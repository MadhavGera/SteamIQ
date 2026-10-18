"""
Job: Train and Evaluate SASRec Sequential Challenger (Phase F).

Evaluates lightweight SASRec (2 layers, 2 heads, d=64, max_len=50, dropout=0.2)
across 3 seeds on the validation split under the identical evaluation protocol:
  - Timestamp-aware masking
  - 5,000 fixed validation evaluation users
  - In-batch sampled softmax loss with logQ popularity correction
  - Measures: parameter count, sec/epoch, total training wall-clock, inference p50/p95 latency
  - Optional 5th candidate source in quota merge recall analysis
  - Comparison vs Clean Two-Tower Champion (Seed 44)

Saves results to:
  - backend/data/recsys/artifacts/recsys_sasrec/seed_{seed}/
  - docs/results/sasrec_benchmark.json
  - backend/data/recsys/reports/sasrec_benchmark.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from recsys.aggregation import aggregate_per_seed_metrics  # noqa: E402
from recsys.data import RecsysData  # noqa: E402
from recsys.evaluate import evaluate_model, get_evaluation_user_split  # noqa: E402
from recsys.metrics import paired_bootstrap_diff  # noqa: E402
from recsys.models.sequential import SASRecConfig, SASRecModel, SASRecModule  # noqa: E402
from recsys.models.two_tower import TwoTowerModel  # noqa: E402
from recsys.registry import log_recsys_run  # noqa: E402
from recsys.train_two_tower import (  # noqa: E402
    build_next_item_training_pairs,
    in_batch_sampled_softmax_loss,
)

logger = logging.getLogger("steamiq.recsys.train_sasrec")


def train_single_seed_sasrec(
    data: RecsysData,
    seed: int,
    epochs: int = 3,
    batch_size: int = 2048,
    pairs_per_epoch: int = 60000,
    lr: float = 2e-3,
    use_text: bool = False,
    data_dir: Path | None = None,
) -> tuple[SASRecModel, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = SASRecConfig(
        n_items=data.n_items,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_len=50,
        dropout=0.2,
        dim_feedforward=256,
        use_text=use_text,
    )

    module = SASRecModule(config)
    full_dataset, q_items = build_next_item_training_pairs(data, max_history=50)
    log_q = torch.tensor(np.log(np.maximum(q_items, 1e-8)), dtype=torch.float32)

    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    epoch_sec_list: list[float] = []
    loss_list: list[float] = []

    print(
        f"\n--- Training SASRec Seed {seed} ({epochs} epochs, {pairs_per_epoch:,} pairs/epoch, batch_size={batch_size}) ---",
        flush=True,
    )
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        module.train()
        total_loss = 0.0

        # Subsample pairs deterministically per epoch
        perm = rng.permutation(len(full_dataset))[:pairs_per_epoch]
        sub_dataset = torch.utils.data.Subset(full_dataset, perm)
        loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for h, m, t in loader:
            optimizer.zero_grad()
            h_tokens = torch.where(m, h + 1, torch.zeros_like(h))
            user_vecs = module.encode_history(h_tokens)
            target_vecs = module.encode_target(t)
            loss = in_batch_sampled_softmax_loss(
                user_vecs=user_vecs,
                item_vecs=target_vecs,
                target_ids=t,
                tau=module.tau,
                log_q=log_q,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        ep_duration = time.perf_counter() - ep_t0
        epoch_sec_list.append(ep_duration)
        avg_loss = total_loss / max(1, len(loader))
        loss_list.append(avg_loss)
        print(
            f"  Epoch {ep}/{epochs}: loss={avg_loss:.4f}, time={ep_duration:.1f}s, lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

    total_training_sec = time.perf_counter() - t_start
    print(
        f"  Seed {seed} finished in {total_training_sec:.1f}s (mean {np.mean(epoch_sec_list):.1f}s/epoch)",
        flush=True,
    )

    model = SASRecModel(config)
    model.module = module
    model.training_time_sec = total_training_sec
    model.epoch_sec_list = epoch_sec_list
    model.precompute_item_vectors(data)

    training_meta = {
        "seed": seed,
        "epochs": epochs,
        "total_training_sec": total_training_sec,
        "mean_epoch_sec": float(np.mean(epoch_sec_list)),
        "epoch_sec_list": epoch_sec_list,
        "final_loss": loss_list[-1] if loss_list else 0.0,
        "parameter_count": model.parameter_count(),
    }

    return model, training_meta


def benchmark_inference_latency(
    model: SASRecModel,
    n_requests: int = 1000,
    history_len: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Measures single-request inference latency distribution (p50, p90, p95, p99)."""
    rng = np.random.default_rng(seed)
    latencies_ms: list[float] = []

    # Warmup 50 requests
    for _ in range(50):
        dummy_hist = [rng.integers(0, model.config.n_items, size=history_len)]
        _ = model.score(dummy_hist)

    # Benchmark loop
    for _ in range(n_requests):
        sample_len = rng.integers(1, 50)
        dummy_hist = [rng.integers(0, model.config.n_items, size=sample_len)]
        t0 = time.perf_counter()
        _ = model.score(dummy_hist)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    lat_arr = np.array(latencies_ms)
    return {
        "n_requests": n_requests,
        "p50_latency_ms": float(np.percentile(lat_arr, 50)),
        "p90_latency_ms": float(np.percentile(lat_arr, 90)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "p99_latency_ms": float(np.percentile(lat_arr, 99)),
        "mean_latency_ms": float(np.mean(lat_arr)),
    }


def evaluate_candidate_union_recall_with_sasrec(
    data: RecsysData,
    sasrec_model: SASRecModel,
    eval_users: np.ndarray,
    two_tower_model: TwoTowerModel,
    k: int = 200,
) -> dict[str, float]:
    """
    Measures union recall on validation users when SASRec is added as an optional 5th source.
    """
    user_train_pos = {}
    train_sorted = data.train_df.sort_values(by=["user_idx", "ts"])
    for u, it in zip(train_sorted["user_idx"].to_numpy(), train_sorted["item_idx"].to_numpy(), strict=True):
        if u not in user_train_pos:
            user_train_pos[u] = []
        user_train_pos[u].append(int(it))

    val_target_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))

    hits_standard = 0
    hits_with_sasrec = 0
    sasrec_individual_hits = 0
    total = 0

    # Evaluate on a representative sample of 1,000 validation users for speed
    sample_users = eval_users[:1000]

    for u in sample_users:
        u_int = int(u)
        target = val_target_map[u_int]
        pos = user_train_pos.get(u_int, [])
        if not pos:
            continue

        total += 1
        # Two-Tower top-K
        tt_scores = two_tower_model.score([np.array(pos, dtype=np.int32)])[0]
        tt_topk = set(np.argsort(tt_scores)[-100:])

        # SASRec top-K
        sas_scores = sasrec_model.score([np.array(pos, dtype=np.int32)])[0]
        sas_topk = set(np.argsort(sas_scores)[-100:])

        if target in sas_topk:
            sasrec_individual_hits += 1

        # Standard Two-Tower pool
        if target in tt_topk:
            hits_standard += 1

        # Combined pool (Two-Tower + SASRec union)
        combined_pool = tt_topk | sas_topk
        if target in combined_pool:
            hits_with_sasrec += 1

    return {
        "n_users_evaluated": total,
        "two_tower_recall_at_100": hits_standard / max(1, total),
        "sasrec_individual_recall_at_100": sasrec_individual_hits / max(1, total),
        "union_recall_at_200": hits_with_sasrec / max(1, total),
        "recall_lift_pct": float((hits_with_sasrec - hits_standard) / max(1, hits_standard) * 100.0),
    }


def run_phase8_sasrec(
    data_dir: Path | str = "backend/data/recsys",
    docs_results_dir: Path | str = "docs/results",
    seeds: list[int] | None = None,
    epochs: int = 3,
    use_text: bool = False,
) -> dict[str, Any]:
    if seeds is None:
        seeds = [42, 43, 44]

    data_path = Path(data_dir)
    if not data_path.exists():
        data_path = backend_dir / "data" / "recsys"
    docs_path = Path(docs_results_dir)
    if not docs_path.exists():
        docs_path = backend_dir.parent / "docs" / "results"
    reports_dir = data_path / "reports"
    artifacts_dir = data_path / "artifacts" / "recsys_sasrec"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading RecsysData from {data_path}...")
    data = RecsysData.load(data_path)

    val_users, _ = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=42, persistence_dir=data_path
    )

    # Load Clean Two-Tower Champion for direct paired comparison
    tt_champ_dir = data_path / "artifacts" / "recsys_two_tower_clean" / "seed_44"
    print(f"Loading Two-Tower Champion from {tt_champ_dir}...")
    tt_champ = TwoTowerModel.load(tt_champ_dir, data=data)

    print("Evaluating Two-Tower Champion on Validation Split...")
    tt_val_res = evaluate_model(tt_champ, data, split="val", user_indices=val_users, seed=42)
    tt_val_ndcg = tt_val_res.metrics["ndcg_at_10"]
    tt_val_user_metrics = tt_val_res.per_user["ndcg_at_10"]

    # Record Hardware Environment
    hardware_info = {
        "cpu_model": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    per_seed_results: list[dict[str, Any]] = []
    models_dict: dict[int, SASRecModel] = {}
    training_metas: dict[int, dict[str, Any]] = {}
    paired_diffs: list[float] = []

    for s in seeds:
        seed_art_dir = artifacts_dir / f"seed_{s}"
        if (seed_art_dir / "sasrec.pt").exists() and (seed_art_dir / "metrics_meta.json").exists():
            print(f"\n--- Loading existing SASRec Seed {s} from {seed_art_dir} ---", flush=True)
            model = SASRecModel.load(seed_art_dir, data=data)
            with open(seed_art_dir / "metrics_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            t_meta = {
                "seed": s,
                "epochs": epochs,
                "total_training_sec": meta.get("training_time_sec", 0.0),
                "mean_epoch_sec": float(np.mean(meta.get("epoch_sec_list", [0.0]))),
                "epoch_sec_list": meta.get("epoch_sec_list", []),
                "parameter_count": meta.get("parameter_count", model.parameter_count()),
            }
        else:
            model, t_meta = train_single_seed_sasrec(
                data=data,
                seed=s,
                epochs=epochs,
                batch_size=2048,
                lr=2e-3,
                use_text=use_text,
                data_dir=data_path,
            )
            # Save artifact
            model.save(seed_art_dir)
            print(f"Saved SASRec seed {s} to {seed_art_dir}")

        models_dict[s] = model
        training_metas[s] = t_meta

        # Evaluate on validation split under identical protocol
        val_res = evaluate_model(model, data, split="val", user_indices=val_users, seed=s)
        m = val_res.metrics
        m["seed"] = s
        m["training_time_sec"] = t_meta["total_training_sec"]
        m["mean_epoch_sec"] = t_meta["mean_epoch_sec"]
        per_seed_results.append(m)

        # Paired comparison vs Two-Tower Champion
        sas_user_metrics = val_res.per_user["ndcg_at_10"]
        diff_mean, diff_lo, diff_hi = paired_bootstrap_diff(sas_user_metrics, tt_val_user_metrics, seed=s)
        paired_diffs.append(diff_mean)
        print(
            f"  Seed {s} Val NDCG@10: {m['ndcg_at_10']:.5f} (Diff vs TT: {diff_mean:+.5f} "
            f"[{diff_lo:+.5f}, {diff_hi:+.5f}])"
        )

        try:
            asyncio.run(
                log_recsys_run(
                    model_name="recsys_sasrec",
                    family="sasrec",
                    dataset_version=data.dataset_version,
                    hyperparameters=model.config.to_dict(),
                    val_metrics=m,
                    artifact_dir=seed_art_dir,
                    extra={"seed": s, "role": "challenger"},
                )
            )
            print(f"  Registered SASRec seed {s} to registry")
        except Exception as e:
            print(f"  Registry logging skipped: {e}")

    # Aggregate across seeds using Phase C aggregation
    agg_metrics = aggregate_per_seed_metrics(per_seed_results, deterministic=False)

    # Benchmark Latency on Champion (Seed closest to mean NDCG)
    mean_ndcg = agg_metrics["summary"]["ndcg_at_10"]["mean"]
    best_seed = min(seeds, key=lambda s: abs(per_seed_results[seeds.index(s)]["ndcg_at_10"] - mean_ndcg))
    champ_model = models_dict[best_seed]
    print(f"\nBenchmarking inference latency on SASRec Seed {best_seed}...")
    latency_res = benchmark_inference_latency(champ_model, n_requests=1000)
    print(
        f"  Measured Latency: p50 = {latency_res['p50_latency_ms']:.2f} ms, "
        f"p95 = {latency_res['p95_latency_ms']:.2f} ms (Mean = {latency_res['mean_latency_ms']:.2f} ms)"
    )

    # Benchmark Candidate Union Recall with SASRec
    print("Benchmarking Candidate Union Recall with SASRec...")
    recall_res = evaluate_candidate_union_recall_with_sasrec(
        data=data,
        sasrec_model=champ_model,
        eval_users=val_users,
        two_tower_model=tt_champ,
    )
    print(
        f"  Two-Tower Recall@100: {recall_res['two_tower_recall_at_100']:.4f}, "
        f"SASRec Recall@100: {recall_res['sasrec_individual_recall_at_100']:.4f}, "
        f"Union Recall@200: {recall_res['union_recall_at_200']:.4f} (Lift: {recall_res['recall_lift_pct']:+.1f}%)"
    )

    # Paired test across pooled user evaluations
    mean_diff = float(np.mean(paired_diffs))
    tt_mean_ndcg = float(tt_val_ndcg)
    sas_std = float(agg_metrics["summary"]["ndcg_at_10"]["std"]) if isinstance(agg_metrics["summary"]["ndcg_at_10"]["std"], int | float) else 0.0
    sig_beats_tt = bool(mean_diff > 0 and (mean_diff - 1.96 * sas_std / np.sqrt(len(seeds))) > 0)

    conclusion = (
        f"SASRec validation NDCG@10 ({agg_metrics['summary']['ndcg_at_10']['formatted']}) vs Two-Tower Champion ({tt_mean_ndcg:.5f}): "
        f"Paired Difference = {mean_diff:+.5f}. "
        f"{'Statistically significantly beats Two-Tower.' if sig_beats_tt else 'Does not beat Two-Tower champion on validation.'}"
    )

    payload = {
        "metadata": {
            "model_name": "recsys_sasrec",
            "model_role": "challenger",
            "dataset_version": data.dataset_version,
            "seeds": seeds,
            "champion_seed": best_seed,
            "epochs": epochs,
            "use_text": use_text,
            "parameter_count": champ_model.parameter_count(),
            "hardware": hardware_info,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "aggregate": agg_metrics,
        "champion": per_seed_results[seeds.index(best_seed)],
        "per_seed": per_seed_results,
        "training_time": {
            "total_training_sec_mean": float(np.mean([t["total_training_sec"] for t in training_metas.values()])),
            "seconds_per_epoch_mean": float(np.mean([t["mean_epoch_sec"] for t in training_metas.values()])),
        },
        "latency_benchmark": latency_res,
        "candidate_union_recall": recall_res,
        "comparison_vs_two_tower": {
            "two_tower_val_ndcg": tt_mean_ndcg,
            "sasrec_val_ndcg": mean_ndcg,
            "mean_paired_diff": mean_diff,
            "significantly_beats_two_tower": sig_beats_tt,
            "conclusion": conclusion,
        },
    }

    # Save reports
    for out_p in [docs_path / "sasrec_benchmark.json", reports_dir / "sasrec_benchmark.json"]:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved benchmark report to {out_p}")

    return payload


if __name__ == "__main__":
    run_phase8_sasrec()
