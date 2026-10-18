"""
Phase L SASRec Sequential Challenger: Converged Training, Hyperparameter Sweep & Slice Analysis.

Resolves Issue L:
  1. Hyperparameter grid sweep on seed 42 across:
     - d_model in {64, 128}
     - dropout in {0.2, 0.4}
     - use_text in {False, True}
     Records trial results in docs/results/sasrec_trials.json.
  2. Trains 3 seeds (42, 43, 44) with winning config up to 40 epochs with early stopping (patience 5).
     Uses learning rate warmup (2 epochs) + Cosine Annealing, gradient clipping (max norm 1.0).
  3. Appends multi-seed training curves to docs/results/training_curves.json.
  4. Performs validation slice analysis comparing SASRec vs Clean Two-Tower Champion across history buckets:
     - 1 to 3 items
     - 4 to 9 items
     - 10+ items
  5. Re-measures CPU inference latency distribution (p50, p90, p95, p99) over 1,000 requests.
  6. Re-measures candidate quota-merge union recall.
  7. Updates docs/results/sasrec_benchmark.json and backend/data/recsys/reports/sasrec_benchmark.json.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

# Add backend to sys.path
sys.path.insert(0, str(Path("backend").resolve()))

from recsys.aggregation import aggregate_per_seed_metrics
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model, get_evaluation_user_split
from recsys.metrics import bootstrap_ci, paired_bootstrap_diff
from recsys.models.sequential import SASRecConfig, SASRecModel, SASRecModule
from recsys.models.two_tower import TwoTowerModel
from recsys.train_two_tower import (
    build_next_item_training_pairs,
    in_batch_sampled_softmax_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steamiq.sasrec_convergence")


def train_single_sasrec_run(
    data: RecsysData,
    seed: int,
    epochs: int = 40,
    batch_size: int = 2048,
    pairs_per_epoch: int = 60000,
    lr: float = 2e-3,
    d_model: int = 64,
    dropout: float = 0.2,
    use_text: bool = False,
    patience: int = 5,
    tuning_users: np.ndarray | None = None,
    save_dir: Path | None = None,
) -> tuple[SASRecModel, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = SASRecConfig(
        n_items=data.n_items,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=50,
        dropout=dropout,
        dim_feedforward=4 * d_model,
        use_text=use_text,
    )

    module = SASRecModule(config)
    full_dataset, q_items = build_next_item_training_pairs(data, max_history=50)
    log_q = torch.tensor(np.log(np.maximum(q_items, 1e-8)), dtype=torch.float32)

    text_table = None
    if use_text:
        text_table = torch.zeros((data.n_items + 1, 384), dtype=torch.float32)
        if data.item_text_emb is not None:
            text_table[1:] = torch.from_numpy(data.item_text_emb.astype(np.float32))

    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.01)

    warmup_epochs = 2
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(1e-5 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    epoch_sec_list: list[float] = []
    loss_list: list[float] = []
    curves: list[dict[str, Any]] = []

    best_tuning_ndcg = -1.0
    best_epoch = -1
    best_state = None
    no_improve = 0

    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    logger.info(
        "Training SASRec [seed=%d, d=%d, drop=%.2f, text=%s, patience=%d, max_ep=%d]...",
        seed, d_model, dropout, use_text, patience, epochs
    )

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        module.train()
        total_loss = 0.0

        n_pairs = min(pairs_per_epoch, len(full_dataset))
        perm = torch.from_numpy(rng.permutation(len(full_dataset))[:n_pairs])
        n_batches = n_pairs // batch_size

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            h = full_dataset.histories[b_idx]
            t = full_dataset.targets[b_idx]

            optimizer.zero_grad()
            if use_text and text_table is not None:
                h_text = F.embedding(h, text_table)
                t_text = text_table[t + 1]
                user_vecs = module.encode_history(h, text_embs=h_text)
                target_vecs = module.encode_target(t, text_embs=t_text)
            else:
                user_vecs = module.encode_history(h)
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
        avg_loss = total_loss / max(1, n_batches)
        loss_list.append(avg_loss)

        # Validation evaluation on tuning users
        val_ndcg10 = 0.0
        val_hr10 = 0.0
        if tuning_users is not None and len(tuning_users) > 0:
            eval_model = SASRecModel(config)
            eval_model.module = module
            eval_model.precompute_item_vectors(data)
            val_res = evaluate_model(
                eval_model,
                data,
                split="val",
                k_list=(10, 20),
                user_indices=tuning_users,
                seed=seed,
            )
            val_ndcg10 = float(val_res.metrics.get("ndcg_at_10", 0.0))
            val_hr10 = float(val_res.metrics.get("hit_rate_at_10", 0.0))

        cur_lr = float(scheduler.get_last_lr()[0])
        epoch_rec = {
            "epoch": ep,
            "train_loss": avg_loss,
            "tau": float(module.tau.item()),
            "val_ndcg_at_10": val_ndcg10,
            "val_hr_at_10": val_hr10,
            "learning_rate": cur_lr,
            "epoch_sec": ep_duration,
            "wall_time": float(time.perf_counter() - t_start),
            "eval_ndcg": val_ndcg10,
        }
        curves.append(epoch_rec)

        if val_ndcg10 > best_tuning_ndcg:
            best_tuning_ndcg = val_ndcg10
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}
            no_improve = 0
            logger.info(
                "  Ep %2d/%2d [%.1fs] loss=%.4f | Val NDCG@10=%.4f (★ NEW BEST) | HR@10=%.4f",
                ep, epochs, ep_duration, avg_loss, val_ndcg10, val_hr10
            )
        else:
            no_improve += 1
            logger.info(
                "  Ep %2d/%2d [%.1fs] loss=%.4f | Val NDCG@10=%.4f (best=%.4f ep %d, %d/%d)",
                ep, epochs, ep_duration, avg_loss, val_ndcg10, best_tuning_ndcg, best_epoch, no_improve, patience
            )
            if no_improve >= patience:
                logger.info("  -> Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Restore best checkpoint
    if best_state is not None:
        module.load_state_dict(best_state)

    total_training_sec = time.perf_counter() - t_start
    model = SASRecModel(config)
    model.module = module
    model.training_time_sec = total_training_sec
    model.epoch_sec_list = epoch_sec_list
    model.precompute_item_vectors(data)

    training_meta = {
        "seed": seed,
        "d_model": d_model,
        "dropout": dropout,
        "use_text": use_text,
        "epochs_trained": len(curves),
        "best_epoch": best_epoch,
        "best_tuning_val_ndcg_at_10": best_tuning_ndcg,
        "total_training_sec": total_training_sec,
        "mean_epoch_sec": float(np.mean(epoch_sec_list)) if epoch_sec_list else 0.0,
        "epoch_sec_list": epoch_sec_list,
        "final_loss": loss_list[-1] if loss_list else 0.0,
        "parameter_count": model.parameter_count(),
        "curves": curves,
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        with open(save_path / "training_curves.json", "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2)
        logger.info("Saved model artifact and curves to %s", save_path)

    return model, training_meta


def _worker_train_sasrec_trial(args: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(4)
    data_path = Path(args["data_path"])
    data = RecsysData.load(data_path)
    _, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=data_path,
    )

    idx = args["trial_id"]
    params = args["params"]
    seed = args["seed"]
    logger.info(">>> Starting Trial %d/8: %s", idx, params)
    _, meta = train_single_sasrec_run(
        data=data,
        seed=seed,
        epochs=args.get("epochs", 15),
        batch_size=2048,
        pairs_per_epoch=args.get("pairs_per_epoch", 25000),
        lr=2e-3,
        d_model=params["d_model"],
        dropout=params["dropout"],
        use_text=params["use_text"],
        patience=args.get("patience", 3),
        tuning_users=tuning_users,
    )
    trial_rec = {
        "trial_id": idx,
        "d_model": params["d_model"],
        "dropout": params["dropout"],
        "use_text": params["use_text"],
        "parameter_count": meta["parameter_count"],
        "epochs_trained": meta["epochs_trained"],
        "best_epoch": meta["best_epoch"],
        "best_tuning_val_ndcg_at_10": meta["best_tuning_val_ndcg_at_10"],
        "total_training_sec": round(meta["total_training_sec"], 1),
        "mean_epoch_sec": round(meta["mean_epoch_sec"], 1),
    }
    logger.info(
        ">>> Trial %d Complete: best_tuning_NDCG=%.4f (best ep %d, %d total eps, %.1fs)",
        idx, trial_rec["best_tuning_val_ndcg_at_10"], trial_rec["best_epoch"],
        trial_rec["epochs_trained"], trial_rec["total_training_sec"]
    )
    return trial_rec


def _worker_train_sasrec_seed(args: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(4)
    data_path = Path(args["data_path"])
    data = RecsysData.load(data_path)
    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=data_path,
    )
    s = args["seed"]
    win_cfg = args["win_cfg"]
    seed_dir = Path(args["seed_dir"])

    logger.info(">>> Training Converged SASRec Seed %d with winning config %s...", s, win_cfg)
    model, meta = train_single_sasrec_run(
        data=data,
        seed=s,
        epochs=args.get("epochs", 35),
        batch_size=2048,
        pairs_per_epoch=args.get("pairs_per_epoch", 35000),
        lr=2e-3,
        d_model=win_cfg["d_model"],
        dropout=win_cfg["dropout"],
        use_text=win_cfg["use_text"],
        patience=args.get("patience", 5),
        tuning_users=tuning_users,
        save_dir=seed_dir,
    )
    v_res = evaluate_model(model, data, split="val", user_indices=eval_users, seed=s)
    m = {k: v for k, v in v_res.metrics.items() if not k.endswith("_ranks")}
    m["seed"] = s
    m["training_time_sec"] = meta["total_training_sec"]
    m["mean_epoch_sec"] = meta["mean_epoch_sec"]
    return {
        "seed": s,
        "metrics": m,
        "meta": meta,
    }


def run_hyperparameter_sweep(
    data_path: Path,
    docs_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs the 8-trial hyperparameter grid on seed 42 across 2 parallel workers."""
    grid = [
        {"d_model": 64, "dropout": 0.2, "use_text": False},
        {"d_model": 64, "dropout": 0.2, "use_text": True},
        {"d_model": 64, "dropout": 0.4, "use_text": False},
        {"d_model": 64, "dropout": 0.4, "use_text": True},
        {"d_model": 128, "dropout": 0.2, "use_text": False},
        {"d_model": 128, "dropout": 0.2, "use_text": True},
        {"d_model": 128, "dropout": 0.4, "use_text": False},
        {"d_model": 128, "dropout": 0.4, "use_text": True},
    ]

    tasks = [
        {
            "trial_id": idx,
            "params": params,
            "seed": seed,
            "data_path": str(data_path),
            "epochs": 15,
            "pairs_per_epoch": 25000,
            "patience": 3,
        }
        for idx, params in enumerate(grid, 1)
    ]

    logger.info("Starting SASRec Hyperparameter Sweep on Seed 42 (%d trials across 2 workers)...", len(tasks))
    trials = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(_worker_train_sasrec_trial, t): t for t in tasks}
        for future in concurrent.futures.as_completed(future_map):
            trials.append(future.result())

    trials.sort(key=lambda t: t["trial_id"])

    # Pick winning configuration
    winner = max(trials, key=lambda t: t["best_tuning_val_ndcg_at_10"])
    winning_config = {
        "d_model": winner["d_model"],
        "dropout": winner["dropout"],
        "use_text": winner["use_text"],
    }
    logger.info("\n★ WINNING CONFIGURATION: Trial %d -> %s (NDCG@10 = %.4f)", winner["trial_id"], winning_config, winner["best_tuning_val_ndcg_at_10"])

    data = RecsysData.load(data_path)
    payload = {
        "metadata": {
            "sweep_name": "sasrec_hyperparameter_sweep",
            "seed": seed,
            "n_trials": len(trials),
            "dataset_version": data.dataset_version,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "trials": trials,
        "winning_config": winning_config,
        "winning_trial": winner,
    }

    trials_file = docs_dir / "sasrec_trials.json"
    with open(trials_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved hyperparameter trials table to %s", trials_file)

    return payload


def evaluate_history_length_slices(
    sasrec_model: SASRecModel,
    tt_model: TwoTowerModel,
    data: RecsysData,
    eval_users: np.ndarray,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Evaluates NDCG@10 and HR@10 broken down by user history length buckets:
      - 1 to 3 items (short)
      - 4 to 9 items (medium)
      - 10+ items (long)
    """
    logger.info("Evaluating Validation History Length Slice Analysis (SASRec vs Two-Tower)...")

    # Evaluate both models on validation split
    sas_res = evaluate_model(sasrec_model, data, split="val", user_indices=eval_users, seed=seed)
    tt_res = evaluate_model(tt_model, data, split="val", user_indices=eval_users, seed=seed)

    sas_u_ndcg = sas_res.per_user["ndcg_at_10"]
    tt_u_ndcg = tt_res.per_user["ndcg_at_10"]
    sas_u_hr = sas_res.per_user["hit_rate_at_10"]
    tt_u_hr = tt_res.per_user["hit_rate_at_10"]
    history_lengths = sas_res.per_user.get("history_length", tt_res.per_user.get("history_length"))

    # Define buckets
    buckets = {
        "short_1_3": (history_lengths >= 1) & (history_lengths <= 3),
        "medium_4_9": (history_lengths >= 4) & (history_lengths <= 9),
        "long_10_plus": (history_lengths >= 10),
    }

    slice_results = {}
    for b_key, mask in buckets.items():
        n_users_bucket = int(np.sum(mask))
        if n_users_bucket == 0:
            continue

        sas_slice_ndcg = sas_u_ndcg[mask]
        tt_slice_ndcg = tt_u_ndcg[mask]
        sas_slice_hr = sas_u_hr[mask]
        tt_slice_hr = tt_u_hr[mask]

        sas_nd_m, sas_nd_lo, sas_nd_hi = bootstrap_ci(sas_slice_ndcg, n_boot=n_boot, seed=seed)
        tt_nd_m, tt_nd_lo, tt_nd_hi = bootstrap_ci(tt_slice_ndcg, n_boot=n_boot, seed=seed)

        sas_hr_m, sas_hr_lo, sas_hr_hi = bootstrap_ci(sas_slice_hr, n_boot=n_boot, seed=seed)
        tt_hr_m, tt_hr_lo, tt_hr_hi = bootstrap_ci(tt_slice_hr, n_boot=n_boot, seed=seed)

        diff_m, diff_lo, diff_hi = paired_bootstrap_diff(sas_slice_ndcg, tt_slice_ndcg, n_boot=n_boot, seed=seed)

        slice_results[b_key] = {
            "n_users": n_users_bucket,
            "sasrec": {
                "ndcg_at_10": sas_nd_m,
                "ndcg_at_10_lo95": sas_nd_lo,
                "ndcg_at_10_hi95": sas_nd_hi,
                "hit_rate_at_10": sas_hr_m,
                "hit_rate_at_10_lo95": sas_hr_lo,
                "hit_rate_at_10_hi95": sas_hr_hi,
            },
            "two_tower": {
                "ndcg_at_10": tt_nd_m,
                "ndcg_at_10_lo95": tt_nd_lo,
                "ndcg_at_10_hi95": tt_nd_hi,
                "hit_rate_at_10": tt_hr_m,
                "hit_rate_at_10_lo95": tt_hr_lo,
                "hit_rate_at_10_hi95": tt_hr_hi,
            },
            "paired_diff_sasrec_vs_tt": {
                "diff_mean": diff_m,
                "diff_lo95": diff_lo,
                "diff_hi95": diff_hi,
                "significant": bool(diff_lo > 0 or diff_hi < 0),
            },
        }

    return slice_results


def benchmark_latency(model: SASRecModel, n_requests: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    latencies: list[float] = []

    # Warmup
    for _ in range(50):
        dummy = [rng.integers(0, model.config.n_items, size=15)]
        _ = model.score(dummy)

    for _ in range(n_requests):
        hist_len = int(rng.integers(1, 50))
        dummy = [rng.integers(0, model.config.n_items, size=hist_len)]
        t0 = time.perf_counter()
        _ = model.score(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "n_requests": n_requests,
        "p50_latency_ms": float(np.percentile(arr, 50)),
        "p90_latency_ms": float(np.percentile(arr, 90)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "p99_latency_ms": float(np.percentile(arr, 99)),
        "mean_latency_ms": float(np.mean(arr)),
    }


def evaluate_quota_recall(
    data: RecsysData,
    sasrec: SASRecModel,
    two_tower: TwoTowerModel,
    eval_users: np.ndarray,
) -> dict[str, float]:
    user_train_pos = {}
    train_sorted = data.train_df.sort_values(by=["user_idx", "ts"])
    for u, it in zip(train_sorted["user_idx"].to_numpy(), train_sorted["item_idx"].to_numpy(), strict=True):
        if u not in user_train_pos:
            user_train_pos[u] = []
        user_train_pos[u].append(int(it))

    val_target_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))

    sample_users = eval_users[:1000]
    valid_targets = []
    valid_histories = []
    for u in sample_users:
        u_int = int(u)
        target = val_target_map.get(u_int)
        pos = user_train_pos.get(u_int, [])
        if target is not None and pos:
            valid_targets.append(int(target))
            valid_histories.append(np.array(pos, dtype=np.int32))

    total = len(valid_targets)
    if total == 0:
        return {
            "n_users_evaluated": 0,
            "two_tower_recall_at_100": 0.0,
            "sasrec_individual_recall_at_100": 0.0,
            "union_recall_at_200": 0.0,
            "recall_lift_pct": 0.0,
        }

    # Batch scoring
    all_tt_sc = two_tower.score(valid_histories)
    all_sas_sc = sasrec.score(valid_histories)

    hits_tt = 0
    hits_sas = 0
    hits_union = 0

    for i in range(total):
        target = valid_targets[i]
        tt_sc = all_tt_sc[i]
        sas_sc = all_sas_sc[i]

        top_tt = set(np.argsort(tt_sc)[-100:])
        top_sas = set(np.argsort(sas_sc)[-100:])
        union_pool = top_tt | top_sas

        if target in top_tt:
            hits_tt += 1
        if target in top_sas:
            hits_sas += 1
        if target in union_pool:
            hits_union += 1

    rec_tt = hits_tt / max(1, total)
    rec_sas = hits_sas / max(1, total)
    rec_union = hits_union / max(1, total)
    lift = float((rec_union - rec_tt) / max(1e-6, rec_tt) * 100.0)

    return {
        "n_users_evaluated": total,
        "two_tower_recall_at_100": rec_tt,
        "sasrec_individual_recall_at_100": rec_sas,
        "union_recall_at_200": rec_union,
        "recall_lift_pct": lift,
    }


def main():
    data_dir = Path("backend/data/recsys")
    docs_dir = Path("docs/results")
    reports_dir = data_dir / "reports"
    artifacts_dir = data_dir / "artifacts" / "recsys_sasrec"

    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = RecsysData.load(data_dir)
    eval_users, tuning_users = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=42, persistence_dir=data_dir
    )

    # 1. Hyperparameter sweep on seed 42
    sweep_payload = run_hyperparameter_sweep(data_dir, docs_dir, seed=42)
    win_cfg = sweep_payload["winning_config"]

    # 2. Train 3 seeds with winning config to full convergence (up to 40 epochs, patience 5)
    seeds = [42, 43, 44]
    tasks = [
        {
            "seed": s,
            "win_cfg": win_cfg,
            "data_path": str(data_dir),
            "seed_dir": str(artifacts_dir / f"seed_{s}"),
            "epochs": 40,
            "pairs_per_epoch": 35000,
            "patience": 5,
        }
        for s in seeds
    ]
    logger.info("Training 3 Converged SASRec Seeds (%s) across 2 workers...", seeds)
    seed_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(_worker_train_sasrec_seed, t): t for t in tasks}
        for future in concurrent.futures.as_completed(future_map):
            seed_results.append(future.result())

    seed_results.sort(key=lambda r: r["seed"])

    models_dict = {}
    training_metas = {}
    per_seed_val_metrics = []
    all_curves_dict = {}

    for res in seed_results:
        s = res["seed"]
        training_metas[s] = res["meta"]
        per_seed_val_metrics.append(res["metrics"])
        all_curves_dict[str(s)] = res["meta"]["curves"]
        models_dict[s] = SASRecModel.load(artifacts_dir / f"seed_{s}", data=data)

    # Aggregate validation metrics across seeds
    agg_metrics = aggregate_per_seed_metrics(per_seed_val_metrics, deterministic=False)

    # Pick champion seed closest to mean NDCG@10
    mean_ndcg = agg_metrics["summary"]["ndcg_at_10"]["mean"]
    champ_seed = min(seeds, key=lambda s: abs(per_seed_val_metrics[seeds.index(s)]["ndcg_at_10"] - mean_ndcg))
    champ_model = models_dict[champ_seed]
    logger.info(
        "\n★ Selected Champion Seed: %d (val NDCG = %.5f, mean = %.5f)",
        champ_seed,
        per_seed_val_metrics[seeds.index(champ_seed)]["ndcg_at_10"],
        mean_ndcg,
    )

    # 3. Append to training_curves.json
    curves_file = docs_dir / "training_curves.json"
    if curves_file.exists():
        with open(curves_file, encoding="utf-8") as f:
            curves_payload = json.load(f)
    else:
        curves_payload = {"metadata": {"report_key": "training_curves"}}

    curves_payload["sasrec_curves"] = {
        "champion_seed": champ_seed,
        "curves": training_metas[champ_seed]["curves"],
        "seeds": all_curves_dict,
    }
    with open(curves_file, "w", encoding="utf-8") as f:
        json.dump(curves_payload, f, indent=2)
    logger.info("Updated docs/results/training_curves.json with converged SASRec curves.")

    # 4. Load Two-Tower Champion for comparison and history slice analysis
    tt_clean_path = data_dir / "artifacts" / "recsys_two_tower_clean" / "seed_44"
    tt_champ = TwoTowerModel.load(tt_clean_path, data=data)

    slices_res = evaluate_history_length_slices(
        sasrec_model=champ_model,
        tt_model=tt_champ,
        data=data,
        eval_users=eval_users,
        seed=42,
    )

    # 5. Benchmark inference latency
    latency_res = benchmark_latency(champ_model, n_requests=1000)

    # 6. Candidate quota recall
    recall_res = evaluate_quota_recall(data, champ_model, tt_champ, eval_users)

    # 7. Hardware details
    hardware_info = {
        "cpu_model": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    # 8. Comparison vs Two-Tower Champion
    tt_val_eval = evaluate_model(tt_champ, data, split="val", user_indices=eval_users, seed=42)
    tt_val_ndcg = float(tt_val_eval.metrics["ndcg_at_10"])
    overall_diff_m, overall_diff_lo, overall_diff_hi = paired_bootstrap_diff(
        evaluate_model(champ_model, data, split="val", user_indices=eval_users, seed=42).per_user["ndcg_at_10"],
        tt_val_eval.per_user["ndcg_at_10"],
        n_boot=1000,
        seed=42,
    )

    benchmark_payload = {
        "metadata": {
            "model_name": "recsys_sasrec",
            "model_role": "challenger",
            "dataset_version": data.dataset_version,
            "seeds": seeds,
            "champion_seed": champ_seed,
            "winning_config": win_cfg,
            "parameter_count": champ_model.parameter_count(),
            "hardware": hardware_info,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "aggregate": agg_metrics,
        "champion": per_seed_val_metrics[seeds.index(champ_seed)],
        "per_seed": per_seed_val_metrics,
        "training_time": {
            "total_training_sec_mean": float(np.mean([t["total_training_sec"] for t in training_metas.values()])),
            "seconds_per_epoch_mean": float(np.mean([t["mean_epoch_sec"] for t in training_metas.values()])),
        },
        "latency_benchmark": latency_res,
        "candidate_union_recall": recall_res,
        "history_length_slice_analysis": slices_res,
        "comparison_vs_two_tower": {
            "two_tower_val_ndcg": tt_val_ndcg,
            "sasrec_val_ndcg": float(mean_ndcg),
            "mean_paired_diff": overall_diff_m,
            "diff_ci95": [overall_diff_lo, overall_diff_hi],
            "significantly_beats_two_tower": bool(overall_diff_lo > 0.0),
        },
    }

    for out_p in [docs_dir / "sasrec_benchmark.json", reports_dir / "sasrec_benchmark.json"]:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(benchmark_payload, f, indent=2)
        logger.info("Saved converged benchmark report to %s", out_p)


if __name__ == "__main__":
    main()
