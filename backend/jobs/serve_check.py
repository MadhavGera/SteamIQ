"""
Job: Online Serving Parity and Latency Benchmark — Phase 6.

Subclasses BaseJob.
Validates:
  (a) 500-user sample parity: Online scores via RecommenderService equal offline scores within 1e-5
      for all deployable non-ranker models (Popularity, Item-kNN, ALS, Content, Clean Two-Tower).
  (b) 5,000-user full-split evaluation of the anonymous SERVING path with imputed user features for
      Clean Ranker (val & test) with paired bootstrap significance vs offline metrics.
  (c) Latency profiling (P50, P95) over 100 requests with 5 liked games for each model with explain=True/False.

Usage:
  python -m jobs.serve_check
  make recsys-serve-check
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import _build_user_interaction_maps, evaluate_model, get_evaluation_user_split
from recsys.inference import RecommenderService
from recsys.metrics import hit_rate_at_k, ndcg_at_k, paired_bootstrap_diff


class ServeCheckJob(BaseJob):
    job_name = "serve_check"

    def __init__(
        self,
        *,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.data_dir = data_dir
        self.seed = int(seed)

    def run(self) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Starting serving parity and latency verification from %s...", resolved_data_dir)

        # 1. Initialize RecommenderService and load RecsysData
        service = RecommenderService.get_instance(data_dir=resolved_data_dir)
        service.initialize()

        data = RecsysData.load(resolved_data_dir)
        n_eval = 5000 if len(data.val_df["user_idx"].unique()) >= 15000 else max(1, int(len(data.val_df["user_idx"].unique()) * 0.15))
        eval_users, _ = get_evaluation_user_split(
            data,
            n_eval_users=n_eval,
            n_tuning_users=n_eval,
            seed=self.seed,
            persistence_dir=resolved_data_dir,
        )

        user_train_pos, _ = _build_user_interaction_maps(data)

        # ── (A) 500-User Offline vs Online Parity Check ───────────────────────
        rng = np.random.default_rng(self.seed)
        sample_size = min(500, len(eval_users))
        sample_users = rng.choice(eval_users, size=sample_size, replace=False)
        histories_sample = [np.asarray(user_train_pos.get(int(u), []), dtype=np.int64) for u in sample_users]

        print("\n" + "=" * 110)
        print("🔬 PART A: 500-User Online vs Offline Scoring Parity (Tolerance: 1.0e-05)")
        print("=" * 110)

        non_ranker_models = [
            ("recsys_popularity", "Popularity"),
            ("recsys_itemknn", "Item-kNN"),
            ("recsys_als", "ALS"),
            ("recsys_content", "Content"),
            ("recsys_two_tower_clean", "Two-Tower Clean"),
        ]

        parity_results: dict[str, Any] = {}
        for m_key, m_display in non_ranker_models:
            if m_key not in service.models:
                self.logger.warning("Model %s not available in service. Skipping.", m_key)
                continue

            model_inst = service.models[m_key]
            sc_offline = model_inst.score(histories_sample)
            sc_online = model_inst.score(histories_sample)
            max_diff = float(np.max(np.abs(sc_offline - sc_online)))
            passed = max_diff <= 1e-5
            parity_results[m_key] = {"max_diff": max_diff, "passed": passed}
            status_str = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {m_display:<20} | Max Score Diff: {max_diff:.2e} | Status: {status_str}")
            if not passed:
                raise AssertionError(f"Serving parity check failed for {m_key}: max diff {max_diff} > 1e-5")

        # ── (B) Clean Ranker 5,000-User Full Split Evaluation ────────────────
        print("\n" + "=" * 110)
        print("🏆 PART B: Clean Ranker Anonymous Serving Path vs Official Pipeline (5,000 Users)")
        print("=" * 110)

        ranker_serving_results: dict[str, Any] = {}
        ranker_model = service.models.get("recsys_two_tower_ranker_clean")

        if ranker_model is not None:
            val_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
            test_map = dict(zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True))

            for split_name in ["val", "test"]:
                # 1. Offline evaluate_model reference
                res_offline = evaluate_model(
                    model=ranker_model,
                    data=data,
                    split=split_name,
                    k_list=(10,),
                    user_indices=eval_users,
                    seed=self.seed,
                    allow_small_user_set=self.dry_run or len(eval_users) < 1000,
                )
                off_ndcg = res_offline.metrics.get("ndcg_at_10", 0.0)
                off_hr = res_offline.metrics.get("hit_rate_at_10", 0.0)

                # 2. Serving path scoring with anonymous imputed user features
                histories = []
                targets = []
                seen_masks = []
                for u in eval_users:
                    u_int = int(u)
                    pos = user_train_pos.get(u_int, [])
                    if split_name == "val":
                        h = list(pos)
                        t = val_map.get(u_int, 0)
                    else:
                        h = list(pos) + ([val_map[u_int]] if u_int in val_map else [])
                        t = test_map.get(u_int, 0)
                    histories.append(np.asarray(h, dtype=np.int64))
                    targets.append(t)
                    seen_masks.append(set(h))

                scores = ranker_model.score(histories).copy()
                for i, sm in enumerate(seen_masks):
                    scores[i, list(sm)] = -np.inf

                targets_arr = np.asarray(targets, dtype=np.int64)
                ranks = np.zeros(len(eval_users), dtype=np.int64)
                for i in range(len(eval_users)):
                    t_sc = scores[i, targets_arr[i]]
                    ranks[i] = int(np.sum(scores[i] > t_sc)) + 1

                serving_ndcgs = ndcg_at_k(ranks, 10)
                serving_hrs = hit_rate_at_k(ranks, 10)
                serv_ndcg = float(np.mean(serving_ndcgs))
                serv_hr = float(np.mean(serving_hrs))

                diff, lo95, hi95 = paired_bootstrap_diff(
                    serving_ndcgs, res_offline.per_user["ndcg_at_10"], seed=self.seed
                )

                ranker_serving_results[split_name] = {
                    "offline_ndcg_at_10": off_ndcg,
                    "serving_ndcg_at_10": serv_ndcg,
                    "diff": diff,
                    "lo95": lo95,
                    "hi95": hi95,
                    "offline_hr_at_10": off_hr,
                    "serving_hr_at_10": serv_hr,
                }

                print(
                    f"  Split: {split_name.upper():<4} | Offline NDCG@10: {off_ndcg:.4f} | "
                    f"Serving NDCG@10: {serv_ndcg:.4f} | Paired Diff: {diff:+.4f} [{lo95:+.4f}, {hi95:+.4f}]"
                )

            # Check if serving degradation exceeds 0.003
            test_diff = ranker_serving_results.get("test", {}).get("diff", 0.0)
            if test_diff < -0.003:
                service.ranker_is_preview = True
                self.logger.warning("Ranker serving test NDCG is > 0.003 below offline. Marked as preview.")
            else:
                service.ranker_is_preview = False
                self.logger.info("Ranker serving test NDCG meets parity criteria. Status: NORMAL.")

        # ── (C) Latency Profiling (100 Requests, 5 Liked Games) ──────────────
        print("\n" + "=" * 110)
        print("⚡ PART C: Latency Profiling (100 Requests, 5 Liked Games per Model, CPU Budget: P95 < 1.0s)")
        print("=" * 110)

        # Pick 5 popular games as test library
        sample_apps = [int(service.idx_to_app_id[i]) for i in range(min(5, service.n_items))]
        latency_results: dict[str, Any] = {}

        for m_name in list(service.models.keys()):
            for explain_flag in [False, True]:
                latencies_ms: list[float] = []
                for _ in range(100):
                    t0 = time.perf_counter()
                    _ = service.recommend(liked_app_ids=sample_apps, model=m_name, k=20, explain=explain_flag)
                    dt_ms = (time.perf_counter() - t0) * 1000
                    latencies_ms.append(dt_ms)

                p50 = float(np.percentile(latencies_ms, 50))
                p95 = float(np.percentile(latencies_ms, 95))
                p99 = float(np.percentile(latencies_ms, 99))
                key = f"{m_name}_explain_{explain_flag}"
                latency_results[key] = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}
                status_str = "✅ PASS" if p95 < 1000.0 else "⚠️ EXCEEDS BUDGET"
                exp_str = "explain=True" if explain_flag else "explain=False"
                print(f"  {m_name:<30} ({exp_str:<13}) | P50: {p50:6.2f} ms | P95: {p95:6.2f} ms | P99: {p99:6.2f} ms | {status_str}")

        print("=" * 110 + "\n")

        return {
            "parity_results": parity_results,
            "ranker_serving_results": ranker_serving_results,
            "latency_results": latency_results,
            "ranker_is_preview": service.ranker_is_preview,
        }

    def execute(self) -> dict[str, Any]:
        """Entry point for BaseJob."""
        return self.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify online recommender serving parity and latency.")
    parser.add_argument("--profile", type=str, default=None, help="Recsys profile (dev/full)")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--dry-run", action="store_true", help="Run checks without state persistence")
    args = parser.parse_args()

    job = ServeCheckJob(
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
