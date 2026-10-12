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

        # Print loaded artifact paths
        ranker_key = (
            "recsys_two_tower_ranker_v2_trainstats"
            if "recsys_two_tower_ranker_v2_trainstats" in service.models
            else (
                "recsys_two_tower_ranker_clean_v2"
                if "recsys_two_tower_ranker_clean_v2" in service.models
                else "recsys_two_tower_ranker_clean"
            )
        )
        ranker_meta = service.model_runs_meta.get(ranker_key, {})
        loaded_art_dir = ranker_meta.get("artifact_path", "unknown")
        print("\n" + "=" * 110)
        print("📦 MODEL ARTIFACT RESOLUTION (via model_runs registry)")
        print("=" * 110)
        print(f"  Ranker Loaded Artifact Dir:     {loaded_art_dir}")
        print(f"  model_runs ({ranker_key}) : {loaded_art_dir} (Run ID: {ranker_meta.get('model_run_id', 'unknown')})")
        print("=" * 110)

        # ── (A) 500-User Offline vs Online Parity Check ───────────────────────
        rng = np.random.default_rng(self.seed)
        # Filter for users with at least 1 interaction
        valid_eval_users = [u for u in eval_users if len(user_train_pos.get(int(u), [])) > 0]
        sample_size = min(500, len(valid_eval_users))
        sample_users = rng.choice(valid_eval_users, size=sample_size, replace=False)

        print("\n" + "=" * 110)
        print("🔬 PART A: 500-User Online (RecommenderService.recommend) vs Offline Scoring Parity (Tolerance: 1.0e-05)")
        print("=" * 110)

        all_models = [
            ("recsys_popularity", "Popularity"),
            ("recsys_itemknn", "Item-kNN"),
            ("recsys_als", "ALS"),
            ("recsys_content", "Content"),
            ("recsys_two_tower_clean", "Two-Tower Clean"),
        ]

        parity_results: dict[str, Any] = {}
        for m_key, m_display in all_models:
            if m_key not in service.models:
                self.logger.warning("Model %s not available in service. Skipping.", m_key)
                continue

            model_inst = service.models[m_key]
            max_diff = 0.0
            order_mismatches = 0

            for u in sample_users:
                u_int = int(u)
                history_indices = user_train_pos.get(u_int, [])[-50:]
                history_app_ids = [int(service.idx_to_app_id[idx]) for idx in history_indices if idx in service.idx_to_app_id]
                if not history_app_ids:
                    continue

                # Online call via RecommenderService.recommend()
                online_res = service.recommend(liked_app_ids=history_app_ids, model=m_key, k=10, explain=False)
                online_top_ids = [r["app_id"] for r in online_res["items"]]
                online_top_scores = np.array([r["score"] for r in online_res["items"]])

                # Offline scoring via model_inst.score()
                known_indices = [service.app_id_to_idx[aid] for aid in history_app_ids if aid in service.app_id_to_idx]
                sc_offline = model_inst.score([np.asarray(known_indices, dtype=np.int64)])[0].copy()
                sc_offline[known_indices] = -np.inf
                offline_top_indices = np.argsort(-sc_offline)[:10]
                offline_top_ids = [int(service.idx_to_app_id[idx]) for idx in offline_top_indices]
                offline_top_scores = sc_offline[offline_top_indices]

                if online_top_ids != offline_top_ids:
                    order_mismatches += 1

                score_diff = np.max(np.abs(online_top_scores - offline_top_scores))
                if score_diff > max_diff:
                    max_diff = float(score_diff)

            passed = (max_diff <= 1e-5) and (order_mismatches == 0)
            parity_results[m_key] = {"max_diff": max_diff, "order_mismatches": order_mismatches, "passed": passed}
            status_str = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {m_display:<20} | Max Score Diff: {max_diff:.2e} | Order Mismatches: {order_mismatches}/500 | Status: {status_str}")
            if not passed:
                raise AssertionError(f"Serving parity check failed for {m_key}: max diff {max_diff} > 1e-5 or mismatches {order_mismatches} > 0")

        # ── (A.2) Ranker vs Two-Tower 3-Way Top-10 Comparison ─────────────────
        tt_key = "recsys_two_tower_clean"
        if ranker_key in service.models and tt_key in service.models:
            ranker_inst = service.models[ranker_key]
            tt_inst = service.models[tt_key]

            print("\n" + "-" * 110)
            print("🔍 RANKER TOP-10 vs RANKER MASKED vs TWO-TOWER CLEAN (3 User Histories)")
            print("-" * 110)
            for h_idx in range(min(3, len(sample_users))):
                u_int = int(sample_users[h_idx])
                history_indices = user_train_pos.get(u_int, [])[-50:]
                history_app_ids = [int(service.idx_to_app_id[idx]) for idx in history_indices if idx in service.idx_to_app_id]
                history_titles = [service.items_metadata.get(idx, {}).get("title", f"App #{service.idx_to_app_id.get(idx)}") for idx in history_indices[:5]]

                print(f"\n[User History #{h_idx + 1} (User idx: {u_int})]: Seed Library: {history_titles} ... ({len(history_app_ids)} items)")

                # 1. service.recommend(ranker, explain=False)
                rec_srv = service.recommend(liked_app_ids=history_app_ids, model=ranker_key, k=10, explain=False)
                srv_ids = [r["app_id"] for r in rec_srv["items"]]

                # 2. ranker.score() masked
                known_indices = [service.app_id_to_idx[aid] for aid in history_app_ids if aid in service.app_id_to_idx]
                sc_rk = ranker_inst.score([np.asarray(known_indices, dtype=np.int64)])[0].copy()
                sc_rk[known_indices] = -np.inf
                rk_top_idx = np.argsort(-sc_rk)[:10]
                rk_ids = [int(service.idx_to_app_id[i]) for i in rk_top_idx]

                # 3. two_tower_clean.score() masked
                sc_tt = tt_inst.score([np.asarray(known_indices, dtype=np.int64)])[0].copy()
                sc_tt[known_indices] = -np.inf
                tt_top_idx = np.argsort(-sc_tt)[:10]
                tt_ids = [int(service.idx_to_app_id[i]) for i in tt_top_idx]

                def _fmt_list(ids: list[int]) -> list[str]:
                    res = []
                    for aid in ids:
                        idx = service.app_id_to_idx.get(aid, -1)
                        t = service.items_metadata.get(idx, {}).get("title", f"App #{aid}") if idx >= 0 else f"App #{aid}"
                        res.append(f"'{t}' (AppID: {aid})")
                    return res

                print("  (1) service.recommend(ranker):")
                for item_str in _fmt_list(srv_ids):
                    print(f"      - {item_str}")
                print("  (2) ranker.score() masked:")
                for item_str in _fmt_list(rk_ids):
                    print(f"      - {item_str}")
                print("  (3) two_tower_clean masked:")
                for item_str in _fmt_list(tt_ids):
                    print(f"      - {item_str}")

                if srv_ids == tt_ids:
                    raise AssertionError(
                        f"Serving ranker output equals two-tower order for user {u_int}! "
                        f"Ranker must perform active re-ranking."
                    )
                if srv_ids != rk_ids:
                    raise AssertionError(
                        f"service.recommend(ranker) differs from ranker.score() masked for user {u_int}!"
                    )
                print("  ✅ Parity Verified: service == ranker_score, service != two_tower_clean.")
            print("-" * 110)

        # ── (B) Clean Ranker 5,000-User Full Split Evaluation ────────────────
        print("\n" + "=" * 110)
        print("🏆 PART B: Clean Ranker Anonymous Serving Path vs Fresh Offline Evaluation (5,000 Users)")
        print("=" * 110)

        ranker_serving_results: dict[str, Any] = {}
        ranker_model = service.models.get(ranker_key)

        if ranker_model is not None:
            from recsys.metrics import compute_ranks

            val_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
            test_map = dict(zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True))

            for split_name in ["val", "test"]:
                # Fresh offline evaluation
                fresh_eval = evaluate_model(ranker_model, data, split=split_name, user_indices=eval_users)
                fresh_ranks = fresh_eval.per_user["rank"]
                fresh_ndcgs = ndcg_at_k(fresh_ranks, 10)
                fresh_hrs = hit_rate_at_k(fresh_ranks, 10)
                off_ndcg = float(np.mean(fresh_ndcgs))
                off_hr = float(np.mean(fresh_hrs))

                # 1. Serving path scoring with anonymous imputed user features & [-50:] history
                histories = []
                targets = []
                seen_masks = []
                for u in eval_users:
                    u_int = int(u)
                    pos = user_train_pos.get(u_int, [])
                    if split_name == "val":
                        h = list(pos[-50:]) if len(pos) > 50 else list(pos)
                        t = val_map.get(u_int, 0)
                    else:
                        full_pos = list(pos) + ([val_map[u_int]] if u_int in val_map else [])
                        h = list(full_pos[-50:]) if len(full_pos) > 50 else list(full_pos)
                        t = test_map.get(u_int, 0)
                    histories.append(np.asarray(h, dtype=np.int64))
                    targets.append(t)
                    seen_masks.append(set(h))

                n_users_eval = len(histories)
                scores = np.empty((n_users_eval, data.n_items), dtype=np.float32)
                for b_i in range(0, n_users_eval, 256):
                    b_slice = slice(b_i, min(b_i + 256, n_users_eval))
                    scores[b_slice] = ranker_model.score(histories[b_slice])
                for i, sm in enumerate(seen_masks):
                    scores[i, list(sm)] = -np.inf

                targets_arr = np.asarray(targets, dtype=np.int64)
                serving_ranks = compute_ranks(scores, targets_arr)

                # Confirm service.recommend() produces identical top-10 recommendations to serving scores
                sample_h_apps = [int(service.idx_to_app_id[idx]) for idx in histories[0] if idx in service.idx_to_app_id]
                rec_srv = service.recommend(liked_app_ids=sample_h_apps, model=ranker_key, k=10, explain=False)
                srv_items = [r["app_id"] for r in rec_srv["items"]]
                sc_top10 = [int(service.idx_to_app_id[i]) for i in np.argsort(-scores[0])[:10]]
                if srv_items != sc_top10:
                    raise AssertionError(f"service.recommend() mismatch with serving scores: {srv_items} vs {sc_top10}")

                serving_ndcgs = ndcg_at_k(serving_ranks, 10)
                serving_hrs = hit_rate_at_k(serving_ranks, 10)
                serv_ndcg = float(np.mean(serving_ndcgs))
                serv_hr = float(np.mean(serving_hrs))

                diff, lo95, hi95 = paired_bootstrap_diff(serving_ndcgs, fresh_ndcgs, seed=self.seed)

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
                    f"  Split: {split_name.upper():<4} | Fresh Eval NDCG@10: {off_ndcg:.4f} (HR@10: {off_hr:.4f}) | "
                    f"Serving NDCG@10: {serv_ndcg:.4f} (HR@10: {serv_hr:.4f}) | Paired Diff: {diff:+.4f} [{lo95:+.4f}, {hi95:+.4f}]"
                )

            # Check if serving degradation exceeds 0.003
            test_diff = ranker_serving_results.get("test", {}).get("diff", 0.0)
            if test_diff < -0.003:
                service.ranker_is_preview = True
                self.logger.warning("Ranker serving test NDCG is > 0.003 below offline. Marked as preview.")
            else:
                service.ranker_is_preview = False
                self.logger.info("Ranker serving test NDCG meets parity criteria. Status: NORMAL.")

        # ── (C) Latency Profiling (100 Distinct Histories, CPU Budget: P95 < 1.0s)
        print("\n" + "=" * 110)
        print("⚡ PART C: Latency Profiling (100 Distinct User Histories per Model, CPU Budget: P95 < 1.0s)")
        print("=" * 110)

        # Build 100 distinct user histories from catalog interactions
        distinct_histories: list[list[int]] = []
        for _u_int, pos in user_train_pos.items():
            if len(pos) >= 3:
                h_apps = [int(service.idx_to_app_id[idx]) for idx in pos[-10:] if idx in service.idx_to_app_id]
                if h_apps and h_apps not in distinct_histories:
                    distinct_histories.append(h_apps)
            if len(distinct_histories) >= 100:
                break

        if len(distinct_histories) < 100:
            rng = np.random.default_rng(self.seed)
            all_aids = list(service.app_id_to_idx.keys())
            while len(distinct_histories) < 100:
                h_len = int(rng.integers(3, 15))
                distinct_histories.append([int(x) for x in rng.choice(all_aids, size=h_len, replace=False)])

        latency_results: dict[str, Any] = {}

        for m_name in list(service.models.keys()):
            for explain_flag in [False, True]:
                latencies_ms: list[float] = []
                for h_apps in distinct_histories:
                    t0 = time.perf_counter()
                    rec_res = service.recommend(liked_app_ids=h_apps, model=m_name, k=20, explain=explain_flag)
                    dt_ms = (time.perf_counter() - t0) * 1000
                    latencies_ms.append(dt_ms)

                    # Verify resolved model name matches expectation
                    resolved_canonical = service.resolve_model_name(m_name)
                    assert rec_res["model_name"] == resolved_canonical, (
                        f"Resolved model mismatch: expected {resolved_canonical}, got {rec_res['model_name']}"
                    )

                    # Verify that explain=True computes SHAP for ranker (assert took_ms > 20)
                    if "ranker" in m_name and explain_flag:
                        took_ms = rec_res["took_ms"]
                        assert took_ms > 20, (
                            f"Expected took_ms > 20 for {m_name} when explain=True (SHAP calculation), but got {took_ms:.2f} ms"
                        )

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
