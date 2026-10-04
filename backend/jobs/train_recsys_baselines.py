"""
Training & Evaluation pipeline for Recommender Baselines — Phase 6.

Subclasses BaseJob.
Trains baseline recommender models (Popularity, Item-kNN, ALS, Content), evaluates them
on validation (and optionally test), serializes model artifacts, logs candidate runs to
MLflow & PostgreSQL model_runs, and prints comparative leaderboards, paired bootstrap differences,
and slice breakdowns.

Usage:
  python -m jobs.train_recsys_baselines --models popularity,itemknn,als,content
  python -m jobs.train_recsys_baselines --models all --evaluate-test
  python -m jobs.train_recsys_baselines --profile dev --trials 6 --evaluate-test
  python -m jobs.train_recsys_baselines --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import EvaluationResult, evaluate_model, get_evaluation_user_split
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender, tune_als_hyperparameters
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.registry import log_recsys_run

MODEL_REGISTRY: dict[str, tuple[type[BaseRecommender], str]] = {
    "popularity": (PopularityRecommender, "popularity"),
    "itemknn": (ItemKNNRecommender, "item_knn"),
    "als": (ALSRecommender, "matrix_factorization"),
    "content": (ContentRecommender, "content"),
}


class TrainRecsysBaselinesJob(BaseJob):
    job_name = "train_recsys_baselines"

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        evaluate_test: bool = False,
        trials: int | None = None,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        raw_models = models or ["popularity"]
        if len(raw_models) == 1 and raw_models[0].lower() == "all":
            self.model_keys = ["popularity", "itemknn", "als", "content"]
        else:
            expanded = []
            for m in raw_models:
                if m.lower() == "all":
                    expanded.extend(["popularity", "itemknn", "als", "content"])
                else:
                    expanded.append(m.lower())
            # Deduplicate while preserving order
            self.model_keys = list(dict.fromkeys(expanded))

        self.evaluate_test = evaluate_test
        self.trials = trials
        self.profile = profile
        self.data_dir = data_dir
        self.seed = seed

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = self.seed if self.seed is not None else cfg.seed
        num_trials = self.trials if self.trials is not None else (6 if cfg.profile == "dev" else 12)

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recommender dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # Disjoint evaluation and tuning user splits (Addendum 2)
        n_tuning = cfg.eval_users if cfg.profile == "dev" else 10000
        eval_users, tuning_users = get_evaluation_user_split(
            data,
            n_eval_users=cfg.eval_users,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        self.logger.info(
            "Dataset ready: %d users, %d items, %d train rows. Split: %d eval users, %d tuning users (seed=%d)",
            data.n_users,
            data.n_items,
            len(data.train_df),
            len(eval_users),
            len(tuning_users),
            seed,
        )

        leaderboard_records: list[dict[str, Any]] = []
        trained_results: dict[str, Any] = {}
        pop_per_user_val: dict[str, Any] = {}
        pop_per_user_test: dict[str, Any] = {}

        for model_key in self.model_keys:
            clean_key = model_key.strip().lower()
            if clean_key not in MODEL_REGISTRY:
                self.logger.warning("Unknown model '%s', skipping. Available: %s", clean_key, list(MODEL_REGISTRY.keys()))
                continue

            model_cls, family = MODEL_REGISTRY[clean_key]
            self.logger.info("═══════════════════════════════════════════════════════════")
            self.logger.info("▶ Initializing and training model: %s (family: %s)...", clean_key, family)

            train_start = time.perf_counter()
            model: BaseRecommender

            if clean_key == "popularity":
                model = PopularityRecommender()
                model.fit(data)

            elif clean_key == "itemknn":
                # Tune k_neighbors on disjoint tuning users
                self.logger.info("Tuning Item-kNN k_neighbors in {50, 100, 200} on %d tuning users...", len(tuning_users))
                best_k = 100
                best_score = -1.0
                for k_cand in [50, 100, 200]:
                    cand = ItemKNNRecommender(k_neighbors=k_cand, shrink=10.0)
                    cand.fit(data)
                    cand_res = evaluate_model(
                        cand,
                        data,
                        split="val",
                        k_list=(10,),
                        user_indices=tuning_users,
                        seed=seed,
                        allow_small_user_set=(len(tuning_users) < 1000),
                    )
                    cand_ndcg = cand_res.metrics.get("ndcg_at_10", 0.0)
                    self.logger.info("  Item-kNN k=%d -> Val NDCG@10 on tuning users: %.4f", k_cand, cand_ndcg)
                    if cand_ndcg > best_score:
                        best_score = cand_ndcg
                        best_k = k_cand

                self.logger.info("Selected best Item-kNN k_neighbors=%d", best_k)
                model = ItemKNNRecommender(k_neighbors=best_k, shrink=10.0)
                model.fit(data)

            elif clean_key == "als":
                # Optuna hyperparameter search on tuning users
                if num_trials > 0:
                    best_params, _ = tune_als_hyperparameters(
                        data=data,
                        tuning_users=tuning_users,
                        n_trials=num_trials,
                        seed=seed,
                    )
                    model = ALSRecommender(**best_params, iterations=15, seed=seed)
                else:
                    model = ALSRecommender(factors=64, regularization=0.05, alpha=15.0, iterations=15, seed=seed)

                model.fit(data)

            elif clean_key == "content":
                model = ContentRecommender(weight_tags=0.5)
                model.fit(data)

            else:
                model = model_cls()
                model.fit(data)

            train_elapsed = time.perf_counter() - train_start
            self.logger.info("Trained %s in %.2fs", model.name, train_elapsed)

            # 1. Validation Evaluation (on fixed evaluation users)
            self.logger.info("Evaluating %s on validation split (%d users)...", model.name, len(eval_users))
            val_res: EvaluationResult = evaluate_model(
                model=model,
                data=data,
                split="val",
                k_list=(10, 20),
                user_indices=eval_users,
                max_history=cfg.max_history,
                seed=seed,
            )

            test_res: EvaluationResult | None = None
            if self.evaluate_test:
                self.logger.info("Evaluating %s on test split (--evaluate-test enabled, %d users)...", model.name, len(eval_users))
                test_res = evaluate_model(
                    model=model,
                    data=data,
                    split="test",
                    k_list=(10, 20),
                    user_indices=eval_users,
                    max_history=cfg.max_history,
                    seed=seed,
                )

            # Save baseline per-user outputs for paired bootstrap difference
            if clean_key == "popularity":
                pop_per_user_val = val_res.per_user
                if test_res is not None:
                    pop_per_user_test = test_res.per_user

            # 2. Serialize Model Artifacts
            run_uuid = uuid.uuid4()
            artifact_dir = self.resolve_path(
                "..", "data", "recsys", "artifacts", model.name, str(run_uuid)
            ).resolve()

            if not self.dry_run:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model.save(artifact_dir)
                if "rank" in val_res.per_user:
                    np.save(artifact_dir / "ranks_val.npy", val_res.per_user["rank"])
                if test_res is not None and "rank" in test_res.per_user:
                    np.save(artifact_dir / "ranks_test.npy", test_res.per_user["rank"])
                self.logger.info("Saved model artifacts to %s", artifact_dir)

                # 3. Log to MLflow & Database (stage='candidate', no promote_champion per Addendum 3)
                await log_recsys_run(
                    model_name=model.name,
                    family=family,
                    dataset_version=data.dataset_version,
                    hyperparameters=model.hyperparameters(),
                    val_metrics=val_res,
                    test_metrics=test_res,
                    artifact_dir=artifact_dir,
                    extra={
                        "seed": seed,
                        "profile": cfg.profile,
                        "eval_users": len(eval_users),
                        "tuning_users": len(tuning_users),
                        "train_time_sec": round(train_elapsed, 2),
                    },
                    session=session,
                )

            # Append to leaderboard records
            leaderboard_records.append({
                "model_key": clean_key,
                "model_name": model.name,
                "family": family,
                "train_time": train_elapsed,
                "val_res": val_res,
                "test_res": test_res,
            })

            trained_results[clean_key] = {
                "val_metrics": val_res.metrics,
                "test_metrics": test_res.metrics if test_res else None,
                "train_time_sec": train_elapsed,
                "artifact_dir": str(artifact_dir),
            }

        # 4. Print Comparative Tables
        self._print_full_evaluation_report(
            records=leaderboard_records,
            pop_per_user_val=pop_per_user_val,
            pop_per_user_test=pop_per_user_test,
            dataset_version=data.dataset_version,
            eval_users_count=len(eval_users),
            seed=seed,
        )

        return {
            "models_trained": len(trained_results),
            "results": trained_results,
            "dataset_version": data.dataset_version,
        }

    def _print_full_evaluation_report(
        self,
        records: list[dict[str, Any]],
        pop_per_user_val: dict[str, Any],
        pop_per_user_test: dict[str, Any],
        dataset_version: str,
        eval_users_count: int,
        seed: int,
    ) -> None:
        """
        Prints the complete report: main leaderboard, paired bootstrap differences vs popularity,
        and slice breakdown tables.
        """
        # Sort records by validation NDCG@10 descending
        sorted_records = sorted(
            records,
            key=lambda r: r["val_res"].metrics.get("ndcg_at_10", 0.0),
            reverse=True,
        )

        div_main = "=" * 148
        sub_main = "-" * 148

        # ── TABLE 1: MAIN LEADERBOARD ─────────────────────────────────────────
        print(f"\n{div_main}")
        print(f"🏆 SteamIQ Recommender Leaderboard | Dataset: {dataset_version} | Eval Users: {eval_users_count} | Seed: {seed}")
        print(div_main)
        print(
            f"{'Model':<18} | {'Split':<5} | {'HR@10 (95% CI)':<23} | {'HR@20 (95% CI)':<23} | "
            f"{'NDCG@10 (95% CI)':<23} | {'NDCG@20 (95% CI)':<23} | {'MRR@10':<7} | {'Cov@10':<7} | {'Nov@10':<7} | {'Div@10':<7} | {'Time (s)':<8}"
        )
        print(sub_main)

        for rec in sorted_records:
            name = rec["model_name"]
            t_time = rec["train_time"]

            # Val row
            vm = rec["val_res"].metrics
            v_hr10 = f"{vm.get('hit_rate_at_10', 0.0):.4f} [{vm.get('hit_rate_at_10_lo95', 0.0):.4f},{vm.get('hit_rate_at_10_hi95', 0.0):.4f}]"
            v_hr20 = f"{vm.get('hit_rate_at_20', 0.0):.4f} [{vm.get('hit_rate_at_20_lo95', 0.0):.4f},{vm.get('hit_rate_at_20_hi95', 0.0):.4f}]"
            v_ndcg10 = f"{vm.get('ndcg_at_10', 0.0):.4f} [{vm.get('ndcg_at_10_lo95', 0.0):.4f},{vm.get('ndcg_at_10_hi95', 0.0):.4f}]"
            v_ndcg20 = f"{vm.get('ndcg_at_20', 0.0):.4f} [{vm.get('ndcg_at_20_lo95', 0.0):.4f},{vm.get('ndcg_at_20_hi95', 0.0):.4f}]"
            v_mrr10 = f"{vm.get('mrr_at_10', 0.0):.4f}"
            v_cov10 = f"{vm.get('catalog_coverage_at_10', 0.0):.4f}"
            v_nov10 = f"{vm.get('novelty_at_10', 0.0):.4f}"
            v_div10 = f"{vm.get('diversity_at_10', 0.0):.4f}"

            print(
                f"{name:<18} | {'val':<5} | {v_hr10:<23} | {v_hr20:<23} | "
                f"{v_ndcg10:<23} | {v_ndcg20:<23} | {v_mrr10:<7} | {v_cov10:<7} | {v_nov10:<7} | {v_div10:<7} | {t_time:<8.2f}"
            )

            # Test row (if evaluated)
            if rec["test_res"] is not None:
                tm = rec["test_res"].metrics
                t_hr10 = f"{tm.get('hit_rate_at_10', 0.0):.4f} [{tm.get('hit_rate_at_10_lo95', 0.0):.4f},{tm.get('hit_rate_at_10_hi95', 0.0):.4f}]"
                t_hr20 = f"{tm.get('hit_rate_at_20', 0.0):.4f} [{tm.get('hit_rate_at_20_lo95', 0.0):.4f},{tm.get('hit_rate_at_20_hi95', 0.0):.4f}]"
                t_ndcg10 = f"{tm.get('ndcg_at_10', 0.0):.4f} [{tm.get('ndcg_at_10_lo95', 0.0):.4f},{tm.get('ndcg_at_10_hi95', 0.0):.4f}]"
                t_ndcg20 = f"{tm.get('ndcg_at_20', 0.0):.4f} [{tm.get('ndcg_at_20_lo95', 0.0):.4f},{tm.get('ndcg_at_20_hi95', 0.0):.4f}]"
                t_mrr10 = f"{tm.get('mrr_at_10', 0.0):.4f}"
                t_cov10 = f"{tm.get('catalog_coverage_at_10', 0.0):.4f}"
                t_nov10 = f"{tm.get('novelty_at_10', 0.0):.4f}"
                t_div10 = f"{tm.get('diversity_at_10', 0.0):.4f}"

                print(
                    f"{name:<18} | {'test':<5} | {t_hr10:<23} | {t_hr20:<23} | "
                    f"{t_ndcg10:<23} | {t_ndcg20:<23} | {t_mrr10:<7} | {t_cov10:<7} | {t_nov10:<7} | {t_div10:<7} | {'-':<8}"
                )

        print(f"{div_main}\n")

        # ── TABLE 2: PAIRED BOOTSTRAP DIFFERENCE VS POPULARITY ─────────────────
        if pop_per_user_val:
            div_paired = "=" * 105
            sub_paired = "-" * 105
            print(f"{div_paired}")
            print("📊 Paired Bootstrap Significance vs Popularity Baseline (Mean Diff & 95% CI)")
            print(f"{div_paired}")
            print(
                f"{'Model':<18} | {'Split':<5} | {'NDCG@10 Diff vs Popularity':<30} | "
                f"{'HR@10 Diff vs Popularity':<30} | {'Beats Pop?':<10}"
            )
            print(f"{sub_paired}")

            for rec in sorted_records:
                name = rec["model_name"]
                if rec["model_key"] == "popularity":
                    print(
                        f"{name:<18} | {'val':<5} | {'0.0000 [baseline]':<30} | "
                        f"{'0.0000 [baseline]':<30} | {'Baseline':<10}"
                    )
                    continue

                # Val Diff
                m_val_ndcg = rec["val_res"].per_user.get("ndcg_at_10")
                p_val_ndcg = pop_per_user_val.get("ndcg_at_10")
                m_val_hr = rec["val_res"].per_user.get("hit_rate_at_10")
                p_val_hr = pop_per_user_val.get("hit_rate_at_10")

                if m_val_ndcg is not None and p_val_ndcg is not None:
                    d_ndcg, ndcg_lo, ndcg_hi = paired_bootstrap_diff(m_val_ndcg, p_val_ndcg, n_boot=1000, seed=seed)
                    d_hr, hr_lo, hr_hi = paired_bootstrap_diff(m_val_hr, p_val_hr, n_boot=1000, seed=seed)

                    ndcg_diff_str = f"{d_ndcg:+.4f} [{ndcg_lo:+.4f}, {ndcg_hi:+.4f}]"
                    hr_diff_str = f"{d_hr:+.4f} [{hr_lo:+.4f}, {hr_hi:+.4f}]"
                    beats = "YES (p<0.05)" if ndcg_lo > 0.0 else ("YES" if d_ndcg > 0.0 else "NO")

                    print(
                        f"{name:<18} | {'val':<5} | {ndcg_diff_str:<30} | {hr_diff_str:<30} | {beats:<10}"
                    )

                # Test Diff
                if rec["test_res"] is not None and pop_per_user_test:
                    m_test_ndcg = rec["test_res"].per_user.get("ndcg_at_10")
                    p_test_ndcg = pop_per_user_test.get("ndcg_at_10")
                    m_test_hr = rec["test_res"].per_user.get("hit_rate_at_10")
                    p_test_hr = pop_per_user_test.get("hit_rate_at_10")

                    if m_test_ndcg is not None and p_test_ndcg is not None:
                        td_ndcg, t_ndcg_lo, t_ndcg_hi = paired_bootstrap_diff(m_test_ndcg, p_test_ndcg, n_boot=1000, seed=seed)
                        td_hr, t_hr_lo, t_hr_hi = paired_bootstrap_diff(m_test_hr, p_test_hr, n_boot=1000, seed=seed)

                        t_ndcg_diff_str = f"{td_ndcg:+.4f} [{t_ndcg_lo:+.4f}, {t_ndcg_hi:+.4f}]"
                        t_hr_diff_str = f"{td_hr:+.4f} [{t_hr_lo:+.4f}, {t_hr_hi:+.4f}]"
                        t_beats = "YES (p<0.05)" if t_ndcg_lo > 0.0 else ("YES" if td_ndcg > 0.0 else "NO")

                        print(
                            f"{name:<18} | {'test':<5} | {t_ndcg_diff_str:<30} | {t_hr_diff_str:<30} | {t_beats:<10}"
                        )

            print(f"{div_paired}\n")

        # ── TABLE 3: SLICE PERFORMANCE BREAKDOWN (NDCG@10) ────────────────────
        div_slice = "=" * 115
        sub_slice = "-" * 115
        print(f"{div_slice}")
        print("🔍 Sliced Evaluation Breakdown (NDCG@10 across Popularity & History Length Buckets)")
        print(f"{div_slice}")
        print(
            f"{'Model':<18} | {'Split':<5} | {'Pop Head (20%)':<14} | {'Pop Torso (30%)':<15} | "
            f"{'Pop Tail (50%)':<14} | {'Hist 3-4':<11} | {'Hist 5-9':<11} | {'Hist 10+':<11}"
        )
        print(f"{sub_slice}")

        for rec in sorted_records:
            name = rec["model_name"]
            vs = rec["val_res"].slices

            vh = vs.get("pop_head", {}).get("ndcg_at_10", 0.0)
            vtr = vs.get("pop_torso", {}).get("ndcg_at_10", 0.0)
            vtl = vs.get("pop_tail", {}).get("ndcg_at_10", 0.0)
            vh34 = vs.get("hist_3_4", {}).get("ndcg_at_10", 0.0)
            vh59 = vs.get("hist_5_9", {}).get("ndcg_at_10", 0.0)
            vh10 = vs.get("hist_10_plus", {}).get("ndcg_at_10", 0.0)

            print(
                f"{name:<18} | {'val':<5} | {vh:<14.4f} | {vtr:<15.4f} | {vtl:<14.4f} | "
                f"{vh34:<11.4f} | {vh59:<11.4f} | {vh10:<11.4f}"
            )

            if rec["test_res"] is not None:
                ts = rec["test_res"].slices
                th = ts.get("pop_head", {}).get("ndcg_at_10", 0.0)
                ttr = ts.get("pop_torso", {}).get("ndcg_at_10", 0.0)
                ttl = ts.get("pop_tail", {}).get("ndcg_at_10", 0.0)
                th34 = ts.get("hist_3_4", {}).get("ndcg_at_10", 0.0)
                th59 = ts.get("hist_5_9", {}).get("ndcg_at_10", 0.0)
                th10 = ts.get("hist_10_plus", {}).get("ndcg_at_10", 0.0)

                print(
                    f"{name:<18} | {'test':<5} | {th:<14.4f} | {ttr:<15.4f} | {ttl:<14.4f} | "
                    f"{th34:<11.4f} | {th59:<11.4f} | {th10:<11.4f}"
                )

        print(f"{div_slice}\n")

    def run(self) -> dict[str, Any]:
        if self.dry_run:
            return asyncio.run(self._run_async(session=None))

        async def _execute_with_engine() -> dict[str, Any]:
            engine = create_async_engine(settings.async_database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                res = await self._run_async(session=session)
            await engine.dispose()
            return res

        return asyncio.run(_execute_with_engine())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate recommender baselines with MLflow & PostgreSQL registration"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models (popularity,itemknn,als,content) or 'all'",
    )
    parser.add_argument(
        "--evaluate-test",
        "--test",
        action="store_true",
        help="Compute final test metrics on test.parquet split",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna tuning trials for ALS (default: 6 for dev, 12 for full)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["dev", "full"],
        help="Recsys execution profile override",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to precomputed recsys data directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override for evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training & evaluation without writing to MLflow or DB",
    )
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(",") if m.strip()]

    job = TrainRecsysBaselinesJob(
        models=models_list,
        evaluate_test=args.evaluate_test,
        trials=args.trials,
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
