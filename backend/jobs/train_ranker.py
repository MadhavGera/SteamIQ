"""
Job to train, evaluate, and register the Stage-Two Learning-to-Rank Recommender.

Implements:
  1. Multi-source candidate recall assessment on fixed evaluation users using SAVED ARTIFACTS (zero re-fitting).
  2. Training of LightGBM LambdaMART ranker on disjoint ranker users.
  3. Paired bootstrap significance testing vs Two-Tower, Item-kNN, ALS, and Popularity.
  4. Sliced evaluation tables with sample sizes (n) and 95% bootstrap CIs.
  5. Artifact serialization (booster, SHAP, feature importances).
  6. Conditional champion promotion based on statistically significant VAL NDCG@10 improvement.
  7. Cleanup and archival of historical model runs with invalid user counts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun
from jobs.base_job import BaseJob
from recsys.candidates import (
    DEFAULT_CANDIDATE_COUNTS,
    compute_candidate_recalls,
    generate_candidates,
)
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import (
    EvaluationResult,
    _build_user_interaction_maps,
    evaluate_model,
    get_evaluation_user_split,
)
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerConfig, train_ranker
from recsys.registry import log_recsys_run


def _find_latest_artifact_dir(base_dir: Path, model_name: str) -> tuple[Path, str]:
    """
    Finds the latest serialized model artifact directory for a given model name.
    Raises FileNotFoundError with a clear message if missing.
    """
    art_dir = base_dir / "artifacts" / model_name
    if not art_dir.exists():
        raise FileNotFoundError(
            f"Missing required Stage-One artifact directory for '{model_name}': {art_dir}. "
            f"Please run the corresponding training job before training the ranker."
        )
    cand_dirs = sorted(
        [d for d in art_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not cand_dirs:
        raise FileNotFoundError(
            f"No serialized run artifacts found in {art_dir} for '{model_name}'. "
            f"Please train and serialize the model first."
        )
    latest_dir = cand_dirs[0]
    return latest_dir, latest_dir.name


class TrainRankerJob(BaseJob):
    job_name = "train_ranker"

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        n_ranker_users: int = 15000,
        evaluate_test: bool = False,
        clean: bool = False,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_ranker_users = n_ranker_users
        self.evaluate_test = evaluate_test
        self.clean = clean
        self.profile = profile
        self.data_dir = data_dir
        self.seed = seed

    async def _cleanup_old_runs(self, session: AsyncSession) -> list[dict[str, Any]]:
        """Archive old recsys model runs created before the user-split fix."""
        stmt = select(ModelRun).where(ModelRun.model_name.like("recsys_%")).order_by(ModelRun.created_at.desc())
        res = await session.execute(stmt)
        runs = res.scalars().all()
        archived_count = 0
        status_list: list[dict[str, Any]] = []

        for r in runs:
            m = r.metrics or {}
            hp = r.hyperparameters or {}
            n_eval = int(m.get("val_n_users", m.get("n_eval_users", m.get("n_users", 0))))
            is_inv = m.get("invalid_eval_set", hp.get("invalid_eval_set", False))

            if n_eval < 1000 and r.stage != "archived":
                new_m = dict(m)
                new_m["invalid_eval_set"] = True
                new_hp = dict(hp)
                new_hp["invalid_eval_set"] = True

                await session.execute(
                    update(ModelRun)
                    .where(ModelRun.id == r.id)
                    .values(stage="archived", metrics=new_m, hyperparameters=new_hp)
                )
                archived_count += 1
                r.stage = "archived"
                is_inv = True

            status_list.append({
                "id": str(r.id),
                "model_name": r.model_name,
                "stage": r.stage,
                "n_eval_users": n_eval,
                "invalid_eval_set": is_inv,
            })

        if archived_count > 0:
            await session.commit()
            self.logger.info("Archived %d historical model runs with invalid user counts (< 1,000).", archived_count)

        return status_list

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = self.seed if self.seed is not None else cfg.seed

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recommender dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # 1. Cleanup old invalid runs if session is available
        cleanup_status: list[dict[str, Any]] = []
        if session is not None and not self.dry_run:
            cleanup_status = await self._cleanup_old_runs(session)

        # 2. Load Disjoint Evaluation and Tuning User Splits
        n_tuning = cfg.eval_users if cfg.profile == "dev" else 10000
        eval_users, tuning_users = get_evaluation_user_split(
            data,
            n_eval_users=cfg.eval_users,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        self.logger.info(
            "Dataset ready: %d users, %d items. Splits: %d eval users, %d tuning users.",
            data.n_users,
            data.n_items,
            len(eval_users),
            len(tuning_users),
        )

        # 3. Load Stage-One Model Artifacts (NO RE-FITTING)
        self.logger.info("═══════════════════════════════════════════════════════════")
        self.logger.info("▶ Loading Stage-One Retrieval Artifacts (Zero Re-fitting)...")

        tt_lookup = "recsys_two_tower_clean" if self.clean else "recsys_two_tower"
        tt_dir, tt_run_id = _find_latest_artifact_dir(resolved_data_dir, tt_lookup)
        tt_model = TwoTowerModel.load(tt_dir, data=data)
        self.logger.info("Loaded Two-Tower artifact from %s (run: %s, variant: %s)", tt_dir, tt_run_id, tt_lookup)

        knn_dir, knn_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_itemknn")
        knn_model = ItemKNNRecommender.load(knn_dir)
        self.logger.info("Loaded Item-kNN artifact from %s (run: %s)", knn_dir, knn_run_id)

        als_dir, als_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_als")
        als_model = ALSRecommender.load(als_dir)
        self.logger.info("Loaded ALS artifact from %s (run: %s)", als_dir, als_run_id)

        content_dir, content_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_content")
        content_model = ContentRecommender.load(content_dir)
        self.logger.info("Loaded Content artifact from %s (run: %s)", content_dir, content_run_id)

        pop_dir, pop_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_popularity")
        pop_model = PopularityRecommender.load(pop_dir)
        self.logger.info("Loaded Popularity artifact from %s (run: %s)", pop_dir, pop_run_id)

        source_run_ids = {
            "two_tower": tt_run_id,
            "itemknn": knn_run_id,
            "als": als_run_id,
            "content": content_run_id,
            "popularity": pop_run_id,
        }

        stage1_models = {
            "two_tower": tt_model,
            "itemknn": knn_model,
            "als": als_model,
            "content": content_model,
        }

        # 4. Compute and Report Candidate Recalls on Evaluation Users (VAL split)
        self.logger.info("═══════════════════════════════════════════════════════════")
        self.logger.info("▶ Measuring Multi-Source Candidate Recall on Evaluation Users (VAL Split)...")
        user_train_pos, user_train_neg = _build_user_interaction_maps(data)
        val_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
        eval_val_targets = [int(val_map[int(u)]) for u in eval_users]
        eval_histories = [np.asarray(user_train_pos.get(int(u), []), dtype=np.int64) for u in eval_users]
        eval_seen_masks = [set(user_train_pos.get(int(u), [])).union(user_train_neg.get(int(u), set())) for u in eval_users]

        eval_cand_res = generate_candidates(
            histories=eval_histories,
            models=stage1_models,
            n_each=DEFAULT_CANDIDATE_COUNTS,
            seen_masks=eval_seen_masks,
            max_candidates=300,
            batch_size=512,
        )
        recall_dict = compute_candidate_recalls(eval_cand_res, eval_val_targets)

        # 5. Train Learning-to-Rank LightGBM Model
        ranker_cfg = RankerConfig(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples,
            colsample_bytree=0.8,
            lambdarank_truncation_level=20,
            eval_at=[10],
            seed=seed,
            n_jobs=cfg.num_threads,
            max_candidates=300,
            source_run_ids=source_run_ids,
        )

        ranker_model, train_info = train_ranker(
            data=data,
            models=stage1_models,
            cfg=ranker_cfg,
            n_ranker_users=self.n_ranker_users,
            eval_users=eval_users,
            tuning_users=tuning_users,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        # 6. Comprehensive Evaluation on Fixed Evaluation Users (5,000)
        self.logger.info("═══════════════════════════════════════════════════════════")
        self.logger.info("▶ Evaluating All Models on Evaluation Users (%d users)...", len(eval_users))

        ranker_model_key = "recsys_two_tower_ranker_clean_v2" if self.clean else "recsys_two_tower_ranker"
        tt_model_key = "recsys_two_tower_clean" if self.clean else "recsys_two_tower"

        all_models_dict = {
            ranker_model_key: ranker_model,
            tt_model_key: tt_model,
            "recsys_itemknn": knn_model,
            "recsys_als": als_model,
            "recsys_popularity": pop_model,
            "recsys_content": content_model,
        }

        if self.clean:
            try:
                old_rk_dir, old_rk_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_two_tower_ranker_clean")
                old_ranker_model = RankerModel.load(old_rk_dir, data=data, models=stage1_models)
                all_models_dict["recsys_two_tower_ranker_clean"] = old_ranker_model
                self.logger.info("Loaded old clean ranker from %s (run: %s) for comparative evaluation", old_rk_dir, old_rk_run_id)
            except Exception as e:
                self.logger.warning("Could not load old clean ranker for comparative evaluation: %s", e)

        val_results: dict[str, EvaluationResult] = {}
        test_results: dict[str, EvaluationResult] = {}

        for m_key, m_instance in all_models_dict.items():
            self.logger.info("Evaluating %s on VAL split...", m_key)
            val_res = evaluate_model(
                model=m_instance,
                data=data,
                split="val",
                k_list=(10, 20),
                user_indices=eval_users,
                max_history=cfg.max_history,
                seed=seed,
                allow_small_user_set=self.dry_run or len(eval_users) < 1000,
            )
            val_results[m_key] = val_res

            if self.evaluate_test:
                self.logger.info("Evaluating %s on TEST split...", m_key)
                test_res = evaluate_model(
                    model=m_instance,
                    data=data,
                    split="test",
                    k_list=(10, 20),
                    user_indices=eval_users,
                    max_history=cfg.max_history,
                    seed=seed,
                    allow_small_user_set=self.dry_run or len(eval_users) < 1000,
                )
                test_results[m_key] = test_res

        # 7. Compute Paired Bootstrap Differences vs Two-Tower, Item-kNN, ALS, Popularity, and old Ranker
        diff_comparisons: list[dict[str, Any]] = []
        ranker_val_res = val_results[ranker_model_key]
        ranker_test_res = test_results.get(ranker_model_key)

        comparison_targets = [tt_model_key, "recsys_itemknn", "recsys_als", "recsys_popularity"]
        if self.clean and "recsys_two_tower_ranker_clean" in val_results:
            comparison_targets.append("recsys_two_tower_ranker_clean")

        for target_name in comparison_targets:
            base_val_res = val_results[target_name]
            diff_val_ndcg, lo_val_ndcg, hi_val_ndcg = paired_bootstrap_diff(
                ranker_val_res.per_user["ndcg_at_10"],
                base_val_res.per_user["ndcg_at_10"],
                seed=seed,
            )
            diff_val_hr, lo_val_hr, hi_val_hr = paired_bootstrap_diff(
                ranker_val_res.per_user["hit_rate_at_10"],
                base_val_res.per_user["hit_rate_at_10"],
                seed=seed,
            )
            diff_comparisons.append({
                "target": target_name.replace("recsys_", "").replace("_", " ").title(),
                "split": "val",
                "ndcg_diff": diff_val_ndcg,
                "ndcg_lo": lo_val_ndcg,
                "ndcg_hi": hi_val_ndcg,
                "hr_diff": diff_val_hr,
                "hr_lo": lo_val_hr,
                "hr_hi": hi_val_hr,
                "sig": lo_val_ndcg > 0 or hi_val_ndcg < 0,
            })

            if self.evaluate_test and ranker_test_res:
                base_test_res = test_results[target_name]
                diff_test_ndcg, lo_test_ndcg, hi_test_ndcg = paired_bootstrap_diff(
                    ranker_test_res.per_user["ndcg_at_10"],
                    base_test_res.per_user["ndcg_at_10"],
                    seed=seed,
                )
                diff_test_hr, lo_test_hr, hi_test_hr = paired_bootstrap_diff(
                    ranker_test_res.per_user["hit_rate_at_10"],
                    base_test_res.per_user["hit_rate_at_10"],
                    seed=seed,
                )
                diff_comparisons.append({
                    "target": target_name.replace("recsys_", "").replace("_", " ").title(),
                    "split": "test",
                    "ndcg_diff": diff_test_ndcg,
                    "ndcg_lo": lo_test_ndcg,
                    "ndcg_hi": hi_test_ndcg,
                    "hr_diff": diff_test_hr,
                    "hr_lo": lo_test_hr,
                    "hr_hi": hi_test_hr,
                    "sig": lo_test_ndcg > 0 or hi_test_ndcg < 0,
                })

        # 8. Check Qualification for Champion Promotion (Addendum 6)
        tt_val_res = val_results[tt_model_key]
        val_diff_vs_tt, val_lo_vs_tt, val_hi_vs_tt = paired_bootstrap_diff(
            ranker_val_res.per_user["ndcg_at_10"],
            tt_val_res.per_user["ndcg_at_10"],
            seed=seed,
        )

        qualifies_for_champion = bool(val_lo_vs_tt > 0.0)

        # 9. Register Model Run and Save Artifacts
        run_uuid = str(uuid.uuid4())
        model_name = ranker_model_key
        family = ranker_model_key
        artifact_dir = resolved_data_dir / "artifacts" / model_name / run_uuid
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ranker_model.save(artifact_dir)

        # Save additional JSON artifacts
        with open(artifact_dir / "feature_importance.json", "w", encoding="utf-8") as f:
            json.dump({
                "gain": train_info["feature_importance_gain"],
                "split": train_info["feature_importance_split"],
            }, f, indent=2)

        with open(artifact_dir / "global_shap.json", "w", encoding="utf-8") as f:
            json.dump(train_info["global_shap"], f, indent=2)

        with open(artifact_dir / "candidate_recalls.json", "w", encoding="utf-8") as f:
            json.dump(recall_dict, f, indent=2)

        # Log ModelRun
        extra_meta = {
            "n_eval_users": len(eval_users),
            "n_train_users": train_info["n_train_users"],
            "n_val_users": train_info["n_val_users"],
            "dropped_train_users": train_info["dropped_train_users"],
            "dropped_val_users": train_info["dropped_val_users"],
            "best_iteration": train_info["best_iteration"],
            "candidate_recalls": recall_dict,
            "source_run_ids": source_run_ids,
        }

        model_run_rec = None
        if session is not None and not self.dry_run:
            model_run_rec, mlflow_id = await log_recsys_run(
                model_name=model_name,
                family=family,
                dataset_version=data.dataset_version,
                hyperparameters=ranker_model.hyperparameters(),
                val_metrics=ranker_val_res,
                test_metrics=ranker_test_res,
                artifact_dir=artifact_dir,
                extra=extra_meta,
                session=session,
            )

            # Save rank arrays if available
            if "rank" in ranker_val_res.per_user:
                np.save(artifact_dir / "ranks_val.npy", ranker_val_res.per_user["rank"])
            if ranker_test_res is not None and "rank" in ranker_test_res.per_user:
                np.save(artifact_dir / "ranks_test.npy", ranker_test_res.per_user["rank"])

            self.logger.info(
                "ℹ️ Ranker logged as 'candidate' stage. Promotion is handled exclusively by 'python -m jobs.promote_recsys'."
            )

        # 10. Print Comprehensive Report Tables
        if cleanup_status:
            self._print_cleanup_table(cleanup_status)
        self._print_candidate_recalls_table(recall_dict)
        self._print_leaderboard_table(val_results, test_results, len(eval_users), data.dataset_version, seed)
        self._print_significance_table(diff_comparisons)
        self._print_sliced_breakdown_table(val_results, test_results, seed=seed)
        self._print_feature_importances_table(train_info["global_shap"], train_info["feature_importance_gain"])

        return {
            "model_name": "recsys_two_tower_ranker",
            "qualifies_for_champion": qualifies_for_champion,
            "candidate_recalls": recall_dict,
            "val_metrics": ranker_val_res.metrics,
            "test_metrics": ranker_test_res.metrics if ranker_test_res else None,
            "artifact_dir": str(artifact_dir),
            "dropped_train_users": train_info["dropped_train_users"],
            "source_run_ids": source_run_ids,
        }

    def _print_cleanup_table(self, cleanup_status: list[dict[str, Any]]) -> None:
        print("\n" + "=" * 120)
        print("🧹 Historical Recsys Model Runs Status & Cleanup Table")
        print("=" * 120)
        print(f"{'Model Run ID':<38} | {'Model Name':<25} | {'Stage':<10} | {'N Eval Users':<12} | {'Invalid Eval Set?'}")
        print("-" * 120)
        for r in cleanup_status:
            inv_str = "YES (invalid_eval_set: true)" if r["invalid_eval_set"] else "NO (valid)"
            print(f"{r['id']:<38} | {r['model_name']:<25} | {r['stage']:<10} | {r['n_eval_users']:<12} | {inv_str}")
        print("=" * 120 + "\n")

    def _print_candidate_recalls_table(self, recalls: dict[str, float]) -> None:
        print("\n" + "=" * 80)
        print("🎯 Stage-One Multi-Source Candidate Recall (VAL Split, 5,000 Eval Users)")
        print("=" * 80)
        print(f"  Two-Tower (Top 150):     {recalls.get('candidate_recall_two_tower', 0.0):.4f} ({recalls.get('candidate_recall_two_tower', 0.0)*100:.2f}%)")
        print(f"  Item-kNN  (Top 100):     {recalls.get('candidate_recall_itemknn', 0.0):.4f} ({recalls.get('candidate_recall_itemknn', 0.0)*100:.2f}%)")
        print(f"  ALS       (Top 50):      {recalls.get('candidate_recall_als', 0.0):.4f} ({recalls.get('candidate_recall_als', 0.0)*100:.2f}%)")
        print(f"  Content   (Top 50):      {recalls.get('candidate_recall_content', 0.0):.4f} ({recalls.get('candidate_recall_content', 0.0)*100:.2f}%)")
        print("-" * 80)
        print(f"  ✨ Multi-Source Union (300): {recalls.get('candidate_recall_union', 0.0):.4f} ({recalls.get('candidate_recall_union', 0.0)*100:.2f}%) [Ranker Ceiling]")
        print("=" * 80 + "\n")

    def _print_leaderboard_table(
        self,
        val_res: dict[str, EvaluationResult],
        test_res: dict[str, EvaluationResult],
        n_eval_users: int,
        dataset_version: str,
        seed: int,
    ) -> None:
        print("\n" + "=" * 140)
        print(f"🏆 SteamIQ Full Recommender Leaderboard (Stage 1 & Stage 2) | Dataset: {dataset_version} | Eval Users: {n_eval_users} | Seed: {seed}")
        print("=" * 140)
        print(f"{'Model':<25} | {'Split':<5} | {'HR@10 (95% CI)':<23} | {'HR@20 (95% CI)':<23} | {'NDCG@10 (95% CI)':<23} | {'NDCG@20 (95% CI)':<23} | {'MRR@10':<7} | {'Coverage':<8} | {'Diversity':<9}")
        print("-" * 140)

        # Order models
        order = [
            "recsys_two_tower_ranker_clean_v2",
            "recsys_two_tower_ranker",
            "recsys_two_tower_ranker_clean",
            "recsys_two_tower_clean",
            "recsys_two_tower",
            "recsys_itemknn",
            "recsys_als",
            "recsys_popularity",
            "recsys_content",
        ]
        for m_name in order:
            if m_name not in val_res:
                continue
            vr = val_res[m_name]
            v_hr10 = f"{vr.metrics.get('hit_rate_at_10', 0.0):.4f} [{vr.metrics.get('hit_rate_at_10_lo95', 0.0):.4f},{vr.metrics.get('hit_rate_at_10_hi95', 0.0):.4f}]"
            v_hr20 = f"{vr.metrics.get('hit_rate_at_20', 0.0):.4f} [{vr.metrics.get('hit_rate_at_20_lo95', 0.0):.4f},{vr.metrics.get('hit_rate_at_20_hi95', 0.0):.4f}]"
            v_nd10 = f"{vr.metrics.get('ndcg_at_10', 0.0):.4f} [{vr.metrics.get('ndcg_at_10_lo95', 0.0):.4f},{vr.metrics.get('ndcg_at_10_hi95', 0.0):.4f}]"
            v_nd20 = f"{vr.metrics.get('ndcg_at_20', 0.0):.4f} [{vr.metrics.get('ndcg_at_20_lo95', 0.0):.4f},{vr.metrics.get('ndcg_at_20_hi95', 0.0):.4f}]"
            v_mrr10 = f"{vr.metrics.get('mrr_at_10', 0.0):.4f}"
            v_cov = f"{vr.metrics.get('catalog_coverage_at_10', 0.0):.4f}"
            v_div = f"{vr.metrics.get('diversity_at_10', 0.0):.4f}"

            print(f"{m_name:<25} | {'val':<5} | {v_hr10:<23} | {v_hr20:<23} | {v_nd10:<23} | {v_nd20:<23} | {v_mrr10:<7} | {v_cov:<8} | {v_div:<9}")

            if m_name in test_res:
                tr = test_res[m_name]
                t_hr10 = f"{tr.metrics.get('hit_rate_at_10', 0.0):.4f} [{tr.metrics.get('hit_rate_at_10_lo95', 0.0):.4f},{tr.metrics.get('hit_rate_at_10_hi95', 0.0):.4f}]"
                t_hr20 = f"{tr.metrics.get('hit_rate_at_20', 0.0):.4f} [{tr.metrics.get('hit_rate_at_20_lo95', 0.0):.4f},{tr.metrics.get('hit_rate_at_20_hi95', 0.0):.4f}]"
                t_nd10 = f"{tr.metrics.get('ndcg_at_10', 0.0):.4f} [{tr.metrics.get('ndcg_at_10_lo95', 0.0):.4f},{tr.metrics.get('ndcg_at_10_hi95', 0.0):.4f}]"
                t_nd20 = f"{tr.metrics.get('ndcg_at_20', 0.0):.4f} [{tr.metrics.get('ndcg_at_20_lo95', 0.0):.4f},{tr.metrics.get('ndcg_at_20_hi95', 0.0):.4f}]"
                t_mrr10 = f"{tr.metrics.get('mrr_at_10', 0.0):.4f}"
                t_cov = f"{tr.metrics.get('catalog_coverage_at_10', 0.0):.4f}"
                t_div = f"{tr.metrics.get('diversity_at_10', 0.0):.4f}"

                print(f"{m_name:<25} | {'test':<5} | {t_hr10:<23} | {t_hr20:<23} | {t_nd10:<23} | {t_nd20:<23} | {t_mrr10:<7} | {t_cov:<8} | {t_div:<9}")

        print("=" * 140 + "\n")

    def _print_significance_table(self, diffs: list[dict[str, Any]]) -> None:
        print("\n" + "=" * 120)
        print("📊 Paired Bootstrap Significance: Ranker vs Baselines (Mean Diff & 95% CI)")
        print("=" * 120)
        print(f"{'Comparison':<30} | {'Split':<5} | {'NDCG@10 Diff (95% CI)':<32} | {'HR@10 Diff (95% CI)':<32} | {'Significant?'}")
        print("-" * 120)
        for d in diffs:
            comp_name = f"Ranker vs {d['target']}"
            nd_str = f"{d['ndcg_diff']:+.4f} [{d['ndcg_lo']:+.4f}, {d['ndcg_hi']:+.4f}]"
            hr_str = f"{d['hr_diff']:+.4f} [{d['hr_lo']:+.4f}, {d['hr_hi']:+.4f}]"
            sig_str = "YES (p<0.05)" if d["sig"] else "TIE"
            print(f"{comp_name:<30} | {d['split']:<5} | {nd_str:<32} | {hr_str:<32} | {sig_str}")
        print("=" * 120 + "\n")

    def _print_sliced_breakdown_table(
        self,
        val_res: dict[str, EvaluationResult],
        test_res: dict[str, EvaluationResult],
        seed: int = 42,
    ) -> None:
        print("\n" + "=" * 140)
        print("🔍 Sliced Evaluation Breakdown with Sample Sizes (N) & 95% Bootstrap CIs (NDCG@10)")
        print("=" * 140)
        print(f"{'Model':<25} | {'Split':<5} | {'Head Pop (20%)':<22} | {'Torso Pop (30%)':<22} | {'Tail Pop (50%)':<22} | {'Hist 3-4':<22} | {'Hist 5-9':<22} | {'Hist 10+':<22}")
        print("-" * 140)

        order = ["recsys_two_tower_ranker", "recsys_two_tower", "recsys_itemknn", "recsys_als", "recsys_popularity", "recsys_content"]
        for m_name in order:
            if m_name not in val_res:
                continue
            vr = val_res[m_name]
            h = vr.slices.get("pop_head", {})
            t = vr.slices.get("pop_torso", {})
            tail_slice = vr.slices.get("pop_tail", {})
            h34 = vr.slices.get("hist_3_4", {})
            h59 = vr.slices.get("hist_5_9", {})
            h10 = vr.slices.get("hist_10_plus", {})

            h_str = f"{h.get('ndcg_at_10',0):.4f} [n={int(h.get('count',0))}]"
            t_str = f"{t.get('ndcg_at_10',0):.4f} [n={int(t.get('count',0))}]"
            l_str = f"{tail_slice.get('ndcg_at_10',0):.4f} [n={int(tail_slice.get('count',0))}]"
            h34_str = f"{h34.get('ndcg_at_10',0):.4f} [n={int(h34.get('count',0))}]"
            h59_str = f"{h59.get('ndcg_at_10',0):.4f} [n={int(h59.get('count',0))}]"
            h10_str = f"{h10.get('ndcg_at_10',0):.4f} [n={int(h10.get('count',0))}]"

            print(f"{m_name:<25} | {'val':<5} | {h_str:<22} | {t_str:<22} | {l_str:<22} | {h34_str:<22} | {h59_str:<22} | {h10_str:<22}")

            if m_name in test_res:
                tr = test_res[m_name]
                th = tr.slices.get("pop_head", {})
                tt = tr.slices.get("pop_torso", {})
                tl = tr.slices.get("pop_tail", {})
                th34 = tr.slices.get("hist_3_4", {})
                th59 = tr.slices.get("hist_5_9", {})
                th10 = tr.slices.get("hist_10_plus", {})

                th_str = f"{th.get('ndcg_at_10',0):.4f} [n={int(th.get('count',0))}]"
                tt_str = f"{tt.get('ndcg_at_10',0):.4f} [n={int(tt.get('count',0))}]"
                tl_str = f"{tl.get('ndcg_at_10',0):.4f} [n={int(tl.get('count',0))}]"
                th34_str = f"{th34.get('ndcg_at_10',0):.4f} [n={int(th34.get('count',0))}]"
                th59_str = f"{th59.get('ndcg_at_10',0):.4f} [n={int(th59.get('count',0))}]"
                th10_str = f"{th10.get('ndcg_at_10',0):.4f} [n={int(th10.get('count',0))}]"

                print(f"{m_name:<25} | {'test':<5} | {th_str:<22} | {tt_str:<22} | {tl_str:<22} | {th34_str:<22} | {th59_str:<22} | {th10_str:<22}")

        print("=" * 140)

        # Slice verdicts with paired bootstrap CIs (Ranker vs Two-Tower) (Step 8)
        r_vr = val_res.get("recsys_two_tower_ranker")
        tt_vr = val_res.get("recsys_two_tower")
        r_tr = test_res.get("recsys_two_tower_ranker")
        tt_tr = test_res.get("recsys_two_tower")

        for split_label, r_res, tt_res in [("VAL", r_vr, tt_vr), ("TEST", r_tr, tt_tr)]:
            if r_res is None or tt_res is None:
                continue
            print(f"\n📌 Detailed Sliced Breakdown & Verdicts: Ranker vs Two-Tower ({split_label} Split):")
            print(f"{'Slice':<16} | {'N':<6} | {'Ranker NDCG@10 (95% CI)':<25} | {'Two-Tower NDCG@10 (95% CI)':<25} | {'Paired Diff (95% CI)':<28} | {'Verdict'}")
            print("-" * 130)

            for slice_key in ["pop_head", "pop_torso", "pop_tail", "hist_3_4", "hist_5_9", "hist_10_plus"]:
                r_sl = r_res.slices.get(slice_key, {})
                tt_sl = tt_res.slices.get(slice_key, {})
                n_slice = int(r_sl.get("count", 0))
                if n_slice == 0:
                    continue

                r_m = r_sl.get("ndcg_at_10", 0.0)
                r_lo = r_sl.get("ndcg_at_10_lo95", 0.0)
                r_hi = r_sl.get("ndcg_at_10_hi95", 0.0)

                tt_m = tt_sl.get("ndcg_at_10", 0.0)
                tt_lo = tt_sl.get("ndcg_at_10_lo95", 0.0)
                tt_hi = tt_sl.get("ndcg_at_10_hi95", 0.0)

                # Approximate slice paired diff from metric differences with CIs
                diff_m = r_m - tt_m
                lo_ci = r_lo - tt_hi
                hi_ci = r_hi - tt_lo

                if lo_ci > 0:
                    verdict = "outperforms"
                elif hi_ci < 0:
                    verdict = "underperforms"
                else:
                    verdict = "no significant difference"

                r_str = f"{r_m:.4f} [{r_lo:.4f}, {r_hi:.4f}]"
                tt_str = f"{tt_m:.4f} [{tt_lo:.4f}, {tt_hi:.4f}]"
                diff_str = f"{diff_m:+.4f} [{lo_ci:+.4f}, {hi_ci:+.4f}]"

                print(f"{slice_key:<16} | {n_slice:<6} | {r_str:<25} | {tt_str:<25} | {diff_str:<28} | {verdict}")

        print("=" * 140 + "\n")

    def _print_feature_importances_table(self, shap_dict: dict[str, float], gain_dict: dict[str, float]) -> None:
        print("\n" + "=" * 80)
        print("💡 Top 10 Features by Global SHAP & LightGBM Gain")
        print("=" * 80)
        top_shap = list(shap_dict.items())[:10]
        top_gain = list(gain_dict.items())[:10]

        print(f"{'Rank':<4} | {'Top Features by Mean |SHAP|':<35} | {'Top Features by Split Gain':<35}")
        print("-" * 80)
        for i in range(10):
            s_name, s_val = top_shap[i] if i < len(top_shap) else ("", 0.0)
            g_name, g_val = top_gain[i] if i < len(top_gain) else ("", 0.0)
            s_str = f"{s_name} ({s_val:.4f})" if s_name else "-"
            g_str = f"{g_name} ({g_val:.1f})" if g_name else "-"
            print(f"{i+1:<4} | {s_str:<35} | {g_str:<35}")
        print("=" * 80 + "\n")

    def execute(self) -> dict[str, Any]:
        """Entry point for executing BaseJob either with async DB session or dry-run."""
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
        description="Train and evaluate Stage-Two Learning-to-Rank Recommender"
    )
    parser.add_argument("--epochs", "--n-estimators", type=int, default=500, dest="n_estimators", help="Max LightGBM boosting rounds")
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.05, dest="learning_rate", help="LightGBM learning rate")
    parser.add_argument("--num-leaves", type=int, default=31, help="Max tree leaves")
    parser.add_argument("--min-child-samples", type=int, default=50, help="Min samples per leaf")
    parser.add_argument("--n-ranker-users", type=int, default=15000, help="Max users to sample for ranker training")
    parser.add_argument("--evaluate-test", "--test", action="store_true", help="Evaluate final model on test split")
    parser.add_argument("--clean", action="store_true", help="Use clean Two-Tower Stage-1 and register as recsys_two_tower_ranker_clean_v2")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow or DB writes")
    args = parser.parse_args()

    job = TrainRankerJob(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        n_ranker_users=args.n_ranker_users,
        evaluate_test=args.evaluate_test,
        clean=args.clean,
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
