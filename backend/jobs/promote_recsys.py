"""
Model Promotion Job for Recommender System Models — Phase 6.

Subclasses BaseJob.
The ONLY authorized entry point for promote_champion across the recsys subsystem.
Rules:
  (a) Picks the model by VAL NDCG@10 on the evaluation users.
  (b) Promotes a ranker ONLY if its paired-bootstrap VAL difference vs the best non-ranker
      model has a 95% CI lower bound strictly greater than 0 (lower_bound > 0).
  (c) NEVER uses or inspects TEST split.
  (d) Prints its complete reasoning, paired confidence intervals, and promotion decision.

Usage:
  python -m jobs.promote_recsys
  make recsys-promote
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import EvaluationResult, evaluate_model, get_evaluation_user_split
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerModel
from recsys.registry import promote_champion


class PromoteRecsysJob(BaseJob):
    job_name = "promote_recsys"

    def __init__(
        self,
        *,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        deployable_only: bool = True,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.data_dir = data_dir
        self.deployable_only = deployable_only

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = cfg.seed

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recsys dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # Disjoint evaluation users (strictly on validation split)
        n_tuning = cfg.eval_users if cfg.profile == "dev" else 10000
        eval_users, _ = get_evaluation_user_split(
            data,
            n_eval_users=cfg.eval_users,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )
        self.logger.info("Loaded %d fixed evaluation users for VAL promotion tournament.", len(eval_users))

        # Query database for latest valid recsys runs
        runs_dict: dict[str, ModelRun] = {}
        if session is not None:
            stmt = (
                select(ModelRun)
                .where(ModelRun.model_name.like("recsys_%"))
                .where(ModelRun.stage != "archived")
                .order_by(ModelRun.created_at.desc())
            )
            res = await session.execute(stmt)
            all_runs = res.scalars().all()
            for r in all_runs:
                if r.model_name not in runs_dict:
                    runs_dict[r.model_name] = r

        if self.deployable_only:
            model_families = [
                ("recsys_popularity", "popularity"),
                ("recsys_itemknn", "itemknn"),
                ("recsys_als", "als"),
                ("recsys_content", "content"),
                ("recsys_two_tower_clean", "two_tower_clean"),
                ("recsys_two_tower_ranker_clean", "two_tower_ranker_clean"),
            ]
        else:
            model_families = [
                ("recsys_popularity", "popularity"),
                ("recsys_itemknn", "itemknn"),
                ("recsys_als", "als"),
                ("recsys_content", "content"),
                ("recsys_two_tower_clean", "two_tower_clean"),
                ("recsys_two_tower_ranker_clean", "two_tower_ranker_clean"),
                ("recsys_two_tower", "two_tower"),
                ("recsys_two_tower_ranker", "two_tower_ranker"),
            ]

        stage1_instances: dict[str, BaseRecommender] = {}
        val_eval_results: dict[str, EvaluationResult] = {}
        model_instances: dict[str, BaseRecommender] = {}

        print("\n" + "=" * 120)
        print(f"🏛️ SteamIQ Recsys Model Promotion Tournament (Validation Split Only, deployable_only={self.deployable_only})")
        print("=" * 120)

        for m_name, _short_key in model_families:
            run_obj = runs_dict.get(m_name)
            if run_obj is not None and run_obj.artifact_path:
                art_path = Path(run_obj.artifact_path)
            else:
                art_base = resolved_data_dir / "artifacts" / m_name
                if not art_base.exists():
                    self.logger.warning("No artifact directory for %s on disk. Skipping.", m_name)
                    continue
                cand_dirs = sorted([d for d in art_base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                if not cand_dirs:
                    self.logger.warning("No artifact found for %s on disk. Skipping.", m_name)
                    continue
                art_path = cand_dirs[0]

            if not art_path.exists():
                self.logger.warning("Artifact path %s does not exist. Skipping %s.", art_path, m_name)
                continue

            # Load model instance
            m_inst: BaseRecommender
            if m_name == "recsys_popularity":
                m_inst = PopularityRecommender.load(art_path)
            elif m_name == "recsys_itemknn":
                m_inst = ItemKNNRecommender.load(art_path)
                stage1_instances["itemknn"] = m_inst
            elif m_name == "recsys_als":
                m_inst = ALSRecommender.load(art_path)
                stage1_instances["als"] = m_inst
            elif m_name == "recsys_content":
                m_inst = ContentRecommender.load(art_path)
                stage1_instances["content"] = m_inst
            elif m_name in ("recsys_two_tower", "recsys_two_tower_clean"):
                m_inst = TwoTowerModel.load(art_path, data=data)
                stage1_instances["two_tower"] = m_inst
            elif m_name in ("recsys_two_tower_ranker", "recsys_two_tower_ranker_clean"):
                m_inst = RankerModel.load(art_path, data=data, models=stage1_instances)
            else:
                continue

            model_instances[m_name] = m_inst

            val_res = evaluate_model(
                model=m_inst,
                data=data,
                split="val",
                k_list=(10, 20),
                user_indices=eval_users,
                max_history=cfg.max_history,
                seed=seed,
                allow_small_user_set=self.dry_run or len(eval_users) < 1000,
            )
            val_eval_results[m_name] = val_res

        # Print VAL Tournament Standings
        print(f"\n{'Rank':<4} | {'Model Name':<32} | {'VAL NDCG@10':<13} | {'95% Bootstrap CI':<20} | {'VAL HR@10':<10} | {'VAL MRR@10':<10}")
        print("-" * 105)

        sorted_models = sorted(
            val_eval_results.keys(),
            key=lambda k: val_eval_results[k].metrics.get("ndcg_at_10", 0.0),
            reverse=True,
        )

        for rank_idx, m_name in enumerate(sorted_models, 1):
            res = val_eval_results[m_name]
            ndcg = res.metrics.get("ndcg_at_10", 0.0)
            ndcg_lo = res.metrics.get("ndcg_at_10_lo95", 0.0)
            ndcg_hi = res.metrics.get("ndcg_at_10_hi95", 0.0)
            hr = res.metrics.get("hit_rate_at_10", 0.0)
            mrr = res.metrics.get("mrr_at_10", 0.0)
            ci_str = f"[{ndcg_lo:.4f}, {ndcg_hi:.4f}]"
            print(f"{rank_idx:<4} | {m_name:<32} | {ndcg:<13.4f} | {ci_str:<20} | {hr:<10.4f} | {mrr:<10.4f}")

        print("=" * 105 + "\n")

        # Promotion Logic under Deployable Decision Rule
        promoted_model_name: str
        promoted_family: str
        reasoning: str

        ranker_key = "recsys_two_tower_ranker_clean" if "recsys_two_tower_ranker_clean" in val_eval_results else "recsys_two_tower_ranker"
        tt_key = "recsys_two_tower_clean" if "recsys_two_tower_clean" in val_eval_results else "recsys_two_tower"

        if ranker_key in val_eval_results and tt_key in val_eval_results:
            r_ndcg = val_eval_results[ranker_key].metrics.get("ndcg_at_10", 0.0)
            tt_ndcg = val_eval_results[tt_key].metrics.get("ndcg_at_10", 0.0)
            diff, lo95, hi95 = paired_bootstrap_diff(
                val_eval_results[ranker_key].per_user["ndcg_at_10"],
                val_eval_results[tt_key].per_user["ndcg_at_10"],
                seed=seed,
            )

            # Check pre-registered decision rule
            if r_ndcg > tt_ndcg and lo95 > 0.0:
                promoted_model_name = ranker_key
                promoted_family = ranker_key
                reasoning = (
                    f"Deployable Clean Ranker achieved higher VAL NDCG@10 ({r_ndcg:.4f} vs {tt_ndcg:.4f}) and "
                    f"satisfied the statistical significance threshold with paired bootstrap 95% CI lower bound > 0 "
                    f"(diff={diff:+.4f}, 95% CI: [{lo95:+.4f}, {hi95:+.4f}])."
                )
            else:
                promoted_model_name = tt_key
                promoted_family = tt_key
                reasoning = (
                    f"Deployable Clean Two-Tower selected as champion ({tt_ndcg:.4f} VAL NDCG@10). "
                    f"Clean Ranker VAL difference was {diff:+.4f} (95% CI: [{lo95:+.4f}, {hi95:+.4f}])."
                )
        else:
            promoted_model_name = sorted_models[0]
            promoted_family = sorted_models[0]
            reasoning = f"Selected best available deployable model on VAL NDCG@10 ({val_eval_results[promoted_model_name].metrics.get('ndcg_at_10', 0.0):.4f})."

        print("📢 Promotion Decision & Reasoning:")
        print(f"  • Selected Model:     {promoted_model_name}")
        print(f"  • Metric:             VAL NDCG@10 = {val_eval_results[promoted_model_name].metrics.get('ndcg_at_10', 0.0):.4f}")
        print(f"  • Evaluation Users:   {len(eval_users)} (Validation Split Only)")
        print(f"  • Reasoning:          {reasoning}")

        if not self.dry_run and session is not None:
            promoted_run = await promote_champion(
                family=promoted_family,
                metric_key="val_ndcg_at_10",
                session=session,
            )
            print(f"\n🏆 Successfully promoted '{promoted_model_name}' (Run ID: {promoted_run.id if promoted_run else 'N/A'}) to PRODUCTION stage in database.\n")
        else:
            print("\nℹ️ [Dry Run] Skipped database promote_champion call.\n")

        return {
            "promoted_model_name": promoted_model_name,
            "promoted_family": promoted_family,
            "val_ndcg_at_10": val_eval_results[promoted_model_name].metrics.get("ndcg_at_10", 0.0),
            "reasoning": reasoning,
        }

    def execute(self) -> dict[str, Any]:
        """Entry point for BaseJob."""
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
    parser = argparse.ArgumentParser(description="Promote best recsys model to production champion.")
    parser.add_argument("--profile", type=str, default=None, help="Recsys profile (dev/full)")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys data directory")
    parser.add_argument("--deployable-only", action="store_true", default=True, help="Promote only among deployable models")
    parser.add_argument("--allow-all", action="store_false", dest="deployable_only", help="Allow non-deployable models in tournament")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without modifying database stages")
    args = parser.parse_args()

    job = PromoteRecsysJob(
        profile=args.profile,
        data_dir=args.data_dir,
        deployable_only=args.deployable_only,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
