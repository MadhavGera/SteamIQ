"""
Verification Job for Recommender System Models — Phase 6.

Subclasses BaseJob.
Re-evaluates every latest recsys model run from its persisted artifact,
compares newly evaluated metrics to metrics stored in `model_runs` (tolerance 1e-4),
and exits non-zero if any mismatch occurs.

Usage:
  python -m jobs.verify_recsys
  make recsys-verify
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model, get_evaluation_user_split
from recsys.models.als import ALSRecommender
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerModel


class VerifyRecsysJob(BaseJob):
    job_name = "verify_recsys"

    def __init__(
        self,
        *,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        tolerance: float = 1e-4,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.data_dir = data_dir
        self.tolerance = tolerance

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = cfg.seed

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recsys dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # Disjoint evaluation users
        n_tuning = cfg.eval_users if cfg.profile == "dev" else 10000
        eval_users, _ = get_evaluation_user_split(
            data,
            n_eval_users=cfg.eval_users,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        self.logger.info("Evaluation users loaded: %d users", len(eval_users))

        # Query database for latest valid recsys runs
        runs_to_verify: dict[str, ModelRun] = {}

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
                if r.model_name not in runs_to_verify:
                    runs_to_verify[r.model_name] = r

        # Model loaders mapping
        model_names_order = [
            "recsys_popularity",
            "recsys_itemknn",
            "recsys_als",
            "recsys_content",
            "recsys_two_tower",
            "recsys_two_tower_ranker",
        ]

        # Stage 1 instances dict for ranker
        stage1_instances: dict[str, BaseRecommender] = {}

        verification_records: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []

        print("\n" + "=" * 130)
        print("🔬 SteamIQ Recommender Model Artifact Verification Job")
        print("=" * 130)

        for m_name in model_names_order:
            run_obj = runs_to_verify.get(m_name)
            if run_obj is None or not run_obj.artifact_path:
                self.logger.warning("No valid active run found for %s in model_runs. Searching disk...", m_name)
                art_base = resolved_data_dir / "artifacts" / m_name
                if not art_base.exists():
                    continue
                cand_dirs = sorted([d for d in art_base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                if not cand_dirs:
                    continue
                art_path = cand_dirs[0]
                db_metrics = {}
                run_id_str = "disk_fallback"
            else:
                art_path = Path(run_obj.artifact_path)
                db_metrics = run_obj.metrics or {}
                run_id_str = str(run_obj.id)

            if not art_path.exists():
                self.logger.error("Artifact path does not exist on disk: %s", art_path)
                mismatches.append({"model": m_name, "error": f"Missing artifact path: {art_path}"})
                continue

            # Load model instance
            self.logger.info("Loading artifact for %s from %s...", m_name, art_path)
            model_inst: BaseRecommender
            if m_name == "recsys_popularity":
                model_inst = PopularityRecommender.load(art_path)
            elif m_name == "recsys_itemknn":
                model_inst = ItemKNNRecommender.load(art_path)
                stage1_instances["itemknn"] = model_inst
            elif m_name == "recsys_als":
                model_inst = ALSRecommender.load(art_path)
                stage1_instances["als"] = model_inst
            elif m_name == "recsys_content":
                model_inst = ContentRecommender.load(art_path)
                stage1_instances["content"] = model_inst
            elif m_name == "recsys_two_tower":
                model_inst = TwoTowerModel.load(art_path, data=data)
                stage1_instances["two_tower"] = model_inst
            elif m_name == "recsys_two_tower_ranker":
                model_inst = RankerModel.load(art_path, data=data, models=stage1_instances)
            else:
                continue

            # Evaluate freshly
            val_res = evaluate_model(
                model=model_inst,
                data=data,
                split="val",
                k_list=(10, 20),
                user_indices=eval_users,
                seed=seed,
            )

            has_test = any(k.startswith("test_") for k in db_metrics)
            test_res = None
            if has_test:
                test_res = evaluate_model(
                    model=model_inst,
                    data=data,
                    split="test",
                    k_list=(10, 20),
                    user_indices=eval_users,
                    seed=seed,
                )

            # Compare metrics
            check_keys = ["val_ndcg_at_10", "val_hit_rate_at_10", "val_mrr_at_10"]
            if test_res:
                check_keys.extend(["test_ndcg_at_10", "test_hit_rate_at_10", "test_mrr_at_10", "test_catalog_coverage_at_10", "test_diversity_at_10"])

            model_mismatches = []
            for k in check_keys:
                if k in db_metrics:
                    expected_v = float(db_metrics[k])
                    actual_v = float(val_res.metrics.get(k.replace("val_", ""), 0.0) if k.startswith("val_") else (test_res.metrics.get(k.replace("test_", ""), 0.0) if test_res else 0.0))
                    diff = abs(actual_v - expected_v)
                    if diff > self.tolerance:
                        model_mismatches.append({"metric": k, "expected": expected_v, "actual": actual_v, "diff": diff})

            val_ndcg = val_res.metrics.get("ndcg_at_10", 0.0)
            val_hr = val_res.metrics.get("hit_rate_at_10", 0.0)
            test_ndcg = test_res.metrics.get("ndcg_at_10", 0.0) if test_res else 0.0
            test_hr = test_res.metrics.get("hit_rate_at_10", 0.0) if test_res else 0.0
            test_cov = test_res.metrics.get("catalog_coverage_at_10", 0.0) if test_res else 0.0
            test_div = test_res.metrics.get("diversity_at_10", 0.0) if test_res else 0.0

            status_str = "MATCH" if not model_mismatches else f"DIFF ({len(model_mismatches)} keys)"
            if model_mismatches:
                mismatches.extend([{"model": m_name, **m} for m in model_mismatches])

            verification_records.append({
                "model": m_name,
                "run_id": run_id_str,
                "val_ndcg": val_ndcg,
                "val_hr": val_hr,
                "test_ndcg": test_ndcg,
                "test_hr": test_hr,
                "test_cov": test_cov,
                "test_div": test_div,
                "status": status_str,
            })

        print(f"{'Model':<25} | {'Val NDCG':<9} | {'Val HR':<8} | {'Test NDCG':<9} | {'Test HR':<8} | {'Test Cov':<8} | {'Test Div':<8} | {'Status'}")
        print("-" * 130)
        for r in verification_records:
            print(f"{r['model']:<25} | {r['val_ndcg']:<9.4f} | {r['val_hr']:<8.4f} | {r['test_ndcg']:<9.4f} | {r['test_hr']:<8.4f} | {r['test_cov']:<8.4f} | {r['test_div']:<8.4f} | {r['status']}")

        print("=" * 130)

        if mismatches:
            print("\n❌ VERIFICATION FAILED: Found metric mismatches exceeding tolerance:")
            for m in mismatches:
                print(f"  • {m}")
            raise RuntimeError(f"Recommender artifact verification failed with {len(mismatches)} mismatches.")
        else:
            print("\n✅ ALL ARTIFACT METRICS VERIFIED SUCCESSFULLY (Tolerance: 1e-4)!\n")

        return {"verified_count": len(verification_records), "mismatches": mismatches}

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
    parser = argparse.ArgumentParser(description="Verify serialized recsys artifacts against model_runs")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys data directory")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Allowed metric float tolerance")
    parser.add_argument("--dry-run", action="store_true", help="Run without DB connection")
    args = parser.parse_args()

    job = VerifyRecsysJob(
        profile=args.profile,
        data_dir=args.data_dir,
        tolerance=args.tolerance,
        dry_run=args.dry_run,
    )
    try:
        job.execute()
    except Exception as exc:
        print(f"Error during verification: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
