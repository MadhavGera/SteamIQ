"""
Job: Generate and Persist Recommender System Evidence Reports — Phase 6.

Builds and saves all 10 report payloads to `backend/data/recsys/reports/*.json`,
writes `leaderboard.md`, and upserts all payloads into `serving_rec_reports`.

Usage:
  python -m jobs.generate_reports
  make recsys-reports
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun, ServingRecReport
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.report import (
    generate_leaderboard_markdown,
)


class GenerateReportsJob(BaseJob):
    job_name = "generate_reports"

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
        self.seed = seed

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        reports_dir = resolved_data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating and synchronizing recommender reports from %s...", reports_dir)
        data = RecsysData.load(resolved_data_dir)

        # Check if reports exist; if not, execute RunRecsysExperimentsJob
        required_keys = [
            "leaderboard",
            "ablations",
            "ranker_ablations",
            "slices",
            "cold_start",
            "beyond_accuracy",
            "shap_global",
            "training_curves",
            "dataset",
            "limitations",
        ]

        missing_keys = [k for k in required_keys if not (reports_dir / f"{k}.json").exists()]
        if missing_keys:
            self.logger.info("Missing report files %s. Running experiment suite to generate them...", missing_keys)
            from jobs.run_recsys_experiments import RunRecsysExperimentsJob
            exp_job = RunRecsysExperimentsJob(
                quick=False,
                profile=self.profile,
                data_dir=self.data_dir,
                seed=self.seed,
                dry_run=self.dry_run,
            )
            return await exp_job._run_async(session=session)

        # Load existing JSON payloads
        all_payloads: dict[str, dict[str, Any]] = {}
        for k in required_keys:
            with open(reports_dir / f"{k}.json", encoding="utf-8") as f:
                all_payloads[k] = json.load(f)

        # Find latest model runs for model_run_id association
        model_runs_dict: dict[str, ModelRun] = {}
        if session is not None:
            stmt = select(ModelRun).where(ModelRun.model_name.like("recsys_%")).order_by(ModelRun.created_at.desc())
            res = await session.execute(stmt)
            for r in res.scalars().all():
                if r.model_name not in model_runs_dict:
                    model_runs_dict[r.model_name] = r

        # Upsert into PostgreSQL serving_rec_reports
        if session is not None and not self.dry_run:
            for rep_key, rep_payload in all_payloads.items():
                stmt = (
                    insert(ServingRecReport)
                    .values(
                        report_key=rep_key,
                        payload=rep_payload,
                        dataset_version=data.dataset_version,
                        model_run_id=model_runs_dict.get("recsys_two_tower").id if model_runs_dict.get("recsys_two_tower") else None,
                    )
                    .on_conflict_do_update(
                        index_elements=[ServingRecReport.report_key],
                        set_={
                            "payload": rep_payload,
                            "dataset_version": data.dataset_version,
                        },
                    )
                )
                await session.execute(stmt)

            await session.commit()
            self.logger.info("Successfully upserted all %d report payloads to serving_rec_reports table.", len(all_payloads))

        # Re-render leaderboard.md
        md_content = generate_leaderboard_markdown(
            leaderboard_payload=all_payloads["leaderboard"],
            ablations_payload=all_payloads.get("ablations"),
            ranker_ablations_payload=all_payloads.get("ranker_ablations"),
            slices_payload=all_payloads.get("slices"),
            cold_start_payload=all_payloads.get("cold_start"),
            beyond_accuracy_payload=all_payloads.get("beyond_accuracy"),
        )
        with open(reports_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(resolved_data_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        print("\n" + md_content)
        return {"reports_count": len(all_payloads), "dataset_version": data.dataset_version}

    def execute(self) -> dict[str, Any]:
        """Entry point for executing BaseJob."""
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
    parser = argparse.ArgumentParser(description="Generate and persist recommender evidence reports")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without DB writes")
    args = parser.parse_args()

    job = GenerateReportsJob(
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
