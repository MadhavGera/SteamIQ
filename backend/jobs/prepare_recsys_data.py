"""
Memory-Safe Data Preparation & Serving Ingestion Job for SteamIQ Recommender.

Executes end-to-end data preparation:
  1. Stage 0: recommendations.csv -> interactions_all.parquet (cached).
  2. Interaction filtering, user/item sampling, deduplication, and leakage-free temporal splitting.
  3. Item feature extraction, sparse tag vocabulary, continuous standardization, and MiniLM embeddings.
  4. Data manifest generation with verifiable SHA1 hashes and 12-char dataset_version.
  5. Bulk loading of universe catalog into Postgres serving_rec_items (inside one transaction).

Rule 2: Subclasses BaseJob, implements run(), supports dry_run, self.resolve_path(), self.logger.
Rule 3: Ingestion/Job writes to serving_rec_items.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ServingRecItem
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.items import build_item_artifacts
from recsys.prepare import prepare_interactions


class PrepareRecsysDataJob(BaseJob):
    """
    BaseJob orchestrating recommender interaction preparation, item embeddings, and serving persistence.
    """

    job_name = "prepare_recsys_data"

    def __init__(
        self,
        *,
        profile: str | None = None,
        config: RecsysConfig | None = None,
        raw_dir: Path | None = None,
        data_dir: Path | None = None,
        seed: int | None = None,
        min_user_positives: int | None = None,
        min_item_positives: int | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.config = config
        self.raw_dir = raw_dir
        self.data_dir = data_dir
        self.seed = seed
        self.min_user_positives = min_user_positives
        self.min_item_positives = min_item_positives

    def _get_config(self) -> RecsysConfig:
        if self.config is not None:
            return self.config
        base_cfg = get_recsys_config(self.profile)
        if (
            self.raw_dir
            or self.data_dir
            or self.seed is not None
            or self.min_user_positives is not None
            or self.min_item_positives is not None
        ):
            # Override with explicit paths/seeds/thresholds if provided
            return RecsysConfig(
                profile=base_cfg.profile,
                max_users=base_cfg.max_users,
                max_items=base_cfg.max_items,
                eval_users=base_cfg.eval_users,
                min_user_positives=self.min_user_positives if self.min_user_positives is not None else base_cfg.min_user_positives,
                min_item_positives=self.min_item_positives if self.min_item_positives is not None else base_cfg.min_item_positives,
                max_history=base_cfg.max_history,
                seed=self.seed if self.seed is not None else base_cfg.seed,
                num_threads=base_cfg.num_threads,
                device=base_cfg.device,
                raw_dir=self.raw_dir or base_cfg.raw_dir,
                data_dir=self.data_dir or base_cfg.data_dir,
            )
        return base_cfg

    async def _persist_to_db(self, items_df: pd.DataFrame, session: AsyncSession | None = None) -> None:
        if session is not None:
            await self._persist_with_session(session, items_df)
            return

        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as s:
            await self._persist_with_session(s, items_df)
        await engine.dispose()

    async def _persist_with_session(self, session: AsyncSession, items_df: pd.DataFrame) -> None:
        await session.execute(delete(ServingRecItem))
        records: list[dict[str, Any]] = []
        for r in items_df.to_dict(orient="records"):
            raw_tags = r.get("tags")
            if isinstance(raw_tags, list | np.ndarray):
                tags_clean = [str(t) for t in raw_tags]
            elif isinstance(raw_tags, dict):
                tags_clean = list(raw_tags.keys())
            else:
                tags_clean = []

            price_final = r.get("price_final")
            pos_ratio = r.get("positive_ratio")
            user_revs = r.get("user_reviews")

            records.append({
                "app_id": int(r["app_id"]),
                "item_idx": int(r["item_idx"]),
                "title": str(r["title"]),
                "date_release": str(r["date_release"]) if pd.notna(r.get("date_release")) and r.get("date_release") is not None else None,
                "price_final": float(price_final) if pd.notna(price_final) and price_final is not None else None,
                "positive_ratio": int(pos_ratio) if pd.notna(pos_ratio) and pos_ratio is not None else None,
                "user_reviews": int(user_revs) if pd.notna(user_revs) and user_revs is not None else None,
                "tags": tags_clean,
                "description_short": str(r["description_short"]) if pd.notna(r.get("description_short")) and r.get("description_short") is not None else None,
                "has_description": bool(r.get("has_description", False)),
                "train_positives": int(r.get("train_positives", 0)),
                "popularity_rank": int(r.get("popularity_rank", 0)),
                "win": bool(r.get("win", False)),
                "mac": bool(r.get("mac", False)),
                "linux": bool(r.get("linux", False)),
                "steam_deck": bool(r.get("steam_deck", False)),
                "header_image_url": str(r["header_image_url"]),
            })

        if records:
            for i in range(0, len(records), 1000):
                chunk = records[i : i + 1000]
                stmt = pg_insert(ServingRecItem).values(chunk)
                await session.execute(stmt)
        await session.commit()

    def run(self) -> dict[str, Any]:
        config = self._get_config()
        raw_dir = config.raw_dir or settings.resolved_recsys_raw_dir
        data_dir = config.data_dir or settings.resolved_recsys_data_dir

        self.logger.info(
            "Starting recommender data preparation (Profile=%s, Seed=%d, DryRun=%s)...",
            config.profile,
            config.seed,
            self.dry_run,
        )

        # ── 1. Prepare Interactions & Splits ─────────────────────────────────
        self.logger.info("Running interactions pipeline (Stages 0 - 1)...")
        inter_stats = prepare_interactions(raw_dir=raw_dir, data_dir=data_dir, config=config)

        # ── 2. Build Item Features & Embeddings ──────────────────────────────
        self.logger.info("Building item features, embeddings & data manifest (Stage 2 - 3)...")
        item_stats = build_item_artifacts(
            raw_dir=raw_dir,
            data_dir=data_dir,
            config=config,
            interaction_stats=inter_stats,
        )

        # ── 3. Database Persistence (serving_rec_items) ──────────────────────
        items_parquet_path = data_dir / "items.parquet"
        items_df = pd.read_parquet(items_parquet_path)

        if not self.dry_run:
            self.logger.info("Persisting %d universe items into serving_rec_items...", len(items_df))
            try:
                asyncio.run(self._persist_to_db(items_df))
                self.logger.info("✅ Successfully persisted %d items into serving_rec_items.", len(items_df))
            except Exception as exc:
                self.logger.error("Database write to serving_rec_items failed: %s", exc)
                raise
        else:
            self.logger.info("Dry run enabled: Skipping database persistence.")

        # ── 4. Metrics & Summary Calculations ────────────────────────────────
        users_df = pd.read_parquet(data_dir / "users.parquet")
        num_users = len(users_df)
        num_items = len(items_df)
        train_rows = inter_stats["train_rows"]
        val_rows = inter_stats["val_rows"]
        test_rows = inter_stats["test_rows"]
        neg_rows = inter_stats["negatives_rows"]

        density = float(train_rows / (num_users * num_items)) if (num_users * num_items) > 0 else 0.0
        median_history = float(np.median(users_df["n_train"])) if not users_df.empty else 0.0
        desc_coverage = item_stats["item_description_coverage"]
        dataset_version = item_stats["dataset_version"]

        # ── 5. Summary Table ─────────────────────────────────────────────────
        summary_table = f"""
================================================================================
STEAM RECOMMENDATION SYSTEM DATA PREPARATION SUMMARY
================================================================================
Profile:                      {config.profile.upper()} (Seed={config.seed})
Dataset Version:              {dataset_version}
--------------------------------------------------------------------------------
COUNTS & DENSITY
Universe Items:               {num_items:,}
Sampled Eligible Users:       {num_users:,}
Train Interactions:           {train_rows:,}
Val Interactions:             {val_rows:,}
Test Interactions:            {test_rows:,}
Negatives (Masking):          {neg_rows:,}
Train Interaction Density:    {density * 100:.4f}%
Median Train History / User:  {median_history:.1f} games
Item Description Coverage:    {desc_coverage * 100:.2f}%
================================================================================
"""
        print(summary_table)

        return {
            "dataset_version": dataset_version,
            "profile": config.profile,
            "seed": config.seed,
            "users": num_users,
            "items": num_items,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "negatives_rows": neg_rows,
            "density": round(density, 6),
            "median_history_length": median_history,
            "item_description_coverage": round(desc_coverage, 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Steam interaction datasets, item features, and serving tables")
    parser.add_argument("--profile", type=str, default=None, help="Recsys profile ('dev' or 'full')")
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory containing raw Steam CSV/JSON files")
    parser.add_argument("--data-dir", type=str, default=None, help="Output directory for processed parquets/artifacts")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Skip database write")
    args = parser.parse_args()

    job = PrepareRecsysDataJob(
        profile=args.profile,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
