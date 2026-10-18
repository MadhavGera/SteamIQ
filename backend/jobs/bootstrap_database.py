"""
Database Bootstrap Job — ensures serving and mart tables are populated on fresh startup.

Idempotent: if tables already contain rows, this job completes in milliseconds.
Designed to guarantee a seamless first-time startup for new developers and Docker runs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import (
    MartGameOverview,
    RawGame,
    ServingRecItem,
    ServingRecItemProjection,
    ServingRecReport,
)

logger = logging.getLogger("steamiq.jobs.bootstrap")


async def bootstrap_database(session: AsyncSession) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    data_dir = Path(settings.resolved_recsys_data_dir)

    # 1. Check serving_rec_reports
    rep_count = (await session.execute(select(func.count()).select_from(ServingRecReport))).scalar_one()
    if rep_count == 0:
        logger.info("serving_rec_reports is empty. Syncing precomputed evidence reports...")
        try:
            from jobs.generate_reports import GenerateReportsJob
            res = await GenerateReportsJob()._run_async(session=session)
            stats["reports_synced"] = res.get("reports_count", 0)
        except Exception as e:
            logger.warning("Could not sync reports during bootstrap: %s", e)
    else:
        stats["reports_synced"] = rep_count

    # 2. Check serving_rec_items
    item_count = (await session.execute(select(func.count()).select_from(ServingRecItem))).scalar_one()
    items_parquet = data_dir / "items.parquet"
    if item_count == 0 and items_parquet.exists():
        logger.info("serving_rec_items is empty. Persisting universe from %s...", items_parquet)
        try:
            df = pd.read_parquet(items_parquet)
            records = []
            for r in df.to_dict(orient="records"):
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

            for i in range(0, len(records), 500):
                chunk = records[i : i + 500]
                stmt = pg_insert(ServingRecItem).values(chunk).on_conflict_do_nothing(index_elements=[ServingRecItem.app_id])
                await session.execute(stmt)
            await session.commit()
            stats["items_persisted"] = len(records)
            logger.info("Persisted %d items to serving_rec_items.", len(records))
        except Exception as e:
            logger.warning("Could not persist items during bootstrap: %s", e)
    else:
        stats["items_persisted"] = item_count

    # 3. Check serving_rec_item_projection
    proj_count = (await session.execute(select(func.count()).select_from(ServingRecItemProjection))).scalar_one()
    if proj_count == 0:
        logger.info("serving_rec_item_projection is empty. Running item projection job...")
        try:
            from jobs.project_item_embeddings import ProjectItemEmbeddingsJob
            res = await ProjectItemEmbeddingsJob()._run_async(session=session)
            stats["projections_persisted"] = res.get("n_items", 0)
        except Exception as e:
            logger.warning("Could not project item embeddings during bootstrap: %s", e)
    else:
        stats["projections_persisted"] = proj_count

    # 4. Check decision marts (mart_game_overview)
    mart_count = (await session.execute(select(func.count()).select_from(MartGameOverview))).scalar_one()
    if mart_count == 0:
        raw_count = (await session.execute(select(func.count()).select_from(RawGame))).scalar_one()
        if raw_count > 0:
            logger.info("mart_game_overview is empty but raw games exist. Materializing marts...")
            try:
                from jobs.materialize_marts import MaterializeMartsJob
                res = await MaterializeMartsJob(all_games=True)._run_async(session=session)
                stats["marts_materialized"] = res.get("games_processed", 0)
            except Exception as e:
                logger.warning("Could not materialize marts during bootstrap: %s", e)
    else:
        stats["marts_materialized"] = mart_count

    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger.info("Running database bootstrap check...")

    async def _run() -> None:
        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            res = await bootstrap_database(session)
            logger.info("Bootstrap complete: %s", res)
        await engine.dispose()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
