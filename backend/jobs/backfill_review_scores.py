"""
Backfill Review Scores Job for SteamIQ.

Queries the Steam appreviews API (query_summary object) for all games in raw_games
to backfill:
  - review_score (real Steam 1-9 score, replacing any mistakenly copied Metacritic scores)
  - review_score_desc (Steam's text sentiment descriptor, e.g. 'Overwhelmingly Positive')

Preserves metacritic_score as its own separate, distinct field.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from db.models import RawGame
from jobs.base_job import BaseJob

logger = logging.getLogger(__name__)

STEAM_REVIEWS_API = "https://store.steampowered.com/appreviews/{appid}"
REQUEST_DELAY_S = 0.5  # Rate limit delay for Steam reviews API


class BackfillReviewScoresJob(BaseJob):
    job_name = "backfill_review_scores"

    def __init__(self, *, concurrency: int = 4, dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self.concurrency = concurrency

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())

    async def _fetch_query_summary(
        self, client: httpx.AsyncClient, app_id: int, semaphore: asyncio.Semaphore
    ) -> tuple[int, dict[str, Any] | None]:
        async with semaphore:
            try:
                await asyncio.sleep(REQUEST_DELAY_S)
                resp = await client.get(
                    STEAM_REVIEWS_API.format(appid=app_id),
                    params={"json": 1, "language": "english", "num_per_page": 1},
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        return app_id, data.get("query_summary")
                return app_id, None
            except Exception as exc:
                self.logger.warning("Failed fetching query_summary for %d: %s", app_id, exc)
                return app_id, None

    async def _run_async(self) -> dict[str, Any]:
        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as session:
            res = await session.execute(select(RawGame.app_id, RawGame.name, RawGame.review_score, RawGame.metacritic_score))
            games = res.all()

        total = len(games)
        self.logger.info("Starting review_score / review_score_desc backfill for %d games...", total)

        semaphore = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(headers={"User-Agent": "SteamIQ-Backfill/1.0"}) as client:
            tasks = [self._fetch_query_summary(client, g.app_id, semaphore) for g in games]
            results = await asyncio.gather(*tasks)

        updated_count = 0
        corrupted_fixed_count = 0

        async with session_factory() as session, session.begin():
            for app_id, q_sum in results:
                if not q_sum:
                    continue

                review_score = q_sum.get("review_score")
                review_score_desc = q_sum.get("review_score_desc")

                # Check if old score was corrupted (e.g. > 9)
                old_row = next((g for g in games if g.app_id == app_id), None)
                if old_row and old_row.review_score is not None and old_row.review_score > 9:
                    corrupted_fixed_count += 1

                if not self.dry_run:
                    stmt = (
                        update(RawGame)
                        .where(RawGame.app_id == app_id)
                        .values(
                            review_score=review_score,
                            review_score_desc=review_score_desc,
                        )
                    )
                    await session.execute(stmt)
                updated_count += 1

        await engine.dispose()
        self.logger.info(
            "Backfill complete: %d/%d games updated, %d previously corrupted review_scores corrected.",
            updated_count,
            total,
            corrupted_fixed_count,
        )
        return {
            "total_games": total,
            "updated_count": updated_count,
            "corrupted_fixed_count": corrupted_fixed_count,
        }


def main() -> None:
    from pathlib import Path
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)

    job = BackfillReviewScoresJob()
    job.execute()


if __name__ == "__main__":
    main()
