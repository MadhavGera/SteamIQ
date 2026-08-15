"""
Steam News & Patch Notes Ingestion Job — Phase 5.

Fetches authentic developer announcements, patch notes, and update releases
from the official Steam News API (ISteamNews/GetNewsForApp/v2).

Data Source:
  - https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid={app_id}&count=25&format=json

Target Table:
  - raw_patch_notes (app_id, gid, title, url, author, contents, feedlabel, published_at)

Usage:
  python -m jobs.ingest_news --all
  python -m jobs.ingest_news --app-id 1145360
  python -m jobs.ingest_news --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import RawGame, RawPatchNote
from jobs.base_job import BaseJob

STEAM_NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"


class IngestNewsJob(BaseJob):
    """
    Ingests official Steam news and patch notes for catalog games.
    """
    job_name = "ingest_news"

    def __init__(
        self,
        *,
        app_id: int | None = None,
        count: int = 25,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.count = count

    async def fetch_news_for_app(
        self,
        client: httpx.AsyncClient,
        app_id: int,
    ) -> list[dict[str, Any]]:
        """
        Fetch news items from Steam Web API for a given app_id.
        """
        params = {
            "appid": app_id,
            "count": self.count,
            "maxlength": 1500,
            "format": "json",
        }
        try:
            resp = await client.get(STEAM_NEWS_URL, params=params, timeout=15.0)
            if resp.status_code != 200:
                self.logger.warning(
                    "Steam News API returned status %d for app_id=%d",
                    resp.status_code,
                    app_id,
                )
                return []

            data = resp.json()
            items = data.get("appnews", {}).get("newsitems", [])
            return items
        except Exception as exc:
            self.logger.warning("Failed to fetch news for app_id=%d: %s", app_id, exc)
            return []

    async def _upsert_patch_notes(
        self,
        session: AsyncSession,
        app_id: int,
        items: list[dict[str, Any]],
    ) -> int:
        """
        Upsert fetched newsitems into raw_patch_notes table.
        """
        if not items:
            return 0

        inserted_count = 0

        for item in items:
            gid = str(item.get("gid", "")).strip()
            if not gid:
                continue

            title = str(item.get("title", "Community Update")).strip()
            url = item.get("url")
            author = item.get("author")
            contents = item.get("contents")
            feedlabel = item.get("feedlabel")
            raw_date = item.get("date", 0)
            published_at = datetime.fromtimestamp(raw_date, tz=UTC) if raw_date else datetime.now(UTC)

            existing = await session.execute(
                select(RawPatchNote).where(RawPatchNote.gid == gid)
            )
            record = existing.scalar_one_or_none()
            if record:
                record.title = title
                record.url = url
                record.author = author
                record.contents = contents
                record.feedlabel = feedlabel
                record.published_at = published_at
            else:
                new_note = RawPatchNote(
                    app_id=app_id,
                    gid=gid,
                    title=title,
                    url=url,
                    author=author,
                    contents=contents,
                    feedlabel=feedlabel,
                    published_at=published_at,
                )
                session.add(new_note)
            inserted_count += 1

        return inserted_count

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        """Execute Steam news ingestion pipeline."""
        close_session = False
        if session is None:
            engine = create_async_engine(settings.async_database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            session = session_factory()
            close_session = True

        try:
            if self.app_id:
                games_stmt = select(RawGame).where(RawGame.app_id == self.app_id)
            else:
                games_stmt = select(RawGame)
            games_res = await session.execute(games_stmt)
            target_games = games_res.scalars().all()

            if not target_games:
                self.logger.warning("No games found to fetch news for.")
                return {"games_processed": 0, "total_patch_notes": 0}

            total_notes = 0
            async with httpx.AsyncClient(headers={"User-Agent": "SteamIQ/1.0"}) as client:
                for game in target_games:
                    items = await self.fetch_news_for_app(client, game.app_id)
                    if not self.dry_run and items:
                        count = await self._upsert_patch_notes(session, game.app_id, items)
                        total_notes += count
                    else:
                        total_notes += len(items)

            if not self.dry_run:
                await session.commit()
                self.logger.info(
                    "Successfully ingested %d patch notes across %d games.",
                    total_notes,
                    len(target_games),
                )
            else:
                self.logger.info(
                    "[DRY RUN] Fetched %d patch notes across %d games.",
                    total_notes,
                    len(target_games),
                )

            return {
                "games_processed": len(target_games),
                "total_patch_notes": total_notes,
            }

        finally:
            if close_session:
                await session.close()
                await engine.dispose()

    def run(self) -> dict[str, Any]:
        """Synchronous entry point."""
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest authentic Steam news and patch notes")
    parser.add_argument("--app-id", type=int, default=None, help="Target specific app_id")
    parser.add_argument("--all", action="store_true", help="Fetch news for all catalog games")
    parser.add_argument("--count", type=int, default=25, help="Number of news items per game")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without DB write")
    args = parser.parse_args()

    job = IngestNewsJob(app_id=args.app_id, count=args.count, dry_run=args.dry_run)
    job.execute()


if __name__ == "__main__":
    main()
