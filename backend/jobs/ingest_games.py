"""
Steam + SteamSpy game ingestion job.

Fetches game metadata, reviews, and player counts from:
  - Steam Store API  (store.steampowered.com/api)
  - Steam Reviews API (store.steampowered.com/appreviews)
  - SteamSpy API    (steamspy.com/api.php)
  - Steam GetNumberOfCurrentPlayers (api.steampowered.com)

Writes only to raw_* tables:
  - raw_games
  - raw_reviews
  - raw_game_tags
  - raw_price_history
  - raw_player_snapshots

ADR 0001, Decision 1: jobs write to raw_*, never to feature_*/serving_*/mart_*.
ADR 0001, Decision 8: all paths are absolute, never os.getcwd()-relative.

Usage:
    # Via Makefile (preferred)
    make ingest APPID=1145360

    # Direct
    cd backend && python -m jobs.ingest_games --appid 1145360 [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from jobs.base_job import BaseJob

logger = logging.getLogger(__name__)

# ─── API endpoints (never hardcoded in source — only the base URLs) ───────────
STEAM_STORE_API     = "https://store.steampowered.com/api/appdetails"
STEAM_REVIEWS_API   = "https://store.steampowered.com/appreviews/{appid}"
STEAM_PLAYERS_API   = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
STEAMSPY_API        = "https://steamspy.com/api.php"

# Review fetch config
REVIEW_BATCH_SIZE   = 100
MAX_REVIEW_PAGES    = 20  # Cap at 2000 reviews per ingestion run
REVIEW_LANGUAGE     = "english"

# Rate limiting — Steam API is strict about request rate
REQUEST_DELAY_S     = 1.5   # seconds between requests to the same API
STEAMSPY_DELAY_S    = 2.0   # SteamSpy is more sensitive


# ─── Ingestion Job ────────────────────────────────────────────────────────────

class IngestGamesJob(BaseJob):
    """
    Full ingestion and recurring price snapshot pipeline for Steam games.
    Writes strictly to raw_* tables only (raw_games, raw_reviews, raw_game_tags,
    raw_price_history, raw_player_snapshots).
    """
    job_name = "ingest_games"

    def __init__(
        self,
        app_id: int | None = None,
        *,
        snapshot_all_prices: bool = False,
        dry_run: bool = False,
        max_reviews: int = MAX_REVIEW_PAGES * REVIEW_BATCH_SIZE,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.snapshot_all_prices = snapshot_all_prices
        self.max_reviews = max_reviews
        self._steam_api_key: str = os.environ.get("STEAM_API_KEY", "")
        if not self._steam_api_key:
            logger.warning(
                "STEAM_API_KEY is not set — some endpoints will fail. "
                "Set it in .env (see .env.example)."
            )

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    async def _get(
        self, client: httpx.AsyncClient, url: str, params: dict[str, Any], delay: float = REQUEST_DELAY_S
    ) -> dict[str, Any] | None:
        """GET with error handling and rate-limit delay."""
        try:
            await asyncio.sleep(delay)
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %s from %s: %s", exc.response.status_code, url, exc)
            return None
        except Exception as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    async def fetch_steam_details(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Fetch game metadata from Steam Store API."""
        data = await self._get(
            client,
            STEAM_STORE_API,
            {"appids": self.app_id, "cc": "us", "l": "english"},
            delay=0,  # first call — no prior call to rate-limit against
        )
        if not data:
            return None
        app_data = data.get(str(self.app_id), {})
        if not app_data.get("success"):
            logger.warning("Steam API returned success=false for app_id=%s", self.app_id)
            return None
        return app_data.get("data")

    async def fetch_steamspy_details(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """Fetch ownership estimates and tags from SteamSpy."""
        return await self._get(
            client,
            STEAMSPY_API,
            {"request": "appdetails", "appid": self.app_id},
            delay=STEAMSPY_DELAY_S,
        )

    async def fetch_reviews(
        self, client: httpx.AsyncClient
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """
        Paginate through Steam review API.
        Stops at max_reviews or when no more reviews are returned.
        Returns (reviews_list, query_summary_dict).
        """
        reviews: list[dict[str, Any]] = []
        query_summary: dict[str, Any] | None = None
        cursor = "*"
        pages_fetched = 0

        while len(reviews) < self.max_reviews:
            data = await self._get(
                client,
                STEAM_REVIEWS_API.format(appid=self.app_id),
                {
                    "json": 1,
                    "language": REVIEW_LANGUAGE,
                    "filter": "recent",
                    "num_per_page": REVIEW_BATCH_SIZE,
                    "cursor": cursor,
                    "purchase_type": "steam",
                },
            )
            if not data or not data.get("success"):
                break

            if query_summary is None and "query_summary" in data:
                query_summary = data.get("query_summary")

            batch = data.get("reviews", [])
            if not batch:
                break

            reviews.extend(batch)
            cursor = data.get("cursor", "")
            pages_fetched += 1

            if pages_fetched >= MAX_REVIEW_PAGES:
                logger.info("Reached max review pages (%s) for app_id=%s", MAX_REVIEW_PAGES, self.app_id)
                break

        logger.info("Fetched %s reviews for app_id=%s", len(reviews), self.app_id)
        return reviews, query_summary

    async def fetch_player_count(self, client: httpx.AsyncClient) -> int | None:
        """Get current concurrent player count."""
        if not self._steam_api_key:
            return None
        data = await self._get(
            client,
            STEAM_PLAYERS_API,
            {"appid": self.app_id, "key": self._steam_api_key},
        )
        if data and data.get("response", {}).get("result") == 1:
            return data["response"].get("player_count")
        return None

    # ── Parse helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_price(steam_data: dict[str, Any]) -> tuple[Decimal | None, Decimal | None, int]:
        """Return (price_usd, final_price_usd, discount_pct) as Decimal."""
        price_overview = steam_data.get("price_overview", {})
        if not price_overview:
            return None, None, 0

        # Steam returns prices in the smallest currency unit (cents for USD)
        initial = price_overview.get("initial")
        final   = price_overview.get("final")
        discount = price_overview.get("discount_percent", 0)

        price_usd       = Decimal(initial) / 100 if initial is not None else None
        final_price_usd = Decimal(final)   / 100 if final   is not None else None
        return price_usd, final_price_usd, discount

    @staticmethod
    def _extract_genres(steam_data: dict[str, Any]) -> list[dict]:
        return steam_data.get("genres", [])

    @staticmethod
    def _extract_categories(steam_data: dict[str, Any]) -> list[dict]:
        return steam_data.get("categories", [])

    # ── DB write helpers ───────────────────────────────────────────────────────

    async def _upsert_game(
        self,
        session: AsyncSession,
        steam_data: dict[str, Any],
        spy_data: dict[str, Any] | None,
        query_summary: dict[str, Any] | None = None,
    ) -> None:
        """Upsert into raw_games. On conflict (app_id), update all fields."""
        from db.models import RawGame  # local import avoids circular at module level

        price_usd, final_price_usd, discount_pct = self._parse_price(steam_data)
        release = steam_data.get("release_date", {})

        # Extract review score (1-9 scale) and description from Steam query_summary
        steam_review_score = query_summary.get("review_score") if query_summary else None
        steam_review_score_desc = query_summary.get("review_score_desc") if query_summary else None

        pos_count = (
            int(spy_data.get("positive", 0) or 0)
            if (spy_data and spy_data.get("positive"))
            else (
                query_summary.get("total_positive")
                if (query_summary and query_summary.get("total_positive") is not None)
                else (steam_data.get("recommendations", {}).get("total", 0) if steam_data.get("recommendations") else 0)
            )
        )
        neg_count = (
            int(spy_data.get("negative", 0) or 0)
            if (spy_data and spy_data.get("negative"))
            else (
                query_summary.get("total_negative", 0)
                if (query_summary and query_summary.get("total_negative") is not None)
                else 0
            )
        )

        row = {
            "app_id": self.app_id,
            "name": steam_data.get("name", ""),
            "description": steam_data.get("detailed_description"),
            "short_description": steam_data.get("short_description"),
            "developer": ", ".join(steam_data.get("developers", [])),
            "publisher": ", ".join(steam_data.get("publishers", [])),
            "release_date": release.get("date") if release else None,
            "coming_soon": release.get("coming_soon", False) if release else False,
            "genres": self._extract_genres(steam_data),
            "categories": self._extract_categories(steam_data),
            "tags": spy_data.get("tags") if spy_data else None,
            "is_free": steam_data.get("is_free", False),
            "price_usd": price_usd,
            "final_price_usd": final_price_usd,
            "discount_pct": discount_pct,
            "platform_windows": steam_data.get("platforms", {}).get("windows", False),
            "platform_mac": steam_data.get("platforms", {}).get("mac", False),
            "platform_linux": steam_data.get("platforms", {}).get("linux", False),
            "positive_reviews": pos_count,
            "negative_reviews": neg_count,
            "review_score": steam_review_score,
            "review_score_desc": steam_review_score_desc,
            "owners_estimate": spy_data.get("owners") if spy_data else None,
            "average_playtime_forever": spy_data.get("average_forever", 0) if spy_data else 0,
            "median_playtime_forever": spy_data.get("median_forever", 0) if spy_data else 0,
            "metacritic_score": steam_data.get("metacritic", {}).get("score") if steam_data.get("metacritic") else None,
            "header_image": steam_data.get("header_image"),
            "website": steam_data.get("website"),
        }

        stmt = pg_insert(RawGame).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["app_id"],
            set_={k: v for k, v in row.items() if k != "app_id"},
        )
        await session.execute(stmt)

    async def _upsert_reviews(
        self, session: AsyncSession, reviews: list[dict[str, Any]]
    ) -> int:
        """Upsert reviews into raw_reviews. Skip duplicates by review_id."""
        from db.models import RawReview

        upserted = 0
        for r in reviews:
            author = r.get("author", {})
            row = {
                "review_id": str(r.get("recommendationid", "")),
                "app_id": self.app_id,
                "author_steam_id": author.get("steamid"),
                "author_playtime_forever": author.get("playtime_forever", 0),
                "author_playtime_at_review": author.get("playtime_at_review", 0),
                "author_num_reviews": author.get("num_reviews", 0),
                "language": r.get("language", "english"),
                "review_text": r.get("review"),
                "voted_up": bool(r.get("voted_up", False)),
                "votes_up": r.get("votes_up", 0),
                "votes_funny": r.get("votes_funny", 0),
                "weighted_vote_score": Decimal(str(r.get("weighted_vote_score", "0"))) if r.get("weighted_vote_score") else None,
                "steam_purchase": bool(r.get("steam_purchase", True)),
                "received_for_free": bool(r.get("received_for_free", False)),
                "written_during_early_access": bool(r.get("written_during_early_access", False)),
                "review_created_at": r.get("timestamp_created"),
                "review_updated_at": r.get("timestamp_updated"),
            }
            stmt = pg_insert(RawReview).values(**row)
            stmt = stmt.on_conflict_do_nothing(index_elements=["review_id"])
            result = await session.execute(stmt)
            upserted += result.rowcount

        return upserted

    async def _upsert_tags(
        self, session: AsyncSession, spy_data: dict[str, Any]
    ) -> None:
        """Upsert tags from SteamSpy into raw_game_tags."""
        from db.models import RawGameTag

        tags: dict[str, int] = spy_data.get("tags") or {}
        for tag_name, votes in tags.items():
            row = {"app_id": self.app_id, "tag_name": tag_name, "votes": votes}
            stmt = pg_insert(RawGameTag).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["app_id", "tag_name"],
                set_={"votes": votes},
            )
            await session.execute(stmt)

    async def _insert_price_snapshot(
        self,
        session: AsyncSession,
        steam_data: dict[str, Any],
        app_id: int | None = None,
    ) -> None:
        """Record current price in raw_price_history."""
        from db.models import RawPriceHistory

        target_id = app_id or self.app_id
        if target_id is None:
            return

        price_usd, final_price_usd, discount_pct = self._parse_price(steam_data)
        row = {
            "app_id": target_id,
            "recorded_at": datetime.now(UTC),
            "price_usd": price_usd,
            "final_price_usd": final_price_usd,
            "discount_pct": discount_pct,
            "currency": "USD",
        }
        stmt = pg_insert(RawPriceHistory).values(**row)
        stmt = stmt.on_conflict_do_nothing(index_elements=["app_id", "recorded_at"])
        await session.execute(stmt)

    async def _insert_player_snapshot(
        self, session: AsyncSession, player_count: int | None
    ) -> None:
        """Record current player count in raw_player_snapshots."""
        if player_count is None or self.app_id is None:
            return
        from db.models import RawPlayerSnapshot

        row = {
            "app_id": self.app_id,
            "snapshot_at": datetime.now(UTC),
            "player_count": player_count,
            "source": "steam_api",
        }
        stmt = pg_insert(RawPlayerSnapshot).values(**row)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["app_id", "snapshot_at", "source"]
        )
        await session.execute(stmt)

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Synchronous entry point — delegates to async _run()."""
        return asyncio.run(self._run())

    async def _run_recurring_price_snapshots(self) -> dict[str, Any]:
        """
        Recurring price snapshot pipeline:
        Iterates over all games in raw_games and records a point-in-time price snapshot
        in raw_price_history to maintain a continuous, reliable historical price time-series.

        TODO(Phase7_Step7.2): Wire this recurring snapshot method to an RQ scheduler / cron daemon.
        Currently invoked manually or via scheduled CLI job runner (`python -m jobs.ingest_games --snapshot-prices`).
        materialize_marts.py degrades gracefully to reporting tracking_since and current price when only
        1 or 2 snapshots exist.
        """
        from sqlalchemy import select

        from db.base import _build_database_url
        from db.models import RawGame

        engine = create_async_engine(_build_database_url(), pool_pre_ping=True)
        SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

        async with SessionLocal() as session:
            res = await session.execute(select(RawGame.app_id, RawGame.name))
            catalog = res.all()

        total = len(catalog)
        self.logger.info("Starting recurring price snapshot run for %d catalog games...", total)
        snapshotted = 0

        async with httpx.AsyncClient(
            headers={"User-Agent": "SteamIQ-PriceTracker/0.1"},
            follow_redirects=True,
        ) as client:
            for g_id, g_name in catalog:
                try:
                    data = await self._get(
                        client,
                        STEAM_STORE_API,
                        {"appids": g_id, "cc": "us", "l": "english"},
                        delay=REQUEST_DELAY_S,
                    )
                    if data:
                        app_data = data.get(str(g_id), {})
                        if app_data.get("success"):
                            steam_data = app_data.get("data", {})
                            if not self.dry_run:
                                async with SessionLocal() as session, session.begin():
                                    await self._insert_price_snapshot(session, steam_data, app_id=g_id)
                            snapshotted += 1
                except Exception as e:
                    self.logger.warning("Price snapshot error for %s (%s): %s", g_name, g_id, e)

        await engine.dispose()
        self.logger.info("Completed recurring price snapshot: %d/%d recorded.", snapshotted, total)
        return {
            "mode": "recurring_price_snapshots",
            "total_games": total,
            "snapshotted": snapshotted,
        }

    async def _run(self) -> dict[str, Any]:
        if self.snapshot_all_prices or (self.app_id is None):
            return await self._run_recurring_price_snapshots()

        stats: dict[str, Any] = {
            "app_id": self.app_id,
            "reviews_upserted": 0,
            "tags_upserted": 0,
        }

        async with httpx.AsyncClient(
            headers={"User-Agent": "SteamIQ-Ingestion/0.1"},
            follow_redirects=True,
        ) as client:
            # 1. Fetch data from all sources
            steam_data = await self.fetch_steam_details(client)
            if not steam_data:
                self.logger.error("No Steam data for app_id=%s — aborting", self.app_id)
                return {"app_id": self.app_id, "error": "steam_api_no_data"}

            stats["game_name"] = steam_data.get("name", "unknown")

            spy_data = await self.fetch_steamspy_details(client)
            reviews, query_summary = await self.fetch_reviews(client)
            player_count = await self.fetch_player_count(client)

            stats["reviews_fetched"] = len(reviews)
            stats["player_count"] = player_count
            if query_summary:
                stats["review_score"] = query_summary.get("review_score")
                stats["review_score_desc"] = query_summary.get("review_score_desc")

            if self.dry_run:
                self.logger.info(
                    "[DRY RUN] Would write game=%r, reviews=%d, player_count=%s",
                    stats["game_name"], len(reviews), player_count,
                )
                return stats

            # 2. Write to DB (raw_* tables only)
            from db.base import _build_database_url
            engine = create_async_engine(_build_database_url(), pool_pre_ping=True)
            SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

            async with SessionLocal() as session, session.begin():
                await self._upsert_game(session, steam_data, spy_data, query_summary=query_summary)
                reviews_count = await self._upsert_reviews(session, reviews)
                if spy_data:
                    await self._upsert_tags(session, spy_data)
                await self._insert_price_snapshot(session, steam_data)
                await self._insert_player_snapshot(session, player_count)

            await engine.dispose()

            stats["reviews_upserted"] = reviews_count
            self.logger.info(
                "Ingested app_id=%s (%s): %d reviews, player_count=%s",
                self.app_id, stats["game_name"], reviews_count, player_count,
            )

        return stats


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Steam game metadata and recurring price history")
    parser.add_argument("--appid", type=int, default=None, help="Steam app ID for single game ingestion")
    parser.add_argument(
        "--snapshot-prices",
        "--all-prices",
        action="store_true",
        help="Run recurring scheduled price snapshots for all catalog games",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't write to DB")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=MAX_REVIEW_PAGES * REVIEW_BATCH_SIZE,
        help=f"Max reviews to fetch (default: {MAX_REVIEW_PAGES * REVIEW_BATCH_SIZE})",
    )
    args = parser.parse_args()

    if not args.appid and not args.snapshot_prices:
        parser.error("Must provide either --appid <ID> or --snapshot-prices")

    # Load .env from the project root (parent of backend/)
    from pathlib import Path
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)
            logger.info("Loaded .env from %s", env_file)
        except ImportError:
            logger.warning("python-dotenv not installed — .env not loaded")

    job = IngestGamesJob(
        app_id=args.appid,
        snapshot_all_prices=args.snapshot_prices,
        dry_run=args.dry_run,
        max_reviews=args.max_reviews,
    )
    job.execute(app_id=args.appid, snapshot_prices=args.snapshot_prices, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
