"""
Catalog Seeding Job for SteamIQ.

Ingests a curated benchmark catalog of top indie and commercial games across
distinct genres (Metroidvania, Roguelike, Deckbuilder, Simulation, Action RPG)
into raw_* tables to provide a rich dataset for:
  - Phase 3: pgvector game embeddings & competitor discovery
  - Phase 4: Tabular ML feature engineering & baseline models

Usage:
    # Seed top 10 benchmark games (fast)
    python -m jobs.seed_catalog --limit 10

    # Seed full 30-game benchmark catalog
    python -m jobs.seed_catalog --all
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Any

import httpx

from jobs.base_job import BaseJob
from jobs.ingest_games import IngestGamesJob

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ─── Curated Benchmark Catalog (30 Diverse Games) ─────────────────────────────
CURATED_CATALOG: list[dict[str, Any]] = [
    # ── Metroidvanias & Action Platformers ──
    {"app_id": 367520, "name": "Hollow Knight", "genre": "Metroidvania"},
    {"app_id": 1145360, "name": "Hades", "genre": "Action Roguelike"},
    {"app_id": 504230, "name": "Celeste", "genre": "Precision Platformer"},
    {"app_id": 588650, "name": "Dead Cells", "genre": "Rogue-lite Metroidvania"},
    {"app_id": 1057090, "name": "Ori and the Will of the Wisps", "genre": "Metroidvania"},
    {"app_id": 774361, "name": "Blasphemous", "genre": "Dark Metroidvania"},
    {"app_id": 1369630, "name": "ENDER LILIES: Quietus of the Knights", "genre": "Dark Fantasy Metroidvania"},
    {"app_id": 250760, "name": "Shovel Knight: Treasure Trove", "genre": "Retro Platformer"},

    # ── Roguelike Deckbuilders & Strategy ──
    {"app_id": 646570, "name": "Slay the Spire", "genre": "Roguelike Deckbuilder"},
    {"app_id": 1382330, "name": "Inscryption", "genre": "Psychological Deckbuilder"},
    {"app_id": 1102190, "name": "Monster Train", "genre": "Strategic Deckbuilder"},
    {"app_id": 2379780, "name": "Balatro", "genre": "Poker Roguelike"},
    {"app_id": 242680, "name": "FTL: Faster Than Light", "genre": "Spaceship Roguelike"},

    # ── Action Roguelikes & Bullet Heaven ──
    {"app_id": 1794680, "name": "Vampire Survivors", "genre": "Bullet Heaven"},
    {"app_id": 632360, "name": "Risk of Rain 2", "genre": "3D Action Roguelike"},
    {"app_id": 250900, "name": "The Binding of Isaac: Rebirth", "genre": "Twin-stick Roguelike"},
    {"app_id": 311690, "name": "Enter the Gungeon", "genre": "Bullet Hell Roguelike"},
    {"app_id": 1942280, "name": "Brotato", "genre": "Arena Shooter Roguelike"},
    {"app_id": 881100, "name": "Noita", "genre": "Pixel Physics Roguelike"},

    # ── Cozy, Simulation & Survival ──
    {"app_id": 413150, "name": "Stardew Valley", "genre": "Farming Simulation"},
    {"app_id": 105600, "name": "Terraria", "genre": "Sandbox Survival"},
    {"app_id": 892970, "name": "Valheim", "genre": "Viking Survival"},
    {"app_id": 1084600, "name": "My Time at Sandrock", "genre": "Crafting Simulation"},

    # ── Commercial Benchmark Games ──
    {"app_id": 1091500, "name": "Cyberpunk 2077", "genre": "Open World RPG"},
    {"app_id": 292030, "name": "The Witcher 3: Wild Hunt", "genre": "Action RPG"},
    {"app_id": 1245620, "name": "ELDEN RING", "genre": "Open World Action RPG"},
    {"app_id": 1086940, "name": "Baldur's Gate 3", "genre": "CRPG"},
    {"app_id": 477160, "name": "Human Fall Flat", "genre": "Physics Co-op"},
    {"app_id": 252490, "name": "Rust", "genre": "Multiplayer Survival"},
    {"app_id": 570, "name": "Dota 2", "genre": "MOBA"},
]


class SeedCatalogJob(BaseJob):
    """
    Seeds a curated catalog of Steam games sequentially with rate-limit respect.
    """
    job_name = "seed_catalog"

    def __init__(
        self,
        *,
        limit: int | None = None,
        reviews_per_game: int = 200,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.limit = limit
        self.reviews_per_game = reviews_per_game

    def run(self) -> dict[str, Any]:
        """Synchronous entry point."""
        targets = CURATED_CATALOG[: self.limit] if self.limit else CURATED_CATALOG
        total_targets = len(targets)

        logger.info("🚀 Starting Catalog Seeder: %d target games (max %d reviews/game)", total_targets, self.reviews_per_game)

        successful: list[str] = []
        failed: list[dict[str, Any]] = []

        start_time = time.perf_counter()

        for idx, item in enumerate(targets, 1):
            app_id = item["app_id"]
            name = item["name"]
            genre = item["genre"]

            logger.info("[%d/%d] Ingesting %s (App ID %d, Genre: %s)...", idx, total_targets, name, app_id, genre)

            try:
                job = IngestGamesJob(
                    app_id=app_id,
                    max_reviews=self.reviews_per_game,
                    dry_run=self.dry_run,
                )
                result = job.run()

                if "error" in result:
                    logger.warning("⚠️ Failed to ingest %s (%d): %s", name, app_id, result["error"])
                    failed.append({"app_id": app_id, "name": name, "error": result["error"]})
                else:
                    reviews_count = result.get("reviews_upserted", 0)
                    logger.info("✅ Success: %s (%d reviews upserted)", name, reviews_count)
                    successful.append(name)

            except Exception as exc:
                logger.error("❌ Exception during %s (%d): %s", name, app_id, exc)
                failed.append({"app_id": app_id, "name": name, "error": str(exc)})

            # Polite pause between games to respect Steam API guidelines
            if idx < total_targets:
                time.sleep(1.5)

        elapsed = round(time.perf_counter() - start_time, 1)

        summary = {
            "total_requested": total_targets,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful_games": successful,
            "failed_games": failed,
            "elapsed_seconds": elapsed,
        }

        logger.info(
            "🎉 Seeding Complete in %.1fs: %d succeeded, %d failed.",
            elapsed,
            len(successful),
            len(failed),
        )

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed SteamIQ database with curated benchmark games")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of games to seed (e.g. 10)")
    parser.add_argument("--all", action="store_true", help="Seed all 30 benchmark games")
    parser.add_argument("--reviews-per-game", type=int, default=200, help="Reviews per game (default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without DB writes")
    args = parser.parse_args()

    limit = None if args.all else (args.limit or 10)

    job = SeedCatalogJob(
        limit=limit,
        reviews_per_game=args.reviews_per_game,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
