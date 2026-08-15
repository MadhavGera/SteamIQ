"""
Game Embedding & Competitor Similarity Pipeline — Phase 3.

Generates dense semantic vector embeddings over game descriptions, tags, and review topics
using sentence-transformers (`all-MiniLM-L6-v2`). Computes pairwise cosine similarity and
populates `model_game_embeddings` and `serving_similar_games`.

Usage:
  python -m jobs.process_embeddings --all
  python -m jobs.process_embeddings --app-id 367520
  python -m jobs.process_embeddings --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from db.models import (
    FeatureReviewTopic,
    ModelGameEmbedding,
    RawGame,
    RawPlayerSnapshot,
    ServingSimilarGame,
)
from jobs.base_job import BaseJob

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_VERSION = "1.0.0"
MAX_SNAPSHOT_AGE_DAYS = 7  # Staleness threshold for CCU telemetry


class ProcessEmbeddingsJob(BaseJob):
    job_name = "process_embeddings"

    def __init__(self, *, app_id: int | None = None, all_games: bool = False, dry_run: bool = False) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.all_games = all_games
        self._model = None

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self.logger.info("Loading sentence transformer model '%s'...", MODEL_NAME)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_NAME)
            except Exception as e:
                self.logger.exception("Failed to load sentence_transformers: %s", e)
                raise
        return self._model

    def build_embedding_text(self, game: RawGame, topics: list[str]) -> str:
        """
        Build rich semantic profile string combining description, genres, tags, and review themes.
        """
        parts = [f"Game: {game.name}"]
        
        if game.short_description:
            parts.append(f"Description: {game.short_description.strip()}")
        elif game.description:
            # First 300 chars of long description
            clean_desc = game.description.strip()[:300]
            parts.append(f"Description: {clean_desc}")

        if game.genres:
            genre_list = [g.get("description", "") for g in game.genres if isinstance(g, dict) and g.get("description")]
            if genre_list:
                parts.append("Genres: " + ", ".join(genre_list))

        if game.tags and isinstance(game.tags, dict):
            # Sort by vote count descending, take top 10 tags
            sorted_tags = sorted(game.tags.items(), key=lambda x: -x[1] if isinstance(x[1], int | float) else 0)[:10]
            tag_names = [t[0] for t in sorted_tags if t[0]]
            if tag_names:
                parts.append("Tags: " + ", ".join(tag_names))

        if topics:
            parts.append("Review themes: " + ", ".join(topics))

        return " | ".join(parts)

    def extract_top_tags(self, game: RawGame) -> set[str]:
        """Extract top SteamSpy tag names for a game."""
        if not game.tags or not isinstance(game.tags, dict):
            return set()
        sorted_tags = sorted(game.tags.items(), key=lambda x: -x[1] if isinstance(x[1], int | float) else 0)[:12]
        return {t[0] for t in sorted_tags if t[0]}

    def derive_market_presence(self, game: RawGame, ccu: int) -> Decimal:
        """
        Derives an additive normalized 0.0 - 100.0 market presence index from review count
        and active player volume (raw_player_snapshots).
        """
        rev_count = (game.positive_reviews or 0) + (game.negative_reviews or 0)
        rev_score = min(60.0, float(np.log10(max(1, rev_count))) * 12.0) if rev_count > 0 else 5.0

        if ccu > 0:
            ccu_score = min(40.0, float(np.log10(max(1, ccu))) * 8.0)
        else:
            ccu_score = min(40.0, (rev_score / 60.0) * 40.0)

        presence = float(np.clip(rev_score + ccu_score, 5.0, 100.0))
        return Decimal(str(round(presence, 2)))

    async def _run_async(self) -> dict[str, Any]:
        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as session:
            # 1. Fetch games to process
            if self.app_id:
                stmt = select(RawGame).where(RawGame.app_id == self.app_id)
            else:
                stmt = select(RawGame).order_by(RawGame.positive_reviews.desc())

            res = await session.execute(stmt)
            target_games = res.scalars().all()

            if not target_games:
                self.logger.warning("No games found to process.")
                return {"games_processed": 0, "pairs_computed": 0}

            # Fetch all games in database for pairwise similarity computation
            all_res = await session.execute(select(RawGame))
            all_games_catalog = all_res.scalars().all()

            # Fetch all review topics for rich embeddings
            topics_res = await session.execute(select(FeatureReviewTopic))
            all_topics = topics_res.scalars().all()
            topics_by_game: dict[int, list[str]] = {}
            for t in all_topics:
                topics_by_game.setdefault(t.app_id, []).append(t.topic_label)

            # Fetch player snapshots for market presence scoring (sorted newest first)
            snapshots_res = await session.execute(
                select(RawPlayerSnapshot).order_by(RawPlayerSnapshot.snapshot_at.desc())
            )
            all_snapshots = snapshots_res.scalars().all()
            snapshots_by_game: dict[int, list[RawPlayerSnapshot]] = {}
            for s in all_snapshots:
                snapshots_by_game.setdefault(s.app_id, []).append(s)

            now_dt = datetime.now(UTC)
            ccu_by_game: dict[int, int] = {}
            for g_id, g_snaps in snapshots_by_game.items():
                latest_snap = g_snaps[0]
                snap_time = latest_snap.snapshot_at if latest_snap.snapshot_at.tzinfo else latest_snap.snapshot_at.replace(tzinfo=UTC)
                age_days = (now_dt - snap_time).total_seconds() / 86400.0

                # Freshness criteria:
                # 1. More than 1 snapshot recorded (indicates recurring monitoring, not a single initial snapshot)
                # 2. Latest snapshot is within MAX_SNAPSHOT_AGE_DAYS (7 days)
                if len(g_snaps) > 1 and age_days <= MAX_SNAPSHOT_AGE_DAYS:
                    recent_snaps = [
                        s for s in g_snaps
                        if (now_dt - (s.snapshot_at if s.snapshot_at.tzinfo else s.snapshot_at.replace(tzinfo=UTC))).total_seconds() / 86400.0 <= MAX_SNAPSHOT_AGE_DAYS
                    ]
                    ccu_by_game[g_id] = max([s.peak_24h or s.player_count or 0 for s in recent_snaps] or [0])
                else:
                    # Stale or isolated snapshot: fall back to review-only market presence calculation
                    ccu_by_game[g_id] = 0

            self.logger.info("Found %d target games (catalog size: %d games)", len(target_games), len(all_games_catalog))

            # 2. Build text representations and compute embeddings
            texts_to_embed = []
            game_records = []
            for game in all_games_catalog:
                topics = topics_by_game.get(game.app_id, [])
                text = self.build_embedding_text(game, topics)
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                texts_to_embed.append(text)
                game_records.append((game, text, text_hash))

            model = self._get_model()
            self.logger.info("Encoding %d game semantic profiles...", len(texts_to_embed))
            
            # Generate unit-normalized embeddings (shape: [N, 384])
            embeddings = model.encode(texts_to_embed, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            emb_by_appid = {record[0].app_id: embeddings[i] for i, record in enumerate(game_records)}

            # 3. Store / update model_game_embeddings
            if not self.dry_run:
                for i, (game, _text, text_hash) in enumerate(game_records):
                    vec_list = embeddings[i].tolist()
                    existing_emb_res = await session.execute(
                        select(ModelGameEmbedding).where(
                            ModelGameEmbedding.app_id == game.app_id,
                            ModelGameEmbedding.model_name == MODEL_NAME,
                        )
                    )
                    existing_emb = existing_emb_res.scalar_one_or_none()

                    if existing_emb:
                        existing_emb.embedding = vec_list
                        existing_emb.text_hash = text_hash
                    else:
                        new_emb = ModelGameEmbedding(
                            app_id=game.app_id,
                            model_name=MODEL_NAME,
                            model_version=MODEL_VERSION,
                            embedding=vec_list,
                            text_hash=text_hash,
                        )
                        session.add(new_emb)

                await session.commit()
                self.logger.info("Saved %d game vectors into model_game_embeddings.", len(game_records))

            # 4. Compute pairwise cosine similarity and populate serving_similar_games
            pairs_computed = 0

            for src_game in target_games:
                src_vec = emb_by_appid.get(src_game.app_id)
                if src_vec is None:
                    continue

                src_tags = self.extract_top_tags(src_game)
                src_price = float(src_game.final_price_usd) if src_game.final_price_usd is not None else 0.0

                # Score against all other games
                candidates = []
                for other_game in all_games_catalog:
                    if other_game.app_id == src_game.app_id:
                        continue
                    
                    other_vec = emb_by_appid.get(other_game.app_id)
                    if other_vec is None:
                        continue

                    # Dot product of normalized vectors = Cosine similarity
                    sim_score = float(np.dot(src_vec, other_vec))
                    # Clamp between 0.0 and 1.0
                    sim_score = max(0.0, min(1.0, sim_score))
                    candidates.append((other_game, sim_score))

                # Sort by similarity score descending
                candidates.sort(key=lambda x: -x[1])
                top_candidates = candidates[:20]  # Store top 20

                if not self.dry_run:
                    # Clear old serving rows for this source game
                    await session.execute(
                        delete(ServingSimilarGame).where(ServingSimilarGame.source_app_id == src_game.app_id)
                    )

                    for rank_idx, (tgt_game, score) in enumerate(top_candidates, start=1):
                        tgt_tags = self.extract_top_tags(tgt_game)
                        shared = list(src_tags.intersection(tgt_tags))[:5]
                        
                        tgt_price = float(tgt_game.final_price_usd) if tgt_game.final_price_usd is not None else 0.0
                        p_delta = round(tgt_price - src_price, 2)
                        market_pres = self.derive_market_presence(tgt_game, ccu_by_game.get(tgt_game.app_id, 0))

                        serving_row = ServingSimilarGame(
                            source_app_id=src_game.app_id,
                            target_app_id=tgt_game.app_id,
                            similarity_score=score,
                            rank=rank_idx,
                            shared_tags=shared,
                            price_delta_usd=Decimal(str(p_delta)),
                            market_presence=market_pres,
                        )
                        session.add(serving_row)
                        pairs_computed += 1

                    await session.commit()

            self.logger.info("Serving table populated: %d similar game pairs stored.", pairs_computed)

        await engine.dispose()
        return {
            "games_processed": len(target_games),
            "catalog_size": len(all_games_catalog),
            "pairs_computed": pairs_computed,
        }

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings & compute competitor similarity (Phase 3)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app-id", type=int, help="Single Steam App ID to compute competitors for")
    group.add_argument("--all", action="store_true", help="Process embeddings and competitors for all games")
    parser.add_argument("--dry-run", action="store_true", help="Validate and encode without database write")
    args = parser.parse_args()

    job = ProcessEmbeddingsJob(
        app_id=args.app_id,
        all_games=args.all,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
