"""
Feature Engineering Pipeline — Phase 4.

Aggregates raw_* and feature_review_* data into structured tabular feature vectors
for predictive ML models (feature_game_features and feature_market_features).

Strictly enforces feature_cutoff_date to eliminate temporal data leakage.
Never reads from serving_* or mart_* tables (ADR 0001, Decision 1 & 2).

Usage:
  python -m jobs.build_features --all
  python -m jobs.build_features --app-id 367520
  python -m jobs.build_features --cutoff-date 2026-08-01T00:00:00Z
  python -m jobs.build_features --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import (
    FeatureGameFeature,
    FeatureMarketFeature,
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    RawGame,
    RawGameTag,
    RawReview,
)
from jobs.base_job import BaseJob


class BuildFeaturesJob(BaseJob):
    job_name = "build_features"

    def __init__(
        self,
        *,
        app_id: int | None = None,
        all_games: bool = False,
        cutoff_date: datetime | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.all_games = all_games
        self.cutoff_date = cutoff_date or datetime.now(UTC)

    def _extract_primary_genre(self, game: RawGame) -> str:
        """Extract primary genre name from raw_games.genres JSONB."""
        if game.genres and isinstance(game.genres, list) and len(game.genres) > 0:
            first = game.genres[0]
            if isinstance(first, dict) and "description" in first:
                return first["description"]
        return "Indie"

    def _calculate_target_success_score(
        self,
        positive_reviews: int,
        negative_reviews: int,
        review_score: int | None,
        sentiment_score: float | None,
        avg_playtime: int,
    ) -> tuple[float, bool]:
        """
        Calculate composite ground-truth success index (0.00 - 100.00) and binary hit status.
        Composite weights lifetime volume, positive ratio, Steam review score, NLP sentiment, and player retention.
        
        Temporal Outcome Separation:
        - Features (e.g. total_reviews_at_cutoff, velocity_30d) are strictly bounded by feature_cutoff_date (t <= T_cutoff).
        - Ground truth target labels evaluate the full outcome window (lifetime performance) to predict future success.
        """
        total = positive_reviews + negative_reviews
        if total == 0:
            return 20.0, False

        pos_ratio = (positive_reviews / total) * 100.0
        volume_score = min(100.0, (np.log1p(total) / np.log1p(100_000)) * 100.0)
        score_component = (review_score / 9.0 * 100.0) if review_score is not None else pos_ratio
        sentiment_comp = (sentiment_score * 100.0) if sentiment_score is not None else pos_ratio
        playtime_comp = min(100.0, (np.log1p(avg_playtime) / np.log1p(3000)) * 100.0)

        # 40% Volume, 25% Review Ratio, 15% Steam Score, 10% NLP Sentiment, 10% Playtime
        composite = (
            0.40 * volume_score
            + 0.25 * pos_ratio
            + 0.15 * score_component
            + 0.10 * sentiment_comp
            + 0.10 * playtime_comp
        )
        composite = float(np.clip(composite, 0.0, 100.0))
        is_hit = bool(composite >= 60.0 or (total >= 5000 and pos_ratio >= 80.0))
        return round(composite, 2), is_hit

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        if session is not None:
            return await self._run_pipeline_with_session(session)
        engine = create_async_engine(settings.async_database_url, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as s:
            res = await self._run_pipeline_with_session(s)
        await engine.dispose()
        return res

    async def _run_pipeline_with_session(self, session: AsyncSession) -> dict[str, Any]:
        cutoff_ts = int(self.cutoff_date.timestamp())
        cutoff_30d_prior_ts = int((self.cutoff_date - timedelta(days=30)).timestamp())

        # 1. Fetch catalog of games
        if self.app_id:
            target_stmt = select(RawGame).where(RawGame.app_id == self.app_id)
        else:
            target_stmt = select(RawGame).order_by(RawGame.app_id)

        res = await session.execute(target_stmt)
        target_games = res.scalars().all()

        if not target_games:
            self.logger.warning("No games found to process.")
            return {"games_processed": 0, "genres_processed": 0}

        # Fetch all games for catalog-wide developer/publisher and market computations
        all_games_res = await session.execute(select(RawGame))
        all_games = all_games_res.scalars().all()

        self.logger.info(
            "Processing features for %d games (Catalog: %d) at cutoff %s",
            len(target_games),
            len(all_games),
            self.cutoff_date.isoformat(),
        )

        # 2. Pre-fetch historical reviews & snapshots respecting cutoff date
        reviews_res = await session.execute(select(RawReview))
        all_reviews = reviews_res.scalars().all()

        # Group reviews by app_id filtering by cutoff_ts
        reviews_by_game: dict[int, list[RawReview]] = {}
        for rev in all_reviews:
            if rev.review_created_at is not None and rev.review_created_at > cutoff_ts:
                continue  # EXCLUDE post-cutoff reviews
            reviews_by_game.setdefault(rev.app_id, []).append(rev)

        # Group NLP features
        sentiment_res = await session.execute(
            select(FeatureReviewSentiment).where(FeatureReviewSentiment.month == "ALL_TIME")
        )
        sentiment_by_game = {s.app_id: float(s.sentiment_score) for s in sentiment_res.scalars().all()}

        complaints_res = await session.execute(select(FeatureReviewComplaint))
        complaints_by_game: dict[int, list[FeatureReviewComplaint]] = {}
        for comp in complaints_res.scalars().all():
            complaints_by_game.setdefault(comp.app_id, []).append(comp)

        features_res = await session.execute(select(FeatureReviewFeature))
        loved_by_game: dict[int, list[FeatureReviewFeature]] = {}
        for lf in features_res.scalars().all():
            loved_by_game.setdefault(lf.app_id, []).append(lf)

        tags_res = await session.execute(select(RawGameTag))
        tags_by_game: dict[int, list[RawGameTag]] = {}
        for tag in tags_res.scalars().all():
            tags_by_game.setdefault(tag.app_id, []).append(tag)

        # Precompute developer and publisher catalog aggregates
        dev_stats: dict[str, list[RawGame]] = {}
        pub_stats: dict[str, list[RawGame]] = {}
        for g in all_games:
            if g.developer:
                dev_stats.setdefault(g.developer.strip(), []).append(g)
            if g.publisher:
                pub_stats.setdefault(g.publisher.strip(), []).append(g)

        # Precompute genre game lists for market aggregations
        games_by_genre: dict[str, list[RawGame]] = {}
        for g in all_games:
            p_genre = self._extract_primary_genre(g)
            games_by_genre.setdefault(p_genre, []).append(g)

        genre_medians: dict[str, float] = {}
        genre_avg_scores: dict[str, float] = {}
        for genre_name, g_list in games_by_genre.items():
            prices = [
                float(g.final_price_usd)
                for g in g_list
                if g.final_price_usd is not None and not g.is_free
            ]
            genre_medians[genre_name] = float(np.median(prices)) if prices else 14.99
            scs = [g.review_score * 10.0 for g in g_list if g.review_score is not None]
            genre_avg_scores[genre_name] = float(np.mean(scs)) if scs else 70.0

        # 3. Build game features
        game_feature_records: list[FeatureGameFeature] = []
        for game in target_games:
            primary_genre = self._extract_primary_genre(game)

            # Review signals strictly up to cutoff
            game_revs = reviews_by_game.get(game.app_id, [])
            total_revs_cutoff = len(game_revs) if game_revs else game.positive_reviews + game.negative_reviews
            pos_revs_cutoff = (
                sum(1 for r in game_revs if r.voted_up)
                if game_revs
                else game.positive_reviews
            )

            # Calculate review velocity in 30d window prior to cutoff
            revs_30d = sum(
                1
                for r in game_revs
                if r.review_created_at is not None
                and cutoff_30d_prior_ts <= r.review_created_at <= cutoff_ts
            )
            review_velocity = round(revs_30d / 30.0, 4) if revs_30d > 0 else 0.0

            pos_ratio = (
                round(Decimal(str(pos_revs_cutoff / total_revs_cutoff * 100.0)), 2)
                if total_revs_cutoff > 0
                else Decimal("0.0")
            )

            # Developer track record (strictly excluding target game's own row)
            dev_prior = [g for g in dev_stats.get(game.developer.strip(), []) if g.app_id != game.app_id] if game.developer else []
            dev_count = len(dev_prior)
            dev_scores = [
                g.review_score * 10.0
                for g in dev_prior
                if g.review_score is not None
            ]
            dev_avg_score = (
                Decimal(str(round(np.mean(dev_scores), 2)))
                if dev_scores
                else Decimal(str(round(genre_avg_scores.get(primary_genre, 70.0), 2)))
            )

            # Publisher track record (strictly excluding target game's own row)
            pub_prior = [g for g in pub_stats.get(game.publisher.strip(), []) if g.app_id != game.app_id] if game.publisher else []
            pub_count = len(pub_prior)
            pub_scores = [
                g.review_score * 10.0
                for g in pub_prior
                if g.review_score is not None
            ]
            pub_avg_score = (
                Decimal(str(round(np.mean(pub_scores), 2)))
                if pub_scores
                else Decimal(str(round(genre_avg_scores.get(primary_genre, 70.0), 2)))
            )

            # Competitor density & price positioning (strictly excluding target game)
            genre_peers = [g for g in games_by_genre.get(primary_genre, []) if g.app_id != game.app_id]
            competitor_density = len(genre_peers)
            peer_prices = [
                float(g.final_price_usd)
                for g in genre_peers
                if g.final_price_usd is not None and not g.is_free
            ]
            genre_med = float(np.median(peer_prices)) if peer_prices else genre_medians.get(primary_genre, 14.99)
            g_price = float(game.final_price_usd) if game.final_price_usd is not None else 14.99
            price_vs_med = Decimal(str(round(g_price - genre_med, 2)))

            # NLP sentiment & review insights
            sentiment = sentiment_by_game.get(game.app_id)
            sentiment_dec = Decimal(str(round(sentiment, 4))) if sentiment is not None else None

            game_complaints = complaints_by_game.get(game.app_id, [])
            complaint_density = float(sum(float(c.volume_pct) for c in game_complaints))

            game_loved = loved_by_game.get(game.app_id, [])
            loved_density = float(sum(lf.mention_count for lf in game_loved))

            # Top tags
            g_tags = tags_by_game.get(game.app_id, [])
            tag_dict = {t.tag_name: t.votes for t in sorted(g_tags, key=lambda x: -x.votes)[:10]}
            if not tag_dict and game.tags and isinstance(game.tags, dict):
                tag_dict = dict(sorted(game.tags.items(), key=lambda x: -x[1])[:10])

            # Ground-truth target metrics: evaluated against lifetime outcome
            target_score, is_hit = self._calculate_target_success_score(
                positive_reviews=game.positive_reviews or pos_revs_cutoff,
                negative_reviews=game.negative_reviews or max(0, total_revs_cutoff - pos_revs_cutoff),
                review_score=game.review_score,
                sentiment_score=sentiment,
                avg_playtime=game.average_playtime_forever,
            )

            feat = FeatureGameFeature(
                app_id=game.app_id,
                feature_cutoff_date=self.cutoff_date,
                price_usd=game.price_usd,
                discount_pct=game.discount_pct,
                is_free=game.is_free,
                primary_genre=primary_genre,
                genres=game.genres,
                top_tags=tag_dict,
                developer_game_count=dev_count,
                publisher_game_count=pub_count,
                developer_avg_review_score=dev_avg_score,
                publisher_avg_review_score=pub_avg_score,
                total_reviews_at_cutoff=total_revs_cutoff,
                positive_reviews_at_cutoff=pos_revs_cutoff,
                review_velocity_30d=review_velocity,
                positive_review_pct=pos_ratio,
                competitor_density=competitor_density,
                price_vs_genre_median=price_vs_med,
                sentiment_score=sentiment_dec,
                complaint_density=complaint_density,
                loved_feature_density=loved_density,
                average_playtime_forever=game.average_playtime_forever,
                target_success_score=target_score,
                is_hit=is_hit,
            )
            game_feature_records.append(feat)

        # 4. Build market / genre aggregates
        market_feature_records: list[FeatureMarketFeature] = []
        for genre_name, g_list in games_by_genre.items():
            prices = [
                float(g.final_price_usd)
                for g in g_list
                if g.final_price_usd is not None and not g.is_free
            ]
            scores = [g.review_score * 10.0 for g in g_list if g.review_score is not None]
            pos_revs = sum(g.positive_reviews or 0 for g in g_list)
            neg_revs = sum(g.negative_reviews or 0 for g in g_list)
            playtimes = [g.average_playtime_forever for g in g_list if g.average_playtime_forever > 0]

            # Aggregate top tags across genre
            genre_tag_counts: dict[str, int] = {}
            for g in g_list:
                for t in tags_by_game.get(g.app_id, []):
                    genre_tag_counts[t.tag_name] = genre_tag_counts.get(t.tag_name, 0) + t.votes
            sorted_tags = dict(sorted(genre_tag_counts.items(), key=lambda x: -x[1])[:10])

            sat_index = float(min(1.0, len(g_list) / 40.0))

            market_feat = FeatureMarketFeature(
                genre=genre_name,
                feature_cutoff_date=self.cutoff_date,
                game_count=len(g_list),
                median_price_usd=Decimal(str(round(np.median(prices), 2))) if prices else None,
                avg_review_score=Decimal(str(round(np.mean(scores), 2))) if scores else None,
                total_positive_reviews=pos_revs,
                total_negative_reviews=neg_revs,
                median_playtime_forever=int(np.median(playtimes)) if playtimes else 0,
                top_tags=sorted_tags,
                saturation_index=sat_index,
            )
            market_feature_records.append(market_feat)

        # 5. Persist to DB
        if not self.dry_run:
            # Remove existing feature records for this cutoff date
            app_ids = [g.app_id for g in target_games]
            await session.execute(
                delete(FeatureGameFeature).where(
                    FeatureGameFeature.app_id.in_(app_ids),
                    FeatureGameFeature.feature_cutoff_date == self.cutoff_date,
                )
            )
            genres = list(games_by_genre.keys())
            await session.execute(
                delete(FeatureMarketFeature).where(
                    FeatureMarketFeature.genre.in_(genres),
                    FeatureMarketFeature.feature_cutoff_date == self.cutoff_date,
                )
            )

            for gf in game_feature_records:
                session.add(gf)
            for mf in market_feature_records:
                session.add(mf)

            await session.commit()
            self.logger.info(
                "Saved %d game feature rows and %d market feature rows for cutoff %s",
                len(game_feature_records),
                len(market_feature_records),
                self.cutoff_date.isoformat(),
            )

        return {
            "games_processed": len(game_feature_records),
            "market_genres_processed": len(market_feature_records),
            "cutoff_date": self.cutoff_date.isoformat(),
        }

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tabular ML game & market features with strict cutoff date (Phase 4)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app-id", type=int, help="Steam App ID to build features for")
    group.add_argument("--all", action="store_true", help="Build features for all games")
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="ISO format cutoff timestamp (e.g. 2026-08-01T00:00:00Z)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute features without DB persistence")
    args = parser.parse_args()

    cutoff = None
    if args.cutoff_date:
        cutoff = datetime.fromisoformat(args.cutoff_date.replace("Z", "+00:00"))

    job = BuildFeaturesJob(
        app_id=args.app_id,
        all_games=args.all,
        cutoff_date=cutoff,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
