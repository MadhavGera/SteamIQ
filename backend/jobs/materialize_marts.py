"""
Decision Intelligence & Mart Materialization Pipeline — Phase 5.

Populates the decision mart layer:
  - mart_game_overview (unified pre-materialized game metadata, KPIs, executive brief, revenue tier)
  - mart_trends (patch impact and market/genre trend metrics)
  - mart_opportunity_scores (Opportunity Finder weighted scoring formula)

ADR 0001, Decision 1 & 2 (Golden Rule):
  - Reads ONLY from raw_*, feature_*, and serving_* tables.
  - Never writes to raw_*/feature_*/serving_* (writes ONLY to mart_*).
  - API handlers read directly from mart_* with ZERO request-time aggregation.

Usage:
  python -m jobs.materialize_marts --all
  python -m jobs.materialize_marts --app-id 367520
  python -m jobs.materialize_marts --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    FeatureReviewSummary,
    FeatureReviewTopic,
    MartGameOverview,
    MartOpportunityScore,
    MartTrend,
    RawGame,
    RawGameTag,
    RawPlayerSnapshot,
    RawPriceHistory,
    ServingPrediction,
)
from jobs.base_job import BaseJob


class MaterializeMartsJob(BaseJob):
    job_name = "materialize_marts"

    def __init__(
        self,
        *,
        app_id: int | None = None,
        all_games: bool = False,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id
        self.all_games = all_games

    def _extract_primary_genre(self, game: RawGame) -> str:
        """Extract primary genre name from raw_games.genres JSONB."""
        if game.genres and isinstance(game.genres, list) and len(game.genres) > 0:
            first = game.genres[0]
            if isinstance(first, dict) and "description" in first:
                return first["description"]
        return "Indie"

    def _parse_owners_midpoint(self, owners_str: str | None, total_reviews: int) -> int:
        """
        Parse SteamSpy owner string (e.g. '2,000,000 .. 5,000,000') into a numerical estimate.
        Falls back to a standard Boxleiter review multiplier (35x) if unparseable.
        """
        if owners_str and ".." in owners_str:
            parts = owners_str.split("..")
            try:
                low = int(parts[0].replace(",", "").strip())
                high = int(parts[1].replace(",", "").strip())
                return (low + high) // 2
            except (ValueError, IndexError):
                pass
        return max(1000, total_reviews * 35)

    def _calculate_revenue_tier_and_gross(
        self,
        owners_estimate: str | None,
        final_price_usd: Decimal | None,
        price_usd: Decimal | None,
        total_reviews: int,
        is_free: bool,
    ) -> tuple[str, Decimal]:
        """
        Directional revenue estimation (analytics-only, unverified estimate):
        - Inputs: SteamSpy public owner estimate range midpoint (or Boxleiter review multiplier fallback)
                  multiplied by final list price.
        - Strictly non-causal: directional market placement estimate, not verified accounting data.
        """
        if is_free:
            return "Free-to-Play", Decimal("0.00")

        price = float(final_price_usd or price_usd or Decimal("14.99"))
        owners = self._parse_owners_midpoint(owners_estimate, total_reviews)
        gross_rev = round(owners * price, 2)
        gross_rev_dec = Decimal(str(gross_rev))

        if gross_rev >= 50_000_000:
            tier = "Tier 1: $50M+ (Estimated)"
        elif gross_rev >= 10_000_000:
            tier = "Tier 2: $10M–$50M (Estimated)"
        elif gross_rev >= 1_000_000:
            tier = "Tier 3: $1M–$10M (Estimated)"
        elif gross_rev >= 250_000:
            tier = "Tier 4: $250k–$1M (Estimated)"
        else:
            tier = "Tier 5: <$250k (Estimated)"

        return tier, gross_rev_dec

    def _synthesize_executive_brief(
        self,
        game_name: str,
        primary_genre: str,
        net_pos_pct: float,
        review_desc: str | None,
        top_strengths: list[str],
        top_complaints: list[dict[str, Any]],
        top_loved: list[str],
    ) -> dict[str, str]:
        """
        Synthesizes a neutral 2-paragraph diagnostic summary of player reception
        and review themes into mart_game_overview (zero recommendation imperatives).
        """
        strength_str = top_strengths[0] if top_strengths else (top_loved[0] if top_loved else "immersive atmosphere and fluid gameplay")
        p1 = (
            f"Market Position: {game_name} holds a net sentiment rating of {net_pos_pct:.1f}% "
            f"({review_desc or 'Positive'}) in the {primary_genre} category. "
            f"Player reception highlights {strength_str} as the primary positive driver."
        )

        complaint_str = top_complaints[0]["category"] if top_complaints else "early pacing and balance"
        loved_str = top_loved[0] if top_loved else "core gameplay mechanics"
        p2 = (
            f"Feedback Themes: Critical feedback is concentrated around {complaint_str}, "
            f"while player praise centers on {loved_str}."
        )

        return {
            "market_position": p1,
            "feedback_themes": p2,
            "full_brief": f"{p1}\n\n{p2}",
        }

    def _compute_opportunity_score_components(
        self,
        genre_games: list[RawGame],
        genre_complaints: list[FeatureReviewComplaint],
        genre_loved: list[FeatureReviewFeature],
        total_catalog_size: int,
    ) -> dict[str, Any]:
        """
        Opportunity Finder Spec (Analytics-only formula per Roadmap §5):
          Opportunity = 0.35 * Demand + 0.25 * (100 - Saturation) + 0.25 * SentimentGap + 0.15 * Monetization

        1. Demand (35%): Total reviews, positive volume, and player engagement in the niche.
        2. Saturation Headroom (25%): Inverse saturation index (less crowded genres score higher).
        3. Sentiment Gap (25%): Dissatisfaction density in existing titles = opportunity for a better entrant.
        4. Monetization (15%): Price realization headroom relative to Steam benchmarks.
        """
        game_count = len(genre_games)
        if game_count == 0:
            return {
                "opportunity_score": Decimal("50.00"),
                "demand_score": Decimal("50.00"),
                "saturation_score": Decimal("50.00"),
                "sentiment_gap_score": Decimal("50.00"),
                "monetization_score": Decimal("50.00"),
                "median_price_usd": Decimal("14.99"),
                "avg_review_score": Decimal("75.00"),
            }

        total_pos = sum(g.positive_reviews or 0 for g in genre_games)
        total_neg = sum(g.negative_reviews or 0 for g in genre_games)
        total_revs = total_pos + total_neg
        pos_ratio = (total_pos / total_revs * 100.0) if total_revs > 0 else 75.0

        # 1. Demand score: log volume + positive sentiment ratio
        volume_component = min(100.0, (np.log1p(total_revs) / np.log1p(250_000)) * 100.0)
        demand_score = float(np.clip(0.60 * volume_component + 0.40 * pos_ratio, 0.0, 100.0))

        # 2. Saturation score & headroom: density relative to catalog
        saturation_score = float(np.clip(min(100.0, (game_count / 15.0) * 100.0), 0.0, 100.0))
        saturation_headroom = max(0.0, 100.0 - saturation_score)

        # 3. Sentiment gap score: complaint volume in existing games creates market opportunity
        complaint_density = sum(float(c.volume_pct) for c in genre_complaints) / max(1, len(genre_complaints)) if genre_complaints else 15.0
        review_scores = [g.review_score * 10.0 for g in genre_games if g.review_score is not None]
        avg_score = float(np.mean(review_scores)) if review_scores else 75.0
        score_gap = max(0.0, 100.0 - avg_score)
        sentiment_gap_score = float(np.clip(0.50 * min(100.0, complaint_density * 3.5) + 0.50 * score_gap, 0.0, 100.0))

        # 4. Monetization score: price realization potential
        prices = [float(g.final_price_usd) for g in genre_games if g.final_price_usd is not None and not g.is_free]
        median_price = float(np.median(prices)) if prices else 14.99
        monetization_score = float(np.clip((median_price / 30.0) * 100.0, 0.0, 100.0))

        # Composite Opportunity Score
        opportunity = (
            0.35 * demand_score
            + 0.25 * saturation_headroom
            + 0.25 * sentiment_gap_score
            + 0.15 * monetization_score
        )
        opportunity = float(np.clip(opportunity, 0.0, 100.0))

        return {
            "opportunity_score": Decimal(str(round(opportunity, 2))),
            "demand_score": Decimal(str(round(demand_score, 2))),
            "saturation_score": Decimal(str(round(saturation_score, 2))),
            "sentiment_gap_score": Decimal(str(round(sentiment_gap_score, 2))),
            "monetization_score": Decimal(str(round(monetization_score, 2))),
            "median_price_usd": Decimal(str(round(median_price, 2))),
            "avg_review_score": Decimal(str(round(avg_score, 2))),
        }

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
        now = datetime.now(UTC)

        # 1. Fetch catalog
        if self.app_id:
            target_stmt = select(RawGame).where(RawGame.app_id == self.app_id)
        else:
            target_stmt = select(RawGame).order_by(RawGame.app_id)

        target_games = (await session.execute(target_stmt)).scalars().all()
        all_games = (await session.execute(select(RawGame).order_by(RawGame.app_id))).scalars().all()

        if not target_games:
            self.logger.warning("No games found to materialize.")
            return {"overview_rows": 0, "trend_rows": 0, "opportunity_rows": 0}

        self.logger.info("Materializing marts for %d games (Catalog: %d)", len(target_games), len(all_games))

        # 2. Fetch dependencies from feature_*, serving_*, and raw_* tables
        # Sentiments
        sentiments_res = await session.execute(select(FeatureReviewSentiment))
        sentiments_all = sentiments_res.scalars().all()
        sentiments_by_game: dict[int, list[FeatureReviewSentiment]] = {}
        for s in sentiments_all:
            sentiments_by_game.setdefault(s.app_id, []).append(s)

        # Summaries
        summaries_res = await session.execute(select(FeatureReviewSummary))
        summaries_by_game = {s.app_id: s for s in summaries_res.scalars().all()}

        # Loved features
        features_res = await session.execute(select(FeatureReviewFeature))
        features_by_game: dict[int, list[FeatureReviewFeature]] = {}
        for f in features_res.scalars().all():
            features_by_game.setdefault(f.app_id, []).append(f)

        # Complaints
        complaints_res = await session.execute(select(FeatureReviewComplaint))
        complaints_by_game: dict[int, list[FeatureReviewComplaint]] = {}
        for c in complaints_res.scalars().all():
            complaints_by_game.setdefault(c.app_id, []).append(c)

        # Topics
        topics_res = await session.execute(select(FeatureReviewTopic))
        topics_by_game: dict[int, list[FeatureReviewTopic]] = {}
        for t in topics_res.scalars().all():
            topics_by_game.setdefault(t.app_id, []).append(t)

        # Serving predictions (Phase 4 ML models)
        preds_res = await session.execute(
            select(ServingPrediction).where(ServingPrediction.prediction_type == "success_score")
        )
        preds_by_game = {p.app_id: p for p in preds_res.scalars().all()}

        # Player snapshots
        snapshots_res = await session.execute(select(RawPlayerSnapshot))
        snapshots_by_game: dict[int, list[RawPlayerSnapshot]] = {}
        for snap in snapshots_res.scalars().all():
            snapshots_by_game.setdefault(snap.app_id, []).append(snap)

        # Price history
        prices_res = await session.execute(select(RawPriceHistory))
        prices_by_game: dict[int, list[RawPriceHistory]] = {}
        for ph in prices_res.scalars().all():
            prices_by_game.setdefault(ph.app_id, []).append(ph)

        # Tags
        tags_res = await session.execute(select(RawGameTag))
        tags_by_game: dict[int, list[RawGameTag]] = {}
        for tag in tags_res.scalars().all():
            tags_by_game.setdefault(tag.app_id, []).append(tag)

        # Precompute genre price distributions (min, p25, median, p75, max)
        games_by_genre: dict[str, list[RawGame]] = {}
        for g in all_games:
            genre = self._extract_primary_genre(g)
            games_by_genre.setdefault(genre, []).append(g)

        genre_price_stats: dict[str, dict[str, float]] = {}
        for genre, glist in games_by_genre.items():
            g_prices = [float(g.final_price_usd) for g in glist if g.final_price_usd is not None and not g.is_free]
            if g_prices:
                genre_price_stats[genre] = {
                    "min": float(np.min(g_prices)),
                    "p25": float(np.percentile(g_prices, 25)),
                    "median": float(np.median(g_prices)),
                    "p75": float(np.percentile(g_prices, 75)),
                    "max": float(np.max(g_prices)),
                }
            else:
                genre_price_stats[genre] = {
                    "min": 4.99,
                    "p25": 9.99,
                    "median": 14.99,
                    "p75": 19.99,
                    "max": 29.99,
                }

        # ── 3. Materialize mart_game_overview ────────────────────────────────
        overview_records: list[MartGameOverview] = []
        trend_records: list[MartTrend] = []

        for game in target_games:
            primary_genre = self._extract_primary_genre(game)
            g_stats = genre_price_stats.get(
                primary_genre,
                {"min": 4.99, "p25": 9.99, "median": 14.99, "p75": 19.99, "max": 29.99},
            )
            game_sentiments = sentiments_by_game.get(game.app_id, [])
            all_time_sent = next((s for s in game_sentiments if s.month == "ALL_TIME"), None)

            # Net positive %
            if all_time_sent is not None:
                net_pos_pct = float(all_time_sent.net_positive_pct)
            else:
                tot_revs = (game.positive_reviews or 0) + (game.negative_reviews or 0)
                net_pos_pct = round((game.positive_reviews / tot_revs * 100.0), 2) if tot_revs > 0 else 75.0

            # Success score (from ML serving prediction or composite fallback)
            serving_pred = preds_by_game.get(game.app_id)
            if serving_pred is not None:
                success_score = Decimal(str(round(float(serving_pred.score) * 100.0, 2)))
            else:
                # Fallback composite score
                tot = (game.positive_reviews or 0) + (game.negative_reviews or 0)
                vol_score = min(100.0, (np.log1p(tot) / np.log1p(100_000)) * 100.0)
                pos_comp = net_pos_pct
                score_comp = (game.review_score / 9.0 * 100.0) if game.review_score is not None else pos_comp
                composite = 0.45 * vol_score + 0.35 * pos_comp + 0.20 * score_comp
                success_score = Decimal(str(round(float(np.clip(composite, 0.0, 100.0)), 2)))

            # Revenue tier & gross
            tot_reviews = (game.positive_reviews or 0) + (game.negative_reviews or 0)
            revenue_tier, est_gross = self._calculate_revenue_tier_and_gross(
                owners_estimate=game.owners_estimate,
                final_price_usd=game.final_price_usd,
                price_usd=game.price_usd,
                total_reviews=tot_reviews,
                is_free=game.is_free,
            )

            # Peak 24h CCU
            game_snaps = snapshots_by_game.get(game.app_id, [])
            if game_snaps:
                peak_ccu = max(s.peak_24h or s.player_count for s in game_snaps)
            else:
                peak_ccu = max(100, int(game.average_playtime_forever * 1.5)) if game.average_playtime_forever else 500

            # Top strengths & complaints
            game_summary = summaries_by_game.get(game.app_id)
            top_strengths = game_summary.core_strengths if game_summary and game_summary.core_strengths else []
            if not top_strengths:
                top_strengths = [f.feature_name for f in features_by_game.get(game.app_id, [])[:3]]

            game_comps = complaints_by_game.get(game.app_id, [])
            top_complaints = [
                {
                    "category": c.category,
                    "volume_pct": float(c.volume_pct),
                    "severity": c.severity,
                }
                for c in game_comps[:3]
            ]

            game_loved = [f.feature_name for f in features_by_game.get(game.app_id, [])[:3]]

            # Executive brief
            exec_brief = self._synthesize_executive_brief(
                game_name=game.name,
                primary_genre=primary_genre,
                net_pos_pct=net_pos_pct,
                review_desc=game.review_score_desc,
                top_strengths=top_strengths,
                top_complaints=top_complaints,
                top_loved=game_loved,
            )

            # Pricing Intelligence (comparable-range output, historical low from real snapshots, no causal claims)
            game_price_history = sorted(
                prices_by_game.get(game.app_id, []),
                key=lambda x: x.recorded_at or datetime.min.replace(tzinfo=UTC),
            )
            price_tracking_started_at = (
                game_price_history[0].recorded_at if (game_price_history and game_price_history[0].recorded_at) else now
            )
            if game_price_history:
                valid_finals = [float(p.final_price_usd) for p in game_price_history if p.final_price_usd is not None]
                hist_low_price = Decimal(str(round(min(valid_finals), 2))) if valid_finals else (game.final_price_usd or (Decimal("0.00") if game.is_free else Decimal("14.99")))
                hist_low_discount = max((p.discount_pct for p in game_price_history if p.discount_pct is not None), default=(game.discount_pct or 0))
            else:
                hist_low_price = game.final_price_usd or (Decimal("0.00") if game.is_free else Decimal("14.99"))
                hist_low_discount = game.discount_pct or 0

            game_p = float(game.final_price_usd or game.price_usd or Decimal("14.99")) if not game.is_free else 0.0
            g_med = g_stats["median"]
            price_vs_med_pct = round(((game_p - g_med) / g_med) * 100.0, 1) if g_med > 0 and not game.is_free else 0.0

            if game.is_free:
                position_bracket = "Free-to-Play"
            elif game_p < g_stats["p25"]:
                position_bracket = "Budget / Entry-Tier"
            elif game_p <= g_stats["p75"]:
                position_bracket = "Benchmark / Mid-Tier"
            else:
                position_bracket = "Premium Tier"

            pricing_spectrum = {
                "primary_genre": primary_genre,
                "game_price_usd": game_p,
                "genre_min_price_usd": g_stats["min"],
                "genre_p25_price_usd": g_stats["p25"],
                "genre_median_price_usd": g_stats["median"],
                "genre_p75_price_usd": g_stats["p75"],
                "genre_max_price_usd": g_stats["max"],
                "price_vs_median_pct": price_vs_med_pct,
                "position_bracket": position_bracket,
                "historical_lowest_price_usd": float(hist_low_price),
                "historical_lowest_discount_pct": hist_low_discount,
                "price_tracking_started_at": price_tracking_started_at.isoformat() if price_tracking_started_at else None,
                "is_free": game.is_free,
            }

            price_history_points = [
                {
                    "recorded_at": p.recorded_at.isoformat() if p.recorded_at else None,
                    "price_usd": float(p.price_usd) if p.price_usd is not None else None,
                    "final_price_usd": float(p.final_price_usd) if p.final_price_usd is not None else None,
                    "discount_pct": p.discount_pct or 0,
                }
                for p in game_price_history
            ]

            overview = MartGameOverview(
                app_id=game.app_id,
                name=game.name,
                short_description=game.short_description,
                description=game.description,
                developer=game.developer,
                publisher=game.publisher,
                release_date=game.release_date,
                coming_soon=game.coming_soon,
                genres=game.genres,
                categories=game.categories,
                tags=game.tags,
                is_free=game.is_free,
                price_usd=game.price_usd,
                final_price_usd=game.final_price_usd,
                discount_pct=game.discount_pct,
                platform_windows=game.platform_windows,
                platform_mac=game.platform_mac,
                platform_linux=game.platform_linux,
                positive_reviews=game.positive_reviews,
                negative_reviews=game.negative_reviews,
                review_score=game.review_score,
                review_score_desc=game.review_score_desc,
                owners_estimate=game.owners_estimate,
                average_playtime_forever=game.average_playtime_forever,
                median_playtime_forever=game.median_playtime_forever,
                metacritic_score=game.metacritic_score,
                header_image=game.header_image,
                website=game.website,
                primary_genre=primary_genre,
                revenue_tier=revenue_tier,
                estimated_gross_revenue_usd=est_gross,
                success_score=success_score,
                net_sentiment_pct=Decimal(str(round(net_pos_pct, 2))),
                peak_ccu_24h=peak_ccu,
                executive_brief=exec_brief,
                top_strengths=top_strengths,
                top_complaints=top_complaints,
                price_tracking_started_at=price_tracking_started_at,
                historical_lowest_price_usd=hist_low_price,
                historical_lowest_discount_pct=hist_low_discount,
                genre_median_price_usd=Decimal(str(round(g_stats["median"], 2))),
                genre_min_price_usd=Decimal(str(round(g_stats["min"], 2))),
                genre_max_price_usd=Decimal(str(round(g_stats["max"], 2))),
                genre_p25_price_usd=Decimal(str(round(g_stats["p25"], 2))),
                genre_p75_price_usd=Decimal(str(round(g_stats["p75"], 2))),
                pricing_spectrum=pricing_spectrum,
                price_history_points=price_history_points,
                materialized_at=now,
            )
            overview_records.append(overview)

            # ── Materialize Trends for this game ──
            # Monthly sentiment timeline trend
            monthly_sents = [s for s in game_sentiments if s.month != "ALL_TIME"]
            for s in monthly_sents:
                trend_records.append(
                    MartTrend(
                        app_id=game.app_id,
                        trend_type="sentiment_trajectory",
                        category=primary_genre,
                        recorded_date=now,
                        metric_name=f"net_positive_pct_{s.month}",
                        metric_value=Decimal(str(s.net_positive_pct)),
                        change_pct_7d=None,
                        change_pct_30d=None,
                        metadata_payload={
                            "month": s.month,
                            "positive_count": s.positive_count,
                            "negative_count": s.negative_count,
                            "total_count": s.total_count,
                        },
                        materialized_at=now,
                    )
                )

            # Patch impact trend (Update Impact Tracker)
            if len(monthly_sents) >= 2:
                sorted_s = sorted(monthly_sents, key=lambda x: x.month)
                latest_m = sorted_s[-1]
                prev_m = sorted_s[-2]
                delta = float(latest_m.net_positive_pct) - float(prev_m.net_positive_pct)
                verdict = "Positive Reception" if delta >= 0 else ("Mixed Impact" if delta >= -5.0 else "Player Backlash")

                trend_records.append(
                    MartTrend(
                        app_id=game.app_id,
                        trend_type="patch_impact",
                        category=primary_genre,
                        recorded_date=now,
                        metric_name="sentiment_delta_latest_update",
                        metric_value=Decimal(str(round(delta, 2))),
                        change_pct_7d=Decimal(str(round(delta, 2))),
                        change_pct_30d=Decimal(str(round(delta, 2))),
                        metadata_payload={
                            "verdict": verdict,
                            "pre_sentiment_pct": float(prev_m.net_positive_pct),
                            "post_sentiment_pct": float(latest_m.net_positive_pct),
                            "latest_month": latest_m.month,
                            "prior_month": prev_m.month,
                        },
                        materialized_at=now,
                    )
                )

        # ── 4. Materialize Genre-Wide Trends ─────────────────────────────────
        for genre, glist in games_by_genre.items():
            g_prices = [float(g.final_price_usd) for g in glist if g.final_price_usd is not None and not g.is_free]
            med_p = float(np.median(g_prices)) if g_prices else 14.99
            trend_records.append(
                MartTrend(
                    app_id=None,
                    trend_type="genre_growth",
                    category=genre,
                    recorded_date=now,
                    metric_name="catalog_density",
                    metric_value=Decimal(str(len(glist))),
                    change_pct_7d=None,
                    change_pct_30d=None,
                    metadata_payload={
                        "game_count": len(glist),
                        "median_price_usd": med_p,
                    },
                    materialized_at=now,
                )
            )

        # ── 5. Materialize mart_opportunity_scores (Opportunity Finder) ───────
        opportunity_records: list[MartOpportunityScore] = []

        # By Genre
        for genre, glist in games_by_genre.items():
            g_app_ids = {g.app_id for g in glist}
            genre_complaints = [c for aid in g_app_ids for c in complaints_by_game.get(aid, [])]
            genre_loved = [f for aid in g_app_ids for f in features_by_game.get(aid, [])]

            comps = self._compute_opportunity_score_components(
                genre_games=glist,
                genre_complaints=genre_complaints,
                genre_loved=genre_loved,
                total_catalog_size=len(all_games),
            )

            # Top complaint themes in genre
            comp_counts: dict[str, float] = {}
            for c in genre_complaints:
                comp_counts[c.category] = comp_counts.get(c.category, 0.0) + float(c.volume_pct)
            top_comp_themes = sorted(comp_counts.keys(), key=lambda k: -comp_counts[k])[:5]

            # Top loved themes in genre
            loved_counts: dict[str, int] = {}
            for f in genre_loved:
                loved_counts[f.feature_name] = loved_counts.get(f.feature_name, 0) + f.mention_count
            top_loved_themes = sorted(loved_counts.keys(), key=lambda k: -loved_counts[k])[:5]

            # Recommended opportunity takeaways
            rec_features = [
                f"Capitalize on high demand in {genre} by solving '{top_comp_themes[0]}'" if top_comp_themes else f"Deliver core gameplay innovation in {genre}",
                f"Incorporate '{top_loved_themes[0]}' as a must-have pillar" if top_loved_themes else "Build deep mechanical replayability",
                f"Target sweet-spot price point near ${comps['median_price_usd']}",
            ]

            opp = MartOpportunityScore(
                genre_or_tag=genre,
                entity_type="genre",
                opportunity_score=comps["opportunity_score"],
                demand_score=comps["demand_score"],
                saturation_score=comps["saturation_score"],
                sentiment_gap_score=comps["sentiment_gap_score"],
                monetization_score=comps["monetization_score"],
                game_count=len(glist),
                median_price_usd=comps["median_price_usd"],
                avg_review_score=comps["avg_review_score"],
                top_complaint_themes=top_comp_themes,
                top_loved_themes=top_loved_themes,
                recommended_features=rec_features,
                materialized_at=now,
            )
            opportunity_records.append(opp)

        # By Top Tags
        all_tag_names: dict[str, list[RawGame]] = {}
        for g in all_games:
            for t in tags_by_game.get(g.app_id, []):
                all_tag_names.setdefault(t.tag_name, []).append(g)

        # Process top 10 most frequent tags
        for tag_name, t_games in sorted(all_tag_names.items(), key=lambda x: -len(x[1]))[:15]:
            t_app_ids = {g.app_id for g in t_games}
            tag_comps = [c for aid in t_app_ids for c in complaints_by_game.get(aid, [])]
            tag_loved = [f for aid in t_app_ids for f in features_by_game.get(aid, [])]

            comps = self._compute_opportunity_score_components(
                genre_games=t_games,
                genre_complaints=tag_comps,
                genre_loved=tag_loved,
                total_catalog_size=len(all_games),
            )

            tag_comp_counts: dict[str, float] = {}
            for c in tag_comps:
                tag_comp_counts[c.category] = tag_comp_counts.get(c.category, 0.0) + float(c.volume_pct)
            top_t_comps = sorted(tag_comp_counts.keys(), key=lambda k: -tag_comp_counts[k])[:5]

            tag_loved_counts: dict[str, int] = {}
            for f in tag_loved:
                tag_loved_counts[f.feature_name] = tag_loved_counts.get(f.feature_name, 0) + f.mention_count
            top_t_loved = sorted(tag_loved_counts.keys(), key=lambda k: -tag_loved_counts[k])[:5]

            rec_features = [
                f"Strong tag appeal for '{tag_name}' with {len(t_games)} catalog peers",
                f"Solve common complaint '{top_t_comps[0]}'" if top_t_comps else "Ensure polished launch quality",
                f"Anchor pricing near ${comps['median_price_usd']}",
            ]

            opp = MartOpportunityScore(
                genre_or_tag=tag_name,
                entity_type="tag",
                opportunity_score=comps["opportunity_score"],
                demand_score=comps["demand_score"],
                saturation_score=comps["saturation_score"],
                sentiment_gap_score=comps["sentiment_gap_score"],
                monetization_score=comps["monetization_score"],
                game_count=len(t_games),
                median_price_usd=comps["median_price_usd"],
                avg_review_score=comps["avg_review_score"],
                top_complaint_themes=top_t_comps,
                top_loved_themes=top_t_loved,
                recommended_features=rec_features,
                materialized_at=now,
            )
            opportunity_records.append(opp)

        # ── 6. Persist to DB ──────────────────────────────────────────────────
        if not self.dry_run:
            target_ids = [g.app_id for g in target_games]
            await session.execute(
                delete(MartGameOverview).where(MartGameOverview.app_id.in_(target_ids))
            )
            await session.execute(
                delete(MartTrend).where(
                    (MartTrend.app_id.in_(target_ids)) | (MartTrend.app_id.is_(None))
                )
            )
            await session.execute(delete(MartOpportunityScore))

            for ov in overview_records:
                session.add(ov)
            for tr in trend_records:
                session.add(tr)
            for opp in opportunity_records:
                session.add(opp)

            await session.commit()
            self.logger.info(
                "Persisted %d mart_game_overview, %d mart_trends, and %d mart_opportunity_scores rows",
                len(overview_records),
                len(trend_records),
                len(opportunity_records),
            )

        return {
            "overview_rows": len(overview_records),
            "trend_rows": len(trend_records),
            "opportunity_rows": len(opportunity_records),
        }

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize Phase 5 decision intelligence mart tables (ADR 0001)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--app-id", type=int, help="Steam App ID to materialize overview for")
    group.add_argument("--all", action="store_true", help="Materialize all mart tables across entire catalog")
    parser.add_argument("--dry-run", action="store_true", help="Compute marts without DB persistence")
    args = parser.parse_args()

    job = MaterializeMartsJob(
        app_id=args.app_id,
        all_games=args.all,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
