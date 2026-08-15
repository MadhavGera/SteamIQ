"""
Recommendation Engine Job — Phase 5.

Synthesizes prioritized, explainable recommendations using a hybrid system
combining predictive model SHAP attributions (carrying model_run_id for traceability)
and domain-specific heuristic rules.

Data Sources (Read-Only):
  - serving_predictions (champion ML predictions, SHAP feature attributions, model_run_id)
  - feature_review_complaints (categorized pain points, severity, snippets)
  - feature_review_features (loved features, praise intensity)
  - feature_review_sentiment (net sentiment)
  - raw_games / mart_game_overview (pricing, discounts, genre benchmarks)

Target Table:
  - serving_recommendations (read by api/recommendations.py)

ADR 0001, Decision 1 & 2 (Golden Rule):
  - Job writes to serving_recommendations.
  - API handlers read directly from serving_recommendations with zero live computation.

Usage:
  python -m jobs.generate_recommendations --all
  python -m jobs.generate_recommendations --app-id 1145360
  python -m jobs.generate_recommendations --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    MartGameOverview,
    RawGame,
    ServingPrediction,
    ServingRecommendation,
)
from jobs.base_job import BaseJob


class GenerateRecommendationsJob(BaseJob):
    """
    Hybrid Recommendation Engine generating prioritized, explainable recommendations.
    """
    job_name = "generate_recommendations"

    def __init__(
        self,
        *,
        app_id: int | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.app_id = app_id

    def _extract_primary_genre(self, game: RawGame | MartGameOverview) -> str:
        """Extract primary genre name from JSON genres."""
        if hasattr(game, "genres") and game.genres and isinstance(game.genres, list) and len(game.genres) > 0:
            first = game.genres[0]
            if isinstance(first, dict):
                return first.get("description", "Indie")
            return str(first)
        return "Indie"

    def _build_game_recommendations(
        self,
        game: RawGame | MartGameOverview,
        prediction: ServingPrediction | None,
        complaints: list[FeatureReviewComplaint],
        loved_features: list[FeatureReviewFeature],
        sentiment: FeatureReviewSentiment | None,
        genre_median_price: float,
    ) -> list[dict[str, Any]]:
        """
        Generate candidate recommendations combining model SHAP attributions and domain heuristics.
        """
        candidates: list[dict[str, Any]] = []
        model_run_id = prediction.model_run_id if prediction else None
        shap_values: dict[str, float] = (prediction.shap_values or {}) if prediction else {}

        price = float(game.final_price_usd or game.price_usd or Decimal("14.99")) if not game.is_free else 0.0
        primary_genre = self._extract_primary_genre(game)
        net_pos = float(sentiment.net_positive_pct) if sentiment else 75.0

        # ── 1. Model-Driven Recommendations (SHAP feature attribution) ────────
        # Check negative drag factor on complaint density
        complaint_shap = shap_values.get("complaint_density", 0.0)
        if complaint_shap < -0.02 and complaints:
            top_comp = complaints[0]
            # Empirical confidence derived from SHAP attribution strength and complaint volume
            conf_val = min(0.96, max(0.65, 0.70 + abs(complaint_shap) * 1.0 + (float(top_comp.volume_pct) / 100.0) * 0.15))
            candidates.append({
                "model_run_id": model_run_id,
                "recommendation_type": "quality_engineering",
                "domain": "Engineering & Quality",
                "impact_level": "High",
                "difficulty_level": "Medium" if top_comp.severity != "high" else "High",
                "confidence_score": Decimal(str(round(conf_val, 2))),
                "title": f"Triage {top_comp.category} to Reclaim Sentiment Velocity",
                "rationale": (
                    f"Predictive model attribution flags complaint density as a primary success drag "
                    f"(SHAP impact {complaint_shap:+.2f}). Review analysis reveals {top_comp.volume_pct:.1f}% "
                    f"of critical feedback is concentrated in {top_comp.category}."
                ),
                "evidence_type": "hybrid",  # Model identifies macro feature drag; NLP isolates root cause category
                "evidence_payload": {
                    "shap_feature": "complaint_density",
                    "shap_value": round(complaint_shap, 4),
                    "complaint_density_total": float(sum(float(c.volume_pct) for c in complaints)),
                    "contributing_category": top_comp.category,
                    "contributing_volume_pct": float(top_comp.volume_pct),
                    "severity": top_comp.severity,
                },
                "action_items": [
                    f"Deploy targeted hotfix addressing player pain points in {top_comp.category}.",
                    "Publish a developer update highlighting resolution in patch notes.",
                    "Monitor 14-day post-update sentiment delta to measure recovery.",
                ],
                "score_weight": 95,
            })
        elif complaints and (complaints[0].severity == "high" or complaints[0].volume_pct >= 25.0):
            # Review NLP heuristic without direct model attribution
            top_comp = complaints[0]
            conf_val = min(0.90, max(0.60, 0.60 + (float(top_comp.volume_pct) / 100.0) * 0.35))
            candidates.append({
                "model_run_id": None,
                "recommendation_type": "quality_engineering",
                "domain": "Engineering & Quality",
                "impact_level": "High" if top_comp.severity == "high" else "Medium",
                "difficulty_level": "Medium",
                "confidence_score": Decimal(str(round(conf_val, 2))),
                "title": f"Resolve Critical Feedback in {top_comp.category}",
                "rationale": (
                    f"Review NLP analysis isolates {top_comp.volume_pct:.1f}% of negative reviews focused on "
                    f"{top_comp.category} (severity: {top_comp.severity})."
                ),
                "evidence_type": "review_nlp",
                "evidence_payload": {
                    "complaint_category": top_comp.category,
                    "complaint_volume_pct": float(top_comp.volume_pct),
                    "severity": top_comp.severity,
                },
                "action_items": [
                    f"Investigate high-frequency issue reports in {top_comp.category}.",
                    "Release patch addressing top reported edge cases.",
                ],
                "score_weight": 90,
            })

        # Check positive driver on loved features
        loved_shap = shap_values.get("loved_feature_density", 0.0)
        if loved_features and (loved_shap > 0.02 or net_pos >= 85.0):
            top_loved = loved_features[0]
            conf_val = min(0.95, max(0.65, 0.65 + (top_loved.praise_intensity / 100.0) * 0.20 + min(0.10, top_loved.mention_count / 1000.0)))
            candidates.append({
                "model_run_id": model_run_id if loved_shap > 0.02 else None,
                "recommendation_type": "content_expansion",
                "domain": "Content & Gameplay",
                "impact_level": "High",
                "difficulty_level": "Medium",
                "confidence_score": Decimal(str(round(conf_val, 2))),
                "title": f"Capitalize on Acclaimed {top_loved.feature_name}",
                "rationale": (
                    f"Player enthusiasm strongly correlates with {top_loved.feature_name} "
                    f"({top_loved.mention_count} positive mentions, {top_loved.praise_intensity}/100 praise intensity). "
                    f"Expanding this core pillar provides maximum leverage for engagement retention."
                ),
                "evidence_type": "hybrid" if (model_run_id and loved_shap > 0.02) else "review_nlp",
                "evidence_payload": {
                    "loved_feature": top_loved.feature_name,
                    "mention_count": top_loved.mention_count,
                    "praise_intensity": top_loved.praise_intensity,
                    "shap_driver": round(loved_shap, 4) if (model_run_id and loved_shap > 0.02) else None,
                },
                "action_items": [
                    f"Highlight {top_loved.feature_name} prominently in store capsule art and trailer cut.",
                    f"Develop content DLC or roadmap expansion centered on {top_loved.feature_name}.",
                ],
                "score_weight": 85,
            })

        # ── 2. Heuristic Domain Rules ─────────────────────────────────────────
        # Pricing & Monetization Rule
        if not game.is_free and genre_median_price > 0:
            if price < (genre_median_price * 0.70) and net_pos >= 80.0:
                price_gap_ratio = (genre_median_price - price) / genre_median_price
                conf_val = min(0.92, max(0.65, 0.65 + price_gap_ratio * 0.35))
                candidates.append({
                    "model_run_id": None,  # Rules-only
                    "recommendation_type": "pricing",
                    "domain": "Pricing & Monetization",
                    "impact_level": "High",
                    "difficulty_level": "Low",
                    "confidence_score": Decimal(str(round(conf_val, 2))),
                    "title": "Pricing Alignment with Genre Benchmark",
                    "rationale": (
                        f"Current list price (${price:.2f}) sits significantly below the {primary_genre} "
                        f"median benchmark (${genre_median_price:.2f}) despite strong player sentiment ({net_pos:.1f}%). "
                        f"A moderate price realignment or premium edition bundle can expand gross revenue."
                    ),
                    "evidence_type": "pricing_comparable",
                    "evidence_payload": {
                        "current_price_usd": price,
                        "genre_median_price_usd": genre_median_price,
                        "discount_pct": game.discount_pct,
                        "net_positive_pct": net_pos,
                        "price_gap_pct": round(price_gap_ratio * 100.0, 1),
                    },
                    "action_items": [
                        f"Evaluate adjusting base price towards ${genre_median_price:.2f} ahead of major content release.",
                        "Introduce a Soundtrack / Deluxe Bundle with a 15% bundle discount.",
                    ],
                    "score_weight": 90,
                })
            elif game.discount_pct == 0:
                candidates.append({
                    "model_run_id": None,
                    "recommendation_type": "marketing_discovery",
                    "domain": "Marketing & Discovery",
                    "impact_level": "Medium",
                    "difficulty_level": "Low",
                    "confidence_score": Decimal("0.78"),
                    "title": "Schedule Seasonal Steam Festival Discount",
                    "rationale": (
                        "Game has no active discount. Scheduling a 15–25% promotional discount during upcoming "
                        "Steam seasonal sales triggers wishlist notification emails to re-accelerate conversion."
                    ),
                    "evidence_type": "pricing_comparable",
                    "evidence_payload": {
                        "current_discount_pct": 0,
                        "primary_genre": primary_genre,
                    },
                    "action_items": [
                        "Register for upcoming Steam Seasonal Sale with a 20% promotional discount.",
                        "Prepare community announcement and streaming broadcast event for sale launch.",
                    ],
                    "score_weight": 75,
                })

        # Community & Retention Rule
        avg_playtime = getattr(game, "average_playtime_forever", 0) or 0
        if avg_playtime > 0 and avg_playtime < 180 and net_pos >= 75.0:
            conf_val = min(0.88, max(0.60, 0.60 + (1.0 - (avg_playtime / 300.0)) * 0.28))
            candidates.append({
                "model_run_id": None,
                "recommendation_type": "retention",
                "domain": "Community & Retention",
                "impact_level": "Medium",
                "difficulty_level": "Medium",
                "confidence_score": Decimal(str(round(conf_val, 2))),
                "title": "Introduce Replayability Loops & Steam Achievements",
                "rationale": (
                    f"Average playtime ({avg_playtime} mins) reflects rapid drop-off despite positive sentiment. "
                    f"Adding unlockable challenges, Steam achievements, or endless game modes will lengthen play sessions."
                ),
                "evidence_type": "retention_telemetry",
                "evidence_payload": {
                    "average_playtime_forever": avg_playtime,
                    "net_positive_pct": net_pos,
                },
                "action_items": [
                    "Design tiered Steam Achievements tied to alternate playstyles.",
                    "Implement a Daily Challenge or Endless Mode with community leaderboards.",
                ],
                "score_weight": 70,
            })

        # Fallback general recommendation if catalog game had few items
        if not candidates:
            candidates.append({
                "model_run_id": model_run_id,
                "recommendation_type": "community",
                "domain": "Marketing & Discovery",
                "impact_level": "Medium",
                "difficulty_level": "Low",
                "confidence_score": Decimal("0.75"),
                "title": "Engage Community Feedback & Store Page Localization",
                "rationale": (
                    f"Expanding language support for top non-English Steam markets (Simplified Chinese, German, Japanese) "
                    f"and running monthly developer Q&A events increases organic global discovery in {primary_genre}."
                ),
                "evidence_type": "review_nlp",
                "evidence_payload": {"primary_genre": primary_genre},
                "action_items": [
                    "Review top geographic wishlist markets and localize store page and subtitles.",
                    "Establish a regular bi-weekly devlog rhythm on Steam Community Hub.",
                ],
                "score_weight": 60,
            })

        # Sort candidates by score_weight descending and assign priority_rank
        candidates.sort(key=lambda c: -c["score_weight"])
        for rank, cand in enumerate(candidates, start=1):
            cand["priority_rank"] = rank

        return candidates

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        """Async execution pipeline populating serving_recommendations."""
        close_session = False
        if session is None:
            engine = create_async_engine(settings.async_database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            session = session_factory()
            close_session = True

        try:
            # 1. Fetch Target Games
            if self.app_id:
                games_stmt = select(RawGame).where(RawGame.app_id == self.app_id)
            else:
                games_stmt = select(RawGame)
            games_res = await session.execute(games_stmt)
            target_games = games_res.scalars().all()

            if not target_games:
                self.logger.warning("No games found to generate recommendations for.")
                return {"recommendation_rows": 0, "games_processed": 0}

            # Fetch all games to compute genre benchmark medians
            all_games_res = await session.execute(select(RawGame))
            all_games = all_games_res.scalars().all()

            genre_medians: dict[str, float] = {}
            games_by_genre: dict[str, list[RawGame]] = {}
            for g in all_games:
                genre = self._extract_primary_genre(g)
                games_by_genre.setdefault(genre, []).append(g)

            for genre, glist in games_by_genre.items():
                prices = [float(g.final_price_usd) for g in glist if g.final_price_usd is not None and not g.is_free]
                genre_medians[genre] = float(Decimal(str(round(sum(prices) / len(prices), 2)))) if prices else 14.99

            # 2. Fetch Review Features, Complaints, and Sentiments
            comp_res = await session.execute(select(FeatureReviewComplaint))
            comps_by_game: dict[int, list[FeatureReviewComplaint]] = {}
            for c in comp_res.scalars().all():
                comps_by_game.setdefault(c.app_id, []).append(c)

            feat_res = await session.execute(select(FeatureReviewFeature))
            feats_by_game: dict[int, list[FeatureReviewFeature]] = {}
            for f in feat_res.scalars().all():
                feats_by_game.setdefault(f.app_id, []).append(f)

            sent_res = await session.execute(
                select(FeatureReviewSentiment).where(FeatureReviewSentiment.month == "ALL_TIME")
            )
            sent_by_game: dict[int, FeatureReviewSentiment] = {
                s.app_id: s for s in sent_res.scalars().all()
            }

            # 3. Fetch Champion Predictions from serving_predictions
            preds_res = await session.execute(
                select(ServingPrediction).where(ServingPrediction.prediction_type == "success_score")
            )
            preds_by_game: dict[int, ServingPrediction] = {
                p.app_id: p for p in preds_res.scalars().all()
            }

            # 4. Generate recommendations
            rec_objects: list[ServingRecommendation] = []
            now = datetime.now(UTC)

            for game in target_games:
                genre = self._extract_primary_genre(game)
                g_median = genre_medians.get(genre, 14.99)
                game_pred = preds_by_game.get(game.app_id)
                game_comps = comps_by_game.get(game.app_id, [])
                game_feats = feats_by_game.get(game.app_id, [])
                game_sent = sent_by_game.get(game.app_id)

                candidate_dicts = self._build_game_recommendations(
                    game=game,
                    prediction=game_pred,
                    complaints=game_comps,
                    loved_features=game_feats,
                    sentiment=game_sent,
                    genre_median_price=g_median,
                )

                for c in candidate_dicts:
                    rec_objects.append(
                        ServingRecommendation(
                            app_id=game.app_id,
                            model_run_id=c.get("model_run_id"),
                            recommendation_type=c["recommendation_type"],
                            domain=c["domain"],
                            priority_rank=c["priority_rank"],
                            title=c["title"],
                            impact_level=c["impact_level"],
                            difficulty_level=c["difficulty_level"],
                            confidence_score=c["confidence_score"],
                            rationale=c["rationale"],
                            evidence_type=c["evidence_type"],
                            evidence_payload=c.get("evidence_payload"),
                            action_items=c.get("action_items"),
                            generated_at=now,
                        )
                    )

            if not self.dry_run:
                target_ids = [g.app_id for g in target_games]
                await session.execute(
                    delete(ServingRecommendation).where(ServingRecommendation.app_id.in_(target_ids))
                )
                session.add_all(rec_objects)
                await session.commit()
                self.logger.info(
                    "Persisted %d recommendations for %d games.",
                    len(rec_objects),
                    len(target_games),
                )
            else:
                self.logger.info(
                    "[DRY RUN] Generated %d recommendations for %d games.",
                    len(rec_objects),
                    len(target_games),
                )

            return {
                "recommendation_rows": len(rec_objects),
                "games_processed": len(target_games),
            }

        finally:
            if close_session:
                await session.close()
                await engine.dispose()

    def run(self) -> dict[str, Any]:
        """Synchronous entry point."""
        return asyncio.run(self._run_async())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hybrid recommendations for Steam games")
    parser.add_argument("--app-id", type=int, default=None, help="Target specific app_id")
    parser.add_argument("--all", action="store_true", help="Generate recommendations for all catalog games")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without writing to DB")
    args = parser.parse_args()

    job = GenerateRecommendationsJob(app_id=args.app_id, dry_run=args.dry_run)
    job.execute()


if __name__ == "__main__":
    main()
