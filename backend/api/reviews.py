"""
Review Intelligence API router — Phase 2.

Endpoints:
  GET /api/v1/games/{app_id}/reviews
  GET /api/v1/games/{app_id}/reviews/sentiment
  GET /api/v1/games/{app_id}/reviews/topics
  GET /api/v1/games/{app_id}/reviews/complaints
  GET /api/v1/games/{app_id}/reviews/features
  GET /api/v1/games/{app_id}/reviews/summary

ADR 0001, Decision 2 (Golden Rule):
  These handlers read from precomputed feature_* tables ONLY.
  ZERO live NLP inference or heavy aggregations in the request path.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.reviews import (
    ComplaintCategorySchema,
    LovedFeatureSchema,
    MonthlySentimentSchema,
    ReviewIntelligenceBundleSchema,
    ReviewSummarySchema,
    ReviewTopicSchema,
    SentimentOverviewSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse
from db.base import get_db
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    FeatureReviewSummary,
    FeatureReviewTopic,
    RawGame,
)

router = APIRouter(prefix="/games/{app_id}/reviews", tags=["reviews"])


# ─── Full Review Intelligence Bundle ───────────────────────────────────────────

@router.get(
    "",
    response_model=ApiResponse[ReviewIntelligenceBundleSchema],
    summary="Get complete Review Intelligence bundle",
    description="Returns precomputed sentiment, timeline, topics, complaints, loved features, and summary.",
)
async def get_review_intelligence(
    app_id: int = Path(..., description="Steam App ID"),
    db: AsyncSession = Depends(get_db),
) -> ApiResponse[ReviewIntelligenceBundleSchema]:
    start = time.perf_counter()

    # 1. Verify game exists
    game = await db.get(RawGame, app_id)
    if game is None:
        raise GameNotFoundError(app_id=app_id)

    # 2. Query precomputed feature tables
    # Sentiment (Overall & Monthly)
    sentiment_stmt = select(FeatureReviewSentiment).where(
        FeatureReviewSentiment.app_id == app_id
    ).order_by(FeatureReviewSentiment.month.asc())
    sentiment_rows = (await db.execute(sentiment_stmt)).scalars().all()

    overall_row = next((r for r in sentiment_rows if r.month == "ALL_TIME"), None)
    monthly_rows = [r for r in sentiment_rows if r.month != "ALL_TIME"]

    # If no feature_review_sentiment exists yet, provide baseline from raw_games
    if overall_row is not None:
        pos_pct = float(overall_row.net_positive_pct)
        pos_cnt = overall_row.positive_count
        neg_cnt = overall_row.negative_count
        tot_cnt = overall_row.total_count
        sentiment_label = game.review_score_desc or "Positive" if pos_pct >= 70 else "Mixed"
    else:
        tot_cnt = game.positive_reviews + game.negative_reviews
        pos_pct = round(game.positive_reviews / tot_cnt * 100, 1) if tot_cnt > 0 else 0.0
        pos_cnt = game.positive_reviews
        neg_cnt = game.negative_reviews
        sentiment_label = game.review_score_desc or "Unknown"

    sentiment_overview = SentimentOverviewSchema(
        positive_pct=pos_pct,
        mixed_pct=max(0.0, round(100.0 - pos_pct - (neg_cnt / tot_cnt * 100 if tot_cnt else 0), 1)),
        negative_pct=round(neg_cnt / tot_cnt * 100, 1) if tot_cnt > 0 else 0.0,
        positive_count=pos_cnt,
        negative_count=neg_cnt,
        total_count=tot_cnt,
        sentiment_label=sentiment_label,
    )

    timeline = [
        MonthlySentimentSchema(
            month=r.month,
            positive_reviews=r.positive_count,
            negative_reviews=r.negative_count,
            net_positive_pct=float(r.net_positive_pct),
        )
        for r in monthly_rows
    ]

    # Topics
    topics_stmt = select(FeatureReviewTopic).where(
        FeatureReviewTopic.app_id == app_id
    ).order_by(FeatureReviewTopic.review_count.desc())
    topic_rows = (await db.execute(topics_stmt)).scalars().all()
    topics = [
        ReviewTopicSchema(
            topic_id=t.topic_id,
            label=t.topic_label,
            review_count=t.review_count,
            sentiment_score=float(t.sentiment_score),
            keywords=t.keywords if isinstance(t.keywords, list) else None,
        )
        for t in topic_rows
    ]

    # Loved Features
    features_stmt = select(FeatureReviewFeature).where(
        FeatureReviewFeature.app_id == app_id
    ).order_by(FeatureReviewFeature.mention_count.desc())
    feature_rows = (await db.execute(features_stmt)).scalars().all()
    loved_features = [
        LovedFeatureSchema(
            feature_name=f.feature_name,
            mention_count=f.mention_count,
            praise_intensity=f.praise_intensity,
        )
        for f in feature_rows
    ]

    # Complaints
    complaints_stmt = select(FeatureReviewComplaint).where(
        FeatureReviewComplaint.app_id == app_id
    ).order_by(FeatureReviewComplaint.volume_pct.desc())
    complaint_rows = (await db.execute(complaints_stmt)).scalars().all()
    complaints = [
        ComplaintCategorySchema(
            category=c.category,
            volume_pct=float(c.volume_pct),
            severity=c.severity,
            representative_snippets=c.representative_snippets if isinstance(c.representative_snippets, list) else [],
        )
        for c in complaint_rows
    ]

    # Summary
    summary_stmt = select(FeatureReviewSummary).where(FeatureReviewSummary.app_id == app_id)
    summary_row = (await db.execute(summary_stmt)).scalar_one_or_none()
    if summary_row is not None:
        summary = ReviewSummarySchema(
            strengths=summary_row.core_strengths if isinstance(summary_row.core_strengths, list) else [],
            pain_points=summary_row.pain_points if isinstance(summary_row.pain_points, list) else [],
            feature_requests=summary_row.feature_requests if isinstance(summary_row.feature_requests, list) else [],
        )
    else:
        summary = ReviewSummarySchema(strengths=[], pain_points=[], feature_requests=[])

    is_processed = len(sentiment_rows) > 0 or len(topic_rows) > 0 or len(complaint_rows) > 0

    bundle = ReviewIntelligenceBundleSchema(
        app_id=app_id,
        sentiment=sentiment_overview,
        timeline=timeline,
        topics=topics,
        loved_features=loved_features,
        complaints=complaints,
        summary=summary,
        is_processed=is_processed,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return ApiResponse.ok(data=bundle, took_ms=elapsed_ms)
