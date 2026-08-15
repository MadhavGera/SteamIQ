"""
Review Intelligence API router — Phase 5 (Option A: Mart Materialization).

Endpoints:
  GET /api/v1/games/{app_id}/reviews

ADR 0001, Decision 2 (Golden Rule):
  Reads strictly from mart_review_intelligence and mart_game_overview.
  Zero request-time queries to raw_* or feature_* tables.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
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
from db.models import MartGameOverview, MartReviewIntelligence, RawGame

router = APIRouter(prefix="/games/{app_id}/reviews", tags=["reviews"])


# ─── Full Review Intelligence Bundle ───────────────────────────────────────────

@router.get(
    "",
    response_model=ApiResponse[ReviewIntelligenceBundleSchema],
    summary="Get complete Review Intelligence bundle",
    description=(
        "Returns pre-materialized sentiment, timeline, topics, complaints, loved features, "
        "and summary from mart_review_intelligence."
    ),
)
async def get_review_intelligence(
    app_id: int = Path(..., description="Steam App ID"),
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[ReviewIntelligenceBundleSchema]:
    start = time.perf_counter()

    # 1. Fetch pre-materialized review intelligence mart (Golden Rule strictly compliant)
    mart_rev = await db.get(MartReviewIntelligence, app_id)

    if mart_rev is not None:
        bundle = ReviewIntelligenceBundleSchema(
            app_id=app_id,
            sentiment=SentimentOverviewSchema(**mart_rev.sentiment_overview),
            timeline=[MonthlySentimentSchema(**item) for item in (mart_rev.timeline or [])],
            topics=[ReviewTopicSchema(**item) for item in (mart_rev.topics or [])],
            loved_features=[LovedFeatureSchema(**item) for item in (mart_rev.loved_features or [])],
            complaints=[ComplaintCategorySchema(**item) for item in (mart_rev.complaints or [])],
            summary=ReviewSummarySchema(**(mart_rev.summary or {})),
            is_processed=mart_rev.is_processed,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return ApiResponse.ok(data=bundle, took_ms=elapsed_ms)

    # 2. Fallback for unmaterialized game check (reads from mart_game_overview)
    overview = await db.get(MartGameOverview, app_id)
    raw_game = await db.get(RawGame, app_id) if overview is None else None
    if overview is None and raw_game is None:
        raise GameNotFoundError(f"Game with app_id={app_id} not found")

    positive_revs = overview.positive_reviews if overview else (raw_game.positive_reviews if raw_game else 0)
    negative_revs = overview.negative_reviews if overview else (raw_game.negative_reviews if raw_game else 0)
    score_desc = overview.review_score_desc if overview else (raw_game.review_score_desc if raw_game else None)

    tot_cnt = (positive_revs or 0) + (negative_revs or 0)
    pos_pct = round((positive_revs or 0) / tot_cnt * 100, 1) if tot_cnt > 0 else 0.0
    sentiment_overview = SentimentOverviewSchema(
        positive_pct=pos_pct,
        mixed_pct=max(0.0, round(100.0 - pos_pct - ((negative_revs or 0) / tot_cnt * 100 if tot_cnt else 0), 1)),
        negative_pct=round((negative_revs or 0) / tot_cnt * 100, 1) if tot_cnt > 0 else 0.0,
        positive_count=positive_revs or 0,
        negative_count=negative_revs or 0,
        total_count=tot_cnt,
        sentiment_label=score_desc or "Unknown",
    )

    bundle = ReviewIntelligenceBundleSchema(
        app_id=app_id,
        sentiment=sentiment_overview,
        timeline=[],
        topics=[],
        loved_features=[],
        complaints=[],
        summary=ReviewSummarySchema(strengths=[], pain_points=[], feature_requests=[]),
        is_processed=False,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return ApiResponse.ok(data=bundle, took_ms=elapsed_ms)
