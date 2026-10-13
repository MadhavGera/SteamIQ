"""
Market & Pricing Intelligence API endpoint (Phase 5).

Provides pre-materialized commercial placement, revenue tiering, comparable price
positioning spectrum (non-causal), historical price trajectory, and genre market
opportunity scores.

Golden Rule (ADR 0001, Decision 2):
Reads strictly from mart_game_overview and mart_opportunity_scores.
Zero live aggregation in the handler.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.market import (
    MarketIntelligenceSchema,
    MarketOpportunitySummarySchema,
    MarketTrendPointSchema,
    OpportunityScoreItemSchema,
    PriceHistoryPointSchema,
    PricingSpectrumSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview, MartOpportunityScore, MartTrend

router = APIRouter(tags=["market"])


@router.get(
    "/games/{app_id}/market",
    response_model=ApiResponse[MarketIntelligenceSchema],
    summary="Get Market & Pricing Intelligence for a game",
    description=(
        "Returns pre-materialized market performance, commercial revenue tiering, "
        "comparable pricing spectrum against genre benchmarks, historical price time-series, "
        "and genre opportunity index from mart_game_overview and mart_opportunity_scores."
    ),
)
async def get_market_intelligence(
    app_id: Annotated[int, Path(description="Steam App ID", ge=1)],
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[MarketIntelligenceSchema]:
    start = time.perf_counter()

    # 1. Fetch pre-materialized game overview
    overview = await db.get(MartGameOverview, app_id)
    if overview is None:
        raise GameNotFoundError(f"Game with app_id={app_id} not found")

    # 2. Extract comparable pricing spectrum
    pricing_data: PricingSpectrumSchema | None = None
    if overview.pricing_spectrum:
        pricing_data = PricingSpectrumSchema(**overview.pricing_spectrum)
    elif overview.genre_median_price_usd is not None:
        pricing_data = PricingSpectrumSchema(
            primary_genre=overview.primary_genre or "General",
            game_price_usd=float(overview.final_price_usd or overview.price_usd or 0.0),
            genre_min_price_usd=float(overview.genre_min_price_usd or 4.99),
            genre_p25_price_usd=float(overview.genre_p25_price_usd or 9.99),
            genre_median_price_usd=float(overview.genre_median_price_usd or 14.99),
            genre_p75_price_usd=float(overview.genre_p75_price_usd or 19.99),
            genre_max_price_usd=float(overview.genre_max_price_usd or 29.99),
            price_vs_median_pct=0.0,
            position_bracket="Benchmark / Mid-Tier" if not overview.is_free else "Free-to-Play",
            historical_lowest_price_usd=float(overview.historical_lowest_price_usd or overview.final_price_usd or 0.0),
            historical_lowest_discount_pct=overview.historical_lowest_discount_pct or 0,
            is_free=overview.is_free,
        )

    # 3. Extract price history points
    price_history: list[PriceHistoryPointSchema] = []
    if overview.price_history_points and isinstance(overview.price_history_points, list):
        price_history = [
            PriceHistoryPointSchema(
                recorded_at=pt.get("recorded_at"),
                price_usd=pt.get("price_usd"),
                final_price_usd=pt.get("final_price_usd"),
                discount_pct=pt.get("discount_pct", 0),
            )
            for pt in overview.price_history_points
        ]

    # 4. Fetch genre market opportunity score if available
    market_opp: MarketOpportunitySummarySchema | None = None
    if overview.primary_genre:
        opp_stmt = select(MartOpportunityScore).where(
            MartOpportunityScore.genre_or_tag == overview.primary_genre,
            MartOpportunityScore.entity_type == "genre",
        )
        opp_res = await db.execute(opp_stmt)
        opp_row = opp_res.scalar_one_or_none()
        if opp_row:
            market_opp = MarketOpportunitySummarySchema(
                genre=opp_row.genre_or_tag,
                opportunity_score=float(opp_row.opportunity_score),
                demand_score=float(opp_row.demand_score),
                saturation_score=float(opp_row.saturation_score),
                sentiment_gap_score=float(opp_row.sentiment_gap_score),
                monetization_score=float(opp_row.monetization_score),
            )

    data = MarketIntelligenceSchema(
        app_id=overview.app_id,
        name=overview.name,
        primary_genre=overview.primary_genre,
        revenue_tier=overview.revenue_tier,
        estimated_gross_revenue_usd=overview.estimated_gross_revenue_usd,
        owners_estimate=overview.owners_estimate,
        success_score=overview.success_score,
        net_sentiment_pct=overview.net_sentiment_pct,
        peak_ccu_24h=overview.peak_ccu_24h,
        price_tracking_started_at=overview.price_tracking_started_at,
        pricing=pricing_data,
        price_history=price_history,
        market_opportunity=market_opp,
    )

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=data,
        meta=ResponseMeta(took_ms=round(took_ms, 2)),
    )


@router.get(
    "/market/opportunities",
    response_model=ApiResponse[list[OpportunityScoreItemSchema]],
    summary="List Pre-Materialized Market Opportunity Scores",
    description=(
        "Returns pre-materialized opportunity scores across genres and tags from "
        "mart_opportunity_scores. Strictly zero live aggregation (ADR 0001, Decision 2)."
    ),
)
async def get_market_opportunities(
    entity_type: Annotated[str | None, Query(description="Filter by 'genre' or 'tag'")] = None,
    limit: Annotated[int, Query(description="Maximum results to return", ge=1, le=100)] = 50,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[list[OpportunityScoreItemSchema]]:
    start = time.perf_counter()

    stmt = select(MartOpportunityScore).order_by(MartOpportunityScore.opportunity_score.desc())
    if entity_type:
        stmt = stmt.where(MartOpportunityScore.entity_type == entity_type)
    stmt = stmt.limit(limit)

    res = await db.execute(stmt)
    rows = res.scalars().all()

    data = [
        OpportunityScoreItemSchema(
            id=r.id,
            genre_or_tag=r.genre_or_tag,
            entity_type=r.entity_type,
            opportunity_score=float(r.opportunity_score),
            demand_score=float(r.demand_score),
            saturation_score=float(r.saturation_score),
            sentiment_gap_score=float(r.sentiment_gap_score),
            monetization_score=float(r.monetization_score),
            game_count=r.game_count,
            median_price_usd=float(r.median_price_usd) if r.median_price_usd is not None else None,
            avg_review_score=float(r.avg_review_score) if r.avg_review_score is not None else None,
            top_complaint_themes=r.top_complaint_themes if isinstance(r.top_complaint_themes, list | dict) else None,
            top_loved_themes=r.top_loved_themes if isinstance(r.top_loved_themes, list | dict) else None,
            recommended_features=r.recommended_features if isinstance(r.recommended_features, list | dict) else None,
            materialized_at=r.materialized_at,
        )
        for r in rows
    ]

    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse(
        success=True,
        data=data,
        meta=ResponseMeta(took_ms=round(took_ms, 2)),
    )


@router.get(
    "/market/trends",
    response_model=ApiResponse[list[MarketTrendPointSchema]],
    summary="List Pre-Materialized Market Trends",
    description=(
        "Returns pre-materialized market and platform trends from mart_trends. "
        "Strictly zero live aggregation (ADR 0001, Decision 2)."
    ),
)
async def get_market_trends(
    trend_type: Annotated[str | None, Query(description="Filter by trend_type (e.g. 'platform', 'genre')")] = None,
    category: Annotated[str | None, Query(description="Filter by category or genre name")] = None,
    limit: Annotated[int, Query(description="Maximum results to return", ge=1, le=500)] = 100,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[list[MarketTrendPointSchema]]:
    start = time.perf_counter()

    stmt = select(MartTrend).order_by(MartTrend.recorded_date.desc())
    if trend_type:
        stmt = stmt.where(MartTrend.trend_type == trend_type)
    if category:
        stmt = stmt.where(MartTrend.category == category)
    stmt = stmt.limit(limit)

    res = await db.execute(stmt)
    rows = res.scalars().all()

    data = [
        MarketTrendPointSchema(
            id=r.id,
            app_id=r.app_id,
            trend_type=r.trend_type,
            category=r.category,
            recorded_date=r.recorded_date,
            metric_name=r.metric_name,
            metric_value=float(r.metric_value),
            change_pct_7d=float(r.change_pct_7d) if r.change_pct_7d is not None else None,
            change_pct_30d=float(r.change_pct_30d) if r.change_pct_30d is not None else None,
            metadata_payload=r.metadata_payload if isinstance(r.metadata_payload, dict) else None,
            materialized_at=r.materialized_at,
        )
        for r in rows
    ]

    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse(
        success=True,
        data=data,
        meta=ResponseMeta(took_ms=round(took_ms, 2)),
    )

