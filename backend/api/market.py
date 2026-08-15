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

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.market import (
    MarketIntelligenceSchema,
    MarketOpportunitySummarySchema,
    PriceHistoryPointSchema,
    PricingSpectrumSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview, MartOpportunityScore

router = APIRouter(prefix="/games", tags=["market"])


@router.get(
    "/{app_id}/market",
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
