"""
Player Activity & Telemetry API router — Phase 5.

Endpoints:
  GET /api/v1/games/{app_id}/activity

ADR 0001, Decision 2 (Golden Rule):
  Reads strictly from mart_game_overview with zero request-time heavy aggregations
  or scans on raw_*/feature_* tables.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.activity import (
    CCUPointSchema,
    EventImpactSummarySchema,
    PlayerActivitySchema,
    PlaytimeCohortSchema,
    PlaytimePercentilesSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview

router = APIRouter(prefix="/games", tags=["activity"])


@router.get(
    "/{app_id}/activity",
    response_model=ApiResponse[PlayerActivitySchema],
    summary="Get Player Activity & Telemetry intelligence for a game",
    description=(
        "Returns pre-materialized player activity telemetry, concurrent player counts (CCU), "
        "empirical playtime percentiles, and retention cohort distributions from mart_game_overview."
    ),
)
async def get_player_activity(
    app_id: Annotated[int, Path(description="Steam App ID", ge=1)],
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[PlayerActivitySchema]:
    start = time.perf_counter()

    # 1. Verify game exists in decision marts
    overview = await db.get(MartGameOverview, app_id)
    if overview is None:
        raise GameNotFoundError(f"Game with app_id={app_id} not found")

    # 2. Extract pre-materialized player activity payload
    if overview.player_activity:
        act_data = overview.player_activity
        activity = PlayerActivitySchema(
            app_id=act_data.get("app_id", overview.app_id),
            game_name=act_data.get("game_name", overview.name),
            current_ccu=act_data.get("current_ccu", overview.peak_ccu_24h),
            peak_ccu_24h=act_data.get("peak_ccu_24h", overview.peak_ccu_24h),
            all_time_peak_ccu=act_data.get("all_time_peak_ccu", overview.peak_ccu_24h),
            owners_estimate=act_data.get("owners_estimate", overview.owners_estimate),
            average_playtime_forever_hours=act_data.get("average_playtime_forever_hours"),
            median_playtime_forever_hours=act_data.get("median_playtime_forever_hours"),
            average_playtime_at_review_hours=act_data.get("average_playtime_at_review_hours"),
            post_review_engagement_delta_hours=act_data.get("post_review_engagement_delta_hours"),
            sample_size_reviews=act_data.get("sample_size_reviews", 0),
            playtime_percentiles=PlaytimePercentilesSchema(
                **(act_data.get("playtime_percentiles") or {})
            ),
            playtime_cohorts=[
                PlaytimeCohortSchema(**c) for c in (act_data.get("playtime_cohorts") or [])
            ],
            ccu_history=[
                CCUPointSchema(**pt) for pt in (act_data.get("ccu_history") or [])
            ],
            recent_event_impacts=[
                EventImpactSummarySchema(**imp) for imp in (act_data.get("recent_event_impacts") or [])
            ],
            telemetry_mode=act_data.get("telemetry_mode", "snapshot_polled"),
            updated_at=act_data.get("updated_at"),
        )
    else:
        # Fallback to direct scalar fields on overview if player_activity json hasn't materialized
        avg_h = round(overview.average_playtime_forever / 60.0, 1) if overview.average_playtime_forever > 0 else None
        med_h = round(overview.median_playtime_forever / 60.0, 1) if overview.median_playtime_forever > 0 else None
        activity = PlayerActivitySchema(
            app_id=overview.app_id,
            game_name=overview.name,
            current_ccu=overview.peak_ccu_24h,
            peak_ccu_24h=overview.peak_ccu_24h,
            all_time_peak_ccu=overview.peak_ccu_24h,
            owners_estimate=overview.owners_estimate,
            average_playtime_forever_hours=avg_h,
            median_playtime_forever_hours=med_h,
            average_playtime_at_review_hours=None,
            post_review_engagement_delta_hours=None,
            sample_size_reviews=0,
            playtime_percentiles=PlaytimePercentilesSchema(
                median_hours=med_h,
            ),
            playtime_cohorts=[],
            ccu_history=[],
            recent_event_impacts=[],
            telemetry_mode="snapshot_polled",
            updated_at=overview.materialized_at.isoformat() if overview.materialized_at else None,
        )

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=activity,
        meta=ResponseMeta(
            took_ms=round(took_ms, 2),
            page=None,
            page_size=None,
            total=None,
        ),
    )
