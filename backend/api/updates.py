"""
Update Impact Tracker API router — Phase 5.

Endpoints:
  GET /api/v1/games/{app_id}/updates

ADR 0001, Decision 2 (Golden Rule):
  Reads strictly from mart_update_impact with zero request-time computation.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.updates import (
    UpdateImpactFeedSchema,
    UpdateImpactItemSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview, MartUpdateImpact

router = APIRouter(prefix="/games", tags=["updates"])


@router.get(
    "/{app_id}/updates",
    response_model=ApiResponse[UpdateImpactFeedSchema],
    summary="Get Update Impact Tracker history for a game",
    description=(
        "Returns pre-materialized before/after window comparisons around patch dates "
        "tracking sentiment, player CCU, and complaint shifts from mart_update_impact."
    ),
)
async def get_game_update_impact(
    app_id: Annotated[int, Path(description="Steam App ID", ge=1)],
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[UpdateImpactFeedSchema]:
    start = time.perf_counter()

    # 1. Verify game exists in decision marts
    overview = await db.get(MartGameOverview, app_id)
    if overview is None:
        raise GameNotFoundError(f"Game with app_id={app_id} not found")

    # 2. Query mart_update_impact (strictly Golden-Rule compliant)
    stmt = (
        select(MartUpdateImpact)
        .where(MartUpdateImpact.app_id == app_id)
        .order_by(MartUpdateImpact.patch_date.desc())
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    item_schemas = [UpdateImpactItemSchema.model_validate(r) for r in records]

    latest_verdict = records[0].observed_sentiment_verdict if records else None
    deltas = [float(r.sentiment_delta_pct) for r in records if r.sentiment_delta_pct is not None]
    avg_delta = round(sum(deltas) / len(deltas), 2) if deltas else None

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=UpdateImpactFeedSchema(
            app_id=app_id,
            total_updates=len(item_schemas),
            latest_verdict=latest_verdict,
            average_sentiment_delta=avg_delta,
            updates=item_schemas,
        ),
        meta=ResponseMeta(took_ms=round(took_ms, 2)),
    )
