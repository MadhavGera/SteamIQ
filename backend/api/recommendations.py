"""
Recommendations API router — Phase 5.

Endpoints:
  GET /api/v1/games/{app_id}/recommendations

ADR 0001, Decision 2 (Golden Rule):
  Reads strictly from serving_recommendations with zero request-time computation.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.recommendations import (
    RecommendationBundleSchema,
    RecommendationSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview, ServingRecommendation

router = APIRouter(prefix="/games", tags=["recommendations"])


@router.get(
    "/{app_id}/recommendations",
    response_model=ApiResponse[RecommendationBundleSchema],
    summary="Get prioritized recommendations for a game",
    description=(
        "Returns priority-ranked, explainable recommendations from serving_recommendations, "
        "including model_run_id traceability for ML-driven recommendations."
    ),
)
async def get_game_recommendations(
    app_id: Annotated[int, Path(description="Steam App ID", ge=1)],
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[RecommendationBundleSchema]:
    start = time.perf_counter()

    # 1. Verify game exists in decision marts
    overview = await db.get(MartGameOverview, app_id)
    if overview is None:
        raise GameNotFoundError(f"Game with app_id={app_id} not found")

    # 2. Query serving_recommendations (strictly Golden-Rule compliant)
    stmt = (
        select(ServingRecommendation)
        .where(ServingRecommendation.app_id == app_id)
        .order_by(ServingRecommendation.priority_rank.asc())
    )
    result = await db.execute(stmt)
    recs = result.scalars().all()

    rec_schemas = [RecommendationSchema.model_validate(r) for r in recs]
    domains = sorted(list({r.domain for r in recs}))

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=RecommendationBundleSchema(
            app_id=app_id,
            total_recommendations=len(rec_schemas),
            domains_covered=domains,
            recommendations=rec_schemas,
        ),
        meta=ResponseMeta(took_ms=round(took_ms, 2)),
    )
