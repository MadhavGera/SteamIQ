"""
Games API router — Phase 5.

Endpoints:
  GET /api/v1/games/search?q={name}&page={n}&page_size={n}
  GET /api/v1/games/{app_id}

ADR 0001, Decision 2 (Golden Rule):
  These handlers read from mart_game_overview directly with zero request-time
  heavy aggregations or scans on raw_*/feature_* tables.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.games import GameDetailSchema, GameSearchResultSchema, GameSummarySchema
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview

router = APIRouter(prefix="/games", tags=["games"])


# ─── Search ───────────────────────────────────────────────────────────────────

@router.get(
    "/search",
    response_model=ApiResponse[GameSearchResultSchema],
    summary="Search games by name",
    description=(
        "Full-text search over pre-materialized game overviews. "
        "Returns paginated GameSummarySchema items from mart_game_overview."
    ),
)
async def search_games(
    q: Annotated[str, Query(min_length=1, max_length=200, description="Game name search term")],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=50, description="Results per page")] = 20,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[GameSearchResultSchema]:
    start = time.perf_counter()

    like_pattern = f"%{q}%"
    base_query = select(MartGameOverview).where(MartGameOverview.name.ilike(like_pattern))

    # Count total matches
    count_query = select(func.count()).select_from(base_query.subquery())
    total: int = (await db.execute(count_query)).scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    result = await db.execute(
        base_query.order_by(MartGameOverview.positive_reviews.desc()).offset(offset).limit(page_size)
    )
    games = result.scalars().all()

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=GameSearchResultSchema(
            games=[GameSummarySchema.model_validate(g) for g in games],
            total=total,
            page=page,
            page_size=page_size,
            query=q,
        ),
        meta=ResponseMeta(
            took_ms=round(took_ms, 2),
            page=page,
            page_size=page_size,
            total=total,
        ),
    )


# ─── Detail ───────────────────────────────────────────────────────────────────

@router.get(
    "/{app_id}",
    response_model=ApiResponse[GameDetailSchema],
    summary="Get game detail by Steam app_id",
    description=(
        "Returns full pre-materialized game metadata, KPIs, and executive intelligence from mart_game_overview."
    ),
)
async def get_game(
    app_id: int,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[GameDetailSchema]:
    start = time.perf_counter()

    result = await db.execute(select(MartGameOverview).where(MartGameOverview.app_id == app_id))
    game = result.scalar_one_or_none()

    if game is None:
        raise GameNotFoundError(
            f"No game found with app_id {app_id}. "
            f"Run 'make ingest APPID={app_id}' to ingest and materialize it."
        )

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse.ok(
        data=GameDetailSchema.model_validate(game),
        took_ms=round(took_ms, 2),
    )
