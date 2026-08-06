"""
Games API router — Phase 1.

Endpoints:
  GET /api/v1/games/search?q={name}&page={n}&page_size={n}
  GET /api/v1/games/{app_id}

ADR 0001, Decision 2 (Golden Rule):
  These handlers read from raw_games directly — this is the ONLY Phase 1
  stopgap. Every raw_* read is marked TODO(Phase5).

  TODO(Phase5): repoint both endpoints to mart_game_overview once it exists.
  DO NOT add any aggregation, NLP inference, or additional raw_* reads here.
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
from db.models import RawGame

router = APIRouter(prefix="/games", tags=["games"])


# ─── Search ───────────────────────────────────────────────────────────────────

@router.get(
    "/search",
    response_model=ApiResponse[GameSearchResultSchema],
    summary="Search games by name",
    description=(
        "Full-text search over ingested game names. "
        "Returns paginated GameSummarySchema items. "
        "**Phase 1 stopgap**: reads from raw_games directly. "
        "TODO(Phase5): repoint to mart_game_overview."
    ),
)
async def search_games(
    q: Annotated[str, Query(min_length=1, max_length=200, description="Game name search term")],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=50, description="Results per page")] = 20,
    db: AsyncSession = Depends(get_db),
) -> ApiResponse[GameSearchResultSchema]:
    start = time.perf_counter()

    # TODO(Phase5): repoint to mart_game_overview once it exists.
    # Phase 1 stopgap: ILIKE search on raw_games.name
    like_pattern = f"%{q}%"
    base_query = select(RawGame).where(RawGame.name.ilike(like_pattern))

    # Count total matches
    count_query = select(func.count()).select_from(base_query.subquery())
    total: int = (await db.execute(count_query)).scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    result = await db.execute(
        base_query.order_by(RawGame.positive_reviews.desc()).offset(offset).limit(page_size)
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
        "Returns full game metadata for a single game. "
        "**Phase 1 stopgap**: reads from raw_games directly. "
        "TODO(Phase5): repoint to mart_game_overview."
    ),
)
async def get_game(
    app_id: int,
    db: AsyncSession = Depends(get_db),
) -> ApiResponse[GameDetailSchema]:
    start = time.perf_counter()

    # TODO(Phase5): repoint to mart_game_overview once it exists.
    # Phase 1 stopgap: read from raw_games by app_id.
    result = await db.execute(select(RawGame).where(RawGame.app_id == app_id))
    game = result.scalar_one_or_none()

    if game is None:
        raise GameNotFoundError(
            f"No game found with app_id {app_id}. "
            f"Run 'make ingest APPID={app_id}' to ingest it."
        )

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse.ok(
        data=GameDetailSchema.model_validate(game),
        took_ms=round(took_ms, 2),
    )
