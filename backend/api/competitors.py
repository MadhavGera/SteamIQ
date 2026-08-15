"""
Competitors API router — Phase 3.

Endpoint:
  GET /api/v1/games/{app_id}/competitors?limit=10

ADR 0001, Decision 2 (Golden Rule):
  This handler reads ONLY from serving_similar_games joined with raw_games for display.
  Zero live vector distance calculations or embedding generations in the request path.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.competitors import (
    CompetitorItemSchema,
    CompetitorListSchema,
    SourceGameHeaderSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview, RawGame, ServingSimilarGame

router = APIRouter(prefix="/games", tags=["competitors"])


@router.get(
    "/{app_id}/competitors",
    response_model=ApiResponse[CompetitorListSchema],
    summary="Get precomputed similar competitor games",
    description=(
        "Returns top-N vector-similar competitor games from `serving_similar_games`. "
        "Generated via semantic embeddings over description, tags, and review topics."
    ),
)
async def get_competitors(
    app_id: int,
    limit: Annotated[int, Query(ge=1, le=50, description="Max competitors to return")] = 10,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[CompetitorListSchema]:
    start = time.perf_counter()

    # 1. Fetch source game (prefer pre-materialized decision mart)
    res_source = await db.execute(select(MartGameOverview).where(MartGameOverview.app_id == app_id))
    source_game = res_source.scalar_one_or_none()
    if source_game is None:
        res_source = await db.execute(select(RawGame).where(RawGame.app_id == app_id))
        source_game = res_source.scalar_one_or_none()

    if source_game is None:
        raise GameNotFoundError(
            f"No game found with app_id {app_id}. "
            f"Run 'make ingest APPID={app_id}' to ingest it."
        )

    # Compute source review %
    src_total = (source_game.positive_reviews or 0) + (source_game.negative_reviews or 0)
    src_pct = round((source_game.positive_reviews or 0) / src_total * 100) if src_total > 0 else None

    source_header = SourceGameHeaderSchema(
        app_id=source_game.app_id,
        name=source_game.name,
        header_image=source_game.header_image,
        final_price_usd=str(source_game.final_price_usd) if source_game.final_price_usd is not None else None,
        positive_reviews=source_game.positive_reviews or 0,
        negative_reviews=source_game.negative_reviews or 0,
        review_pct=src_pct,
        genres=source_game.genres,
    )

    # 2. Fetch precomputed similar games from serving_similar_games joined with decision mart
    query = (
        select(ServingSimilarGame, MartGameOverview)
        .join(MartGameOverview, ServingSimilarGame.target_app_id == MartGameOverview.app_id)
        .where(ServingSimilarGame.source_app_id == app_id)
        .order_by(ServingSimilarGame.rank.asc())
        .limit(limit)
    )
    result = await db.execute(query)
    rows = result.all()

    if not rows:
        # Fallback to raw_games join if decision marts are unmaterialized in unit tests
        query_raw = (
            select(ServingSimilarGame, RawGame)
            .join(RawGame, ServingSimilarGame.target_app_id == RawGame.app_id)
            .where(ServingSimilarGame.source_app_id == app_id)
            .order_by(ServingSimilarGame.rank.asc())
            .limit(limit)
        )
        rows = (await db.execute(query_raw)).all()

    if not rows:
        took_ms = (time.perf_counter() - start) * 1000
        return ApiResponse.ok(
            data=CompetitorListSchema(
                app_id=app_id,
                source_game=source_header,
                competitors=[],
                total=0,
                model_name="all-MiniLM-L6-v2",
                is_processed=False,
            ),
            took_ms=round(took_ms, 2),
        )

    competitor_items: list[CompetitorItemSchema] = []
    for sim_row, target_game in rows:
        sim_score = float(sim_row.similarity_score)
        t_total = target_game.positive_reviews + target_game.negative_reviews
        t_pct = round(target_game.positive_reviews / t_total * 100) if t_total > 0 else None
        p_delta = float(sim_row.price_delta_usd) if sim_row.price_delta_usd is not None else None

        competitor_items.append(
            CompetitorItemSchema(
                rank=sim_row.rank,
                app_id=target_game.app_id,
                name=target_game.name,
                similarity_score=sim_score,
                similarity_pct=round(sim_score * 100),
                shared_tags=sim_row.shared_tags if isinstance(sim_row.shared_tags, list) else [],
                price_usd=str(target_game.final_price_usd) if target_game.final_price_usd is not None else None,
                price_delta_usd=p_delta,
                positive_reviews=target_game.positive_reviews,
                negative_reviews=target_game.negative_reviews,
                review_pct=t_pct,
                market_presence=float(sim_row.market_presence) if sim_row.market_presence is not None else None,
                header_image=target_game.header_image,
                genres=target_game.genres,
            )
        )

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=CompetitorListSchema(
            app_id=app_id,
            source_game=source_header,
            competitors=competitor_items,
            total=len(competitor_items),
            model_name="all-MiniLM-L6-v2",
            is_processed=True,
        ),
        meta=ResponseMeta(
            took_ms=round(took_ms, 2),
            total=len(competitor_items),
        ),
    )
