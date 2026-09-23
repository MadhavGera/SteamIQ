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
import logging
import httpx
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.games import GameDetailSchema, GameSearchResultSchema, GameSummarySchema
from core.errors import GameNotFoundError
from core.response import ApiResponse, ResponseMeta
from db.base import get_db
from db.models import MartGameOverview

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/games", tags=["games"])

# Simple in-memory cache for Steam API search results
_steam_search_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
STEAM_SEARCH_CACHE_TTL_SEC = 60

async def fetch_steam_search(query: str) -> list[dict[str, Any]]:
    """Fetch search results from Steam Store API with a 3s timeout and 60s cache."""
    now = time.time()
    if query in _steam_search_cache:
        cached_time, cached_results = _steam_search_cache[query]
        if now - cached_time < STEAM_SEARCH_CACHE_TTL_SEC:
            return cached_results

    url = f"https://store.steampowered.com/api/storesearch/?term={query}&l=english&cc=US"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            _steam_search_cache[query] = (now, items)
            return items
    except Exception as e:
        logger.error(f"Steam search API failed for query {query!r}: {e}")
        return []



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

    # Paginate local database matches
    offset = (page - 1) * page_size
    result = await db.execute(
        base_query.order_by(MartGameOverview.positive_reviews.desc()).offset(offset).limit(page_size)
    )
    db_games = result.scalars().all()
    
    # Track which app_ids are in the local database matches
    local_app_ids = {g.app_id for g in db_games}
    
    # Fetch from Steam API (only realistically needed for the first page)
    steam_items = await fetch_steam_search(q) if page == 1 else []
    
    merged_games = [GameSummarySchema.model_validate(g) for g in db_games]
    
    # Append uningested games from Steam
    for item in steam_items:
        app_id = item.get("id")
        if app_id and app_id not in local_app_ids:
            # HONEST-FALLBACK: Explicitly nulling fields not provided by storesearch API
            merged_games.append(
                GameSummarySchema(
                    app_id=app_id,
                    name=item.get("name", "Unknown Game"),
                    header_image=item.get("tiny_image"),
                    is_ingested=False,
                    short_description=None,
                    developer=None,
                    publisher=None,
                    release_date=None,
                    final_price_usd=None,
                    owners_estimate=None,
                    genres=None,
                    primary_genre=None,
                    revenue_tier=None,
                    success_score=None,
                    net_sentiment_pct=None,
                )
            )
            local_app_ids.add(app_id)  # Deduplicate within Steam results if necessary

    total += len(steam_items)  # Approximate total increase

    took_ms = (time.perf_counter() - start) * 1000

    return ApiResponse(
        success=True,
        data=GameSearchResultSchema(
            games=merged_games,
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
