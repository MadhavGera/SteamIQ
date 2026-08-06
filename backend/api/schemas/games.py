"""
Pydantic response schemas for the games API.

These are the shapes returned by api/games.py. They are intentionally
separate from the ORM models so the API contract is decoupled from the
database schema.

Phase 1: schemas represent raw_games data.
TODO(Phase5): create GameOverviewSchema backed by mart_game_overview.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class GenreSchema(BaseModel):
    id: str
    description: str


class PlatformSchema(BaseModel):
    windows: bool = False
    mac: bool = False
    linux: bool = False


class GameSummarySchema(BaseModel):
    """
    Lightweight game summary for search results.
    Returned by GET /api/v1/games/search.
    """
    app_id: int
    name: str
    short_description: Optional[str] = None
    header_image: Optional[str] = None
    developer: Optional[str] = None
    publisher: Optional[str] = None
    release_date: Optional[str] = None
    is_free: bool = False
    final_price_usd: Optional[Decimal] = None
    discount_pct: int = 0
    positive_reviews: int = 0
    negative_reviews: int = 0
    owners_estimate: Optional[str] = None
    genres: Optional[list[dict[str, Any]]] = None

    model_config = {"from_attributes": True}

    @property
    def review_total(self) -> int:
        return self.positive_reviews + self.negative_reviews

    @property
    def positive_pct(self) -> float | None:
        total = self.review_total
        if total == 0:
            return None
        return round(self.positive_reviews / total * 100, 1)


class GameDetailSchema(BaseModel):
    """
    Full game detail for a single game page.
    Returned by GET /api/v1/games/{app_id}.

    TODO(Phase5): repoint to mart_game_overview once it exists.
    This is the Phase 1 stopgap — reading directly from raw_games.
    """
    app_id: int
    name: str
    description: Optional[str] = None
    short_description: Optional[str] = None
    header_image: Optional[str] = None
    website: Optional[str] = None

    developer: Optional[str] = None
    publisher: Optional[str] = None
    release_date: Optional[str] = None
    coming_soon: bool = False

    genres: Optional[list[dict[str, Any]]] = None
    categories: Optional[list[dict[str, Any]]] = None
    tags: Optional[dict[str, Any]] = None

    is_free: bool = False
    price_usd: Optional[Decimal] = None
    final_price_usd: Optional[Decimal] = None
    discount_pct: int = 0

    platform_windows: bool = False
    platform_mac: bool = False
    platform_linux: bool = False

    positive_reviews: int = 0
    negative_reviews: int = 0
    review_score: Optional[int] = None
    review_score_desc: Optional[str] = None

    owners_estimate: Optional[str] = None
    average_playtime_forever: int = 0
    median_playtime_forever: int = 0

    metacritic_score: Optional[int] = None

    model_config = {"from_attributes": True}

    @property
    def review_total(self) -> int:
        return self.positive_reviews + self.negative_reviews

    @property
    def positive_pct(self) -> float | None:
        total = self.review_total
        if total == 0:
            return None
        return round(self.positive_reviews / total * 100, 1)


class GameSearchResultSchema(BaseModel):
    """Wrapper for paginated search results."""
    games: list[GameSummarySchema]
    total: int
    page: int
    page_size: int
    query: str
