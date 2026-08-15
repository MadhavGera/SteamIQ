"""
Pydantic response schemas for the games API — Phase 5.

These are the shapes returned by api/games.py, backed directly by mart_game_overview.
ADR 0001, Decision 2: API handlers read only from mart_* or serving_* tables.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel


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
    Backed by mart_game_overview.
    """
    app_id: int
    name: str
    short_description: str | None = None
    header_image: str | None = None
    developer: str | None = None
    publisher: str | None = None
    release_date: str | None = None
    is_free: bool = False
    final_price_usd: Decimal | None = None
    discount_pct: int = 0
    positive_reviews: int = 0
    negative_reviews: int = 0
    owners_estimate: str | None = None
    genres: list[dict[str, Any]] | None = None
    primary_genre: str | None = None
    revenue_tier: str | None = None
    success_score: Decimal | None = None
    net_sentiment_pct: Decimal | None = None

    model_config = {"from_attributes": True}

    @property
    def review_total(self) -> int:
        return self.positive_reviews + self.negative_reviews

    @property
    def positive_pct(self) -> float | None:
        if self.net_sentiment_pct is not None:
            return float(self.net_sentiment_pct)
        total = self.review_total
        if total == 0:
            return None
        return round(self.positive_reviews / total * 100, 1)


class GameDetailSchema(BaseModel):
    """
    Full game detail for a single game page.
    Returned by GET /api/v1/games/{app_id}.
    Backed by mart_game_overview.
    """
    app_id: int
    name: str
    description: str | None = None
    short_description: str | None = None
    header_image: str | None = None
    website: str | None = None

    developer: str | None = None
    publisher: str | None = None
    release_date: str | None = None
    coming_soon: bool = False

    genres: list[dict[str, Any]] | None = None
    categories: list[dict[str, Any]] | None = None
    tags: dict[str, Any] | None = None

    is_free: bool = False
    price_usd: Decimal | None = None
    final_price_usd: Decimal | None = None
    discount_pct: int = 0

    platform_windows: bool = False
    platform_mac: bool = False
    platform_linux: bool = False

    positive_reviews: int = 0
    negative_reviews: int = 0
    review_score: int | None = None
    review_score_desc: str | None = None

    owners_estimate: str | None = None
    average_playtime_forever: int = 0
    median_playtime_forever: int = 0

    metacritic_score: int | None = None

    # Decision / Mart layer additions (Phase 5)
    primary_genre: str | None = None
    # Directional estimate based on public SteamSpy owner estimate range & final price (not verified financial data)
    revenue_tier: str | None = None
    estimated_gross_revenue_usd: Decimal | None = None
    success_score: Decimal | None = None
    net_sentiment_pct: Decimal | None = None
    peak_ccu_24h: int | None = None
    executive_brief: dict[str, Any] | None = None
    top_strengths: list[str] | None = None
    top_complaints: list[dict[str, Any]] | None = None
    shap_values: dict[str, float] | None = None
    model_run_id: str | None = None

    model_config = {"from_attributes": True}

    @property
    def review_total(self) -> int:
        return self.positive_reviews + self.negative_reviews

    @property
    def positive_pct(self) -> float | None:
        if self.net_sentiment_pct is not None:
            return float(self.net_sentiment_pct)
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
