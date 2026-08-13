"""
Pydantic schemas for Competitor Intelligence API (Phase 3).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompetitorItemSchema(BaseModel):
    """A single similar competitor game."""
    rank: int
    app_id: int
    name: str
    similarity_score: float
    similarity_pct: int
    shared_tags: list[str] = Field(default_factory=list)
    price_usd: str | None = None
    price_delta_usd: float | None = None
    positive_reviews: int = 0
    negative_reviews: int = 0
    review_pct: int | None = None
    header_image: str | None = None
    genres: list[dict[str, Any]] | None = None


class SourceGameHeaderSchema(BaseModel):
    """Header context for the source game being analyzed."""
    app_id: int
    name: str
    header_image: str | None = None
    final_price_usd: str | None = None
    positive_reviews: int = 0
    negative_reviews: int = 0
    review_pct: int | None = None
    genres: list[dict[str, Any]] | None = None


class CompetitorListSchema(BaseModel):
    """Complete Competitor discovery bundle for a game."""
    app_id: int
    source_game: SourceGameHeaderSchema
    competitors: list[CompetitorItemSchema] = Field(default_factory=list)
    total: int = 0
    model_name: str = "all-MiniLM-L6-v2"
    is_processed: bool = True
