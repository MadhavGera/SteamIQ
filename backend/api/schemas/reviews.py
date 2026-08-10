"""
Pydantic schemas for the Review Intelligence API (Phase 2 — NLP Core).

Decoupled from ORM models. Returned by api/reviews.py.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class SentimentOverviewSchema(BaseModel):
    """Overall sentiment summary metrics."""
    positive_pct: float = Field(..., description="Net positive percentage (0-100)")
    mixed_pct: float = Field(0.0, description="Mixed / neutral percentage (0-100)")
    negative_pct: float = Field(0.0, description="Negative percentage (0-100)")
    positive_count: int = Field(0, description="Total positive reviews analyzed")
    negative_count: int = Field(0, description="Total negative reviews analyzed")
    total_count: int = Field(0, description="Total reviews analyzed")
    sentiment_label: str = Field("Unknown", description="e.g. Overwhelmingly Positive, Mostly Positive")


class MonthlySentimentSchema(BaseModel):
    """Historical sentiment for a single month."""
    month: str = Field(..., description="YYYY-MM (e.g. '2024-05')")
    positive_reviews: int = Field(0, description="Positive reviews in month")
    negative_reviews: int = Field(0, description="Negative reviews in month")
    net_positive_pct: float = Field(0.0, description="Net positive % for the month")


class ReviewTopicSchema(BaseModel):
    """BERTopic semantic cluster."""
    topic_id: int
    label: str = Field(..., description="User-facing topic name (e.g. Combat & Boss Design)")
    review_count: int = Field(0, description="Number of reviews mentioning this topic")
    sentiment_score: float = Field(0.5, description="Topic sentiment score 0.0 - 1.0")
    keywords: Optional[list[str]] = Field(default=None, description="Top keywords in cluster")


class LovedFeatureSchema(BaseModel):
    """Positively appreciated gameplay feature from HDBSCAN embeddings."""
    feature_name: str = Field(..., description="e.g. Soundtrack & Audio")
    mention_count: int = Field(0, description="Count of positive mentions")
    praise_intensity: int = Field(80, description="0-100 score indicating depth of praise")


class ComplaintCategorySchema(BaseModel):
    """Zero-shot classified complaint category."""
    category: str = Field(..., description="e.g. Performance / FPS Drops")
    volume_pct: float = Field(0.0, description="Share of negative reviews mentioning this")
    severity: str = Field("moderate", description="high | moderate | low")
    representative_snippets: list[str] = Field(default_factory=list, description="3 representative quote snippets")


class ReviewSummarySchema(BaseModel):
    """Hierarchical structured summary of all player feedback."""
    strengths: list[str] = Field(default_factory=list, description="Core acclaimed elements")
    pain_points: list[str] = Field(default_factory=list, description="Primary player friction points")
    feature_requests: list[str] = Field(default_factory=list, description="Common feature requests or QoL wishes")


class ReviewIntelligenceBundleSchema(BaseModel):
    """Full bundle for Game Intelligence → Tab 2 (Reviews)."""
    app_id: int
    sentiment: SentimentOverviewSchema
    timeline: list[MonthlySentimentSchema]
    topics: list[ReviewTopicSchema]
    loved_features: list[LovedFeatureSchema]
    complaints: list[ComplaintCategorySchema]
    summary: ReviewSummarySchema
    is_processed: bool = Field(True, description="True if NLP pipeline has run, False if raw data only")
