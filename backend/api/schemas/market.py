"""
Pydantic schemas for Market & Pricing Intelligence (Phase 5).
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PricingSpectrumSchema(BaseModel):
    """Comparable pricing spectrum positioning (strictly non-causal)."""
    primary_genre: str = Field(..., description="Primary genre classification")
    game_price_usd: float = Field(..., description="Current list price of the game")
    genre_min_price_usd: float = Field(..., description="Lowest non-free price in genre comparable set")
    genre_p25_price_usd: float = Field(..., description="25th percentile price in genre comparable set")
    genre_median_price_usd: float = Field(..., description="Median price benchmark in genre comparable set")
    genre_p75_price_usd: float = Field(..., description="75th percentile price in genre comparable set")
    genre_max_price_usd: float = Field(..., description="Highest price in genre comparable set")
    price_vs_median_pct: float = Field(..., description="Percentage difference vs genre median price")
    position_bracket: str = Field(..., description="Non-causal market pricing bracket classification")
    historical_lowest_price_usd: float = Field(..., description="Historical lowest price observed across time series")
    historical_lowest_discount_pct: int = Field(default=0, description="Highest discount percentage observed across time series")
    price_tracking_started_at: str | None = Field(default=None, description="ISO timestamp when price tracking began for this game")
    is_free: bool = Field(default=False, description="Whether the game is free to play")


class PriceHistoryPointSchema(BaseModel):
    """Historical price data point from time series."""
    recorded_at: str | None = Field(default=None, description="ISO timestamp of price record")
    price_usd: float | None = Field(default=None, description="Base/original list price")
    final_price_usd: float | None = Field(default=None, description="Discounted final price")
    discount_pct: int = Field(default=0, description="Discount percentage")


class MarketOpportunitySummarySchema(BaseModel):
    """Genre market opportunity context."""
    genre: str = Field(..., description="Genre identifier")
    opportunity_score: float = Field(..., description="Weighted market opportunity score [0-100]")
    demand_score: float = Field(..., description="Demand index [0-100]")
    saturation_score: float = Field(..., description="Market saturation index [0-100]")
    sentiment_gap_score: float = Field(..., description="Sentiment gap / complaint headroom [0-100]")
    monetization_score: float = Field(..., description="Monetization index [0-100]")


class GenrePricingSpectrumSchema(BaseModel):
    """Comparable pricing spectrum summary for genre (Frontend Tab 4 compatibility)."""
    genre: str = Field(..., description="Primary genre classification")
    median_price_usd: float = Field(..., description="Median price benchmark")
    q25_price_usd: float = Field(..., description="25th percentile price benchmark")
    q75_price_usd: float = Field(..., description="75th percentile price benchmark")
    min_price_usd: float = Field(..., description="Lowest observed price benchmark")
    max_price_usd: float = Field(..., description="Highest observed price benchmark")
    discounted_game_share_pct: float = Field(default=0.0, description="Percentage of genre titles currently discounted")


class MarketIntelligenceSchema(BaseModel):
    """Unified Market & Pricing Intelligence view for Tab 4."""
    app_id: int = Field(..., description="Steam App ID")
    name: str = Field(..., description="Game title")
    primary_genre: str | None = Field(default=None, description="Primary genre classification")
    revenue_tier: str | None = Field(default=None, description="Directional commercial bracket estimate")
    estimated_gross_revenue_usd: Decimal | None = Field(default=None, description="Directional gross revenue estimate")
    owners_estimate: str | None = Field(default=None, description="Public owner estimate bracket")
    success_score: Decimal | None = Field(default=None, description="Success prediction score [0-100]")
    net_sentiment_pct: Decimal | None = Field(default=None, description="Net positive sentiment percentage")
    peak_ccu_24h: int | None = Field(default=None, description="Peak 24-hour concurrent users")

    # Pricing Intelligence (Phase 5 / Frontend contract compatibility)
    current_price_usd: float | None = Field(default=None, description="Current list price")
    price_tier: str | None = Field(default=None, description="Pricing bracket classification")
    historical_lowest_price_usd: float | None = Field(default=None, description="Historical lowest observed price")
    price_tracking_started_at: datetime | None = Field(default=None, description="Timestamp when price tracking began for this game")
    genre_pricing_spectrum: GenrePricingSpectrumSchema | None = Field(default=None, description="Genre pricing spectrum")
    pricing: PricingSpectrumSchema | None = Field(default=None, description="Comparable pricing spectrum")
    competitor_price_distribution: dict[str, int] = Field(default_factory=dict, description="Distribution of competitors across price tiers")
    recent_price_snapshots: list[PriceHistoryPointSchema] = Field(default_factory=list, description="Recent price snapshot records")
    price_history: list[PriceHistoryPointSchema] = Field(default_factory=list, description="Historical price time-series records")

    # Market Opportunity Context
    market_opportunity: MarketOpportunitySummarySchema | None = Field(default=None, description="Market opportunity metrics for genre")

    model_config = {"from_attributes": True}


class OpportunityScoreItemSchema(BaseModel):
    """Pre-materialized genre or tag market opportunity score item."""
    id: int = Field(..., description="Opportunity score record ID")
    genre_or_tag: str = Field(..., description="Genre or tag cluster identifier")
    entity_type: str = Field(..., description="Entity classification ('genre' or 'tag')")
    opportunity_score: float = Field(..., description="Weighted market opportunity score [0-100]")
    demand_score: float = Field(..., description="Demand index [0-100]")
    saturation_score: float = Field(..., description="Market saturation index [0-100]")
    sentiment_gap_score: float = Field(..., description="Sentiment gap / complaint headroom [0-100]")
    monetization_score: float = Field(..., description="Monetization index [0-100]")
    game_count: int = Field(default=0, description="Total game count in cluster")
    median_price_usd: float | None = Field(default=None, description="Median list price in USD")
    avg_review_score: float | None = Field(default=None, description="Average review score [0-100]")
    top_complaint_themes: list[str] | dict | None = Field(default=None, description="Top negative review themes")
    top_loved_themes: list[str] | dict | None = Field(default=None, description="Top positive review themes")
    recommended_features: list[str] | dict | None = Field(default=None, description="Recommended feature signals")
    materialized_at: datetime | None = Field(default=None, description="Timestamp of mart calculation")

    model_config = {"from_attributes": True}


class MarketTrendPointSchema(BaseModel):
    """Pre-materialized market or platform trend data point."""
    id: int = Field(..., description="Trend record ID")
    app_id: int | None = Field(default=None, description="Associated Steam App ID if game-specific")
    trend_type: str = Field(..., description="Trend category (e.g. 'platform', 'genre', 'patch_impact')")
    category: str | None = Field(default=None, description="Sub-category or genre name")
    recorded_date: datetime = Field(..., description="Time point of observation")
    metric_name: str = Field(..., description="Metric name (e.g. 'avg_sentiment', 'avg_ccu')")
    metric_value: float = Field(..., description="Numeric value of metric")
    change_pct_7d: float | None = Field(default=None, description="7-day percentage change")
    change_pct_30d: float | None = Field(default=None, description="30-day percentage change")
    metadata_payload: dict | None = Field(default=None, description="Additional context payload")
    materialized_at: datetime | None = Field(default=None, description="Timestamp of mart calculation")

    model_config = {"from_attributes": True}
