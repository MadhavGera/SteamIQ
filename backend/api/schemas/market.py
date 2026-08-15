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

    # Pricing Intelligence
    price_tracking_started_at: datetime | None = Field(default=None, description="Timestamp when price tracking began for this game")
    pricing: PricingSpectrumSchema | None = Field(default=None, description="Comparable pricing spectrum")
    price_history: list[PriceHistoryPointSchema] = Field(default_factory=list, description="Historical price time-series records")

    # Market Opportunity Context
    market_opportunity: MarketOpportunitySummarySchema | None = Field(default=None, description="Market opportunity metrics for genre")

    model_config = {"from_attributes": True}
