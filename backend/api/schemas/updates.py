"""
Pydantic schemas for Update Impact Tracker API (Phase 5).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UpdateImpactItemSchema(BaseModel):
    """Single patch event before/after window impact comparison."""
    id: int = Field(..., description="Unique record ID")
    app_id: int = Field(..., description="Steam App ID")
    patch_name: str = Field(..., description="Patch or content release title")
    patch_version: str | None = Field(default=None, description="Semantic version string")
    patch_date: datetime = Field(..., description="Date when update was released")
    is_inferred: bool = Field(default=False, description="Whether patch event was inferred from review/telemetry surge")
    window_days: int = Field(default=14, description="Evaluation window duration in days")

    # Observed Sentiment Delta
    pre_sentiment_positive_pct: float | None = Field(default=None, description="Pre-patch window net positive sentiment %")
    post_sentiment_positive_pct: float | None = Field(default=None, description="Post-patch window net positive sentiment %")
    sentiment_delta_pct: float | None = Field(default=None, description="Observed sentiment delta %")
    observed_sentiment_verdict: str = Field(..., description="Categorical verdict (Observed Positive / Mixed / Backlash)")

    # Observed Player Activity Delta
    pre_avg_ccu: int | None = Field(default=None, description="Average concurrent players in pre-window")
    post_avg_ccu: int | None = Field(default=None, description="Average concurrent players in post-window")
    ccu_change_pct: float | None = Field(default=None, description="Observed concurrent player shift %")

    # Observed Complaint Topic Shifts
    pre_complaint_distribution: dict[str, float] | None = Field(default=None, description="Complaint category volume % pre-patch")
    post_complaint_distribution: dict[str, float] | None = Field(default=None, description="Complaint category volume % post-patch")
    top_resolved_complaints: list[dict[str, Any]] | None = Field(default=None, description="Complaints with largest observed drop")
    top_emerging_complaints: list[dict[str, Any]] | None = Field(default=None, description="Complaints with largest observed rise")

    # Observational Note
    correlation_summary: str = Field(..., description="Explicit non-causal summary of observed telemetry")
    materialized_at: datetime = Field(..., description="Timestamp when decision record was materialized")

    model_config = {"from_attributes": True}


class UpdateImpactFeedSchema(BaseModel):
    """Full update impact history feed for Tab 6 (Updates)."""
    app_id: int = Field(..., description="Steam App ID")
    total_updates: int = Field(..., description="Total tracked patch events")
    latest_verdict: str | None = Field(default=None, description="Verdict from most recent patch event")
    average_sentiment_delta: float | None = Field(default=None, description="Average sentiment delta across tracked updates")
    updates: list[UpdateImpactItemSchema] = Field(default_factory=list, description="Chronological patch impact records (newest first)")

    model_config = {"from_attributes": True}
