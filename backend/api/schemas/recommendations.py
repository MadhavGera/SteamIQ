"""
Pydantic schemas for Recommendation Engine API (Phase 5).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecommendationSchema(BaseModel):
    """Single actionable recommendation card representation."""
    id: int = Field(..., description="Unique recommendation ID")
    app_id: int = Field(..., description="Steam App ID")
    model_run_id: uuid.UUID | None = Field(
        default=None,
        description="Model run ID if influenced by ML / SHAP predictions (traceability)",
    )
    recommendation_type: str = Field(..., description="Type category identifier")
    domain: str = Field(..., description="Domain tag (Pricing, Engineering, Content, Marketing, etc.)")
    priority_rank: int = Field(..., description="Priority ranking position (1 = highest ROI)")
    title: str = Field(..., description="Recommendation action title")
    impact_level: str = Field(..., description="Impact tier (High, Medium, Low)")
    difficulty_level: str = Field(..., description="Implementation difficulty (Low, Medium, High)")
    confidence_score: float = Field(..., description="Algorithm confidence score [0.0 to 1.0]")
    rationale: str = Field(..., description="Detailed diagnostic rationale")
    evidence_type: str = Field(..., description="Supporting evidence source (model_shap, review_nlp, pricing_comparable, etc.)")
    evidence_payload: dict[str, Any] | None = Field(default=None, description="Structured supporting metrics and snippets")
    action_items: list[str] | None = Field(default=None, description="Concrete step-by-step action items")
    generated_at: datetime = Field(..., description="Timestamp when recommendation was generated")

    model_config = {"from_attributes": True}


class RecommendationBundleSchema(BaseModel):
    """Full recommendation feed for the Recommendations tab."""
    app_id: int = Field(..., description="Steam App ID")
    total_recommendations: int = Field(..., description="Total recommendation count")
    domains_covered: list[str] = Field(default_factory=list, description="Unique domain tags present in feed")
    recommendations: list[RecommendationSchema] = Field(default_factory=list, description="Priority-ordered recommendations list")

    model_config = {"from_attributes": True}
