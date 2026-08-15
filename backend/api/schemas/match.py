"""
Pydantic schemas for the Game Match API — Phase 5 (Roadmap v2 §3).

Defines request/response shapes for:
  - Game Match 6-dimension intensity profiles
  - User preference vectors (<=6 values, never persisted server-side)
  - Game match score and dimension alignment breakdown
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GameMatchProfileSchema(BaseModel):
    """Pre-materialized 6-dimension intensity profile for a game."""
    app_id: int
    difficulty: float
    story_weight: float
    exploration: float
    combat: float
    multiplayer: float
    session_length: float
    confidence_score: float
    profile_summary: dict[str, Any] | None = None
    materialized_at: datetime

    model_config = {"from_attributes": True}


class UserMatchPreferencesSchema(BaseModel):
    """
    User-supplied preference vector for Game Match scoring.
    <= 6 intensity preference values (0.0 to 10.0 scale).
    Never persisted server-side.
    """
    difficulty: float | None = Field(None, ge=0.0, le=10.0, description="Preferred challenge intensity (0=relaxing, 10=hardcore)")
    story_weight: float | None = Field(None, ge=0.0, le=10.0, description="Preferred narrative depth (0=pure gameplay, 10=story rich)")
    exploration: float | None = Field(None, ge=0.0, le=10.0, description="Preferred world exploration (0=linear, 10=open world / secrets)")
    combat: float | None = Field(None, ge=0.0, le=10.0, description="Preferred action/combat intensity (0=pacifist/puzzle, 10=hack-and-slash/FPS)")
    multiplayer: float | None = Field(None, ge=0.0, le=10.0, description="Preferred multiplayer focus (0=pure solo, 10=heavy co-op/PvP)")
    session_length: float | None = Field(None, ge=0.0, le=10.0, description="Preferred session/game commitment (0=bite-sized, 10=epic 40h+)")


class DimensionMatchScoreSchema(BaseModel):
    """Detailed score comparison on an individual gameplay dimension."""
    dimension: str
    user_preference: float
    game_intensity: float
    delta: float
    dimension_match_pct: float


class GameMatchResultSchema(BaseModel):
    """
    Match evaluation output comparing user preferences against pre-materialized game profile.
    Computed on-the-fly from mart_game_match_profile (Golden Rule addendum exception).
    """
    app_id: int
    game_name: str
    overall_match_pct: float
    match_verdict: str  # Exceptional Match, Strong Match, Moderate Match, Low Match
    dimension_scores: dict[str, DimensionMatchScoreSchema]
    alignment_highlights: list[str]
    friction_points: list[str]
    confidence_score: float
    materialized_at: datetime
