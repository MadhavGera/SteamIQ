"""
Game Match API — Phase 5 (Roadmap v2 §3).

Endpoints:
  - GET  /api/v1/games/{app_id}/match/profile : Pre-materialized 6D game match profile
  - POST /api/v1/games/{app_id}/match         : Evaluate user preference vector against profile

GOLDEN RULE COMPLIANCE & ADDENDUM (ADR 0001 / core/README.md):
  - Reads exclusively from mart_game_match_profile and mart_game_overview.
  - The in-memory comparison in match_game() is a documented, narrow exception
    to the Golden Rule: it computes a fixed-size mathematical distance (<= 6 numbers)
    against a precomputed mart row, performing zero live table scans or heavy aggregation.
  - User preference vectors are evaluated ephemerally and NEVER stored server-side.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.match import (
    DimensionMatchScoreSchema,
    GameMatchProfileSchema,
    GameMatchResultSchema,
    UserMatchPreferencesSchema,
)
from core.errors import GameNotFoundError
from core.response import ApiResponse
from db.base import get_db
from db.models import MartGameMatchProfile, MartGameOverview

router = APIRouter(tags=["Game Match"])


# ── GET /api/v1/games/{app_id}/match/profile ──────────────────────────────────

@router.get(
    "/games/{app_id}/match/profile",
    response_model=ApiResponse[GameMatchProfileSchema],
    summary="Get 6-dimension intensity match profile for a game",
)
async def get_game_match_profile(
    app_id: Annotated[int, Path(description="Steam App ID")],
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[GameMatchProfileSchema]:
    """
    Returns the pre-materialized 6-dimension intensity profile for a game:
    difficulty, story_weight, exploration, combat, multiplayer, session_length.
    Reads exclusively from mart_game_match_profile.
    """
    start = time.perf_counter()

    profile_res = await db.execute(
        select(MartGameMatchProfile).where(MartGameMatchProfile.app_id == app_id)
    )
    profile = profile_res.scalar_one_or_none()

    if profile is None:
        # Check if game exists in mart_game_overview to give proper 404
        game_res = await db.execute(
            select(MartGameOverview.app_id).where(MartGameOverview.app_id == app_id)
        )
        if game_res.scalar_one_or_none() is None:
            raise GameNotFoundError(f"Game with app_id {app_id} not found in catalog")
        raise GameNotFoundError(f"Match profile for game {app_id} has not been materialized yet")

    schema = GameMatchProfileSchema(
        app_id=profile.app_id,
        difficulty=float(profile.difficulty),
        story_weight=float(profile.story_weight),
        exploration=float(profile.exploration),
        combat=float(profile.combat),
        multiplayer=float(profile.multiplayer),
        session_length=float(profile.session_length),
        confidence_score=float(profile.confidence_score),
        profile_summary=profile.profile_summary,
        materialized_at=profile.materialized_at,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=schema, took_ms=elapsed_ms)


# ── POST /api/v1/games/{app_id}/match ─────────────────────────────────────────

@router.post(
    "/games/{app_id}/match",
    response_model=ApiResponse[GameMatchResultSchema],
    summary="Calculate percentage match for a game given user preferences",
)
async def match_game(
    app_id: Annotated[int, Path(description="Steam App ID")],
    prefs: UserMatchPreferencesSchema,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> ApiResponse[GameMatchResultSchema]:
    """
    Evaluates an ephemeral user preference vector against the game's pre-materialized profile.

    GOLDEN RULE ADDENDUM (core/README.md):
    Performs a fixed-size mathematical distance calculation (<= 6 numbers) against
    the pre-materialized row in mart_game_match_profile. Zero live table scanning.
    """
    start = time.perf_counter()

    # 1. Fetch pre-materialized profile and game name from mart tables
    profile_res = await db.execute(
        select(MartGameMatchProfile).where(MartGameMatchProfile.app_id == app_id)
    )
    profile = profile_res.scalar_one_or_none()

    if profile is None:
        game_res = await db.execute(
            select(MartGameOverview.name).where(MartGameOverview.app_id == app_id)
        )
        if game_res.scalar_one_or_none() is None:
            raise GameNotFoundError(f"Game with app_id {app_id} not found in catalog")
        raise GameNotFoundError(f"Match profile for game {app_id} has not been materialized yet")

    overview_res = await db.execute(
        select(MartGameOverview.name).where(MartGameOverview.app_id == app_id)
    )
    game_name = overview_res.scalar_one_or_none() or f"Steam Game #{app_id}"

    # 2. Extract profile intensity map
    game_intensity_map: dict[str, float] = {
        "difficulty": float(profile.difficulty),
        "story_weight": float(profile.story_weight),
        "exploration": float(profile.exploration),
        "combat": float(profile.combat),
        "multiplayer": float(profile.multiplayer),
        "session_length": float(profile.session_length),
    }

    # 3. Compute Dimension-by-Dimension Alignment
    # Map provided user preferences (default to 5.0 for dimensions left unselected)
    dimension_scores: dict[str, DimensionMatchScoreSchema] = {}
    alignment_highlights: list[str] = []
    friction_points: list[str] = []
    total_delta = 0.0
    evaluated_dims = 0

    dim_labels = {
        "difficulty": "Challenge & Difficulty",
        "story_weight": "Narrative & Story Depth",
        "exploration": "World Exploration",
        "combat": "Combat & Action Intensity",
        "multiplayer": "Multiplayer Focus",
        "session_length": "Session Length & Scope",
    }

    pref_dict = prefs.model_dump(exclude_unset=False)

    for dim_key, game_val in game_intensity_map.items():
        user_val = pref_dict.get(dim_key)
        # If user did not provide preference, treat as non-penalized or default
        if user_val is None:
            user_val = 5.0
            is_active_pref = False
        else:
            is_active_pref = True

        delta = abs(user_val - game_val)
        # Distance on 0..10 scale converted to percentage match
        dim_pct = max(0.0, round((1.0 - (delta / 10.0)) * 100.0, 1))

        dimension_scores[dim_key] = DimensionMatchScoreSchema(
            dimension=dim_labels.get(dim_key, dim_key),
            user_preference=round(user_val, 1),
            game_intensity=round(game_val, 1),
            delta=round(delta, 2),
            dimension_match_pct=dim_pct,
        )

        if is_active_pref:
            total_delta += delta
            evaluated_dims += 1

            if delta <= 1.5:
                alignment_highlights.append(
                    f"Strong alignment on {dim_labels[dim_key]} (Game: {game_val:.1f}/10 vs Preference: {user_val:.1f}/10)"
                )
            elif delta >= 3.5:
                friction_points.append(
                    f"Potential divergence on {dim_labels[dim_key]} (Game: {game_val:.1f}/10 vs Preference: {user_val:.1f}/10)"
                )

    # 4. Overall Match Percentage
    if evaluated_dims > 0:
        avg_delta = total_delta / evaluated_dims
        overall_match_pct = max(0.0, min(100.0, round((1.0 - (avg_delta / 10.0)) * 100.0, 1)))
    else:
        # Default baseline match if no specific vector specified
        overall_match_pct = 75.0
        alignment_highlights.append("Default neutral profile comparison")

    # 5. Verdict Classification
    if overall_match_pct >= 85.0:
        verdict = "Exceptional Match"
    elif overall_match_pct >= 70.0:
        verdict = "Strong Match"
    elif overall_match_pct >= 50.0:
        verdict = "Moderate Match"
    else:
        verdict = "Low Match / Divergent Taste"

    result = GameMatchResultSchema(
        app_id=app_id,
        game_name=game_name,
        overall_match_pct=overall_match_pct,
        match_verdict=verdict,
        dimension_scores=dimension_scores,
        alignment_highlights=alignment_highlights,
        friction_points=friction_points,
        confidence_score=float(profile.confidence_score),
        materialized_at=profile.materialized_at,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=result, took_ms=elapsed_ms)
