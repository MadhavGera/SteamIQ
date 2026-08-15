"""
Unit & Integration Tests for Phase 5 Game Match (Roadmap v2 §3).

Tests:
  - MaterializeMatchProfilesJob execution & 6D intensity scoring
  - Idempotency on re-run
  - GET /api/v1/games/{app_id}/match/profile endpoint
  - POST /api/v1/games/{app_id}/match scoring endpoint
  - Ephemeral user preference evaluation
  - Architectural guardrails (zero raw/feature table scans in handler)
"""
from __future__ import annotations

import inspect
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import match
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewTopic,
    MartGameMatchProfile,
    RawGame,
    ServingSimilarGame,
)
from jobs.materialize_marts import MaterializeMartsJob
from jobs.materialize_match_profiles import MaterializeMatchProfilesJob
from jobs.process_embeddings import ProcessEmbeddingsJob

# ─── Fixtures ─────────────────────────────────────────────────────────────────

async def _seed_match_test_game(session: AsyncSession) -> int:
    """Seed game, tags, and review features for game match testing."""
    app_id = 95001

    # 1. Raw Game
    game = RawGame(
        app_id=app_id,
        name="Abyssal Soul: Dark Odyssey",
        developer="FromDark Devs",
        publisher="FromDark Pub",
        positive_reviews=9000,
        negative_reviews=1000,
        review_score=9,
        review_score_desc="Overwhelmingly Positive",
        owners_estimate="500,000 .. 1,000,000",
        final_price_usd=Decimal("39.99"),
        price_usd=Decimal("39.99"),
        discount_pct=0,
        is_free=False,
        genres=[{"id": "1", "description": "Action"}, {"id": "2", "description": "RPG"}],
        categories=[{"id": "1", "description": "Single-player"}, {"id": "2", "description": "Co-op"}],
        tags={"Souls-like": 500, "Difficult": 450, "Story Rich": 380, "Open World": 320, "Action": 400, "Co-op": 120},
        average_playtime_forever=1800,  # 30 hours
        median_playtime_forever=1500,
    )
    session.add(game)

    # 2. NLP Topics & Features
    topic = FeatureReviewTopic(
        app_id=app_id,
        topic_id=1,
        topic_label="Challenging Boss Encounters & Combat",
        review_count=450,
        sentiment_score=Decimal("0.92"),
    )
    session.add(topic)

    complaint = FeatureReviewComplaint(
        app_id=app_id,
        category="Difficulty Spikes & Late-Game Balance",
        volume_pct=Decimal("22.5"),
        severity="moderate",
        representative_snippets=["The endgame boss phase jump is brutally hard."],
    )
    session.add(complaint)

    feature = FeatureReviewFeature(
        app_id=app_id,
        feature_name="Deep Lore & Atmosphere",
        mention_count=380,
        praise_intensity=94,
    )
    session.add(feature)

    await session.commit()
    return app_id


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_materialize_match_profiles_job(db_session: AsyncSession) -> None:
    """Verify MaterializeMatchProfilesJob calculates 6 intensity dimensions."""
    app_id = await _seed_match_test_game(db_session)

    job = MaterializeMatchProfilesJob(app_id=app_id, dry_run=False)
    await job._run_async(session=db_session)

    profile = (
        await db_session.execute(
            select(MartGameMatchProfile).where(MartGameMatchProfile.app_id == app_id)
        )
    ).scalar_one_or_none()

    assert profile is not None
    assert profile.app_id == app_id
    # Souls-like + Difficult + complaints => High difficulty
    assert float(profile.difficulty) >= 7.0
    # Story rich + Deep lore => High story weight
    assert float(profile.story_weight) >= 6.0
    # Open world => High exploration
    assert float(profile.exploration) >= 6.0
    # Action + Hack and slash => High combat
    assert float(profile.combat) >= 6.0
    # Co-op tag => Moderate multiplayer
    assert float(profile.multiplayer) >= 2.0
    # 30 hours avg playtime => Long session length
    assert float(profile.session_length) >= 6.0

    assert profile.profile_summary is not None
    assert "primary_traits" in profile.profile_summary


@pytest.mark.anyio
async def test_match_profiles_idempotency(db_session: AsyncSession) -> None:
    """Verify that re-running MaterializeMatchProfilesJob replaces existing row idempotently."""
    app_id = await _seed_match_test_game(db_session)

    job = MaterializeMatchProfilesJob(app_id=app_id, dry_run=False)
    await job._run_async(session=db_session)
    await job._run_async(session=db_session)

    res = (
        await db_session.execute(
            select(MartGameMatchProfile).where(MartGameMatchProfile.app_id == app_id)
        )
    ).scalars().all()

    assert len(res) == 1


@pytest.mark.anyio
async def test_get_match_profile_api_endpoint(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test GET /api/v1/games/{app_id}/match/profile."""
    app_id = await _seed_match_test_game(db_session)

    job = MaterializeMatchProfilesJob(app_id=app_id, dry_run=False)
    await job._run_async(session=db_session)

    mart_job = MaterializeMartsJob(app_id=app_id, dry_run=False)
    await mart_job._run_async(session=db_session)

    resp = await client.get(f"/api/v1/games/{app_id}/match/profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["app_id"] == app_id
    assert "difficulty" in data
    assert "story_weight" in data
    assert "exploration" in data
    assert "combat" in data
    assert "multiplayer" in data
    assert "session_length" in data


@pytest.mark.anyio
async def test_post_match_game_scoring(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test POST /api/v1/games/{app_id}/match with user preference vector."""
    app_id = await _seed_match_test_game(db_session)

    job = MaterializeMatchProfilesJob(app_id=app_id, dry_run=False)
    await job._run_async(session=db_session)

    mart_job = MaterializeMartsJob(app_id=app_id, dry_run=False)
    await mart_job._run_async(session=db_session)

    # 1. Matching Preferences (Hardcore RPG player)
    user_prefs = {
        "difficulty": 8.5,
        "story_weight": 7.5,
        "exploration": 8.0,
        "combat": 8.0,
        "multiplayer": 2.0,
        "session_length": 7.0,
    }

    resp = await client.post(f"/api/v1/games/{app_id}/match", json=user_prefs)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["app_id"] == app_id
    assert data["overall_match_pct"] >= 75.0
    assert data["match_verdict"] in ["Exceptional Match", "Strong Match"]
    assert len(data["alignment_highlights"]) > 0

    # 2. Divergent Preferences (Casual Cozy non-combat player)
    cozy_prefs = {
        "difficulty": 1.0,
        "combat": 0.0,
        "story_weight": 2.0,
    }

    resp_cozy = await client.post(f"/api/v1/games/{app_id}/match", json=cozy_prefs)
    assert resp_cozy.status_code == 200
    body_cozy = resp_cozy.json()
    assert body_cozy["success"] is True
    data_cozy = body_cozy["data"]
    assert data_cozy["overall_match_pct"] < 65.0
    assert len(data_cozy["friction_points"]) > 0


@pytest.mark.anyio
async def test_match_game_404_handling(client: AsyncClient) -> None:
    """Verify proper 404 for unknown game."""
    resp = await client.post("/api/v1/games/99999999/match", json={"difficulty": 5.0})
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "game_not_found"


@pytest.mark.anyio
async def test_golden_rule_guardrail_in_match_api() -> None:
    """
    Permanent Architectural Guardrail:
    Assert api/match.py does not import or query RawGame, RawReview, or raw_*/feature_* tables.
    """
    source = inspect.getsource(match)
    assert "RawGame" not in source, "Golden Rule Violation: api/match.py references RawGame!"
    assert "RawReview" not in source, "Golden Rule Violation: api/match.py references RawReview!"
    assert "FeatureReview" not in source, "Golden Rule Violation: api/match.py references FeatureReview*!"
    assert "MartGameMatchProfile" in source
    assert "MartGameOverview" in source


@pytest.mark.anyio
async def test_serving_similar_games_market_presence_derivation(db_session: AsyncSession) -> None:
    """Verify market_presence derivation logic and storage in serving_similar_games."""
    app_id = await _seed_match_test_game(db_session)

    job = ProcessEmbeddingsJob(dry_run=True)
    game = (
        await db_session.execute(select(RawGame).where(RawGame.app_id == app_id))
    ).scalar_one()

    # 10,000 reviews + 1500 CCU => expected market presence between 65.0 and 95.0
    presence = job.derive_market_presence(game, ccu=1500)
    assert isinstance(presence, Decimal)
    assert float(presence) >= 60.0
    assert float(presence) <= 100.0

    # Verify zero CCU fallback still produces valid presence from reviews
    presence_no_ccu = job.derive_market_presence(game, ccu=0)
    assert isinstance(presence_no_ccu, Decimal)
    assert float(presence_no_ccu) >= 40.0

    # Direct persistence check into ServingSimilarGame
    serving_entry = ServingSimilarGame(
        source_app_id=app_id,
        target_app_id=app_id,
        similarity_score=0.95,
        rank=1,
        shared_tags=["Souls-like", "Difficult"],
        price_delta_usd=Decimal("0.00"),
        market_presence=presence,
    )
    db_session.add(serving_entry)
    await db_session.commit()

    saved = (
        await db_session.execute(
            select(ServingSimilarGame).where(
                ServingSimilarGame.source_app_id == app_id,
                ServingSimilarGame.target_app_id == app_id,
            )
        )
    ).scalar_one_or_none()

    assert saved is not None
    assert float(saved.market_presence) == float(presence)
