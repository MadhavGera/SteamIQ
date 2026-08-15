"""
Unit & Integration Tests for Phase 5 Recommendation Engine (Step 5.4).

Tests:
  - GenerateRecommendationsJob hybrid synthesis (rules + model output)
  - model_run_id traceability on model-driven recommendations
  - serving_recommendations table population, ranking, and idempotency
  - GET /api/v1/games/{app_id}/recommendations API endpoint
  - Architectural guardrails (zero live generation or raw table scans in handler)
"""
from __future__ import annotations

import inspect
import uuid
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import recommendations
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    ModelRun,
    RawGame,
    ServingPrediction,
    ServingRecommendation,
)
from jobs.generate_recommendations import GenerateRecommendationsJob
from jobs.materialize_marts import MaterializeMartsJob

# ─── Fixtures ─────────────────────────────────────────────────────────────────

async def _seed_recommendation_pipeline_data(session: AsyncSession) -> tuple[int, uuid.UUID]:
    """Seed data required for hybrid recommendation generation."""
    app_id = 70001
    run_id = uuid.uuid4()

    # 1. Raw Game
    game = RawGame(
        app_id=app_id,
        name="Cyber Roguelite 2088",
        developer="Neon Devs",
        publisher="Neon Pubs",
        positive_reviews=12000,
        negative_reviews=3000,
        review_score=8,
        review_score_desc="Very Positive",
        owners_estimate="500,000 .. 1,000,000",
        final_price_usd=Decimal("12.99"),
        price_usd=Decimal("12.99"),
        discount_pct=0,
        is_free=False,
        genres=[{"id": "1", "description": "Roguelike"}],
        average_playtime_forever=120,
    )
    session.add(game)
    await session.commit()

    # 2. Model Run & Serving Prediction (ML output + SHAP attribution)
    model_run = ModelRun(
        id=run_id,
        model_name="success_classifier_v2",
        stage="production",
        dataset_version="v2026.1",
        metrics={"f1": 0.88, "roc_auc": 0.93},
    )
    session.add(model_run)
    await session.commit()

    pred = ServingPrediction(
        app_id=app_id,
        model_run_id=run_id,
        prediction_type="success_score",
        score=Decimal("0.7850"),
        feature_importance={"complaint_density": 0.35, "price_usd": 0.25},
        shap_values={
            "complaint_density": -0.22,
            "loved_feature_density": 0.18,
            "price_usd": -0.08,
        },
    )
    session.add(pred)

    # 3. Feature Review Complaints & Features
    complaint = FeatureReviewComplaint(
        app_id=app_id,
        category="Crashing & Performance",
        volume_pct=Decimal("34.50"),
        severity="high",
        representative_snippets=["Crashes on level 3 boss every time."],
    )
    loved = FeatureReviewFeature(
        app_id=app_id,
        feature_name="Fast-Paced Combat",
        mention_count=620,
        praise_intensity=94,
    )
    sentiment = FeatureReviewSentiment(
        app_id=app_id,
        month="ALL_TIME",
        positive_count=12000,
        negative_count=3000,
        total_count=15000,
        net_positive_pct=Decimal("80.00"),
        sentiment_score=Decimal("0.8000"),
    )

    session.add_all([complaint, loved, sentiment])
    await session.commit()

    return app_id, run_id


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_generate_recommendations_job_hybrid_synthesis(db_session: AsyncSession) -> None:
    """
    Test that GenerateRecommendationsJob generates a hybrid blend of:
      1. Model-driven recommendations (with model_run_id traceability and SHAP evidence)
      2. Rules-driven recommendations (with model_run_id=None and domain heuristics)
    """
    app_id, run_id = await _seed_recommendation_pipeline_data(db_session)

    job = GenerateRecommendationsJob(app_id=app_id, dry_run=False)
    result = await job._run_async(session=db_session)

    assert result["recommendation_rows"] >= 2
    assert result["games_processed"] == 1

    # Query serving_recommendations
    recs_res = await db_session.execute(
        select(ServingRecommendation)
        .where(ServingRecommendation.app_id == app_id)
        .order_by(ServingRecommendation.priority_rank.asc())
    )
    recs = recs_res.scalars().all()
    assert len(recs) >= 2

    # Check rank 1 is highest priority
    rec1 = recs[0]
    assert rec1.priority_rank == 1
    assert rec1.impact_level in ["High", "Medium"]
    assert rec1.domain in [
        "Engineering & Quality",
        "Pricing & Monetization",
        "Content & Gameplay",
        "Marketing & Discovery",
        "Community & Retention",
    ]

    # Verify at least one recommendation carries model_run_id from SHAP attribution
    model_driven = [r for r in recs if r.model_run_id is not None]
    assert len(model_driven) >= 1
    assert model_driven[0].model_run_id == run_id
    assert model_driven[0].evidence_type in ["model_shap", "hybrid"]
    assert "shap_value" in model_driven[0].evidence_payload or "shap_driver" in model_driven[0].evidence_payload

    # Verify at least one rule-based recommendation exists with model_run_id=None
    rule_driven = [r for r in recs if r.model_run_id is None]
    assert len(rule_driven) >= 1
    assert rule_driven[0].evidence_type in ["pricing_comparable", "review_nlp", "retention_telemetry"]


@pytest.mark.anyio
async def test_generate_recommendations_idempotency(db_session: AsyncSession) -> None:
    """Verify that running GenerateRecommendationsJob multiple times produces identical clean rows."""
    app_id, _ = await _seed_recommendation_pipeline_data(db_session)

    job = GenerateRecommendationsJob(app_id=app_id, dry_run=False)
    res1 = await job._run_async(session=db_session)
    res2 = await job._run_async(session=db_session)

    assert res1["recommendation_rows"] == res2["recommendation_rows"]

    recs_count = (
        await db_session.execute(
            select(ServingRecommendation).where(ServingRecommendation.app_id == app_id)
        )
    ).scalars().all()
    assert len(recs_count) == res1["recommendation_rows"]


@pytest.mark.anyio
async def test_get_recommendations_api_endpoint(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test GET /api/v1/games/{app_id}/recommendations endpoint."""
    app_id, run_id = await _seed_recommendation_pipeline_data(db_session)

    # Populate marts and recommendations
    mart_job = MaterializeMartsJob(app_id=app_id, dry_run=False)
    await mart_job._run_async(session=db_session)

    rec_job = GenerateRecommendationsJob(app_id=app_id, dry_run=False)
    await rec_job._run_async(session=db_session)

    resp = await client.get(f"/api/v1/games/{app_id}/recommendations")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["app_id"] == app_id
    assert data["total_recommendations"] >= 2
    assert len(data["recommendations"]) == data["total_recommendations"]

    # Verify model_run_id is correctly serialized
    rec_items = data["recommendations"]
    has_model_run = any(r.get("model_run_id") == str(run_id) for r in rec_items)
    assert has_model_run is True


@pytest.mark.anyio
async def test_get_recommendations_not_found(client: AsyncClient) -> None:
    """Test 404 behavior for unknown game."""
    resp = await client.get("/api/v1/games/99999999/recommendations")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "game_not_found"


@pytest.mark.anyio
async def test_golden_rule_no_raw_reads_in_recommendations_api() -> None:
    """
    Permanent Architectural Guardrail:
    Assert api/recommendations.py does not import or query RawGame or raw_*/feature_* tables.
    """
    source = inspect.getsource(recommendations)
    assert "RawGame" not in source, "Golden Rule Violation: api/recommendations.py references RawGame!"
    assert "raw_games" not in source, "Golden Rule Violation: api/recommendations.py references raw_games!"
    assert "FeatureReview" not in source, "Golden Rule Violation: api/recommendations.py references FeatureReview*!"
    assert "ServingRecommendation" in source
    assert "MartGameOverview" in source
