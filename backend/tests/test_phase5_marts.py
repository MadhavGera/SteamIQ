"""
Unit & Integration Tests for Phase 5 (Decision Mart Layer, Pricing Intelligence, & Opportunity Finder).

Tests:
  - MaterializeMartsJob end-to-end execution and idempotency
  - mart_game_overview population with revenue tiers, neutral executive briefs, and pricing intelligence
  - Pricing Intelligence: comparable spectrum, genre benchmark distribution, historical low from time-series
  - mart_trends population for patch impact and time-series metrics
  - mart_opportunity_scores weighted-scoring formula (analytics only)
  - api/games.py and api/market.py reading exclusively from mart_* tables (Golden Rule compliance)
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import games, market
from db.models import (
    FeatureReviewComplaint,
    FeatureReviewFeature,
    FeatureReviewSentiment,
    FeatureReviewSummary,
    MartGameOverview,
    MartTrend,
    RawGame,
    RawGameTag,
    RawPriceHistory,
)
from jobs.materialize_marts import MaterializeMartsJob

# ─── Fixtures & Helpers ───────────────────────────────────────────────────────

async def _seed_full_pipeline_data(session: AsyncSession) -> None:
    """Helper to seed raw_* and feature_* tables for materialization testing."""
    # 1. Games
    g1 = RawGame(
        app_id=50001,
        name="Metroid Legend",
        developer="Indie Dev Studio",
        publisher="Indie Pubs",
        positive_reviews=10000,
        negative_reviews=1000,
        review_score=9,
        review_score_desc="Very Positive",
        owners_estimate="500,000 .. 1,000,000",
        final_price_usd=Decimal("17.99"),
        price_usd=Decimal("19.99"),
        discount_pct=10,
        is_free=False,
        genres=[{"id": "1", "description": "Action"}, {"id": "2", "description": "Metroidvania"}],
        average_playtime_forever=1200,
    )
    g2 = RawGame(
        app_id=50002,
        name="Roguelike Spire",
        developer="Deck Studio",
        publisher="Deck Pubs",
        positive_reviews=50000,
        negative_reviews=2000,
        review_score=9,
        review_score_desc="Overwhelmingly Positive",
        owners_estimate="2,000,000 .. 5,000,000",
        final_price_usd=Decimal("24.99"),
        price_usd=Decimal("24.99"),
        discount_pct=0,
        is_free=False,
        genres=[{"id": "3", "description": "Roguelike"}],
        average_playtime_forever=3500,
    )
    session.add_all([g1, g2])
    await session.commit()

    # 2. Sentiments
    s1_all = FeatureReviewSentiment(
        app_id=50001,
        month="ALL_TIME",
        positive_count=10000,
        negative_count=1000,
        total_count=11000,
        net_positive_pct=Decimal("90.91"),
        sentiment_score=Decimal("0.9091"),
    )
    s1_m1 = FeatureReviewSentiment(
        app_id=50001,
        month="2026-06",
        positive_count=500,
        negative_count=100,
        total_count=600,
        net_positive_pct=Decimal("83.33"),
        sentiment_score=Decimal("0.8333"),
    )
    s1_m2 = FeatureReviewSentiment(
        app_id=50001,
        month="2026-07",
        positive_count=800,
        negative_count=50,
        total_count=850,
        net_positive_pct=Decimal("94.12"),
        sentiment_score=Decimal("0.9412"),
    )

    s2_all = FeatureReviewSentiment(
        app_id=50002,
        month="ALL_TIME",
        positive_count=50000,
        negative_count=2000,
        total_count=52000,
        net_positive_pct=Decimal("96.15"),
        sentiment_score=Decimal("0.9615"),
    )
    session.add_all([s1_all, s1_m1, s1_m2, s2_all])

    # 3. Summaries, Complaints, Loved Features, Tags
    sum1 = FeatureReviewSummary(
        app_id=50001,
        core_strengths=["Fluid Combat Mechanics", "Atmospheric Worldbuilding"],
        pain_points=["Steep early difficulty spike"],
        feature_requests=["Boss rush mode"],
    )
    comp1 = FeatureReviewComplaint(
        app_id=50001,
        category="Difficulty Spikes",
        volume_pct=Decimal("28.50"),
        severity="high",
        representative_snippets=["The third boss is way too hard."],
    )
    feat1 = FeatureReviewFeature(
        app_id=50001,
        feature_name="Fluid Combat",
        mention_count=450,
        praise_intensity=95,
    )

    tag1 = RawGameTag(app_id=50001, tag_name="Metroidvania", votes=500)
    tag2 = RawGameTag(app_id=50002, tag_name="Roguelike", votes=1200)

    # 4. Multi-point Raw Price History for Pricing Intelligence
    now_dt = datetime.now(UTC)
    p1 = RawPriceHistory(
        app_id=50001,
        recorded_at=now_dt - timedelta(days=180),
        price_usd=Decimal("19.99"),
        final_price_usd=Decimal("19.99"),
        discount_pct=0,
        currency="USD",
    )
    p2 = RawPriceHistory(
        app_id=50001,
        recorded_at=now_dt - timedelta(days=90),
        price_usd=Decimal("19.99"),
        final_price_usd=Decimal("9.99"),
        discount_pct=50,
        currency="USD",
    )
    p3 = RawPriceHistory(
        app_id=50001,
        recorded_at=now_dt,
        price_usd=Decimal("19.99"),
        final_price_usd=Decimal("17.99"),
        discount_pct=10,
        currency="USD",
    )
    p4 = RawPriceHistory(
        app_id=50002,
        recorded_at=now_dt,
        price_usd=Decimal("24.99"),
        final_price_usd=Decimal("24.99"),
        discount_pct=0,
        currency="USD",
    )

    session.add_all([sum1, comp1, feat1, tag1, tag2, p1, p2, p3, p4])
    await session.commit()


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_materialize_marts_job_execution(db_session: AsyncSession) -> None:
    """Test full execution of MaterializeMartsJob and verify row counts."""
    await _seed_full_pipeline_data(db_session)

    job = MaterializeMartsJob(dry_run=False)
    result = await job._run_async(session=db_session)

    assert result["overview_rows"] == 2
    assert result["trend_rows"] >= 3
    assert result["opportunity_rows"] >= 2

    # Query mart_game_overview
    ov_res = await db_session.execute(select(MartGameOverview).order_by(MartGameOverview.app_id))
    overviews = ov_res.scalars().all()
    assert len(overviews) == 2

    g1_ov = overviews[0]
    assert g1_ov.app_id == 50001
    assert g1_ov.name == "Metroid Legend"
    assert g1_ov.primary_genre == "Action"
    assert g1_ov.revenue_tier is not None
    assert "Tier" in g1_ov.revenue_tier
    assert g1_ov.estimated_gross_revenue_usd is not None
    assert g1_ov.estimated_gross_revenue_usd > 0
    assert g1_ov.executive_brief is not None
    assert "market_position" in g1_ov.executive_brief
    assert "feedback_themes" in g1_ov.executive_brief


@pytest.mark.anyio
async def test_pricing_intelligence_materialization(db_session: AsyncSession) -> None:
    """Verify that Pricing Intelligence fields and comparable spectrum are materialized properly."""
    await _seed_full_pipeline_data(db_session)

    job = MaterializeMartsJob(dry_run=False)
    await job._run_async(session=db_session)

    ov_res = await db_session.execute(select(MartGameOverview).where(MartGameOverview.app_id == 50001))
    g1_ov = ov_res.scalar_one()

    # Price tracking started at should be populated
    assert g1_ov.price_tracking_started_at is not None

    # Pricing spectrum comparable object
    assert g1_ov.pricing_spectrum is not None
    assert g1_ov.pricing_spectrum["primary_genre"] == "Action"
    assert g1_ov.pricing_spectrum["historical_lowest_price_usd"] == 9.99
    assert g1_ov.pricing_spectrum["price_tracking_started_at"] is not None
    assert g1_ov.pricing_spectrum["position_bracket"] in [
        "Free-to-Play",
        "Budget / Entry-Tier",
        "Benchmark / Mid-Tier",
        "Premium Tier",
    ]

    # Price history time series points
    assert g1_ov.price_history_points is not None
    assert len(g1_ov.price_history_points) == 3

    # Game 50002 has only 1 snapshot: historical lowest should equal current price cleanly without synthetic data
    ov2_res = await db_session.execute(select(MartGameOverview).where(MartGameOverview.app_id == 50002))
    g2_ov = ov2_res.scalar_one()
    assert g2_ov.historical_lowest_price_usd == Decimal("24.99")
    assert g2_ov.price_tracking_started_at is not None
    assert len(g2_ov.price_history_points) == 1


@pytest.mark.anyio
async def test_materialize_marts_idempotency(db_session: AsyncSession) -> None:
    """Verify that running MaterializeMartsJob multiple times produces identical clean state."""
    await _seed_full_pipeline_data(db_session)

    job = MaterializeMartsJob(dry_run=False)
    res1 = await job._run_async(session=db_session)
    res2 = await job._run_async(session=db_session)

    assert res1["overview_rows"] == res2["overview_rows"]
    assert res1["opportunity_rows"] == res2["opportunity_rows"]

    ov_count = (await db_session.execute(select(MartGameOverview))).scalars().all()
    assert len(ov_count) == 2


@pytest.mark.anyio
async def test_mart_trends_patch_impact_calculation(db_session: AsyncSession) -> None:
    """Verify Update Impact Tracker trends (sentiment deltas and verdicts)."""
    await _seed_full_pipeline_data(db_session)

    job = MaterializeMartsJob(dry_run=False)
    await job._run_async(session=db_session)

    trends_res = await db_session.execute(
        select(MartTrend).where(
            MartTrend.app_id == 50001,
            MartTrend.trend_type == "patch_impact",
        )
    )
    patch_trend = trends_res.scalar_one_or_none()
    assert patch_trend is not None
    assert float(patch_trend.metric_value) > 0
    assert patch_trend.metadata_payload["verdict"] == "Positive Reception"
    assert patch_trend.metadata_payload["latest_month"] == "2026-07"


@pytest.mark.anyio
async def test_opportunity_scores_weighted_formula() -> None:
    """Test Opportunity Finder weighted scoring formula (analytics only)."""
    job = MaterializeMartsJob(dry_run=True)

    g1 = RawGame(
        app_id=1,
        name="Popular Game",
        positive_reviews=50000,
        negative_reviews=5000,
        review_score=9,
        final_price_usd=Decimal("29.99"),
        is_free=False,
    )
    complaint = FeatureReviewComplaint(
        app_id=1,
        category="Bugs",
        volume_pct=Decimal("35.00"),
        severity="high",
    )
    loved = FeatureReviewFeature(
        app_id=1,
        feature_name="Combat",
        mention_count=200,
        praise_intensity=90,
    )

    comps = job._compute_opportunity_score_components(
        genre_games=[g1],
        genre_complaints=[complaint],
        genre_loved=[loved],
        total_catalog_size=10,
    )

    opp_score = float(comps["opportunity_score"])
    assert 0.0 <= opp_score <= 100.0
    assert float(comps["demand_score"]) > 50.0
    assert float(comps["sentiment_gap_score"]) > 0.0
    assert float(comps["monetization_score"]) > 0.0


@pytest.mark.anyio
async def test_get_market_intelligence_endpoint(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test GET /api/v1/games/{app_id}/market endpoint."""
    await _seed_full_pipeline_data(db_session)

    job = MaterializeMartsJob(dry_run=False)
    await job._run_async(session=db_session)

    resp = await client.get("/api/v1/games/50001/market")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["app_id"] == 50001
    assert data["name"] == "Metroid Legend"
    assert data["primary_genre"] == "Action"
    assert data["price_tracking_started_at"] is not None
    assert data["pricing"] is not None
    assert data["pricing"]["price_tracking_started_at"] is not None
    assert data["pricing"]["historical_lowest_price_usd"] == 9.99
    assert data["pricing"]["historical_lowest_discount_pct"] == 50
    assert len(data["price_history"]) == 3


@pytest.mark.anyio
async def test_get_market_intelligence_not_found(client: AsyncClient) -> None:
    """Test 404 behavior for unknown game in Market endpoint."""
    resp = await client.get("/api/v1/games/99999999/market")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "game_not_found"


@pytest.mark.anyio
async def test_golden_rule_no_raw_reads_in_apis() -> None:
    """
    Permanent Architectural Guardrail:
    Assert api/games.py and api/market.py do not import or query RawGame or raw_* tables.
    """
    games_source = inspect.getsource(games)
    assert "RawGame" not in games_source, "Golden Rule Violation: api/games.py references RawGame!"
    assert "raw_games" not in games_source, "Golden Rule Violation: api/games.py references raw_games!"
    assert "MartGameOverview" in games_source

    market_source = inspect.getsource(market)
    assert "RawGame" not in market_source, "Golden Rule Violation: api/market.py references RawGame!"
    assert "raw_games" not in market_source, "Golden Rule Violation: api/market.py references raw_games!"
    assert "RawPriceHistory" not in market_source, "Golden Rule Violation: api/market.py references RawPriceHistory!"
    assert "MartGameOverview" in market_source
