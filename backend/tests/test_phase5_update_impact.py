"""
Unit & Integration Tests for Phase 5 Update Impact Tracker (Step 5.3).

Tests:
  - MaterializeUpdateImpactJob execution and before/after window comparisons
  - Non-causal observation labeling ("observed/correlated", never "caused")
  - Shift calculations in sentiment, player CCU, and complaint topic rates
  - Idempotency on re-run
  - GET /api/v1/games/{app_id}/updates API endpoint
  - Architectural guardrails (zero live computation or raw reads in handler)
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api import updates
from db.models import (
    FeatureReviewComplaint,
    MartUpdateImpact,
    RawGame,
    RawPatchNote,
    RawPlayerSnapshot,
    RawReview,
)
from jobs.ingest_news import IngestNewsJob
from jobs.materialize_marts import MaterializeMartsJob
from jobs.materialize_update_impact import MaterializeUpdateImpactJob

# ─── Fixtures ─────────────────────────────────────────────────────────────────

async def _seed_update_impact_data(session: AsyncSession) -> int:
    """Seed game, reviews, player snapshots, and complaints for update impact testing."""
    app_id = 80001
    now_dt = datetime.now(UTC)

    # 1. Raw Game
    game = RawGame(
        app_id=app_id,
        name="Tactical Cyber Squad",
        developer="Matrix Studio",
        publisher="Matrix Pub",
        positive_reviews=8000,
        negative_reviews=2000,
        review_score=8,
        review_score_desc="Very Positive",
        owners_estimate="200,000 .. 500,000",
        final_price_usd=Decimal("19.99"),
        price_usd=Decimal("19.99"),
        discount_pct=0,
        is_free=False,
        genres=[{"id": "1", "description": "Strategy"}],
        average_playtime_forever=240,
    )
    session.add(game)
    await session.commit()

    # 2. Raw Reviews (spread across pre- and post-patch windows around 45 days ago)
    patch_45d_ago = now_dt - timedelta(days=45)
    p_ts = int(patch_45d_ago.timestamp())

    reviews_list = []
    # Pre-patch (50 days ago): 60% positive
    for i in range(10):
        reviews_list.append(
            RawReview(
                app_id=app_id,
                review_id=f"rev_pre_{i}",
                review_text=f"Pre patch review {i}",
                voted_up=(i < 6),  # 6/10 = 60% positive
                review_created_at=p_ts - int(timedelta(days=5).total_seconds()),
            )
        )
    # Post-patch (40 days ago): 90% positive
    for i in range(10):
        reviews_list.append(
            RawReview(
                app_id=app_id,
                review_id=f"rev_post_{i}",
                review_text=f"Post patch review {i}",
                voted_up=(i < 9),  # 9/10 = 90% positive
                review_created_at=p_ts + int(timedelta(days=5).total_seconds()),
            )
        )
    session.add_all(reviews_list)

    # 3. Raw Player Snapshots (CCU telemetry)
    snap_pre = RawPlayerSnapshot(
        app_id=app_id,
        player_count=1200,
        snapshot_at=patch_45d_ago - timedelta(days=5),
    )
    snap_post = RawPlayerSnapshot(
        app_id=app_id,
        player_count=1800,
        snapshot_at=patch_45d_ago + timedelta(days=5),
    )
    session.add_all([snap_pre, snap_post])

    # 4. Feature Complaints
    comp1 = FeatureReviewComplaint(
        app_id=app_id,
        category="Crashing on Match Start",
        volume_pct=Decimal("40.00"),
        severity="high",
        representative_snippets=["Crashes as soon as match begins."],
    )
    comp2 = FeatureReviewComplaint(
        app_id=app_id,
        category="Weapon Balance",
        volume_pct=Decimal("15.00"),
        severity="medium",
        representative_snippets=["Sniper rifle is overpowered."],
    )
    session.add_all([comp1, comp2])
    await session.commit()

    return app_id


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_materialize_update_impact_job(db_session: AsyncSession) -> None:
    """Test full execution of MaterializeUpdateImpactJob and verify window metrics."""
    app_id = await _seed_update_impact_data(db_session)

    job = MaterializeUpdateImpactJob(app_id=app_id, dry_run=False)
    result = await job._run_async(session=db_session)

    assert result["update_rows"] >= 1
    assert result["games_processed"] == 1

    # Query mart_update_impact
    records_res = await db_session.execute(
        select(MartUpdateImpact)
        .where(MartUpdateImpact.app_id == app_id)
        .order_by(MartUpdateImpact.patch_date.desc())
    )
    records = records_res.scalars().all()
    assert len(records) >= 1

    r = records[0]
    assert r.app_id == app_id
    assert r.window_days == 14
    assert r.pre_sentiment_positive_pct is not None
    assert r.post_sentiment_positive_pct is not None
    assert r.sentiment_delta_pct is not None
    assert r.observed_sentiment_verdict in [
        "Observed Positive Reception",
        "Observed Mixed / Neutral Impact",
        "Observed Player Backlash",
    ]
    # Verify non-causal branding
    assert "observed" in r.correlation_summary.lower()
    assert "correlation" in r.correlation_summary.lower()
    assert "sole causality" in r.correlation_summary.lower()

    # CCU telemetry
    assert r.pre_avg_ccu is not None
    assert r.post_avg_ccu is not None
    assert r.ccu_change_pct is not None


@pytest.mark.anyio
async def test_materialize_update_impact_idempotency(db_session: AsyncSession) -> None:
    """Verify that running MaterializeUpdateImpactJob multiple times produces identical clean rows."""
    app_id = await _seed_update_impact_data(db_session)

    job = MaterializeUpdateImpactJob(app_id=app_id, dry_run=False)
    res1 = await job._run_async(session=db_session)
    res2 = await job._run_async(session=db_session)

    assert res1["update_rows"] == res2["update_rows"]

    count = (
        await db_session.execute(
            select(MartUpdateImpact).where(MartUpdateImpact.app_id == app_id)
        )
    ).scalars().all()
    assert len(count) == res1["update_rows"]


@pytest.mark.anyio
async def test_get_updates_api_endpoint(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test GET /api/v1/games/{app_id}/updates endpoint."""
    app_id = await _seed_update_impact_data(db_session)

    # Populate marts and update impact
    mart_job = MaterializeMartsJob(app_id=app_id, dry_run=False)
    await mart_job._run_async(session=db_session)

    impact_job = MaterializeUpdateImpactJob(app_id=app_id, dry_run=False)
    await impact_job._run_async(session=db_session)

    resp = await client.get(f"/api/v1/games/{app_id}/updates")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["app_id"] == app_id
    assert data["total_updates"] >= 1
    assert data["latest_verdict"] is not None
    assert len(data["updates"]) == data["total_updates"]

    first_update = data["updates"][0]
    assert "correlation_summary" in first_update
    assert "observed_sentiment_verdict" in first_update
    assert first_update["window_days"] == 14
    assert "is_inferred" in first_update


@pytest.mark.anyio
async def test_get_updates_empty_honest_degradation(client: AsyncClient, db_session: AsyncSession) -> None:
    """Verify honest degradation when a game has no recorded patch events."""
    app_id = 90002
    game = RawGame(
        app_id=app_id,
        name="Untracked Indie Game",
        review_score=7,
        final_price_usd=Decimal("9.99"),
        genres=[{"id": "1", "description": "Action"}],
    )
    db_session.add(game)
    await db_session.commit()

    mart_job = MaterializeMartsJob(app_id=app_id, dry_run=False)
    await mart_job._run_async(session=db_session)

    resp = await client.get(f"/api/v1/games/{app_id}/updates")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total_updates"] == 0
    assert body["data"]["updates"] == []
    assert body["data"]["latest_verdict"] is None


@pytest.mark.anyio
async def test_get_updates_not_found(client: AsyncClient) -> None:
    """Test 404 behavior for unknown game."""
    resp = await client.get("/api/v1/games/99999999/updates")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "game_not_found"


@pytest.mark.anyio
async def test_golden_rule_no_raw_reads_in_updates_api() -> None:
    """
    Permanent Architectural Guardrail:
    Assert api/updates.py does not import or query RawGame, RawReview, or raw_*/feature_* tables.
    """
    source = inspect.getsource(updates)
    assert "RawGame" not in source, "Golden Rule Violation: api/updates.py references RawGame!"
    assert "RawReview" not in source, "Golden Rule Violation: api/updates.py references RawReview!"
    assert "raw_reviews" not in source, "Golden Rule Violation: api/updates.py references raw_reviews!"
    assert "raw_player_snapshots" not in source, "Golden Rule Violation: api/updates.py references raw_player_snapshots!"
    assert "FeatureReview" not in source, "Golden Rule Violation: api/updates.py references FeatureReview*!"
    assert "MartUpdateImpact" in source
    assert "MartGameOverview" in source


@pytest.mark.anyio
async def test_real_patch_notes_take_priority_over_inferred_surge(db_session: AsyncSession) -> None:
    """Verify that authentic raw_patch_notes take precedence over inferred surges with is_inferred=False."""
    app_id = await _seed_update_impact_data(db_session)
    now_dt = datetime.now(UTC)

    # Insert authentic developer patch note 45 days ago
    real_note = RawPatchNote(
        app_id=app_id,
        gid="steam_announcement_101",
        title="Major Tactical Update v1.2 — Systems Overhaul",
        url="https://store.steampowered.com/news/app/80001/view/101",
        author="Matrix Studio Dev",
        contents="Full balance overhaul notes and server improvements.",
        feedlabel="Community Announcements",
        published_at=now_dt - timedelta(days=45),
    )
    db_session.add(real_note)
    await db_session.commit()

    job = MaterializeUpdateImpactJob(app_id=app_id, dry_run=False)
    await job._run_async(session=db_session)

    records = (
        await db_session.execute(
            select(MartUpdateImpact).where(MartUpdateImpact.app_id == app_id)
        )
    ).scalars().all()

    assert len(records) >= 1
    top_record = records[0]
    assert top_record.patch_name == "Major Tactical Update v1.2 — Systems Overhaul"
    assert top_record.is_inferred is False


@pytest.mark.anyio
async def test_ingest_news_job_upsert(db_session: AsyncSession) -> None:
    """Verify IngestNewsJob fetches and upserts Steam news items into raw_patch_notes."""
    app_id = 85001
    game = RawGame(
        app_id=app_id,
        name="News Test Game",
        review_score=9,
        final_price_usd=Decimal("29.99"),
        genres=[{"id": "1", "description": "RPG"}],
    )
    db_session.add(game)
    await db_session.commit()

    sample_news_items = [
        {
            "gid": "test_gid_999",
            "title": "Season 2 Expansion Launch Notes",
            "url": "https://store.steampowered.com/news/sample",
            "author": "RPG Team",
            "contents": "Patch details and new quest lines.",
            "feedlabel": "Community Announcements",
            "date": 1720000000,
        }
    ]

    job = IngestNewsJob(app_id=app_id, dry_run=False)
    count = await job._upsert_patch_notes(db_session, app_id, sample_news_items)
    await db_session.commit()

    assert count == 1
    saved = (
        await db_session.execute(
            select(RawPatchNote).where(RawPatchNote.gid == "test_gid_999")
        )
    ).scalar_one_or_none()
    assert saved is not None
    assert saved.title == "Season 2 Expansion Launch Notes"
    assert saved.app_id == app_id


