"""
Tests for the games API endpoints (backed by mart_game_overview in Phase 5).
"""
from __future__ import annotations

from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MartGameOverview, RawGame

# ─── Fixtures ─────────────────────────────────────────────────────────────────

async def _insert_game(session: AsyncSession, **overrides) -> MartGameOverview:
    """Helper: insert a minimal RawGame + MartGameOverview for testing."""
    defaults = {
        "app_id": 1145360,
        "name": "Hollow Knight",
        "short_description": "A challenging 2D action adventure.",
        "developer": "Team Cherry",
        "publisher": "Team Cherry",
        "positive_reviews": 150000,
        "negative_reviews": 3000,
        "is_free": False,
        "final_price_usd": "14.99",
        "discount_pct": 0,
        "platform_windows": True,
    }
    defaults.update(overrides)
    game = RawGame(**defaults)
    session.add(game)
    await session.commit()
    await session.refresh(game)

    overview = MartGameOverview(
        app_id=game.app_id,
        name=game.name,
        short_description=game.short_description,
        developer=game.developer,
        publisher=game.publisher,
        positive_reviews=game.positive_reviews,
        negative_reviews=game.negative_reviews,
        is_free=game.is_free,
        final_price_usd=Decimal(str(game.final_price_usd)) if game.final_price_usd is not None else None,
        discount_pct=game.discount_pct,
        platform_windows=game.platform_windows,
        primary_genre="Metroidvania",
        revenue_tier="Tier 2: $10M–$50M (Major Hit)",
        success_score=Decimal("88.50"),
        net_sentiment_pct=Decimal("98.00"),
    )
    session.add(overview)
    await session.commit()
    await session.refresh(overview)
    return overview


# ─── Search tests ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_search_returns_game(client: AsyncClient, db_session: AsyncSession) -> None:
    await _insert_game(db_session)

    resp = await client.get("/api/v1/games/search", params={"q": "Hollow"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total"] >= 1
    assert body["data"]["games"][0]["name"] == "Hollow Knight"


@pytest.mark.anyio
async def test_search_empty_returns_zero(client: AsyncClient, db_session: AsyncSession) -> None:
    resp = await client.get("/api/v1/games/search", params={"q": "xyzzy_nonexistent_game"})
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] == 0
    assert resp.json()["data"]["games"] == []


@pytest.mark.anyio
async def test_search_missing_query_returns_422(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/games/search")
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"


@pytest.mark.anyio
async def test_search_pagination(client: AsyncClient, db_session: AsyncSession) -> None:
    # Insert 3 games
    for i in range(3):
        await _insert_game(db_session, app_id=9990 + i, name=f"Test Game {i}")

    resp = await client.get(
        "/api/v1/games/search", params={"q": "Test Game", "page": 1, "page_size": 2}
    )
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["games"]) <= 2
    assert body["meta"]["page"] == 1
    assert body["meta"]["page_size"] == 2


# ─── Detail tests ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_get_game_detail(client: AsyncClient, db_session: AsyncSession) -> None:
    await _insert_game(db_session)

    resp = await client.get("/api/v1/games/1145360")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert body["data"]["app_id"] == 1145360
    assert body["data"]["name"] == "Hollow Knight"
    assert body["data"]["developer"] == "Team Cherry"


@pytest.mark.anyio
async def test_get_game_not_found(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/games/99999999")
    assert resp.status_code == 404

    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "game_not_found"
    assert body["data"] is None


@pytest.mark.anyio
async def test_response_envelope_always_present(client: AsyncClient) -> None:
    """Verify ApiResponse envelope on both success and error responses."""
    resp = await client.get("/api/v1/games/99999999")
    body = resp.json()

    # All four fields must always be present
    assert "success" in body
    assert "data" in body
    assert "error" in body
    # meta may be None on errors
