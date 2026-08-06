"""
Tests for the /health endpoint.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert body["data"]["db"] == "connected"
    assert "version" in body["data"]
    assert body["meta"]["took_ms"] is not None


@pytest.mark.anyio
async def test_health_envelope_shape(client: AsyncClient) -> None:
    """Verify the ApiResponse envelope shape on a healthy response."""
    resp = await client.get("/health")
    body = resp.json()

    assert "success" in body
    assert "data" in body
    assert "error" in body
    assert "meta" in body
    assert body["error"] is None
