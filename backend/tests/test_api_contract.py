"""
API Contract Verification Test Suite (Phase 6).

Asserts that every API endpoint path called by the Next.js frontend (in frontend/lib/api.ts)
has a corresponding route registered in the FastAPI backend application.
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest
from httpx import AsyncClient

from core.app import create_app


def normalize_route_path(path: str) -> str:
    """
    Normalizes path parameters (e.g., {app_id}, {appId}, ${appId}, {name}, {key})
    into a canonical {param} token for agnostic schema matching.
    """
    # Strip query parameters
    clean = path.split("?")[0].strip()
    # Normalize template literal interpolation like ${appId} -> {param}
    clean = re.sub(r"\$\{[^}]+\}", "{param}", clean)
    # Normalize FastAPI path parameters like {app_id} -> {param}
    clean = re.sub(r"\{[^}]+\}", "{param}", clean)
    return clean


def extract_frontend_api_endpoints() -> list[tuple[str, str]]:
    """
    Parses frontend/lib/api.ts and extracts all backend route strings along with their HTTP methods.
    """
    api_ts_path = Path("frontend/lib/api.ts")
    if not api_ts_path.exists():
        api_ts_path = Path("../frontend/lib/api.ts")

    assert api_ts_path.exists(), f"Could not find api.ts at {api_ts_path}"

    content = api_ts_path.read_text(encoding="utf-8")
    
    # Matches patterns like apiFetch<...>(`/api/v1/...` or "/api/v1/..." or fetch(`${baseUrl}/api/v1/...`)
    endpoint_pattern = re.compile(
        r"""(?:apiFetch<[^>]*>\(\s*|fetch\(`\$\{baseUrl\}\s*)(["'`])(/api/v1/[^"'`\?]+|/health)(?:[^"'`]*)?\1""",
        re.MULTILINE
    )

    found_endpoints = []
    for match in endpoint_pattern.finditer(content):
        path = match.group(2)
        found_endpoints.append(path)

    return sorted(list(set(found_endpoints)))


def test_all_frontend_endpoints_exist_in_fastapi_app():
    """
    Contract invariant: Every API endpoint called by the frontend must exist in FastAPI's route table.
    """
    app = create_app()

    # Extract all registered backend routes (normalized)
    backend_routes = set()
    for route in app.routes:
        if hasattr(route, "path"):
            backend_routes.add(normalize_route_path(route.path))

    # Extract all frontend invoked routes (normalized)
    frontend_endpoints = extract_frontend_api_endpoints()
    assert len(frontend_endpoints) >= 15, f"Expected at least 15 frontend API routes, found {len(frontend_endpoints)}"

    unmatched = []
    for fe_path in frontend_endpoints:
        norm_fe = normalize_route_path(fe_path)
        if norm_fe not in backend_routes:
            unmatched.append((fe_path, norm_fe))

    assert not unmatched, (
        f"Frontend calls {len(unmatched)} endpoints not registered in FastAPI backend:\n"
        + "\n".join(f"  {raw} -> normalized: {norm}" for raw, norm in unmatched)
    )


@pytest.mark.asyncio
async def test_essential_endpoints_contract_status(client: AsyncClient):
    """
    Smoke test: Core contract endpoints respond with expected status codes without unhandled exceptions.
    """
    # 1. Health check
    res_health = await client.get("/health")
    assert res_health.status_code == 200

    # 2. Recsys Models
    res_models = await client.get("/api/v1/recsys/models")
    assert res_models.status_code == 200

    # 3. Recsys Map
    res_map = await client.get("/api/v1/recsys/map")
    assert res_map.status_code == 200

    # 4. Ops Metrics
    res_metrics = await client.get("/api/v1/ops/metrics")
    assert res_metrics.status_code == 200

    # 5. Ops Drift
    res_drift = await client.get("/api/v1/ops/drift")
    assert res_drift.status_code == 200

    # 6. Ops Model Cards
    res_cards = await client.get("/api/v1/ops/model-cards")
    assert res_cards.status_code == 200

    # 7. Ops Evidence
    res_ev = await client.get("/api/v1/ops/evidence/retrain_gate")
    assert res_ev.status_code == 200
