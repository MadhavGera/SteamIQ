"""
Unit tests for Dev Safety, Gating, and RecSys Report Fallback (Phase 3).
"""
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch
from pathlib import Path

from core.app import create_app
from core.config import settings
from core.errors import ReportsMissingError
from jobs.generate_reports import GenerateReportsJob


@pytest.mark.asyncio
async def test_generate_reports_raises_409_when_missing_and_no_experiments(tmp_path):
    """
    GenerateReportsJob._run_async must raise ReportsMissingError when required report
    files are missing and allow_experiments is False.
    """
    job = GenerateReportsJob(data_dir=tmp_path, allow_experiments=False)
    
    with pytest.raises(ReportsMissingError) as exc_info:
        await job._run_async(session=None, allow_experiments=False)
    
    err = exc_info.value
    assert err.status_code == 409
    assert err.code == "missing_reports"
    assert "missing_keys" in err.details
    assert "leaderboard" in err.details["missing_keys"]


@pytest.mark.asyncio
async def test_materialize_reports_returns_403_when_enable_ops_write_false(client: AsyncClient, monkeypatch):
    """
    POST /api/v1/ops/materialize-reports must return 403 when settings.enable_ops_write is False.
    """
    monkeypatch.setattr(settings, "enable_ops_write", False)
    
    response = await client.post("/api/v1/ops/materialize-reports")
    assert response.status_code == 403
    data = response.json()
    assert "error" in data or "detail" in data
    error_info = data.get("error", {})
    assert error_info.get("code") == "forbidden" or "disabled" in str(data).lower()


@pytest.mark.asyncio
async def test_recsys_report_rejects_path_traversal(client: AsyncClient):
    """
    GET /api/v1/recsys/reports/{key} must return 404 when key is not in ALLOWED_REPORT_KEYS.
    """
    # Attempt directory traversal or unauthorized keys
    traversal_keys = [
        "..%2F..%2Fetc%2Fpasswd",
        "secret",
        "../../docs/results/leaderboard",
        "arbitrary_file",
    ]
    for key in traversal_keys:
        response = await client.get(f"/api/v1/recsys/reports/{key}")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] in ("report_not_found", "not_found")


@pytest.mark.asyncio
async def test_recsys_report_fallback_source(client: AsyncClient):
    """
    GET /api/v1/recsys/reports/leaderboard must return source='file_fallback'
    when not present in SQLite database.
    """
    response = await client.get("/api/v1/recsys/reports/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    report_data = data["data"]
    assert report_data["report_key"] == "leaderboard"
    assert report_data["source"] in ("database", "file_fallback")
    assert "models" in report_data["payload"]


@pytest.mark.asyncio
async def test_ops_evidence_whitelisting_and_retrieval(client: AsyncClient):
    """
    GET /api/v1/ops/evidence/{name} must return 200 for whitelisted names
    and 404 for arbitrary unwhitelisted keys.
    """
    # 404 on unwhitelisted name
    res_bad = await client.get("/api/v1/ops/evidence/unwhitelisted_key")
    assert res_bad.status_code == 404

    # 200 on whitelisted name
    for valid_name in ["retrain_gate", "drift_detector_power", "monitor_models"]:
        res_ok = await client.get(f"/api/v1/ops/evidence/{valid_name}")
        assert res_ok.status_code == 200
        data = res_ok.json()
        assert data["success"] is True
        ev = data["data"]
        assert ev["name"] == valid_name
        assert "log" in ev
        assert "data" in ev


@pytest.mark.asyncio
async def test_ops_model_cards_retrieval(client: AsyncClient):
    """
    GET /api/v1/ops/model-cards must return a list of model cards with metadata and markdown content.
    """
    res = await client.get("/api/v1/ops/model-cards")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    cards = data["data"]
    assert isinstance(cards, list)
    assert len(cards) >= 3
    card_names = {c["model_name"] for c in cards}
    assert "recsys_two_tower_clean" in card_names
    for c in cards:
        assert "filename" in c
        assert "title" in c
        assert "content" in c
        assert len(c["content"]) > 50

