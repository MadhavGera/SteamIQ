"""
Tests for provenance metadata, serving latency/SHAP verification, and model registry flags.
Covers requirements 1, 2, and 3 from the verification review.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from core.app import create_app
from db.base import AsyncSessionLocal
from recsys.inference import MODEL_ALIASES, RecommenderService


@pytest.mark.asyncio
async def test_provenance_metadata_non_null_and_positive() -> None:
    """
    Item 1: service.recommend must return the real model_run_id and the registered val NDCG@10
    for every model, including v2. Must fail on null or 0.0.
    """
    real_data_dir = Path("backend/data/recsys")
    if not (real_data_dir / "data_manifest.json").exists():
        real_data_dir = Path("data/recsys")
    RecommenderService._instance = None
    service = RecommenderService.get_instance(data_dir=real_data_dir)
    service.initialize()

    assert service.is_initialized, "RecommenderService must be initialized"
    assert len(service.models) > 0, "RecommenderService must have loaded models"

    # Test games (Counter-Strike, Team Fortress Classic, Day of Defeat)
    seed_games = [10, 20, 30]

    for model_name in list(service.models.keys()):
        res = service.recommend(liked_app_ids=seed_games, model=model_name, k=5, explain=False)

        # 1. model_run_id must be non-null, non-empty string
        model_run_id = res.get("model_run_id")
        assert model_run_id is not None, f"model_run_id is None for model '{model_name}'"
        assert str(model_run_id).strip() != "", f"model_run_id is empty string for model '{model_name}'"

        # 2. val_ndcg_at_10 must be float > 0.0
        val_ndcg = res.get("val_ndcg_at_10")
        assert val_ndcg is not None, f"val_ndcg_at_10 is None for model '{model_name}'"
        assert float(val_ndcg) > 0.0, f"val_ndcg_at_10 is {val_ndcg} (<= 0.0) for model '{model_name}'"


def test_latency_resolved_model_and_shap_assertion() -> None:
    """
    Item 2: Benchmark verifies resolved model name, and that explain=True computes SHAP
    (assert took_ms > 20 for ranker).
    """
    real_data_dir = Path("backend/data/recsys")
    if not (real_data_dir / "data_manifest.json").exists():
        real_data_dir = Path("data/recsys")
    service = RecommenderService.get_instance(data_dir=real_data_dir)
    if not service.is_initialized:
        service.initialize()

    # 1. Test model name resolution
    assert service.resolve_model_name("two_tower") == "recsys_two_tower_clean"
    assert service.resolve_model_name("ranker") == "recsys_two_tower_ranker_v2_trainstats"
    assert service.resolve_model_name("recsys_popularity") == "recsys_popularity"

    # Verify old aliases do not silently remap to v2
    assert "ranker_clean" not in MODEL_ALIASES or MODEL_ALIASES["ranker_clean"] != "recsys_two_tower_ranker_clean_v2"
    assert "two_tower_ranker_clean" not in MODEL_ALIASES or MODEL_ALIASES["two_tower_ranker_clean"] != "recsys_two_tower_ranker_clean_v2"

    # 2. Test ranker with explain=True computes SHAP (took_ms > 20)
    ranker_key = (
        "recsys_two_tower_ranker_v2_trainstats"
        if "recsys_two_tower_ranker_v2_trainstats" in service.models
        else (
            "recsys_two_tower_ranker_clean_v2"
            if "recsys_two_tower_ranker_clean_v2" in service.models
            else "recsys_two_tower_ranker_clean"
        )
    )
    if ranker_key in service.models:
        rec_res = service.recommend(liked_app_ids=[10, 20, 30], model=ranker_key, k=10, explain=True)
        assert rec_res["model_name"] == ranker_key
        took_ms = rec_res["took_ms"]
        assert took_ms > 20.0, f"Ranker with explain=True took {took_ms} ms <= 20 ms (did SHAP compute?)"

        # Check that reasons do NOT include review/popularity concepts
        for item in rec_res["items"]:
            for reason in item.get("reasons", []):
                assert "positive reviews" not in reason.lower(), f"Review concept leaked in reason: {reason}"
                assert "community favorite" not in reason.lower(), f"Popularity concept leaked in reason: {reason}"

            # Check that anchors have cosine similarity >= 0.5
            for anchor in item.get("because_you_liked", []):
                assert anchor["similarity"] >= 0.5, f"Anchor similarity {anchor['similarity']} < 0.5"


@pytest.mark.asyncio
async def test_models_registry_api_flags_and_candidates() -> None:
    """
    Item 3: GET /api/v1/recsys/models sets deployable/leak_free/clean flags on v2 explicitly,
    marks old ranker non-deployable, and ensures no candidate runs have promoted_* set.
    """
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/recsys/models")
        assert resp.status_code == 200, f"GET /models returned {resp.status_code}: {resp.text}"
        payload = resp.json()
        assert payload["success"] is True
        models_data = payload["data"]

        model_dict = {m["model_name"]: m for m in models_data}

        # Verify old ranker is non-deployable
        if "recsys_two_tower_ranker_clean" in model_dict:
            old_r = model_dict["recsys_two_tower_ranker_clean"]
            assert old_r["deployable"] is False, "Old ranker must have deployable=False"
            assert old_r["leak_free"] is False, "Old ranker must have leak_free=False"
            assert old_r["clean"] is False, "Old ranker must have clean=False"

        # Verify v2 has explicit reference, non-deployable, leak_free=False, clean=False flags
        if "recsys_two_tower_ranker_clean_v2" in model_dict:
            v2_r = model_dict["recsys_two_tower_ranker_clean_v2"]
            assert v2_r["deployable"] is False, "v2 ranker must have deployable=False (reference)"
            assert v2_r["leak_free"] is False, "v2 ranker must have leak_free=False (reference)"
            assert v2_r["clean"] is False, "v2 ranker must have clean=False (reference)"
            assert v2_r["stage"] == "reference", "v2 ranker stage must be reference"

        # Verify trainstats is marked as the leak-free ranker candidate
        if "recsys_two_tower_ranker_v2_trainstats" in model_dict:
            ts_r = model_dict["recsys_two_tower_ranker_v2_trainstats"]
            assert ts_r["deployable"] is True, "trainstats ranker must have deployable=True"
            assert ts_r["leak_free"] is True, "trainstats ranker must have leak_free=True"
            assert ts_r["clean"] is True, "trainstats ranker must have clean=True"
            assert ts_r["stage"] == "candidate", "trainstats ranker stage must be candidate"

        # Verify clean TT has explicit flags
        if "recsys_two_tower_clean" in model_dict:
            tt_r = model_dict["recsys_two_tower_clean"]
            assert tt_r["deployable"] is True
            assert tt_r["leak_free"] is True
            assert tt_r["clean"] is True

    # Verify database has zero promoted_* columns on candidate stage runs
    async with AsyncSessionLocal() as session:
        check_stmt = text(
            "SELECT count(*) FROM model_runs WHERE stage = 'candidate' AND (promoted_at IS NOT NULL OR promoted_by IS NOT NULL)"
        )
        res = await session.execute(check_stmt)
        cand_with_promoted = res.scalar() or 0
        assert cand_with_promoted == 0, f"Found {cand_with_promoted} candidate runs with promoted_* non-null"
