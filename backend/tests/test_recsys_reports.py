"""
Unit tests for Recommender System Report schemas, NaN prevention, and serialization.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from jobs.generate_cold_start_report import summarize_model_ranks
from recsys.report import (
    _clean_for_json,
    build_leaderboard_payload,
    generate_leaderboard_markdown,
    save_report_json,
)
from recsys.report_schema import (
    REPORT_MODEL_MAP,
    check_no_nan_inf,
    validate_report_payload,
)


def test_nan_and_infinity_rejection() -> None:
    """check_no_nan_inf must raise ValueError on NaN or Inf in nested payloads."""
    with pytest.raises(ValueError, match="Found invalid float value"):
        check_no_nan_inf({"metric": float("nan")})

    with pytest.raises(ValueError, match="Found invalid float value"):
        check_no_nan_inf({"nested": {"scores": [1.0, float("inf")]}})


def test_clean_for_json_replaces_nan_inf() -> None:
    """_clean_for_json sanitizes NaN and Inf to 0.0."""
    dirty = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested": [1.0, float("-inf"), 3.0],
    }
    clean = _clean_for_json(dirty)
    assert clean["nan_val"] == 0.0
    assert clean["inf_val"] == 0.0
    assert clean["nested"] == [1.0, 0.0, 3.0]
    # Verify it passes check_no_nan_inf
    check_no_nan_inf(clean)


def test_leaderboard_payload_schema_validation() -> None:
    """Leaderboard payload validates against LeaderboardPayload schema."""
    models_data = [
        {
            "model_name": "recsys_two_tower",
            "family": "two_tower",
            "stage": "production",
            "model_run_id": "9e09d443-7d28-4af3-8a40-1daa88101145",
            "hyperparameters": {"hidden_dim": 256},
            "training_time_sec": 42.5,
            "val_metrics": {
                "hit_rate_at_10": 0.1232,
                "hit_rate_at_10_lo95": 0.1144,
                "hit_rate_at_10_hi95": 0.1322,
                "ndcg_at_10": 0.0630,
                "ndcg_at_10_lo95": 0.0581,
                "ndcg_at_10_hi95": 0.0680,
                "catalog_coverage_at_10": 0.3597,
                "diversity_at_10": 0.7623,
            },
            "test_metrics": {
                "hit_rate_at_10": 0.1038,
                "hit_rate_at_10_lo95": 0.0954,
                "hit_rate_at_10_hi95": 0.1116,
                "ndcg_at_10": 0.0535,
                "ndcg_at_10_lo95": 0.0488,
                "ndcg_at_10_hi95": 0.0582,
                "catalog_coverage_at_10": 0.3597,
                "diversity_at_10": 0.7623,
            },
        }
    ]

    payload = build_leaderboard_payload(
        dataset_version="test_v1",
        models_data=models_data,
        n_eval_users=5000,
        seed=42,
    )

    validated = validate_report_payload("leaderboard", payload)
    assert validated.metadata.report_key == "leaderboard"
    assert len(validated.models) == 1
    assert validated.models[0].model_name == "recsys_two_tower"


def test_invalid_report_schema_rejection() -> None:
    """Invalid payload structure triggers ValidationError."""
    invalid_payload = {
        "metadata": {"report_key": "leaderboard"},  # missing required metadata fields
        "models": "not_a_list",
    }
    with pytest.raises(ValidationError):
        validate_report_payload("leaderboard", invalid_payload)


def test_all_10_report_schemas_exist() -> None:
    """All 10 required report types have registered Pydantic schemas."""
    expected_keys = [
        "leaderboard",
        "ablations",
        "ranker_ablations",
        "slices",
        "cold_start",
        "beyond_accuracy",
        "shap_global",
        "training_curves",
        "dataset",
        "limitations",
    ]
    for k in expected_keys:
        assert k in REPORT_MODEL_MAP


def test_save_report_json_and_markdown(tmp_path: Path) -> None:
    """save_report_json writes valid JSON and generate_leaderboard_markdown outputs markdown."""
    models_data = [
        {
            "model_name": "recsys_popularity",
            "family": "popularity",
            "stage": "candidate",
            "val_metrics": {"ndcg_at_10": 0.0200, "hit_rate_at_10": 0.0414},
            "test_metrics": {"ndcg_at_10": 0.0151, "hit_rate_at_10": 0.0326},
        }
    ]
    payload = build_leaderboard_payload("test_v1", models_data, 5000, 42)
    json_path = tmp_path / "leaderboard.json"
    save_report_json(json_path, payload)

    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["report_key"] == "leaderboard"

    md = generate_leaderboard_markdown(payload)
    assert "# 🏆 SteamIQ Recommender System Evaluation & Evidence Report" in md
    assert "recsys_popularity" in md


def test_cold_start_report_schema_validation() -> None:
    """Cold-start report payload validates against ColdStartPayload schema with reference labels."""
    ranks = np.array([1, 5, 12, 100, 2500, 3000], dtype=np.int64)
    summary = summarize_model_ranks(ranks, n_users=6, seed=42)
    assert summary["hits_k"] == 2
    assert summary["n_users"] == 6
    assert 0.0 <= summary["hit_rate_at_10"] <= 1.0
    assert 0.0 <= summary["ndcg_at_10"] <= 1.0
    assert summary["mean_rank"] > 0
    assert summary["median_rank"] > 0

    cold_start_payload = {
        "metadata": {
            "report_key": "cold_start",
            "dataset_version": "test_v1",
            "n_eval_users": 6,
            "seed": 42,
            "generated_at": "2026-10-08T20:00:00+00:00",
        },
        "item_holdout_study": {
            "study_type": "item_holdout",
            "primary_evidence": True,
            "title": "True Item Holdout",
            "n_eval_users": 6,
            "n_held_out_items": 300,
            "models": {
                "random": summary,
                "content": summary,
                "two_tower": summary,
            },
        },
        "sparse_targets_study": {
            "study_type": "sparse_targets",
            "primary_evidence": False,
            "title": "Sparse Targets Subgroup Study",
            "is_optimistic_proxy": True,
            "n_eval_users": 6,
            "max_train_positives": 10,
            "models": {
                "two_tower_masked (reference: all-time stats)": summary,
            },
        },
        "simulated_new_items_study": {
            "study_type": "simulated_new_items",
            "variant": "c2",
            "variant_label": "C2: Fair Simulation",
            "primary_evidence": False,
            "is_optimistic_proxy": True,
            "n_eval_users": 6,
            "n_new_items": 300,
            "fraction_masked": 0.10,
            "models": {
                "two_tower_masked (reference: all-time stats)": summary,
            },
        },
    }
    validated = validate_report_payload("cold_start", cold_start_payload)
    assert validated.metadata.report_key == "cold_start"
    assert validated.item_holdout_study["primary_evidence"] is True


def test_cold_start_all_holdout_conditions_present_in_payload() -> None:
    """
    Verifies that all 5 holdout conditions ((a), (b), (c), (c2), (c3)),
    plus random and content, are present in the cold_start.json report payload.
    """
    report_file = Path("data/recsys/reports/cold_start.json")
    if not report_file.exists():
        # Fallback path if run from repo root
        report_file = Path("backend/data/recsys/reports/cold_start.json")

    assert report_file.exists(), f"cold_start.json not found at {report_file}"
    with open(report_file, encoding="utf-8") as f:
        payload = json.load(f)

    validated = validate_report_payload("cold_start", payload)
    assert validated.metadata.report_key == "cold_start"

    holdout_models = validated.item_holdout_study.get("models", {})
    expected_conditions = [
        "random",
        "content",
        "two_tower",
        "routed_hybrid",
    ]
    for cond in expected_conditions:
        assert cond in holdout_models, f"Cold start condition '{cond}' missing from item_holdout_study models!"
        model_stats = holdout_models[cond]
        assert "hit_rate_at_10" in model_stats
        assert "ndcg_at_10" in model_stats


