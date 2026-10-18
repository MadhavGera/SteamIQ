"""
Unit and Artifact-Level Tests for Multi-Seed Aggregation & Integrity (Issue I2).

Verifies:
1. Known synthetic inputs produce exact expected mean and std.
2. Deterministic baselines report std as 'n/a' instead of 0.0000.
3. Artifact-level test fails if a multi-seed mean equals the champion value on all metrics while per-seed values differ.
4. Validation sweep artifact conforms to strict multi-seed aggregation integrity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recsys.aggregation import (
    aggregate_per_seed_metrics,
    verify_aggregation_integrity,
)


def test_aggregate_known_inputs() -> None:
    per_seed = [
        {"seed": 42, "ndcg_at_10": 0.060, "hit_rate_at_10": 0.120, "coverage_at_10": 0.40},
        {"seed": 43, "ndcg_at_10": 0.070, "hit_rate_at_10": 0.130, "coverage_at_10": 0.45},
        {"seed": 44, "ndcg_at_10": 0.080, "hit_rate_at_10": 0.140, "coverage_at_10": 0.50},
    ]

    agg = aggregate_per_seed_metrics(per_seed, deterministic=False)

    assert agg["n_seeds"] == 3
    assert agg["seeds"] == [42, 43, 44]
    assert agg["deterministic"] is False

    # Known means
    assert np.isclose(agg["means"]["ndcg_at_10"], 0.070, atol=1e-6)
    assert np.isclose(agg["means"]["hit_rate_at_10"], 0.130, atol=1e-6)
    assert np.isclose(agg["means"]["coverage_at_10"], 0.450, atol=1e-6)

    # Known stds
    expected_ndcg_std = float(np.std([0.060, 0.070, 0.080]))
    assert np.isclose(agg["stds"]["ndcg_at_10"], expected_ndcg_std, atol=1e-6)

    # Formatted strings
    assert agg["summary"]["ndcg_at_10"]["formatted"] == f"0.0700 ± {expected_ndcg_std:.4f}"


def test_deterministic_baseline_reports_na_std() -> None:
    per_seed = [
        {"seed": 42, "ndcg_at_10": 0.055286, "hit_rate_at_10": 0.1014},
        {"seed": 43, "ndcg_at_10": 0.055286, "hit_rate_at_10": 0.1014},
        {"seed": 44, "ndcg_at_10": 0.055286, "hit_rate_at_10": 0.1014},
    ]

    agg = aggregate_per_seed_metrics(per_seed, deterministic=True)

    assert agg["deterministic"] is True
    assert agg["stds"]["ndcg_at_10"] == "n/a"
    assert agg["stds"]["hit_rate_at_10"] == "n/a"
    assert agg["summary"]["ndcg_at_10"]["formatted"] == "0.0553 (n/a)"
    assert agg["summary"]["hit_rate_at_10"]["formatted"] == "0.1014 (n/a)"


def test_copy_paste_detection_fails_when_champion_copied() -> None:
    # Seeds differ in reality:
    per_seed = [
        {"seed": 42, "ndcg_at_10": 0.040, "hit_rate_at_10": 0.080, "coverage_at_10": 0.10},
        {"seed": 43, "ndcg_at_10": 0.050, "hit_rate_at_10": 0.100, "coverage_at_10": 0.20},
    ]

    champion_entry = {
        "seed": 42,
        "ndcg_at_10": 0.040,
        "hit_rate_at_10": 0.080,
        "coverage_at_10": 0.10,
    }

    # Malicious or bug-ridden aggregate where champion values were copied as means
    fraudulent_agg = {
        "deterministic": False,
        "means": {
            "ndcg_at_10": 0.040,  # Copied champion value instead of true mean 0.045
            "hit_rate_at_10": 0.080,
            "coverage_at_10": 0.10,
        },
        "stds": {
            "ndcg_at_10": 0.005,
            "hit_rate_at_10": 0.010,
            "coverage_at_10": 0.05,
        },
    }

    with pytest.raises(ValueError, match="Aggregation math error|Suspicious aggregation detected"):
        verify_aggregation_integrity(per_seed, champion_entry, fraudulent_agg)


def test_validation_sweep_artifact_integrity() -> None:
    sweep_path = Path(__file__).resolve().parent.parent.parent / "docs" / "results" / "validation_sweep.json"
    assert sweep_path.exists(), f"validation_sweep.json not found at {sweep_path}"

    with open(sweep_path, encoding="utf-8") as f:
        sweep_data = json.load(f)

    tt_section = sweep_data.get("two_tower_clean_5seeds", {})
    seed_results = tt_section.get("seed_results", {})
    assert len(seed_results) == 5, f"Expected 5 seeds, got {len(seed_results)}"

    per_seed = [{"seed": int(s), **m} for s, m in seed_results.items()]
    champ_seed = str(tt_section.get("champion_seed", 44))
    champ_metrics = seed_results[champ_seed]

    agg = aggregate_per_seed_metrics(per_seed, deterministic=False)

    # Verify that multi-seed mean does NOT identically equal champion on all metrics
    verify_aggregation_integrity(per_seed, champ_metrics, agg)

    # Check that reported mean matches computed mean to 5 decimal places
    reported_mean_ndcg = tt_section.get("mean_ndcg_at_10")
    computed_mean_ndcg = agg["means"]["val_ndcg_at_10"]
    assert np.isclose(reported_mean_ndcg, computed_mean_ndcg, atol=1e-5)

    # Check ranker section as well
    rk_section = sweep_data.get("ranker_clean_5seeds", {})
    rk_seed_results = rk_section.get("seed_results", {})
    assert len(rk_seed_results) == 5

    rk_per_seed = [{"seed": int(s), **m} for s, m in rk_seed_results.items()]
    rk_champ_seed = str(rk_section.get("champion_seed", 44))
    rk_champ_metrics = rk_seed_results[rk_champ_seed]

    rk_agg = aggregate_per_seed_metrics(rk_per_seed, deterministic=False)
    verify_aggregation_integrity(rk_per_seed, rk_champ_metrics, rk_agg)

    reported_rk_mean = rk_section.get("mean_ndcg_at_10")
    computed_rk_mean = rk_agg["means"]["val_ndcg_at_10"]
    assert np.isclose(reported_rk_mean, computed_rk_mean, atol=1e-5)
