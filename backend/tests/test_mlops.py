"""
Unit and Integration Tests for MLOps Engine — Phase 7.

Validates:
  1. Statistical drift detection engine (PSI, KS, JSD).
  2. Bounded telemetry registry and privacy preservation invariants.
  3. ResponseCache hashing, TTL, and graceful degradation.
  4. Retraining gate pre-registered decision paths (PROMOTE, REJECT, INCONCLUSIVE).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.cache import ResponseCache
from jobs.retrain_gate import evaluate_gate_decision
from mlops.drift import (
    calculate_js_distance,
    calculate_ks,
    calculate_psi,
    evaluate_categorical_drift,
    evaluate_distribution_drift,
)
from mlops.telemetry import TelemetryRegistry


# ─── 1. Statistical Drift Detector Tests ──────────────────────────────────────


def test_psi_identical_distributions():
    """Identical reference and current distributions must yield PSI approximately 0.0."""
    np.random.seed(42)
    sample = np.random.normal(loc=10.0, scale=2.0, size=2000)
    psi = calculate_psi(sample, sample)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_shifted_distribution_triggers_alert():
    """Large distribution shift (e.g. +2 sigma) must produce PSI >= 0.20 (ALERT)."""
    np.random.seed(42)
    ref = np.random.normal(loc=10.0, scale=2.0, size=2000)
    curr = np.random.normal(loc=14.0, scale=2.0, size=2000)
    psi = calculate_psi(ref, curr)
    assert psi >= 0.20

    eval_res = evaluate_distribution_drift(ref, curr)
    assert eval_res["status"] == "ALERT"


def test_ks_test_p_value():
    """Two-sample KS test should pass for identical and reject for shifted."""
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=1000)
    curr_null = np.random.normal(loc=0.0, scale=1.0, size=1000)
    curr_shift = np.random.normal(loc=1.0, scale=1.0, size=1000)

    _, p_null = calculate_ks(ref, curr_null)
    assert p_null > 0.01  # Should not reject null at 0.01

    _, p_shift = calculate_ks(ref, curr_shift)
    assert p_shift < 1e-4  # Extreme significance


def test_jsd_categorical_drift():
    """Jensen-Shannon distance should be ~0 for identical categories and trigger on skew."""
    ref_counts = {"a": 100, "b": 100, "c": 100}
    curr_identical = {"a": 100, "b": 100, "c": 100}
    curr_shifted = {"a": 500, "b": 10, "c": 5}

    eval_id = evaluate_categorical_drift(ref_counts, curr_identical)
    assert eval_id["status"] == "PASS"
    assert eval_id["js_distance"] == pytest.approx(0.0, abs=1e-3)

    eval_shift = evaluate_categorical_drift(ref_counts, curr_shifted)
    assert eval_shift["status"] in ("WARN", "ALERT")
    assert eval_shift["js_distance"] > 0.15


# ─── 2. Telemetry & Privacy Preservation Tests ────────────────────────────────


def test_telemetry_registry_privacy_preservation():
    """
    Privacy Invariant: Telemetry registry MUST NOT store raw query strings
    or raw user library arrays. Only bounded aggregate samples and counts are kept.
    """
    reg = TelemetryRegistry()
    reg.record_recsys_request(
        model_name="recsys_two_tower_clean",
        took_ms=15.2,
        library_len=14,
        scores=[0.95, 0.82, 0.71],
        cached=False,
    )
    reg.record_assistant_request(
        intent="game_overview",
        confidence=0.98,
        took_ms=45.0,
        cached=False,
    )

    # Inspect all attributes of the registry
    for attr, val in reg.__dict__.items():
        assert "query" not in attr.lower()
        assert "liked_app" not in attr.lower()
        if isinstance(val, list):
            # Elements must strictly be numeric scalars
            for elem in val:
                assert isinstance(elem, (int, float, np.number))

    prom_text = reg.render_prometheus_text()
    assert "steamiq_recsys_requests_total" in prom_text
    assert "steamiq_assistant_requests_total" in prom_text
    assert "steamiq_cache_misses_total" in prom_text


def test_telemetry_registry_bounded_memory():
    """Telemetry buffers must not grow unboundedly under heavy traffic."""
    reg = TelemetryRegistry()
    reg.max_sample_buffer = 50

    for i in range(200):
        reg.record_recsys_request("test_model", took_ms=10.0, library_len=i + 1, scores=[float(i)])
        reg.record_assistant_request("overview", confidence=0.9, took_ms=15.0)

    assert len(reg.live_library_lengths) <= 50
    assert len(reg.live_recommendation_scores) <= 50
    assert len(reg.live_assistant_confidences) <= 50


# ─── 3. Response Cache Tests ──────────────────────────────────────────────────


def test_response_cache_key_generation():
    """Cache keys must incorporate model_run_id so model changes invalidate cache."""
    cache = ResponseCache()
    key1 = cache.make_recsys_key("run_1", [10, 20, 30], k=10, explain=False)
    key2 = cache.make_recsys_key("run_2", [10, 20, 30], k=10, explain=False)
    key3 = cache.make_recsys_key("run_1", [30, 20, 10], k=10, explain=False)  # same items, different order

    assert key1 != key2, "Different model runs must have distinct cache keys"
    assert key1 == key3, "Item order permutation must produce identical canonical key"


@pytest.mark.asyncio
async def test_response_cache_graceful_degradation():
    """When Redis is not reachable or disabled, cache operations must return None without raising."""
    cache = ResponseCache(redis_url="redis://localhost:9999/9")
    val = await cache.get("dummy_key")
    assert val is None
    # Setting in cache should safely degrade without crashing
    await cache.set("dummy_key", {"items": []})


# ─── 4. Retrain Gate Decision Paths ───────────────────────────────────────────


def test_retrain_gate_promote_decision():
    """
    PROMOTE path: Challenger out-performs champion by margin > seed std
    and satisfies non-inferiority.
    """
    champion_val = 0.060
    # Challenger seeds: mean 0.070, std ~0.002
    challenger_seeds = [0.069, 0.070, 0.071]

    n_users = 200
    rng = np.random.default_rng(42)
    champ_user = rng.normal(0.060, 0.02, size=n_users)
    chall_user = champ_user + 0.010  # strictly better for every user

    res = evaluate_gate_decision(
        champion_val_ndcg=champion_val,
        challenger_seed_ndcgs=challenger_seeds,
        challenger_user_metrics=chall_user,
        champion_user_metrics=champ_user,
        margin=0.005,
        epsilon=0.001,
    )

    assert res["decision"] == "PROMOTE"
    assert res["criteria"]["margin_exceeds_std"] is True
    assert res["criteria"]["non_inferiority_pass"] is True


def test_retrain_gate_reject_decision():
    """
    REJECT path: Challenger significantly under-performs champion.
    """
    champion_val = 0.065
    challenger_seeds = [0.050, 0.052, 0.051]

    n_users = 200
    rng = np.random.default_rng(42)
    champ_user = rng.normal(0.065, 0.02, size=n_users)
    chall_user = champ_user - 0.014  # strictly worse for every user

    res = evaluate_gate_decision(
        champion_val_ndcg=champion_val,
        challenger_seed_ndcgs=challenger_seeds,
        challenger_user_metrics=chall_user,
        champion_user_metrics=champ_user,
        margin=0.005,
        epsilon=0.001,
    )

    assert res["decision"] == "REJECT"


def test_retrain_gate_inconclusive_decision():
    """
    INCONCLUSIVE path: Challenger difference is within variance/noise band.
    """
    champion_val = 0.060
    challenger_seeds = [0.0605, 0.0595, 0.0600]

    n_users = 200
    rng = np.random.default_rng(42)
    champ_user = rng.normal(0.060, 0.02, size=n_users)
    chall_user = champ_user + rng.normal(0.0001, 0.005, size=n_users)

    res = evaluate_gate_decision(
        champion_val_ndcg=champion_val,
        challenger_seed_ndcgs=challenger_seeds,
        challenger_user_metrics=chall_user,
        champion_user_metrics=champ_user,
        margin=0.005,
        epsilon=0.001,
    )

    assert res["decision"] == "INCONCLUSIVE"
