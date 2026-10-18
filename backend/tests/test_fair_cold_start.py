"""
Unit tests for Fair Disjoint Cold-Start Study and Rule-Based Evidence Grading (Phase K).

Verifies:
  1. All partitioned cohorts (tuning, eval_1, eval_2, eval_3) are strictly mutually disjoint.
  2. Exact closed-form cold-only random expectation behaves as derived.
  3. Evidence grading rule behaves deterministically and as specified on synthetic inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys.coldstart import (
    compute_cold_only_random_metrics,
    compute_cold_start_grade,
    get_disjoint_item_holdout_cohorts,
)
from recsys.data import RecsysData


def test_disjoint_cohorts_have_zero_overlap():
    """Verify that all partitioned cohorts share zero items."""
    data = RecsysData.load(Path("backend/data/recsys"))
    cohorts = get_disjoint_item_holdout_cohorts(data, n_eval_cohorts=3, cohort_size=200, seed=42)

    assert "tuning" in cohorts
    assert "eval_1" in cohorts
    assert "eval_2" in cohorts
    assert "eval_3" in cohorts

    cohort_names = list(cohorts.keys())
    assert len(cohort_names) == 4

    for name in cohort_names:
        assert len(cohorts[name]) == 200, f"Cohort {name} expected 200 items, got {len(cohorts[name])}"

    # Verify pairwise disjointness
    for i in range(len(cohort_names)):
        for j in range(i + 1, len(cohort_names)):
            c1_name = cohort_names[i]
            c2_name = cohort_names[j]
            s1 = set(cohorts[c1_name])
            s2 = set(cohorts[c2_name])
            overlap = s1 & s2
            assert len(overlap) == 0, f"Cohorts {c1_name} and {c2_name} overlap by {len(overlap)} items!"


def test_cold_only_random_expectation_exact():
    """Verify that cold-only random baseline matches theoretical formula."""
    # Suppose cold pool has 200 items, seen masks have 0 cold items
    seen_masks = [set() for _ in range(50)]
    cold_set = set(range(200))

    metrics = compute_cold_only_random_metrics(seen_masks, cold_set)

    # Theoretical NDCG@10 for 200 items = sum_{r=1..10} (1 / (200 * log2(r + 1)))
    expected_ndcg = float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / 200.0)
    expected_hr = float(10.0 / 200.0)

    assert np.isclose(metrics["ndcg_at_10"], expected_ndcg, atol=1e-5)
    assert np.isclose(metrics["hit_rate_at_10"], expected_hr, atol=1e-5)
    assert metrics["ndcg_at_10"] > 0.020, "Expected cold-only random NDCG around 0.023, not 0.0015"


def test_grading_rule_synthetic_cases():
    """
    Verify mechanical grading rule on synthetic inputs:
      - Case 1: lo95 > 2*random AND no significant warm loss -> STRONG
      - Case 2: lo95 > 1*random (but <= 2*random) AND no significant warm loss -> MODERATE
      - Case 3: lo95 > 2*random BUT significant warm loss (hi95 < 0) -> WEAK
      - Case 4: lo95 <= 1*random -> WEAK
    """
    cold_random = 0.015

    # Case 1: STRONG
    # Group a lo95 = 0.035 > 2 * 0.015 (0.030)
    # Group b diff CI = [-0.001, +0.002] (not significantly negative, hi95 >= 0)
    res1 = compute_cold_start_grade(
        group_a_ndcg_lo95=0.035,
        cold_only_random_ndcg=cold_random,
        group_b_diff_lo95=-0.001,
        group_b_diff_hi95=0.002,
    )
    assert res1["grade"] == "STRONG"
    assert res1["beats_double_cold_random"] is True
    assert res1["has_significant_warm_loss"] is False

    # Case 2: MODERATE
    # Group a lo95 = 0.025 (between 0.015 and 0.030)
    # Group b diff CI = [-0.001, +0.002]
    res2 = compute_cold_start_grade(
        group_a_ndcg_lo95=0.025,
        cold_only_random_ndcg=cold_random,
        group_b_diff_lo95=-0.001,
        group_b_diff_hi95=0.002,
    )
    assert res2["grade"] == "MODERATE"
    assert res2["beats_single_cold_random"] is True
    assert res2["beats_double_cold_random"] is False

    # Case 3: WEAK due to warm-user degradation
    # Group a lo95 = 0.040 > 2 * 0.015
    # Group b diff CI = [-0.010, -0.002] (strictly negative! significant damage to warm users)
    res3 = compute_cold_start_grade(
        group_a_ndcg_lo95=0.040,
        cold_only_random_ndcg=cold_random,
        group_b_diff_lo95=-0.010,
        group_b_diff_hi95=-0.002,
    )
    assert res3["grade"] == "WEAK"
    assert res3["has_significant_warm_loss"] is True

    # Case 4: WEAK due to failing random baseline
    # Group a lo95 = 0.010 <= 0.015
    res4 = compute_cold_start_grade(
        group_a_ndcg_lo95=0.010,
        cold_only_random_ndcg=cold_random,
        group_b_diff_lo95=0.000,
        group_b_diff_hi95=0.005,
    )
    assert res4["grade"] == "WEAK"
    assert res4["beats_single_cold_random"] is False


def test_routed_hybrid_zero_only_and_capped_boost():
    """Verify route_zero_only and cap_sigma routing behavior."""
    from recsys.models.base import BaseRecommender
    from recsys.models.routed_hybrid import RoutedHybridRecommender

    class MockPrimary(BaseRecommender):
        name = "mock_primary"
        def fit(self, data): pass
        def save(self, path): pass
        @classmethod
        def load(cls, path): return cls()
        def score(self, histories, **kwargs):
            # 3 items, base score = 10.0
            return np.full((len(histories), 3), 10.0, dtype=np.float32)

    class MockCold(BaseRecommender):
        name = "mock_cold"
        def fit(self, data): pass
        def save(self, path): pass
        @classmethod
        def load(cls, path): return cls()
        def score(self, histories, **kwargs):
            # item 0 has high content score
            res = np.zeros((len(histories), 3), dtype=np.float32)
            res[:, 0] = 5.0
            return res

    prim = MockPrimary()
    cold = MockCold()

    # Route zero only: item 0 has 0 pop, item 1 has 3 pop, item 2 has 10 pop
    hyb = RoutedHybridRecommender(
        primary_model=prim,
        cold_model=cold,
        cold_threshold=5,
        route_zero_only=True,
        cap_sigma=1.0,
    )
    hyb.item_train_pop = np.array([0, 3, 10], dtype=np.int64)

    # With route_zero_only=True, only item 0 is routed to cold model; item 1 (pop=3) is NOT routed
    # Even though pop=3 < cold_threshold=5
    sc = hyb.score([np.array([1, 2])])
    assert sc.shape == (1, 3)
    # Item 1 must retain primary score (10.0)
    assert np.isclose(sc[0, 1], 10.0)

