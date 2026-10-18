"""
Golden Reference Tests for SNIPS Exposure-Debiased Ranking Metrics — Phase 7.

Validates exact mathematical correctness of:
  1. compute_item_propensities: p_i proportional to (pop_i + 1)^gamma
  2. compute_snips_weights: w_u = 1.0 / max(p(t_u), p_min)
  3. snips_ndcg_at_k: sum_u(w_u * NDCG_u@k) / sum_u(w_u)
  4. snips_bootstrap_ci: 95% bootstrap confidence interval bounds

All tests verify analytical closed-form derivations calculated by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.metrics import (
    compute_item_propensities,
    compute_snips_weights,
    ndcg_at_k,
    snips_bootstrap_ci,
    snips_ndcg_at_k,
)


def test_compute_item_propensities_hand_calculated():
    """
    Hand-calculated analytical ground truth for item propensities:
    Catalog of 3 items:
      Item 0: pop = 99
      Item 1: pop = 24
      Item 2: pop = 3

    Smoothing: (pop + 1)
      Item 0: 99 + 1 = 100
      Item 1: 24 + 1 = 25
      Item 2: 3 + 1 = 4

    Power gamma = 0.5 (square root):
      Item 0: sqrt(100) = 10.0
      Item 1: sqrt(25) = 5.0
      Item 2: sqrt(4) = 2.0
      Sum = 10.0 + 5.0 + 2.0 = 17.0

    Propensities:
      p_0 = 10 / 17
      p_1 = 5 / 17
      p_2 = 2 / 17
    """
    item_train_pop = np.array([99, 24, 3], dtype=np.int64)
    props = compute_item_propensities(item_train_pop, gamma=0.5)

    assert len(props) == 3
    assert pytest.approx(float(np.sum(props)), abs=1e-9) == 1.0
    assert pytest.approx(props[0], rel=1e-6) == 10.0 / 17.0
    assert pytest.approx(props[1], rel=1e-6) == 5.0 / 17.0
    assert pytest.approx(props[2], rel=1e-6) == 2.0 / 17.0


def test_snips_weights_and_ndcg_hand_calculated():
    """
    Analytical ground truth for SNIPS NDCG@10 without propensity clipping:
    Using same 3 items with pop = [99, 24, 3] and gamma = 0.5:
      p = [10/17, 5/17, 2/17]
      p_min = 0.01 (smaller than min(p) = 2/17 ~ 0.1176, so zero clipping occurs)

    Weights:
      w_0 = 1 / p_0 = 17 / 10 = 1.7
      w_1 = 1 / p_1 = 17 / 5 = 3.4
      w_2 = 1 / p_2 = 17 / 2 = 8.5

    3 Users:
      User 0: target = 0 (head item), model ranks at 1 (within k=10)
        NDCG_0@10 = 1 / log2(1 + 1) = 1.0
      User 1: target = 1 (torso item), model ranks at 2 (within k=10)
        NDCG_1@10 = 1 / log2(2 + 1) = 1 / log2(3) ~ 0.63092975
      User 2: target = 2 (tail item), model ranks at 15 (exceeds k=10)
        NDCG_2@10 = 0.0

    Sum of weights:
      1.7 + 3.4 + 8.5 = 13.6

    Weighted sum of NDCG:
      1.7 * 1.0 + 3.4 * (1 / log2(3)) + 8.5 * 0.0
      = 1.7 + 3.4 / log2(3)

    Expected SNIPS NDCG@10:
      (1.7 + 3.4 / log2(3)) / 13.6
    """
    item_train_pop = np.array([99, 24, 3], dtype=np.int64)
    targets = np.array([0, 1, 2], dtype=np.int64)
    ranks = np.array([1, 2, 15], dtype=np.int64)

    # 1. Verify weights
    weights = compute_snips_weights(targets, item_train_pop, gamma=0.5, p_min=0.01)
    assert pytest.approx(weights[0], rel=1e-6) == 1.7
    assert pytest.approx(weights[1], rel=1e-6) == 3.4
    assert pytest.approx(weights[2], rel=1e-6) == 8.5

    # 2. Verify raw unweighted NDCG vs SNIPS NDCG
    raw_ndcg = float(np.mean(ndcg_at_k(ranks, k=10)))
    expected_raw = (1.0 + 1.0 / math.log2(3) + 0.0) / 3.0
    assert pytest.approx(raw_ndcg, rel=1e-6) == expected_raw

    # 3. Verify exact SNIPS NDCG
    expected_snips = (1.7 * 1.0 + 3.4 * (1.0 / math.log2(3)) + 8.5 * 0.0) / 13.6
    actual_snips = snips_ndcg_at_k(ranks, targets, item_train_pop, k=10, gamma=0.5, p_min=0.01)

    assert pytest.approx(actual_snips, rel=1e-6) == expected_snips

    # Debiased metric must penalize the model for missing the rare item
    assert actual_snips < raw_ndcg


def test_snips_clipping_with_p_min():
    """
    Verifies that p_min correctly clips rare item propensities to prevent infinite weights:
    Item 0 pop = 99 -> powered = 10
    Item 1 pop = 0  -> smoothed = 1 -> powered = 1
    Total = 11.
    p_0 = 10 / 11 ~ 0.909
    p_1 = 1 / 11 ~ 0.0909

    If p_min = 0.20:
      p_1 (0.0909) is clipped to 0.20
      w_1 = 1 / 0.20 = 5.0 (instead of 11.0)
    """
    item_train_pop = np.array([99, 0], dtype=np.int64)
    targets = np.array([0, 1], dtype=np.int64)

    weights = compute_snips_weights(targets, item_train_pop, gamma=0.5, p_min=0.20)
    assert pytest.approx(weights[0], rel=1e-6) == 11.0 / 10.0
    assert pytest.approx(weights[1], rel=1e-6) == 1.0 / 0.20


def test_snips_uniform_popularity_equivalence():
    """
    Mathematical theorem: When all items have identical training popularity,
    propensities are uniform (p_i = 1/M) and SNIPS NDCG@k is mathematically
    IDENTICAL to standard unweighted NDCG@k.
    """
    n_items = 20
    item_train_pop = np.full(n_items, 42, dtype=np.int64)
    rng = np.random.default_rng(123)
    n_users = 100
    targets = rng.integers(0, n_items, size=n_users)
    ranks = rng.integers(1, 50, size=n_users)

    raw_ndcg = float(np.mean(ndcg_at_k(ranks, k=10)))
    snips_ndcg = snips_ndcg_at_k(ranks, targets, item_train_pop, k=10, gamma=0.5, p_min=1e-5)

    assert pytest.approx(snips_ndcg, rel=1e-6) == raw_ndcg


def test_snips_bootstrap_ci():
    """
    Verifies that bootstrap confidence interval contains the point estimate
    and produces sound ordering: lo95 <= point_estimate <= hi95.
    """
    item_train_pop = np.array([1000, 200, 50, 10, 2], dtype=np.int64)
    rng = np.random.default_rng(42)
    n_users = 200
    targets = rng.integers(0, 5, size=n_users)
    ranks = rng.integers(1, 30, size=n_users)

    mean_val, lo95, hi95 = snips_bootstrap_ci(
        ranks=ranks,
        targets=targets,
        item_train_pop=item_train_pop,
        k=10,
        gamma=0.5,
        p_min=1e-4,
        n_boot=500,
        seed=42,
    )

    assert lo95 <= mean_val <= hi95
    assert 0.0 <= lo95 <= 1.0
    assert 0.0 <= hi95 <= 1.0


def test_snips_empty_and_single_edge_cases():
    """
    Edge case handling for empty or single element arrays.
    """
    pop = np.array([10, 20], dtype=np.int64)
    empty_ranks = np.array([], dtype=np.int64)
    empty_targets = np.array([], dtype=np.int64)

    assert snips_ndcg_at_k(empty_ranks, empty_targets, pop) == 0.0

    mean_v, lo_v, hi_v = snips_bootstrap_ci(empty_ranks, empty_targets, pop)
    assert (mean_v, lo_v, hi_v) == (0.0, 0.0, 0.0)

    # Single user
    one_rk = np.array([1], dtype=np.int64)
    one_tg = np.array([0], dtype=np.int64)
    s_val = snips_ndcg_at_k(one_rk, one_tg, pop, k=10)
    assert pytest.approx(s_val) == 1.0

    m1, l1, h1 = snips_bootstrap_ci(one_rk, one_tg, pop, k=10)
    assert pytest.approx(m1) == 1.0
    assert pytest.approx(l1) == 1.0
    assert pytest.approx(h1) == 1.0
