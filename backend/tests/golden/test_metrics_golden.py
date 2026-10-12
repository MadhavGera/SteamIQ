"""
Golden Reference Tests for Recommender Evaluation Metrics — Phase 7.

Validates exact mathematical correctness of rank and metric computations against
hand-calculated analytical ground truths:
  1. compute_ranks (with pessimistic tie-breaking)
  2. hit_rate_at_k
  3. ndcg_at_k
  4. mrr_at_k
  5. catalog_coverage_at_k
  6. novelty_at_k
  7. diversity_at_k
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.metrics import (
    catalog_coverage_at_k,
    compute_ranks,
    diversity_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    paired_bootstrap_diff,
)


def test_compute_ranks_hand_calculated():
    """
    User 0 scores: [0.9, 0.1, 0.8, 0.4, 0.6] -> Sorted items: [0, 2, 4, 3, 1]
      Item 0 is rank 1
      Item 2 is rank 2
      Item 4 is rank 3
      Item 3 is rank 4
      Item 1 is rank 5

    User 1 scores: [0.2, 0.7, 0.5, 0.9, 0.1] -> Sorted items: [3, 1, 2, 0, 4]
      Item 3 is rank 1
      Item 1 is rank 2
      Item 2 is rank 3
      Item 0 is rank 4
      Item 4 is rank 5
    """
    scores = np.array([
        [0.9, 0.1, 0.8, 0.4, 0.6],
        [0.2, 0.7, 0.5, 0.9, 0.1],
    ], dtype=np.float32)

    # Targets: User 0 target is item 4 (rank 3); User 1 target is item 3 (rank 1)
    target_items = np.array([4, 3], dtype=np.int64)

    ranks = compute_ranks(scores, target_items)
    assert ranks[0] == 3
    assert ranks[1] == 1


def test_hit_rate_at_k_hand_calculated():
    """
    Ranks: [1, 3, 5, 10, 15]
    At k=3: ranks <= 3 are [1, 3] -> 2 / 5 = 0.40
    At k=5: ranks <= 5 are [1, 3, 5] -> 3 / 5 = 0.60
    At k=10: ranks <= 10 are [1, 3, 5, 10] -> 4 / 5 = 0.80
    """
    ranks = np.array([1, 3, 5, 10, 15], dtype=np.int64)

    hr_3 = float(np.mean(hit_rate_at_k(ranks, k=3)))
    hr_5 = float(np.mean(hit_rate_at_k(ranks, k=5)))
    hr_10 = float(np.mean(hit_rate_at_k(ranks, k=10)))

    assert pytest.approx(hr_3) == 0.40
    assert pytest.approx(hr_5) == 0.60
    assert pytest.approx(hr_10) == 0.80


def test_ndcg_at_k_hand_calculated():
    """
    Single positive item per user (Binary relevance):
      IDCG = 1.0 (since best possible rank is 1, where DCG = 1 / log2(1 + 1) = 1.0)
      DCG at rank r: 1.0 / log2(r + 1)
      If rank > k, NDCG = 0.0

    Ranks: [1, 2, 3, 4, 10]
    At k=3:
      User 0 (rank 1): 1 / log2(2) = 1.0
      User 1 (rank 2): 1 / log2(3) = 1 / 1.5849625 = 0.63092975
      User 2 (rank 3): 1 / log2(4) = 1 / 2.0 = 0.50
      User 3 (rank 4): 0.0 (rank > 3)
      User 4 (rank 10): 0.0 (rank > 3)
      Mean NDCG@3 = (1.0 + 0.63092975 + 0.50 + 0.0 + 0.0) / 5 = 2.13092975 / 5 = 0.42618595
    """
    ranks = np.array([1, 2, 3, 4, 10], dtype=np.int64)

    expected_ndcg_at_3 = (1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4) + 0.0 + 0.0) / 5.0
    actual_ndcg_3 = float(np.mean(ndcg_at_k(ranks, k=3)))

    assert pytest.approx(actual_ndcg_3, rel=1e-5) == expected_ndcg_at_3


def test_mrr_at_k_hand_calculated():
    """
    RR at rank r: 1.0 / r if r <= k else 0.0

    Ranks: [1, 2, 4, 5, 8]
    At k=4:
      User 0 (rank 1): 1/1 = 1.0
      User 1 (rank 2): 1/2 = 0.5
      User 2 (rank 4): 1/4 = 0.25
      User 3 (rank 5): 0.0
      User 4 (rank 8): 0.0
      Mean MRR@4 = (1.0 + 0.5 + 0.25 + 0.0 + 0.0) / 5 = 1.75 / 5 = 0.35
    """
    ranks = np.array([1, 2, 4, 5, 8], dtype=np.int64)

    mrr_4 = float(np.mean(mrr_at_k(ranks, k=4)))
    assert pytest.approx(mrr_4, rel=1e-5) == 0.35


def test_catalog_coverage_hand_calculated():
    """
    Catalog has 10 items (0..9).
    User 0 top-2: [0, 1]
    User 1 top-2: [1, 2]
    User 2 top-2: [2, 3]
    Unique recommended items: {0, 1, 2, 3} -> 4 unique items.
    Coverage = 4 / 10 = 0.40.
    """
    top_k = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ], dtype=np.int64)

    cov = catalog_coverage_at_k(top_k, n_items=10)
    assert pytest.approx(cov) == 0.40


def test_novelty_at_k_hand_calculated():
    """
    Self-information: -log2(p(i)) = -log2(count(i) / total_interactions)
    Item 0 count: 8
    Item 1 count: 2
    Total interactions: 10
    p(0) = 0.8 -> -log2(0.8) = 0.321928 bits
    p(1) = 0.2 -> -log2(0.2) = 2.321928 bits

    If user gets [0, 1]:
      Mean novelty = (0.321928 + 2.321928) / 2 = 1.321928 bits
    """
    item_counts = np.array([8, 2], dtype=np.float32)
    top_k = np.array([[0, 1]], dtype=np.int64)

    nov = novelty_at_k(top_k, item_counts)
    expected_nov = (-math.log2(0.8) + -math.log2(0.2)) / 2.0
    assert pytest.approx(nov, rel=1e-5) == expected_nov


def test_diversity_at_k_hand_calculated():
    """
    Cosine distance between orthogonal items is 1.0 (Cosine similarity = 0.0).
    Items: [0, 1] with orthogonal unit vectors [1, 0] and [0, 1].
    Cosine similarity: 0.0 -> Cosine distance: 1.0.
    Intra-list diversity for 2 items: sum(1.0) / (2 * 1 / 2) = 1.0.
    """
    top_k = np.array([[0, 1]], dtype=np.int64)
    item_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    div = diversity_at_k(top_k, item_embeddings)
    assert pytest.approx(div, rel=1e-5) == 1.0


def test_paired_bootstrap_diff_exact():
    """
    Model A and Model B identical -> diff mean 0, CI symmetric around 0.
    """
    a = np.array([0.5, 0.8, 0.2, 0.9, 0.4])
    b = np.array([0.5, 0.8, 0.2, 0.9, 0.4])

    mean_diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=200, seed=42)
    assert pytest.approx(mean_diff) == 0.0
    assert pytest.approx(lo) == 0.0
    assert pytest.approx(hi) == 0.0
