"""
Unit tests for multi-source candidate generation with source quotas and merge strategies.
Phase 2 Evidence Hardening.
"""
from __future__ import annotations

import numpy as np
import pytest

from recsys.candidates import (
    DEFAULT_CANDIDATE_COUNTS,
    DEFAULT_QUOTAS,
    CandidateResult,
    generate_candidates,
)
from recsys.models.base import BaseRecommender


class MockScoreRecommender(BaseRecommender):
    """Synthetic recommender scoring designated item subsets with high values."""

    def __init__(self, name: str, item_range: tuple[int, int], n_items: int = 1000) -> None:
        self.name = name
        self.item_range = item_range
        self.n_items = n_items

    def fit(self, data, **kwargs):
        return self

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, **kwargs):
        return cls("mock", (0, 100))

    def score(self, histories: list[np.ndarray], user_indices: list[int] | None = None) -> np.ndarray:
        n_users = len(histories)
        scores = np.zeros((n_users, self.n_items), dtype=np.float32)
        start, end = self.item_range
        # Give linearly decreasing positive scores to items in item_range
        n_in_range = end - start
        if n_in_range > 0:
            assigned = np.linspace(10.0, 1.0, n_in_range, dtype=np.float32)
            scores[:, start:end] = assigned
        return scores


@pytest.fixture
def mock_models() -> dict[str, BaseRecommender]:
    n_total = 1000
    return {
        "two_tower": MockScoreRecommender("two_tower", (0, 200), n_items=n_total),
        "itemknn": MockScoreRecommender("itemknn", (200, 400), n_items=n_total),
        "als": MockScoreRecommender("als", (400, 600), n_items=n_total),
        "content": MockScoreRecommender("content", (600, 800), n_items=n_total),
    }


def test_pool_never_exceeds_cap(mock_models):
    histories = [np.array([1, 2], dtype=np.int32) for _ in range(5)]
    for cap in (10, 50, 150, 300):
        res = generate_candidates(
            histories=histories,
            models=mock_models,
            strategy="quota",
            max_candidates=cap,
        )
        for u_cands in res.user_candidates:
            assert len(u_cands) <= cap, f"Pool length {len(u_cands)} exceeded cap {cap}"


def test_no_duplicates_in_pool(mock_models):
    # Overlapping models
    overlapping_models = {
        "two_tower": MockScoreRecommender("two_tower", (0, 150), n_items=500),
        "itemknn": MockScoreRecommender("itemknn", (50, 200), n_items=500),
        "als": MockScoreRecommender("als", (100, 250), n_items=500),
        "content": MockScoreRecommender("content", (150, 300), n_items=500),
    }
    histories = [np.array([5, 6], dtype=np.int32) for _ in range(3)]
    for strat in ("quota", "interleave", "legacy_ordered"):
        res = generate_candidates(
            histories=histories,
            models=overlapping_models,
            strategy=strat,
            max_candidates=200,
        )
        for u_cands in res.user_candidates:
            assert len(u_cands) == len(set(u_cands)), f"Duplicates found under strategy '{strat}'"


def test_seen_items_never_appear(mock_models):
    histories = [np.array([10, 20], dtype=np.int32)]
    # User 0 has seen items: 5, 205, 405, 605 (which are top candidates in each source)
    seen_masks = [{5, 205, 405, 605, 10, 20}]
    for strat in ("quota", "interleave", "legacy_ordered"):
        res = generate_candidates(
            histories=histories,
            models=mock_models,
            strategy=strat,
            seen_masks=seen_masks,
            max_candidates=300,
        )
        u_cands = set(res.user_candidates[0])
        assert u_cands.isdisjoint(seen_masks[0]), f"Seen items appeared in candidates under '{strat}'"


def test_every_source_gets_at_least_quota_when_available(mock_models):
    histories = [np.array([1, 2], dtype=np.int32)]
    quotas = {"two_tower": 120, "itemknn": 80, "als": 50, "content": 50}
    res = generate_candidates(
        histories=histories,
        models=mock_models,
        quotas=quotas,
        strategy="quota",
        max_candidates=300,
    )
    u_cands = set(res.user_candidates[0])
    assert len(u_cands) == 300

    tt_items = set(range(0, 200))
    knn_items = set(range(200, 400))
    als_items = set(range(400, 600))
    cnt_items = set(range(600, 800))

    tt_count = len(u_cands.intersection(tt_items))
    knn_count = len(u_cands.intersection(knn_items))
    als_count = len(u_cands.intersection(als_items))
    cnt_count = len(u_cands.intersection(cnt_items))

    assert tt_count >= 120, f"Expected >= 120 TT items, got {tt_count}"
    assert knn_count >= 80, f"Expected >= 80 ItemKNN items, got {knn_count}"
    assert als_count >= 50, f"Expected >= 50 ALS items, got {als_count}"
    assert cnt_count >= 50, f"Expected >= 50 Content items, got {cnt_count}"


def test_legacy_ordered_starvation_demonstration(mock_models):
    histories = [np.array([1, 2], dtype=np.int32)]
    # Legacy ordered with 150 TT, 100 kNN, 50 ALS -> exactly 300 items!
    res = generate_candidates(
        histories=histories,
        models=mock_models,
        strategy="legacy_ordered",
        max_candidates=300,
    )
    u_cands = set(res.user_candidates[0])
    cnt_items = set(range(600, 800))
    cnt_count = len(u_cands.intersection(cnt_items))
    # Under legacy ordered, content items are completely starved (0 items)
    assert cnt_count == 0, "Under legacy_ordered with 150+100+50, content items should be starved (0)"


def test_interleave_strategy_distribution(mock_models):
    histories = [np.array([1, 2], dtype=np.int32)]
    res = generate_candidates(
        histories=histories,
        models=mock_models,
        strategy="interleave",
        max_candidates=200,
    )
    u_cands = set(res.user_candidates[0])
    # Interleave should take approximately 50 from each of the 4 sources
    for start, end, name in [
        (0, 200, "two_tower"),
        (200, 400, "itemknn"),
        (400, 600, "als"),
        (600, 800, "content"),
    ]:
        c = len(u_cands.intersection(set(range(start, end))))
        assert c == 50, f"Expected 50 items for {name} under balanced interleave, got {c}"
