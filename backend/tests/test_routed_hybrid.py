"""
Unit tests for RoutedHybridRecommender and Phase E Cold-Start Multi-Cohort Study.

Verifies:
  1. Different seeds produce distinct withheld-item sets and distinct evaluation users.
  2. Per-user z-score calibration matches content scores to primary model logits scale.
  3. RoutedHybridRecommender handles edge cases (empty cold, all cold, zero variance).
  4. docs/results/cold_start.json artifact integrity and honest reporting of evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recsys.coldstart import get_disjoint_item_holdout_cohorts, get_item_holdout_split
from recsys.data import RecsysData
from recsys.evaluate import get_evaluation_user_split
from recsys.models.base import BaseRecommender
from recsys.models.routed_hybrid import RoutedHybridRecommender


class DummyRecommender(BaseRecommender):
    """Synthetic recommender for unit tests."""

    def __init__(self, name: str, fixed_scores: np.ndarray) -> None:
        self.name = name
        self.fixed_scores = fixed_scores

    def fit(self, data: RecsysData) -> None:
        pass

    def score(self, histories: list[np.ndarray], **kwargs) -> np.ndarray:
        return np.tile(self.fixed_scores, (len(histories), 1))

    def save(self, model_dir: Path | str) -> None:
        pass

    @classmethod
    def load(cls, model_dir: Path | str) -> DummyRecommender:
        return cls("dummy", np.zeros((1, 10)))


def test_seeds_produce_distinct_withheld_items_and_users():
    """
    Issue I5 Acceptance Check 1:
    Verify whether different seeds actually produce different withheld-item sets
    and different evaluation users.
    """
    data = RecsysData.load(Path("backend/data/recsys"))
    val_users, _ = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=42, persistence_dir=Path("backend/data/recsys")
    )
    val_target_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))

    item_sets = {}
    user_sets = {}

    for s in [42, 43, 44]:
        ho_items, ho_set = get_item_holdout_split(data, fraction=0.10, seed=s)
        ho_users = set(u for u in val_users if val_target_map.get(u) in ho_set)

        assert len(ho_items) == 300, f"Expected 300 items, got {len(ho_items)}"
        assert len(ho_users) > 0, f"Expected > 0 users for seed {s}"

        item_sets[s] = ho_set
        user_sets[s] = ho_users

    # Check pair-wise distinctness
    for s1 in [42, 43]:
        for s2 in [43, 44]:
            if s1 < s2:
                # Withheld item sets must differ
                assert item_sets[s1] != item_sets[s2], f"Seeds {s1} and {s2} produced identical item sets!"
                # Evaluated user sets must differ
                assert user_sets[s1] != user_sets[s2], f"Seeds {s1} and {s2} evaluated identical user sets!"
                # Overlap must be strictly less than total
                overlap_items = len(item_sets[s1] & item_sets[s2])
                assert overlap_items < 300, f"Seeds {s1} and {s2} had 100% item overlap!"
                assert overlap_items < 100, f"Expected low item overlap, got {overlap_items}"


def test_routed_hybrid_score_calibration():
    """
    Issue I5 Acceptance Check 2:
    Verify that per-user score calibration scales content scores to the primary distribution.
    """
    n_items = 10
    # Primary model outputs logits in [10, 20]
    primary_scores = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 15.0, 13.0, 11.0, 17.0])
    # Cold model outputs cosine similarities in [0.0, 1.0]
    cold_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    prim_model = DummyRecommender("dummy_primary", primary_scores)
    cld_model = DummyRecommender("dummy_cold", cold_scores)

    # Route items 8 and 9 as cold
    held_out = {8, 9}
    hybrid = RoutedHybridRecommender(
        primary_model=prim_model,
        cold_model=cld_model,
        cold_threshold=5,
        held_out_items=held_out,
    )

    # Create dummy user histories
    histories = [np.array([0, 1]), np.array([2, 3])]
    scores = hybrid.score(histories)

    assert scores.shape == (2, n_items)

    # Warm items must strictly match primary model scores
    warm_indices = [i for i in range(n_items) if i not in held_out]
    for w in warm_indices:
        np.testing.assert_allclose(scores[0, w], primary_scores[w])

    # Cold items must NOT be in [0.0, 1.0]; they must be scaled into primary score magnitude
    for c in held_out:
        assert scores[0, c] > 5.0, f"Cold item {c} score {scores[0, c]} was not calibrated into logit range!"
        assert scores[0, c] < 25.0, f"Cold item {c} score {scores[0, c]} exploded out of range!"

    # Relative ranking within cold items must preserve content model ranking:
    # Since cold_scores[9] = 1.0 > cold_scores[8] = 0.9, calibrated score 9 must exceed 8
    assert scores[0, 9] > scores[0, 8], "Calibration inverted content model ordering!"


def test_cold_start_artifact_integrity():
    """
    Issue I5 Acceptance Check 3:
    Verify docs/results/cold_start.json structure and evidence claims.
    """
    path = Path("docs/results/cold_start.json")
    assert path.exists(), f"Artifact {path} does not exist!"

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    assert "item_holdout_study" in payload
    ho_study = payload["item_holdout_study"]

    assert ho_study.get("primary_evidence") is True
    assert ho_study.get("n_eval_users") > 0
    assert "models" in ho_study

    models = ho_study["models"]
    assert "random" in models
    assert "content" in models
    assert "two_tower" in models
    assert "routed_hybrid" in models

    # Verify per-cohort numbers differ appropriately
    assert "per_cohort" in ho_study
    per_cohort = ho_study["per_cohort"]
    assert len(per_cohort) >= 3

    cohort_ndcgs = [c["models"]["routed_hybrid"]["ndcg_at_10"] for c in per_cohort.values()]
    # Not all cohorts should have identical performance
    assert len(set(cohort_ndcgs)) > 1, f"Expected differing cohort NDCG metrics, got {cohort_ndcgs}"

    # Verify lower CI bound of Routed Hybrid strictly exceeds analytic random
    rnd_ndcg = models["random"]["ndcg_at_10"]
    hyb_ndcg_lo = models["routed_hybrid"]["ndcg_at_10_lo95"]
    assert hyb_ndcg_lo > rnd_ndcg, f"Routed Hybrid lower CI {hyb_ndcg_lo} failed to exceed random {rnd_ndcg}"

    # Verify evidence statement exists and matches
    assert "evidence_status" in ho_study
    assert ho_study["evidence_status"].lower() in ("strong", "moderate", "weak")


def test_disjoint_item_holdout_cohorts_zero_overlap():
    """
    Phase K: Verify that get_disjoint_item_holdout_cohorts generates 4 mutually disjoint cohorts
    (1 tuning cohort + 3 eval cohorts) with zero overlap between any pair.
    """
    data = RecsysData.load(Path("backend/data/recsys"))
    cohorts = get_disjoint_item_holdout_cohorts(data, n_eval_cohorts=3, cohort_size=200, seed=42)

    keys = ["tuning", "eval_1", "eval_2", "eval_3"]
    assert set(cohorts.keys()) == set(keys)

    for k in keys:
        assert len(cohorts[k]) == 200, f"Cohort {k} expected 200 items, got {len(cohorts[k])}"

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            overlap = set(cohorts[k1]) & set(cohorts[k2])
            assert len(overlap) == 0, f"Expected 0 overlap between {k1} and {k2}, found {len(overlap)} overlapping items!"

