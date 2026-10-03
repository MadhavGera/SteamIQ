"""
Comprehensive unit tests for recommender metrics, evaluation pipeline,
popularity baseline, and registry isolation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sqlalchemy import select

from db.models import ModelRun
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model
from recsys.metrics import (
    bootstrap_ci,
    catalog_coverage_at_k,
    compute_ranks,
    diversity_at_k,
    head_share_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    paired_bootstrap_diff,
)
from recsys.models.base import BaseRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.registry import log_recsys_run, promote_champion

# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics & Pessimistic Ties
# ─────────────────────────────────────────────────────────────────────────────

def test_hand_computed_5_item_metrics():
    """
    Tests exact hand-computed rank, HR@k, NDCG@k, and MRR@k on a 5-item catalog.
    """
    # User 0: scores [10, 5, 20, 8, 3], target=0 (score 10).
    # Items with score >= 10: item 0 (10), item 2 (20).
    # Rank = 2.
    #
    # User 1: scores [1, 2, 3, 4, 5], target=4 (score 5).
    # Items with score >= 5: item 4 (5).
    # Rank = 1.
    #
    # User 2: scores [50, 40, 30, 20, 10], target=4 (score 10).
    # Items with score >= 10: all 5 items.
    # Rank = 5.
    scores = np.array([
        [10.0, 5.0, 20.0, 8.0, 3.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [50.0, 40.0, 30.0, 20.0, 10.0],
    ], dtype=np.float32)
    targets = np.array([0, 4, 4], dtype=np.int64)

    ranks = compute_ranks(scores, targets)
    assert np.array_equal(ranks, [2, 1, 5])

    # HR@k
    hr1 = hit_rate_at_k(ranks, 1)
    assert np.array_equal(hr1, [0.0, 1.0, 0.0])

    hr2 = hit_rate_at_k(ranks, 2)
    assert np.array_equal(hr2, [1.0, 1.0, 0.0])

    hr5 = hit_rate_at_k(ranks, 5)
    assert np.array_equal(hr5, [1.0, 1.0, 1.0])

    # NDCG@k
    ndcg1 = ndcg_at_k(ranks, 1)
    assert np.allclose(ndcg1, [0.0, 1.0, 0.0])

    ndcg2 = ndcg_at_k(ranks, 2)
    expected_ndcg2_u0 = 1.0 / math.log2(2 + 1)
    assert np.allclose(ndcg2, [expected_ndcg2_u0, 1.0, 0.0])

    ndcg5 = ndcg_at_k(ranks, 5)
    expected_ndcg5_u2 = 1.0 / math.log2(5 + 1)
    assert np.allclose(ndcg5, [expected_ndcg2_u0, 1.0, expected_ndcg5_u2])

    # MRR@k
    mrr1 = mrr_at_k(ranks, 1)
    assert np.allclose(mrr1, [0.0, 1.0, 0.0])

    mrr2 = mrr_at_k(ranks, 2)
    assert np.allclose(mrr2, [0.5, 1.0, 0.0])

    mrr5 = mrr_at_k(ranks, 5)
    assert np.allclose(mrr5, [0.5, 1.0, 0.2])


def test_pessimistic_tie_handling():
    """
    Constant scores must receive the worst possible rank (n_items), preventing
    flat or constant scorers from appearing competent.
    """
    # 5 items all equal score 7.0
    scores = np.full((3, 5), 7.0, dtype=np.float32)
    targets = np.array([0, 2, 4], dtype=np.int64)

    ranks = compute_ranks(scores, targets)
    # Every user's target ties with all 5 items -> rank is 5
    assert np.array_equal(ranks, [5, 5, 5])

    # Hit rate @ 4 must be 0.0
    assert np.all(hit_rate_at_k(ranks, 4) == 0.0)
    assert np.all(ndcg_at_k(ranks, 4) == 0.0)
    assert np.all(mrr_at_k(ranks, 4) == 0.0)

    # Hit rate @ 5 is 1.0
    assert np.all(hit_rate_at_k(ranks, 5) == 1.0)
    assert np.allclose(ndcg_at_k(ranks, 5), 1.0 / math.log2(6))
    assert np.allclose(mrr_at_k(ranks, 5), 0.2)


def test_partial_tie_handling():
    """
    Items tying for top place must all be ranked behind each other.
    """
    # 3 items tied for 1st place: [10, 10, 10, 2, 1]
    scores = np.array([[10.0, 10.0, 10.0, 2.0, 1.0]], dtype=np.float32)
    target = np.array([0], dtype=np.int64)

    rank = compute_ranks(scores, target)
    assert rank[0] == 3
    assert hit_rate_at_k(rank, 2)[0] == 0.0
    assert hit_rate_at_k(rank, 3)[0] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. List & Catalog Metrics (Coverage, Novelty, Diversity, Head Share)
# ─────────────────────────────────────────────────────────────────────────────

def test_catalog_coverage():
    # 2 users, k=2, catalog size = 10
    # user 0 recommended [0, 1], user 1 recommended [1, 2]
    # unique = {0, 1, 2} (3 items)
    top_k = np.array([[0, 1], [1, 2]], dtype=np.int64)
    cov = catalog_coverage_at_k(top_k, n_items=10)
    assert math.isclose(cov, 0.30, rel_tol=1e-5)


def test_novelty():
    # item 0 pop=100, item 1 pop=10, item 2 pop=1 (total=111)
    pop = np.array([100.0, 10.0, 1.0])
    total = 111.0

    # User 0 gets item 2 (rare): self-info = -log2(1/111) = log2(111)
    top_rare = np.array([[2]], dtype=np.int64)
    nov_rare = novelty_at_k(top_rare, pop, total_train_positives=total)
    assert math.isclose(nov_rare, math.log2(111), rel_tol=1e-5)

    # User 0 gets item 0 (common): self-info = -log2(100/111)
    top_comm = np.array([[0]], dtype=np.int64)
    nov_comm = novelty_at_k(top_comm, pop, total_train_positives=total)
    assert math.isclose(nov_comm, -math.log2(100 / 111), rel_tol=1e-5)
    assert nov_rare > nov_comm


def test_diversity():
    # 2 orthogonal items: cosine = 0.0 -> diversity = 1.0 - 0.0 = 1.0
    embs_orth = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    top_k = np.array([[0, 1]], dtype=np.int64)
    div_orth = diversity_at_k(top_k, embs_orth)
    assert math.isclose(div_orth, 1.0, abs_tol=1e-5)

    # 2 identical items: cosine = 1.0 -> diversity = 1.0 - 1.0 = 0.0
    embs_ident = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    div_ident = diversity_at_k(top_k, embs_ident)
    assert math.isclose(div_ident, 0.0, abs_tol=1e-5)


def test_head_share():
    # 100 items, item 0 is top 1% (rank 1 of 100)
    pop = np.arange(100, 0, -1, dtype=np.float32)
    # top_k recommending item 0 in 1 of 2 slots for user
    top_k = np.array([[0, 50]], dtype=np.int64)
    share = head_share_at_k(top_k, pop, head_fraction=0.01)
    assert math.isclose(share, 0.5, abs_tol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bootstrap CI & Paired Diff
# ─────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_determinism_and_bounds():
    rng = np.random.default_rng(123)
    values = rng.uniform(0.0, 1.0, size=50)

    mean1, lo1, hi1 = bootstrap_ci(values, n_boot=500, seed=42)
    mean2, lo2, hi2 = bootstrap_ci(values, n_boot=500, seed=42)

    assert mean1 == mean2
    assert lo1 == lo2
    assert hi1 == hi2
    assert lo1 <= mean1 <= hi1


def test_paired_bootstrap_diff():
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 0.0])

    mean_diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=500, seed=42)
    assert math.isclose(mean_diff, 0.5, abs_tol=1e-5)
    assert lo <= mean_diff <= hi


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Framework, Masking, and Oracles
# ─────────────────────────────────────────────────────────────────────────────

def _create_mock_recsys_data(n_users: int = 10, n_items: int = 20) -> RecsysData:
    """Helper to build a valid minimal RecsysData fixture."""
    train_rows = []
    for u in range(n_users):
        for i in range(4):
            train_rows.append({"user_idx": u, "item_idx": i, "ts": 100 + i, "hours": 10.0})

    val_rows = [{"user_idx": u, "item_idx": 5, "ts": 200, "hours": 5.0} for u in range(n_users)]
    test_rows = [{"user_idx": u, "item_idx": 6, "ts": 300, "hours": 5.0} for u in range(n_users)]
    neg_rows = [{"user_idx": u, "item_idx": 10, "ts": 150, "hours": 1.0, "split": "train"} for u in range(n_users)]

    users_df = pd.DataFrame({"user_idx": list(range(n_users)), "original_user_id": list(range(n_users))})
    mappings_df = pd.DataFrame({"item_idx": list(range(n_items)), "app_id": [1000 + i for i in range(n_items)]})
    items_df = pd.DataFrame({"item_idx": list(range(n_items)), "name": [f"Game {i}" for i in range(n_items)]})

    return RecsysData(
        train_df=pd.DataFrame(train_rows),
        val_df=pd.DataFrame(val_rows),
        test_df=pd.DataFrame(test_rows),
        negatives_df=pd.DataFrame(neg_rows),
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=np.random.default_rng(42).normal(size=(n_items, 16)).astype(np.float32),
        item_train_pop=np.full(n_items, 50.0, dtype=np.float32),
        item_tags_sparse=sp.csr_matrix((n_items, 10), dtype=np.float32),
        dataset_version="test_version_123",
        manifest={"dataset_version": "test_version_123"},
    )


class OracleRecommender(BaseRecommender):
    name = "recsys_oracle"

    def __init__(self, target_item: int) -> None:
        self.target_item = target_item

    def fit(self, data: RecsysData) -> None:
        self.n_items = data.n_items

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        b = len(histories)
        scores = np.zeros((b, self.n_items), dtype=np.float32)
        scores[:, self.target_item] = 1000.0  # highest score
        return scores

    def save(self, model_dir: Path | str) -> None:
        pass

    @classmethod
    def load(cls, model_dir: Path | str) -> OracleRecommender:
        return cls(target_item=5)


class ReverseOracleRecommender(BaseRecommender):
    name = "recsys_reverse_oracle"

    def __init__(self, target_item: int) -> None:
        self.target_item = target_item

    def fit(self, data: RecsysData) -> None:
        self.n_items = data.n_items

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        b = len(histories)
        scores = np.zeros((b, self.n_items), dtype=np.float32)
        scores[:, self.target_item] = -1000.0  # lowest score
        return scores

    def save(self, model_dir: Path | str) -> None:
        pass

    @classmethod
    def load(cls, model_dir: Path | str) -> ReverseOracleRecommender:
        return cls(target_item=5)


def test_oracle_and_reverse_oracle():
    data = _create_mock_recsys_data(n_users=5, n_items=20)

    # Oracle scores val target (item 5) highest
    oracle = OracleRecommender(target_item=5)
    oracle.fit(data)
    val_res = evaluate_model(oracle, data, split="val", k_list=(1, 10), max_users=5, allow_small_user_set=True)
    assert val_res.metrics["hit_rate_at_1"] == 1.0
    assert val_res.metrics["ndcg_at_10"] == 1.0
    assert val_res.metrics["mrr_at_10"] == 1.0

    # Reverse Oracle scores val target lowest
    rev = ReverseOracleRecommender(target_item=5)
    rev.fit(data)
    val_rev_res = evaluate_model(rev, data, split="val", k_list=(1, 10), max_users=5, allow_small_user_set=True)
    assert val_rev_res.metrics["hit_rate_at_10"] == 0.0


def test_masking_prevents_seen_items_from_top_k():
    data = _create_mock_recsys_data(n_users=5, n_items=20)

    # Item 0 is in user train history. A model scoring item 0 highest (1000.0)
    # should have item 0 masked to -inf during evaluation.
    class TrainScorer(BaseRecommender):
        name = "recsys_train_scorer"

        def fit(self, data: RecsysData) -> None:
            self.n_items = data.n_items

        def score(self, histories: list[np.ndarray]) -> np.ndarray:
            b = len(histories)
            scores = np.zeros((b, self.n_items), dtype=np.float32)
            scores[:, 0] = 1000.0  # seen item
            scores[:, 5] = 10.0    # val target
            return scores

        def save(self, model_dir: Path | str) -> None:
            pass

        @classmethod
        def load(cls, model_dir: Path | str) -> BaseRecommender:
            return cls()

    model = TrainScorer()
    model.fit(data)
    res = evaluate_model(model, data, split="val", k_list=(1,), max_users=5, allow_small_user_set=True)
    # Since item 0 was masked to -inf, item 5 (score 10) becomes rank 1!
    assert res.metrics["hit_rate_at_1"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Popularity Recommender Save/Load Roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_popularity_recommender_roundtrip(tmp_path: Path):
    data = _create_mock_recsys_data(n_users=5, n_items=20)
    model = PopularityRecommender()
    model.fit(data)

    test_histories = [np.array([1, 2]), np.array([3])]
    orig_scores = model.score(test_histories)

    # Save
    save_dir = tmp_path / "popularity_artifact"
    model.save(save_dir)
    assert (save_dir / "item_scores.npy").exists()
    assert (save_dir / "metadata.json").exists()

    # Load
    loaded = PopularityRecommender.load(save_dir)
    loaded_scores = loaded.score(test_histories)

    assert np.array_equal(orig_scores, loaded_scores)
    assert loaded.name == "recsys_popularity"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Registry Safety Isolation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_prefix_isolation(db_session):
    """
    Verifies that promote_champion only touches runs starting with 'recsys_'
    and never modifies non-recsys models (e.g., game_success_predictor).
    """
    # 1. Insert existing production run for game_success_predictor
    success_model = ModelRun(
        model_name="game_success_predictor",
        stage="production",
        dataset_version="2026-08-01",
        hyperparameters={"model_type": "LightGBM"},
        metrics={"roc_auc": 0.82},
    )
    db_session.add(success_model)
    await db_session.commit()

    # 2. Log a candidate recsys run
    val_metrics = {"val_ndcg_at_10": 0.085, "val_hit_rate_at_10": 0.15}
    recsys_run, _ = await log_recsys_run(
        model_name="recsys_popularity",
        family="popularity",
        dataset_version="test_version",
        hyperparameters={"transform": "log1p"},
        val_metrics=val_metrics,
        session=db_session,
    )
    assert recsys_run.stage == "candidate"

    # 3. Promote champion
    champion = await promote_champion(family="popularity", session=db_session)
    assert champion is not None
    assert champion.id == recsys_run.id
    assert champion.stage == "production"

    # 4. Verify game_success_predictor was NOT modified or archived
    stmt = select(ModelRun).where(ModelRun.model_name == "game_success_predictor")
    res = await db_session.execute(stmt)
    success_record = res.scalar_one()
    assert success_record.stage == "production"  # remains production!


@pytest.mark.asyncio
async def test_registry_rejects_invalid_model_name(db_session):
    """
    Rejects any model name not starting with 'recsys_'.
    """
    with pytest.raises(ValueError, match="All recsys models MUST start with 'recsys_'"):
        await log_recsys_run(
            model_name="invalid_name",
            family="popularity",
            dataset_version="v1",
            session=db_session,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reproducibility & Hand-Computed Metric Verification (Step 6a, 6b)
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluating_same_saved_model_twice_gives_identical_metrics(tmp_path: Path):
    """
    (6a) Verifies that evaluating the same saved model artifact twice in fresh
    evaluations produces strictly identical metrics across VAL and TEST.
    """
    data = _create_mock_recsys_data(n_users=10, n_items=20)
    model = PopularityRecommender()
    model.fit(data)

    save_dir = tmp_path / "pop_reproducibility"
    model.save(save_dir)

    # First load and eval
    model_run1 = PopularityRecommender.load(save_dir)
    val_res1 = evaluate_model(model_run1, data, split="val", k_list=(1, 5, 10), max_users=10, allow_small_user_set=True, seed=42)
    test_res1 = evaluate_model(model_run1, data, split="test", k_list=(1, 5, 10), max_users=10, allow_small_user_set=True, seed=42)

    # Second load and eval
    model_run2 = PopularityRecommender.load(save_dir)
    val_res2 = evaluate_model(model_run2, data, split="val", k_list=(1, 5, 10), max_users=10, allow_small_user_set=True, seed=42)
    test_res2 = evaluate_model(model_run2, data, split="test", k_list=(1, 5, 10), max_users=10, allow_small_user_set=True, seed=42)

    # Check that all val metrics are identical
    for k, v1 in val_res1.metrics.items():
        assert k in val_res2.metrics
        v2 = val_res2.metrics[k]
        assert v1 == pytest.approx(v2, abs=1e-9), f"Val metric {k} differed: {v1} vs {v2}"

    # Check that all test metrics are identical
    for k, v1 in test_res1.metrics.items():
        assert k in test_res2.metrics
        v2 = test_res2.metrics[k]
        assert v1 == pytest.approx(v2, abs=1e-9), f"Test metric {k} differed: {v1} vs {v2}"


def test_hand_computed_fixture_known_metrics():
    """
    (6b) Verifies HR, NDCG, MRR, catalog coverage, and diversity against exact hand-computed values.

    Hand-computed setup:
      Catalog size: 4 items (0, 1, 2, 3)
      Embeddings:
        item 0: [1, 0]
        item 1: [0, 1]
        item 2: [1, 0]
        item 3: [0, 1]

      User 0:
        Scores: [0.9, 0.7, 0.4, 0.1]
        Target item: 1 (score 0.7)
        Pessimistic rank: 2 (scores >= 0.7 are item 0 and item 1)
        Top-2 list: [0, 1]
        Cosine(0, 1) = 0.0 -> diversity = 1.0 - 0.0 = 1.0
        HR@2 = 1.0
        NDCG@2 = 1 / log2(2 + 1) = 1 / log2(3)
        MRR@2 = 1 / 2 = 0.5

      User 1:
        Scores: [0.2, 0.8, 0.1, 0.9]
        Target item: 0 (score 0.2)
        Pessimistic rank: 3 (scores >= 0.2 are item 3, item 1, item 0)
        Top-2 list: [3, 1]
        Cosine(3, 1) = 1.0 -> diversity = 1.0 - 1.0 = 0.0
        HR@2 = 0.0
        NDCG@2 = 0.0
        MRR@2 = 0.0

      Aggregate ground truth:
        Mean HR@2 = (1.0 + 0.0) / 2 = 0.5
        Mean NDCG@2 = (1.0 / log2(3) + 0.0) / 2 ≈ 0.3154648767857287
        Mean MRR@2 = (0.5 + 0.0) / 2 = 0.25
        Coverage@2: unique items {0, 1, 3} out of 4 = 3 / 4 = 0.75
        Diversity@2 = (1.0 + 0.0) / 2 = 0.5
    """
    scores = np.array([
        [0.9, 0.7, 0.4, 0.1],
        [0.2, 0.8, 0.1, 0.9],
    ], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int64)
    item_embs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    # 1. Ranks
    ranks = compute_ranks(scores, targets)
    assert np.array_equal(ranks, [2, 3])

    # 2. HR@2
    hr2 = hit_rate_at_k(ranks, 2)
    assert np.array_equal(hr2, [1.0, 0.0])
    assert float(np.mean(hr2)) == 0.5

    # 3. NDCG@2
    ndcg2 = ndcg_at_k(ranks, 2)
    expected_user0_ndcg = 1.0 / math.log2(3.0)
    assert math.isclose(ndcg2[0], expected_user0_ndcg, rel_tol=1e-6)
    assert ndcg2[1] == 0.0
    expected_mean_ndcg = (expected_user0_ndcg + 0.0) / 2.0
    assert math.isclose(float(np.mean(ndcg2)), expected_mean_ndcg, rel_tol=1e-6)

    # 4. MRR@2
    mrr2 = mrr_at_k(ranks, 2)
    assert np.array_equal(mrr2, [0.5, 0.0])
    assert float(np.mean(mrr2)) == 0.25

    # 5. Coverage & Diversity
    # Top-2 lists: user 0 -> [0, 1], user 1 -> [3, 1]
    top_k = np.array([
        [0, 1],
        [3, 1],
    ], dtype=np.int64)

    cov = catalog_coverage_at_k(top_k, n_items=4)
    assert math.isclose(cov, 0.75, rel_tol=1e-6)

    div = diversity_at_k(top_k, item_embs)
    assert math.isclose(div, 0.5, rel_tol=1e-6)

