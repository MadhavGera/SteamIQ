"""
Unit tests for Recommender System Experiments, Ablations, Cold-Start, and MMR Re-Ranking.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
import pytest

from db.models import ModelRun
from recsys.coldstart import (
    get_simulated_new_items_test_users,
    get_sparse_target_test_users,
)
from recsys.data import RecsysData
from recsys.metrics import gini_coefficient_at_k, mmr_rerank
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.registry import promote_champion
from tests.test_recsys_two_tower import _create_planted_block_recsys_data


@pytest.fixture
def sample_recsys_data() -> RecsysData:
    """Fixture providing synthetic planted block recsys dataset."""
    return _create_planted_block_recsys_data(n_users=60, n_items=40, seed=42)



def test_gini_coefficient_uniform_exposure() -> None:
    """Gini coefficient should be 0.0 for perfectly uniform item exposure."""
    n_items = 50
    # Each item appears exactly 2 times across top-k lists
    repeated_items = np.tile(np.arange(n_items), 2).reshape(10, 10)
    gini = gini_coefficient_at_k(repeated_items, n_items=n_items)
    assert pytest.approx(gini, abs=1e-6) == 0.0


def test_gini_coefficient_single_exposed_item() -> None:
    """Gini coefficient should be (n_items - 1) / n_items (close to 1.0) when 1 item receives all recommendations."""
    n_items = 100
    single_item_recs = np.full((20, 10), fill_value=42, dtype=np.int64)
    gini = gini_coefficient_at_k(single_item_recs, n_items=n_items)
    expected_gini = (n_items - 1.0) / n_items
    assert pytest.approx(gini, abs=1e-4) == expected_gini
    assert gini > 0.98


def test_gini_coefficient_empty_or_edge_cases() -> None:
    """Handles empty top-k or zero items gracefully without errors."""
    assert gini_coefficient_at_k(np.empty((0, 10)), n_items=100) == 0.0
    assert gini_coefficient_at_k(np.zeros((10, 10)), n_items=0) == 0.0


def test_mmr_rerank_lambda_one_preserves_order() -> None:
    """When lambda=1.0, MMR output MUST match descending candidate_scores order."""
    candidate_items = np.array([[10, 20, 30, 40, 50]], dtype=np.int64)
    candidate_scores = np.array([[0.1, 0.9, 0.4, 0.8, 0.2]], dtype=np.float32)
    # Synthetic orthogonal embeddings
    embs = np.eye(60, dtype=np.float32)

    reranked = mmr_rerank(
        candidate_items=candidate_items,
        candidate_scores=candidate_scores,
        item_embeddings=embs,
        lambda_param=1.0,
        k=3,
    )

    # Expected top 3 by score: 20 (0.9), 40 (0.8), 30 (0.4)
    expected = np.array([[20, 40, 30]], dtype=np.int64)
    assert np.array_equal(reranked, expected)


def test_mmr_rerank_diversification() -> None:
    """Lower lambda penalizes candidate items that are highly similar to already selected items."""
    # Item 0 and Item 1 are identical (cosine sim = 1.0)
    # Item 2 is orthogonal (cosine sim = 0.0)
    candidate_items = np.array([[0, 1, 2]], dtype=np.int64)
    # Scores: Item 0 = 1.0, Item 1 = 0.95 (nearly as good as 0), Item 2 = 0.80
    candidate_scores = np.array([[1.0, 0.95, 0.80]], dtype=np.float32)

    embs = np.array([
        [1.0, 0.0],  # item 0
        [1.0, 0.0],  # item 1 (identical to 0)
        [0.0, 1.0],  # item 2 (diverse)
    ], dtype=np.float32)

    # With lambda=0.3 (strong diversity preference):
    # Step 1: picks item 0 (highest score).
    # Step 2: item 1 gets similarity penalty 1.0, so item 2 is picked instead!
    reranked = mmr_rerank(
        candidate_items=candidate_items,
        candidate_scores=candidate_scores,
        item_embeddings=embs,
        lambda_param=0.3,
        k=2,
    )

    expected = np.array([[0, 2]], dtype=np.int64)
    assert np.array_equal(reranked, expected)


def test_sparse_targets_selection(sample_recsys_data: RecsysData) -> None:
    """Sparse target user selection picks only users whose target has <= 10 train positives."""
    eval_users = sample_recsys_data.val_df["user_idx"].unique()
    sparse_users, sparse_targets = get_sparse_target_test_users(
        data=sample_recsys_data,
        eval_users=eval_users,
        max_train_positives=10,
    )

    item_pop = np.asarray(sample_recsys_data.item_train_pop)
    for t in sparse_targets:
        assert item_pop[t] <= 10


def test_simulated_new_items_selection(sample_recsys_data: RecsysData) -> None:
    """Simulated 10% new items selection designates ~10% items and filters matching target users."""
    eval_users = sample_recsys_data.val_df["user_idx"].unique()
    new_items, masked_users, masked_targets = get_simulated_new_items_test_users(
        data=sample_recsys_data,
        eval_users=eval_users,
        fraction=0.10,
        seed=42,
    )

    n_expected_new = int(round(0.10 * sample_recsys_data.n_items))
    assert len(new_items) == n_expected_new
    new_set = set(new_items.tolist())
    for t in masked_targets:
        assert t in new_set


def test_two_tower_cold_start_masking(sample_recsys_data: RecsysData) -> None:
    """TwoTowerModel produces masked ID embeddings when mask_ids is provided."""
    cfg = TwoTowerConfig(
        n_items=sample_recsys_data.n_items,
        hidden_dim=32,
        output_dim=16,
        id_dim=16,
        text_proj_dim=16,
        tag_proj_dim=8,
        numeric_proj_dim=8,
    )
    model = TwoTowerModel(cfg=cfg)
    model.fit(sample_recsys_data)

    unmasked_vecs = model.item_vectors()
    masked_vecs = model.item_vectors(mask_ids=[0, 1])

    # Masked items should differ from unmasked items (due to ID branch replacement with unknown embedding)
    assert not np.allclose(unmasked_vecs[0], masked_vecs[0])
    assert not np.allclose(unmasked_vecs[1], masked_vecs[1])

    # Unmasked items should remain identical
    if sample_recsys_data.n_items > 2:
        assert np.allclose(unmasked_vecs[2], masked_vecs[2])


@pytest.mark.asyncio
async def test_promote_champion_ignores_experiment_runs(db_session) -> None:
    """promote_champion MUST ignore candidate runs tagged with experiment='ablation'."""
    # Create a normal candidate run
    normal_run = ModelRun(
        id=uuid.uuid4(),
        model_name="recsys_two_tower",
        stage="candidate",
        dataset_version="test_v1",
        hyperparameters={"family": "two_tower"},
        metrics={"val_ndcg_at_10": 0.0600},
        created_at=datetime.now(UTC),
    )
    # Create an experiment ablation run with higher metric
    ablation_run = ModelRun(
        id=uuid.uuid4(),
        model_name="recsys_two_tower_no_text",
        stage="candidate",
        dataset_version="test_v1",
        hyperparameters={"family": "two_tower", "experiment": "ablation", "is_experiment": True},
        metrics={"val_ndcg_at_10": 0.0999},  # High metric, but should be ignored
        created_at=datetime.now(UTC),
    )

    db_session.add(normal_run)
    db_session.add(ablation_run)
    await db_session.commit()

    promoted = await promote_champion(family="two_tower", session=db_session)
    assert promoted is not None
    assert promoted.id == normal_run.id
    assert promoted.model_name == "recsys_two_tower"
    assert promoted.id != ablation_run.id
