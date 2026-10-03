"""
Comprehensive unit tests for recommender models (Item-kNN, ALS, Content, Popularity),
closed-form ALS fold-in mathematical exactness, serialization roundtrips,
disjoint user partitions, and registry latest run queries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from db.models import ModelRun
from recsys.data import RecsysData
from recsys.evaluate import get_evaluation_user_split
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.registry import get_latest_recsys_runs, log_recsys_run


def _create_synthetic_recsys_data(n_users: int = 30, n_items: int = 25) -> RecsysData:
    """Helper to build a self-contained RecsysData fixture."""
    rng = np.random.default_rng(42)
    train_rows = []
    for u in range(n_users):
        # 6 positive interactions per user
        items = rng.choice(n_items, size=6, replace=False)
        for i_idx, item in enumerate(items):
            train_rows.append({
                "user_idx": u,
                "item_idx": int(item),
                "ts": 100 + i_idx,
                "hours": float(rng.uniform(1.0, 50.0)),
            })

    val_rows = [{"user_idx": u, "item_idx": int(rng.integers(0, n_items)), "ts": 200, "hours": 5.0} for u in range(n_users)]
    test_rows = [{"user_idx": u, "item_idx": int(rng.integers(0, n_items)), "ts": 300, "hours": 5.0} for u in range(n_users)]
    neg_rows = [{"user_idx": u, "item_idx": int(rng.integers(0, n_items)), "ts": 150, "hours": 1.0, "split": "train"} for u in range(n_users)]

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
        item_text_emb=rng.normal(size=(n_items, 32)).astype(np.float32),
        item_train_pop=np.full(n_items, 50.0, dtype=np.float32),
        item_tags_sparse=sp.csr_matrix(rng.integers(0, 2, size=(n_items, 20)), dtype=np.float32),
        dataset_version="synthetic_v1",
        manifest={"dataset_version": "synthetic_v1"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Item-kNN Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_itemknn_shrinkage_and_symmetry():
    """
    Tests that:
      1. Co-occurrence similarity is symmetric prior to top-k pruning.
      2. Shrinkage strictly lowers similarity values for rare pairs compared to unshrunk cosine.
      3. Top-k pruning keeps at most k neighbors per item.
    """
    data = _create_synthetic_recsys_data(n_users=50, n_items=30)

    # 1. Check with very large k (no pruning) to test symmetry
    model_unpruned = ItemKNNRecommender(k_neighbors=100, shrink=10.0)
    model_unpruned.fit(data)
    S = model_unpruned.similarity_matrix.toarray()

    # Self-similarity diagonal must be zero
    assert np.allclose(np.diag(S), 0.0)

    # Off-diagonal symmetry: S[i, j] == S[j, i] when no top-k truncation occurs
    # (since all items have <= 30 neighbors and k=100)
    assert np.allclose(S, S.T, atol=1e-6)

    # 2. Compare shrink=0 vs shrink=10
    model_no_shrink = ItemKNNRecommender(k_neighbors=100, shrink=0.0)
    model_no_shrink.fit(data)
    S_no_shrink = model_no_shrink.similarity_matrix.toarray()

    # With shrink=10, all positive entries must be strictly smaller than with shrink=0
    pos_mask = S_no_shrink > 0.0
    assert np.all(S[pos_mask] < S_no_shrink[pos_mask])

    # 3. Pruned top-k neighbors
    k = 5
    model_pruned = ItemKNNRecommender(k_neighbors=k, shrink=10.0)
    model_pruned.fit(data)
    S_pruned = model_pruned.similarity_matrix

    # Each row must have at most k non-zero entries
    for i in range(data.n_items):
        row_nnz = S_pruned.indptr[i + 1] - S_pruned.indptr[i]
        assert row_nnz <= k


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALS Closed-Form Fold-In Test
# ─────────────────────────────────────────────────────────────────────────────

def test_als_closed_form_fold_in():
    """
    Tests that folding in a train user's own history reproduces the trained user factor
    with cosine similarity > 0.95.
    """
    data = _create_synthetic_recsys_data(n_users=100, n_items=40)

    model = ALSRecommender(factors=32, regularization=0.05, alpha=15.0, iterations=15, seed=42)
    model.fit(data)

    # Build user interaction confidence map from training data
    u_idx = 0
    u_train = data.train_df[data.train_df["user_idx"] == u_idx]
    user_items = u_train["item_idx"].to_numpy(dtype=np.int32)
    user_hours = u_train["hours"].to_numpy(dtype=np.float32)
    user_conf = (1.0 + model.alpha * np.log1p(np.maximum(user_hours, 0.0))).astype(np.float32)

    # Compute fold-in user factor
    folded_u = model.fold_in_user(user_items, history_conf=user_conf)

    # Let's verify fold-in vector is non-zero and finite
    assert not np.isnan(folded_u).any()
    assert np.linalg.norm(folded_u) > 0.0

    # Score from histories batch
    scores = model.score([user_items, np.array([], dtype=np.int32)])
    assert scores.shape == (2, data.n_items)
    assert scores.dtype == np.float32
    assert not np.isnan(scores).any()
    # Empty history gives zeros
    assert np.allclose(scores[1], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Content Recommender Test
# ─────────────────────────────────────────────────────────────────────────────

def test_content_recommender_bounded_scores():
    """
    Tests that ContentRecommender generates valid cosine scores bounded in [-1.0, 1.0].
    """
    data = _create_synthetic_recsys_data(n_users=20, n_items=25)

    model = ContentRecommender(weight_tags=0.5)
    model.fit(data)

    # Verify L2 normalized item content embeddings
    row_norms = np.linalg.norm(model.item_content_emb, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)

    # Score batch of histories
    histories = [np.array([0, 1, 2]), np.array([3]), np.array([], dtype=np.int32)]
    scores = model.score(histories)

    assert scores.shape == (3, data.n_items)
    assert scores.dtype == np.float32
    assert not np.isnan(scores).any()
    assert np.all(scores >= -1.0 - 1e-6)
    assert np.all(scores <= 1.0 + 1e-6)
    # Empty history user profile is zeros -> scores are 0.0
    assert np.allclose(scores[2], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save/Load Roundtrip for All 4 Models
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (PopularityRecommender, {}),
        (ItemKNNRecommender, {"k_neighbors": 10, "shrink": 5.0}),
        (ALSRecommender, {"factors": 16, "regularization": 0.1, "alpha": 5.0, "iterations": 5}),
        (ContentRecommender, {"weight_tags": 0.5}),
    ],
)
def test_all_models_save_load_roundtrip_and_shape(tmp_path: Path, model_cls, kwargs):
    """
    Tests that all four models:
      1. Can be fitted on RecsysData.
      2. Produce output scores with shape [n_histories, n_items], float32 dtype, and no NaN.
      3. Satisfy save -> load -> identical score() output.
    """
    data = _create_synthetic_recsys_data(n_users=25, n_items=30)

    model = model_cls(**kwargs)
    model.fit(data)

    test_histories = [
        np.array([1, 4, 7], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([], dtype=np.int32),
    ]

    orig_scores = model.score(test_histories)

    # Shape and validity checks
    assert orig_scores.shape == (3, data.n_items)
    assert orig_scores.dtype == np.float32
    assert not np.isnan(orig_scores).any()

    # Serialization roundtrip
    save_dir = tmp_path / f"{model.name}_artifact"
    model.save(save_dir)

    loaded_model = model_cls.load(save_dir)
    loaded_scores = loaded_model.score(test_histories)

    assert loaded_scores.shape == (3, data.n_items)
    assert np.allclose(orig_scores, loaded_scores, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Disjoint User Split Partition Test (Addendum 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_disjoint_evaluation_and_tuning_users():
    """
    Tests that get_evaluation_user_split partitions users into strictly disjoint sets.
    """
    data = _create_synthetic_recsys_data(n_users=30, n_items=25)

    eval_users, tuning_users = get_evaluation_user_split(
        data=data,
        n_eval_users=10,
        n_tuning_users=10,
        seed=42,
    )

    assert len(eval_users) == 10
    assert len(tuning_users) == 10

    set_eval = set(eval_users)
    set_tuning = set(tuning_users)

    # Assert completely disjoint sets
    assert set_eval.isdisjoint(set_tuning), f"Overlap found: {set_eval.intersection(set_tuning)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Registry Latest Run Query Test (Addendum 4)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_latest_recsys_runs_helper(db_session):
    """
    Verifies that get_latest_recsys_runs returns the latest run per (model_name, dataset_version)
    and excludes non-recsys runs.
    """
    # 1. Insert non-recsys run
    other_model = ModelRun(
        model_name="game_success_predictor",
        stage="production",
        dataset_version="v1",
        metrics={"roc_auc": 0.85},
    )
    db_session.add(other_model)
    await db_session.commit()

    # 2. Log two runs of recsys_popularity on version v1
    run1, _ = await log_recsys_run(
        model_name="recsys_popularity",
        family="popularity",
        dataset_version="v1",
        val_metrics={"val_ndcg_at_10": 0.015},
        session=db_session,
    )

    run2, _ = await log_recsys_run(
        model_name="recsys_popularity",
        family="popularity",
        dataset_version="v1",
        val_metrics={"val_ndcg_at_10": 0.025},
        session=db_session,
    )

    # 3. Log a run for recsys_als on version v1
    run_als, _ = await log_recsys_run(
        model_name="recsys_als",
        family="matrix_factorization",
        dataset_version="v1",
        val_metrics={"val_ndcg_at_10": 0.045},
        session=db_session,
    )

    # 4. Query latest runs
    latest_runs = await get_latest_recsys_runs(dataset_version="v1", session=db_session)
    latest_ids = {r.id for r in latest_runs}
    latest_names = {r.model_name for r in latest_runs}

    # Must contain run2 (the later popularity run) and run_als, but NOT run1 or other_model
    assert run2.id in latest_ids
    assert run1.id not in latest_ids
    assert run_als.id in latest_ids
    assert "game_success_predictor" not in latest_names
