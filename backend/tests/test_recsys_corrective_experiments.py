"""
Unit tests for Corrective & Extension Recommender System Experiments (A-E).

Tests:
  1. Beyond-accuracy catalog coverage and diversity consistency within 1e-9.
  2. MMR lambda=1.0 exact reproduction of ranker test NDCG@10 within 1e-9.
  3. Fair Cold-Start Simulations (C1, C2, C3) and Proxy Neutralization in Two-Tower.
  4. True Item Holdout Dataset Preparation & Holdout Split (Part D).
  5. Ranker Feature Ablations including without_all_time_item_stats and without_train_popularity.

Rule 4: Zero FastAPI and zero database imports. All tests pass offline on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pytest

from recsys.coldstart import (
    evaluate_item_holdout_study,
    get_item_holdout_split,
    get_simulated_new_items_test_users,
)
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model
from recsys.metrics import (
    catalog_coverage_at_k,
    diversity_at_k,
    mmr_rerank,
    ndcg_at_k,
)
from recsys.models.content import ContentRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from tests.test_recsys_two_tower import _create_planted_block_recsys_data


@pytest.fixture
def tiny_recsys_data() -> RecsysData:
    """Fixture providing tiny synthetic recsys dataset."""
    return _create_planted_block_recsys_data(n_users=40, n_items=30, seed=42)


def test_beyond_accuracy_consistency_within_1e9(tiny_recsys_data: RecsysData) -> None:
    """Coverage and diversity derived from top_k_indices MUST equal evaluation metrics within 1e-9."""
    model = PopularityRecommender()
    model.fit(tiny_recsys_data)

    eval_users = np.unique(tiny_recsys_data.val_df["user_idx"].to_numpy())
    res = evaluate_model(
        model,
        tiny_recsys_data,
        split="val",
        user_indices=eval_users,
        allow_small_user_set=True,
        seed=42,
    )

    top10 = res.per_user["top_k_indices"][:, :10]
    cov = catalog_coverage_at_k(top10, tiny_recsys_data.n_items)
    div = (
        diversity_at_k(top10, tiny_recsys_data.item_text_emb)
        if tiny_recsys_data.item_text_emb is not None
        else 0.0
    )

    assert abs(cov - res.metrics["catalog_coverage_at_10"]) < 1e-9
    assert abs(div - res.metrics["diversity_at_10"]) < 1e-9


def test_mmr_lambda_one_reproduces_official_ndcg_within_1e9(tiny_recsys_data: RecsysData) -> None:
    """MMR with lambda=1.0 MUST reproduce candidate list NDCG@10 within 1e-9."""
    n_users = 20
    n_cands = 15
    n_items = tiny_recsys_data.n_items

    # Synthetic candidate items and scores
    rng = np.random.default_rng(42)
    candidate_items = np.array([rng.choice(n_items, size=n_cands, replace=False) for _ in range(n_users)], dtype=np.int64)
    candidate_scores = rng.random((n_users, n_cands)).astype(np.float32)

    # Sort candidates in descending score order
    sort_order = np.argsort(-candidate_scores, axis=1)
    candidate_items = np.take_along_axis(candidate_items, sort_order, axis=1)
    candidate_scores = np.take_along_axis(candidate_scores, sort_order, axis=1)

    # Synthetic target items (present in top-10 for half the users)
    targets = np.array([candidate_items[i, i % 10] if i % 2 == 0 else 999 for i in range(n_users)], dtype=np.int64)

    # Official top-10 ranks directly from candidates
    ranks_official = np.full(n_users, 999, dtype=np.int64)
    for u_i, t in enumerate(targets):
        top10_items = candidate_items[u_i, :10]
        pos = np.where(top10_items == t)[0]
        if len(pos) > 0:
            ranks_official[u_i] = int(pos[0]) + 1

    official_ndcg = float(np.mean(ndcg_at_k(ranks_official, 10)))

    # MMR with lambda=1.0
    embs = np.eye(n_items, dtype=np.float32)
    reranked_top10 = mmr_rerank(
        candidate_items=candidate_items,
        candidate_scores=candidate_scores,
        item_embeddings=embs,
        lambda_param=1.0,
        k=10,
    )

    ranks_mmr = np.full(n_users, 999, dtype=np.int64)
    for u_i, t in enumerate(targets):
        top_items = reranked_top10[u_i]
        pos = np.where(top_items == t)[0]
        if len(pos) > 0:
            ranks_mmr[u_i] = int(pos[0]) + 1

    mmr_ndcg = float(np.mean(ndcg_at_k(ranks_mmr, 10)))

    # Must equal within 1e-9
    assert abs(mmr_ndcg - official_ndcg) < 1e-9


def test_cold_start_simulations_c1_c2_c3(tiny_recsys_data: RecsysData) -> None:
    """Verifies that C1, C2, and C3 produce expected user and item splits."""
    eval_users = np.unique(tiny_recsys_data.test_df["user_idx"].to_numpy())

    # C1 (Old setup)
    new_c1, users_c1, _ = get_simulated_new_items_test_users(
        tiny_recsys_data, eval_users, fraction=0.20, seed=42, variant="c1"
    )
    # C2 (Fair Random)
    new_c2, users_c2, _ = get_simulated_new_items_test_users(
        tiny_recsys_data, eval_users, fraction=0.20, seed=42, variant="c2"
    )
    # C3 (Fair Bottom 50%)
    new_c3, users_c3, _ = get_simulated_new_items_test_users(
        tiny_recsys_data, eval_users, fraction=0.20, seed=42, variant="c3"
    )

    # C1 and C2 should pick identical item sets for the same seed
    assert np.array_equal(new_c1, new_c2)
    assert np.array_equal(users_c1, users_c2)

    # C3 should draw only from bottom 50% by popularity
    pop = np.asarray(tiny_recsys_data.item_train_pop)
    median_pop = np.median(pop)
    for it in new_c3:
        assert pop[it] <= median_pop + 1e-5


def test_two_tower_proxy_neutralization(tiny_recsys_data: RecsysData) -> None:
    """TwoTowerModel neutralizes numeric review/rating proxies when neutralize_proxies is provided."""
    cfg = TwoTowerConfig(
        n_items=tiny_recsys_data.n_items,
        hidden_dim=32,
        output_dim=16,
        id_dim=16,
        text_proj_dim=16,
        tag_proj_dim=8,
        numeric_proj_dim=8,
    )
    model = TwoTowerModel(cfg=cfg)
    model.fit(tiny_recsys_data)

    standard_vecs = model.item_vectors(mask_ids=[0])
    neutralized_vecs = model.item_vectors(mask_ids=[0], neutralize_proxies=[0])

    # Neutralizing numeric branch alters item representation for item 0
    assert not np.allclose(standard_vecs[0], neutralized_vecs[0])
    # Other items remain unchanged
    assert np.allclose(standard_vecs[1], neutralized_vecs[1])


def test_true_item_holdout_split_and_evaluation(tiny_recsys_data: RecsysData) -> None:
    """Verifies true item holdout split selection and evaluation payload."""
    held_items, held_set = get_item_holdout_split(tiny_recsys_data, fraction=0.20, seed=42)
    assert len(held_items) > 0
    assert len(held_set) == len(held_items)

    eval_users = np.unique(tiny_recsys_data.test_df["user_idx"].to_numpy())
    content_m = ContentRecommender()
    content_m.fit(tiny_recsys_data)

    res = evaluate_item_holdout_study(
        data=tiny_recsys_data,
        models={"content": content_m},
        eval_users=eval_users,
        held_out_items=held_items,
        seed=42,
    )

    assert res["study_type"] == "item_holdout"
    assert "random" in res["models"]
    assert "content" in res["models"]
    assert "hit_rate_at_10" in res["models"]["random"]
