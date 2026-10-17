"""
Unit tests for recsys audit: rank array mathematical properties,
metric reproducibility from saved rank arrays, TwoTower loading error guard,
and model promotion tournament logic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from jobs.promote_recsys import PromoteRecsysJob
from recsys.data import RecsysData
from recsys.evaluate import compute_metrics_from_ranks
from recsys.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel, TwoTowerModule

# ─────────────────────────────────────────────────────────────────────────────
# 1. Rank Array Mathematical Ordering Test
# ─────────────────────────────────────────────────────────────────────────────

def test_hand_built_rank_array_metric_ordering():
    """
    Step 2 Test: For ANY rank array, MRR@10 <= NDCG@10 <= HR@10 holds for every user and in the mean.
    Proof:
      - For r = 1: 1/1 = 1 <= 1/log2(2) = 1 <= 1 (equality).
      - For 2 <= r <= 10: log2(r+1) <= r => 1/r <= 1/log2(r+1) <= 1.
      - For r > 10: 0 <= 0 <= 0 (equality).
    """
    hand_built_ranks = np.array([1, 2, 3, 4, 5, 7, 9, 10, 11, 20, 50, 100, 3000], dtype=np.int64)

    mrr10 = mrr_at_k(hand_built_ranks, 10)
    ndcg10 = ndcg_at_k(hand_built_ranks, 10)
    hr10 = hit_rate_at_k(hand_built_ranks, 10)

    # Pointwise comparison for EVERY user
    assert np.all(mrr10 <= ndcg10 + 1e-12), f"Violated MRR <= NDCG: MRR={mrr10}, NDCG={ndcg10}"
    assert np.all(ndcg10 <= hr10 + 1e-12), f"Violated NDCG <= HR: NDCG={ndcg10}, HR={hr10}"

    # Mean comparison
    mean_mrr = float(np.mean(mrr10))
    mean_ndcg = float(np.mean(ndcg10))
    mean_hr = float(np.mean(hr10))

    assert mean_mrr <= mean_ndcg <= mean_hr
    assert mean_mrr > 0.0
    assert mean_ndcg > mean_mrr
    assert mean_hr > mean_ndcg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metric Reproducibility from Saved Rank Array
# ─────────────────────────────────────────────────────────────────────────────

def test_recomputing_metrics_from_rank_array(tmp_path: Path):
    """
    Step 2 Test: Recomputing all metrics from a saved rank array reproduces the stored metrics exactly.
    """
    n_users = 20
    n_items = 50
    rng = np.random.default_rng(42)

    fake_ranks = rng.integers(1, n_items + 1, size=n_users, dtype=np.int64)
    fake_targets = rng.integers(0, n_items, size=n_users, dtype=np.int64)
    fake_history_lens = rng.integers(3, 25, size=n_users, dtype=np.int64)
    fake_item_pop = rng.integers(1, 1000, size=n_items, dtype=np.int64)

    dummy_data = RecsysData(
        train_df=pd.DataFrame(),
        val_df=pd.DataFrame({"user_idx": np.arange(n_users), "item_idx": fake_targets}),
        test_df=pd.DataFrame({"user_idx": np.arange(n_users), "item_idx": fake_targets}),
        negatives_df=pd.DataFrame(),
        users_df=pd.DataFrame({"user_idx": np.arange(n_users)}),
        mappings_df=pd.DataFrame(),
        items_df=pd.DataFrame({"item_idx": np.arange(n_items)}),
        item_text_emb=np.zeros((n_items, 384), dtype=np.float32),
        item_train_pop=fake_item_pop,
        item_tags_sparse=sp.csr_matrix((n_items, 10)),
        dataset_version="test_v1",
        manifest={},
    )

    metrics_1, per_user_1, slices_1 = compute_metrics_from_ranks(
        ranks=fake_ranks,
        top_k_indices=None,
        targets=fake_targets,
        history_lengths=fake_history_lens,
        data=dummy_data,
        k_list=(10, 20),
        seed=42,
    )

    # Save ranks to disk (.npy) and reload
    rank_file = tmp_path / "ranks_val.npy"
    np.save(rank_file, fake_ranks)
    loaded_ranks = np.load(rank_file)

    metrics_2, per_user_2, slices_2 = compute_metrics_from_ranks(
        ranks=loaded_ranks,
        top_k_indices=None,
        targets=fake_targets,
        history_lengths=fake_history_lens,
        data=dummy_data,
        k_list=(10, 20),
        seed=42,
    )

    for k in metrics_1:
        assert math.isclose(metrics_1[k], metrics_2[k], rel_tol=1e-9), f"Mismatch for metric {k}"

    for bucket in slices_1:
        for sk in slices_1[bucket]:
            assert math.isclose(slices_1[bucket][sk], slices_2[bucket][sk], rel_tol=1e-9), f"Mismatch for slice {bucket}.{sk}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TwoTower Load Path Guard Test
# ─────────────────────────────────────────────────────────────────────────────

def test_two_tower_load_missing_features_raises(tmp_path: Path):
    """
    Step 4 Test: TwoTowerModel.load raises ValueError if item_features.pt is missing and no data is provided.
    """
    cfg = TwoTowerConfig(n_items=10, text_dim=384, tag_dim=10, numeric_dim=4)
    module = TwoTowerModule(cfg)

    # Save partial model (without item_features.pt)
    torch.save(module.state_dict(), tmp_path / "two_tower.pt")
    with open(tmp_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f)
    np.save(tmp_path / "item_vecs.npy", np.zeros((10, 64), dtype=np.float32))

    # Calling load without data must raise ValueError
    with pytest.raises(ValueError, match="item_features.pt.*is missing"):
        TwoTowerModel.load(tmp_path, data=None)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Promotion Job Execution Test
# ─────────────────────────────────────────────────────────────────────────────

def test_promote_recsys_job_dry_run():
    """
    Step 5 Test: PromoteRecsysJob executes in dry_run mode on real dataset without errors.
    """
    real_data_dir = Path(__file__).resolve().parent.parent / "data" / "recsys"
    job = PromoteRecsysJob(data_dir=real_data_dir, profile="dev", dry_run=True)
    res = job.execute()

    assert "promoted_model_name" in res
    assert "promoted_family" in res
    assert "val_ndcg_at_10" in res
    assert "reasoning" in res
    assert res["val_ndcg_at_10"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Timestamp-Aware Masking Audit Test (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_timestamp_aware_masking_synthetic():
    """
    Phase 1 Test: Proves that in evaluate_model / build_user_evaluation_inputs:
      (a) An earlier disliked game (neg_ts < target_ts) IS masked.
      (b) The evaluation target item itself is NEVER masked.
      (c) A later interaction (ts > target_ts) is NOT masked.
    """
    from recsys.evaluate import build_user_evaluation_inputs

    # Setup 1 user, 10 items
    u = 0
    # Items:
    # 1: train positive at ts=100
    # 2: held-out negative at ts=150 (earlier than test target) -> MUST BE MASKED
    # 3: negative at ts=300 (later than test target) -> MUST NOT BE MASKED
    # 4: test target at ts=200 -> MUST NEVER BE MASKED
    # 5: validation positive at ts=120 (earlier than test target) -> MUST BE MASKED
    # 6: unconsumed item -> MUST NOT BE MASKED

    train_df = pd.DataFrame([{"user_idx": u, "item_idx": 1, "ts": 100.0, "hours": 10.0}])
    val_df = pd.DataFrame([{"user_idx": u, "item_idx": 5, "ts": 120.0, "hours": 5.0}])
    test_df = pd.DataFrame([{"user_idx": u, "item_idx": 4, "ts": 200.0, "hours": 8.0}])
    negatives_df = pd.DataFrame([
        {"user_idx": u, "item_idx": 2, "ts": 150.0, "hours": 1.0, "split": "holdout"},
        {"user_idx": u, "item_idx": 3, "ts": 300.0, "hours": 0.5, "split": "holdout"},
    ])
    users_df = pd.DataFrame([{"user_idx": u, "products": 5, "reviews": 3}])
    mappings_df = pd.DataFrame([{"item_idx": i, "app_id": 1000 + i} for i in range(10)])
    items_df = pd.DataFrame([{"item_idx": i, "name": f"Game {i}"} for i in range(10)])

    synthetic_data = RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=negatives_df,
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=np.zeros((10, 384), dtype=np.float32),
        item_train_pop=np.ones(10, dtype=np.float32),
        item_tags_sparse=sp.csr_matrix((10, 50), dtype=np.float32),
        dataset_version="test_v1",
        manifest={},
    )

    histories, targets, seen_masks, history_lengths = build_user_evaluation_inputs(
        data=synthetic_data,
        eval_users=[u],
        split="test",
        max_history=50,
    )

    seen = set(seen_masks[0])

    # (a) Earlier disliked game (item 2, ts=150 < 200) MUST be masked
    assert 2 in seen, "Earlier disliked game (ts=150) must be included in seen mask"

    # (b) The target item itself (item 4, ts=200) MUST NEVER be masked
    assert 4 not in seen, "Target item must NEVER be masked"
    assert targets[0] == 4

    # (c) Later interaction (item 3, ts=300 > 200) MUST NOT be masked
    assert 3 not in seen, "Later interaction (ts=300 > target_ts=200) must NOT be masked"

    # Also verify train positive (item 1) and val positive (item 5) are masked
    assert 1 in seen, "Train positive (ts=100) must be masked"
    assert 5 in seen, "Val positive (ts=120) prior to test target must be masked"
    # Unconsumed item 6 must not be masked
    assert 6 not in seen, "Unconsumed item 6 must not be masked"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Provenance Isolation Test (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_provenance_train_only_isolation():
    """
    Phase 1 Test: Proves UserFeatureCache(train_stats_only=True) uses strictly
    train-period interactions and never leaks post-cutoff crawl-time totals from users_df.
    """
    from recsys.ranker_features import UserFeatureCache

    u = 42
    # users_df has post-cutoff total counts from crawl time (e.g. 500 products, 200 reviews)
    users_df = pd.DataFrame([{"user_idx": u, "products": 500, "reviews": 200}])

    # train_df has only 3 train interactions
    train_df = pd.DataFrame([
        {"user_idx": u, "item_idx": 0, "ts": 100.0, "hours": 2.0},
        {"user_idx": u, "item_idx": 1, "ts": 110.0, "hours": 5.0},
        {"user_idx": u, "item_idx": 2, "ts": 120.0, "hours": 10.0},
    ])
    # negatives_df has 1 train negative and 5 post-cutoff holdout negatives
    negatives_df = pd.DataFrame([
        {"user_idx": u, "item_idx": 3, "ts": 130.0, "hours": 1.0, "split": "train"},
        {"user_idx": u, "item_idx": 4, "ts": 500.0, "hours": 1.0, "split": "holdout"},
        {"user_idx": u, "item_idx": 5, "ts": 510.0, "hours": 1.0, "split": "holdout"},
    ])

    val_df = pd.DataFrame([{"user_idx": u, "item_idx": 6, "ts": 300.0, "hours": 4.0}])
    test_df = pd.DataFrame([{"user_idx": u, "item_idx": 7, "ts": 400.0, "hours": 6.0}])
    mappings_df = pd.DataFrame([{"item_idx": i, "app_id": 1000 + i} for i in range(10)])
    items_df = pd.DataFrame([{"item_idx": i, "name": f"Game {i}"} for i in range(10)])

    test_data = RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=negatives_df,
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=np.zeros((10, 384), dtype=np.float32),
        item_train_pop=np.ones(10, dtype=np.float32),
        item_tags_sparse=sp.csr_matrix((10, 50), dtype=np.float32),
        dataset_version="test_v1",
        manifest={},
    )

    # 1. train_stats_only=True: strictly train interactions
    clean_cache = UserFeatureCache(test_data, train_stats_only=True)
    expected_clean_prods = float(np.log1p(3))  # 3 train positives
    expected_clean_revs = float(np.log1p(3 + 1))  # 3 train pos + 1 train neg = 4

    assert math.isclose(clean_cache.user_log_products[u], expected_clean_prods, rel_tol=1e-5), (
        f"Expected train-only log_products {expected_clean_prods}, got {clean_cache.user_log_products[u]}"
    )
    assert math.isclose(clean_cache.user_log_reviews[u], expected_clean_revs, rel_tol=1e-5), (
        f"Expected train-only log_reviews {expected_clean_revs}, got {clean_cache.user_log_reviews[u]}"
    )

    # 2. train_stats_only=False (legacy mode): uses crawl-time users_df
    leaky_cache = UserFeatureCache(test_data, train_stats_only=False)
    assert math.isclose(leaky_cache.user_log_products[u], float(np.log1p(500)), rel_tol=1e-5)
    assert math.isclose(leaky_cache.user_log_reviews[u], float(np.log1p(200)), rel_tol=1e-5)

