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
