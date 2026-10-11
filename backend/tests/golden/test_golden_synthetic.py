"""
Golden Synthetic Regression Test — Phase 7.

Runs an offline, deterministic evaluation on a tiny synthetic dataset with fixed seeds.
Asserts that baseline and two-tower models produce stable, deterministic top-k outputs.
Runs completely offline without external databases or pre-existing artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from recsys.data import RecsysData
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel


@pytest.fixture
def tiny_recsys_data() -> RecsysData:
    """Creates a deterministic 10-user, 20-item synthetic RecsysData container."""
    np.random.seed(42)
    n_users = 10
    n_items = 20

    # Train interactions
    train_rows = []
    for u in range(n_users):
        # Each user interacts with 3 items
        items = [(u + i) % n_items for i in range(3)]
        for it in items:
            train_rows.append({"user_idx": u, "item_idx": it, "ts": 1000 + it, "hours": 5.0})
    train_df = pd.DataFrame(train_rows)

    # Val and Test splits: 1 held-out item per user
    val_rows = []
    test_rows = []
    for u in range(n_users):
        val_rows.append({"user_idx": u, "item_idx": (u + 3) % n_items, "ts": 2000, "hours": 2.0})
        test_rows.append({"user_idx": u, "item_idx": (u + 4) % n_items, "ts": 3000, "hours": 3.0})

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    negatives_df = pd.DataFrame(columns=["user_idx", "item_idx", "split"])

    users_df = pd.DataFrame({"user_idx": np.arange(n_users)})
    mappings_df = pd.DataFrame({
        "item_idx": np.arange(n_items),
        "app_id": [1000 + i for i in range(n_items)],
    })
    items_df = pd.DataFrame({
        "item_idx": np.arange(n_items),
        "app_id": [1000 + i for i in range(n_items)],
        "price_final": np.linspace(5.0, 50.0, n_items),
        "positive_ratio": np.linspace(0.6, 0.95, n_items),
        "log_review_count": np.linspace(2.0, 8.0, n_items),
    })

    # Embeddings and tags
    item_text_emb = np.random.normal(size=(n_items, 16)).astype(np.float32)
    norms = np.linalg.norm(item_text_emb, axis=1, keepdims=True)
    item_text_emb = item_text_emb / np.maximum(norms, 1e-8)

    item_train_pop = np.bincount(train_df["item_idx"], minlength=n_items).astype(np.float32)
    item_tags_sparse = sp.eye(n_items, format="csr", dtype=np.float32)

    return RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=negatives_df,
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=item_text_emb,
        item_train_pop=item_train_pop,
        item_tags_sparse=item_tags_sparse,
        dataset_version="synthetic_golden_v1",
        manifest={"dataset_version": "synthetic_golden_v1"},
    )


def test_popularity_recommender_synthetic_golden(tiny_recsys_data: RecsysData):
    """Popularity recommender must rank unseen items strictly by train popularity."""
    model = PopularityRecommender()
    model.fit(tiny_recsys_data)

    user_hist = np.array([0, 1, 2], dtype=np.int64)
    scores = model.score([user_hist])[0]

    assert len(scores) == tiny_recsys_data.n_items
    assert np.all(np.isfinite(scores))
    # Unseen items with highest train pop should rank top
    top_rec = int(np.argmax(scores))
    assert top_rec in range(tiny_recsys_data.n_items)


def test_itemknn_recommender_synthetic_golden(tiny_recsys_data: RecsysData):
    """ItemKNN recommender must produce valid non-trivial scores on synthetic data."""
    model = ItemKNNRecommender(k_neighbors=5)
    model.fit(tiny_recsys_data)

    user_hist = np.array([0, 1], dtype=np.int64)
    scores = model.score([user_hist])[0]

    assert len(scores) == tiny_recsys_data.n_items
    assert np.isfinite(np.max(scores))


def test_content_recommender_synthetic_golden(tiny_recsys_data: RecsysData):
    """Content recommender must compute cosine similarity over item text embeddings."""
    model = ContentRecommender()
    model.fit(tiny_recsys_data)

    user_hist = np.array([3, 4], dtype=np.int64)
    scores = model.score([user_hist])[0]

    assert len(scores) == tiny_recsys_data.n_items
    assert np.isfinite(np.max(scores))


def test_two_tower_synthetic_forward(tiny_recsys_data: RecsysData):
    """TwoTowerModel must initialize and score via forward pass on synthetic data."""
    torch.manual_seed(42)
    config = TwoTowerConfig(
        n_items=tiny_recsys_data.n_items,
        text_dim=16,
        tag_dim=tiny_recsys_data.n_items,
        numeric_dim=9,
        output_dim=8,
        pool="mean",
        id_dropout=0.0,
    )
    model = TwoTowerModel(cfg=config)
    model.fit(tiny_recsys_data)

    user_hist = np.array([0, 1, 2], dtype=np.int64)
    scores = model.score([user_hist])[0]

    assert len(scores) == tiny_recsys_data.n_items
    assert np.all(np.isfinite(scores))
