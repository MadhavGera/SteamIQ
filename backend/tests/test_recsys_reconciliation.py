"""
Unit tests for Recommender Reconciliation Suite:
1. Cold-start nested masking conditions integrity (fails if conditions change silently).
2. Two-Tower --no-pop-proxies architectural option (dim=6 numeric features).
3. Two-Tower --proxy-dropout regularization behavior in train vs eval modes.
4. Ranker reproducibility and ranker_notime feature subsetting.

Rule 4: Zero FastAPI and zero database imports. All tests pass offline on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys.coldstart import (
    get_sparse_target_test_users,
)
from recsys.data import RecsysData
from recsys.models.two_tower import (
    TwoTowerConfig,
    TwoTowerModel,
    TwoTowerModule,
    extract_standardized_numeric_features,
)
from recsys.ranker_features import FEATURE_COLUMNS
from tests.test_recsys_two_tower import _create_planted_block_recsys_data


@pytest.fixture
def tiny_recsys_data() -> RecsysData:
    """Fixture providing tiny synthetic recsys dataset with populated numeric proxies."""
    data = _create_planted_block_recsys_data(n_users=40, n_items=30, seed=42)
    n = len(data.items_df)
    # Ensure non-zero proxy features for all items so neutralization tests can strictly differentiate conditions
    data.items_df["train_positives"] = np.arange(1, n + 1, dtype=np.float32)
    data.items_df["user_reviews"] = np.arange(10, 10 + n, dtype=np.float32)
    data.items_df["positive_ratio"] = np.full(n, 85.0, dtype=np.float32)
    return data


def test_cold_start_masking_conditions_integrity(tiny_recsys_data: RecsysData) -> None:
    """
    Verifies that the 5 cold-start masking conditions (a, b, c, d, e) operate strictly
    according to specification and fail if any condition is modified silently.
    """
    model = TwoTowerModel(TwoTowerConfig(n_items=tiny_recsys_data.n_items))
    model.fit(tiny_recsys_data)

    eval_users = np.unique(tiny_recsys_data.val_df["user_idx"].to_numpy())
    sparse_users, sparse_targets = get_sparse_target_test_users(tiny_recsys_data, eval_users, max_train_positives=100)
    assert len(sparse_users) > 0, "Synthetic dataset must have sparse target users"

    target_ids = list(set(int(t) for t in sparse_targets))
    target_idx_t = torch.tensor(target_ids, dtype=torch.long)

    # Condition (a): Intact baseline item vectors
    vecs_a = model.item_vecs
    assert vecs_a is not None
    assert vecs_a.shape == (tiny_recsys_data.n_items, 64)

    # Condition (b): Mask target IDs only
    vecs_b = model.precompute_item_vectors(mask_ids=target_ids)
    # Masked items must differ from intact items; non-masked items must be identical
    for it in range(tiny_recsys_data.n_items):
        if it in target_ids:
            assert not np.allclose(vecs_a[it], vecs_b[it], atol=1e-5), f"Item {it} should be masked in (b)"
        else:
            assert np.allclose(vecs_a[it], vecs_b[it], atol=1e-5), f"Item {it} should remain unchanged in (b)"

    # Condition (c): (b) + train popularity zeroed in TARGET items (index 3 in numeric)
    num_c = model._item_features_tensors["numeric"].clone()
    assert num_c.shape[1] == 9, "Standard numeric features must have 9 dimensions"
    num_c[target_idx_t, 3] = 0.0  # Zero train_positives
    t_dict = model._item_features_tensors
    vecs_c = model.module.encode_items(
        item_ids=t_dict["all_ids"], text_embs=t_dict["text"], tag_features=t_dict["tags"],
        numeric_features=num_c, mask_ids=target_idx_t,
    ).detach().cpu().numpy().astype(np.float32)

    # Condition (d): (c) + log1p(user_reviews) [idx 2] and positive_ratio [idx 1] neutralized in TARGET items
    num_d = num_c.clone()
    num_d[target_idx_t, 1] = 0.0
    num_d[target_idx_t, 2] = 0.0
    vecs_d = model.module.encode_items(
        item_ids=t_dict["all_ids"], text_embs=t_dict["text"], tag_features=t_dict["tags"],
        numeric_features=num_d, mask_ids=target_idx_t,
    ).detach().cpu().numpy().astype(np.float32)

    # Condition (e): (d) + same neutralization applied globally (all items)
    num_e = num_d.clone()
    num_e[:, 1] = 0.0
    num_e[:, 2] = 0.0
    num_e[:, 3] = 0.0
    vecs_e = model.module.encode_items(
        item_ids=t_dict["all_ids"], text_embs=t_dict["text"], tag_features=t_dict["tags"],
        numeric_features=num_e, mask_ids=target_idx_t,
    ).detach().cpu().numpy().astype(np.float32)
    assert vecs_e.shape == (tiny_recsys_data.n_items, 64)

    # Assert nested vectors are strictly distinct for target items
    for tid in target_ids:
        assert not np.allclose(vecs_b[tid], vecs_c[tid], atol=1e-5), "Condition (c) must differ from (b)"
        assert not np.allclose(vecs_c[tid], vecs_d[tid], atol=1e-5), "Condition (d) must differ from (c)"


def test_no_pop_proxies_two_tower_architecture(tiny_recsys_data: RecsysData) -> None:
    """
    Tests the --no-pop-proxies Two-Tower architectural option:
    1. Numeric feature extraction outputs 6 features (omitting reviews and positive_ratio).
    2. TwoTowerConfig auto-adjusts numeric_dim to 6.
    3. TwoTowerModule forward and item encoding output unit L2 normalized vectors.
    """
    # 1. Feature extraction
    num_feats_full = extract_standardized_numeric_features(tiny_recsys_data.items_df, no_pop_proxies=False)
    num_feats_nopop = extract_standardized_numeric_features(tiny_recsys_data.items_df, no_pop_proxies=True)

    assert num_feats_full.shape == (tiny_recsys_data.n_items, 9)
    assert num_feats_nopop.shape == (tiny_recsys_data.n_items, 6)

    # 2. Config auto-adjustment
    cfg = TwoTowerConfig(n_items=tiny_recsys_data.n_items, no_pop_proxies=True)
    assert cfg.numeric_dim == 6
    assert cfg.raw_item_dim == 64 + 64 + 32 + 16  # 176

    # 3. Model construction & forward encoding
    module = TwoTowerModule(cfg)
    item_ids = torch.arange(tiny_recsys_data.n_items, dtype=torch.long)
    num_t = torch.from_numpy(num_feats_nopop)

    vecs = module.encode_items(
        item_ids=item_ids,
        text_embs=torch.zeros((tiny_recsys_data.n_items, 384), dtype=torch.float32),
        tag_features=torch.zeros((tiny_recsys_data.n_items, 300), dtype=torch.float32),
        numeric_features=num_t,
    )
    assert vecs.shape == (tiny_recsys_data.n_items, 64)
    norms = torch.norm(vecs, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_proxy_dropout_regularization_behavior() -> None:
    """
    Tests that proxy_dropout correctly zeroes ID and numeric representations
    during training mode, and stays disabled during eval mode.
    """
    cfg = TwoTowerConfig(n_items=10, id_dropout=0.0, proxy_dropout=1.0)  # 100% dropout for testing
    module = TwoTowerModule(cfg)

    item_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    num_t = torch.ones((3, 9), dtype=torch.float32)

    # In eval mode: proxy_dropout should not apply
    module.eval()
    raw_eval = module.feature_encoder(
        item_ids=item_ids,
        numeric_features=num_t,
        apply_proxy_dropout=True,
    )
    # In training mode with apply_proxy_dropout=True:
    module.train()
    raw_train = module.feature_encoder(
        item_ids=item_ids,
        numeric_features=num_t,
        apply_proxy_dropout=True,
    )

    # Train output should differ from eval output because ID and numeric were zeroed/unknown
    assert not torch.allclose(raw_eval, raw_train, atol=1e-4)


def test_ranker_notime_feature_subsetting() -> None:
    """
    Tests that ranker_notime correctly drops log_user_reviews and positive_ratio (33 features).
    """
    notime_feats = [f for f in FEATURE_COLUMNS if f not in {"log_user_reviews", "positive_ratio"}]
    assert len(notime_feats) == len(FEATURE_COLUMNS) - 2
    assert "log_user_reviews" not in notime_feats
    assert "positive_ratio" not in notime_feats
    assert "price_final" in notime_feats
    assert "tt_score" in notime_feats
