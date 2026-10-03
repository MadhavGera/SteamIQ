"""
Unit tests for the PyTorch Two-Tower Hybrid Retrieval Recommender.

Tests include:
  (i) User and item vectors have unit L2 norm and expected shapes ([B, 64], [N, 64]).
  (ii) Next-item training pairs never contain the target item in their history.
  (iii) Loss decreases over training steps and the model overfits a planted block dataset (HR@5 > 0.8).
  (iv) In-batch false negatives (duplicate target items in a batch) are masked out.
  (v) ID dropout changes embeddings, and item_vectors(mask_ids=...) changes only masked items.
  (vi) Serialization roundtrip (save -> load) produces bitwise/numerically identical scores.
  (vii) Determinism: identical random seeds produce identical weights after training steps.
  (viii) Eval and tuning user sets are disjoint, persist to disk, and reload identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from recsys.data import RecsysData
from recsys.evaluate import evaluate_model, get_evaluation_user_split
from recsys.models.two_tower import (
    TwoTowerConfig,
    TwoTowerModel,
)
from recsys.train_two_tower import (
    build_next_item_training_pairs,
    in_batch_sampled_softmax_loss,
    train_two_tower_model,
)


def _create_planted_block_recsys_data(
    n_users: int = 60,
    n_items: int = 40,
    n_groups: int = 4,
    seed: int = 42,
) -> RecsysData:
    """
    Constructs a synthetic dataset with planted group structure:
    Users in group g interact exclusively with items in group g.
    Item content features (text, tags, numeric) also cluster by group.
    """
    rng = np.random.default_rng(seed)
    users_per_group = max(1, n_users // n_groups)
    items_per_group = max(8, n_items // n_groups)
    actual_items = items_per_group * n_groups

    train_rows = []
    val_rows = []
    test_rows = []
    neg_rows = []

    for u in range(n_users):
        group = min(u // users_per_group, n_groups - 1)
        group_items = list(range(group * items_per_group, (group + 1) * items_per_group))

        # Chronological sequence of 7 items in group
        chosen_items = rng.choice(group_items, size=min(7, len(group_items)), replace=False)

        # First 5 items go to train
        for t_idx, item in enumerate(chosen_items[:5]):
            train_rows.append({
                "user_idx": u,
                "item_idx": int(item),
                "ts": 100 + t_idx,
                "hours": float(rng.uniform(5.0, 50.0)),
            })

        # 6th item is val target
        val_rows.append({
            "user_idx": u,
            "item_idx": int(chosen_items[5 % len(chosen_items)]),
            "ts": 200,
            "hours": 10.0,
        })

        # 7th item is test target
        test_rows.append({
            "user_idx": u,
            "item_idx": int(chosen_items[6 % len(chosen_items)]),
            "ts": 300,
            "hours": 10.0,
        })

        # Random negative from other groups
        other_items = [i for i in range(actual_items) if i not in group_items]
        if not other_items:
            other_items = group_items
        neg_rows.append({
            "user_idx": u,
            "item_idx": int(rng.choice(other_items)),
            "ts": 150,
            "hours": 0.5,
            "split": "train",
        })

    # Content features: group-clustered text embeddings (384-d), tags (300-d), numeric
    text_emb = np.zeros((actual_items, 384), dtype=np.float32)
    tag_dense = np.zeros((actual_items, 300), dtype=np.float32)
    prices = np.zeros(actual_items, dtype=np.float32)
    review_scores = np.zeros(actual_items, dtype=np.float32)
    total_reviews = np.zeros(actual_items, dtype=np.float32)

    for i in range(actual_items):
        g = min(i // items_per_group, n_groups - 1)
        # Text embedding center per group
        center = np.zeros(384, dtype=np.float32)
        center[g * 90 : (g + 1) * 90] = 1.0
        text_emb[i] = center + rng.normal(0, 0.05, size=384).astype(np.float32)
        text_emb[i] /= np.linalg.norm(text_emb[i]) + 1e-8

        # Tags specific to group
        tag_dense[i, g * 50 : (g + 1) * 50] = 1.0

        # Numeric features
        prices[i] = 10.0 * (g + 1) + rng.uniform(-1, 1)
        review_scores[i] = 80.0 + 3.0 * g + rng.uniform(-2, 2)
        total_reviews[i] = 500.0 * (g + 1) + rng.uniform(0, 50)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    negatives_df = pd.DataFrame(neg_rows)

    users_df = pd.DataFrame({"user_idx": list(range(n_users)), "original_user_id": list(range(n_users))})
    mappings_df = pd.DataFrame({"item_idx": list(range(actual_items)), "app_id": [1000 + i for i in range(actual_items)]})
    items_df = pd.DataFrame({
        "item_idx": list(range(actual_items)),
        "name": [f"Game {i}" for i in range(actual_items)],
        "price": prices,
        "review_score": review_scores,
        "total_reviews": total_reviews,
    })

    # Item popularity counts
    item_train_pop = np.zeros(actual_items, dtype=np.float32)
    for i in train_df["item_idx"]:
        item_train_pop[i] += 1.0

    return RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=negatives_df,
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=text_emb,
        item_train_pop=item_train_pop,
        item_tags_sparse=sp.csr_matrix(tag_dense),
        dataset_version="planted_test_v1",
        manifest={"dataset_version": "planted_test_v1"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vector Norms and Shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_shapes_and_unit_l2_norm():
    """
    (i) Verify that user and item vectors have unit L2 norm and expected shapes ([B, 64], [N, 64]).
    """
    data = _create_planted_block_recsys_data(n_users=40, n_items=40)
    config = TwoTowerConfig(id_dim=64, hidden_dim=128, pool="attn")
    model = TwoTowerModel(config)
    model.fit(data)

    # Check precomputed item vectors
    assert model.item_vecs is not None
    assert model.item_vecs.shape == (data.n_items, 64)
    item_norms = np.linalg.norm(model.item_vecs, axis=1)
    np.testing.assert_allclose(item_norms, 1.0, atol=1e-5)

    # Check user vectors from scoring
    histories = [np.array([0, 1, 2]), np.array([3]), np.array([], dtype=int)]
    scores = model.score(histories)
    assert scores.shape == (3, data.n_items)

    # Forward pass through module directly
    model.module.eval()
    with torch.no_grad():
        dummy_hist = torch.tensor([[0, 1, 2, -1], [3, -1, -1, -1]], dtype=torch.long)
        user_vecs = model.module.encode_user(dummy_hist)
        assert user_vecs.shape == (2, 64)
        u_norms = torch.norm(user_vecs, p=2, dim=-1).numpy()
        np.testing.assert_allclose(u_norms, 1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. History Generation: Target Never in History
# ─────────────────────────────────────────────────────────────────────────────

def test_training_pairs_never_contain_target_in_history():
    """
    (ii) Verify that next-item pairs built from user trajectories never contain the target in history.
    """
    data = _create_planted_block_recsys_data(n_users=40, n_items=40)
    dataset, q_items = build_next_item_training_pairs(data, max_history=50)

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        hist_t, mask_t, target_t = dataset[idx]
        hist = hist_t[mask_t].numpy().tolist()
        target = int(target_t.item())
        assert len(hist) > 0
        assert target not in hist, f"Target item {target} was found inside history {hist}"
        assert len(hist) <= 50


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training & Planted Block Overfit Test
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_decreases_and_overfits_planted_blocks():
    """
    (iii) Loss decreases over training steps and model achieves HR@5 > 0.8 on planted block data.
    """
    data = _create_planted_block_recsys_data(n_users=60, n_items=40, n_groups=4, seed=123)
    config = TwoTowerConfig(
        id_dim=64,
        hidden_dim=128,
        id_dropout=0.0,
        pool="attn",
        use_id=True,
        use_text=True,
        use_tags=True,
        use_numeric=True,
    )
    model = TwoTowerModel(config)

    # Train for 35 epochs on planted dataset
    model, training_curves = train_two_tower_model(
        model=model,
        data=data,
        epochs=35,
        lr=5e-3,
        batch_size=32,
        pairs_per_epoch=500,
        patience=35,
        seed=123,
    )

    history = training_curves["history"]
    # Loss must decrease over training steps
    train_losses = [h["train_loss"] for h in history]
    assert len(train_losses) >= 10
    assert train_losses[-1] < train_losses[0], f"Loss did not decrease: {train_losses[0]} -> {train_losses[-1]}"

    # Evaluate on val split of planted data
    eval_res = evaluate_model(
        model=model,
        data=data,
        split="val",
        k_list=(5, 10),
        max_users=60,
        allow_small_user_set=True,
    )

    hr5 = eval_res.metrics.get("hit_rate_at_5", 0.0)
    assert hr5 >= 0.80, f"Planted data sanity check failed: HR@5 is {hr5:.4f} (expected >= 0.80)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. In-Batch Duplicate Item Masking (False Negatives)
# ─────────────────────────────────────────────────────────────────────────────

def test_duplicate_items_masked_as_false_negatives():
    """
    (iv) Construct a batch with a repeated item and assert the masked logits.
    """
    # Batch size = 4, embedding_dim = 64
    B = 4
    dim = 64
    rng = np.random.default_rng(42)

    user_vecs = torch.from_numpy(rng.normal(size=(B, dim)).astype(np.float32))
    user_vecs = torch.nn.functional.normalize(user_vecs, p=2, dim=-1)

    # Target items: item 7 appears at index 0 and index 2
    # Item indices: [7, 12, 7, 19]
    target_indices = torch.tensor([7, 12, 7, 19], dtype=torch.long)

    # Distinct item representations
    item_reps = torch.from_numpy(rng.normal(size=(25, dim)).astype(np.float32))
    item_reps = torch.nn.functional.normalize(item_reps, p=2, dim=-1)
    target_item_vecs = item_reps[target_indices]

    log_q = torch.zeros(B, dtype=torch.float32)
    log_tau = torch.tensor(np.log(0.07), dtype=torch.float32)

    loss, logits = in_batch_sampled_softmax_loss(
        user_vecs=user_vecs,
        target_item_vecs=target_item_vecs,
        target_item_indices=target_indices,
        log_q=log_q,
        log_tau=log_tau,
        return_logits=True,
    )

    # Logits shape: [4, 4]
    assert logits.shape == (4, 4)

    # Check false-negative mask:
    # Row 0 (item 7): col 0 is diagonal (pos), col 2 is identical item 7 (must be -1e9)
    assert logits[0, 2].item() <= -1e8
    # Row 2 (item 7): col 2 is diagonal (pos), col 0 is identical item 7 (must be -1e9)
    assert logits[2, 0].item() <= -1e8

    # Non-duplicate off-diagonals must be finite
    assert logits[0, 1].item() > -1e4
    assert logits[0, 3].item() > -1e4
    assert logits[1, 0].item() > -1e4
    assert logits[3, 2].item() > -1e4

    # Loss is finite
    assert torch.isfinite(loss)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ID Dropout & Masking for Cold Start
# ─────────────────────────────────────────────────────────────────────────────

def test_id_dropout_and_mask_ids_cold_start():
    """
    (v) Verify ID dropout modifies representations, and item_vectors(mask_ids=...)
        changes only the masked items (simulating cold start).
    """
    data = _create_planted_block_recsys_data(n_users=40, n_items=40)
    config = TwoTowerConfig(id_dim=64, hidden_dim=128, id_dropout=0.5)
    model = TwoTowerModel(config)
    model.fit(data)

    unmasked_vecs = model.item_vectors(mask_ids=None)
    assert unmasked_vecs.shape == (data.n_items, 64)

    # Mask items 2 and 5
    masked_items = [2, 5]
    masked_vecs = model.item_vectors(mask_ids=masked_items)

    # Masked items must differ from unmasked items (as ID embedding is replaced by unknown embedding)
    for m_idx in masked_items:
        diff = np.linalg.norm(masked_vecs[m_idx] - unmasked_vecs[m_idx])
        assert diff > 1e-4, f"Masked item {m_idx} was not changed by mask_ids"

    # Unmasked items must be exactly identical
    for u_idx in range(data.n_items):
        if u_idx not in masked_items:
            np.testing.assert_allclose(
                masked_vecs[u_idx],
                unmasked_vecs[u_idx],
                atol=1e-6,
                err_msg=f"Unmasked item {u_idx} was unexpectedly changed",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save and Load Roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_save_load_roundtrip_identical_scores(tmp_path: Path):
    """
    (vi) Verify that save -> load produces identical scoring output.
    """
    data = _create_planted_block_recsys_data(n_users=40, n_items=40)
    config = TwoTowerConfig(id_dim=64, hidden_dim=128, pool="mean")
    model = TwoTowerModel(config)
    model.fit(data)

    histories = [np.array([1, 4, 7]), np.array([2]), np.array([0, 3, 6, 9])]
    orig_scores = model.score(histories)

    save_dir = tmp_path / "two_tower_model"
    model.save(save_dir)

    loaded_model = TwoTowerModel.load(save_dir)
    loaded_scores = loaded_model.score(histories)

    np.testing.assert_allclose(orig_scores, loaded_scores, atol=1e-5)
    assert loaded_model.config.id_dim == model.config.id_dim
    assert loaded_model.config.pool == model.config.pool


# ─────────────────────────────────────────────────────────────────────────────
# 7. Seed Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_determinism():
    """
    (vii) Verify two runs with identical seed produce identical weights after 5 steps.
    """
    data = _create_planted_block_recsys_data(n_users=40, n_items=40, seed=10)

    config1 = TwoTowerConfig(id_dim=32, hidden_dim=64)
    model1 = TwoTowerModel(config1)
    model1, _ = train_two_tower_model(model1, data, epochs=5, batch_size=16, pairs_per_epoch=50, seed=42)

    config2 = TwoTowerConfig(id_dim=32, hidden_dim=64)
    model2 = TwoTowerModel(config2)
    model2, _ = train_two_tower_model(model2, data, epochs=5, batch_size=16, pairs_per_epoch=50, seed=42)

    for p1, p2 in zip(model1.module.parameters(), model2.module.parameters(), strict=True):
        torch.testing.assert_close(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 8. User Splits & Addendum Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_and_tuning_user_splits_disjoint_and_deterministic(tmp_path: Path):
    """
    (viii) Verify evaluation and tuning user sets are disjoint, persist to disk, and reload identically.
    """
    data = _create_planted_block_recsys_data(n_users=60, n_items=40)
    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=20,
        n_tuning_users=20,
        seed=42,
        persistence_dir=tmp_path,
    )

    assert len(eval_users) == 20
    assert len(tuning_users) == 20
    assert len(set(eval_users).intersection(set(tuning_users))) == 0

    # Verify JSON files exist on disk
    assert (tmp_path / "eval_users.json").exists()
    assert (tmp_path / "tuning_users.json").exists()

    # Reload from disk
    eval_reloaded, tuning_reloaded = get_evaluation_user_split(
        data,
        n_eval_users=20,
        n_tuning_users=20,
        persistence_dir=tmp_path,
    )

    np.testing.assert_array_equal(eval_users, eval_reloaded)
    np.testing.assert_array_equal(tuning_users, tuning_reloaded)


def test_early_stopping_uses_tuning_split():
    """
    Verify early stopping evaluates against the specified tuning user indices.
    """
    data = _create_planted_block_recsys_data(n_users=60, n_items=40)
    eval_users, tuning_users = get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, seed=42)

    config = TwoTowerConfig(id_dim=32, hidden_dim=64)
    model = TwoTowerModel(config)

    model, training_curves = train_two_tower_model(
        model=model,
        data=data,
        epochs=3,
        batch_size=32,
        pairs_per_epoch=50,
        val_user_indices=tuning_users,
        seed=42,
    )

    history = training_curves["history"]
    assert len(history) == 3
    for entry in history:
        assert "val_ndcg_at_10" in entry
        assert "val_hr_at_10" in entry
        assert "train_loss" in entry


def test_persisted_split_wrong_version_or_invalid_ids_raises(tmp_path: Path):
    """
    Step 6: A persisted split with a wrong dataset_version or out-of-range ids raises ValueError.
    """
    data = _create_planted_block_recsys_data(n_users=60, n_items=40)
    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=20,
        n_tuning_users=20,
        seed=42,
        persistence_dir=tmp_path,
    )

    # 1. Test wrong dataset_version
    eval_file = tmp_path / "eval_users.json"
    with open(eval_file, encoding="utf-8") as f:
        content = json.load(f)
    content["dataset_version"] = "mismatched_version_999"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(content, f)

    with pytest.raises(ValueError, match="dataset_version"):
        get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, persistence_dir=tmp_path)

    # Reset correct version
    content["dataset_version"] = data.dataset_version

    # 2. Test out-of-range user IDs
    content["eval_users"][0] = 999999
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(content, f)

    with pytest.raises(ValueError, match="out-of-range"):
        get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, persistence_dir=tmp_path)

    # 3. Test length mismatch
    content["eval_users"] = [1, 2, 3]  # too short
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(content, f)

    with pytest.raises(ValueError, match="length mismatch"):
        get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, persistence_dir=tmp_path)


def test_evaluate_model_guard_raises_on_small_user_set():
    """
    Step 6: evaluate_model guard raises if fewer than 1,000 users are evaluated without allow_small_user_set.
    """
    data = _create_planted_block_recsys_data(n_users=60, n_items=40)
    config = TwoTowerConfig(id_dim=32, hidden_dim=64, n_items=40)
    model = TwoTowerModel(config)
    model.fit(data)

    with pytest.raises(ValueError, match="safety threshold of 1,000 users"):
        evaluate_model(
            model=model,
            data=data,
            split="val",
            user_indices=np.arange(20),
            allow_small_user_set=False,
        )


def test_real_sized_synthetic_split_reloads_identically(tmp_path: Path):
    """
    Step 6: A real-sized synthetic split (5,000 eval, 5,000 tuning) persists and reloads identically.
    """
    # Create large synthetic dataset with 12,000 users
    n_users = 12000
    n_items = 100
    user_ids = np.arange(n_users)
    train_users = np.repeat(user_ids, 5)
    train_items = np.random.randint(0, n_items, size=len(train_users))
    train_df = pd.DataFrame({"user_idx": train_users, "item_idx": train_items, "ts": np.arange(len(train_users))})
    val_df = pd.DataFrame({"user_idx": user_ids, "item_idx": np.random.randint(0, n_items, size=n_users), "ts": np.full(n_users, 1000000)})
    test_df = pd.DataFrame({"user_idx": user_ids, "item_idx": np.random.randint(0, n_items, size=n_users), "ts": np.full(n_users, 2000000)})
    users_df = pd.DataFrame({"user_idx": user_ids})
    items_df = pd.DataFrame({"item_idx": np.arange(n_items)})
    mappings_df = pd.DataFrame({"item_idx": np.arange(n_items), "app_id": np.arange(n_items)})

    data = RecsysData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        negatives_df=pd.DataFrame(columns=["user_idx", "item_idx", "split"]),
        users_df=users_df,
        mappings_df=mappings_df,
        items_df=items_df,
        item_text_emb=np.zeros((n_items, 384), dtype=np.float32),
        item_train_pop=np.ones(n_items, dtype=np.float32),
        item_tags_sparse=sp.csr_matrix((n_items, 300), dtype=np.float32),
        dataset_version="synthetic_12k",
        manifest={},
    )

    eval_users, tuning_users = get_evaluation_user_split(
        data=data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=tmp_path,
    )

    assert len(eval_users) == 5000
    assert len(tuning_users) == 5000
    assert set(eval_users).isdisjoint(set(tuning_users))

    # Reload from disk
    eval_reloaded, tuning_reloaded = get_evaluation_user_split(
        data=data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=tmp_path,
    )

    np.testing.assert_array_equal(eval_users, eval_reloaded)
    np.testing.assert_array_equal(tuning_users, tuning_reloaded)


def test_tests_leave_recsys_data_dir_unchanged(tmp_path: Path):
    """
    Step 6: Running tests leaves backend/data/recsys completely unchanged.
    """
    recsys_dir = Path("data/recsys")
    files_before = {p.name: p.stat().st_mtime_ns for p in recsys_dir.iterdir()} if recsys_dir.exists() else {}

    data = _create_planted_block_recsys_data(n_users=60, n_items=40)
    # Call without persistence_dir (purely in-memory)
    get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, seed=42)

    # Call with tmp_path persistence_dir (isolated to tmp_path)
    get_evaluation_user_split(data, n_eval_users=20, n_tuning_users=20, seed=42, persistence_dir=tmp_path)

    files_after = {p.name: p.stat().st_mtime_ns for p in recsys_dir.iterdir()} if recsys_dir.exists() else {}

    # Directory state and timestamps before and after must be strictly identical
    assert files_before == files_after

