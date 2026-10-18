"""
Unit tests for lightweight SASRec sequential model (Phase F).
"""
import numpy as np
import torch

from recsys.models.sequential import SASRecConfig, SASRecModel, SASRecModule


def test_sasrec_module_shapes():
    config = SASRecConfig(n_items=100, d_model=64, n_layers=2, n_heads=2, max_len=50, dropout=0.1)
    module = SASRecModule(config)

    batch_size = 4
    # Right-aligned tokens (0 is padding, 1..100 are item tokens)
    seq = torch.randint(1, 101, (batch_size, 50))
    # Some padding on the left
    seq[:2, :30] = 0

    user_vecs = module.encode_history(seq)
    assert user_vecs.shape == (batch_size, 64)
    # Verify L2 normalized
    norms = torch.norm(user_vecs, p=2, dim=-1)
    np.testing.assert_allclose(norms.detach().numpy(), np.ones(batch_size), atol=1e-5)

    targets = torch.tensor([0, 15, 99, 42], dtype=torch.long)
    target_vecs = module.encode_target(targets)
    assert target_vecs.shape == (batch_size, 64)
    target_norms = torch.norm(target_vecs, p=2, dim=-1)
    np.testing.assert_allclose(target_norms.detach().numpy(), np.ones(batch_size), atol=1e-5)

    all_vecs = module.all_item_vectors()
    assert all_vecs.shape == (100, 64)


def test_sasrec_model_scoring_and_fold_in():
    config = SASRecConfig(n_items=50, d_model=32, n_layers=1, n_heads=2, max_len=20, dropout=0.0)
    model = SASRecModel(config)
    model.module = SASRecModule(config)
    model.precompute_item_vectors()

    # Create dummy user histories (0-indexed item IDs)
    histories = [
        np.array([1, 5, 10, 20], dtype=np.int32),
        np.array([49], dtype=np.int32),
        np.array([], dtype=np.int32),  # empty history
    ]

    scores = model.score(histories)
    assert scores.shape == (3, 50)
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()


def test_sasrec_save_load(tmp_path):
    config = SASRecConfig(n_items=30, d_model=32, n_layers=1, n_heads=2, max_len=10)
    model = SASRecModel(config)
    model.module = SASRecModule(config)
    model.precompute_item_vectors()

    save_dir = tmp_path / "sasrec_model"
    model.save(save_dir)

    assert (save_dir / "config.json").exists()
    assert (save_dir / "sasrec.pt").exists()
    assert (save_dir / "item_vectors.npy").exists()

    loaded = SASRecModel.load(save_dir)
    assert loaded.config.n_items == 30
    assert loaded.config.d_model == 32
    assert loaded.module is not None
    assert loaded._item_vectors is not None
    assert loaded._item_vectors.shape == (30, 32)
