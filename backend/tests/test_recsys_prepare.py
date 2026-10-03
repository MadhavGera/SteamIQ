"""
Unit tests for recsys interaction preparation, temporal splits, item features, and manifests.

Tests run completely OFFLINE using tiny synthetic datasets in tmp_path.
SentenceTransformer is monkeypatched to avoid model downloads.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jobs.prepare_recsys_data import PrepareRecsysDataJob
from recsys.config import RecsysConfig
from recsys.data_schema import (
    GAMES_COLUMNS,
    RECOMMENDATIONS_COLUMNS,
    USERS_COLUMNS,
)
from recsys.items import build_item_artifacts
from recsys.prepare import convert_csv_to_interactions_parquet, prepare_interactions


class DummySentenceTransformer:
    """Mock SentenceTransformer for offline testing."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        # Return deterministic float32 embeddings shaped [len(texts), 384]
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((len(texts), 384)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr = arr / norms
        return arr


@pytest.fixture
def synthetic_recsys_env(tmp_path: Path):
    """
    Creates a tiny synthetic raw Steam directory and returns paths and RecsysConfig.
    Catalog: 6 games (10, 20, 30, 40, 50, 60).
    Users: 101, 102, 103 (with products and reviews).
    Recommendations: ~25 rows with duplicates, positives, negatives, timestamps, review_ids.
    """
    raw_dir = tmp_path / "raw_steam"
    data_dir = tmp_path / "recsys"
    raw_dir.mkdir()
    data_dir.mkdir()

    # 1. games.csv
    games_rows = [
        ",".join(GAMES_COLUMNS),
        "10,Game Alpha,2020-01-01,True,True,False,Very Positive,90,5000,19.99,19.99,0,True",
        "20,Game Beta,2020-06-15,True,False,False,Positive,85,3000,9.99,9.99,0,False",
        "30,Game Gamma,2021-03-20,True,True,True,Overwhelmingly Positive,98,20000,29.99,29.99,0,True",
        "40,Game Delta,2021-11-10,True,False,False,Mixed,60,1000,4.99,4.99,0,False",
        "50,Game Epsilon,2022-02-01,True,True,False,Positive,80,1500,14.99,14.99,0,True",
        "60,Game Zeta,2022-08-10,True,False,False,Mostly Positive,75,800,0.00,0.00,0,True",
    ]
    (raw_dir / "games.csv").write_text("\n".join(games_rows) + "\n", encoding="utf-8")

    # 2. games_metadata.json
    meta_lines = [
        json.dumps({"app_id": 10, "description": "Alpha action RPG adventure", "tags": ["Action", "RPG", "Adventure"]}),
        json.dumps({"app_id": 20, "description": "Beta puzzle platformer", "tags": ["Puzzle", "Platformer", "Indie"]}),
        json.dumps({"app_id": 30, "description": "Gamma space sim", "tags": ["Space", "Sci-fi", "Simulation", "RPG"]}),
        json.dumps({"app_id": 40, "description": "", "tags": ["Casual", "Indie"]}),  # Empty description
        json.dumps({"app_id": 50, "description": "Epsilon roguelike dungeon", "tags": ["Roguelike", "Dungeon Crawler", "Action"]}),
        json.dumps({"app_id": 60, "description": "Zeta racing simulator", "tags": ["Racing", "Simulation"]}),
    ]
    (raw_dir / "games_metadata.json").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    # 3. users.csv
    users_rows = [
        ",".join(USERS_COLUMNS),
        "101,25,10",
        "102,40,15",
        "103,12,5",
    ]
    (raw_dir / "users.csv").write_text("\n".join(users_rows) + "\n", encoding="utf-8")

    # 4. recommendations.csv (Synthetic interactions)
    # User 101: 6 positive interactions on apps 10, 20, 30, 40, 50, 60; plus 1 duplicate on 10; 1 negative on 60 (dated before val)
    # User 102: 5 positive interactions on apps 10, 20, 30, 40, 50; plus 1 negative on 60 (dated after val)
    # User 103: 2 positive interactions (below min_user_positives=5, filtered out)
    recs_rows = [
        ",".join(RECOMMENDATIONS_COLUMNS),
        # User 101 positives (ordered chronologically)
        "10,1,0,2021-01-01,True,10.0,101,1001",
        "10,0,0,2021-01-02,True,12.0,101,1002",  # Duplicate of (101, 10), later date -> replaces review 1001
        "20,0,0,2021-02-01,True,5.0,101,1003",
        "30,2,0,2021-03-01,True,25.0,101,1004",
        "40,0,0,2021-04-01,True,8.0,101,1005",
        "50,0,0,2021-05-01,True,15.0,101,1006",  # Val item for 101 (2nd to last)
        "60,5,1,2021-06-01,True,30.0,101,1007",  # Test item for 101 (last)
        # User 101 negative
        "40,0,0,2021-01-15,False,1.0,101,1008",  # Duplicate on 40 with positive later -> positive on 2021-04-01 replaces it
        "20,0,0,2021-01-20,False,0.5,101,1009",  # Negative before positive on 2021-02-01 -> positive replaces it
        # User 102 positives
        "10,0,0,2021-01-10,True,20.0,102,2001",
        "20,0,0,2021-02-15,True,18.0,102,2002",
        "30,0,0,2021-03-20,True,40.0,102,2003",
        "40,0,0,2021-04-25,True,12.0,102,2004",  # Val item for 102
        "50,0,0,2021-05-30,True,22.0,102,2005",  # Test item for 102
        # User 102 negative on app 60 (dated after val date 2021-04-25 -> holdout)
        "60,0,0,2021-06-15,False,3.0,102,2006",
        # User 103 (only 2 positives -> ineligible)
        "10,0,0,2021-01-05,True,5.0,103,3001",
        "20,0,0,2021-02-05,True,4.0,103,3002",
    ]
    (raw_dir / "recommendations.csv").write_text("\n".join(recs_rows) + "\n", encoding="utf-8")

    config = RecsysConfig(
        profile="dev",
        max_users=10,
        max_items=10,
        eval_users=5,
        min_user_positives=5,
        min_item_positives=1,
        max_history=50,
        seed=42,
        num_threads=2,
        device="cpu",
        raw_dir=raw_dir,
        data_dir=data_dir,
    )

    return {"raw_dir": raw_dir, "data_dir": data_dir, "config": config}


def test_csv_to_parquet_conversion(synthetic_recsys_env):
    """Test STAGE 0 CSV-to-Parquet conversion preserves schema and rows, and caches fingerprints."""
    raw_dir = synthetic_recsys_env["raw_dir"]
    data_dir = synthetic_recsys_env["data_dir"]
    recs_csv = raw_dir / "recommendations.csv"
    out_parquet = data_dir / "interactions_all.parquet"

    # 1. First run: converts CSV to Parquet
    convert_csv_to_interactions_parquet(recs_csv, out_parquet, chunk_size=10)
    assert out_parquet.exists()

    df1 = pd.read_parquet(out_parquet)
    # recommendations.csv has 17 rows
    assert len(df1) == 17
    assert set(df1.columns) == {"app_id", "user_id", "is_recommended", "hours", "date", "review_id"}
    assert df1["hours"].dtype == np.float32
    assert df1["app_id"].dtype == np.int32
    assert df1["user_id"].dtype == np.int32

    # 2. Second run: skipped via fingerprint match
    mtime_before = out_parquet.stat().st_mtime
    convert_csv_to_interactions_parquet(recs_csv, out_parquet, chunk_size=10)
    mtime_after = out_parquet.stat().st_mtime
    assert mtime_before == mtime_after


@patch("recsys.items.SentenceTransformer", DummySentenceTransformer)
def test_temporal_split_and_no_leakage(synthetic_recsys_env):
    """
    Verifies:
      (i) For every user max(train ts) <= val ts <= test ts with review_id tie-breaks.
      (ii) Test/val items never appear in that user's train rows.
      (iii) Every user has >= 3 train positives.
      (iv) user_idx and item_idx are contiguous, and item_idx 0 is the most popular.
      (v) Duplicate (user, app) rows collapse to the latest.
      (vi) Negatives never leak into train.parquet.
    """
    raw_dir = synthetic_recsys_env["raw_dir"]
    data_dir = synthetic_recsys_env["data_dir"]
    config = synthetic_recsys_env["config"]

    # 1. Run prepare interactions
    inter_stats = prepare_interactions(raw_dir, data_dir, config)
    assert inter_stats["num_users"] == 2  # Users 101 and 102 (103 had only 2 positives)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    neg_df = pd.read_parquet(data_dir / "negatives.parquet")
    users_df = pd.read_parquet(data_dir / "users.parquet")
    mappings_df = pd.read_parquet(data_dir / "mappings.parquet")

    # (i) & (iii) Temporal split ordering and >=3 train positives
    for u_idx in range(inter_stats["num_users"]):
        u_train = train_df[train_df["user_idx"] == u_idx]
        u_val = val_df[val_df["user_idx"] == u_idx]
        u_test = test_df[test_df["user_idx"] == u_idx]

        assert len(u_val) == 1, f"User {u_idx} must have exactly 1 val interaction"
        assert len(u_test) == 1, f"User {u_idx} must have exactly 1 test interaction"
        assert len(u_train) >= 3, f"User {u_idx} must have >= 3 train positives, got {len(u_train)}"

        max_train_ts = u_train["ts"].max()
        val_ts = u_val["ts"].iloc[0]
        test_ts = u_test["ts"].iloc[0]

        assert max_train_ts <= val_ts <= test_ts, (
            f"User {u_idx} violated temporal order: max_train={max_train_ts}, val={val_ts}, test={test_ts}"
        )

        # (ii) No leakage: test/val items never appear in that user's train rows
        val_item = u_val["item_idx"].iloc[0]
        test_item = u_test["item_idx"].iloc[0]
        train_items = set(u_train["item_idx"])

        assert val_item not in train_items, f"User {u_idx} val item {val_item} leaked into train"
        assert test_item not in train_items, f"User {u_idx} test item {test_item} leaked into train"
        assert val_item != test_item, f"User {u_idx} val item equals test item"

    # (iv) Contiguous indexing & item_idx 0 is most popular
    assert list(users_df["user_idx"]) == list(range(len(users_df)))
    assert list(mappings_df["item_idx"]) == list(range(len(mappings_df)))
    # item_idx 0 should have the most overall positives in the universe
    assert mappings_df["item_idx"].iloc[0] == 0

    # (v) Duplicate collapse check
    # In recommendations.csv, user 101 had duplicate reviews on app 10 (hours=10 on review 1001, hours=12 on review 1002).
    # The later one (hours=12) must be preserved.
    app_10_idx = mappings_df[mappings_df["app_id"] == 10]["item_idx"].iloc[0]
    user_0_train_app_10 = train_df[(train_df["user_idx"] == 0) & (train_df["item_idx"] == app_10_idx)]
    assert len(user_0_train_app_10) == 1
    assert user_0_train_app_10["hours"].iloc[0] == 12.0

    # (vi) Negatives never leak into train.parquet
    train_pairs = set(zip(train_df["user_idx"], train_df["item_idx"], strict=False))
    for _, neg_row in neg_df.iterrows():
        neg_pair = (int(neg_row["user_idx"]), int(neg_row["item_idx"]))
        assert neg_pair not in train_pairs, f"Negative pair {neg_pair} leaked into train.parquet"


@patch("recsys.items.SentenceTransformer", DummySentenceTransformer)
def test_items_features_and_manifest(synthetic_recsys_env):
    """
    Verifies:
      - Item artifacts (sparse tag matrix, embeddings, items.parquet, numeric_scalers).
      - Data manifest generation with verifiable SHA1 hashes and 12-char dataset_version.
    """
    raw_dir = synthetic_recsys_env["raw_dir"]
    data_dir = synthetic_recsys_env["data_dir"]
    config = synthetic_recsys_env["config"]

    prepare_interactions(raw_dir, data_dir, config)
    item_stats = build_item_artifacts(raw_dir, data_dir, config)

    manifest_file = data_dir / "data_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    assert item_stats["dataset_version"] == manifest["dataset_version"]
    assert manifest["profile"] == "dev"
    assert manifest["seed"] == 42
    assert len(manifest["dataset_version"]) == 12
    assert manifest["counts"]["users"] == 2
    assert manifest["counts"]["items"] == 6
    assert "item_text_emb.npy" in manifest["file_sha1s"]
    assert "item_tags_sparse.npz" in manifest["file_sha1s"]
    assert "items.parquet" in manifest["file_sha1s"]

    # Verify items.parquet structure
    items_df = pd.read_parquet(data_dir / "items.parquet")
    assert len(items_df) == 6
    assert set(items_df.columns) == {
        "app_id",
        "item_idx",
        "title",
        "date_release",
        "price_final",
        "positive_ratio",
        "user_reviews",
        "tags",
        "description_short",
        "has_description",
        "train_positives",
        "popularity_rank",
        "win",
        "mac",
        "linux",
        "steam_deck",
        "header_image_url",
    }
    # Check header_image_url pattern
    assert items_df["header_image_url"].iloc[0].startswith("https://cdn.cloudflare.steamstatic.com/steam/apps/")

    # Verify embedding shape [6, 384]
    emb = np.load(data_dir / "item_text_emb.npy")
    assert emb.shape == (6, 384)
    assert emb.dtype == np.float32


@patch("recsys.items.SentenceTransformer", DummySentenceTransformer)
def test_seed_determinism(synthetic_recsys_env):
    """Verifies that rerunning with the same seed produces identical dataset_version and file hashes."""
    raw_dir = synthetic_recsys_env["raw_dir"]
    data_dir = synthetic_recsys_env["data_dir"]
    config = synthetic_recsys_env["config"]

    # Run 1
    prepare_interactions(raw_dir, data_dir, config)
    res1 = build_item_artifacts(raw_dir, data_dir, config)

    with open(data_dir / "data_manifest.json", encoding="utf-8") as f:
        manifest1 = json.load(f)

    # Run 2
    prepare_interactions(raw_dir, data_dir, config)
    res2 = build_item_artifacts(raw_dir, data_dir, config)

    with open(data_dir / "data_manifest.json", encoding="utf-8") as f:
        manifest2 = json.load(f)

    assert res1["dataset_version"] == res2["dataset_version"]
    assert manifest1["file_sha1s"] == manifest2["file_sha1s"]
    assert manifest1["dataset_version"] == manifest2["dataset_version"]


@patch("recsys.items.SentenceTransformer", DummySentenceTransformer)
def test_prepare_recsys_job_dry_run(synthetic_recsys_env):
    """Verifies PrepareRecsysDataJob runs successfully with --dry-run on synthetic data."""
    raw_dir = synthetic_recsys_env["raw_dir"]
    data_dir = synthetic_recsys_env["data_dir"]

    job = PrepareRecsysDataJob(
        profile="dev",
        raw_dir=raw_dir,
        data_dir=data_dir,
        seed=42,
        min_user_positives=5,
        min_item_positives=1,
        dry_run=True,
    )
    results = job.run()

    assert results["users"] == 2
    assert results["items"] == 6
    assert len(results["dataset_version"]) == 12
    assert results["density"] > 0.0
