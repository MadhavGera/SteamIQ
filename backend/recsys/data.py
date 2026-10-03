"""
In-memory RecsysData container and artifact loader.

Loads all precomputed Parquet splits, item embeddings, tag vocabulary, and manifest
metadata for offline training and evaluation.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from core.config import settings

logger = logging.getLogger("steamiq.recsys.data")


@dataclass
class RecsysData:
    """
    Standard dataset bundle passed to all recommender models and evaluation routines.
    """
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    negatives_df: pd.DataFrame
    users_df: pd.DataFrame
    mappings_df: pd.DataFrame
    items_df: pd.DataFrame
    item_text_emb: np.ndarray
    item_train_pop: np.ndarray
    item_tags_sparse: sp.csr_matrix
    dataset_version: str
    manifest: dict[str, Any]
    scalers: dict[str, dict[str, float]] | None = None
    tag_vocab: dict[str, Any] | None = None

    @property
    def n_users(self) -> int:
        return len(self.users_df)

    @property
    def n_items(self) -> int:
        return len(self.mappings_df)

    def get_train_matrix(self, binary: bool = True) -> sp.csr_matrix:
        """
        Constructs a sparse CSR interaction matrix [n_users, n_items] from train.parquet.
        """
        u_indices = self.train_df["user_idx"].to_numpy(dtype=np.int32)
        i_indices = self.train_df["item_idx"].to_numpy(dtype=np.int32)
        if binary:
            values = np.ones(len(u_indices), dtype=np.float32)
        else:
            values = self.train_df["hours"].to_numpy(dtype=np.float32)

        return sp.csr_matrix(
            (values, (u_indices, i_indices)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

    def save(self, data_dir: Path | str) -> None:
        """
        Persists all dataset splits, matrices, embeddings, and manifest to data_dir.
        """
        dir_path = Path(data_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.train_df.to_parquet(dir_path / "train.parquet")
        self.val_df.to_parquet(dir_path / "val.parquet")
        self.test_df.to_parquet(dir_path / "test.parquet")
        self.negatives_df.to_parquet(dir_path / "negatives.parquet")
        self.users_df.to_parquet(dir_path / "users.parquet")
        self.mappings_df.to_parquet(dir_path / "mappings.parquet")
        self.items_df.to_parquet(dir_path / "items.parquet")

        np.save(dir_path / "item_text_emb.npy", self.item_text_emb)
        np.save(dir_path / "item_train_pop.npy", self.item_train_pop)
        sp.save_npz(dir_path / "item_tags_sparse.npz", self.item_tags_sparse)

        manifest = dict(self.manifest) if self.manifest else {}
        manifest["dataset_version"] = self.dataset_version
        with open(dir_path / "data_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if self.scalers is not None:
            with open(dir_path / "numeric_scalers.json", "w", encoding="utf-8") as f:
                json.dump(self.scalers, f, indent=2)
        if self.tag_vocab is not None:
            with open(dir_path / "tag_vocab.json", "w", encoding="utf-8") as f:
                json.dump(self.tag_vocab, f, indent=2)

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> RecsysData:
        """
        Loads all recommender data artifacts from data_dir.
        """
        dir_path = Path(data_dir) if data_dir else settings.resolved_recsys_data_dir

        if not dir_path.exists():
            raise FileNotFoundError(f"Recsys data directory does not exist: {dir_path}")

        manifest_path = dir_path / "data_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"data_manifest.json not found in {dir_path}. Run recsys-prepare first.")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_version = manifest.get("dataset_version", "unknown")

        logger.info("Loading RecsysData from %s (dataset_version=%s)...", dir_path, dataset_version)

        train_df = pd.read_parquet(dir_path / "train.parquet")
        val_df = pd.read_parquet(dir_path / "val.parquet")
        test_df = pd.read_parquet(dir_path / "test.parquet")
        negatives_df = pd.read_parquet(dir_path / "negatives.parquet")
        users_df = pd.read_parquet(dir_path / "users.parquet")
        mappings_df = pd.read_parquet(dir_path / "mappings.parquet")
        items_df = pd.read_parquet(dir_path / "items.parquet")

        item_text_emb = np.load(dir_path / "item_text_emb.npy")
        item_train_pop = np.load(dir_path / "item_train_pop.npy")
        item_tags_sparse = sp.load_npz(dir_path / "item_tags_sparse.npz")

        # Optional auxiliary files
        scalers_path = dir_path / "numeric_scalers.json"
        scalers = None
        if scalers_path.exists():
            try:
                with open(scalers_path, encoding="utf-8") as f:
                    scalers = json.load(f)
            except Exception:
                scalers = None

        tag_vocab_path = dir_path / "tag_vocab.json"
        tag_vocab = None
        if tag_vocab_path.exists():
            try:
                with open(tag_vocab_path, encoding="utf-8") as f:
                    tag_vocab = json.load(f)
            except Exception:
                tag_vocab = None

        return cls(
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
            dataset_version=dataset_version,
            manifest=manifest,
            scalers=scalers,
            tag_vocab=tag_vocab,
        )
