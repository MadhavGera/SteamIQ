"""
Item-to-Item k-Nearest Neighbors Collaborative Filter.

Implements:
  - Binary user-item co-occurrence matrix C = X^T X.
  - Cosine similarity with shrinkage: sim(i, j) = C_ij / (sqrt(n_i * n_j) + shrink).
  - Sparsification keeping only top-k neighbors per item.
  - History fold-in scoring via sparse matrix multiplication: score(H) = sum_{h in H} S_h.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.data import RecsysData
from recsys.models.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """
    item-item collaborative filtering with shrinkage cosine similarity, top-k neighbors per item (k=100).
    """

    name: str = "recsys_itemknn"

    def __init__(
        self,
        k_neighbors: int = 100,
        shrink: float = 10.0,
        similarity_matrix: sp.csr_matrix | None = None,
        n_items: int = 0,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.shrink = float(shrink)
        self.similarity_matrix: sp.csr_matrix | None = similarity_matrix
        self.n_items: int = n_items if similarity_matrix is None else similarity_matrix.shape[0]

    def fit(self, data: RecsysData) -> None:
        """
        Fits item-item similarity matrix from RecsysData training interactions.
        """
        n_items = data.n_items
        self.n_items = n_items

        # 1. Build binary user-item sparse CSR matrix X: [n_users, n_items]
        X = data.get_train_matrix(binary=True).tocsr()

        # 2. Compute item co-occurrence matrix C = X^T X: [n_items, n_items]
        C = (X.T @ X).tocsr()

        # Item interaction counts (diagonal entries of C)
        n_interactions = C.diagonal().astype(np.float64)

        # 3. Compute shrinkage cosine similarity & prune to top-k neighbors per row
        row_indices: list[int] = []
        col_indices: list[int] = []
        data_values: list[float] = []

        for i in range(n_items):
            start = C.indptr[i]
            end = C.indptr[i + 1]
            if start == end:
                continue

            cols = C.indices[start:end]
            co_counts = C.data[start:end].astype(np.float64)

            # Exclude self-similarity (i == j)
            mask = cols != i
            if not np.any(mask):
                continue

            cols = cols[mask]
            co_counts = co_counts[mask]
            n_j = n_interactions[cols]
            n_i = n_interactions[i]

            # Shrinkage cosine formula: C_ij / (sqrt(n_i * n_j) + shrink)
            denom = np.sqrt(n_i * n_j) + self.shrink
            sims = co_counts / denom

            # Filter out zero or negative similarities
            valid_sim = sims > 0.0
            if not np.any(valid_sim):
                continue

            cols = cols[valid_sim]
            sims = sims[valid_sim]

            # Top-k pruning
            if len(sims) > self.k_neighbors:
                top_k_idx = np.argpartition(-sims, self.k_neighbors)[: self.k_neighbors]
                cols = cols[top_k_idx]
                sims = sims[top_k_idx]

            row_indices.extend([i] * len(cols))
            col_indices.extend(cols.tolist())
            data_values.extend(sims.tolist())

        # Construct final sparse similarity matrix S: [n_items, n_items]
        self.similarity_matrix = sp.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Computes candidate item scores for a batch of user histories.

        score(history) = sum_{h in history} similarity_matrix[h, :]
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        b = len(histories)
        if b == 0:
            return np.empty((0, self.n_items), dtype=np.float32)

        # Build sparse binary batch interaction matrix [B, n_items]
        batch_rows: list[int] = []
        batch_cols: list[int] = []

        for row_idx, hist in enumerate(histories):
            if len(hist) > 0:
                # filter valid item indices
                valid_items = hist[(hist >= 0) & (hist < self.n_items)]
                if len(valid_items) > 0:
                    unique_items = np.unique(valid_items)
                    batch_rows.extend([row_idx] * len(unique_items))
                    batch_cols.extend(unique_items.tolist())

        if not batch_rows:
            return np.zeros((b, self.n_items), dtype=np.float32)

        values = np.ones(len(batch_rows), dtype=np.float32)
        batch_matrix = sp.csr_matrix(
            (values, (batch_rows, batch_cols)),
            shape=(b, self.n_items),
            dtype=np.float32,
        )

        # Sparse matrix multiplication: [B, n_items] @ [n_items, n_items] -> [B, n_items]
        scores = batch_matrix @ self.similarity_matrix
        dense_scores = scores.toarray().astype(np.float32)

        # Replace NaNs if any with 0.0
        np.nan_to_num(dense_scores, copy=False, nan=0.0)
        return dense_scores

    def save(self, model_dir: Path | str) -> None:
        """
        Serializes sparse similarity matrix and metadata to model_dir.
        """
        if self.similarity_matrix is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        sp.save_npz(path / "similarity_matrix.npz", self.similarity_matrix)

        metadata = {
            "name": self.name,
            "k_neighbors": self.k_neighbors,
            "shrink": self.shrink,
            "n_items": self.n_items,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> ItemKNNRecommender:
        """
        Deserializes ItemKNNRecommender from model_dir.
        """
        path = Path(model_dir)
        mat_path = path / "similarity_matrix.npz"
        meta_path = path / "metadata.json"

        if not mat_path.exists():
            raise FileNotFoundError(f"Missing similarity_matrix.npz in {path}")

        similarity_matrix = sp.load_npz(mat_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            k_neighbors=metadata.get("k_neighbors", 100),
            shrink=metadata.get("shrink", 10.0),
            similarity_matrix=similarity_matrix,
            n_items=metadata.get("n_items", similarity_matrix.shape[0]),
        )

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "k_neighbors": self.k_neighbors,
            "shrink": self.shrink,
            "n_items": self.n_items,
        }
