"""
Content-based Recommender using MiniLM Text Embeddings and Steam Tags.

Implements:
  - Item representation: Concat(L2_norm(text_emb), 0.5 * L2_norm(tag_multi_hot)), L2-normalized again.
  - User profile: Mean of history item content vectors.
  - Score: Cosine similarity between user profile and all candidate item content vectors.
  - Pure inference without parameter optimization.

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


class ContentRecommender(BaseRecommender):
    """
    Pure content-based recommender scoring items by cosine similarity to the user's centroid profile.
    """

    name: str = "recsys_content"

    def __init__(
        self,
        weight_tags: float = 0.5,
        item_content_emb: np.ndarray | None = None,
        n_items: int = 0,
    ) -> None:
        self.weight_tags = float(weight_tags)
        self.item_content_emb: np.ndarray | None = (
            item_content_emb.astype(np.float32) if item_content_emb is not None else None
        )
        self.n_items: int = n_items if item_content_emb is None else item_content_emb.shape[0]

    @property
    def dim(self) -> int:
        return self.item_content_emb.shape[1] if self.item_content_emb is not None else 0

    def fit(self, data: RecsysData) -> None:
        """
        Constructs and normalizes the unified item content matrix from text embeddings and tags.
        """
        n_items = data.n_items
        self.n_items = n_items

        # 1. Text Embeddings (MiniLM [n_items, 384])
        if data.item_text_emb is not None and len(data.item_text_emb) == n_items:
            text_embs = data.item_text_emb.astype(np.float32)
            t_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
            normed_text = text_embs / np.maximum(t_norms, 1e-12)
        else:
            normed_text = np.zeros((n_items, 384), dtype=np.float32)

        # 2. Tag Multi-Hot Sparse Matrix ([n_items, 300])
        if data.item_tags_sparse is not None and data.item_tags_sparse.shape[0] == n_items:
            if sp.issparse(data.item_tags_sparse):
                dense_tags = data.item_tags_sparse.toarray().astype(np.float32)
            else:
                dense_tags = np.asarray(data.item_tags_sparse, dtype=np.float32)
            tag_norms = np.linalg.norm(dense_tags, axis=1, keepdims=True)
            normed_tags = dense_tags / np.maximum(tag_norms, 1e-12)
        else:
            normed_tags = np.zeros((n_items, 1), dtype=np.float32)

        # 3. Apply tag weighting
        weighted_tags = normed_tags * self.weight_tags

        # 4. Concatenate text + tags
        combined = np.hstack([normed_text, weighted_tags])

        # 5. Final L2 normalization
        c_norms = np.linalg.norm(combined, axis=1, keepdims=True)
        self.item_content_emb = (combined / np.maximum(c_norms, 1e-12)).astype(np.float32)

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Scores candidate items by cosine similarity against the user's history centroid.
        """
        if self.item_content_emb is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        b = len(histories)
        if b == 0:
            return np.empty((0, self.n_items), dtype=np.float32)

        d = self.dim
        user_profiles = np.zeros((b, d), dtype=np.float32)

        for row_idx, hist in enumerate(histories):
            if len(hist) == 0:
                continue

            valid_items = hist[(hist >= 0) & (hist < self.n_items)]
            if len(valid_items) > 0:
                user_profiles[row_idx] = np.mean(self.item_content_emb[valid_items], axis=0)

        # Batch cosine scores: [B, d] @ [d, n_items] -> [B, n_items]
        scores = user_profiles @ self.item_content_emb.T
        np.nan_to_num(scores, copy=False, nan=0.0)
        # Ensure exact clipping within [-1.0, 1.0]
        np.clip(scores, -1.0, 1.0, out=scores)
        return scores.astype(np.float32)

    def save(self, model_dir: Path | str) -> None:
        """
        Serializes unified item content embeddings and metadata to model_dir.
        """
        if self.item_content_emb is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "item_content_emb.npy", self.item_content_emb)

        metadata = {
            "name": self.name,
            "weight_tags": self.weight_tags,
            "dim": self.dim,
            "n_items": self.n_items,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> ContentRecommender:
        """
        Deserializes ContentRecommender from model_dir.
        """
        path = Path(model_dir)
        emb_file = path / "item_content_emb.npy"
        meta_file = path / "metadata.json"

        if not emb_file.exists():
            raise FileNotFoundError(f"Missing item_content_emb.npy in {path}")

        item_content_emb = np.load(emb_file)
        metadata = {}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                metadata = json.load(f)

        inst = cls(
            weight_tags=metadata.get("weight_tags", 0.5),
            item_content_emb=item_content_emb,
            n_items=metadata.get("n_items", item_content_emb.shape[0]),
        )
        inst.model_dir = str(path)
        inst.model_run_id = path.name
        return inst

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "weight_tags": self.weight_tags,
            "dim": self.dim,
            "n_items": self.n_items,
        }
