"""
Popularity-based baseline recommender.

Scores items unconditionally according to log1p(training_positive_count).
Every user receives the identical unmasked score vector.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recsys.data import RecsysData
from recsys.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Non-personalized baseline scoring items by log-transformed training popularity.
    """

    name: str = "recsys_popularity"

    def __init__(self, scores: np.ndarray | None = None) -> None:
        self.item_scores = scores.astype(np.float32) if scores is not None else None
        self.n_items: int = len(scores) if scores is not None else 0

    def fit(self, data: RecsysData) -> None:
        """
        Fits item scores as log1p(item_train_pop).
        """
        pop = data.item_train_pop.astype(np.float32)
        self.item_scores = np.log1p(pop)
        self.n_items = len(self.item_scores)

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Returns identical score vector for each user history in the batch.

        Args:
            histories: List of 1D item_idx arrays representing user histories.

        Returns:
            np.ndarray of shape [len(histories), n_items], dtype float32.
        """
        if self.item_scores is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        b = len(histories)
        if b == 0:
            return np.empty((0, self.n_items), dtype=np.float32)

        # Broadcast 1D scores to [B, n_items]
        return np.tile(self.item_scores, (b, 1))

    def save(self, model_dir: Path | str) -> None:
        """
        Serializes model scores and metadata to model_dir.
        """
        if self.item_scores is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "item_scores.npy", self.item_scores)

        metadata = {
            "name": self.name,
            "n_items": self.n_items,
            "transform": "log1p",
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> PopularityRecommender:
        """
        Deserializes a PopularityRecommender from model_dir.
        """
        path = Path(model_dir)
        scores_file = path / "item_scores.npy"
        if not scores_file.exists():
            raise FileNotFoundError(f"Missing item_scores.npy in {path}")

        scores = np.load(scores_file)
        return cls(scores=scores)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "transform": "log1p",
            "n_items": self.n_items,
        }
