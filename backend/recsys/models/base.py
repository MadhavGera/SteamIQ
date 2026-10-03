"""
Abstract base class for all recommender system models.

Enforces:
  - History-only (fold-in) scoring interface: models score strictly from user interaction histories
    (list of item_idx arrays), never from stored user IDs.
  - Serialization (save/load) to artifact directories.
  - Standard hyperparameter reporting.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from recsys.data import RecsysData


class BaseRecommender(ABC):
    """
    Abstract base recommender interface.
    """

    name: str = "base_recommender"

    @abstractmethod
    def fit(self, data: RecsysData) -> None:
        """
        Trains or fits model parameters from the supplied RecsysData bundle.
        """
        pass

    @abstractmethod
    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Scores all catalog items for a batch of user interaction histories.

        Args:
            histories: List of 1D numpy arrays, where each array contains the item_idx
                       of games the user previously interacted with (positive history).

        Returns:
            Float32 numpy array of shape [len(histories), n_items] containing unmasked
            ranking scores for every catalog item.
        """
        pass

    @abstractmethod
    def save(self, model_dir: Path | str) -> None:
        """
        Serializes model artifacts to model_dir.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path | str) -> BaseRecommender:
        """
        Deserializes and constructs a model instance from model_dir.
        """
        pass

    def hyperparameters(self) -> dict[str, Any]:
        """
        Returns model hyperparameter dictionary for logging and reproduction.
        """
        return {}
