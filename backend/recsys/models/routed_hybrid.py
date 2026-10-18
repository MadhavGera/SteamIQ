"""
Routed Hybrid Recommender for Cold-Start and Multi-Channel Recommendation.

Routes items according to their interaction frequency:
  - Items with >= cold_threshold training interactions are scored by the primary
    collaborative model (e.g. Two-Tower).
  - Items with < cold_threshold interactions (or explicitly held-out cold items)
    are scored by the content model (text/tag cosine similarities).

Score Calibration:
  Because content cosine similarities lie in [-1, 1] while Two-Tower logits typically
  range in [5, 25], direct merging pushes cold items to the catalog bottom.
  This model performs per-user z-score matching calibration over warm items:
    S_cold_calib = mu_warm_prim + sigma_warm_prim * ((S_cold - mu_warm_cold) / (sigma_warm_cold + eps))

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recsys.data import RecsysData
from recsys.models.base import BaseRecommender


class RoutedHybridRecommender(BaseRecommender):
    """
    Routed Hybrid Recommender combining a primary collaborative model with
    a content-based recommender using per-user score calibration.
    """

    name: str = "recsys_routed_hybrid"

    def __init__(
        self,
        primary_model: BaseRecommender,
        cold_model: BaseRecommender,
        cold_threshold: int = 5,
        held_out_items: list[int] | set[int] | None = None,
        route_zero_only: bool = False,
        cap_sigma: float | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.cold_model = cold_model
        self.cold_threshold = int(cold_threshold)
        self.held_out_items: set[int] = set(int(x) for x in held_out_items) if held_out_items else set()
        self.route_zero_only = bool(route_zero_only)
        self.cap_sigma = float(cap_sigma) if cap_sigma is not None else None
        self.item_train_pop: np.ndarray | None = None
        self.n_items: int = 0

    def fit(self, data: RecsysData) -> None:
        """
        Fits or configures the recommender using the supplied RecsysData bundle.
        """
        self.item_train_pop = np.asarray(data.item_train_pop, dtype=np.int64)
        self.n_items = int(data.n_items)
        if hasattr(self.cold_model, "fit"):
            self.cold_model.fit(data)

    def score(
        self,
        histories: list[np.ndarray],
        item_vecs: np.ndarray | None = None,
        mask_ids: list[int] | None = None,
        neutralize_proxies: list[int] | None = None,
    ) -> np.ndarray:
        """
        Scores all catalog items for a batch of user histories with per-user calibration.
        """
        n_users = len(histories)
        if n_users == 0:
            return np.empty((0, self.n_items), dtype=np.float32)

        # Primary scores
        if item_vecs is not None or mask_ids is not None or neutralize_proxies is not None:
            # Pass two-tower kwargs if supported
            try:
                sc_primary = self.primary_model.score(
                    histories,
                    item_vecs=item_vecs,
                    mask_ids=mask_ids,
                    neutralize_proxies=neutralize_proxies,
                ).copy()
            except TypeError:
                sc_primary = self.primary_model.score(histories).copy()
        else:
            sc_primary = self.primary_model.score(histories).copy()

        n_items = sc_primary.shape[1]
        self.n_items = n_items

        # Cold scores
        sc_cold = self.cold_model.score(histories).copy()

        # Identify cold item set
        cold_set = set(self.held_out_items)
        if mask_ids:
            cold_set.update(int(it) for it in mask_ids)
        if self.item_train_pop is not None:
            if self.route_zero_only:
                sparse_indices = np.where(self.item_train_pop[:n_items] == 0)[0]
            else:
                sparse_indices = np.where(self.item_train_pop[:n_items] < self.cold_threshold)[0]
            cold_set.update(int(it) for it in sparse_indices)

        # Edge cases: no cold items or all cold items
        if not cold_set:
            return sc_primary
        if len(cold_set) >= n_items:
            return sc_cold

        cold_indices = np.array(sorted(cold_set), dtype=np.int64)
        all_indices = np.arange(n_items, dtype=np.int64)
        warm_indices = np.setdiff1d(all_indices, cold_indices)

        sc_hybrid = sc_primary.copy()

        # Per-user z-score matching calibration over warm items
        for u in range(n_users):
            u_prim_warm = sc_primary[u, warm_indices]
            u_cold_warm = sc_cold[u, warm_indices]

            # Filter finite values to avoid issues with pre-masked infinities
            fin_prim = u_prim_warm[np.isfinite(u_prim_warm)]
            fin_cold = u_cold_warm[np.isfinite(u_cold_warm)]

            mu_prim = float(np.mean(fin_prim)) if len(fin_prim) > 0 else 0.0
            std_prim = float(np.std(fin_prim)) if len(fin_prim) > 0 else 1.0
            std_prim = max(std_prim, 1e-6)

            mu_cold = float(np.mean(fin_cold)) if len(fin_cold) > 0 else 0.0
            std_cold = float(np.std(fin_cold)) if len(fin_cold) > 0 else 1.0
            std_cold = max(std_cold, 1e-6)

            # Match cold item distribution to primary warm distribution
            calib_cold = mu_prim + std_prim * ((sc_cold[u, cold_indices] - mu_cold) / std_cold)
            if self.cap_sigma is not None:
                calib_cold = np.minimum(calib_cold, mu_prim + float(self.cap_sigma) * std_prim)
            sc_hybrid[u, cold_indices] = calib_cold

        return sc_hybrid

    def save(self, model_dir: Path | str) -> None:
        """
        Saves metadata configuration.
        """
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "cold_threshold": self.cold_threshold,
            "held_out_items": sorted(self.held_out_items),
            "route_zero_only": self.route_zero_only,
            "cap_sigma": self.cap_sigma,
            "primary_model_name": getattr(self.primary_model, "name", "unknown"),
            "cold_model_name": getattr(self.cold_model, "name", "unknown"),
            "n_items": self.n_items,
        }
        with open(path / "routed_hybrid_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> RoutedHybridRecommender:
        """
        Deserializes configuration stub (submodels must be wired by caller).
        """
        path = Path(model_dir)
        with open(path / "routed_hybrid_config.json", encoding="utf-8") as f:
            meta = json.load(f)
        instance = cls(
            primary_model=None,  # type: ignore[arg-type]
            cold_model=None,  # type: ignore[arg-type]
            cold_threshold=meta.get("cold_threshold", 5),
            held_out_items=meta.get("held_out_items", []),
            route_zero_only=meta.get("route_zero_only", False),
            cap_sigma=meta.get("cap_sigma", None),
        )
        instance.n_items = meta.get("n_items", 0)
        return instance

    def hyperparameters(self) -> dict[str, Any]:
        """
        Returns model hyperparameter dictionary for logging and reproduction.
        """
        return {
            "name": self.name,
            "cold_threshold": self.cold_threshold,
            "route_zero_only": self.route_zero_only,
            "cap_sigma": self.cap_sigma,
            "primary_model": getattr(self.primary_model, "name", "unknown"),
            "cold_model": getattr(self.cold_model, "name", "unknown"),
            "n_held_out_items": len(self.held_out_items),
        }
