"""
Implicit Alternating Least Squares (ALS) Collaborative Filter.

Implements:
  - Implicit feedback confidence weights: c_ui = 1 + alpha * log1p(hours).
  - Implicit.als.AlternatingLeastSquares matrix factorization on CPU.
  - Closed-form history fold-in user factor inference:
      u = solve(Y^T Y + Y_H^T diag(c - 1) Y_H + reg * I, Y_H^T c)
  - Vectorized batch scoring for history-only inference.
  - Optuna hyperparameter optimization with nested MLflow run tracking.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# Configure thread counts before importing implicit (per official recommendation)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import threadpoolctl  # noqa: E402
from implicit.als import AlternatingLeastSquares  # noqa: E402

from recsys.data import RecsysData  # noqa: E402
from recsys.models.base import BaseRecommender  # noqa: E402

logger = logging.getLogger("steamiq.recsys.als")


class ALSRecommender(BaseRecommender):
    """
    Implicit Alternating Least Squares recommender with closed-form history fold-in.
    """

    name: str = "recsys_als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 15.0,
        iterations: int = 15,
        seed: int = 42,
        item_factors: np.ndarray | None = None,
        n_items: int = 0,
    ) -> None:
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.item_factors: np.ndarray | None = (
            item_factors.astype(np.float32) if item_factors is not None else None
        )
        self.n_items: int = n_items if item_factors is None else item_factors.shape[0]

        # Precomputed cached Gram matrix Y^T Y
        self._YtY: np.ndarray | None = None
        if self.item_factors is not None:
            self._update_cached_matrices()

    def _update_cached_matrices(self) -> None:
        if self.item_factors is not None:
            Y = self.item_factors.astype(np.float32)
            self._YtY = Y.T @ Y

    def fit(self, data: RecsysData) -> None:
        """
        Fits item factor representations using training interaction confidence weights.
        """
        n_users = data.n_users
        n_items = data.n_items
        self.n_items = n_items

        # Build confidence-weighted user-item CSR interaction matrix
        # c_ui = 1 + alpha * log1p(hours)
        u_idx = data.train_df["user_idx"].to_numpy(dtype=np.int32)
        i_idx = data.train_df["item_idx"].to_numpy(dtype=np.int32)
        hours = data.train_df["hours"].to_numpy(dtype=np.float32)

        conf = (1.0 + self.alpha * np.log1p(np.maximum(hours, 0.0))).astype(np.float32)

        user_item_matrix = sp.csr_matrix(
            (conf, (u_idx, i_idx)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(
            "Fitting ALS model (factors=%d, reg=%.4f, alpha=%.1f, iter=%d)...",
            self.factors,
            self.regularization,
            self.alpha,
            self.iterations,
        )

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=False,
        )
        with threadpoolctl.threadpool_limits(1, "blas"):
            model.fit(user_item_matrix, show_progress=False)

        self.item_factors = model.item_factors.astype(np.float32)
        self._update_cached_matrices()


    def fold_in_user(
        self,
        history_items: np.ndarray,
        history_conf: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the exact closed-form user factor for a single user history:
            u = solve(Y^T Y + Y_H^T diag(c - 1) Y_H + reg * I, Y_H^T c)
        """
        if self.item_factors is None or self._YtY is None:
            raise RuntimeError("Model has not been fitted.")

        f = self.factors
        if len(history_items) == 0:
            return np.zeros(f, dtype=np.float32)

        valid_mask = (history_items >= 0) & (history_items < self.n_items)
        valid_items = history_items[valid_mask]
        if len(valid_items) == 0:
            return np.zeros(f, dtype=np.float32)

        if history_conf is not None:
            c = history_conf[valid_mask].astype(np.float32)
        else:
            # Default confidence weight for positive history interactions: 1 + alpha * ln(2)
            c_val = float(1.0 + self.alpha * np.log1p(1.0))
            c = np.full(len(valid_items), c_val, dtype=np.float32)

        Y_H = self.item_factors[valid_items]  # [m, f]

        # Regularized system: A = Y^T Y + reg * I + Y_H^T diag(c - 1) Y_H
        # Y_H^T diag(c - 1) Y_H = (Y_H * (c - 1)[:, None]).T @ Y_H
        c_minus_1 = (c - 1.0)[:, None]
        weighted_YH = Y_H * c_minus_1
        A = self._YtY + (self.regularization * np.eye(f, dtype=np.float32)) + (weighted_YH.T @ Y_H)
        b = Y_H.T @ c

        try:
            u = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        return u

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Scores all catalog items for a batch of user histories via closed-form fold-in.
        """
        if self.item_factors is None or self._YtY is None:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        b = len(histories)
        if b == 0:
            return np.empty((0, self.n_items), dtype=np.float32)

        f = self.factors
        reg_I = self.regularization * np.eye(f, dtype=np.float32)
        base_A = self._YtY + reg_I
        c_default = float(1.0 + self.alpha * np.log1p(1.0))
        c_minus_1 = c_default - 1.0

        user_vectors = np.zeros((b, f), dtype=np.float32)

        for row_idx, hist in enumerate(histories):
            if len(hist) == 0:
                continue

            valid_items = hist[(hist >= 0) & (hist < self.n_items)]
            if len(valid_items) == 0:
                continue

            Y_H = self.item_factors[valid_items]  # [m, f]
            # Fast vectorized rank-k update for constant confidence
            A = base_A + (c_minus_1 * (Y_H.T @ Y_H))
            b_vec = c_default * np.sum(Y_H, axis=0)

            try:
                u = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                u = np.linalg.lstsq(A, b_vec, rcond=None)[0]

            user_vectors[row_idx] = u

        # Batch scoring: [B, f] @ [f, n_items] -> [B, n_items]
        scores = user_vectors @ self.item_factors.T
        np.nan_to_num(scores, copy=False, nan=0.0)
        return scores.astype(np.float32)

    def save(self, model_dir: Path | str) -> None:
        """
        Serializes item factors and hyperparameters to model_dir.
        """
        if self.item_factors is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "item_factors.npy", self.item_factors)

        metadata = {
            "name": self.name,
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "seed": self.seed,
            "n_items": self.n_items,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_dir: Path | str) -> ALSRecommender:
        """
        Deserializes ALSRecommender from model_dir.
        """
        path = Path(model_dir)
        factors_file = path / "item_factors.npy"
        meta_file = path / "metadata.json"

        if not factors_file.exists():
            raise FileNotFoundError(f"Missing item_factors.npy in {path}")

        item_factors = np.load(factors_file)
        metadata = {}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            factors=metadata.get("factors", item_factors.shape[1]),
            regularization=metadata.get("regularization", 0.05),
            alpha=metadata.get("alpha", 15.0),
            iterations=metadata.get("iterations", 15),
            seed=metadata.get("seed", 42),
            item_factors=item_factors,
            n_items=metadata.get("n_items", item_factors.shape[0]),
        )

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "seed": self.seed,
            "n_items": self.n_items,
        }


def tune_als_hyperparameters(
    data: RecsysData,
    tuning_users: np.ndarray,
    n_trials: int = 6,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """
    Tuning routine optimizing ALS hyperparameters with Optuna against validation NDCG@10
    on the disjoint tuning users subset.

    Logs all trials as nested MLflow runs.
    """
    logger.info("Starting Optuna tuning for ALS (%d trials on %d tuning users)...", n_trials, len(tuning_users))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from recsys.evaluate import evaluate_model


    def objective(trial: optuna.Trial) -> float:
        factors = trial.suggest_categorical("factors", [32, 64, 128])
        regularization = trial.suggest_float("regularization", 0.01, 0.5, log=True)
        alpha = trial.suggest_categorical("alpha", [1, 5, 15, 40])

        model = ALSRecommender(
            factors=factors,
            regularization=regularization,
            alpha=float(alpha),
            iterations=15,
            seed=seed,
        )
        model.fit(data)

        val_res = evaluate_model(
            model=model,
            data=data,
            split="val",
            k_list=(10,),
            user_indices=tuning_users,
            seed=seed,
            allow_small_user_set=(len(tuning_users) < 1000),
        )
        ndcg_val = float(val_res.metrics.get("ndcg_at_10", 0.0))

        # Log nested MLflow run for this trial
        try:
            active_run = mlflow.active_run()
            if active_run:
                with mlflow.start_run(run_name=f"als_trial_{trial.number}", nested=True):
                    mlflow.log_params({
                        "hp_factors": factors,
                        "hp_regularization": regularization,
                        "hp_alpha": alpha,
                    })
                    mlflow.log_metrics({"val_ndcg_at_10": ndcg_val})
        except Exception as exc:
            logger.debug("Could not log nested MLflow run: %s", exc)

        return ndcg_val

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna ALS tuning complete. Best params: %s (Val NDCG@10: %.4f)", study.best_params, study.best_value)
    return study.best_params, study.best_value
