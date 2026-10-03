"""
Learning-to-Rank (LambdaMART / LightGBM) Stage-Two Recommender.

Implements:
  1. RankerModel(BaseRecommender): Multi-source candidate retrieval, feature extraction,
     and LambdaMART score reranking with full-catalog Two-Tower tail ordering.
  2. train_ranker: Training pipeline on disjoint ranker users (90% train / 10% early-stopping val)
     using LightGBM LGBMRanker with NDCG@10 optimization and candidate miss filtering.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from recsys.candidates import (
    DEFAULT_CANDIDATE_COUNTS,
    CandidateResult,
    generate_candidates,
)
from recsys.data import RecsysData
from recsys.evaluate import _build_user_interaction_maps, get_evaluation_user_split
from recsys.explain import global_shap
from recsys.models.base import BaseRecommender
from recsys.ranker_features import (
    FEATURE_COLUMNS,
    ItemFeatureCache,
    UserFeatureCache,
    build_features,
)

logger = logging.getLogger("steamiq.recsys.ranker")


@dataclass
class RankerConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 50
    colsample_bytree: float = 0.8
    lambdarank_truncation_level: int = 20
    eval_at: list[int] = None
    seed: int = 42
    n_jobs: int = 4
    max_candidates: int = 300

    def __post_init__(self) -> None:
        if self.eval_at is None:
            self.eval_at = [10]


class RankerModel(BaseRecommender):
    """
    Stage-Two Learning-to-Rank Recommender.
    Reranks candidates retrieved by Stage-One models and orders non-candidate items
    by Two-Tower scores so metric computation over the full item catalog remains exact.
    """
    name = "recsys_two_tower_ranker"

    def __init__(
        self,
        ranker: LGBMRanker | lgb.Booster | None = None,
        models: dict[str, BaseRecommender] | None = None,
        config: RankerConfig | None = None,
        feature_names: list[str] | None = None,
        data: RecsysData | None = None,
    ) -> None:
        self.ranker = ranker
        self.models = models or {}
        self.config = config or RankerConfig()
        self.feature_names = feature_names or list(FEATURE_COLUMNS)
        self.data = data
        self.item_cache: ItemFeatureCache | None = ItemFeatureCache(data) if data else None
        self.user_cache: UserFeatureCache | None = UserFeatureCache(data) if data else None
        self.n_items = data.n_items if data else 0

    def fit(self, data: RecsysData) -> None:
        """Required by BaseRecommender interface. Use train_ranker for full training."""
        self.data = data
        self.n_items = data.n_items
        self.item_cache = ItemFeatureCache(data)
        self.user_cache = UserFeatureCache(data)

    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        """
        Scores user histories against the full item catalog:
          1. Generates top 300 candidates from stage 1 models.
          2. Extracts tabular ranking features for candidate items.
          3. Predicts ranker scores for candidates.
          4. Assigns ranker scores to candidates and offsets Two-Tower scores for remaining items.

        Returns:
            np.ndarray of shape [len(histories), n_items], float32.
        """
        n_users = len(histories)
        if self.n_items == 0 and self.data is not None:
            self.n_items = self.data.n_items

        if n_users == 0 or self.n_items == 0:
            return np.empty((n_users, self.n_items), dtype=np.float32)

        # Ensure caches are initialized
        if self.item_cache is None and self.data is not None:
            self.item_cache = ItemFeatureCache(self.data)
        if self.user_cache is None and self.data is not None:
            self.user_cache = UserFeatureCache(self.data)

        # 1. Retrieve stage 1 Two-Tower scores as base score
        tt_model = self.models.get("two_tower") or self.models.get("recsys_two_tower")
        if tt_model is not None:
            base_scores = tt_model.score(histories)  # [B, n_items]
        else:
            base_scores = np.zeros((n_users, self.n_items), dtype=np.float32)

        # 2. Generate multi-source candidates (union capped at 300)
        cand_res: CandidateResult = generate_candidates(
            histories=histories,
            models=self.models,
            n_each=DEFAULT_CANDIDATE_COUNTS,
            max_candidates=self.config.max_candidates,
            batch_size=512,
        )

        # 3. Extract features for all candidates
        if self.data is not None:
            feat_df = build_features(
                histories=histories,
                candidates=cand_res,
                data=self.data,
                item_cache=self.item_cache,
                user_cache=self.user_cache,
            )
        else:
            return base_scores

        if feat_df.empty or self.ranker is None:
            return base_scores

        # 4. Predict ranker scores for candidates
        X_mat = feat_df[self.feature_names].to_numpy(dtype=np.float32)
        if isinstance(self.ranker, lgb.Booster):
            cand_preds = self.ranker.predict(X_mat)
        else:
            cand_preds = self.ranker.predict(X_mat)

        cand_preds = np.asarray(cand_preds, dtype=np.float32)

        # 5. Populate full-catalog score matrix
        # Initialize with Two-Tower base scores shifted down below min candidate score
        final_scores = np.zeros((n_users, self.n_items), dtype=np.float32)

        row_ptr = 0
        for u_i in range(n_users):
            u_cands = cand_res.user_candidates[u_i]
            n_cands = len(u_cands)

            if n_cands == 0:
                final_scores[u_i] = base_scores[u_i]
                continue

            u_preds = cand_preds[row_ptr:row_ptr + n_cands]
            row_ptr += n_cands

            min_pred = float(np.min(u_preds)) if len(u_preds) > 0 else 0.0

            # Scale and offset base scores for non-candidate items strictly below min_pred
            u_base = base_scores[u_i]
            max_base = float(np.max(u_base)) if len(u_base) > 0 else 1.0
            # Offset non-candidates: max possible base score becomes min_pred - 1.0
            final_scores[u_i] = u_base - max_base + min_pred - 1.0

            # Set candidate items to their model predictions
            for cand_it, p_score in zip(u_cands, u_preds, strict=True):
                if 0 <= cand_it < self.n_items:
                    final_scores[u_i, cand_it] = p_score

        return final_scores

    def save(self, model_dir: Path | str) -> None:
        """Persists ranker booster, config, and feature names to directory."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM booster model
        if self.ranker is not None:
            booster = getattr(self.ranker, "booster_", self.ranker)
            booster.save_model(str(path / "ranker_lgb.txt"))

        # Save config
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save feature names
        with open(path / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_names, f, indent=2)

    @classmethod
    def load(
        cls,
        model_dir: Path | str,
        data: RecsysData | None = None,
        models: dict[str, BaseRecommender] | None = None,
    ) -> RankerModel:
        """Loads RankerModel from disk."""
        path = Path(model_dir)
        booster_file = path / "ranker_lgb.txt"
        config_file = path / "config.json"
        feat_file = path / "feature_names.json"

        if not booster_file.exists():
            raise FileNotFoundError(f"Missing ranker booster file: {booster_file}")

        booster = lgb.Booster(model_file=str(booster_file))

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                cfg_dict = json.load(f)
            config = RankerConfig(**cfg_dict)
        else:
            config = RankerConfig()

        if feat_file.exists():
            with open(feat_file, encoding="utf-8") as f:
                feature_names = json.load(f)
        else:
            feature_names = list(FEATURE_COLUMNS)

        return cls(
            ranker=booster,
            models=models or {},
            config=config,
            feature_names=feature_names,
            data=data,
        )

    def hyperparameters(self) -> dict[str, Any]:
        return asdict(self.config)


def train_ranker(
    data: RecsysData,
    models: dict[str, BaseRecommender],
    cfg: RankerConfig | None = None,
    n_ranker_users: int = 15000,
    seed: int = 42,
    persistence_dir: Path | str | None = None,
    eval_users: list[int] | None = None,
    tuning_users: list[int] | None = None,
) -> tuple[RankerModel, dict[str, Any]]:
    """
    Trains the Stage-Two LightGBM LambdaMART ranker.

    Rules & Design:
      - Ranker training users are strictly disjoint from both evaluation and tuning users (Addendum 3).
      - Split ranker users into 90% training / 10% held-out validation users for early stopping (30 rounds).
      - History = user TRAIN positives; Label = 1.0 for VAL target, 0.0 for other candidates.
      - Users whose VAL target is not in the top 300 candidate union are dropped (no positive signal).

    Args:
        data: RecsysData bundle.
        models: Dict of stage 1 models (two_tower, itemknn, als, content).
        cfg: Optional RankerConfig.
        n_ranker_users: Maximum candidate training users to sample.
        seed: Random seed.
        persistence_dir: Directory to verify user splits against.

    Returns:
        (RankerModel, training_info_dict)
    """
    config = cfg or RankerConfig(seed=seed)
    logger.info("═══════════════════════════════════════════════════════════")
    logger.info("▶ Preparing Stage-Two Ranker Dataset (NDCG@10 LambdaMART)...")

    # 1. Get disjoint eval and tuning users to exclude them from ranker training
    if eval_users is None or tuning_users is None:
        n_total = len(data.val_df["user_idx"].unique())
        n_eval = 5000 if n_total >= 15000 else max(1, int(n_total * 0.15))
        n_tuning = 5000 if n_total >= 15000 else max(1, int(n_total * 0.15))
        eval_users, tuning_users = get_evaluation_user_split(
            data,
            n_eval_users=n_eval,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=persistence_dir,
        )
    excluded_users = set(eval_users).union(set(tuning_users))

    # Identify eligible ranker users with valid val targets
    val_user_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
    all_val_users = np.unique(data.val_df["user_idx"].to_numpy())
    eligible_users = np.array([u for u in all_val_users if u not in excluded_users], dtype=np.int64)
    if len(eligible_users) == 0:
        eligible_users = all_val_users

    rng = np.random.default_rng(seed)
    if len(eligible_users) > n_ranker_users:
        sampled_users = rng.choice(eligible_users, size=n_ranker_users, replace=False)
    else:
        sampled_users = eligible_users

    sampled_users = rng.permutation(sampled_users)

    # 90% train / 10% val split by user (with minimum 1 val user if >= 10 users)
    n_val_users = max(1, int(len(sampled_users) * 0.1)) if len(sampled_users) >= 10 else (1 if len(sampled_users) > 3 else 0)
    n_train_users = len(sampled_users) - n_val_users
    train_users = sampled_users[:n_train_users]
    val_users = sampled_users[n_train_users:] if n_val_users > 0 else np.array([], dtype=np.int64)

    logger.info(
        "Ranker users: %d total (train: %d users, val: %d users). Disjoint from %d eval + %d tuning users.",
        len(sampled_users),
        len(train_users),
        len(val_users),
        len(eval_users),
        len(tuning_users),
    )

    # 2. Build training and validation pairs
    user_train_pos, user_train_neg = _build_user_interaction_maps(data)

    def prepare_split_dataset(u_list: np.ndarray, split_name: str) -> tuple[pd.DataFrame, np.ndarray, list[int], int]:
        if len(u_list) == 0:
            return pd.DataFrame(columns=FEATURE_COLUMNS), np.array([], dtype=np.float32), [], 0

        histories: list[np.ndarray] = []
        seen_masks: list[set[int]] = []
        valid_u_ids: list[int] = []
        targets: list[int] = []

        for u in u_list:
            u_int = int(u)
            pos_items = user_train_pos.get(u_int, [])
            neg_items = user_train_neg.get(u_int, set())
            t_item = int(val_user_map[u_int])

            histories.append(np.asarray(pos_items, dtype=np.int64))
            seen_masks.append(set(pos_items).union(neg_items))
            valid_u_ids.append(u_int)
            targets.append(t_item)

        # Generate candidates
        logger.info("Generating candidate pools for %s (%d users)...", split_name, len(u_list))
        cand_res = generate_candidates(
            histories=histories,
            models=models,
            n_each=DEFAULT_CANDIDATE_COUNTS,
            seen_masks=seen_masks,
            max_candidates=config.max_candidates,
            batch_size=512,
        )

        # Filter users whose target is not in their candidate pool
        filtered_histories: list[np.ndarray] = []
        filtered_cands: list[list[int]] = []
        filtered_meta: list[list[Any]] = []
        filtered_u_ids: list[int] = []
        labels_list: list[float] = []
        groups: list[int] = []
        dropped_count = 0

        for i, (u_id, target, c_items, c_meta) in enumerate(
            zip(valid_u_ids, targets, cand_res.user_candidates, cand_res.candidate_metadata, strict=True)
        ):
            if target not in c_items:
                dropped_count += 1
                continue

            filtered_histories.append(histories[i])
            filtered_cands.append(c_items)
            filtered_meta.append(c_meta)
            filtered_u_ids.append(u_id)
            groups.append(len(c_items))

            for it in c_items:
                labels_list.append(1.0 if it == target else 0.0)

        logger.info(
            "%s: retained %d users (%d candidates), dropped %d users without target in candidates (miss rate: %.1f%%).",
            split_name,
            len(filtered_u_ids),
            sum(groups),
            dropped_count,
            (dropped_count / max(1, len(u_list))) * 100,
        )

        # Build features
        cand_container = CandidateResult(
            user_candidates=filtered_cands,
            candidate_metadata=filtered_meta,
            n_users=len(filtered_u_ids),
        )
        feat_df = build_features(
            histories=filtered_histories,
            candidates=cand_container,
            data=data,
            user_indices=filtered_u_ids,
        )

        return feat_df, np.array(labels_list, dtype=np.float32), groups, dropped_count

    # Build train and early-stopping validation feature matrices
    t0 = time.perf_counter()
    X_train_df, y_train, group_train, dropped_train = prepare_split_dataset(train_users, "Train Split")
    X_val_df, y_val, group_val, dropped_val = prepare_split_dataset(val_users, "Early-Stopping Val Split")
    feature_prep_time = time.perf_counter() - t0
    logger.info("Feature extraction completed in %.2fs", feature_prep_time)

    # 3. Fit LightGBM LambdaMART Ranker
    ranker = LGBMRanker(
        objective="lambdarank",
        eval_at=config.eval_at,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_child_samples=config.min_child_samples,
        colsample_bytree=config.colsample_bytree,
        lambdarank_truncation_level=config.lambdarank_truncation_level,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )

    fit_start = time.perf_counter()
    if len(group_val) > 0 and len(X_val_df) > 0:
        logger.info("Training LGBMRanker with early stopping (30 rounds)...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=25),
        ]
        ranker.fit(
            X=X_train_df[FEATURE_COLUMNS],
            y=y_train,
            group=group_train,
            eval_set=[(X_val_df[FEATURE_COLUMNS], y_val)],
            eval_group=[group_val],
            callbacks=callbacks,
        )
    else:
        logger.info("Training LGBMRanker without early stopping validation set...")
        ranker.fit(
            X=X_train_df[FEATURE_COLUMNS],
            y=y_train,
            group=group_train,
        )
    fit_time = time.perf_counter() - fit_start
    best_iteration = int(ranker.best_iteration_) if hasattr(ranker, "best_iteration_") else config.n_estimators
    logger.info("LGBMRanker fit complete in %.2fs (best iteration: %d)", fit_time, best_iteration)

    # 4. Feature importances and global SHAP
    booster: lgb.Booster = ranker.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    feat_importance_gain = {
        name: float(g)
        for name, g in sorted(zip(FEATURE_COLUMNS, gains, strict=True), key=lambda x: x[1], reverse=True)
    }
    feat_importance_split = {
        name: int(s)
        for name, s in sorted(zip(FEATURE_COLUMNS, splits, strict=True), key=lambda x: x[1], reverse=True)
    }

    # Global SHAP on 5,000 sampled rows
    shap_sample = X_train_df[FEATURE_COLUMNS].sample(n=min(5000, len(X_train_df)), random_state=seed)
    shap_dict = global_shap(booster, shap_sample, feature_names=FEATURE_COLUMNS, seed=seed)

    model_wrapper = RankerModel(
        ranker=booster,
        models=models,
        config=config,
        feature_names=FEATURE_COLUMNS,
        data=data,
    )

    training_info = {
        "best_iteration": best_iteration,
        "fit_time_sec": fit_time,
        "feature_prep_time_sec": feature_prep_time,
        "n_train_users": len(train_users),
        "n_val_users": len(val_users),
        "dropped_train_users": dropped_train,
        "dropped_val_users": dropped_val,
        "feature_importance_gain": feat_importance_gain,
        "feature_importance_split": feat_importance_split,
        "global_shap": shap_dict,
    }

    return model_wrapper, training_info
