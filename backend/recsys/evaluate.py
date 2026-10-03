"""
Unified evaluation framework for recommender systems.

Implements:
  - Fold-in user history construction (train for val; train+val for test, capped to max_history).
  - Seen-item masking (-inf) preventing recommendation of previously consumed or negative items.
  - Vectorized metric computation with 95% bootstrap confidence intervals.
  - Sliced evaluations by item popularity (head: top 20%, torso: next 30%, tail: bottom 50%)
    and user history length (3-4, 5-9, >=10).
  - Seeded, deterministic user sampling.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recsys.data import RecsysData
from recsys.metrics import (
    bootstrap_ci,
    catalog_coverage_at_k,
    compute_ranks,
    diversity_at_k,
    get_top_k_indices,
    head_share_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
)
from recsys.models.base import BaseRecommender

logger = logging.getLogger("steamiq.recsys.evaluate")


@dataclass
class EvaluationResult:
    """
    Standard evaluation output holding scalar metrics, bootstrap intervals,
    per-user metric arrays, and slice breakdowns.
    """
    split: str
    n_users: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_user: dict[str, np.ndarray] = field(default_factory=dict)
    slices: dict[str, dict[str, float]] = field(default_factory=dict)
    evaluated_user_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def get_prefixed_metrics(self, prefix: str | None = None) -> dict[str, float]:
        """
        Returns all scalar metrics with an optional prefix (e.g., 'val_' or 'test_').
        """
        p = f"{prefix.rstrip('_')}_" if prefix else f"{self.split}_"
        prefixed = {}
        for k, v in self.metrics.items():
            prefixed[f"{p}{k}"] = float(v)
        for s_name, s_dict in self.slices.items():
            for m_name, m_val in s_dict.items():
                prefixed[f"{p}{s_name}_{m_name}"] = float(m_val)
        return prefixed


def _build_user_interaction_maps(
    data: RecsysData,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """
    Constructs per-user chronological train sequences and train negative item sets.
    """
    # Sort train by user and timestamp
    train_sorted = data.train_df.sort_values(by=["user_idx", "ts"])
    user_train_pos: dict[int, list[int]] = {}
    for u, item in zip(train_sorted["user_idx"].to_numpy(), train_sorted["item_idx"].to_numpy(), strict=True):
        if u not in user_train_pos:
            user_train_pos[u] = []
        user_train_pos[u].append(int(item))

    # Negatives before cutoff (split == 'train')
    user_train_neg: dict[int, set[int]] = {}
    if not data.negatives_df.empty:
        neg_train = data.negatives_df[data.negatives_df["split"] == "train"]
        for u, item in zip(neg_train["user_idx"].to_numpy(), neg_train["item_idx"].to_numpy(), strict=True):
            if u not in user_train_neg:
                user_train_neg[u] = set()
            user_train_neg[u].add(int(item))

    return user_train_pos, user_train_neg



def get_evaluation_user_split(
    data: RecsysData,
    n_eval_users: int = 5000,
    n_tuning_users: int = 5000,
    seed: int = 42,
    persistence_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partitions users into disjoint evaluation and tuning sets for leak-free hyperparameter search.
    Persists eval_users.json and tuning_users.json ONLY when persistence_dir is explicitly provided.

    Args:
        data: RecsysData container.
        n_eval_users: Number of users in the fixed evaluation set (default: 5000).
        n_tuning_users: Number of users in the disjoint tuning set (default: 5000).
        seed: Random seed for partition.
        persistence_dir: Optional directory where split JSON files are saved/loaded. If None, computes in-memory only.

    Returns:
        (eval_users, tuning_users) where set(eval_users).isdisjoint(set(tuning_users)) is True.
    """
    available_users = np.unique(data.val_df["user_idx"].to_numpy())
    if len(available_users) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Calculate expected sizes
    if n_eval_users + n_tuning_users > len(available_users):
        expected_eval_len = max(1, int(len(available_users) * 0.5))
        expected_tuning_len = len(available_users) - expected_eval_len
    else:
        expected_eval_len = min(n_eval_users, len(available_users))
        expected_tuning_len = min(n_tuning_users, max(0, len(available_users) - expected_eval_len))

    if persistence_dir is not None:
        target_dir = Path(persistence_dir)
        eval_file = target_dir / "eval_users.json"
        tuning_file = target_dir / "tuning_users.json"

        if eval_file.exists() and tuning_file.exists():
            with open(eval_file, encoding="utf-8") as f:
                eval_raw = json.load(f)
            with open(tuning_file, encoding="utf-8") as f:
                tuning_raw = json.load(f)

            eval_list = eval_raw.get("eval_users", eval_raw if isinstance(eval_raw, list) else [])
            tuning_list = tuning_raw.get("tuning_users", tuning_raw if isinstance(tuning_raw, list) else [])

            # 1. Dataset version validation
            file_version = eval_raw.get("dataset_version") if isinstance(eval_raw, dict) else None
            if file_version is not None and file_version != data.dataset_version:
                raise ValueError(
                    f"Persisted user split dataset_version ({file_version}) does not match current "
                    f"data.dataset_version ({data.dataset_version}). Please delete {eval_file} and {tuning_file} and re-run."
                )

            # 2. Length validation
            if len(eval_list) != expected_eval_len or len(tuning_list) != expected_tuning_len:
                raise ValueError(
                    f"Persisted user split length mismatch: eval={len(eval_list)} (expected {expected_eval_len}), "
                    f"tuning={len(tuning_list)} (expected {expected_tuning_len}). "
                    f"Please delete {eval_file} and {tuning_file} and re-run."
                )

            # 3. ID range and valid user validation
            max_user_limit = len(data.users_df) if not data.users_df.empty else (available_users.max() + 1)
            eval_arr = np.array(eval_list, dtype=np.int64)
            tuning_arr = np.array(tuning_list, dtype=np.int64)

            if len(eval_arr) > 0 and (eval_arr.min() < 0 or eval_arr.max() >= max_user_limit):
                raise ValueError(
                    f"Persisted eval_users contain out-of-range user IDs (min={eval_arr.min()}, max={eval_arr.max()}, "
                    f"limit={max_user_limit}). Please delete {eval_file} and {tuning_file} and re-run."
                )
            if len(tuning_arr) > 0 and (tuning_arr.min() < 0 or tuning_arr.max() >= max_user_limit):
                raise ValueError(
                    f"Persisted tuning_users contain out-of-range user IDs (min={tuning_arr.min()}, max={tuning_arr.max()}, "
                    f"limit={max_user_limit}). Please delete {eval_file} and {tuning_file} and re-run."
                )

            # 4. Disjointness check
            if not set(eval_list).isdisjoint(set(tuning_list)):
                raise ValueError(
                    f"Persisted eval_users and tuning_users are not disjoint. Please delete {eval_file} and {tuning_file} and re-run."
                )

            return eval_arr, tuning_arr

    # Generate split deterministically
    rng = np.random.default_rng(seed)
    perm = rng.permutation(available_users)

    if n_eval_users + n_tuning_users > len(perm):
        eval_count = max(1, int(len(perm) * 0.5))
        eval_users = np.sort(perm[:eval_count])
        tuning_users = np.sort(perm[eval_count:])
    else:
        eval_count = min(n_eval_users, len(perm))
        eval_users = np.sort(perm[:eval_count])
        remaining = perm[eval_count:]
        tuning_count = min(n_tuning_users, len(remaining))
        tuning_users = np.sort(remaining[:tuning_count])

    # Persist to disk ONLY if persistence_dir was explicitly passed
    if persistence_dir is not None:
        target_dir = Path(persistence_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        eval_file = target_dir / "eval_users.json"
        tuning_file = target_dir / "tuning_users.json"

        payload = {
            "dataset_version": data.dataset_version,
            "seed": seed,
            "n_users": len(available_users),
            "eval_users": eval_users.tolist(),
            "tuning_users": tuning_users.tolist(),
        }
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(tuning_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        manifest_file = target_dir / "data_manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    manifest = json.load(f)
                file_sha1s = manifest.get("file_sha1s", {})
                file_sha1s["eval_users.json"] = hashlib.sha1(eval_file.read_bytes()).hexdigest()
                file_sha1s["tuning_users.json"] = hashlib.sha1(tuning_file.read_bytes()).hexdigest()
                manifest["file_sha1s"] = file_sha1s
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
            except Exception as exc:
                logger.warning("Could not update data_manifest.json SHA1s: %s", exc)

    return eval_users, tuning_users



def evaluate_model(
    model: BaseRecommender,
    data: RecsysData,
    split: str = "val",
    k_list: tuple[int, ...] = (10, 20),
    user_indices: np.ndarray | None = None,
    max_users: int = 5000,
    batch_size: int = 1024,
    max_history: int = 50,
    seed: int = 42,
    n_boot: int = 1000,
    allow_small_user_set: bool = False,
) -> EvaluationResult:
    """
    Evaluates a recommender model on the validation or test split.

    Args:
        model: BaseRecommender model instance (scoring via history fold-in).
        data: RecsysData artifact container.
        split: 'val' or 'test'.
        k_list: Cutoffs for ranking metrics.
        user_indices: Optional explicit array of user_idx to evaluate on (overrides max_users sampling).
        max_users: Max evaluation users to sample if user_indices is None.
        batch_size: User batch size for model.score().
        max_history: Max recent items to retain in fold-in history.
        seed: Random seed for sampling and bootstrap reproducibility.
        n_boot: Number of bootstrap resamples for confidence intervals.
        allow_small_user_set: If False, guards against running on < 1,000 users.

    Returns:
        EvaluationResult containing metrics, CIs, per-user arrays, and slices.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")

    split_df = data.val_df if split == "val" else data.test_df
    if split_df.empty:
        raise ValueError(f"Split dataframe '{split}' is empty")

    # 1. Select evaluation users (explicit user_indices or fixed random sample)
    if user_indices is not None:
        eval_users = np.sort(np.asarray(user_indices, dtype=np.int64))
    else:
        all_users = np.unique(split_df["user_idx"].to_numpy())
        if len(all_users) > max_users:
            rng = np.random.default_rng(seed)
            eval_users = rng.choice(all_users, size=max_users, replace=False)
            eval_users = np.sort(eval_users)
        else:
            eval_users = np.sort(all_users)

    n_eval = len(eval_users)
    if n_eval < 1000 and not allow_small_user_set:
        raise ValueError(
            f"Evaluating on {n_eval} users is below the safety threshold of 1,000 users. "
            f"If this is a unit test, pass allow_small_user_set=True."
        )

    print(f"Evaluating on {n_eval} users ({split} split)...")
    logger.info("Evaluating %s on %s split (n_eval=%d, seed=%d)...", model.name, split, n_eval, seed)

    user_train_pos, user_train_neg = _build_user_interaction_maps(data)


    # Map target items
    target_map = dict(zip(split_df["user_idx"].to_numpy(), split_df["item_idx"].to_numpy(), strict=True))
    val_target_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))


    # 2. Build histories, targets, and seen masks for evaluated users
    histories: list[np.ndarray] = []
    targets: list[int] = []
    seen_masks: list[list[int]] = []
    history_lengths: list[int] = []

    for u in eval_users:
        t_item = int(target_map[u])
        targets.append(t_item)

        pos_items = user_train_pos.get(u, [])
        neg_items = user_train_neg.get(u, set())

        if split == "val":
            # Fold-in history: train positives
            user_hist = pos_items[-max_history:] if len(pos_items) > max_history else pos_items
            # Seen items to mask out: train positives + train negatives
            seen = set(pos_items) | neg_items
            hist_len = len(pos_items)
        else:
            # test split: fold-in history: train positives + val positive
            val_item = val_target_map.get(u)
            full_pos = list(pos_items)
            if val_item is not None:
                full_pos.append(int(val_item))
            user_hist = full_pos[-max_history:] if len(full_pos) > max_history else full_pos
            seen = set(full_pos) | neg_items
            hist_len = len(full_pos)

        histories.append(np.array(user_hist, dtype=np.int32))
        seen_masks.append(list(seen))
        history_lengths.append(hist_len)

    targets_arr = np.array(targets, dtype=np.int64)
    history_lengths_arr = np.array(history_lengths, dtype=np.int32)

    # 3. Batch scoring and seen-item masking
    all_scores = np.empty((n_eval, data.n_items), dtype=np.float32)

    for i in range(0, n_eval, batch_size):
        batch_slice = slice(i, min(i + batch_size, n_eval))
        batch_hist = histories[batch_slice]
        batch_scores = model.score(batch_hist).astype(np.float32, copy=False)

        # Mask seen items to -inf for each user in batch
        for idx, global_idx in enumerate(range(batch_slice.start, batch_slice.stop)):
            seen_items = seen_masks[global_idx]
            if seen_items:
                batch_scores[idx, seen_items] = -np.inf

        all_scores[batch_slice] = batch_scores

    # 4. Compute Ranks and Per-User Metrics
    ranks = compute_ranks(all_scores, targets_arr)

    per_user: dict[str, np.ndarray] = {
        "rank": ranks,
        "history_length": history_lengths_arr,
        "target_item": targets_arr,
    }

    metrics: dict[str, float] = {
        "n_users": float(n_eval),
    }

    for k in k_list:
        hr = hit_rate_at_k(ranks, k)
        ndcg = ndcg_at_k(ranks, k)
        mrr = mrr_at_k(ranks, k)

        per_user[f"hit_rate_at_{k}"] = hr
        per_user[f"ndcg_at_{k}"] = ndcg
        per_user[f"mrr_at_{k}"] = mrr

        # Point estimates & Bootstrap 95% CIs
        hr_m, hr_lo, hr_hi = bootstrap_ci(hr, n_boot=n_boot, seed=seed)
        ndcg_m, ndcg_lo, ndcg_hi = bootstrap_ci(ndcg, n_boot=n_boot, seed=seed)
        mrr_m, mrr_lo, mrr_hi = bootstrap_ci(mrr, n_boot=n_boot, seed=seed)

        metrics[f"hit_rate_at_{k}"] = hr_m
        metrics[f"hit_rate_at_{k}_lo95"] = hr_lo
        metrics[f"hit_rate_at_{k}_hi95"] = hr_hi

        metrics[f"ndcg_at_{k}"] = ndcg_m
        metrics[f"ndcg_at_{k}_lo95"] = ndcg_lo
        metrics[f"ndcg_at_{k}_hi95"] = ndcg_hi

        metrics[f"mrr_at_{k}"] = mrr_m
        metrics[f"mrr_at_{k}_lo95"] = mrr_lo
        metrics[f"mrr_at_{k}_hi95"] = mrr_hi

    # Candidate recall depths (e.g. recall@50, recall@100 if n_items >= depth)
    for depth in (50, 100):
        if data.n_items >= depth:
            rec = hit_rate_at_k(ranks, depth)
            rec_m, rec_lo, rec_hi = bootstrap_ci(rec, n_boot=n_boot, seed=seed)
            metrics[f"candidate_recall_at_{depth}"] = rec_m
            metrics[f"candidate_recall_at_{depth}_lo95"] = rec_lo
            metrics[f"candidate_recall_at_{depth}_hi95"] = rec_hi

    # 5. Catalog-level and Distributional Metrics
    max_k = max(k_list) if k_list else 10
    top_max_k = get_top_k_indices(all_scores, max_k)

    for k in k_list:
        top_k = top_max_k[:, :k]
        metrics[f"catalog_coverage_at_{k}"] = catalog_coverage_at_k(top_k, data.n_items)
        metrics[f"novelty_at_{k}"] = novelty_at_k(top_k, data.item_train_pop)
        metrics[f"head_share_at_{k}"] = head_share_at_k(top_k, data.item_train_pop, head_fraction=0.01)

        if data.item_text_emb is not None and len(data.item_text_emb) == data.n_items:
            metrics[f"diversity_at_{k}"] = diversity_at_k(top_k, data.item_text_emb)

    # 6. Sliced Evaluations
    slices: dict[str, dict[str, float]] = {}

    # Popularity Buckets: Head (top 20%), Torso (next 30%), Tail (last 50%)
    n_items = data.n_items
    sorted_pop_items = np.argsort(-data.item_train_pop)
    head_cutoff = int(0.20 * n_items)
    torso_cutoff = int(0.50 * n_items)

    head_set = set(sorted_pop_items[:head_cutoff])
    torso_set = set(sorted_pop_items[head_cutoff:torso_cutoff])

    pop_bucket_indices: dict[str, list[int]] = {"pop_head": [], "pop_torso": [], "pop_tail": []}
    for idx, t in enumerate(targets_arr):
        if t in head_set:
            pop_bucket_indices["pop_head"].append(idx)
        elif t in torso_set:
            pop_bucket_indices["pop_torso"].append(idx)
        else:
            pop_bucket_indices["pop_tail"].append(idx)

    for bucket_name, u_indices in pop_bucket_indices.items():
        if len(u_indices) > 0:
            idx_arr = np.array(u_indices, dtype=np.int64)
            slice_ranks = ranks[idx_arr]
            slice_ndcg = ndcg_at_k(slice_ranks, 10)
            slice_hr = hit_rate_at_k(slice_ranks, 10)
            ndcg_m, ndcg_lo, ndcg_hi = bootstrap_ci(slice_ndcg, n_boot=n_boot, seed=seed)
            hr_m, hr_lo, hr_hi = bootstrap_ci(slice_hr, n_boot=n_boot, seed=seed)
            slices[bucket_name] = {
                "count": float(len(u_indices)),
                "hit_rate_at_10": hr_m,
                "hit_rate_at_10_lo95": hr_lo,
                "hit_rate_at_10_hi95": hr_hi,
                "ndcg_at_10": ndcg_m,
                "ndcg_at_10_lo95": ndcg_lo,
                "ndcg_at_10_hi95": ndcg_hi,
                "mrr_at_10": float(np.mean(mrr_at_k(slice_ranks, 10))),
            }
        else:
            slices[bucket_name] = {
                "count": 0.0,
                "hit_rate_at_10": 0.0,
                "hit_rate_at_10_lo95": 0.0,
                "hit_rate_at_10_hi95": 0.0,
                "ndcg_at_10": 0.0,
                "ndcg_at_10_lo95": 0.0,
                "ndcg_at_10_hi95": 0.0,
                "mrr_at_10": 0.0,
            }

    # History Length Buckets: 3-4, 5-9, >=10
    hist_bucket_indices: dict[str, list[int]] = {"hist_3_4": [], "hist_5_9": [], "hist_10_plus": []}
    for idx, length in enumerate(history_lengths_arr):
        if length <= 4:
            hist_bucket_indices["hist_3_4"].append(idx)
        elif length <= 9:
            hist_bucket_indices["hist_5_9"].append(idx)
        else:
            hist_bucket_indices["hist_10_plus"].append(idx)

    for bucket_name, u_indices in hist_bucket_indices.items():
        if len(u_indices) > 0:
            idx_arr = np.array(u_indices, dtype=np.int64)
            slice_ranks = ranks[idx_arr]
            slice_ndcg = ndcg_at_k(slice_ranks, 10)
            slice_hr = hit_rate_at_k(slice_ranks, 10)
            ndcg_m, ndcg_lo, ndcg_hi = bootstrap_ci(slice_ndcg, n_boot=n_boot, seed=seed)
            hr_m, hr_lo, hr_hi = bootstrap_ci(slice_hr, n_boot=n_boot, seed=seed)
            slices[bucket_name] = {
                "count": float(len(u_indices)),
                "hit_rate_at_10": hr_m,
                "hit_rate_at_10_lo95": hr_lo,
                "hit_rate_at_10_hi95": hr_hi,
                "ndcg_at_10": ndcg_m,
                "ndcg_at_10_lo95": ndcg_lo,
                "ndcg_at_10_hi95": ndcg_hi,
                "mrr_at_10": float(np.mean(mrr_at_k(slice_ranks, 10))),
            }
        else:
            slices[bucket_name] = {
                "count": 0.0,
                "hit_rate_at_10": 0.0,
                "hit_rate_at_10_lo95": 0.0,
                "hit_rate_at_10_hi95": 0.0,
                "ndcg_at_10": 0.0,
                "ndcg_at_10_lo95": 0.0,
                "ndcg_at_10_hi95": 0.0,
                "mrr_at_10": 0.0,
            }

    logger.info(
        "Finished evaluation for %s on %s: HR@10=%.4f (95%% CI: [%.4f, %.4f]), NDCG@10=%.4f (95%% CI: [%.4f, %.4f])",
        model.name,
        split,
        metrics.get("hit_rate_at_10", 0.0),
        metrics.get("hit_rate_at_10_lo95", 0.0),
        metrics.get("hit_rate_at_10_hi95", 0.0),
        metrics.get("ndcg_at_10", 0.0),
        metrics.get("ndcg_at_10_lo95", 0.0),
        metrics.get("ndcg_at_10_hi95", 0.0),
    )

    return EvaluationResult(
        split=split,
        n_users=n_eval,
        metrics=metrics,
        per_user=per_user,
        slices=slices,
        evaluated_user_indices=eval_users,
    )
