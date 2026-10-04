"""
Cold-Start Evaluation & Simulation Study for Recommender Systems.

Implements three complementary cold-start evaluation protocols:
  1. Sparse-Target Subgroup Study (Natural Cold-Start):
     Evaluates test users whose target item has <= 10 training positives.
     Compares: Random, Popularity, Content-Only, ALS, Two-Tower (masked IDs), Two-Tower (intact).
  2. Three Simulated New-Item Setups (Inference-time cold-start, no retraining):
     - C1: Old setup (random 10% items, train pop zeroed, ID masked, proxies leaked).
     - C2: Fair random simulation (random 10% items, train pop zeroed, ID masked, rating/review proxies neutralized).
     - C3: Fair torso/tail simulation (10% items from bottom 50% pop, ID masked, proxies neutralized).
     Compares: Random, Popularity, ALS, Item-kNN, Content-Only, Two-Tower (masked), Two-Tower (no-id).
  3. True Item Holdout Study (Strongest cold-start evidence, trained from scratch):
     Removes 10% items (torso/tail, seed 42) from training pairs/histories.
     Evaluates test users whose target is a held-out item.
     Compares: Two-Tower, Two-Tower (no-id), Content-Only, Random.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from recsys.data import RecsysData
from recsys.evaluate import _build_user_interaction_maps
from recsys.metrics import (
    bootstrap_ci,
    compute_ranks,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)
from recsys.models.base import BaseRecommender
from recsys.models.two_tower import TwoTowerModel

logger = logging.getLogger("steamiq.recsys.coldstart")


def get_sparse_target_test_users(
    data: RecsysData,
    eval_users: np.ndarray | list[int],
    max_train_positives: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies test evaluation users whose target item has <= max_train_positives.

    Args:
        data: RecsysData artifact bundle.
        eval_users: Array or list of fixed evaluation user indices.
        max_train_positives: Upper threshold for training positive count (default: 10).

    Returns:
        (sparse_user_indices, sparse_target_items)
    """
    eval_users_arr = np.asarray(eval_users, dtype=np.int64)
    test_target_map = dict(
        zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True)
    )

    item_pop = np.asarray(data.item_train_pop, dtype=np.int64)

    sparse_users: list[int] = []
    sparse_targets: list[int] = []

    for u in eval_users_arr:
        t_item = int(test_target_map[int(u)])
        pop = int(item_pop[t_item]) if t_item < len(item_pop) else 0
        if pop <= max_train_positives:
            sparse_users.append(int(u))
            sparse_targets.append(t_item)

    return np.array(sparse_users, dtype=np.int64), np.array(sparse_targets, dtype=np.int64)


def get_simulated_new_items_test_users(
    data: RecsysData,
    eval_users: np.ndarray | list[int],
    fraction: float = 0.10,
    seed: int = 42,
    variant: str = "c2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Designates a fraction of catalog items as 'new items' (seed 42)
    and identifies test users whose held-out target item is one of these new items.

    Args:
        data: RecsysData artifact bundle.
        eval_users: Array or list of fixed evaluation user indices.
        fraction: Fraction of catalog items to mask as new (default: 0.10 = 10%).
        seed: Random seed for new item selection.
        variant: 'c1' (random 10%), 'c2' (random 10% fair), or 'c3' (bottom 50% items fair).

    Returns:
        (new_item_ids, masked_user_indices, masked_target_items)
    """
    n_items = data.n_items
    n_new = int(round(fraction * n_items))
    rng = np.random.default_rng(seed)

    if variant == "c3":
        # Draw new items only from bottom 50% of items by training popularity
        sorted_pop_items = np.argsort(data.item_train_pop)  # ascending order
        bottom_50 = sorted_pop_items[: int(0.50 * n_items)]
        new_item_ids = rng.choice(bottom_50, size=min(n_new, len(bottom_50)), replace=False)
    else:
        # C1 and C2: Draw random 10% from entire catalog
        new_item_ids = rng.choice(n_items, size=n_new, replace=False)

    new_items_set = set(int(it) for it in new_item_ids)

    eval_users_arr = np.asarray(eval_users, dtype=np.int64)
    test_target_map = dict(
        zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True)
    )

    masked_users: list[int] = []
    masked_targets: list[int] = []

    for u in eval_users_arr:
        t_item = int(test_target_map[int(u)])
        if t_item in new_items_set:
            masked_users.append(int(u))
            masked_targets.append(t_item)

    return new_item_ids, np.array(masked_users, dtype=np.int64), np.array(masked_targets, dtype=np.int64)


def _build_test_foldin_data(
    data: RecsysData,
    user_indices: np.ndarray,
    max_history: int = 50,
) -> tuple[list[np.ndarray], list[int], list[set[int]]]:
    """Builds test fold-in histories, targets, and seen masks for specific evaluation users."""
    user_train_pos, user_train_neg = _build_user_interaction_maps(data)
    test_target_map = dict(zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True))
    val_target_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))

    histories: list[np.ndarray] = []
    targets: list[int] = []
    seen_masks: list[set[int]] = []

    for u in user_indices:
        u_int = int(u)
        t_item = int(test_target_map[u_int])
        targets.append(t_item)

        pos_items = user_train_pos.get(u_int, [])
        neg_items = user_train_neg.get(u_int, set())
        val_item = val_target_map.get(u_int)

        full_pos = list(pos_items)
        if val_item is not None:
            full_pos.append(int(val_item))

        user_hist = full_pos[-max_history:] if len(full_pos) > max_history else full_pos
        seen = set(full_pos) | neg_items

        histories.append(np.array(user_hist, dtype=np.int32))
        seen_masks.append(seen)

    return histories, targets, seen_masks


def evaluate_sparse_targets_study(
    data: RecsysData,
    models: dict[str, BaseRecommender],
    eval_users: np.ndarray | list[int],
    max_train_positives: int = 10,
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Executes Part 1 of the cold-start study: TEST users whose target item has <= 10 train positives.

    Compares:
      - Random Scorer reference (HR@10 = 10 / n_items)
      - Popularity baseline
      - Content-only
      - ALS (cannot score items without CF signals; reports what it gets)
      - Two-Tower with target item ID embeddings MASKED via item_vectors(mask_ids=...)
      - Two-Tower with IDs intact
    """
    sparse_users, sparse_targets = get_sparse_target_test_users(
        data, eval_users, max_train_positives=max_train_positives
    )
    n_sparse = len(sparse_users)
    logger.info("Found %d test evaluation users whose target item has <= %d train positives", n_sparse, max_train_positives)

    # Reference Random Scorer
    random_hr = float(10.0 / data.n_items)
    ranks_10 = np.arange(1, 11)
    random_ndcg = float(np.sum(1.0 / np.log2(ranks_10 + 1.0)) / data.n_items)
    random_mrr = float(np.sum(1.0 / ranks_10) / data.n_items)

    if n_sparse == 0:
        return {
            "study_type": "sparse_targets",
            "n_eval_users": 0,
            "max_train_positives": max_train_positives,
            "models": {
                "random": {
                    "hit_rate_at_10": random_hr,
                    "hit_rate_at_10_lo95": random_hr,
                    "hit_rate_at_10_hi95": random_hr,
                    "ndcg_at_10": random_ndcg,
                    "ndcg_at_10_lo95": random_ndcg,
                    "ndcg_at_10_hi95": random_ndcg,
                    "mrr_at_10": random_mrr,
                }
            },
            "simulation_disclaimer": "Simulated cold-start via ID masking; not a true prospective trial.",
        }

    histories, targets, seen_masks = _build_test_foldin_data(data, sparse_users)
    targets_arr = np.asarray(targets, dtype=np.int64)

    results: dict[str, dict[str, float]] = {
        "random": {
            "hit_rate_at_10": random_hr,
            "hit_rate_at_10_lo95": random_hr,
            "hit_rate_at_10_hi95": random_hr,
            "ndcg_at_10": random_ndcg,
            "ndcg_at_10_lo95": random_ndcg,
            "ndcg_at_10_hi95": random_ndcg,
            "mrr_at_10": random_mrr,
        }
    }

    # 1. Popularity
    if "popularity" in models:
        scores = models["popularity"].score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["popularity"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 2. Content-only
    if "content" in models:
        scores = models["content"].score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["content"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 3. ALS
    if "als" in models:
        scores = models["als"].score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["als"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 4. Two-Tower (IDs intact)
    if "two_tower" in models and isinstance(models["two_tower"], TwoTowerModel):
        tt_model = models["two_tower"]
        scores = tt_model.score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["two_tower_intact"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

        # 5. Two-Tower (Target Item ID MASKED)
        unique_targets = list(set(sparse_targets.tolist()))
        masked_item_vecs = tt_model.item_vectors(mask_ids=unique_targets)
        scores_masked = tt_model.score(histories, item_vecs=masked_item_vecs, mask_ids=unique_targets).copy()
        for i, s_mask in enumerate(seen_masks):
            scores_masked[i, list(s_mask)] = -np.inf
        ranks_masked = compute_ranks(scores_masked, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks_masked, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks_masked, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks_masked, 10)))
        results["two_tower_masked"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    return {
        "study_type": "sparse_targets",
        "n_eval_users": n_sparse,
        "max_train_positives": max_train_positives,
        "models": results,
        "limitations_note": (
            "Masking the target item ID embedding simulates a brand-new item cold-start. "
            "This is a simulated offline ablation and not a true prospective trial."
        ),
    }


def evaluate_simulated_new_items_study(
    data: RecsysData,
    models: dict[str, BaseRecommender],
    eval_users: np.ndarray | list[int],
    fraction: float = 0.10,
    seed: int = 42,
    variant: str = "c2",
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Executes a simulated new-item study.

    Variants:
      - 'c1': Old setup (random 10% items, only train popularity zeroed, ID masked; proxies leaked).
      - 'c2': Fair simulation (random 10% items, train pop zeroed, ID masked, rating/review proxies neutralized).
      - 'c3': Fair simulation (10% items drawn from bottom 50% pop, ID masked, proxies neutralized).

    Models evaluated:
      - Random Scorer (HR@10 = 10 / n_items)
      - Popularity (zeroed for new items)
      - ALS (zeroed for new items)
      - Item-kNN (zeroed for new items)
      - Content-Only (cosine similarity with history)
      - Two-Tower Masked (with proxies neutralized in C2/C3)
      - Two-Tower No-ID (with proxies neutralized in C2/C3)
    """
    new_item_ids, masked_users, masked_targets = get_simulated_new_items_test_users(
        data, eval_users, fraction=fraction, seed=seed, variant=variant
    )
    n_masked_users = len(masked_users)
    variant_labels = {
        "c1": "C1: Old Setup (Random 10%, Pop Zeroed, Proxies Leaked)",
        "c2": "C2: Fair Simulation (Random 10%, Pop Zeroed, Proxies Neutralized)",
        "c3": "C3: Fair Simulation (Bottom 50% Torso/Tail, Proxies Neutralized)",
    }
    v_label = variant_labels.get(variant, f"Simulation {variant.upper()}")
    logger.info("%s: %d new items, %d evaluated test users", v_label, len(new_item_ids), n_masked_users)

    # Reference Random Scorer
    random_hr = float(10.0 / data.n_items)
    ranks_10 = np.arange(1, 11)
    random_ndcg = float(np.sum(1.0 / np.log2(ranks_10 + 1.0)) / data.n_items)
    random_mrr = float(np.sum(1.0 / ranks_10) / data.n_items)

    if n_masked_users == 0:
        return {
            "study_type": "simulated_new_items",
            "variant": variant,
            "variant_label": v_label,
            "is_fair_simulation": variant in ("c2", "c3"),
            "n_eval_users": 0,
            "n_new_items": len(new_item_ids),
            "models": {
                "random": {
                    "hit_rate_at_10": random_hr,
                    "hit_rate_at_10_lo95": random_hr,
                    "hit_rate_at_10_hi95": random_hr,
                    "ndcg_at_10": random_ndcg,
                    "ndcg_at_10_lo95": random_ndcg,
                    "ndcg_at_10_hi95": random_ndcg,
                    "mrr_at_10": random_mrr,
                }
            },
            "random_reference": {"hit_rate_at_10": random_hr, "formula": "10 / n_items"},
            "limitations_note": "Simulated new item evaluation: content encoders saw item metadata during training.",
        }

    histories, targets, seen_masks = _build_test_foldin_data(data, masked_users)
    targets_arr = np.asarray(targets, dtype=np.int64)
    new_items_list = new_item_ids.tolist()

    neutralize_list = new_items_list if variant in ("c2", "c3") else None

    results: dict[str, dict[str, float]] = {
        "random": {
            "hit_rate_at_10": random_hr,
            "hit_rate_at_10_lo95": random_hr,
            "hit_rate_at_10_hi95": random_hr,
            "ndcg_at_10": random_ndcg,
            "ndcg_at_10_lo95": random_ndcg,
            "ndcg_at_10_hi95": random_ndcg,
            "mrr_at_10": random_mrr,
        }
    }

    # 1. Popularity (zeroed for new items)
    if "popularity" in models:
        scores = models["popularity"].score(histories).copy()
        scores[:, new_item_ids] = 0.0
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["popularity"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 2. ALS (zeroed for new items)
    if "als" in models:
        scores = models["als"].score(histories).copy()
        scores[:, new_item_ids] = 0.0
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["als"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 3. Item-kNN (zeroed for new items)
    if "itemknn" in models:
        scores = models["itemknn"].score(histories).copy()
        scores[:, new_item_ids] = 0.0
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["itemknn"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 4. Content-only
    if "content" in models:
        scores = models["content"].score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["content"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 5. Two-Tower (Masked New Items)
    if "two_tower" in models and isinstance(models["two_tower"], TwoTowerModel):
        tt_model = models["two_tower"]
        masked_vecs = tt_model.item_vectors(mask_ids=new_items_list, neutralize_proxies=neutralize_list)
        scores = tt_model.score(
            histories,
            item_vecs=masked_vecs,
            mask_ids=new_items_list,
            neutralize_proxies=neutralize_list,
        ).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["two_tower_masked"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # 6. No-ID Two-Tower Variant
    if "two_tower_no_id" in models and isinstance(models["two_tower_no_id"], TwoTowerModel):
        noid_model = models["two_tower_no_id"]
        masked_vecs = noid_model.item_vectors(mask_ids=new_items_list, neutralize_proxies=neutralize_list)
        scores = noid_model.score(
            histories,
            item_vecs=masked_vecs,
            mask_ids=new_items_list,
            neutralize_proxies=neutralize_list,
        ).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["two_tower_no_id"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    return {
        "study_type": "simulated_new_items",
        "variant": variant,
        "variant_label": v_label,
        "is_fair_simulation": variant in ("c2", "c3"),
        "n_eval_users": n_masked_users,
        "n_new_items": len(new_item_ids),
        "fraction_masked": fraction,
        "models": results,
        "random_reference": {
            "hit_rate_at_10": random_hr,
            "ndcg_at_10": random_ndcg,
            "formula": "10 / n_items",
        },
        "limitations_note": (
            "This is a simulation: the content encoders saw item metadata during training. "
            f"{'Rating and review proxies were neutralized to standardized 0.' if variant in ('c2', 'c3') else 'Popularity proxy was not neutralized in C1.'}"
        ),
    }


def get_item_holdout_split(
    data: RecsysData,
    fraction: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, set[int]]:
    """
    Selects 10% of items from the torso and tail (bottom 80% by train popularity) for true item holdout.
    """
    n_items = data.n_items
    n_held = int(round(fraction * n_items))
    sorted_pop = np.argsort(data.item_train_pop)  # ascending (tail to head)
    # Exclude top 20% head items: choose from bottom 80%
    torso_tail = sorted_pop[: int(0.80 * n_items)]

    rng = np.random.default_rng(seed)
    held_out_items = rng.choice(torso_tail, size=min(n_held, len(torso_tail)), replace=False)
    return held_out_items, set(int(it) for it in held_out_items)


def evaluate_item_holdout_study(
    data: RecsysData,
    models: dict[str, BaseRecommender],
    eval_users: np.ndarray | list[int],
    held_out_items: np.ndarray | set[int],
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Executes Part D: True Item Holdout Evaluation.
    Evaluates TEST users whose target item is a held-out item.

    Compares:
      - Two-Tower (retrained without held-out item train interactions, unknown ID used at inference)
      - Two-Tower No-ID (retrained without held-out item train interactions)
      - Content-Only (cosine similarity with history)
      - Random Scorer (HR@10 = 10 / n_items)
    """
    held_out_set = set(int(it) for it in held_out_items)
    held_out_list = [int(it) for it in held_out_items]

    eval_users_arr = np.asarray(eval_users, dtype=np.int64)
    test_target_map = dict(
        zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True)
    )

    holdout_users: list[int] = []
    holdout_targets: list[int] = []

    for u in eval_users_arr:
        t_item = int(test_target_map[int(u)])
        if t_item in held_out_set:
            holdout_users.append(int(u))
            holdout_targets.append(t_item)

    n_holdout = len(holdout_users)
    logger.info("True Item Holdout Study: %d held-out items, %d evaluated test users", len(held_out_set), n_holdout)

    random_hr = float(10.0 / data.n_items)
    ranks_10 = np.arange(1, 11)
    random_ndcg = float(np.sum(1.0 / np.log2(ranks_10 + 1.0)) / data.n_items)
    random_mrr = float(np.sum(1.0 / ranks_10) / data.n_items)

    results: dict[str, dict[str, float]] = {
        "random": {
            "hit_rate_at_10": random_hr,
            "hit_rate_at_10_lo95": random_hr,
            "hit_rate_at_10_hi95": random_hr,
            "ndcg_at_10": random_ndcg,
            "ndcg_at_10_lo95": random_ndcg,
            "ndcg_at_10_hi95": random_ndcg,
            "mrr_at_10": random_mrr,
        }
    }

    if n_holdout == 0:
        return {
            "study_type": "item_holdout",
            "n_eval_users": 0,
            "n_held_out_items": len(held_out_set),
            "models": results,
            "random_reference": {"hit_rate_at_10": random_hr, "formula": "10 / n_items"},
            "note": "No test users found with target in held-out set.",
        }

    histories, targets, seen_masks = _build_test_foldin_data(data, np.array(holdout_users, dtype=np.int64))
    targets_arr = np.asarray(targets, dtype=np.int64)

    # Content-Only
    if "content" in models:
        scores = models["content"].score(histories).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["content"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # Two-Tower (Holdout retrained, unknown ID for held-out items)
    if "two_tower" in models and isinstance(models["two_tower"], TwoTowerModel):
        tt_model = models["two_tower"]
        masked_vecs = tt_model.item_vectors(mask_ids=held_out_list, neutralize_proxies=held_out_list)
        scores = tt_model.score(
            histories,
            item_vecs=masked_vecs,
            mask_ids=held_out_list,
            neutralize_proxies=held_out_list,
        ).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["two_tower"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    # Two-Tower No-ID (Holdout retrained)
    if "two_tower_no_id" in models and isinstance(models["two_tower_no_id"], TwoTowerModel):
        noid_model = models["two_tower_no_id"]
        masked_vecs = noid_model.item_vectors(mask_ids=held_out_list, neutralize_proxies=held_out_list)
        scores = noid_model.score(
            histories,
            item_vecs=masked_vecs,
            mask_ids=held_out_list,
            neutralize_proxies=held_out_list,
        ).copy()
        for i, s_mask in enumerate(seen_masks):
            scores[i, list(s_mask)] = -np.inf
        ranks = compute_ranks(scores, targets_arr)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hit_rate_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_at_k(ranks, 10), n_boot=n_boot, seed=seed)
        mrr_val = float(np.mean(mrr_at_k(ranks, 10)))
        results["two_tower_no_id"] = {
            "hit_rate_at_10": hr_m,
            "hit_rate_at_10_lo95": hr_lo,
            "hit_rate_at_10_hi95": hr_hi,
            "ndcg_at_10": nd_m,
            "ndcg_at_10_lo95": nd_lo,
            "ndcg_at_10_hi95": nd_hi,
            "mrr_at_10": mrr_val,
        }

    return {
        "study_type": "item_holdout",
        "n_eval_users": n_holdout,
        "n_held_out_items": len(held_out_set),
        "models": results,
        "random_reference": {
            "hit_rate_at_10": random_hr,
            "ndcg_at_10": random_ndcg,
            "formula": "10 / n_items",
        },
        "note": "True Item Holdout: Held-out items had zero interactions during training.",
    }


def run_full_cold_start_study(
    data: RecsysData,
    models: dict[str, BaseRecommender],
    eval_users: np.ndarray | list[int],
    seed: int = 42,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """
    Runs:
      - Sparse-targets study (Part 1, with random reference)
      - Three Fair Simulations (C1 old setup, C2 fair random, C3 fair bottom 50%)
    """
    sparse_res = evaluate_sparse_targets_study(
        data=data,
        models=models,
        eval_users=eval_users,
        max_train_positives=10,
        seed=seed,
        n_boot=n_boot,
    )

    c1_res = evaluate_simulated_new_items_study(
        data=data,
        models=models,
        eval_users=eval_users,
        fraction=0.10,
        seed=seed,
        variant="c1",
        n_boot=n_boot,
    )

    c2_res = evaluate_simulated_new_items_study(
        data=data,
        models=models,
        eval_users=eval_users,
        fraction=0.10,
        seed=seed,
        variant="c2",
        n_boot=n_boot,
    )

    c3_res = evaluate_simulated_new_items_study(
        data=data,
        models=models,
        eval_users=eval_users,
        fraction=0.10,
        seed=seed,
        variant="c3",
        n_boot=n_boot,
    )

    return {
        "sparse_targets_study": sparse_res,
        "simulated_new_items_study": c2_res,  # Default fair simulation for schema
        "c1_old_setup": c1_res,
        "c2_fair_random": c2_res,
        "c3_fair_bottom50": c3_res,
    }
