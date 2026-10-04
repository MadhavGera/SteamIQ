"""
Vectorized evaluation metrics for recommender systems in pure NumPy.

Assumes a leave-one-out evaluation setup where each evaluated user has exactly
ONE held-out target item in the given evaluation split.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def compute_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Computes 1-indexed ranks for target items with pessimistic tie handling.

    For user u and target item t, rank is defined as:
        rank = 1 + count_{i != t}(score_{u, i} >= score_{u, t})

    Vectorized implementation:
        target_scores = scores[u, target[u]]
        ranks = sum_i(score_{u, i} >= target_scores)

    When an item ties with other items, it is placed behind all of them.
    If all items in the catalog have identical scores, the rank is n_items (the worst possible rank).

    Args:
        scores: 2D float array of shape [n_users, n_items]
        targets: 1D int array of shape [n_users] containing item_idx for each user

    Returns:
        1D int64 array of shape [n_users] containing 1-indexed ranks.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be a 2D array [n_users, n_items], got ndim={scores.ndim}")
    if targets.ndim != 1 or len(targets) != scores.shape[0]:
        raise ValueError(
            f"targets must be 1D array with length matching scores rows ({scores.shape[0]}), got len={len(targets)}"
        )

    n_users = scores.shape[0]
    if n_users == 0:
        return np.empty((0,), dtype=np.int64)

    target_scores = scores[np.arange(n_users), targets, None]  # shape [n_users, 1]
    # In float comparison, target item itself satisfies score[u, t] >= target_scores[u] (counts as 1)
    # Plus every non-target item with score >= target score counts as 1.
    # Total sum equals 1 + sum_{i != t} I(score_{u, i} >= score_{u, t}).
    ranks = np.sum(scores >= target_scores, axis=1).astype(np.int64)
    return ranks


def hit_rate_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """
    Computes per-user Hit Rate @ k (binary indicator whether rank <= k).

    Args:
        ranks: 1D array of 1-indexed ranks.
        k: Cutoff threshold.

    Returns:
        1D float64 array of 0.0 or 1.0 values.
    """
    return (ranks <= k).astype(np.float64)


def ndcg_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """
    Computes per-user Normalized Discounted Cumulative Gain @ k for single-target evaluation.

    Since there is exactly 1 relevant item, ideal DCG (IDCG) is 1.0 (at rank 1, 1/log2(1+1) = 1.0).
    NDCG@k = 1 / log2(rank + 1) if rank <= k else 0.0.

    Args:
        ranks: 1D array of 1-indexed ranks.
        k: Cutoff threshold.

    Returns:
        1D float64 array of NDCG values.
    """
    in_k = ranks <= k
    result = np.zeros(len(ranks), dtype=np.float64)
    if np.any(in_k):
        result[in_k] = 1.0 / np.log2(ranks[in_k] + 1.0)
    return result


def mrr_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """
    Computes per-user Mean Reciprocal Rank @ k (reciprocal rank if rank <= k else 0.0).

    MRR@k = 1 / rank if rank <= k else 0.0.

    Args:
        ranks: 1D array of 1-indexed ranks.
        k: Cutoff threshold.

    Returns:
        1D float64 array of reciprocal rank values.
    """
    in_k = ranks <= k
    result = np.zeros(len(ranks), dtype=np.float64)
    if np.any(in_k):
        result[in_k] = 1.0 / ranks[in_k].astype(np.float64)
    return result


def candidate_recall_at_n(ranks: np.ndarray, n: int) -> np.ndarray:
    """
    Computes per-user recall at candidate depth n (indicator whether target is in top n).
    """
    return hit_rate_at_k(ranks, n)


def get_top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Extracts sorted top-k item indices for each user.

    Args:
        scores: 2D array [n_users, n_items]
        k: Number of recommendations per user.

    Returns:
        2D int64 array [n_users, k] of item indices in descending score order.
    """
    n_users, n_items = scores.shape
    if n_users == 0:
        return np.empty((0, k), dtype=np.int64)

    k = min(k, n_items)
    # Partition top k
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    # Sort partitioned top k
    row_idx = np.arange(n_users)[:, None]
    sub_scores = scores[row_idx, idx]
    sub_sort = np.argsort(-sub_scores, axis=1)
    top_k = np.take_along_axis(idx, sub_sort, axis=1)
    return top_k.astype(np.int64)


def catalog_coverage_at_k(top_k: np.ndarray, n_items: int) -> float:
    """
    Fraction of unique items appearing in all top-k recommendation lists combined.

    Coverage = len(unique(top_k)) / n_items

    Args:
        top_k: 2D array [n_users, k] of recommended item indices.
        n_items: Total number of items in the catalog.

    Returns:
        Float in [0.0, 1.0].
    """
    if n_items <= 0:
        return 0.0
    if top_k.size == 0:
        return 0.0
    unique_items = np.unique(top_k)
    return float(len(unique_items) / n_items)


def novelty_at_k(
    top_k: np.ndarray,
    item_train_pop: np.ndarray,
    total_train_positives: float | None = None,
) -> float:
    """
    Computes catalog novelty (mean self-information of recommended items).

    Novelty = mean_{u, j} -log2( P(item_j) )
    where P(item_j) = count(item_j) / total_train_positives.

    Args:
        top_k: 2D array [n_users, k] of recommended item indices.
        item_train_pop: 1D array of shape [n_items] with interaction counts.
        total_train_positives: Total number of positive training interactions.

    Returns:
        Float self-information in bits.
    """
    if top_k.size == 0:
        return 0.0

    if total_train_positives is None or total_train_positives <= 0:
        total_train_positives = float(np.sum(item_train_pop))
    if total_train_positives <= 0:
        return 0.0

    # Minimum positive count of 1.0 to prevent division by zero / log2(0)
    pop_matrix = item_train_pop[top_k].astype(np.float64)
    probs = np.maximum(pop_matrix, 1.0) / total_train_positives
    self_info = -np.log2(probs)
    return float(np.mean(self_info))


def diversity_at_k(
    top_k: np.ndarray,
    item_embeddings: np.ndarray | sp.csr_matrix,
) -> float:
    """
    Computes intra-list diversity: 1.0 - mean pairwise cosine similarity among top-k items.

    Args:
        top_k: 2D array [n_users, k] of recommended item indices.
        item_embeddings: Dense [n_items, dim] or sparse [n_items, vocab_size] item features.

    Returns:
        Float diversity in [0.0, 2.0] (typically [0.0, 1.0]).
    """
    n_users, k = top_k.shape
    if n_users == 0 or k <= 1:
        return 0.0

    if sp.issparse(item_embeddings):
        # Convert sparse rows to dense or compute sparse dot products
        # For evaluation users, converting the slice of used items to dense is fast
        unique_items = np.unique(top_k)
        dense_subset = item_embeddings[unique_items].toarray().astype(np.float32)
        # Map item_idx to index in dense_subset
        item_to_dense = {item: idx for idx, item in enumerate(unique_items)}
        flat_mapped = np.vectorize(item_to_dense.get)(top_k)
        user_embs = dense_subset[flat_mapped]  # [n_users, k, dim]
    else:
        user_embs = item_embeddings[top_k].astype(np.float32)  # [n_users, k, dim]

    # L2 normalize each item embedding vector along the feature dimension
    norms = np.linalg.norm(user_embs, axis=2, keepdims=True)
    user_embs = user_embs / np.maximum(norms, 1e-12)

    # Compute pairwise cosine similarity matrix for each user: [n_users, k, k]
    # user_embs[u, i] . user_embs[u, j]
    sims = np.einsum("ukd,umd->ukm", user_embs, user_embs)

    # Sum off-diagonal similarities: total sum minus diagonal (which is k * 1.0)
    # Number of off-diagonal pairs is k * (k - 1)
    off_diag_sum = np.sum(sims, axis=(1, 2)) - float(k)
    user_mean_sim = off_diag_sum / float(k * (k - 1))
    mean_sim = np.mean(user_mean_sim)

    return float(1.0 - mean_sim)


def head_share_at_k(
    top_k: np.ndarray,
    item_train_pop: np.ndarray,
    head_fraction: float = 0.01,
) -> float:
    """
    Computes the share of recommendation slots occupied by the top head_fraction items.

    Args:
        top_k: 2D array [n_users, k] of recommended item indices.
        item_train_pop: 1D array [n_items] of popularity counts.
        head_fraction: Fraction of top items considered "head" (default: 0.01 = 1%).

    Returns:
        Float share in [0.0, 1.0].
    """
    if top_k.size == 0 or len(item_train_pop) == 0:
        return 0.0

    n_items = len(item_train_pop)
    n_head = max(1, int(np.ceil(n_items * head_fraction)))
    head_item_indices = np.argsort(-item_train_pop)[:n_head]

    is_head = np.zeros(n_items, dtype=bool)
    is_head[head_item_indices] = True

    head_slots = np.sum(is_head[top_k])
    total_slots = top_k.size
    return float(head_slots / total_slots)


def bootstrap_ci(
    per_user_values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes 95% bootstrap confidence interval for a 1D array of per-user metric values.

    Args:
        per_user_values: 1D array of metric values across users.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        (mean, lo95, hi95)
    """
    if len(per_user_values) == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(per_user_values))
    if len(per_user_values) == 1:
        return mean_val, mean_val, mean_val

    n = len(per_user_values)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(per_user_values[boot_indices], axis=1)

    lo95 = float(np.percentile(boot_means, 2.5))
    hi95 = float(np.percentile(boot_means, 97.5))
    return mean_val, lo95, hi95


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes 95% bootstrap confidence interval on the difference (a - b) between two models.

    Args:
        a: 1D array of metric values from model A across users.
        b: 1D array of metric values from model B across identical users.
        n_boot: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        (mean_diff, lo95, hi95)
    """
    if len(a) != len(b):
        raise ValueError(f"Arrays must have matching lengths, got {len(a)} and {len(b)}")
    if len(a) == 0:
        return 0.0, 0.0, 0.0

    diff = (a - b).astype(np.float64)
    mean_diff = float(np.mean(diff))
    if len(diff) == 1:
        return mean_diff, mean_diff, mean_diff

    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = np.mean(diff[boot_indices], axis=1)

    lo95 = float(np.percentile(boot_diffs, 2.5))
    hi95 = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, lo95, hi95


def gini_coefficient_at_k(top_k: np.ndarray, n_items: int) -> float:
    """
    Computes the Gini coefficient of item exposure across all users' top-k recommendation lists.

    Exposure counts f_i are the number of times item i appears across all top-k lists.
    Gini coefficient:
        G = (2 * sum_{i=1}^M i * f_{(i)} - (M + 1) * sum_{i=1}^M f_{(i)}) / (M * sum_{i=1}^M f_{(i)})
    where f is sorted in ascending order and M = n_items.
    - Returns 0.0 for perfectly uniform exposure (all items exposed equally).
    - Returns (n_items - 1) / n_items (~1.0) if a single item receives all recommendation slots.

    Args:
        top_k: 2D array [n_users, k] of recommended item indices.
        n_items: Total number of items in catalog.

    Returns:
        Float Gini coefficient in [0.0, 1.0].
    """
    if n_items <= 0 or top_k.size == 0:
        return 0.0

    flat_items = top_k.flatten()
    valid_items = flat_items[(flat_items >= 0) & (flat_items < n_items)]
    if len(valid_items) == 0:
        return 0.0

    counts = np.bincount(valid_items, minlength=n_items).astype(np.float64)
    total_exposure = np.sum(counts)
    if total_exposure == 0:
        return 0.0

    sorted_counts = np.sort(counts)
    m = float(n_items)
    idx = np.arange(1, n_items + 1, dtype=np.float64)

    numerator = 2.0 * np.sum(idx * sorted_counts) - (m + 1.0) * total_exposure
    denominator = m * total_exposure
    gini = float(numerator / denominator)
    return float(max(0.0, min(1.0, gini)))


def mmr_rerank(
    candidate_items: np.ndarray,
    candidate_scores: np.ndarray,
    item_embeddings: np.ndarray | sp.csr_matrix,
    lambda_param: float = 0.8,
    k: int = 10,
) -> np.ndarray:
    """
    Applies Maximal Marginal Relevance (MMR) re-ranking over candidate lists per user.

    MMR Selection Objective:
        argmax_{i in C \\ S} [ lambda * norm_score(i) - (1 - lambda) * max_{j in S} CosineSim(i, j) ]

    When lambda_param == 1.0, MMR produces the exact same top-k order as sorting candidate_scores descending.

    Args:
        candidate_items: 2D array [n_users, n_candidates] (or 1D [n_candidates]) of item indices.
        candidate_scores: 2D array [n_users, n_candidates] (or 1D [n_candidates]) of base relevance scores.
        item_embeddings: [n_items, dim] dense embedding array or sparse CSR matrix.
        lambda_param: Float trade-off parameter in [0.0, 1.0] (default: 0.8).
        k: Output recommendation list length (default: 10).

    Returns:
        2D int64 array [n_users, k] (or 1D [k] if 1D input) of selected item indices.
    """
    is_1d = candidate_items.ndim == 1
    c_items = np.atleast_2d(candidate_items).astype(np.int64)
    c_scores = np.atleast_2d(candidate_scores).astype(np.float32)

    n_users, n_cands = c_items.shape
    k = min(k, n_cands)

    if n_users == 0 or n_cands == 0 or k <= 0:
        empty_res = np.empty((n_users, k), dtype=np.int64)
        return empty_res[0] if is_1d else empty_res

    # Pre-extract and normalize embeddings for unique candidates
    unique_items = np.unique(c_items)
    valid_items = unique_items[(unique_items >= 0) & (unique_items < (item_embeddings.shape[0] if hasattr(item_embeddings, 'shape') else len(item_embeddings)))]

    if sp.issparse(item_embeddings):
        emb_dense = item_embeddings[valid_items].toarray().astype(np.float32)
    else:
        emb_dense = np.asarray(item_embeddings[valid_items], dtype=np.float32)

    norms = np.linalg.norm(emb_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb_dense / norms

    item_to_norm_idx = {int(it): idx for idx, it in enumerate(valid_items)}
    dim = emb_norm.shape[1] if len(emb_norm) > 0 else 1

    results = np.empty((n_users, k), dtype=np.int64)

    for u_idx in range(n_users):
        user_cands = c_items[u_idx]
        user_scores = c_scores[u_idx]

        # Initial sort by score descending
        initial_order = np.argsort(-user_scores)
        sorted_cands = user_cands[initial_order]
        sorted_scores = user_scores[initial_order]

        if lambda_param >= 1.0 or k == 1:
            results[u_idx] = sorted_cands[:k]
            continue

        # Score normalization to [0, 1]
        s_min = float(np.min(sorted_scores))
        s_max = float(np.max(sorted_scores))
        if s_max > s_min:
            norm_scores = (sorted_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(sorted_scores, dtype=np.float32)

        # Get unit-norm vectors for candidate items
        cand_vecs = np.zeros((n_cands, dim), dtype=np.float32)
        for i_c, it in enumerate(sorted_cands):
            if int(it) in item_to_norm_idx:
                cand_vecs[i_c] = emb_norm[item_to_norm_idx[int(it)]]

        # Precompute candidate-to-candidate cosine similarities: [n_cands, n_cands]
        cand_sims = cand_vecs @ cand_vecs.T

        selected: list[int] = [0]  # Start with the highest-scoring candidate
        available = set(range(1, n_cands))

        for _ in range(1, k):
            if not available:
                break
            avail_list = list(available)
            avail_arr = np.array(avail_list, dtype=np.int64)

            # Max similarity to already selected items for each available candidate
            sim_to_selected = np.max(cand_sims[avail_arr[:, None], selected], axis=1)

            # MMR score = lambda * relevance - (1 - lambda) * max_sim
            mmr_scores = lambda_param * norm_scores[avail_arr] - (1.0 - lambda_param) * sim_to_selected

            best_idx = int(np.argmax(mmr_scores))
            chosen = avail_list[best_idx]
            selected.append(chosen)
            available.remove(chosen)

        results[u_idx] = sorted_cands[selected[:k]]

    return results[0] if is_1d else results

