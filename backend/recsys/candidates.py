"""
Multi-source candidate retrieval for two-stage recommender systems.

Retrieves top-N candidates per user across 4 models:
  - Two-Tower: top 150
  - Item-kNN: top 100
  - ALS: top 50
  - Content: top 50
Seen items are masked (-inf), and candidates are deduplicated and capped at 300 per user.
For each candidate, per-source scores, 1-indexed ranks (301 if absent), and presence flags are retained.

Rule 4: Zero FastAPI and zero database imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from recsys.models.base import BaseRecommender

logger = logging.getLogger("steamiq.recsys.candidates")

DEFAULT_CANDIDATE_COUNTS: dict[str, int] = {
    "two_tower": 150,
    "itemknn": 100,
    "als": 50,
    "content": 50,
}


@dataclass
class CandidateInfo:
    """Per-candidate source attribution for a single user."""
    item_idx: int
    tt_score: float = -1.0
    tt_rank: int = 301
    in_tt: bool = False
    itemknn_score: float = 0.0
    itemknn_rank: int = 301
    in_itemknn: bool = False
    als_score: float = 0.0
    als_rank: int = 301
    in_als: bool = False
    content_score: float = 0.0
    content_rank: int = 301
    in_content: bool = False


@dataclass
class CandidateResult:
    """Container holding multi-source candidate lists and source metadata for a batch of users."""
    user_candidates: list[list[int]] = field(default_factory=list)
    candidate_metadata: list[list[CandidateInfo]] = field(default_factory=list)
    source_top_items: dict[str, list[list[int]]] = field(default_factory=dict)
    n_users: int = 0
    max_candidates: int = 300


def generate_candidates(
    histories: list[np.ndarray],
    models: dict[str, BaseRecommender],
    n_each: dict[str, int] | None = None,
    seen_masks: list[set[int]] | list[list[int]] | None = None,
    max_candidates: int = 300,
    batch_size: int = 512,
) -> CandidateResult:
    """
    Generates multi-source candidate pools for a list of user histories.

    Args:
        histories: List of user fold-in history arrays (each array contains item_idx).
        models: Dictionary of models, keys in {"two_tower", "itemknn", "als", "content"}.
        n_each: Dictionary specifying top-N per source (defaults to 150 TT, 100 kNN, 50 ALS, 50 Content).
        seen_masks: Optional list of seen/consumed item sets per user to mask with -inf.
        max_candidates: Maximum deduplicated candidate union size per user (default: 300).
        batch_size: Batch size for scoring large user sets.

    Returns:
        CandidateResult containing candidate item lists and detailed source scores/ranks.
    """
    counts = dict(DEFAULT_CANDIDATE_COUNTS)
    if n_each is not None:
        counts.update(n_each)

    n_users = len(histories)
    if n_users == 0:
        return CandidateResult(n_users=0, max_candidates=max_candidates)

    # Standardize source model keys
    canonical_sources = ["two_tower", "itemknn", "als", "content"]
    active_models: dict[str, BaseRecommender] = {}
    for key, model in models.items():
        k_clean = key.lower().replace("-", "_").replace("recsys_", "")
        if "two_tower" in k_clean or "twotower" in k_clean:
            active_models["two_tower"] = model
        elif "itemknn" in k_clean or "knn" in k_clean:
            active_models["itemknn"] = model
        elif "als" in k_clean:
            active_models["als"] = model
        elif "content" in k_clean:
            active_models["content"] = model

    # Structure to hold per-source top items and scores for each user
    # source_user_items[source_name][user_idx] = (items_list, scores_list, min_score)
    source_results: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    source_top_items_dict: dict[str, list[list[int]]] = {s: [] for s in canonical_sources}

    for s_name in canonical_sources:
        model = active_models.get(s_name)
        k_req = counts.get(s_name, 50)
        source_user_tuples: list[tuple[np.ndarray, np.ndarray, float]] = []

        if model is None:
            for _ in range(n_users):
                source_user_tuples.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0.0))
                source_top_items_dict[s_name].append([])
            source_results[s_name] = source_user_tuples
            continue

        # Score in batches to avoid OOM
        all_top_items: list[list[int]] = []
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            b_histories = histories[start_idx:end_idx]

            raw_scores = model.score(b_histories)  # [B, n_items]
            n_items = raw_scores.shape[1]
            k_eff = min(k_req, n_items)

            # Apply seen item masks
            if seen_masks is not None:
                for b_i, global_i in enumerate(range(start_idx, end_idx)):
                    user_seen = seen_masks[global_i]
                    if len(user_seen) > 0:
                        valid_seen = [it for it in user_seen if 0 <= it < n_items]
                        if valid_seen:
                            raw_scores[b_i, valid_seen] = -np.inf

            # Extract top k items and scores per row
            for b_i in range(len(b_histories)):
                row_scores = raw_scores[b_i]
                # Partition top k
                if k_eff < n_items:
                    part_idx = np.argpartition(row_scores, -k_eff)[-k_eff:]
                    sorted_part = part_idx[np.argsort(-row_scores[part_idx])]
                else:
                    sorted_part = np.argsort(-row_scores)

                top_items = sorted_part[:k_eff]
                top_scores = row_scores[top_items]

                # If -inf was selected due to small catalog, filter out
                valid_mask = ~np.isneginf(top_scores)
                top_items = top_items[valid_mask]
                top_scores = top_scores[valid_mask]

                min_sc = float(np.min(top_scores)) if len(top_scores) > 0 else 0.0
                source_user_tuples.append((top_items, top_scores, min_sc))
                all_top_items.append(top_items.tolist())

        source_results[s_name] = source_user_tuples
        source_top_items_dict[s_name] = all_top_items

    # 2. Merge candidate pools into deduplicated unions per user (capped at max_candidates)
    merged_candidates: list[list[int]] = []
    merged_metadata: list[list[CandidateInfo]] = []

    for u_idx in range(n_users):
        u_cand_set: set[int] = set()
        u_cand_list: list[int] = []

        # Order of merging: two_tower -> itemknn -> als -> content
        for s_name in canonical_sources:
            items_arr, _, _ = source_results[s_name][u_idx]
            for item in items_arr:
                item_int = int(item)
                if item_int not in u_cand_set:
                    u_cand_set.add(item_int)
                    u_cand_list.append(item_int)
                    if len(u_cand_list) >= max_candidates:
                        break
            if len(u_cand_list) >= max_candidates:
                break

        # Build candidate info records for each union candidate
        u_cand_infos: list[CandidateInfo] = []

        # Fast lookup per source: item -> (score, rank)
        source_lookup: dict[str, dict[int, tuple[float, int]]] = {}
        source_min_scores: dict[str, float] = {}

        for s_name in canonical_sources:
            items_arr, scores_arr, min_sc = source_results[s_name][u_idx]
            source_min_scores[s_name] = min_sc
            source_lookup[s_name] = {}
            for rank_0, (it, sc) in enumerate(zip(items_arr, scores_arr, strict=True)):
                source_lookup[s_name][int(it)] = (float(sc), rank_0 + 1)

        for item_idx in u_cand_list:
            info = CandidateInfo(item_idx=item_idx)

            # Two-Tower
            if item_idx in source_lookup["two_tower"]:
                sc, rk = source_lookup["two_tower"][item_idx]
                info.tt_score = sc
                info.tt_rank = rk
                info.in_tt = True
            else:
                info.tt_score = source_min_scores["two_tower"]
                info.tt_rank = 301
                info.in_tt = False

            # Item-kNN
            if item_idx in source_lookup["itemknn"]:
                sc, rk = source_lookup["itemknn"][item_idx]
                info.itemknn_score = sc
                info.itemknn_rank = rk
                info.in_itemknn = True
            else:
                info.itemknn_score = source_min_scores["itemknn"]
                info.itemknn_rank = 301
                info.in_itemknn = False

            # ALS
            if item_idx in source_lookup["als"]:
                sc, rk = source_lookup["als"][item_idx]
                info.als_score = sc
                info.als_rank = rk
                info.in_als = True
            else:
                info.als_score = source_min_scores["als"]
                info.als_rank = 301
                info.in_als = False

            # Content
            if item_idx in source_lookup["content"]:
                sc, rk = source_lookup["content"][item_idx]
                info.content_score = sc
                info.content_rank = rk
                info.in_content = True
            else:
                info.content_score = source_min_scores["content"]
                info.content_rank = 301
                info.in_content = False

            u_cand_infos.append(info)

        merged_candidates.append(u_cand_list)
        merged_metadata.append(u_cand_infos)

    return CandidateResult(
        user_candidates=merged_candidates,
        candidate_metadata=merged_metadata,
        source_top_items=source_top_items_dict,
        n_users=n_users,
        max_candidates=max_candidates,
    )


def compute_candidate_recalls(
    candidate_result: CandidateResult,
    target_items: list[int] | np.ndarray,
) -> dict[str, float]:
    """
    Computes candidate recall per source and for the union.

    Args:
        candidate_result: CandidateResult from generate_candidates.
        target_items: List of ground-truth target item indices (one per user).

    Returns:
        Dictionary of recall metrics, e.g. candidate_recall_two_tower, candidate_recall_union.
    """
    targets = [int(t) for t in target_items]
    n = len(targets)
    if n == 0 or candidate_result.n_users == 0:
        return {
            "candidate_recall_two_tower": 0.0,
            "candidate_recall_itemknn": 0.0,
            "candidate_recall_als": 0.0,
            "candidate_recall_content": 0.0,
            "candidate_recall_union": 0.0,
        }

    hits_tt = 0
    hits_knn = 0
    hits_als = 0
    hits_content = 0
    hits_union = 0

    tt_items = candidate_result.source_top_items.get("two_tower", [])
    knn_items = candidate_result.source_top_items.get("itemknn", [])
    als_items = candidate_result.source_top_items.get("als", [])
    content_items = candidate_result.source_top_items.get("content", [])
    union_items = candidate_result.user_candidates

    for i, t in enumerate(targets):
        if i < len(tt_items) and t in tt_items[i]:
            hits_tt += 1
        if i < len(knn_items) and t in knn_items[i]:
            hits_knn += 1
        if i < len(als_items) and t in als_items[i]:
            hits_als += 1
        if i < len(content_items) and t in content_items[i]:
            hits_content += 1
        if i < len(union_items) and t in union_items[i]:
            hits_union += 1

    return {
        "candidate_recall_two_tower": hits_tt / n,
        "candidate_recall_itemknn": hits_knn / n,
        "candidate_recall_als": hits_als / n,
        "candidate_recall_content": hits_content / n,
        "candidate_recall_union": hits_union / n,
    }
