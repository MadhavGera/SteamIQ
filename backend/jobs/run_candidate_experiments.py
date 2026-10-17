"""
Validation Retrieval Experiment Script for Multi-Source Candidate Generation.
Phase 2 Evidence Hardening.

Compares merge strategies ('quota', 'interleave', 'legacy_ordered') on the validation split:
  - Union recall@K for K in {50, 100, 200, 300}
  - Per-source recall
  - Share of candidate pool contributed by each source
  - Sliced recall by item popularity (head, torso, tail) and history length bucket
  - Decides and validates the default candidate strategy
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from recsys.candidates import (
    DEFAULT_CANDIDATE_COUNTS,
    DEFAULT_QUOTAS,
    CandidateResult,
    generate_candidates,
)
from recsys.data import RecsysData
from recsys.evaluate import build_user_evaluation_inputs
from recsys.models.als import ALSRecommender
from recsys.models.base import BaseRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.two_tower import TwoTowerModel

logger = logging.getLogger("steamiq.jobs.run_candidate_experiments")


def _get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _load_stage1_models(data_dir: Path, data: RecsysData) -> dict[str, BaseRecommender]:
    """Loads existing Stage-1 retrieval models."""
    art_dir = data_dir / "artifacts"
    models: dict[str, BaseRecommender] = {}

    # Two-Tower
    tt_dir = art_dir / "recsys_two_tower_clean"
    if not tt_dir.exists():
        tt_dir = art_dir / "recsys_two_tower"
    if tt_dir.exists():
        cand_runs = [d for d in tt_dir.iterdir() if d.is_dir()]
        if cand_runs:
            latest = sorted(cand_runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            models["two_tower"] = TwoTowerModel.load(latest, data=data)

    # ItemKNN
    knn_dir = art_dir / "recsys_itemknn"
    if knn_dir.exists():
        cand_runs = [d for d in knn_dir.iterdir() if d.is_dir()]
        if cand_runs:
            latest = sorted(cand_runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            models["itemknn"] = ItemKNNRecommender.load(latest)

    # ALS
    als_dir = art_dir / "recsys_als"
    if als_dir.exists():
        cand_runs = [d for d in als_dir.iterdir() if d.is_dir()]
        if cand_runs:
            latest = sorted(cand_runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            models["als"] = ALSRecommender.load(latest)

    # Content
    content_dir = art_dir / "recsys_content"
    if content_dir.exists():
        cand_runs = [d for d in content_dir.iterdir() if d.is_dir()]
        if cand_runs:
            latest = sorted(cand_runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            models["content"] = ContentRecommender.load(latest)

    return models


def run_candidate_retrieval_experiments(
    data_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    max_eval_users: int = 5000,
) -> dict[str, Any]:
    resolved_dir = Path(data_dir) if data_dir else Path(settings.resolved_recsys_data_dir)
    data = RecsysData.load(resolved_dir)

    eval_meta_file = resolved_dir / "eval_users.json"
    if eval_meta_file.exists():
        eval_raw = json.load(open(eval_meta_file, encoding="utf-8"))
        eval_users = eval_raw["eval_users"] if isinstance(eval_raw, dict) else eval_raw
    else:
        eval_users = np.unique(data.val_df["user_idx"].to_numpy())

    if len(eval_users) > max_eval_users:
        eval_users = eval_users[:max_eval_users]

    print(f"Running candidate retrieval experiments on {len(eval_users)} validation users...")
    models = _load_stage1_models(resolved_dir, data)
    print(f"Loaded Stage-1 models: {list(models.keys())}")

    histories, targets, seen_masks, history_lengths = build_user_evaluation_inputs(
        data=data,
        eval_users=eval_users,
        split="val",
        max_history=50,
    )
    n_users = len(eval_users)
    targets_arr = np.array(targets, dtype=np.int64)
    hist_lens_arr = np.array(history_lengths, dtype=np.int32)

    # Define popularity buckets
    n_items = data.n_items
    sorted_pop_items = np.argsort(-data.item_train_pop)
    head_cutoff = int(0.20 * n_items)
    torso_cutoff = int(0.50 * n_items)
    head_set = set(sorted_pop_items[:head_cutoff])
    torso_set = set(sorted_pop_items[head_cutoff:torso_cutoff])

    pop_masks = {
        "head": np.array([t in head_set for t in targets_arr], dtype=bool),
        "torso": np.array([t in torso_set for t in targets_arr], dtype=bool),
        "tail": np.array([t not in head_set and t not in torso_set for t in targets_arr], dtype=bool),
    }

    hist_masks = {
        "hist_3_4": hist_lens_arr <= 4,
        "hist_5_9": (hist_lens_arr >= 5) & (hist_lens_arr <= 9),
        "hist_10_plus": hist_lens_arr >= 10,
    }

    strategies = ["quota", "interleave", "legacy_ordered"]
    k_cutoffs = [50, 100, 200, 300]
    results_by_strategy: dict[str, Any] = {}

    for strat in strategies:
        print(f"\nEvaluating strategy '{strat}'...")
        cand_res = generate_candidates(
            histories=histories,
            models=models,
            strategy=strat,
            seen_masks=seen_masks,
            max_candidates=300,
            batch_size=512,
        )

        strat_stats: dict[str, Any] = {}

        # 1. Union Recall@K and sliced recall
        union_recall_at_k: dict[str, float] = {}
        sliced_recall_at_k: dict[str, dict[str, float]] = {f"k_{k}": {} for k in k_cutoffs}

        for k in k_cutoffs:
            hits = np.zeros(n_users, dtype=bool)
            for u_i, (u_cands, t_item) in enumerate(zip(cand_res.user_candidates, targets_arr, strict=True)):
                if t_item in u_cands[:k]:
                    hits[u_i] = True

            union_recall_at_k[f"recall_at_{k}"] = float(np.mean(hits))

            # Popularity slices
            for pop_key, p_mask in pop_masks.items():
                if p_mask.any():
                    sliced_recall_at_k[f"k_{k}"][f"pop_{pop_key}"] = float(np.mean(hits[p_mask]))
                else:
                    sliced_recall_at_k[f"k_{k}"][f"pop_{pop_key}"] = 0.0

            # History length slices
            for h_key, h_mask in hist_masks.items():
                if h_mask.any():
                    sliced_recall_at_k[f"k_{k}"][h_key] = float(np.mean(hits[h_mask]))
                else:
                    sliced_recall_at_k[f"k_{k}"][h_key] = 0.0

        strat_stats["union_recall_at_k"] = union_recall_at_k
        strat_stats["sliced_recall_at_k"] = sliced_recall_at_k

        # 2. Per-source individual recall
        per_source_recall: dict[str, float] = {}
        for s_name in ["two_tower", "itemknn", "als", "content"]:
            s_items_list = cand_res.source_top_items.get(s_name, [])
            s_hits = 0
            for u_i, t_item in enumerate(targets_arr):
                if u_i < len(s_items_list) and t_item in s_items_list[u_i]:
                    s_hits += 1
            per_source_recall[f"recall_{s_name}"] = float(s_hits / max(1, n_users))
        strat_stats["per_source_recall"] = per_source_recall

        # 3. Share of candidate pool contributed by each source
        source_presence_counts: dict[str, int] = {s: 0 for s in ["two_tower", "itemknn", "als", "content"]}
        total_slots = 0
        for u_meta in cand_res.candidate_metadata:
            total_slots += len(u_meta)
            for info in u_meta:
                if info.in_tt:
                    source_presence_counts["two_tower"] += 1
                if info.in_itemknn:
                    source_presence_counts["itemknn"] += 1
                if info.in_als:
                    source_presence_counts["als"] += 1
                if info.in_content:
                    source_presence_counts["content"] += 1

        pool_shares: dict[str, float] = {}
        for s_name, count in source_presence_counts.items():
            pool_shares[s_name] = float(count / max(1, total_slots))
        strat_stats["source_pool_share"] = pool_shares

        results_by_strategy[strat] = strat_stats
        print(f"Strategy '{strat}': Union Recall@300 = {union_recall_at_k['recall_at_300']:.4f}, "
              f"Tail Recall@300 = {sliced_recall_at_k['k_300']['pop_tail']:.4f}, "
              f"Head Recall@300 = {sliced_recall_at_k['k_300']['pop_head']:.4f}")

    # Decision logic
    legacy_union = results_by_strategy["legacy_ordered"]["union_recall_at_k"]["recall_at_300"]
    legacy_head = results_by_strategy["legacy_ordered"]["sliced_recall_at_k"]["k_300"]["pop_head"]
    quota_union = results_by_strategy["quota"]["union_recall_at_k"]["recall_at_300"]
    quota_head = results_by_strategy["quota"]["sliced_recall_at_k"]["k_300"]["pop_head"]
    quota_tail = results_by_strategy["quota"]["sliced_recall_at_k"]["k_300"]["pop_tail"]
    legacy_tail = results_by_strategy["legacy_ordered"]["sliced_recall_at_k"]["k_300"]["pop_tail"]

    head_drop = legacy_head - quota_head
    union_gain = quota_union - legacy_union

    promote_quota = (union_gain >= 0.0) and (head_drop <= 0.01)

    decision = {
        "chosen_default_strategy": "quota" if promote_quota else "legacy_ordered",
        "quota_union_recall_at_300": round(quota_union, 5),
        "legacy_union_recall_at_300": round(legacy_union, 5),
        "quota_tail_recall_at_300": round(quota_tail, 5),
        "legacy_tail_recall_at_300": round(legacy_tail, 5),
        "quota_head_recall_at_300": round(quota_head, 5),
        "legacy_head_recall_at_300": round(legacy_head, 5),
        "union_gain": round(union_gain, 5),
        "head_drop": round(head_drop, 5),
        "meets_criteria": promote_quota,
        "rationale": (
            f"Quota strategy yields Union Recall@300 of {quota_union:.4f} (vs legacy {legacy_union:.4f}, "
            f"gain: {union_gain:+.4f}) with Tail Recall@300 of {quota_tail:.4f} (vs legacy {legacy_tail:.4f}). "
            f"Head recall difference is {head_drop:+.4f} (threshold <= 0.01)."
        ),
    }

    full_payload = {
        "metadata": {
            "report_key": "candidate_recall",
            "dataset_version": data.dataset_version,
            "git_commit": _get_git_commit(),
            "n_eval_users": n_users,
            "seed": 42,
            "quotas": DEFAULT_QUOTAS,
            "source_counts": DEFAULT_CANDIDATE_COUNTS,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "strategies": results_by_strategy,
        "decision": decision,
    }

    out_file = (
        Path(output_path)
        if output_path
        else Path(__file__).resolve().parent.parent.parent / "docs" / "results" / "candidate_recall.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\nSaved candidate recall results to {out_file}")

    return full_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run candidate retrieval experiments")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-users", type=int, default=5000)
    args = parser.parse_args()

    run_candidate_retrieval_experiments(
        data_dir=args.data_dir,
        output_path=args.output,
        max_eval_users=args.max_users,
    )
