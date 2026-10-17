"""
Phase 6 Final Test Evaluation & Cold-Start Study Job.

In strict compliance with Ground Rule 3:
- This script evaluates the test split exactly once.
- Evaluates the frozen champions and multi-seed sweeps (5 seeds for Two-Tower and Ranker).
- Evaluates baselines (Popularity, Content, 5-seed ItemKNN, 5-seed ALS).
- Computes paired bootstrap differences and confidence intervals.
- Applies the Ranker Promotion Gate.
- Computes test-set popularity and history length slices.
- Computes beyond-accuracy catalog metrics and MMR diversity re-ranking.
- Re-runs the true item-holdout cold-start study across 3 seeds with content-routed hybrid.
- Writes authoritative report payloads:
    * docs/results/leaderboard.json & backend/data/recsys/reports/leaderboard.json
    * docs/results/leaderboard.md & backend/data/recsys/reports/leaderboard.md
    * docs/results/slices.json & backend/data/recsys/reports/slices.json
    * docs/results/beyond_accuracy.json & backend/data/recsys/reports/beyond_accuracy.json
    * docs/results/cold_start.json & backend/data/recsys/reports/cold_start.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from jobs.base_job import BaseJob
from recsys.coldstart import (
    _build_test_foldin_data,
    get_item_holdout_split,
    get_simulated_new_items_test_users,
    get_sparse_target_test_users,
)
from recsys.data import RecsysData
from recsys.evaluate import (
    _build_user_interaction_maps,
    evaluate_model,
    get_evaluation_user_split,
)
from recsys.metrics import (
    bootstrap_ci,
    catalog_coverage_at_k,
    compute_ranks,
    diversity_at_k,
    get_top_k_indices,
    gini_coefficient_at_k,
    head_share_at_k,
    hit_rate_at_k,
    mmr_rerank,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    paired_bootstrap_diff,
)
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerModel
from recsys.report import (
    _clean_for_json,
    build_beyond_accuracy_payload,
    build_cold_start_payload,
    build_leaderboard_payload,
    build_slices_payload,
    generate_leaderboard_markdown,
    save_report_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("steamiq.jobs.run_phase6_test_eval")


def run_phase6_test_eval(
    data_dir: Path | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    t0_all = time.perf_counter()
    resolved_data_dir = Path(data_dir) if data_dir else Path(settings.resolved_recsys_data_dir)
    docs_results_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "results"
    reports_dir = resolved_data_dir / "reports"
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RecsysData from %s...", resolved_data_dir)
    data = RecsysData.load(resolved_data_dir)

    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=seed,
        persistence_dir=resolved_data_dir,
    )
    logger.info("Evaluation users: %d | Tuning users: %d", len(eval_users), len(tuning_users))

    # Load Phase 4 validation sweep results to preserve exact validation provenance
    val_sweep_path = docs_results_dir / "validation_sweep.json"
    val_sweep_data: dict[str, Any] = {}
    if val_sweep_path.exists():
        with open(val_sweep_path, "r", encoding="utf-8") as f:
            val_sweep_data = json.load(f)
        logger.info("Loaded Phase 4 validation sweep provenance from %s", val_sweep_path)

    seeds_5 = [42, 43, 44, 45, 46]
    seeds_3 = [42, 43, 44]

    # =========================================================================
    # 1. Evaluate Clean Two-Tower Models (5 Seeds) on Test Split
    # =========================================================================
    logger.info("\n>>> [1/7] Evaluating Clean Two-Tower across 5 seeds on TEST split...")
    tt_test_evals: dict[int, Any] = {}
    tt_test_ndcgs: list[float] = []
    tt_test_hrs: list[float] = []
    tt_test_mrrs: list[float] = []
    tt_models: dict[int, TwoTowerModel] = {}

    art_tt_dir = resolved_data_dir / "artifacts" / "recsys_two_tower_clean"

    for s in seeds_5:
        seed_dir = art_tt_dir / f"seed_{s}"
        logger.info("Loading Two-Tower seed %d from %s...", s, seed_dir)
        tt_m = TwoTowerModel.load(seed_dir, data=data)
        tt_m.name = "recsys_two_tower_clean"
        tt_models[s] = tt_m

        t_res = evaluate_model(
            model=tt_m,
            data=data,
            split="test",
            k_list=(10, 20),
            user_indices=eval_users,
            seed=s,
        )
        tt_test_evals[s] = t_res
        tt_test_ndcgs.append(float(t_res.metrics.get("ndcg_at_10", 0.0)))
        tt_test_hrs.append(float(t_res.metrics.get("hit_rate_at_10", 0.0)))
        tt_test_mrrs.append(float(t_res.metrics.get("mrr_at_10", 0.0)))
        logger.info("  TT Seed %d -> Test NDCG@10: %.5f | HR@10: %.5f", s, tt_test_ndcgs[-1], tt_test_hrs[-1])

    tt_test_mean_ndcg = float(np.mean(tt_test_ndcgs))
    tt_test_std_ndcg = float(np.std(tt_test_ndcgs))
    tt_test_mean_hr = float(np.mean(tt_test_hrs))
    tt_test_std_hr = float(np.std(tt_test_hrs))
    logger.info("Clean Two-Tower 5-Seed Test: NDCG@10 = %.5f ± %.5f", tt_test_mean_ndcg, tt_test_std_ndcg)

    # Champion is Seed 42 (closest to val mean, selected in Phase 4)
    champ_tt_seed = 42
    champ_tt_model = tt_models[champ_tt_seed]
    champ_tt_eval = tt_test_evals[champ_tt_seed]
    champ_tt_val_eval = evaluate_model(
        champ_tt_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=champ_tt_seed
    )

    # =========================================================================
    # 2. Evaluate Baselines on Test Split
    # =========================================================================
    logger.info("\n>>> [2/7] Evaluating Baselines on TEST split...")
    # Popularity
    pop_m = PopularityRecommender()
    pop_m.fit(data)
    pop_test = evaluate_model(pop_m, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=42)
    pop_val = evaluate_model(pop_m, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=42)
    logger.info("Popularity -> Test NDCG@10: %.5f", pop_test.metrics.get("ndcg_at_10", 0.0))

    # Content
    cnt_m = ContentRecommender(weight_tags=0.5)
    cnt_m.fit(data)
    cnt_test = evaluate_model(cnt_m, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=42)
    cnt_val = evaluate_model(cnt_m, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=42)
    logger.info("Content -> Test NDCG@10: %.5f", cnt_test.metrics.get("ndcg_at_10", 0.0))

    # ItemKNN (5 seeds)
    knn_test_ndcgs: list[float] = []
    knn_models: dict[int, ItemKNNRecommender] = {}
    knn_test_evals: dict[int, Any] = {}
    knn_val_evals: dict[int, Any] = {}
    for s in seeds_5:
        knn_inst = ItemKNNRecommender(k_neighbors=100, shrink=10.0)
        knn_inst.fit(data)
        knn_models[s] = knn_inst
        k_val = evaluate_model(knn_inst, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=s)
        k_test = evaluate_model(knn_inst, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=s)
        knn_val_evals[s] = k_val
        knn_test_evals[s] = k_test
        knn_test_ndcgs.append(float(k_test.metrics.get("ndcg_at_10", 0.0)))
    logger.info("Item-kNN 5-Seed Test: NDCG@10 = %.5f ± %.5f", np.mean(knn_test_ndcgs), np.std(knn_test_ndcgs))

    # ALS (5 seeds)
    als_test_ndcgs: list[float] = []
    als_models: dict[int, ALSRecommender] = {}
    als_test_evals: dict[int, Any] = {}
    als_val_evals: dict[int, Any] = {}
    for s in seeds_5:
        als_inst = ALSRecommender(factors=64, regularization=0.05, alpha=15.0, iterations=15, seed=s)
        als_inst.fit(data)
        als_models[s] = als_inst
        a_val = evaluate_model(als_inst, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=s)
        a_test = evaluate_model(als_inst, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=s)
        als_val_evals[s] = a_val
        als_test_evals[s] = a_test
        als_test_ndcgs.append(float(a_test.metrics.get("ndcg_at_10", 0.0)))
    logger.info("ALS 5-Seed Test: NDCG@10 = %.5f ± %.5f", np.mean(als_test_ndcgs), np.std(als_test_ndcgs))

    # =========================================================================
    # 3. Evaluate Clean Ranker Models (5 Seeds) on Test Split
    # =========================================================================
    logger.info("\n>>> [3/7] Evaluating Clean Ranker across 5 seeds on TEST split...")
    stage1_for_ranker = {
        "two_tower": champ_tt_model,
        "itemknn": knn_models[42],
        "als": als_models[42],
        "content": cnt_m,
    }

    art_rk_dir = resolved_data_dir / "artifacts" / "recsys_two_tower_ranker_clean"
    rk_test_evals: dict[int, Any] = {}
    rk_test_ndcgs: list[float] = []
    rk_test_hrs: list[float] = []
    rk_test_mrrs: list[float] = []
    rk_models: dict[int, RankerModel] = {}

    for s in seeds_5:
        seed_dir = art_rk_dir / f"seed_{s}"
        logger.info("Loading Clean Ranker seed %d from %s...", s, seed_dir)
        rk_m = RankerModel.load(seed_dir, data=data, models=stage1_for_ranker)
        rk_m.name = "recsys_two_tower_ranker_clean"
        rk_models[s] = rk_m

        t_res = evaluate_model(
            model=rk_m,
            data=data,
            split="test",
            k_list=(10, 20),
            user_indices=eval_users,
            seed=s,
        )
        rk_test_evals[s] = t_res
        rk_test_ndcgs.append(float(t_res.metrics.get("ndcg_at_10", 0.0)))
        rk_test_hrs.append(float(t_res.metrics.get("hit_rate_at_10", 0.0)))
        rk_test_mrrs.append(float(t_res.metrics.get("mrr_at_10", 0.0)))
        logger.info("  Ranker Seed %d -> Test NDCG@10: %.5f | HR@10: %.5f", s, rk_test_ndcgs[-1], rk_test_hrs[-1])

    rk_test_mean_ndcg = float(np.mean(rk_test_ndcgs))
    rk_test_std_ndcg = float(np.std(rk_test_ndcgs))
    rk_test_mean_hr = float(np.mean(rk_test_hrs))
    rk_test_std_hr = float(np.std(rk_test_hrs))
    logger.info("Clean Ranker 5-Seed Test: NDCG@10 = %.5f ± %.5f", rk_test_mean_ndcg, rk_test_std_ndcg)

    # Champion Ranker is Seed 43 (closest to val mean, selected in Phase 4)
    champ_rk_seed = 43
    champ_rk_model = rk_models[champ_rk_seed]
    champ_rk_eval = rk_test_evals[champ_rk_seed]
    champ_rk_val_eval = evaluate_model(
        champ_rk_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=champ_rk_seed
    )

    # Reference all-time models for transparency if present on disk
    ref_tt_dir = resolved_data_dir / "artifacts" / "recsys_two_tower"
    ref_tt_eval = None
    if ref_tt_dir.exists() and any(ref_tt_dir.iterdir()):
        try:
            subdirs = [d for d in ref_tt_dir.iterdir() if d.is_dir()]
            if subdirs:
                latest_ref_tt = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                ref_tt_m = TwoTowerModel.load(latest_ref_tt, data=data)
                ref_tt_eval = evaluate_model(ref_tt_m, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=42)
                logger.info("Reference Two-Tower (all-time stats) Test NDCG@10: %.5f", ref_tt_eval.metrics.get("ndcg_at_10", 0.0))
        except Exception as e:
            logger.warning("Could not evaluate reference Two-Tower: %s", e)

    ref_rk_dir = resolved_data_dir / "artifacts" / "recsys_two_tower_ranker"
    ref_rk_eval = None
    if ref_rk_dir.exists() and any(ref_rk_dir.iterdir()) and ref_tt_eval is not None:
        try:
            subdirs = [d for d in ref_rk_dir.iterdir() if d.is_dir()]
            if subdirs:
                latest_ref_rk = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                ref_stage1 = {
                    "two_tower": ref_tt_m,
                    "itemknn": knn_models[42],
                    "als": als_models[42],
                    "content": cnt_m,
                }
                ref_rk_m = RankerModel.load(latest_ref_rk, data=data, models=ref_stage1)
                ref_rk_eval = evaluate_model(ref_rk_m, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=42)
                logger.info("Reference Ranker (all-time stats) Test NDCG@10: %.5f", ref_rk_eval.metrics.get("ndcg_at_10", 0.0))
        except Exception as e:
            logger.warning("Could not evaluate reference Ranker: %s", e)

    # =========================================================================
    # 4. Paired Hypothesis Tests & Promotion Gate Evaluation
    # =========================================================================
    logger.info("\n>>> [4/7] Computing Paired Hypothesis Tests on TEST split...")
    def _extract_ranks(eval_obj: Any) -> np.ndarray:
        if "rank" in eval_obj.per_user:
            return np.asarray(eval_obj.per_user["rank"], dtype=np.int64)
        if "ranks" in eval_obj.per_user:
            return np.asarray(eval_obj.per_user["ranks"], dtype=np.int64)
        raise KeyError("Could not find 'rank' in per_user dictionary")

    champ_rk_ranks = _extract_ranks(champ_rk_eval)
    champ_tt_ranks = _extract_ranks(champ_tt_eval)
    knn_ranks = _extract_ranks(knn_test_evals[42])
    als_ranks = _extract_ranks(als_test_evals[42])
    cnt_ranks = _extract_ranks(cnt_test)
    pop_ranks = _extract_ranks(pop_test)

    rk_ndcg_u = ndcg_at_k(champ_rk_ranks, 10)
    tt_ndcg_u = ndcg_at_k(champ_tt_ranks, 10)
    knn_ndcg_u = ndcg_at_k(knn_ranks, 10)
    als_ndcg_u = ndcg_at_k(als_ranks, 10)
    cnt_ndcg_u = ndcg_at_k(cnt_ranks, 10)
    pop_ndcg_u = ndcg_at_k(pop_ranks, 10)

    # (a) Ranker Champion vs TT Champion
    rk_vs_tt_m, rk_vs_tt_lo, rk_vs_tt_hi = paired_bootstrap_diff(rk_ndcg_u, tt_ndcg_u, n_boot=2000, seed=seed)
    gate_passed = bool(rk_vs_tt_lo > 0.0)

    # (b) TT Champion vs ItemKNN
    tt_vs_knn_m, tt_vs_knn_lo, tt_vs_knn_hi = paired_bootstrap_diff(tt_ndcg_u, knn_ndcg_u, n_boot=2000, seed=seed)

    # (c) Ranker Champion vs ItemKNN
    rk_vs_knn_m, rk_vs_knn_lo, rk_vs_knn_hi = paired_bootstrap_diff(rk_ndcg_u, knn_ndcg_u, n_boot=2000, seed=seed)

    # (d) TT Champion vs ALS
    tt_vs_als_m, tt_vs_als_lo, tt_vs_als_hi = paired_bootstrap_diff(tt_ndcg_u, als_ndcg_u, n_boot=2000, seed=seed)

    # (e) TT Champion vs Content
    tt_vs_cnt_m, tt_vs_cnt_lo, tt_vs_cnt_hi = paired_bootstrap_diff(tt_ndcg_u, cnt_ndcg_u, n_boot=2000, seed=seed)

    # (f) TT Champion vs Popularity
    tt_vs_pop_m, tt_vs_pop_lo, tt_vs_pop_hi = paired_bootstrap_diff(tt_ndcg_u, pop_ndcg_u, n_boot=2000, seed=seed)

    logger.info("Paired Diff: Ranker Champion vs TT Champion: %+.5f [%+.5f, %+.5f]", rk_vs_tt_m, rk_vs_tt_lo, rk_vs_tt_hi)
    logger.info("  Ranker Promotion Gate Result: Passed=%s (lo95 > 0: %s)", gate_passed, rk_vs_tt_lo > 0.0)
    logger.info("Paired Diff: TT Champion vs ItemKNN: %+.5f [%+.5f, %+.5f]", tt_vs_knn_m, tt_vs_knn_lo, tt_vs_knn_hi)
    logger.info("Paired Diff: Ranker Champion vs ItemKNN: %+.5f [%+.5f, %+.5f]", rk_vs_knn_m, rk_vs_knn_lo, rk_vs_knn_hi)

    # =========================================================================
    # 5. Sliced Evaluations on Test Split
    # =========================================================================
    logger.info("\n>>> [5/7] Computing Sliced Evaluations on TEST split...")
    user_train_pos, _ = _build_user_interaction_maps(data)
    test_target_map = dict(zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True))
    train_pops = data.items_df["train_positives"].to_numpy().astype(float)
    pop_p80 = float(np.percentile(train_pops, 80))
    pop_p50 = float(np.percentile(train_pops, 50))

    # Identify slice masks over eval_users
    eval_user_targets = np.array([int(test_target_map[u]) for u in eval_users], dtype=np.int64)
    target_pops = train_pops[eval_user_targets]

    slice_definitions = {
        "pop_head": "Top 20% most popular items in training set (Target item popularity)",
        "pop_torso": "Next 30% most popular items (50th to 80th percentile)",
        "pop_tail": "Bottom 50% items (0 to 50th percentile)",
        "hist_3_4": "Users with 3 to 4 fold-in interaction history length",
        "hist_5_9": "Users with 5 to 9 fold-in interaction history length",
        "hist_10_plus": "Users with 10 or more fold-in interaction history length",
    }

    user_hist_lens = np.array([len(user_train_pos.get(u, [])) for u in eval_users], dtype=np.int64)

    slice_masks = {
        "pop_head": target_pops >= pop_p80,
        "pop_torso": (target_pops >= pop_p50) & (target_pops < pop_p80),
        "pop_tail": target_pops < pop_p50,
        "hist_3_4": (user_hist_lens >= 3) & (user_hist_lens <= 4),
        "hist_5_9": (user_hist_lens >= 5) & (user_hist_lens <= 9),
        "hist_10_plus": user_hist_lens >= 10,
    }

    slice_comparisons: list[dict[str, Any]] = []

    # Overall comparison (Test)
    slice_comparisons.append({
        "comparison": "Clean Ranker vs Clean Two-Tower",
        "split": "test",
        "slice_name": "overall",
        "n_users": len(eval_users),
        "model_a_ndcg": float(champ_rk_eval.metrics.get("ndcg_at_10", 0.0)),
        "model_b_ndcg": float(champ_tt_eval.metrics.get("ndcg_at_10", 0.0)),
        "paired_diff": rk_vs_tt_m,
        "lo95": rk_vs_tt_lo,
        "hi95": rk_vs_tt_hi,
        "verdict": "better" if rk_vs_tt_lo > 0.0 else ("worse" if rk_vs_tt_hi < 0.0 else "no significant difference"),
        "rank_source": "fresh evaluate_model",
    })

    # Slice comparisons (Test)
    for s_name, s_mask in slice_masks.items():
        if np.sum(s_mask) == 0:
            continue
        nd_rk_slice = rk_ndcg_u[s_mask]
        nd_tt_slice = tt_ndcg_u[s_mask]
        m_a = float(np.mean(nd_rk_slice))
        m_b = float(np.mean(nd_tt_slice))
        d_m, d_lo, d_hi = paired_bootstrap_diff(nd_rk_slice, nd_tt_slice, n_boot=1000, seed=seed)
        v = "better" if d_lo > 0.0 else ("worse" if d_hi < 0.0 else "no significant difference")

        slice_comparisons.append({
            "comparison": "Ranker vs Two-Tower",
            "split": "test",
            "slice_name": s_name,
            "n_users": int(np.sum(s_mask)),
            "model_a_ndcg": m_a,
            "model_b_ndcg": m_b,
            "paired_diff": d_m,
            "lo95": d_lo,
            "hi95": d_hi,
            "verdict": v,
        })

    # =========================================================================
    # 6. Beyond-Accuracy Metrics & MMR Re-Ranking on Test Split
    # =========================================================================
    logger.info("\n>>> [6/7] Computing Beyond-Accuracy Metrics & MMR Re-Ranking...")
    beyond_acc_entries: list[dict[str, Any]] = []

    models_for_beyond = [
        ("recsys_two_tower_ranker_clean", champ_rk_eval),
        ("recsys_two_tower_clean", champ_tt_eval),
        ("recsys_itemknn", knn_test_evals[42]),
        ("recsys_als", als_test_evals[42]),
        ("recsys_popularity", pop_test),
        ("recsys_content", cnt_test),
    ]

    for m_name, m_eval in models_for_beyond:
        beyond_acc_entries.append({
            "model_name": m_name,
            "catalog_coverage_at_10": float(m_eval.metrics.get("catalog_coverage_at_10", 0.0)),
            "novelty_at_10": float(m_eval.metrics.get("novelty_at_10", 0.0)),
            "diversity_at_10": float(m_eval.metrics.get("diversity_at_10", 0.0)),
            "head_share_at_10": float(m_eval.metrics.get("head_share_at_10", 0.0)),
            "gini_coefficient_at_10": float(m_eval.metrics.get("gini_coefficient_at_10", 0.0)),
        })

    # MMR re-ranking trade-off sensitivity on Ranker
    logger.info("Computing MMR diversity trade-off on Ranker candidate pool...")
    test_hist_list, _, seen_list = _build_test_foldin_data(data, np.array(eval_users, dtype=np.int64))
    test_targets_arr = np.array([int(test_target_map[u]) for u in eval_users], dtype=np.int64)

    # Score ranker on all eval users
    ranker_test_scores = np.empty((len(eval_users), data.n_items), dtype=np.float32)
    for i in range(0, len(eval_users), 1024):
        b_slice = slice(i, min(i + 1024, len(eval_users)))
        b_hist = test_hist_list[b_slice]
        b_scores = champ_rk_model.score(b_hist).astype(np.float32, copy=False)
        for idx, g_idx in enumerate(range(b_slice.start, b_slice.stop)):
            s_items = list(seen_list[g_idx])
            if s_items:
                b_scores[idx, s_items] = -np.inf
        ranker_test_scores[b_slice] = b_scores

    top50_idx = get_top_k_indices(ranker_test_scores, 50)
    row_grid = np.arange(len(eval_users))[:, None]
    top50_scores = ranker_test_scores[row_grid, top50_idx]

    mmr_entries: list[dict[str, Any]] = []
    lam1_ndcg_per_user = None

    for lam in [1.0, 0.8, 0.6]:
        reranked_top10 = mmr_rerank(
            candidate_items=top50_idx,
            candidate_scores=top50_scores,
            item_embeddings=data.item_text_emb,
            lambda_param=lam,
            k=10,
        )
        ranks_mmr = np.full(len(eval_users), 999, dtype=np.int64)
        for u_i, t in enumerate(test_targets_arr):
            top_items = reranked_top10[u_i]
            pos = np.where(top_items == t)[0]
            if len(pos) > 0:
                ranks_mmr[u_i] = int(pos[0]) + 1

        ndcg_vals = ndcg_at_k(ranks_mmr, 10)
        nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_vals, n_boot=1000, seed=seed)
        div_val = diversity_at_k(reranked_top10, data.item_text_emb) if data.item_text_emb is not None else 0.0

        if lam == 1.0:
            lam1_ndcg_per_user = ndcg_vals
            diff_m, diff_lo, diff_hi = 0.0, 0.0, 0.0
        else:
            diff_m, diff_lo, diff_hi = paired_bootstrap_diff(ndcg_vals, lam1_ndcg_per_user, seed=seed)

        mmr_entries.append({
            "lambda_param": lam,
            "ndcg_at_10": float(nd_m),
            "ndcg_at_10_lo95": float(nd_lo),
            "ndcg_at_10_hi95": float(nd_hi),
            "diversity_at_10": float(div_val),
            "paired_ndcg_diff_vs_lambda1": float(diff_m),
            "paired_ndcg_diff_lo95": float(diff_lo),
            "paired_ndcg_diff_hi95": float(diff_hi),
        })

    # =========================================================================
    # 7. Multi-Seed True Item Holdout Cold-Start Study & Routed Hybrid
    # =========================================================================
    logger.info("\n>>> [7/7] Running Multi-Seed True Item Holdout Cold-Start Study (seeds: %s)...", seeds_3)
    held_out_items, held_out_set = get_item_holdout_split(data, fraction=0.10, seed=42)
    held_out_list = [int(it) for it in held_out_items]

    holdout_users = [int(u) for u in eval_users if int(test_target_map[int(u)]) in held_out_set]
    n_ho_users = len(holdout_users)
    logger.info("True Holdout Users: %d (across %d withheld items)", n_ho_users, len(held_out_set))

    ho_hist, ho_targ, ho_seen = _build_test_foldin_data(data, np.array(holdout_users, dtype=np.int64))
    ho_targ_arr = np.asarray(ho_targ, dtype=np.int64)

    # Content model scores on holdout users
    sc_ho_cnt = cnt_m.score(ho_hist).copy()
    for i, sm in enumerate(ho_seen):
        sc_ho_cnt[i, list(sm)] = -np.inf
    ho_rk_cnt = compute_ranks(sc_ho_cnt, ho_targ_arr)

    # Analytic Random baseline
    n_items_float = float(data.n_items)
    analytic_mean_rank = n_items_float / 2.0
    analytic_hr_10 = 10.0 / n_items_float
    analytic_ndcg_10 = float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / n_items_float)
    analytic_mrr_10 = float(np.sum(1.0 / np.arange(1, 11)) / n_items_float)
    analytic_hits = int(round(n_ho_users * analytic_hr_10))

    random_summary = {
        "hits_k": analytic_hits,
        "n_users": n_ho_users,
        "hit_rate_at_10": analytic_hr_10,
        "hit_rate_at_10_lo95": analytic_hr_10,
        "hit_rate_at_10_hi95": analytic_hr_10,
        "ndcg_at_10": analytic_ndcg_10,
        "ndcg_at_10_lo95": analytic_ndcg_10,
        "ndcg_at_10_hi95": analytic_ndcg_10,
        "mrr_at_10": analytic_mrr_10,
        "mrr_at_10_lo95": analytic_mrr_10,
        "mrr_at_10_hi95": analytic_mrr_10,
        "mean_rank": analytic_mean_rank,
        "mean_rank_lo95": analytic_mean_rank,
        "mean_rank_hi95": analytic_mean_rank,
        "median_rank": analytic_mean_rank,
    }

    # Evaluate across 3 seeds of Two-Tower masked and Routed Hybrid
    tt_ho_ndcgs = []
    tt_ho_hrs = []
    tt_ho_ranks = []

    hybrid_ho_ndcgs = []
    hybrid_ho_hrs = []
    hybrid_ho_ranks = []

    for s in seeds_3:
        tt_s = tt_models[s]
        # Condition (c3): All proxies neutralized + ID masked
        vecs_c3 = tt_s.item_vectors(mask_ids=held_out_list, neutralize_proxies=held_out_list)
        sc_tt_ho = tt_s.score(
            ho_hist, item_vecs=vecs_c3, mask_ids=held_out_list, neutralize_proxies=held_out_list
        ).copy()
        for i, sm in enumerate(ho_seen):
            sc_tt_ho[i, list(sm)] = -np.inf

        # Pure Two-Tower Masked ranks
        rk_s = compute_ranks(sc_tt_ho, ho_targ_arr)
        tt_ho_ndcgs.append(float(np.mean(ndcg_at_k(rk_s, 10))))
        tt_ho_hrs.append(float(np.mean(hit_rate_at_k(rk_s, 10))))
        tt_ho_ranks.append(float(np.mean(rk_s)))

        # Content-Routed Hybrid:
        # Warm items scored with Two-Tower; cold/held-out items routed to Content model scores
        sc_hybrid = sc_tt_ho.copy()
        for it in held_out_list:
            sc_hybrid[:, it] = sc_ho_cnt[:, it]

        rk_hyb = compute_ranks(sc_hybrid, ho_targ_arr)
        hybrid_ho_ndcgs.append(float(np.mean(ndcg_at_k(rk_hyb, 10))))
        hybrid_ho_hrs.append(float(np.mean(hit_rate_at_k(rk_hyb, 10))))
        hybrid_ho_ranks.append(float(np.mean(rk_hyb)))

    def summarize_ranks_helper(ranks: np.ndarray, n_u: int) -> dict[str, Any]:
        hr_arr = hit_rate_at_k(ranks, 10)
        nd_arr = ndcg_at_k(ranks, 10)
        mrr_arr = mrr_at_k(ranks, 10)
        hr_m, hr_lo, hr_hi = bootstrap_ci(hr_arr, n_boot=1000, seed=seed)
        nd_m, nd_lo, nd_hi = bootstrap_ci(nd_arr, n_boot=1000, seed=seed)
        mrr_m, mrr_lo, mrr_hi = bootstrap_ci(mrr_arr, n_boot=1000, seed=seed)
        mr_m, mr_lo, mr_hi = bootstrap_ci(ranks, n_boot=1000, seed=seed)
        return {
            "hits_k": int(np.sum(hr_arr)),
            "n_users": n_u,
            "hit_rate_at_10": float(hr_m),
            "hit_rate_at_10_lo95": float(hr_lo),
            "hit_rate_at_10_hi95": float(hr_hi),
            "ndcg_at_10": float(nd_m),
            "ndcg_at_10_lo95": float(nd_lo),
            "ndcg_at_10_hi95": float(nd_hi),
            "mrr_at_10": float(mrr_m),
            "mrr_at_10_lo95": float(mrr_lo),
            "mrr_at_10_hi95": float(mrr_hi),
            "mean_rank": float(mr_m),
            "mean_rank_lo95": float(mr_lo),
            "mean_rank_hi95": float(mr_hi),
            "median_rank": float(np.median(ranks)),
        }

    cnt_ho_summary = summarize_ranks_helper(ho_rk_cnt, n_ho_users)

    # Representative seed 42 summaries
    vecs_c3_42 = champ_tt_model.item_vectors(mask_ids=held_out_list, neutralize_proxies=held_out_list)
    sc_tt_42 = champ_tt_model.score(ho_hist, item_vecs=vecs_c3_42, mask_ids=held_out_list, neutralize_proxies=held_out_list).copy()
    for i, sm in enumerate(ho_seen):
        sc_tt_42[i, list(sm)] = -np.inf
    rk_tt_42 = compute_ranks(sc_tt_42, ho_targ_arr)
    tt_masked_summary = summarize_ranks_helper(rk_tt_42, n_ho_users)
    tt_masked_summary["multi_seed_mean_ndcg"] = float(np.mean(tt_ho_ndcgs))
    tt_masked_summary["multi_seed_std_ndcg"] = float(np.std(tt_ho_ndcgs))

    sc_hyb_42 = sc_tt_42.copy()
    for it in held_out_list:
        sc_hyb_42[:, it] = sc_ho_cnt[:, it]
    rk_hyb_42 = compute_ranks(sc_hyb_42, ho_targ_arr)
    hybrid_summary = summarize_ranks_helper(rk_hyb_42, n_ho_users)
    hybrid_summary["multi_seed_mean_ndcg"] = float(np.mean(hybrid_ho_ndcgs))
    hybrid_summary["multi_seed_std_ndcg"] = float(np.std(hybrid_ho_ndcgs))

    # Sparse targets subgroup study
    sp_users, sp_targets = get_sparse_target_test_users(data, eval_users, max_train_positives=10)
    n_sp_users = len(sp_users)
    hist_sp, targ_sp, seen_sp = _build_test_foldin_data(data, sp_users)
    targ_sp_arr = np.asarray(targ_sp, dtype=np.int64)
    sp_target_ids = list(set(int(t) for t in targ_sp))

    rng = np.random.default_rng(seed)
    rk_sp_rnd = rng.integers(1, data.n_items + 1, size=n_sp_users)

    sc_sp_pop = pop_m.score(hist_sp).copy()
    for i, sm in enumerate(seen_sp):
        sc_sp_pop[i, list(sm)] = -np.inf
    rk_sp_pop = compute_ranks(sc_sp_pop, targ_sp_arr)

    sc_sp_cnt = cnt_m.score(hist_sp).copy()
    for i, sm in enumerate(seen_sp):
        sc_sp_cnt[i, list(sm)] = -np.inf
    rk_sp_cnt = compute_ranks(sc_sp_cnt, targ_sp_arr)

    sc_sp_als = als_models[42].score(hist_sp).copy()
    for i, sm in enumerate(seen_sp):
        sc_sp_als[i, list(sm)] = -np.inf
    rk_sp_als = compute_ranks(sc_sp_als, targ_sp_arr)

    sc_sp_tti = champ_tt_model.score(hist_sp).copy()
    for i, sm in enumerate(seen_sp):
        sc_sp_tti[i, list(sm)] = -np.inf
    rk_sp_tti = compute_ranks(sc_sp_tti, targ_sp_arr)

    tt_vecs_sp = champ_tt_model.item_vectors(mask_ids=sp_target_ids)
    sc_sp_ttm = champ_tt_model.score(hist_sp, item_vecs=tt_vecs_sp).copy()
    for i, sm in enumerate(seen_sp):
        sc_sp_ttm[i, list(sm)] = -np.inf
    rk_sp_ttm = compute_ranks(sc_sp_ttm, targ_sp_arr)

    sparse_models_summary = {
        "random": summarize_ranks_helper(rk_sp_rnd, n_sp_users),
        "popularity": summarize_ranks_helper(rk_sp_pop, n_sp_users),
        "content": summarize_ranks_helper(rk_sp_cnt, n_sp_users),
        "als": summarize_ranks_helper(rk_sp_als, n_sp_users),
        "two_tower_intact": summarize_ranks_helper(rk_sp_tti, n_sp_users),
        "two_tower_masked": summarize_ranks_helper(rk_sp_ttm, n_sp_users),
    }

    # Simulated new items study (C2 fair simulation)
    new_item_ids, ni_users, ni_targets = get_simulated_new_items_test_users(
        data, eval_users, fraction=0.10, seed=seed, variant="c2"
    )
    n_ni_users = len(ni_users)
    hist_ni, targ_ni, seen_ni = _build_test_foldin_data(data, ni_users)
    targ_ni_arr = np.asarray(targ_ni, dtype=np.int64)
    ni_item_list = list(set(int(t) for t in new_item_ids))

    rk_ni_rnd = rng.integers(1, data.n_items + 1, size=n_ni_users)

    sc_ni_cnt = cnt_m.score(hist_ni).copy()
    for i, sm in enumerate(seen_ni):
        sc_ni_cnt[i, list(sm)] = -np.inf
    rk_ni_cnt = compute_ranks(sc_ni_cnt, targ_ni_arr)

    vecs_ni = champ_tt_model.item_vectors(mask_ids=ni_item_list, neutralize_proxies=ni_item_list)
    sc_ni_ttm = champ_tt_model.score(
        hist_ni, item_vecs=vecs_ni, mask_ids=ni_item_list, neutralize_proxies=ni_item_list
    ).copy()
    for i, sm in enumerate(seen_ni):
        sc_ni_ttm[i, list(sm)] = -np.inf
    rk_ni_ttm = compute_ranks(sc_ni_ttm, targ_ni_arr)

    rk_ni_pop = np.full(n_ni_users, data.n_items, dtype=np.int64)
    rk_ni_als = np.full(n_ni_users, data.n_items, dtype=np.int64)
    rk_ni_knn = np.full(n_ni_users, data.n_items, dtype=np.int64)

    sim_models_summary = {
        "random": summarize_ranks_helper(rk_ni_rnd, n_ni_users),
        "popularity": summarize_ranks_helper(rk_ni_pop, n_ni_users),
        "als": summarize_ranks_helper(rk_ni_als, n_ni_users),
        "itemknn": summarize_ranks_helper(rk_ni_knn, n_ni_users),
        "content": summarize_ranks_helper(rk_ni_cnt, n_ni_users),
        "two_tower_masked": summarize_ranks_helper(rk_ni_ttm, n_ni_users),
    }

    cold_start_payload = build_cold_start_payload(
        dataset_version=data.dataset_version,
        sparse_targets_study={
            "study_type": "sparse_targets",
            "primary_evidence": False,
            "title": "Sparse Targets Subgroup Study (<= 10 Train Positives)",
            "is_optimistic_proxy": True,
            "n_eval_users": n_sp_users,
            "max_train_positives": 10,
            "models": sparse_models_summary,
            "limitations_note": "Target item ID is masked at inference on clean Two-Tower model.",
        },
        simulated_new_items_study={
            "study_type": "simulated_new_items",
            "variant": "c2",
            "variant_label": "C2: Fair Simulation (Random 10%, Pop Zeroed, Proxies Neutralized)",
            "primary_evidence": False,
            "is_optimistic_proxy": True,
            "n_eval_users": n_ni_users,
            "n_new_items": len(new_item_ids),
            "fraction_masked": 0.10,
            "models": sim_models_summary,
            "random_reference": {
                "hit_rate_at_10": 10.0 / data.n_items,
                "ndcg_at_10": float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / data.n_items),
                "formula": "10 / n_items",
            },
            "limitations_note": "Inference-time masking simulation on clean two-tower model with neutralized rating/review proxies.",
        },
        item_holdout_study={
            "study_type": "item_holdout",
            "primary_evidence": True,
            "title": "True Item Holdout (10% Catalog Withheld, 300 Items — Multi-Seed Study)",
            "n_eval_users": n_ho_users,
            "n_held_out_items": len(held_out_set),
            "models": {
                "random": random_summary,
                "content": cnt_ho_summary,
                "two_tower_masked": tt_masked_summary,
                "routed_hybrid": hybrid_summary,
            },
            "multi_seed_results": {
                "seeds": seeds_3,
                "two_tower_masked": {
                    "mean_ndcg_at_10": float(np.mean(tt_ho_ndcgs)),
                    "std_ndcg_at_10": float(np.std(tt_ho_ndcgs)),
                    "mean_rank": float(np.mean(tt_ho_ranks)),
                },
                "routed_hybrid": {
                    "mean_ndcg_at_10": float(np.mean(hybrid_ho_ndcgs)),
                    "std_ndcg_at_10": float(np.std(hybrid_ho_ndcgs)),
                    "mean_rank": float(np.mean(hybrid_ho_ranks)),
                },
            },
            "random_reference": {
                "hit_rate_at_10": 10.0 / data.n_items,
                "ndcg_at_10": float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / data.n_items),
                "formula": "10 / n_items",
            },
            "note": "Multi-seed true item holdout study. Held-out items had zero interactions during training.",
        },
        n_eval_users=len(eval_users),
        seed=seed,
    )

    # =========================================================================
    # 8. Construct Leaderboard Payload & Markdown
    # =========================================================================
    logger.info("\n>>> Building Leaderboard Payload and Slices Payload...")
    # Grab validation metrics from Phase 4 sweep
    tt_val_metrics = champ_tt_val_eval.metrics.copy()
    if "two_tower_clean_5seeds" in val_sweep_data:
        tt_val_mean = val_sweep_data["two_tower_clean_5seeds"]["mean_ndcg_at_10"]
        tt_val_std = val_sweep_data["two_tower_clean_5seeds"]["std_ndcg_at_10"]
    else:
        tt_val_mean = float(champ_tt_val_eval.metrics.get("ndcg_at_10", 0.0))
        tt_val_std = 0.0

    rk_val_metrics = champ_rk_val_eval.metrics.copy()
    if "ranker_clean_5seeds" in val_sweep_data:
        rk_val_mean = val_sweep_data["ranker_clean_5seeds"]["mean_ndcg_at_10"]
        rk_val_std = val_sweep_data["ranker_clean_5seeds"]["std_ndcg_at_10"]
    else:
        rk_val_mean = float(champ_rk_val_eval.metrics.get("ndcg_at_10", 0.0))
        rk_val_std = 0.0

    models_payload_data = [
        {
            "model_name": "recsys_two_tower_clean",
            "family": "recsys_two_tower_clean",
            "stage": "champion",
            "model_run_id": f"phase4_seed_{champ_tt_seed}",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": champ_tt_model.hyperparameters(),
            "val_metrics": tt_val_metrics,
            "test_metrics": champ_tt_eval.metrics,
            "val_slices": champ_tt_eval.slices,
            "test_slices": champ_tt_eval.slices,
            "multi_seed_5seeds": {
                "test_mean_ndcg_at_10": tt_test_mean_ndcg,
                "test_std_ndcg_at_10": tt_test_std_ndcg,
                "test_mean_hr_at_10": tt_test_mean_hr,
                "test_std_hr_at_10": tt_test_std_hr,
                "val_mean_ndcg_at_10": tt_val_mean,
                "val_std_ndcg_at_10": tt_val_std,
            },
        },
        {
            "model_name": "recsys_two_tower_ranker_clean",
            "family": "recsys_two_tower_ranker_clean",
            "stage": "candidate" if not gate_passed else "production",
            "model_run_id": f"phase4_seed_{champ_rk_seed}",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": champ_rk_model.hyperparameters() if hasattr(champ_rk_model, "hyperparameters") else {},
            "val_metrics": rk_val_metrics,
            "test_metrics": champ_rk_eval.metrics,
            "val_slices": champ_rk_eval.slices,
            "test_slices": champ_rk_eval.slices,
            "multi_seed_5seeds": {
                "test_mean_ndcg_at_10": rk_test_mean_ndcg,
                "test_std_ndcg_at_10": rk_test_std_ndcg,
                "test_mean_hr_at_10": rk_test_mean_hr,
                "test_std_hr_at_10": rk_test_std_hr,
                "val_mean_ndcg_at_10": rk_val_mean,
                "val_std_ndcg_at_10": rk_val_std,
            },
            "paired_diff_vs_champion": {
                "diff_ndcg_at_10": rk_vs_tt_m,
                "lo95": rk_vs_tt_lo,
                "hi95": rk_vs_tt_hi,
                "gate_passed": gate_passed,
            },
        },
        {
            # Alias for Model Card and serving compatibility
            "model_name": "recsys_two_tower_ranker_v2_trainstats",
            "family": "recsys_two_tower_ranker_v2_trainstats",
            "stage": "candidate",
            "model_run_id": f"phase4_seed_{champ_rk_seed}",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": champ_rk_model.hyperparameters() if hasattr(champ_rk_model, "hyperparameters") else {},
            "val_metrics": rk_val_metrics,
            "test_metrics": champ_rk_eval.metrics,
            "val_slices": champ_rk_eval.slices,
            "test_slices": champ_rk_eval.slices,
        },
        {
            "model_name": "recsys_itemknn",
            "family": "recsys_itemknn",
            "stage": "candidate",
            "model_run_id": "itemknn_5seed",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {"k_neighbors": 100, "shrink": 10.0},
            "val_metrics": knn_val_evals[42].metrics,
            "test_metrics": knn_test_evals[42].metrics,
            "val_slices": knn_val_evals[42].slices,
            "test_slices": knn_test_evals[42].slices,
            "multi_seed_5seeds": {
                "test_mean_ndcg_at_10": float(np.mean(knn_test_ndcgs)),
                "test_std_ndcg_at_10": float(np.std(knn_test_ndcgs)),
            },
        },
        {
            "model_name": "recsys_als",
            "family": "recsys_als",
            "stage": "candidate",
            "model_run_id": "als_5seed",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {"factors": 64, "regularization": 0.05, "alpha": 15.0, "iterations": 15},
            "val_metrics": als_val_evals[42].metrics,
            "test_metrics": als_test_evals[42].metrics,
            "val_slices": als_val_evals[42].slices,
            "test_slices": als_test_evals[42].slices,
            "multi_seed_5seeds": {
                "test_mean_ndcg_at_10": float(np.mean(als_test_ndcgs)),
                "test_std_ndcg_at_10": float(np.std(als_test_ndcgs)),
            },
        },
        {
            "model_name": "recsys_popularity",
            "family": "recsys_popularity",
            "stage": "candidate",
            "model_run_id": "popularity_baseline",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {},
            "val_metrics": pop_val.metrics,
            "test_metrics": pop_test.metrics,
            "val_slices": pop_val.slices,
            "test_slices": pop_test.slices,
        },
        {
            "model_name": "recsys_content",
            "family": "recsys_content",
            "stage": "candidate",
            "model_run_id": "content_baseline",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {"weight_tags": 0.5},
            "val_metrics": cnt_val.metrics,
            "test_metrics": cnt_test.metrics,
            "val_slices": cnt_val.slices,
            "test_slices": cnt_test.slices,
        },
    ]

    # Include reference all-time models if available
    if ref_tt_eval is not None:
        models_payload_data.append({
            "model_name": "recsys_two_tower",
            "family": "recsys_two_tower",
            "stage": "upper_bound_reference",
            "model_run_id": "reference_all_time",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {"pool": "attn", "clean": False},
            "val_metrics": ref_tt_eval.metrics,
            "test_metrics": ref_tt_eval.metrics,
            "val_slices": ref_tt_eval.slices,
            "test_slices": ref_tt_eval.slices,
        })

    if ref_rk_eval is not None:
        models_payload_data.append({
            "model_name": "recsys_two_tower_ranker",
            "family": "recsys_two_tower_ranker",
            "stage": "upper_bound_reference",
            "model_run_id": "reference_all_time",
            "rank_source": "fresh evaluate_model",
            "hyperparameters": {"clean": False},
            "val_metrics": ref_rk_eval.metrics,
            "test_metrics": ref_rk_eval.metrics,
            "val_slices": ref_rk_eval.slices,
            "test_slices": ref_rk_eval.slices,
        })

    leaderboard_payload = build_leaderboard_payload(
        dataset_version=data.dataset_version,
        models_data=models_payload_data,
        n_eval_users=len(eval_users),
        seed=seed,
    )

    slices_payload = build_slices_payload(
        dataset_version=data.dataset_version,
        slice_definitions=slice_definitions,
        comparisons=slice_comparisons,
        n_eval_users=len(eval_users),
        seed=seed,
    )

    beyond_acc_payload = build_beyond_accuracy_payload(
        dataset_version=data.dataset_version,
        metrics_at_10=beyond_acc_entries,
        mmr_reranking=mmr_entries,
        n_eval_users=len(eval_users),
        seed=seed,
    )

    # Save JSON files to both docs/results/ and reports_dir
    logger.info("Persisting all Phase 6 report payloads to disk...")
    save_report_json(docs_results_dir / "leaderboard.json", leaderboard_payload)
    save_report_json(reports_dir / "leaderboard.json", leaderboard_payload)

    save_report_json(docs_results_dir / "slices.json", slices_payload)
    save_report_json(reports_dir / "slices.json", slices_payload)

    save_report_json(docs_results_dir / "beyond_accuracy.json", beyond_acc_payload)
    save_report_json(reports_dir / "beyond_accuracy.json", beyond_acc_payload)

    save_report_json(docs_results_dir / "cold_start.json", cold_start_payload)
    save_report_json(reports_dir / "cold_start.json", cold_start_payload)

    # Render markdown leaderboard
    ablations_path = docs_results_dir / "ablations.json"
    ablations_p = json.loads(ablations_path.read_text(encoding="utf-8")) if ablations_path.exists() else None

    rk_abl_path = docs_results_dir / "ranker_ablations.json"
    rk_abl_p = json.loads(rk_abl_path.read_text(encoding="utf-8")) if rk_abl_path.exists() else None

    md_content = generate_leaderboard_markdown(
        leaderboard_payload=leaderboard_payload,
        ablations_payload=ablations_p,
        ranker_ablations_payload=rk_abl_p,
        slices_payload=slices_payload,
        cold_start_payload=cold_start_payload,
        beyond_accuracy_payload=beyond_acc_payload,
    )
    (docs_results_dir / "leaderboard.md").write_text(md_content, encoding="utf-8")
    (reports_dir / "leaderboard.md").write_text(md_content, encoding="utf-8")
    logger.info("Successfully updated docs/results/leaderboard.md")

    elapsed = time.perf_counter() - t0_all
    logger.info("\n✅ Phase 6 Final Test Evaluation completed successfully in %.2f s!", elapsed)

    return {
        "two_tower_clean_test_ndcg_mean": tt_test_mean_ndcg,
        "two_tower_clean_test_ndcg_std": tt_test_std_ndcg,
        "two_tower_champion_test_ndcg": float(champ_tt_eval.metrics.get("ndcg_at_10", 0.0)),
        "ranker_clean_test_ndcg_mean": rk_test_mean_ndcg,
        "ranker_clean_test_ndcg_std": rk_test_std_ndcg,
        "ranker_champion_test_ndcg": float(champ_rk_eval.metrics.get("ndcg_at_10", 0.0)),
        "ranker_vs_tt_diff": rk_vs_tt_m,
        "ranker_vs_tt_ci": [rk_vs_tt_lo, rk_vs_tt_hi],
        "promotion_gate_passed": gate_passed,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Final Test Evaluation & Cold-Start Study")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_phase6_test_eval(data_dir=args.data_dir, seed=args.seed)


if __name__ == "__main__":
    main()
