"""
Phase 4: Multi-Seed Experiment Sweep (Validation-Driven).

Executes a comprehensive, leak-free validation sweep:
  1. Clean Two-Tower across 5 seeds [42, 43, 44, 45, 46] (train-only features, Phase 1 & 3 logging).
  2. Baselines: Deterministic (Popularity, Content) on 1 seed; Stochastic (ItemKNN, ALS) on 5 seeds.
  3. Two-Tower Architectural Ablations (no_tags, no_numeric, no_id) on 3 seeds [42, 43, 44].
  4. Clean Ranker across 5 seeds [42, 43, 44, 45, 46] (train-only features, quota candidate generation).
  5. Ranker Feature-Group Ablations on 3 seeds [42, 43, 44].
  6. Champion selection based strictly on validation split proximity to seed mean.
  7. Multi-seed training curves persistence into docs/results/training_curves.json.

GROUND RULE 3: Test split is strictly protected and NEVER evaluated in this phase.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings
from recsys.candidates import DEFAULT_CANDIDATE_COUNTS, generate_candidates
from recsys.data import RecsysData
from recsys.evaluate import (
    EvaluationResult,
    evaluate_model,
    get_evaluation_user_split,
)
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.ranker import (
    FEATURE_COLUMNS,
    RankerConfig,
    RankerModel,
    train_ranker,
)
from recsys.report import build_training_curves_payload, save_report_json
from recsys.train_two_tower import train_two_tower_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("phase4_sweep")


def run_sweep(
    data_dir: Path | str = "data/recsys",
    results_dir: Path | str = "docs/results",
    epochs_tt: int = 15,
    pairs_per_epoch: int | None = None,
    patience: int = 5,
    n_ranker_users: int = 15000,
    skip_ablations: bool = False,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info("================================================================================")
    logger.info("Starting Phase 4 Multi-Seed Sweep (Validation-Driven, Test-Protected)")
    logger.info("================================================================================")

    data = RecsysData.load(data_path)
    logger.info("Loaded dataset: %d items, %d users, %d train rows", data.n_items, data.n_users, len(data.train_df))

    # Disjoint user splits: 5,000 evaluation users, 5,000 tuning users
    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=data_path,
    )
    logger.info("User split: %d evaluation users, %d tuning users", len(eval_users), len(tuning_users))

    seeds_5 = [42, 43, 44, 45, 46]
    seeds_3 = [42, 43, 44]

    # ==============================================================================
    # 1. Clean Two-Tower Multi-Seed Sweep (5 seeds)
    # ==============================================================================
    logger.info("\n>>> [1/5] Training Clean Two-Tower across 5 seeds: %s", seeds_5)
    tt_results: dict[int, dict[str, Any]] = {}
    tt_models: dict[int, TwoTowerModel] = {}
    tt_curves_by_seed: dict[str, list[dict[str, Any]]] = {}

    tt_base_cfg = TwoTowerConfig(
        n_items=data.n_items,
        text_dim=384,
        tag_dim=300,
        numeric_dim=6,
        id_dim=64,
        text_proj_dim=64,
        tag_proj_dim=32,
        numeric_proj_dim=16,
        hidden_dim=256,
        output_dim=64,
        pool="attn",
        id_dropout=0.25,
        proxy_dropout=0.25,
        no_pop_proxies=True,
    )

    art_tt_dir = data_path / "artifacts" / "recsys_two_tower_clean"
    art_tt_dir.mkdir(parents=True, exist_ok=True)

    for s in seeds_5:
        seed_art_dir = art_tt_dir / f"seed_{s}"
        if (seed_art_dir / "two_tower.pt").exists() and (seed_art_dir / "training_curves.json").exists():
            logger.info("--- Loading cached Two-Tower Seed %d from %s ---", s, seed_art_dir)
            t0 = time.perf_counter()
            m_wrapper = TwoTowerModel.load(seed_art_dir, data=data)
            m_wrapper.name = "recsys_two_tower_clean"
            with open(seed_art_dir / "training_curves.json", "r", encoding="utf-8") as f:
                t_curves = json.load(f)
        else:
            logger.info("--- Training Two-Tower Seed %d ---", s)
            t0 = time.perf_counter()
            m_wrapper, t_curves = train_two_tower_model(
                data=data,
                tuning_users=tuning_users,
                cfg=tt_base_cfg,
                epochs=epochs_tt,
                batch_size=2048,
                lr=2e-3,
                patience=patience,
                pairs_per_epoch=pairs_per_epoch,
                seed=s,
            )
            m_wrapper.name = "recsys_two_tower_clean"
            seed_art_dir.mkdir(parents=True, exist_ok=True)
            m_wrapper.save(seed_art_dir)
            with open(seed_art_dir / "training_curves.json", "w", encoding="utf-8") as f:
                json.dump(t_curves, f, indent=2)

        tt_models[s] = m_wrapper
        tt_curves_by_seed[str(s)] = t_curves.get("history", [])

        # Evaluate strictly on validation split
        val_eval = evaluate_model(
            model=m_wrapper,
            data=data,
            split="val",
            k_list=(10, 20),
            user_indices=eval_users,
            seed=s,
        )
        t_elapsed = time.perf_counter() - t0

        tt_results[s] = {
            "val_ndcg_at_10": float(val_eval.metrics.get("ndcg_at_10", 0.0)),
            "val_ndcg_at_10_lo95": float(val_eval.metrics.get("ndcg_at_10_lo95", 0.0)),
            "val_ndcg_at_10_hi95": float(val_eval.metrics.get("ndcg_at_10_hi95", 0.0)),
            "val_hr_at_10": float(val_eval.metrics.get("hit_rate_at_10", 0.0)),
            "val_hr_at_10_lo95": float(val_eval.metrics.get("hit_rate_at_10_lo95", 0.0)),
            "val_hr_at_10_hi95": float(val_eval.metrics.get("hit_rate_at_10_hi95", 0.0)),
            "val_mrr_at_10": float(val_eval.metrics.get("mrr_at_10", 0.0)),
            "coverage_at_10": float(val_eval.metrics.get("catalog_coverage_at_10", 0.0)),
            "novelty_at_10": float(val_eval.metrics.get("novelty_at_10", 0.0)),
            "diversity_at_10": float(val_eval.metrics.get("diversity_at_10", 0.0)),
            "snips_ndcg_at_10": float(val_eval.metrics.get("snips_ndcg_at_10", 0.0)),
            "snips_ndcg_at_10_lo95": float(val_eval.metrics.get("snips_ndcg_at_10_lo95", 0.0)),
            "snips_ndcg_at_10_hi95": float(val_eval.metrics.get("snips_ndcg_at_10_hi95", 0.0)),
            "training_time_sec": round(t_elapsed, 2),
            "val_eval": val_eval,
        }
        logger.info(
            "Seed %d -> Val NDCG@10: %.5f | HR@10: %.4f | Coverage: %.4f (%.1fs)",
            s,
            tt_results[s]["val_ndcg_at_10"],
            tt_results[s]["val_hr_at_10"],
            tt_results[s]["coverage_at_10"],
            t_elapsed,
        )

    tt_ndcgs = [tt_results[s]["val_ndcg_at_10"] for s in seeds_5]
    tt_hrs = [tt_results[s]["val_hr_at_10"] for s in seeds_5]
    tt_mean_ndcg, tt_std_ndcg = float(np.mean(tt_ndcgs)), float(np.std(tt_ndcgs))
    tt_mean_hr, tt_std_hr = float(np.mean(tt_hrs)), float(np.std(tt_hrs))

    # Champion selection: closest to validation mean
    diffs = [abs(tt_results[s]["val_ndcg_at_10"] - tt_mean_ndcg) for s in seeds_5]
    champion_tt_seed = seeds_5[int(np.argmin(diffs))]
    champion_tt_model = tt_models[champion_tt_seed]
    logger.info(
        "Two-Tower 5-Seed Validation NDCG@10: %.5f ± %.5f | Champion Seed: %d (Val NDCG@10: %.5f)",
        tt_mean_ndcg,
        tt_std_ndcg,
        champion_tt_seed,
        tt_results[champion_tt_seed]["val_ndcg_at_10"],
    )

    # Serialize champion Two-Tower artifact to disk
    champ_tt_run_id = f"phase4_seed_{champion_tt_seed}"
    champ_tt_path = art_tt_dir / champ_tt_run_id
    champion_tt_model.save(champ_tt_path)
    # Save ranks_val
    champ_tt_val_eval = tt_results[champion_tt_seed]["val_eval"]
    champ_tt_ranks = champ_tt_val_eval.per_user.get("ranks")
    if champ_tt_ranks is not None:
        np.save(champ_tt_path / "ranks_val.npy", champ_tt_ranks)
    logger.info("Saved champion Two-Tower artifact to %s", champ_tt_path)

    # ==============================================================================
    # 2. Baselines Sweep
    # ==============================================================================
    logger.info("\n>>> [2/5] Training and Evaluating Baselines...")
    baseline_results: dict[str, Any] = {}

    # Popularity (deterministic)
    pop_m = PopularityRecommender()
    pop_m.fit(data)
    pop_val = evaluate_model(pop_m, data, split="val", k_list=(10,), user_indices=eval_users, seed=42)
    baseline_results["recsys_popularity"] = {
        "val_ndcg_at_10": float(pop_val.metrics.get("ndcg_at_10", 0.0)),
        "val_hr_at_10": float(pop_val.metrics.get("hit_rate_at_10", 0.0)),
        "coverage_at_10": float(pop_val.metrics.get("catalog_coverage_at_10", 0.0)),
        "val_eval": pop_val,
    }
    logger.info("Popularity -> Val NDCG@10: %.5f", baseline_results["recsys_popularity"]["val_ndcg_at_10"])

    # Content (deterministic)
    content_m = ContentRecommender(weight_tags=0.5)
    content_m.fit(data)
    content_val = evaluate_model(content_m, data, split="val", k_list=(10,), user_indices=eval_users, seed=42)
    baseline_results["recsys_content"] = {
        "val_ndcg_at_10": float(content_val.metrics.get("ndcg_at_10", 0.0)),
        "val_hr_at_10": float(content_val.metrics.get("hit_rate_at_10", 0.0)),
        "coverage_at_10": float(content_val.metrics.get("catalog_coverage_at_10", 0.0)),
        "val_eval": content_val,
    }
    logger.info("Content -> Val NDCG@10: %.5f", baseline_results["recsys_content"]["val_ndcg_at_10"])

    # ItemKNN (5 seeds)
    knn_ndcgs = []
    knn_models = {}
    for s in seeds_5:
        knn_inst = ItemKNNRecommender(k_neighbors=100, shrink=10.0)
        knn_inst.fit(data)
        knn_val = evaluate_model(knn_inst, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
        knn_ndcgs.append(float(knn_val.metrics.get("ndcg_at_10", 0.0)))
        knn_models[s] = knn_inst
    baseline_results["recsys_itemknn"] = {
        "val_ndcg_at_10_mean": float(np.mean(knn_ndcgs)),
        "val_ndcg_at_10_std": float(np.std(knn_ndcgs)),
        "seed_ndcgs": knn_ndcgs,
        "val_eval": knn_val,
    }
    logger.info("Item-kNN (5 seeds) -> Val NDCG@10: %.5f ± %.5f", np.mean(knn_ndcgs), np.std(knn_ndcgs))

    # Paired diff: Champion Two-Tower vs Item-kNN on Val
    tt_u_ndcg = champ_tt_val_eval.per_user.get("ndcg_at_10", np.array([]))
    knn_u_ndcg = knn_val.per_user.get("ndcg_at_10", np.array([]))
    tt_vs_knn_diff, tt_vs_knn_lo, tt_vs_knn_hi = paired_bootstrap_diff(tt_u_ndcg, knn_u_ndcg, seed=42)
    logger.info(
        "Two-Tower Champion (Seed %d) vs Item-kNN Val NDCG@10 Paired Diff: %+.5f [%.5f, %.5f]",
        champion_tt_seed,
        tt_vs_knn_diff,
        tt_vs_knn_lo,
        tt_vs_knn_hi,
    )

    # ALS (5 seeds)
    als_ndcgs = []
    als_models = {}
    for s in seeds_5:
        als_inst = ALSRecommender(factors=64, regularization=0.05, alpha=15.0, iterations=15, seed=s)
        als_inst.fit(data)
        als_val = evaluate_model(als_inst, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
        als_ndcgs.append(float(als_val.metrics.get("ndcg_at_10", 0.0)))
        als_models[s] = als_inst
    baseline_results["recsys_als"] = {
        "val_ndcg_at_10_mean": float(np.mean(als_ndcgs)),
        "val_ndcg_at_10_std": float(np.std(als_ndcgs)),
        "seed_ndcgs": als_ndcgs,
        "val_eval": als_val,
    }
    logger.info("ALS (5 seeds) -> Val NDCG@10: %.5f ± %.5f", np.mean(als_ndcgs), np.std(als_ndcgs))

    # ==============================================================================
    # 3. Two-Tower Architectural Ablations (3 seeds: [42, 43, 44])
    # ==============================================================================
    tt_ablation_results: dict[str, Any] = {}
    if skip_ablations:
        logger.info("\n>>> [3/5] Skipping Two-Tower Architectural Ablations (--skip-ablations set)")
        old_sweep_file = results_path / "validation_sweep_reduced_budget.json"
        if old_sweep_file.exists():
            with open(old_sweep_file, "r", encoding="utf-8") as f:
                tt_ablation_results = json.load(f).get("two_tower_ablations", {})
    else:
        logger.info("\n>>> [3/5] Running Two-Tower Architectural Ablations across seeds: %s", seeds_3)
        tt_ablation_defs = [
            ("no_tags", "Without Tag Multi-Hot Projection", {"use_tags": False}),
            ("no_numeric", "Without Standardized Numeric Projection", {"use_numeric": False}),
            ("no_id", "Without Item ID Embedding", {"use_id": False}),
        ]

        for var_key, desc, overrides in tt_ablation_defs:
            var_ndcgs = []
            var_hrs = []
            var_user_metrics = []
            art_abl_dir = data_path / "artifacts" / f"recsys_two_tower_{var_key}"
            art_abl_dir.mkdir(parents=True, exist_ok=True)
            for s in seeds_3:
                s_abl_dir = art_abl_dir / f"seed_{s}"
                if (s_abl_dir / "two_tower.pt").exists():
                    logger.info("Loading cached Two-Tower ablation '%s' Seed %d from %s...", var_key, s, s_abl_dir)
                    v_m = TwoTowerModel.load(s_abl_dir, data=data)
                else:
                    v_cfg = TwoTowerConfig(
                        n_items=data.n_items,
                        text_dim=384,
                        tag_dim=300,
                        numeric_dim=6,
                        id_dim=64,
                        text_proj_dim=64,
                        tag_proj_dim=32,
                        numeric_proj_dim=16,
                        hidden_dim=256,
                        output_dim=64,
                        pool="attn",
                        id_dropout=0.25,
                        proxy_dropout=0.25,
                        no_pop_proxies=True,
                        **overrides,
                    )
                    v_m, _ = train_two_tower_model(
                        data=data,
                        tuning_users=tuning_users,
                        cfg=v_cfg,
                        epochs=epochs_tt,
                        batch_size=2048,
                        lr=2e-3,
                        patience=patience,
                        pairs_per_epoch=pairs_per_epoch,
                        seed=s,
                    )
                    s_abl_dir.mkdir(parents=True, exist_ok=True)
                    v_m.save(s_abl_dir)

                v_eval = evaluate_model(v_m, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
                var_ndcgs.append(float(v_eval.metrics.get("ndcg_at_10", 0.0)))
                var_hrs.append(float(v_eval.metrics.get("hit_rate_at_10", 0.0)))
                if s == 42:
                    var_user_metrics = v_eval.per_user.get("ndcg_at_10", np.array([]))

            # Bootstrap paired diff vs full Two-Tower seed 42
            champ_user_metrics = tt_results[42]["val_eval"].per_user.get("ndcg_at_10", np.array([]))
            mean_diff, lo95, hi95 = paired_bootstrap_diff(var_user_metrics, champ_user_metrics, seed=42)

            tt_ablation_results[var_key] = {
                "description": desc,
                "val_ndcg_at_10_mean": float(np.mean(var_ndcgs)),
                "val_ndcg_at_10_std": float(np.std(var_ndcgs)),
                "val_hr_at_10_mean": float(np.mean(var_hrs)),
                "val_hr_at_10_std": float(np.std(var_hrs)),
                "paired_diff_vs_full": mean_diff,
                "paired_diff_ci": [lo95, hi95],
                "significant_loss": hi95 < 0.0,
            }
            logger.info(
                "Ablation '%s' -> Val NDCG@10: %.5f ± %.5f (diff vs full: %+.5f [%.5f, %.5f], significant: %s)",
                var_key,
                np.mean(var_ndcgs),
                np.std(var_ndcgs),
                mean_diff,
                lo95,
                hi95,
                hi95 < 0.0,
            )

    # ==============================================================================
    # 4. Clean Ranker Multi-Seed Sweep (5 seeds)
    # ==============================================================================
    logger.info("\n>>> [4/5] Training Clean Ranker across 5 seeds: %s", seeds_5)
    stage1_for_ranker = {
        "two_tower": champion_tt_model,
        "itemknn": knn_models[42],
        "als": als_models[42],
        "content": content_m,
    }

    art_r_dir = data_path / "artifacts" / "recsys_two_tower_ranker_clean"
    art_r_dir.mkdir(parents=True, exist_ok=True)

    ranker_results: dict[int, dict[str, Any]] = {}
    ranker_models: dict[int, RankerModel] = {}
    ranker_curves_by_seed: dict[str, list[dict[str, Any]]] = {}
    cached_train_val_by_seed: dict[int, dict[str, Any]] = {}

    for s in seeds_5:
        seed_art_dir = art_r_dir / f"seed_{s}"
        if (seed_art_dir / "ranker_lgb.txt").exists() and (seed_art_dir / "training_curves.json").exists():
            logger.info("--- Loading cached Ranker Seed %d from %s ---", s, seed_art_dir)
            t0 = time.perf_counter()
            r_model = RankerModel.load(seed_art_dir, data=data, models=stage1_for_ranker)
            r_model.name = "recsys_two_tower_ranker_clean"
            with open(seed_art_dir / "training_curves.json", "r", encoding="utf-8") as f:
                r_curves_payload = json.load(f)
            r_curves = r_curves_payload.get("curves", [])
        else:
            logger.info("--- Training Ranker Seed %d (train_stats_only=True, quota candidates) ---", s)
            r_cfg = RankerConfig(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=300,
                min_child_samples=50,
                colsample_bytree=0.8,
                lambdarank_truncation_level=20,
                eval_at=[10],
                seed=s,
                n_jobs=4,
                max_candidates=300,
                train_stats_only=True,
            )
            t0 = time.perf_counter()
            r_model, r_info = train_ranker(
                data=data,
                models=stage1_for_ranker,
                cfg=r_cfg,
                n_ranker_users=n_ranker_users,
                eval_users=eval_users,
                tuning_users=tuning_users,
                feature_columns=list(FEATURE_COLUMNS),
                seed=s,
                persistence_dir=data_path,
            )
            r_model.name = "recsys_two_tower_ranker_clean"
            cached_train_val_by_seed[s] = r_info.get("precomputed_train_val", {})
            r_curves_payload = r_info.get("training_curves", {})
            r_curves = r_curves_payload.get("curves", [])
            seed_art_dir.mkdir(parents=True, exist_ok=True)
            r_model.save(seed_art_dir)
            with open(seed_art_dir / "training_curves.json", "w", encoding="utf-8") as f:
                json.dump(r_curves_payload, f, indent=2)

        ranker_models[s] = r_model
        ranker_curves_by_seed[str(s)] = r_curves

        # Evaluate strictly on validation split
        r_val_eval = evaluate_model(
            model=r_model,
            data=data,
            split="val",
            k_list=(10, 20),
            user_indices=eval_users,
            seed=s,
        )
        t_elapsed = time.perf_counter() - t0

        ranker_results[s] = {
            "val_ndcg_at_10": float(r_val_eval.metrics.get("ndcg_at_10", 0.0)),
            "val_ndcg_at_10_lo95": float(r_val_eval.metrics.get("ndcg_at_10_lo95", 0.0)),
            "val_ndcg_at_10_hi95": float(r_val_eval.metrics.get("ndcg_at_10_hi95", 0.0)),
            "val_hr_at_10": float(r_val_eval.metrics.get("hit_rate_at_10", 0.0)),
            "val_hr_at_10_lo95": float(r_val_eval.metrics.get("hit_rate_at_10_lo95", 0.0)),
            "val_hr_at_10_hi95": float(r_val_eval.metrics.get("hit_rate_at_10_hi95", 0.0)),
            "val_mrr_at_10": float(r_val_eval.metrics.get("mrr_at_10", 0.0)),
            "coverage_at_10": float(r_val_eval.metrics.get("catalog_coverage_at_10", 0.0)),
            "novelty_at_10": float(r_val_eval.metrics.get("novelty_at_10", 0.0)),
            "diversity_at_10": float(r_val_eval.metrics.get("diversity_at_10", 0.0)),
            "snips_ndcg_at_10": float(r_val_eval.metrics.get("snips_ndcg_at_10", 0.0)),
            "snips_ndcg_at_10_lo95": float(r_val_eval.metrics.get("snips_ndcg_at_10_lo95", 0.0)),
            "snips_ndcg_at_10_hi95": float(r_val_eval.metrics.get("snips_ndcg_at_10_hi95", 0.0)),
            "training_time_sec": round(t_elapsed, 2),
            "val_eval": r_val_eval,
        }
        logger.info(
            "Ranker Seed %d -> Val NDCG@10: %.5f | HR@10: %.4f | Coverage: %.4f (%.1fs)",
            s,
            ranker_results[s]["val_ndcg_at_10"],
            ranker_results[s]["val_hr_at_10"],
            ranker_results[s]["coverage_at_10"],
            t_elapsed,
        )

    r_ndcgs = [ranker_results[s]["val_ndcg_at_10"] for s in seeds_5]
    r_hrs = [ranker_results[s]["val_hr_at_10"] for s in seeds_5]
    r_mean_ndcg, r_std_ndcg = float(np.mean(r_ndcgs)), float(np.std(r_ndcgs))
    r_mean_hr, r_std_hr = float(np.mean(r_hrs)), float(np.std(r_hrs))

    # Champion selection for Ranker
    r_diffs = [abs(ranker_results[s]["val_ndcg_at_10"] - r_mean_ndcg) for s in seeds_5]
    champion_r_seed = seeds_5[int(np.argmin(r_diffs))]
    champion_r_model = ranker_models[champion_r_seed]
    logger.info(
        "Ranker 5-Seed Validation NDCG@10: %.5f ± %.5f | Champion Seed: %d (Val NDCG@10: %.5f)",
        r_mean_ndcg,
        r_std_ndcg,
        champion_r_seed,
        ranker_results[champion_r_seed]["val_ndcg_at_10"],
    )

    # Save champion Ranker artifact to disk
    art_r_dir = data_path / "artifacts" / "recsys_two_tower_ranker_clean"
    art_r_dir.mkdir(parents=True, exist_ok=True)
    champ_r_run_id = f"phase4_seed_{champion_r_seed}"
    champ_r_path = art_r_dir / champ_r_run_id
    champion_r_model.save(champ_r_path)
    # Save ranks_val
    champ_r_val_eval = ranker_results[champion_r_seed]["val_eval"]
    champ_r_ranks = champ_r_val_eval.per_user.get("ranks")
    if champ_r_ranks is not None:
        np.save(champ_r_path / "ranks_val.npy", champ_r_ranks)
    logger.info("Saved champion Ranker artifact to %s", champ_r_path)

    # Paired difference: Ranker Champion vs Two-Tower Champion on Val
    r_u_ndcg = champ_r_val_eval.per_user.get("ndcg_at_10", np.array([]))
    tt_u_ndcg = champ_tt_val_eval.per_user.get("ndcg_at_10", np.array([]))
    r_vs_tt_diff, r_vs_tt_lo, r_vs_tt_hi = paired_bootstrap_diff(r_u_ndcg, tt_u_ndcg, seed=42)
    logger.info(
        "Champion Ranker vs Two-Tower Val NDCG@10 Paired Diff: %+.5f [%.5f, %.5f]",
        r_vs_tt_diff,
        r_vs_tt_lo,
        r_vs_tt_hi,
    )

    # ==============================================================================
    # 5. Ranker Feature-Group Ablations (3 seeds: [42, 43, 44])
    # ==============================================================================
    ranker_ablation_results: dict[str, Any] = {}
    if skip_ablations:
        logger.info("\n>>> [5/5] Skipping Ranker Feature-Group Ablations (--skip-ablations set)")
        old_sweep_file = results_path / "validation_sweep_reduced_budget.json"
        if old_sweep_file.exists():
            with open(old_sweep_file, "r", encoding="utf-8") as f:
                ranker_ablation_results = json.load(f).get("ranker_ablations", {})
    else:
        logger.info("\n>>> [5/5] Running Ranker Feature-Group Ablations across seeds: %s", seeds_3)
        source_feats = [
            "source_two_tower_score", "source_itemknn_score", "source_als_score", "source_content_score",
            "rank_two_tower", "rank_itemknn", "rank_als", "rank_content", "in_two_tower", "in_itemknn", "in_als", "in_content"
        ]
        content_sim_feats = ["text_sim_cosine", "tags_sim_jaccard"]
        price_feats = ["price_final", "discount", "price_gap", "abs_price_gap"]
        user_feats = ["history_len", "log_products", "log_reviews", "mean_log_hours"]
        item_quality_feats = [
            "log_train_pop", "popularity_rank_pct", "age_years", "positive_ratio",
            "log_user_reviews", "win", "mac", "linux", "steam_deck", "has_description", "release_year_gap"
        ]

        r_ablation_defs = [
            ("without_source_scores", "Without Multi-Source Retrieval Scores/Ranks", set(source_feats)),
            ("without_content_sim", "Without Content Similarity Features (Cosine & Jaccard)", set(content_sim_feats)),
            ("without_price_features", "Without Price & Discount Features", set(price_feats)),
            ("without_user_features", "Without User Profile & Hours Features", set(user_feats)),
            ("without_item_quality", "Without Item Quality & Popularity Features", set(item_quality_feats)),
            ("without_train_popularity", "Without Train Popularity Features", {"log_train_pop", "popularity_rank_pct"}),
        ]

        for grp_key, grp_desc, excluded in r_ablation_defs:
            active_cols = [f for f in FEATURE_COLUMNS if f not in excluded]
            grp_ndcgs = []
            grp_hrs = []
            seed42_user_metrics = np.array([])

            r_abl_dir = data_path / "artifacts" / f"recsys_two_tower_ranker_{grp_key}"
            r_abl_dir.mkdir(parents=True, exist_ok=True)

            for s in seeds_3:
                s_art_dir = r_abl_dir / f"seed_{s}"
                if (s_art_dir / "ranker_lgb.txt").exists():
                    logger.info("Loading cached ranker ablation '%s' Seed %d from %s...", grp_key, s, s_art_dir)
                    m_abl = RankerModel.load(s_art_dir, data=data, models=stage1_for_ranker)
                    m_abl.name = f"recsys_two_tower_ranker_{grp_key}"
                else:
                    r_cfg = RankerConfig(
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=300,
                        min_child_samples=50,
                        seed=s,
                        n_jobs=4,
                        train_stats_only=True,
                    )
                    pre_tv = cached_train_val_by_seed.get(s)
                    m_abl, abl_info = train_ranker(
                        data=data,
                        models=stage1_for_ranker,
                        cfg=r_cfg,
                        n_ranker_users=n_ranker_users,
                        eval_users=eval_users,
                        tuning_users=tuning_users,
                        feature_columns=active_cols,
                        seed=s,
                        persistence_dir=data_path,
                        precomputed_train_val=pre_tv,
                    )
                    m_abl.name = f"recsys_two_tower_ranker_{grp_key}"
                    s_art_dir.mkdir(parents=True, exist_ok=True)
                    m_abl.save(s_art_dir)
                    if s not in cached_train_val_by_seed and "precomputed_train_val" in abl_info:
                        cached_train_val_by_seed[s] = abl_info["precomputed_train_val"]

                val_ev = evaluate_model(m_abl, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
                grp_ndcgs.append(float(val_ev.metrics.get("ndcg_at_10", 0.0)))
                grp_hrs.append(float(val_ev.metrics.get("hit_rate_at_10", 0.0)))
                if s == 42:
                    seed42_user_metrics = val_ev.per_user.get("ndcg_at_10", np.array([]))

            full_u_metrics = ranker_results[42]["val_eval"].per_user.get("ndcg_at_10", np.array([]))
            m_diff, lo95, hi95 = paired_bootstrap_diff(seed42_user_metrics, full_u_metrics, seed=42)

            ranker_ablation_results[grp_key] = {
                "description": grp_desc,
                "active_features_count": len(active_cols),
                "val_ndcg_at_10_mean": float(np.mean(grp_ndcgs)),
                "val_ndcg_at_10_std": float(np.std(grp_ndcgs)),
                "val_hr_at_10_mean": float(np.mean(grp_hrs)),
                "val_hr_at_10_std": float(np.std(grp_hrs)),
                "paired_diff_vs_full": m_diff,
                "paired_diff_ci": [lo95, hi95],
                "significant_loss": hi95 < 0.0,
            }
            logger.info(
                "Ranker Ablation '%s' (%d feats) -> Val NDCG@10: %.5f ± %.5f (diff vs full: %+.5f [%.5f, %.5f])",
                grp_key,
                len(active_cols),
                np.mean(grp_ndcgs),
                np.std(grp_ndcgs),
                m_diff,
                lo95,
                hi95,
            )

    # ==============================================================================
    # 6. Build and Save docs/results/training_curves.json
    # ==============================================================================
    logger.info("\n>>> Generating docs/results/training_curves.json...")
    curves_payload_data = {
        "two_tower_curves": {
            "curves": tt_curves_by_seed[str(champion_tt_seed)],
            "history": tt_curves_by_seed[str(champion_tt_seed)],
            "seeds": tt_curves_by_seed,
            "champion_seed": champion_tt_seed,
        },
        "ranker_curves": {
            "curves": ranker_curves_by_seed[str(champion_r_seed)],
            "history": ranker_curves_by_seed[str(champion_r_seed)],
            "seeds": ranker_curves_by_seed,
            "champion_seed": champion_r_seed,
        },
    }

    curves_payload = build_training_curves_payload(
        dataset_version=data.dataset_version,
        two_tower_curves=curves_payload_data["two_tower_curves"],
        ranker_curves=curves_payload_data["ranker_curves"],
        n_eval_users=len(eval_users),
        seed=42,
    )
    save_report_json(results_path / "training_curves.json", curves_payload)
    logger.info("Saved docs/results/training_curves.json successfully.")

    # ==============================================================================
    # 7. Build and Save Validation Sweep Summary
    # ==============================================================================
    sweep_summary = {
        "metadata": {
            "dataset_version": data.dataset_version,
            "n_eval_users": len(eval_users),
            "n_tuning_users": len(tuning_users),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_sweep_time_sec": round(time.perf_counter() - t_start, 2),
        },
        "two_tower_clean_5seeds": {
            "mean_ndcg_at_10": tt_mean_ndcg,
            "std_ndcg_at_10": tt_std_ndcg,
            "mean_hr_at_10": tt_mean_hr,
            "std_hr_at_10": tt_std_hr,
            "champion_seed": champion_tt_seed,
            "paired_diff_vs_itemknn": {
                "mean": tt_vs_knn_diff,
                "lo95": tt_vs_knn_lo,
                "hi95": tt_vs_knn_hi,
            },
            "seed_results": {s: {k: v for k, v in res.items() if k != "val_eval"} for s, res in tt_results.items()},
        },
        "ranker_clean_5seeds": {
            "mean_ndcg_at_10": r_mean_ndcg,
            "std_ndcg_at_10": r_std_ndcg,
            "mean_hr_at_10": r_mean_hr,
            "std_hr_at_10": r_std_hr,
            "champion_seed": champion_r_seed,
            "paired_diff_vs_tt_champion": {
                "mean": r_vs_tt_diff,
                "lo95": r_vs_tt_lo,
                "hi95": r_vs_tt_hi,
            },
            "seed_results": {s: {k: v for k, v in res.items() if k != "val_eval"} for s, res in ranker_results.items()},
        },
        "baselines": {
            k: {k2: v2 for k2, v2 in v.items() if k2 != "val_eval"} for k, v in baseline_results.items()
        },
        "two_tower_ablations": tt_ablation_results,
        "ranker_ablations": ranker_ablation_results,
    }

    sweep_file = results_path / "validation_sweep.json"
    with open(sweep_file, "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)
    logger.info("Saved validation sweep summary to %s", sweep_file)

    logger.info("\n================================================================================")
    logger.info("Phase 4 Multi-Seed Sweep Finished in %.1fs", time.perf_counter() - t_start)
    logger.info("Two-Tower Clean: %.5f ± %.5f (Champion: Seed %d, %.5f)", tt_mean_ndcg, tt_std_ndcg, champion_tt_seed, tt_results[champion_tt_seed]["val_ndcg_at_10"])
    logger.info("Ranker Clean:    %.5f ± %.5f (Champion: Seed %d, %.5f)", r_mean_ndcg, r_std_ndcg, champion_r_seed, ranker_results[champion_r_seed]["val_ndcg_at_10"])
    logger.info("Ranker vs TT Paired Diff: %+.5f [%.5f, %.5f]", r_vs_tt_diff, r_vs_tt_lo, r_vs_tt_hi)
    logger.info("================================================================================")

    return sweep_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 Validation Multi-Seed Sweep")
    parser.add_argument("--data-dir", type=str, default="data/recsys")
    parser.add_argument("--results-dir", type=str, default="docs/results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--pairs-per-epoch", type=lambda x: None if str(x).lower() in ("none", "null", "") else int(x), default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-ranker-users", type=int, default=15000)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    run_sweep(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs_tt=args.epochs,
        pairs_per_epoch=args.pairs_per_epoch,
        patience=args.patience,
        n_ranker_users=args.n_ranker_users,
        skip_ablations=args.skip_ablations,
    )
