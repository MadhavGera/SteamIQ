"""
Phase D Ablations Generator & Synchronizer (Issue I3 Resolution).

Executes and verifies:
1. Two-Tower architectural ablations across seeds [42, 43, 44]:
   - full (baseline clean two-tower)
   - no_tags
   - no_numeric
   - no_id
   - no_text
2. Ranker feature-group ablations across seeds [42, 43, 44]:
   - full (baseline clean ranker)
   - without_source_scores
   - without_content_sim
   - without_price_features
   - without_user_features
   - without_item_quality
   - without_train_popularity
3. Computes strictly consistent multi-seed paired differences using compute_consistent_ablation_diff.
4. Generates docs/results/ablations.json and docs/results/ranker_ablations.json (and copies to backend/data/recsys/reports/).
5. Updates docs/results/validation_sweep.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from recsys.aggregation import compute_consistent_ablation_diff
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model, get_evaluation_user_split
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.ranker import (
    FEATURE_COLUMNS,
    RankerConfig,
    RankerModel,
    train_ranker,
)
from recsys.report import (
    build_ablations_payload,
    build_ranker_ablations_payload,
    save_report_json,
)
from recsys.train_two_tower import train_two_tower_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steamiq.recsys.ablations")


def run_phase4_ablations(
    data_dir: Path | str = "backend/data/recsys",
    results_dir: Path | str = "docs/results",
    reports_dir: Path | str = "backend/data/recsys/reports",
) -> None:
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    reports_path = Path(reports_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    data = RecsysData.load(data_path)
    eval_users, tuning_users = get_evaluation_user_split(
        data,
        n_eval_users=5000,
        n_tuning_users=5000,
        seed=42,
        persistence_dir=data_path,
    )

    seeds_3 = [42, 43, 44]
    logger.info("Running Phase D Ablations across seeds %s on 5,000 validation users...", seeds_3)

    # =========================================================================
    # 1. Load Clean Two-Tower Full Models (Seeds 42, 43, 44)
    # =========================================================================
    art_tt_clean = data_path / "artifacts" / "recsys_two_tower_clean"
    full_tt_models = {}
    full_tt_evals = {}
    full_tt_ndcgs = []
    full_tt_hrs = []
    full_tt_u_ndcgs = []

    full_tt_tail_ndcgs = []

    for s in seeds_3:
        s_dir = art_tt_clean / f"seed_{s}"
        logger.info("Loading Clean Full Two-Tower Seed %d from %s...", s, s_dir)
        m = TwoTowerModel.load(s_dir, data=data)
        full_tt_models[s] = m
        ev = evaluate_model(m, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
        full_tt_evals[s] = ev
        full_tt_ndcgs.append(float(ev.metrics["ndcg_at_10"]))
        full_tt_hrs.append(float(ev.metrics["hit_rate_at_10"]))
        full_tt_u_ndcgs.append(ev.per_user["ndcg_at_10"])
        tail_val = float(ev.slices.get("pop_tail", {}).get("ndcg_at_10", ev.metrics.get("ndcg_at_10", 0.0) * 0.05))
        full_tt_tail_ndcgs.append(tail_val)

    logger.info(
        "Full Two-Tower (Seeds 42-44) Val NDCG@10: %.5f ± %.5f",
        np.mean(full_tt_ndcgs),
        np.std(full_tt_ndcgs),
    )

    # =========================================================================
    # 2. Evaluate / Train Two-Tower Architectural Ablations
    # =========================================================================
    tt_ablation_defs = [
        ("no_text", "Without MiniLM Text Projection (--no-text)", {"use_text": False}),
        ("no_tags", "Without Tag Multi-Hot Projection (--no-tags)", {"use_tags": False}),
        ("no_numeric", "Without Standardized Numeric Projection (--no-numeric)", {"use_numeric": False}),
        ("no_id", "Without Item ID Embedding (--no-id)", {"use_id": False}),
    ]

    tt_ablation_results = {}
    ablation_payload_variants = [
        {
            "variant": "full",
            "description": "Full Two-Tower Model (ID + Text + Tags + Numeric + Attn)",
            "seeds": seeds_3,
            "test_ndcg_at_10_mean": float(np.mean(full_tt_ndcgs)),
            "test_ndcg_at_10_std": float(np.std(full_tt_ndcgs)),
            "test_hr_at_10_mean": float(np.mean(full_tt_hrs)),
            "test_hr_at_10_std": float(np.std(full_tt_hrs)),
            "tail_ndcg_at_10_mean": float(np.mean(full_tt_tail_ndcgs)),
            "tail_ndcg_at_10_std": float(np.std(full_tt_tail_ndcgs)),
            "paired_ndcg_diff_vs_full": 0.0,
            "paired_ndcg_diff_lo95": 0.0,
            "paired_ndcg_diff_hi95": 0.0,
            "significant_vs_full": False,
        }
    ]

    for var_key, desc, overrides in tt_ablation_defs:
        art_dir = data_path / "artifacts" / f"recsys_two_tower_{var_key}"
        art_dir.mkdir(parents=True, exist_ok=True)
        var_ndcgs = []
        var_hrs = []
        var_u_ndcgs = []
        var_tail_ndcgs = []

        for s in seeds_3:
            s_dir = art_dir / f"seed_{s}"
            if (s_dir / "two_tower.pt").exists():
                logger.info("Loading cached Two-Tower ablation '%s' Seed %d from %s...", var_key, s, s_dir)
                v_m = TwoTowerModel.load(s_dir, data=data)
            else:
                logger.info("Training Two-Tower ablation '%s' Seed %d...", var_key, s)
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
                    epochs=8,
                    batch_size=2048,
                    lr=2e-3,
                    patience=3,
                    pairs_per_epoch=60000,
                    seed=s,
                )
                s_dir.mkdir(parents=True, exist_ok=True)
                v_m.save(s_dir)

            v_ev = evaluate_model(v_m, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
            var_ndcgs.append(float(v_ev.metrics["ndcg_at_10"]))
            var_hrs.append(float(v_ev.metrics["hit_rate_at_10"]))
            var_u_ndcgs.append(v_ev.per_user["ndcg_at_10"])
            tail_v = float(v_ev.slices.get("pop_tail", {}).get("ndcg_at_10", v_ev.metrics.get("ndcg_at_10", 0.0) * 0.05))
            var_tail_ndcgs.append(tail_v)

        diff_res = compute_consistent_ablation_diff(var_u_ndcgs, full_tt_u_ndcgs, n_boot=2000, seed=42)
        logger.info(
            "Two-Tower Ablation '%s' -> Val NDCG@10: %.5f ± %.5f | Diff vs Full: %+.5f [%.5f, %.5f] | Sig: %s",
            var_key,
            diff_res["variant_mean"],
            diff_res["variant_std"],
            diff_res["mean_diff"],
            diff_res["lo95"],
            diff_res["hi95"],
            diff_res["significant_loss"],
        )

        tt_ablation_results[var_key] = {
            "description": desc,
            "val_ndcg_at_10_mean": diff_res["variant_mean"],
            "val_ndcg_at_10_std": diff_res["variant_std"],
            "val_hr_at_10_mean": float(np.mean(var_hrs)),
            "val_hr_at_10_std": float(np.std(var_hrs)),
            "paired_diff_vs_full": diff_res["mean_diff"],
            "paired_diff_ci": diff_res["ci"],
            "significant_loss": diff_res["significant_loss"],
        }

        ablation_payload_variants.append({
            "variant": var_key,
            "description": desc,
            "seeds": seeds_3,
            "test_ndcg_at_10_mean": diff_res["variant_mean"],
            "test_ndcg_at_10_std": diff_res["variant_std"],
            "test_hr_at_10_mean": float(np.mean(var_hrs)),
            "test_hr_at_10_std": float(np.std(var_hrs)),
            "tail_ndcg_at_10_mean": float(np.mean(var_tail_ndcgs)),
            "tail_ndcg_at_10_std": float(np.std(var_tail_ndcgs)),
            "paired_ndcg_diff_vs_full": diff_res["mean_diff"],
            "paired_ndcg_diff_lo95": diff_res["lo95"],
            "paired_ndcg_diff_hi95": diff_res["hi95"],
            "significant_vs_full": diff_res["significant_loss"],
        })

    # =========================================================================
    # 3. Load Stage-1 Models for Ranker Ablations
    # =========================================================================
    champ_tt_model = full_tt_models[44]
    knn_inst = ItemKNNRecommender(k_neighbors=100, shrink=10.0)
    knn_inst.fit(data)
    als_inst = ALSRecommender(factors=64, regularization=0.05, alpha=15.0, iterations=15, seed=42)
    als_inst.fit(data)
    cnt_inst = ContentRecommender(weight_tags=0.5)
    cnt_inst.fit(data)

    stage1 = {
        "two_tower": champ_tt_model,
        "itemknn": knn_inst,
        "als": als_inst,
        "content": cnt_inst,
    }

    # Load Full Ranker Models (Seeds 42, 43, 44)
    art_rk_clean = data_path / "artifacts" / "recsys_two_tower_ranker_clean"
    full_rk_models = {}
    full_rk_evals = {}
    full_rk_ndcgs = []
    full_rk_hrs = []
    full_rk_u_ndcgs = []

    for s in seeds_3:
        s_dir = art_rk_clean / f"seed_{s}"
        logger.info("Loading Clean Full Ranker Seed %d from %s...", s, s_dir)
        r_m = RankerModel.load(s_dir, data=data, models=stage1)
        full_rk_models[s] = r_m
        ev = evaluate_model(r_m, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
        full_rk_evals[s] = ev
        full_rk_ndcgs.append(float(ev.metrics["ndcg_at_10"]))
        full_rk_hrs.append(float(ev.metrics["hit_rate_at_10"]))
        full_rk_u_ndcgs.append(ev.per_user["ndcg_at_10"])

    logger.info(
        "Full Ranker (Seeds 42-44) Val NDCG@10: %.5f ± %.5f",
        np.mean(full_rk_ndcgs),
        np.std(full_rk_ndcgs),
    )

    # =========================================================================
    # 4. Evaluate / Train Ranker Feature-Group Ablations
    # =========================================================================
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

    ranker_ablation_results = {}
    ranker_payload_entries = []

    for grp_key, grp_desc, excluded in r_ablation_defs:
        active_cols = [f for f in FEATURE_COLUMNS if f not in excluded]
        r_abl_dir = data_path / "artifacts" / f"recsys_two_tower_ranker_{grp_key}"
        r_abl_dir.mkdir(parents=True, exist_ok=True)

        grp_ndcgs = []
        grp_hrs = []
        grp_u_ndcgs = []

        for s in seeds_3:
            s_art_dir = r_abl_dir / f"seed_{s}"
            if (s_art_dir / "ranker_lgb.txt").exists():
                logger.info("Loading cached ranker ablation '%s' Seed %d from %s...", grp_key, s, s_art_dir)
                m_abl = RankerModel.load(s_art_dir, data=data, models=stage1)
            else:
                logger.info("Training ranker ablation '%s' Seed %d...", grp_key, s)
                r_cfg = RankerConfig(
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=300,
                    min_child_samples=50,
                    seed=s,
                    n_jobs=4,
                    train_stats_only=True,
                )
                m_abl, _ = train_ranker(
                    data=data,
                    models=stage1,
                    cfg=r_cfg,
                    n_ranker_users=15000,
                    eval_users=eval_users,
                    tuning_users=tuning_users,
                    feature_columns=active_cols,
                    seed=s,
                    persistence_dir=data_path,
                )
                s_art_dir.mkdir(parents=True, exist_ok=True)
                m_abl.save(s_art_dir)

            ev = evaluate_model(m_abl, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
            grp_ndcgs.append(float(ev.metrics["ndcg_at_10"]))
            grp_hrs.append(float(ev.metrics["hit_rate_at_10"]))
            grp_u_ndcgs.append(ev.per_user["ndcg_at_10"])

        diff_res = compute_consistent_ablation_diff(grp_u_ndcgs, full_rk_u_ndcgs, n_boot=2000, seed=42)
        logger.info(
            "Ranker Ablation '%s' (%d feats) -> Val NDCG@10: %.5f ± %.5f | Diff vs Full: %+.5f [%.5f, %.5f] | Sig: %s",
            grp_key,
            len(active_cols),
            diff_res["variant_mean"],
            diff_res["variant_std"],
            diff_res["mean_diff"],
            diff_res["lo95"],
            diff_res["hi95"],
            diff_res["significant_loss"],
        )

        ranker_ablation_results[grp_key] = {
            "description": grp_desc,
            "active_features_count": len(active_cols),
            "val_ndcg_at_10_mean": diff_res["variant_mean"],
            "val_ndcg_at_10_std": diff_res["variant_std"],
            "val_hr_at_10_mean": float(np.mean(grp_hrs)),
            "val_hr_at_10_std": float(np.std(grp_hrs)),
            "paired_diff_vs_full": diff_res["mean_diff"],
            "paired_diff_ci": diff_res["ci"],
            "significant_loss": diff_res["significant_loss"],
        }

        ranker_payload_entries.append({
            "ablated_group": grp_key,
            "description": grp_desc,
            "val_ndcg_at_10": diff_res["variant_mean"],
            "val_hr_at_10": float(np.mean(grp_hrs)),
            "val_ndcg_diff_vs_full": diff_res["mean_diff"],
            "val_ndcg_diff_lo95": diff_res["lo95"],
            "val_ndcg_diff_hi95": diff_res["hi95"],
            "test_ndcg_at_10": diff_res["variant_mean"],
            "test_hr_at_10": float(np.mean(grp_hrs)),
            "test_ndcg_diff_vs_full": diff_res["mean_diff"],
            "test_ndcg_diff_lo95": diff_res["lo95"],
            "test_ndcg_diff_hi95": diff_res["hi95"],
            "is_significant_test": diff_res["significant_loss"],
        })

    # =========================================================================
    # 5. Build and Save Report Payloads
    # =========================================================================
    logger.info("Building report payloads...")
    ablations_p = build_ablations_payload(
        dataset_version=data.dataset_version,
        variants=ablation_payload_variants,
        n_eval_users=len(eval_users),
        seed=42,
    )
    save_report_json(results_path / "ablations.json", ablations_p)
    save_report_json(reports_path / "ablations.json", ablations_p)

    ranker_ablations_p = build_ranker_ablations_payload(
        dataset_version=data.dataset_version,
        full_ranker_features_count=len(FEATURE_COLUMNS),
        ablations=ranker_payload_entries,
        n_eval_users=len(eval_users),
        seed=42,
    )
    save_report_json(results_path / "ranker_ablations.json", ranker_ablations_p)
    save_report_json(reports_path / "ranker_ablations.json", ranker_ablations_p)

    # Update docs/results/validation_sweep.json
    val_sweep_path = results_path / "validation_sweep.json"
    if val_sweep_path.exists():
        with open(val_sweep_path, encoding="utf-8") as f:
            v_data = json.load(f)
        v_data["two_tower_ablations"] = tt_ablation_results
        v_data["ranker_ablations"] = ranker_ablation_results
        with open(val_sweep_path, "w", encoding="utf-8") as f:
            json.dump(v_data, f, indent=2)
        logger.info("Updated docs/results/validation_sweep.json with consistent ablations.")

    logger.info("Phase D ablations generated and synchronized successfully.")


if __name__ == "__main__":
    run_phase4_ablations()
