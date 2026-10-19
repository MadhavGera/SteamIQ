"""
Phase U: Train and Evaluate Clean Ranker Without User Profile Features.

Protocol (Validation-Only, 5 Seeds [42, 43, 44, 45, 46]):
  1. Feature set excludes user profile features:
     ['history_len', 'log_products', 'log_reviews', 'mean_log_hours']
     Retains 31 interaction and item features.
  2. Trains any missing seeds using exact clean protocol:
     - 15,000 disjoint ranker users
     - LightGBM LambdaMART ranker with NDCG@10 optimization
     - train_stats_only = True
  3. Evaluates all 5 seeds on 5,000 validation users.
  4. Computes seed-matched paired bootstrap differences vs Full Clean Ranker.
  5. Computes paired bootstrap differences vs Two-Tower Clean Champion.
  6. Outputs docs/results/ranker_without_user_features.json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add backend to sys.path
sys.path.insert(0, str(Path("backend").resolve()))

from recsys.aggregation import aggregate_per_seed_metrics
from recsys.data import RecsysData
from recsys.evaluate import evaluate_model, get_evaluation_user_split
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.two_tower import TwoTowerModel
from recsys.ranker import RankerConfig, RankerModel, train_ranker
from recsys.ranker_features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("steamiq.ranker_without_user_features")


def run_phase_u(
    data_dir: Path | str = "backend/data/recsys",
    docs_dir: Path | str = "docs/results",
    seeds: list[int] | None = None,
    n_ranker_users: int = 15000,
) -> dict[str, Any]:
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    data_path = Path(data_dir)
    docs_path = Path(docs_dir)
    reports_dir = data_path / "reports"
    docs_path.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = RecsysData.load(data_path)
    eval_users, tuning_users = get_evaluation_user_split(
        data, n_eval_users=5000, n_tuning_users=5000, seed=42, persistence_dir=data_path
    )

    # 1. Load Stage 1 Retriever Models
    tt_champ_path = data_path / "artifacts" / "recsys_two_tower_clean" / "seed_44"
    logger.info("Loading Clean Two-Tower Champion from %s...", tt_champ_path)
    tt_champ = TwoTowerModel.load(tt_champ_path, data=data)

    itemknn_path = data_path / "artifacts" / "recsys_itemknn"
    if itemknn_path.exists() and any(itemknn_path.iterdir()):
        latest_knn = sorted([d for d in itemknn_path.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[0]
        iknn = ItemKNNRecommender.load(latest_knn)
    else:
        iknn = ItemKNNRecommender(k=50)
        iknn.fit(data)

    als_path = data_path / "artifacts" / "recsys_als"
    if als_path.exists() and any(als_path.iterdir()):
        latest_als = sorted([d for d in als_path.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[0]
        als_m = ALSRecommender.load(latest_als)
    else:
        als_m = ALSRecommender(factors=64, iterations=15, regularization=0.05, seed=42)
        als_m.fit(data)

    cnt_path = data_path / "artifacts" / "recsys_content"
    if cnt_path.exists() and any(cnt_path.iterdir()):
        latest_cnt = sorted([d for d in cnt_path.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[0]
        cnt_m = ContentRecommender.load(latest_cnt)
    else:
        cnt_m = ContentRecommender(weight_tags=0.5)
        cnt_m.fit(data)

    stage1_models = {
        "two_tower": tt_champ,
        "itemknn": iknn,
        "als": als_m,
        "content": cnt_m,
    }

    # Evaluate Two-Tower Champion on Validation
    logger.info("Evaluating Two-Tower Champion on Validation Split...")
    tt_val_res = evaluate_model(tt_champ, data, split="val", k_list=(10,), user_indices=eval_users, seed=42)
    tt_val_ndcg = float(tt_val_res.metrics["ndcg_at_10"])
    tt_u_ndcg = tt_val_res.per_user["ndcg_at_10"]

    # 2. Define Features Without User Features
    user_features_to_exclude = {"history_len", "log_products", "log_reviews", "mean_log_hours"}
    active_cols = [f for f in FEATURE_COLUMNS if f not in user_features_to_exclude]
    logger.info("Active feature count: %d / %d (excluded %s)", len(active_cols), len(FEATURE_COLUMNS), user_features_to_exclude)

    # 3. Train / Load 5 Seeds of Ranker Without User Features
    out_art_dir = data_path / "artifacts" / "recsys_two_tower_ranker_without_user_features"
    out_art_dir.mkdir(parents=True, exist_ok=True)

    full_art_dir = data_path / "artifacts" / "recsys_two_tower_ranker_clean"

    seed_results: list[dict[str, Any]] = []
    paired_diffs_vs_full: list[dict[str, Any]] = []
    paired_diffs_vs_tt: list[dict[str, Any]] = []

    for s in seeds:
        seed_dir = out_art_dir / f"seed_{s}"
        full_seed_dir = full_art_dir / f"seed_{s}"

        if (seed_dir / "ranker_lgb.txt").exists():
            logger.info("Loading cached ranker without user features Seed %d from %s...", s, seed_dir)
            model = RankerModel.load(seed_dir, data=data, models=stage1_models)
            model.name = "recsys_two_tower_ranker_without_user_features"
        else:
            logger.info("Training clean ranker without user features Seed %d...", s)
            cfg = RankerConfig(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=300,
                min_child_samples=50,
                seed=s,
                n_jobs=4,
                train_stats_only=True,
            )
            model, _ = train_ranker(
                data=data,
                models=stage1_models,
                cfg=cfg,
                n_ranker_users=n_ranker_users,
                eval_users=eval_users,
                tuning_users=tuning_users,
                feature_columns=active_cols,
                seed=s,
                persistence_dir=data_path,
            )
            model.name = "recsys_two_tower_ranker_without_user_features"
            seed_dir.mkdir(parents=True, exist_ok=True)
            model.save(seed_dir)

        # Evaluate on validation split
        logger.info("Evaluating Seed %d on validation users (N=5,000)...", s)
        ev_res = evaluate_model(model, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
        val_ndcg = float(ev_res.metrics["ndcg_at_10"])
        val_hr = float(ev_res.metrics["hit_rate_at_10"])
        u_ndcg = ev_res.per_user["ndcg_at_10"]

        m_dict = {k: v for k, v in ev_res.metrics.items() if not k.endswith("_ranks")}
        m_dict["seed"] = s
        seed_results.append(m_dict)

        # Paired difference vs Full Clean Ranker (seed-matched)
        if (full_seed_dir / "ranker_lgb.txt").exists():
            full_model = RankerModel.load(full_seed_dir, data=data, models=stage1_models)
            full_ev = evaluate_model(full_model, data, split="val", k_list=(10,), user_indices=eval_users, seed=s)
            full_u_ndcg = full_ev.per_user["ndcg_at_10"]

            diff_m, lo, hi = paired_bootstrap_diff(u_ndcg, full_u_ndcg, n_boot=1000, seed=42)
            paired_diffs_vs_full.append({
                "seed": s,
                "diff_mean": diff_m,
                "diff_lo95": lo,
                "diff_hi95": hi,
                "significant": bool(lo > 0.0 or hi < 0.0),
                "favors_full": bool(hi < 0.0),
                "favors_without_user": bool(lo > 0.0),
            })
            logger.info("Seed %d Paired Diff vs Full Clean Ranker: %+.5f [%.5f, %.5f]", s, diff_m, lo, hi)

        # Paired difference vs Two-Tower Clean Champion
        tt_diff_m, tt_lo, tt_hi = paired_bootstrap_diff(u_ndcg, tt_u_ndcg, n_boot=1000, seed=42)
        paired_diffs_vs_tt.append({
            "seed": s,
            "diff_mean": tt_diff_m,
            "diff_lo95": tt_lo,
            "diff_hi95": tt_hi,
            "beats_two_tower": bool(tt_lo > 0.0),
            "trails_two_tower": bool(tt_hi < 0.0),
        })
        logger.info("Seed %d Paired Diff vs Two-Tower Champion: %+.5f [%.5f, %.5f]", s, tt_diff_m, tt_lo, tt_hi)

    # 4. Multi-Seed Aggregation
    agg_res = aggregate_per_seed_metrics(seed_results, deterministic=False)
    mean_ndcg = agg_res["summary"]["ndcg_at_10"]["mean"]
    std_ndcg = agg_res["summary"]["ndcg_at_10"]["std"]

    # Mean paired differences across 5 seeds
    mean_diff_vs_full = float(np.mean([d["diff_mean"] for d in paired_diffs_vs_full]))
    mean_diff_vs_tt = float(np.mean([d["diff_mean"] for d in paired_diffs_vs_tt]))

    # Load validation sweep for baseline reference numbers
    vs_file = docs_path / "validation_sweep.json"
    vs_data = json.loads(vs_file.read_text(encoding="utf-8")) if vs_file.exists() else {}
    full_rk_mean = vs_data.get("ranker_clean_5seeds", {}).get("mean_ndcg_at_10", 0.05860)
    full_rk_std = vs_data.get("ranker_clean_5seeds", {}).get("std_ndcg_at_10", 0.00198)
    tt_val_mean = vs_data.get("two_tower_clean_5seeds", {}).get("mean_ndcg_at_10", 0.06126)
    tt_val_std = vs_data.get("two_tower_clean_5seeds", {}).get("std_ndcg_at_10", 0.00103)

    payload = {
        "metadata": {
            "report_key": "ranker_without_user_features",
            "study_type": "exploratory_ranker_ablation_validation_only",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "excluded_features": sorted(list(user_features_to_exclude)),
            "active_features_count": len(active_cols),
            "n_eval_users": len(eval_users),
            "split": "val",
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "aggregate": agg_res,
        "two_tower_val_ndcg": tt_val_ndcg,
        "per_seed_validation": seed_results,
        "seed_matched_paired_diff_vs_full_ranker": {
            "mean_diff": mean_diff_vs_full,
            "per_seed": paired_diffs_vs_full,
        },
        "paired_diff_vs_two_tower": {
            "mean_diff": mean_diff_vs_tt,
            "per_seed": paired_diffs_vs_tt,
        },
        "summary": {
            "val_ndcg_at_10_mean": mean_ndcg,
            "val_ndcg_at_10_std": std_ndcg,
            "full_clean_ranker_val_ndcg_mean": full_rk_mean,
            "two_tower_val_ndcg_mean": tt_val_mean,
            "delta_vs_full_clean_ranker": mean_diff_vs_full,
            "delta_vs_two_tower": mean_diff_vs_tt,
            "beats_two_tower": bool(mean_diff_vs_tt > 0.0),
        },
    }

    for p in [docs_path / "ranker_without_user_features.json", reports_dir / "ranker_without_user_features.json"]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved ranker without user features report to %s", p)

    print("\n" + "=" * 80)
    print("PHASE U: RANKER WITHOUT USER FEATURES RESULTS (5 Seeds on Validation Split)")
    print("=" * 80)
    print(f"Ranker Without User Features (5 Seeds): NDCG@10 = {mean_ndcg:.5f} ± {std_ndcg:.5f}")
    print(f"Full Clean Ranker (5 Seeds):            NDCG@10 = {full_rk_mean:.5f} ± {full_rk_std:.5f}")
    print(f"Two-Tower Clean Champion:               NDCG@10 = {tt_val_ndcg:.5f}")
    print(f"Paired Diff vs Full Clean Ranker:       {mean_diff_vs_full:+.5f}")
    print(f"Paired Diff vs Two-Tower Champion:      {mean_diff_vs_tt:+.5f}")
    print(f"Does Ranker beat Two-Tower without user features? {'YES' if mean_diff_vs_tt > 0 else 'NO'}")
    print("=" * 80 + "\n")

    return payload


if __name__ == "__main__":
    run_phase_u()
