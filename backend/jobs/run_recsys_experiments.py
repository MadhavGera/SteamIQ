"""
Batch Job: Run Recommender System Experiments & Evidence Generation — Corrected & Extended Suite.

Executes:
  1. Two-Tower Architectural Ablations (full, --no-text, --no-tags, --no-numeric, --no-id, pool mean, id_dropout 0)
     with multi-seed (42, 43, 44) variance analysis and paired bootstrap differences vs full seed 42.
  2. Ranker Multi-Seed Robustness (5 seeds: 42, 43, 44, 45, 46) & 8 Feature-Group Ablations
     (without source scores, without content-similarity, without price features, without user features,
      without item-quality features, 3-source ranker, without all-time item stats, without train popularity)
     evaluating paired differences against the full ranker of the same seed.
  3. Slices Paired Bootstrap Significance Tests across popularity and history-length buckets.
  4. Cold-Start Studies:
     - Part A: Sparse-target subgroup (<= 10 train positives, including random scorer reference).
     - Part B: Three fair new-item simulations (C1 old setup, C2 fair random, C3 fair bottom 50% pop).
     - Part C: True Item Holdout Study (10% torso/tail items completely removed from training pairs/histories,
       Two-Tower retrained from scratch).
  5. Beyond-Accuracy Catalog Metrics & Masked vs Unmasked Coverage Mystery Analysis.
  6. MMR Diversity Re-Ranking Trade-Off (lambda in {1.0, 0.8, 0.6}) matching ranker test NDCG@10 at lambda=1.0.
  7. Machine-Readable Report Serialization & DB Upsert to `serving_rec_reports`.

Usage:
  python -m jobs.run_recsys_experiments
  python -m jobs.run_recsys_experiments --quick
  make recsys-experiments
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ModelRun, ServingRecReport
from jobs.base_job import BaseJob
from recsys.coldstart import (
    evaluate_item_holdout_study,
    get_item_holdout_split,
    run_full_cold_start_study,
)
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import (
    EvaluationResult,
    _build_user_interaction_maps,
    compute_metrics_from_ranks,
    evaluate_model,
    get_evaluation_user_split,
)
from recsys.metrics import (
    bootstrap_ci,
    catalog_coverage_at_k,
    diversity_at_k,
    get_top_k_indices,
    gini_coefficient_at_k,
    head_share_at_k,
    mmr_rerank,
    ndcg_at_k,
    novelty_at_k,
    paired_bootstrap_diff,
)
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerConfig, TwoTowerModel
from recsys.ranker import RankerConfig, RankerModel, train_ranker
from recsys.ranker_features import FEATURE_COLUMNS
from recsys.registry import log_recsys_run
from recsys.report import (
    build_ablations_payload,
    build_beyond_accuracy_payload,
    build_cold_start_payload,
    build_dataset_payload,
    build_leaderboard_payload,
    build_limitations_payload,
    build_ranker_ablations_payload,
    build_shap_global_payload,
    build_slices_payload,
    build_training_curves_payload,
    generate_leaderboard_markdown,
    save_report_json,
)
from recsys.train_two_tower import train_two_tower_model


def _find_latest_artifact_dir(base_dir: Path, model_name: str) -> tuple[Path, str]:
    art_dir = base_dir / "artifacts" / model_name
    if not art_dir.exists():
        raise FileNotFoundError(f"Missing artifact directory for '{model_name}': {art_dir}")
    cand_dirs = [d for d in art_dir.iterdir() if d.is_dir()]
    if not cand_dirs:
        raise FileNotFoundError(f"No serialized run artifacts in {art_dir} for '{model_name}'")
    cand_dirs.sort(key=lambda x: ((x / "ranks_test.npy").exists(), x.stat().st_mtime), reverse=True)
    latest_dir = cand_dirs[0]
    return latest_dir, latest_dir.name


def _make_eval_result_from_ranks(
    ranks: np.ndarray,
    split: str,
    data: RecsysData,
    eval_users: np.ndarray,
    seed: int = 42,
) -> EvaluationResult:
    """Builds an EvaluationResult directly from saved rank array via compute_metrics_from_ranks."""
    split_df = data.val_df if split == "val" else data.test_df
    target_map = dict(zip(split_df["user_idx"].to_numpy(), split_df["item_idx"].to_numpy(), strict=True))
    targets_arr = np.array([int(target_map[u]) for u in eval_users], dtype=np.int64)
    user_train_pos, _ = _build_user_interaction_maps(data)
    if split == "val":
        hist_lens = np.array([len(user_train_pos.get(u, [])) for u in eval_users], dtype=np.int32)
    else:
        val_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
        hist_lens = np.array(
            [len(user_train_pos.get(u, [])) + (1 if u in val_map else 0) for u in eval_users],
            dtype=np.int32,
        )
    metrics, per_user, slices = compute_metrics_from_ranks(
        ranks=ranks,
        top_k_indices=None,
        targets=targets_arr,
        history_lengths=hist_lens,
        data=data,
        k_list=(10, 20),
        seed=seed,
    )
    return EvaluationResult(
        split=split,
        n_users=len(ranks),
        metrics=metrics,
        per_user=per_user,
        slices=slices,
        evaluated_user_indices=eval_users,
    )


class RunRecsysExperimentsJob(BaseJob):
    job_name = "run_recsys_experiments"

    def __init__(
        self,
        *,
        quick: bool = False,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.quick = quick
        self.profile = profile
        self.data_dir = data_dir
        self.seed = seed

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = self.seed

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recsys dataset from %s (quick=%s)...", resolved_data_dir, self.quick)
        data = RecsysData.load(resolved_data_dir)

        # Disjoint evaluation & tuning users
        n_tuning = 1000 if self.quick else (cfg.eval_users if cfg.profile == "dev" else 10000)
        n_eval = 5000 if not self.quick else min(5000, cfg.eval_users)

        eval_users, tuning_users = get_evaluation_user_split(
            data,
            n_eval_users=n_eval,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        self.logger.info("Splits ready: %d evaluation users, %d tuning users", len(eval_users), len(tuning_users))

        # ── 1. Load Existing Baseline Artifacts ──
        self.logger.info("Loading baseline model artifacts...")
        tt_dir, tt_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_two_tower")
        tt_model = TwoTowerModel.load(tt_dir, data=data)

        ranker_dir, ranker_run_id = _find_latest_artifact_dir(resolved_data_dir, "recsys_two_tower_ranker")
        knn_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_itemknn")
        knn_model = ItemKNNRecommender.load(knn_dir)

        als_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_als")
        als_model = ALSRecommender.load(als_dir)

        content_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_content")
        content_model = ContentRecommender.load(content_dir)

        pop_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_popularity")
        pop_model = PopularityRecommender.load(pop_dir)

        stage1_models = {
            "two_tower": tt_model,
            "itemknn": knn_model,
            "als": als_model,
            "content": content_model,
        }
        ranker_model = RankerModel.load(ranker_dir, data=data, models=stage1_models)

        # ── 2. Evaluate All 6 Standard Models for Leaderboard & Slices ──
        self.logger.info("Evaluating all 6 baseline models on evaluation users (%d users)...", len(eval_users))
        all_models = {
            "recsys_two_tower_ranker": ranker_model,
            "recsys_two_tower": tt_model,
            "recsys_itemknn": knn_model,
            "recsys_als": als_model,
            "recsys_popularity": pop_model,
            "recsys_content": content_model,
        }

        eval_val_results: dict[str, EvaluationResult] = {}
        eval_test_results: dict[str, EvaluationResult] = {}

        for m_name, m_inst in all_models.items():
            eval_val_results[m_name] = evaluate_model(
                m_inst, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=seed
            )
            eval_test_results[m_name] = evaluate_model(
                m_inst, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=seed
            )

        # ── 3. Two-Tower Architectural Ablations ──
        print("\n" + "=" * 130)
        print("🔬 Running Two-Tower Architectural Ablations...")
        print("=" * 130)

        tt_ablation_defs = [
            ("full", "Full Two-Tower Model (ID + Text + Tags + Numeric + Attn)", {}, [42, 43, 44]),
            ("no_text", "Without MiniLM Text Projection (--no-text)", {"use_text": False}, [42, 43, 44]),
            ("no_tags", "Without Tag Multi-Hot Projection (--no-tags)", {"use_tags": False}, [42]),
            ("no_numeric", "Without Standardized Numeric Projection (--no-numeric)", {"use_numeric": False}, [42]),
            ("no_id", "Without Item ID Embedding (--no-id)", {"use_id": False}, [42, 43, 44]),
            ("pool_mean", "Masked Mean History Pooling (--pool mean)", {"pool": "mean"}, [42]),
            ("id_dropout_0", "Without ID Dropout (--id-dropout 0.0)", {"id_dropout": 0.0}, [42]),
        ]

        tt_epochs = 3 if self.quick else 15
        tt_patience = 2 if self.quick else (5 if cfg.profile == "dev" else 3)

        ablation_payload_entries: list[dict[str, Any]] = []
        trained_ablation_models: dict[str, TwoTowerModel] = {}
        tt_curves_history: dict[str, Any] = {}

        full_seed42_test_res: EvaluationResult | None = None

        for var_name, desc, overrides, var_seeds in tt_ablation_defs:
            seed_test_ndcgs: list[float] = []
            seed_test_hrs: list[float] = []
            seed_tail_ndcgs: list[float] = []
            seed_test_eval_results: list[EvaluationResult] = []

            art_base = resolved_data_dir / "artifacts" / f"recsys_two_tower_{var_name}"
            cached_runs = (
                sorted([d for d in art_base.iterdir() if d.is_dir() and (d / "ranks_test.npy").exists()], key=lambda x: x.stat().st_mtime)
                if art_base.exists()
                else []
            )

            for s_idx, s in enumerate(var_seeds):
                if s_idx < len(cached_runs):
                    cand_run_dir = cached_runs[s_idx]
                    self.logger.info("Loading cached Two-Tower variant '%s' from %s (seed=%d)...", var_name, cand_run_dir.name, s)
                    trained_m = TwoTowerModel.load(cand_run_dir, data=data)
                    trained_m.name = f"recsys_two_tower_{var_name}"
                    ranks_v = np.load(cand_run_dir / "ranks_val.npy")
                    ranks_t = np.load(cand_run_dir / "ranks_test.npy")
                    val_res = _make_eval_result_from_ranks(ranks_v, "val", data, eval_users, seed=s)
                    test_res = _make_eval_result_from_ranks(ranks_t, "test", data, eval_users, seed=s)
                else:
                    self.logger.info("Training Two-Tower variant '%s' (seed=%d)...", var_name, s)
                    var_cfg = TwoTowerConfig(
                        n_items=data.n_items,
                        text_dim=384,
                        tag_dim=300,
                        numeric_dim=9,
                        id_dim=64,
                        text_proj_dim=64,
                        tag_proj_dim=32,
                        numeric_proj_dim=16,
                        hidden_dim=256,
                        output_dim=64,
                        init_tau=0.07,
                        **overrides,
                    )
                    trained_m, history = train_two_tower_model(
                        data=data,
                        tuning_users=tuning_users,
                        cfg=var_cfg,
                        epochs=tt_epochs,
                        patience=tt_patience,
                        batch_size=512,
                        lr=1e-3,
                        seed=s,
                    )
                    trained_m.name = f"recsys_two_tower_{var_name}"
                    val_res = evaluate_model(trained_m, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=s)
                    test_res = evaluate_model(trained_m, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=s)

                    if var_name == "full" and s == 42:
                        tt_curves_history = history

                    # Save variant artifact
                    run_uuid = uuid.uuid4()
                    art_path = resolved_data_dir / "artifacts" / f"recsys_two_tower_{var_name}" / str(run_uuid)
                    if not self.dry_run:
                        art_path.mkdir(parents=True, exist_ok=True)
                        trained_m.save(art_path)
                        np.save(art_path / "ranks_val.npy", val_res.per_user["rank"])
                        np.save(art_path / "ranks_test.npy", test_res.per_user["rank"])

                        if session is not None:
                            await log_recsys_run(
                                model_name=f"recsys_two_tower_{var_name}",
                                family="two_tower",
                                dataset_version=data.dataset_version,
                                hyperparameters={**trained_m.hyperparameters(), "seed": s, "experiment": "ablation"},
                                val_metrics=val_res,
                                test_metrics=test_res,
                                artifact_dir=art_path,
                                extra={"experiment": "ablation", "is_experiment": True},
                                session=session,
                            )

                if var_name == "full" and s == 42:
                    full_seed42_test_res = test_res

                if s == 42:
                    trained_ablation_models[var_name] = trained_m

                seed_test_ndcgs.append(float(test_res.metrics.get("ndcg_at_10", 0.0)))
                seed_test_hrs.append(float(test_res.metrics.get("hit_rate_at_10", 0.0)))
                tail_ndcg = float(test_res.slices.get("pop_tail", {}).get("ndcg_at_10", 0.0))
                seed_tail_ndcgs.append(tail_ndcg)
                seed_test_eval_results.append(test_res)

            # Compute mean and std over seeds
            mean_ndcg = float(np.mean(seed_test_ndcgs))
            std_ndcg = float(np.std(seed_test_ndcgs)) if len(seed_test_ndcgs) > 1 else 0.0

            mean_hr = float(np.mean(seed_test_hrs))
            std_hr = float(np.std(seed_test_hrs)) if len(seed_test_hrs) > 1 else 0.0

            mean_tail = float(np.mean(seed_tail_ndcgs))
            std_tail = float(np.std(seed_tail_ndcgs)) if len(seed_tail_ndcgs) > 1 else 0.0

            # Paired diff vs full seed 42
            if full_seed42_test_res is not None:
                primary_test_res = seed_test_eval_results[0]
                diff_m, diff_lo, diff_hi = paired_bootstrap_diff(
                    primary_test_res.per_user["ndcg_at_10"],
                    full_seed42_test_res.per_user["ndcg_at_10"],
                    seed=seed,
                )
                sig = bool(diff_hi < 0.0 or diff_lo > 0.0)
            else:
                diff_m, diff_lo, diff_hi, sig = 0.0, 0.0, 0.0, False

            ablation_payload_entries.append({
                "variant": var_name,
                "description": desc,
                "seeds": var_seeds,
                "test_ndcg_at_10_mean": mean_ndcg,
                "test_ndcg_at_10_std": std_ndcg,
                "test_hr_at_10_mean": mean_hr,
                "test_hr_at_10_std": std_hr,
                "tail_ndcg_at_10_mean": mean_tail,
                "tail_ndcg_at_10_std": std_tail,
                "paired_ndcg_diff_vs_full": diff_m,
                "paired_ndcg_diff_lo95": diff_lo,
                "paired_ndcg_diff_hi95": diff_hi,
                "significant_vs_full": sig,
            })

        # ── 4. Ranker Robustness & Feature-Group Ablations (Multi-Seed) ──
        print("\n" + "=" * 130)
        print("🔬 Running Learning-to-Rank Robustness (5 Seeds) & Feature-Group Ablations...")
        print("=" * 130)

        ranker_seeds = [42, 43, 44, 45, 46] if not self.quick else [42]

        # Feature groups to ablate
        source_feats = [
            "tt_score", "tt_rank", "in_tt", "itemknn_score", "itemknn_rank", "in_itemknn",
            "als_score", "als_rank", "in_als", "content_score", "content_rank", "in_content"
        ]
        content_sim_feats = ["tag_jaccard", "content_max_cos", "content_mean_cos", "cos_to_last_item"]
        price_feats = ["price_final", "discount", "price_gap", "abs_price_gap"]
        user_feats = ["history_len", "log_products", "log_reviews", "mean_log_hours"]
        item_quality_feats = [
            "log_train_pop", "popularity_rank_pct", "age_years", "positive_ratio",
            "log_user_reviews", "win", "mac", "linux", "steam_deck", "has_description", "release_year_gap"
        ]

        ranker_ablation_defs = [
            ("without_source_scores", "Without Multi-Source Retrieval Scores/Ranks", set(source_feats), stage1_models),
            ("without_content_sim", "Without Content Similarity Features (Cosine & Jaccard)", set(content_sim_feats), stage1_models),
            ("without_price_features", "Without Price & Discount Features", set(price_feats), stage1_models),
            ("without_user_features", "Without User Profile & Hours Features", set(user_feats), stage1_models),
            ("without_item_quality", "Without Item Quality & Popularity Features", set(item_quality_feats), stage1_models),
            (
                "three_sources_only",
                "Stage 1 with 3 Sources Only (Without Item-kNN)",
                set(),
                {"two_tower": tt_model, "als": als_model, "content": content_model},
            ),
            (
                "without_all_time_item_stats",
                "Without All-Time Item Stats (Drops log_user_reviews & positive_ratio only)",
                {"log_user_reviews", "positive_ratio"},
                stage1_models,
            ),
            (
                "without_train_popularity",
                "Without Train Popularity Features (Drops log_train_pop & popularity_rank_pct only)",
                {"log_train_pop", "popularity_rank_pct"},
                stage1_models,
            ),
        ]

        # 4a. Train Full Ranker with 5 seeds
        self.logger.info("Training full ranker with %d seeds (%s)...", len(ranker_seeds), ranker_seeds)
        full_ranker_seed_val_res: dict[int, EvaluationResult] = {}
        full_ranker_seed_test_res: dict[int, EvaluationResult] = {}
        cached_train_val_by_seed: dict[int, dict[str, Any]] = {}

        for r_seed in ranker_seeds:
            r_cfg = RankerConfig(
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100 if self.quick else 500,
                min_child_samples=50,
                seed=r_seed,
                n_jobs=cfg.num_threads,
            )
            r_full_model, r_train_info = train_ranker(
                data=data,
                models=stage1_models,
                cfg=r_cfg,
                n_ranker_users=2000 if self.quick else 15000,
                eval_users=eval_users,
                tuning_users=tuning_users,
                feature_columns=list(FEATURE_COLUMNS),
                seed=r_seed,
                persistence_dir=resolved_data_dir,
            )
            cached_train_val_by_seed[r_seed] = r_train_info.get("precomputed_train_val")

            r_v_res = evaluate_model(r_full_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=r_seed)
            r_t_res = evaluate_model(r_full_model, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=r_seed)
            full_ranker_seed_val_res[r_seed] = r_v_res
            full_ranker_seed_test_res[r_seed] = r_t_res

        full_val_ndcgs = [res.metrics.get("ndcg_at_10", 0.0) for res in full_ranker_seed_val_res.values()]
        full_test_ndcgs = [res.metrics.get("ndcg_at_10", 0.0) for res in full_ranker_seed_test_res.values()]
        full_test_hrs = [res.metrics.get("hit_rate_at_10", 0.0) for res in full_ranker_seed_test_res.values()]

        full_val_mean, full_val_std = float(np.mean(full_val_ndcgs)), float(np.std(full_val_ndcgs))
        full_test_mean, full_test_std = float(np.mean(full_test_ndcgs)), float(np.std(full_test_ndcgs))
        full_test_hr_mean, full_test_hr_std = float(np.mean(full_test_hrs)), float(np.std(full_test_hrs))

        print(
            f"Full Ranker ({len(ranker_seeds)} seeds): Val NDCG@10 = {full_val_mean:.4f} ± {full_val_std:.4f} | "
            f"Test NDCG@10 = {full_test_mean:.4f} ± {full_test_std:.4f} | "
            f"Test HR@10 = {full_test_hr_mean:.4f} ± {full_test_hr_std:.4f}"
        )

        # 4b. Evaluate 8 feature ablations over 5 seeds
        ranker_ablation_payload_entries: list[dict[str, Any]] = []

        for grp_key, grp_desc, excluded_feats, grp_stage1_models in ranker_ablation_defs:
            active_feats = [f for f in FEATURE_COLUMNS if f not in excluded_feats]
            self.logger.info("Evaluating ranker ablation '%s' (%d features) across %d seeds...", grp_key, len(active_feats), len(ranker_seeds))

            seed_val_diffs: list[float] = []
            seed_test_diffs: list[float] = []
            seed_val_ndcgs: list[float] = []
            seed_test_ndcgs: list[float] = []
            seed_test_hrs: list[float] = []

            seed42_val_res: EvaluationResult | None = None
            seed42_test_res: EvaluationResult | None = None

            for r_seed in ranker_seeds:
                r_cfg = RankerConfig(
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=100 if self.quick else 500,
                    min_child_samples=50,
                    seed=r_seed,
                    n_jobs=cfg.num_threads,
                )
                pre_tv = cached_train_val_by_seed.get(r_seed) if grp_key != "three_sources_only" else None

                m_abl, _ = train_ranker(
                    data=data,
                    models=grp_stage1_models,
                    cfg=r_cfg,
                    n_ranker_users=2000 if self.quick else 15000,
                    eval_users=eval_users,
                    tuning_users=tuning_users,
                    feature_columns=active_feats,
                    seed=r_seed,
                    persistence_dir=resolved_data_dir,
                    precomputed_train_val=pre_tv,
                )
                m_abl.name = f"recsys_two_tower_ranker_{grp_key}"

                v_res = evaluate_model(m_abl, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=r_seed)
                t_res = evaluate_model(m_abl, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=r_seed)

                full_v = full_ranker_seed_val_res[r_seed]
                full_t = full_ranker_seed_test_res[r_seed]

                v_ndcg = float(v_res.metrics.get("ndcg_at_10", 0.0))
                t_ndcg = float(t_res.metrics.get("ndcg_at_10", 0.0))
                t_hr = float(t_res.metrics.get("hit_rate_at_10", 0.0))

                diff_v = v_ndcg - float(full_v.metrics.get("ndcg_at_10", 0.0))
                diff_t = t_ndcg - float(full_t.metrics.get("ndcg_at_10", 0.0))

                seed_val_ndcgs.append(v_ndcg)
                seed_test_ndcgs.append(t_ndcg)
                seed_test_hrs.append(t_hr)
                seed_val_diffs.append(diff_v)
                seed_test_diffs.append(diff_t)

                if r_seed == 42:
                    seed42_val_res = v_res
                    seed42_test_res = t_res

                    # Save seed 42 ablation artifact
                    run_uuid = uuid.uuid4()
                    art_path = resolved_data_dir / "artifacts" / f"recsys_two_tower_ranker_{grp_key}" / str(run_uuid)
                    if not self.dry_run:
                        art_path.mkdir(parents=True, exist_ok=True)
                        m_abl.save(art_path)
                        np.save(art_path / "ranks_val.npy", v_res.per_user["rank"])
                        np.save(art_path / "ranks_test.npy", t_res.per_user["rank"])

                        if session is not None:
                            await log_recsys_run(
                                model_name=f"recsys_two_tower_ranker_{grp_key}",
                                family="recsys_two_tower_ranker",
                                dataset_version=data.dataset_version,
                                hyperparameters={**m_abl.hyperparameters(), "experiment": "ablation", "ablation_group": grp_key},
                                val_metrics=v_res,
                                test_metrics=t_res,
                                artifact_dir=art_path,
                                extra={"experiment": "ablation", "is_experiment": True},
                                session=session,
                            )

            mean_val_diff = float(np.mean(seed_val_diffs))
            mean_test_diff = float(np.mean(seed_test_diffs))
            mean_val_ndcg = float(np.mean(seed_val_ndcgs))
            mean_test_ndcg = float(np.mean(seed_test_ndcgs))
            mean_test_hr = float(np.mean(seed_test_hrs))

            # Paired difference CI over users on seed 42
            full_s42_test = full_ranker_seed_test_res[42]
            t_diff42, t_lo, t_hi = paired_bootstrap_diff(
                seed42_test_res.per_user["ndcg_at_10"], full_s42_test.per_user["ndcg_at_10"], seed=42
            )
            v_diff42, v_lo, v_hi = paired_bootstrap_diff(
                seed42_val_res.per_user["ndcg_at_10"], full_ranker_seed_val_res[42].per_user["ndcg_at_10"], seed=42
            )

            is_sig = bool(t_hi < 0.0 or t_lo > 0.0)
            is_outside_noise = bool(abs(mean_test_diff) > full_test_std)

            status_sentence = (
                f"Effect is outside seed noise (mean diff: {mean_test_diff:+.4f} vs full std: {full_test_std:.4f}; p < 0.05)"
                if is_sig and is_outside_noise
                else f"Effect is within seed noise (mean diff: {mean_test_diff:+.4f} vs full std: {full_test_std:.4f})"
            )

            ranker_ablation_payload_entries.append({
                "ablated_group": grp_key,
                "description": grp_desc,
                "val_ndcg_at_10": mean_val_ndcg,
                "val_hr_at_10": float(np.mean(seed_val_ndcgs)),
                "val_ndcg_diff_vs_full": mean_val_diff,
                "val_ndcg_diff_lo95": v_lo,
                "val_ndcg_diff_hi95": v_hi,
                "test_ndcg_at_10": mean_test_ndcg,
                "test_hr_at_10": mean_test_hr,
                "test_ndcg_diff_vs_full": mean_test_diff,
                "test_ndcg_diff_lo95": t_lo,
                "test_ndcg_diff_hi95": t_hi,
                "is_significant_test": is_sig,
                "is_outside_noise": is_outside_noise,
                "analysis_sentence": status_sentence,
            })

        # ── 5. Slice Paired Tests (Ranker vs Two-Tower, Ranker vs Item-kNN, Two-Tower vs Item-kNN) ──
        print("\n" + "=" * 130)
        print("🔬 Running Slices Paired Bootstrap Significance Tests...")
        print("=" * 130)

        slice_keys = ["pop_head", "pop_torso", "pop_tail", "hist_3_4", "hist_5_9", "hist_10_plus"]
        slice_definitions = {
            "pop_head": "Top 20% most popular items in training set (Target item popularity)",
            "pop_torso": "Next 30% most popular items (20th to 50th percentile)",
            "pop_tail": "Bottom 50% items (50th to 100th percentile)",
            "hist_3_4": "Users with 3 to 4 fold-in interaction history length",
            "hist_5_9": "Users with 5 to 9 fold-in interaction history length",
            "hist_10_plus": "Users with 10 or more fold-in interaction history length",
        }

        comparisons_plan = [
            ("Ranker vs Two-Tower", "recsys_two_tower_ranker", "recsys_two_tower"),
            ("Ranker vs Item-kNN", "recsys_two_tower_ranker", "recsys_itemknn"),
            ("Two-Tower vs Item-kNN", "recsys_two_tower", "recsys_itemknn"),
        ]

        slice_payload_entries: list[dict[str, Any]] = []

        for comp_name, m_a_key, m_b_key in comparisons_plan:
            for split in ["val", "test"]:
                res_a = eval_val_results[m_a_key] if split == "val" else eval_test_results[m_a_key]
                res_b = eval_val_results[m_b_key] if split == "val" else eval_test_results[m_b_key]

                for s_key in slice_keys:
                    s_a = res_a.slices.get(s_key, {})
                    s_b = res_b.slices.get(s_key, {})
                    n_u = int(s_a.get("count", 0))

                    if n_u == 0:
                        continue

                    # Slice-specific user mask
                    if s_key.startswith("pop_"):
                        sorted_pop_items = np.argsort(-data.item_train_pop)
                        n_items = data.n_items
                        head_cut = int(0.20 * n_items)
                        torso_cut = int(0.50 * n_items)
                        if s_key == "pop_head":
                            item_set = set(sorted_pop_items[:head_cut])
                        elif s_key == "pop_torso":
                            item_set = set(sorted_pop_items[head_cut:torso_cut])
                        else:
                            item_set = set(sorted_pop_items[torso_cut:])

                        targets = res_a.per_user["target_item"]
                        mask = np.array([t in item_set for t in targets], dtype=bool)
                    else:
                        lengths = res_a.per_user["history_length"]
                        if s_key == "hist_3_4":
                            mask = (lengths >= 3) & (lengths <= 4)
                        elif s_key == "hist_5_9":
                            mask = (lengths >= 5) & (lengths <= 9)
                        else:
                            mask = lengths >= 10

                    ndcg_a_slice = res_a.per_user["ndcg_at_10"][mask]
                    ndcg_b_slice = res_b.per_user["ndcg_at_10"][mask]

                    diff_m, diff_lo, diff_hi = paired_bootstrap_diff(ndcg_a_slice, ndcg_b_slice, seed=seed)

                    if diff_lo > 0.0:
                        verdict = "better"
                    elif diff_hi < 0.0:
                        verdict = "worse"
                    else:
                        verdict = "no significant difference"

                    slice_payload_entries.append({
                        "comparison": comp_name,
                        "split": split,
                        "slice_name": s_key,
                        "n_users": int(np.sum(mask)),
                        "model_a_ndcg": float(s_a.get("ndcg_at_10", 0.0)),
                        "model_b_ndcg": float(s_b.get("ndcg_at_10", 0.0)),
                        "paired_diff": diff_m,
                        "lo95": diff_lo,
                        "hi95": diff_hi,
                        "verdict": verdict,
                    })

        # ── 6. Cold-Start Studies ──
        print("\n" + "=" * 130)
        print("🔬 Running Cold-Start Studies (Sparse Target Subgroup & Simulated C1, C2, C3)...")
        print("=" * 130)

        coldstart_models_dict = {
            "popularity": pop_model,
            "content": content_model,
            "als": als_model,
            "itemknn": knn_model,
            "two_tower": tt_model,
            "two_tower_no_id": trained_ablation_models.get("no_id"),
        }

        cold_start_res = run_full_cold_start_study(
            data=data,
            models=coldstart_models_dict,
            eval_users=eval_users,
            seed=seed,
            n_boot=1000,
        )

        # ── 6b. True Item Holdout Study (Part D) ──
        print("\n" + "=" * 130)
        print("🔬 Running True Item Holdout Study (Retraining Two-Tower on 90% Catalog)...")
        print("=" * 130)

        held_out_items, held_out_set = get_item_holdout_split(data, fraction=0.10, seed=42)

        # Filter train_df: remove all interactions with held-out items
        train_filtered = data.train_df[~data.train_df["item_idx"].isin(held_out_set)].copy().reset_index(drop=True)
        self.logger.info("Item holdout: dropped %d training interactions on %d held-out items", len(data.train_df) - len(train_filtered), len(held_out_set))

        holdout_pop = np.zeros(data.n_items, dtype=np.int64)
        pop_counts = train_filtered["item_idx"].value_counts()
        holdout_pop[pop_counts.index.to_numpy()] = pop_counts.to_numpy()

        data_holdout = RecsysData(
            train_df=train_filtered,
            val_df=data.val_df,
            test_df=data.test_df,
            negatives_df=data.negatives_df,
            users_df=data.users_df,
            mappings_df=data.mappings_df,
            items_df=data.items_df,
            item_text_emb=data.item_text_emb,
            item_train_pop=holdout_pop,
            item_tags_sparse=data.item_tags_sparse,
            dataset_version=f"{data.dataset_version}_holdout",
            manifest=dict(data.manifest),
            scalers=data.scalers,
            tag_vocab=data.tag_vocab,
        )

        # Retrain full Two-Tower and No-ID Two-Tower on holdout dataset (seed 42)
        tt_holdout_cfg = TwoTowerConfig(
            n_items=data.n_items,
            hidden_dim=256,
            output_dim=64,
            use_id=True,
            id_dropout=0.25,
        )
        tt_holdout_model, _ = train_two_tower_model(
            data=data_holdout,
            tuning_users=tuning_users,
            cfg=tt_holdout_cfg,
            epochs=3 if self.quick else 15,
            patience=2 if self.quick else (5 if cfg.profile == "dev" else 3),
            batch_size=512,
            lr=1e-3,
            seed=seed,
        )

        tt_noid_holdout_cfg = TwoTowerConfig(
            n_items=data.n_items,
            hidden_dim=256,
            output_dim=64,
            use_id=False,
        )
        tt_noid_holdout_model, _ = train_two_tower_model(
            data=data_holdout,
            tuning_users=tuning_users,
            cfg=tt_noid_holdout_cfg,
            epochs=3 if self.quick else 15,
            patience=2 if self.quick else (5 if cfg.profile == "dev" else 3),
            batch_size=512,
            lr=1e-3,
            seed=seed,
        )

        holdout_models_dict = {
            "two_tower": tt_holdout_model,
            "two_tower_no_id": tt_noid_holdout_model,
            "content": content_model,
        }

        item_holdout_res = evaluate_item_holdout_study(
            data=data,
            models=holdout_models_dict,
            eval_users=eval_users,
            held_out_items=held_out_items,
            seed=seed,
            n_boot=1000,
        )

        # Register holdout runs tagged experiment=item_holdout (candidate, never promotable)
        if session is not None and not self.dry_run:
            run_uuid_holdout = uuid.uuid4()
            art_holdout_tt = resolved_data_dir / "artifacts" / "recsys_two_tower_item_holdout" / str(run_uuid_holdout)
            art_holdout_tt.mkdir(parents=True, exist_ok=True)
            tt_holdout_model.save(art_holdout_tt)
            await log_recsys_run(
                model_name="recsys_two_tower_item_holdout",
                family="two_tower",
                dataset_version=data.dataset_version,
                hyperparameters={**tt_holdout_model.hyperparameters(), "experiment": "item_holdout"},
                val_metrics=eval_val_results["recsys_two_tower"],
                test_metrics=eval_test_results["recsys_two_tower"],
                artifact_dir=art_holdout_tt,
                extra={"experiment": "item_holdout", "is_experiment": True},
                session=session,
            )

        # ── 7. Beyond-Accuracy Catalog Metrics & MMR Re-Ranking ──
        print("\n" + "=" * 130)
        print("🔬 Computing Beyond-Accuracy Metrics & MMR Diversity Re-Ranking Trade-Off...")
        print("=" * 130)

        beyond_acc_entries: list[dict[str, Any]] = []
        coverage_mystery_info: dict[str, Any] = {}
        masked_unmasked_comparisons: list[dict[str, Any]] = []

        # Build fold-in test histories and seen masks using evaluate_model infrastructure
        user_train_pos, user_train_neg = _build_user_interaction_maps(data)
        v_map = dict(zip(data.val_df["user_idx"].to_numpy(), data.val_df["item_idx"].to_numpy(), strict=True))
        t_map = dict(zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True))

        hist_list = []
        seen_list = []
        for u in eval_users:
            p_items = list(user_train_pos.get(u, []))
            v_it = v_map.get(u)
            if v_it is not None:
                p_items.append(int(v_it))
            user_hist = p_items[-50:] if len(p_items) > 50 else p_items
            seen = set(p_items) | user_train_neg.get(u, set())
            hist_list.append(np.array(user_hist, dtype=np.int32))
            seen_list.append(seen)

        for m_key, m_inst in all_models.items():
            test_res = eval_test_results[m_key]

            # (a) Official masked top-10 recommendations directly from test evaluation
            top10_masked = test_res.per_user["top_k_indices"][:, :10]

            cov_masked = catalog_coverage_at_k(top10_masked, data.n_items)
            div_masked = diversity_at_k(top10_masked, data.item_text_emb) if data.item_text_emb is not None else 0.0
            nov_val = novelty_at_k(top10_masked, data.item_train_pop)
            head_sh = head_share_at_k(top10_masked, data.item_train_pop, head_fraction=0.01)
            gini_val = gini_coefficient_at_k(top10_masked, data.n_items)

            # Strict consistency assertion against official test evaluation metrics
            official_cov = test_res.metrics.get("catalog_coverage_at_10", 0.0)
            official_div = test_res.metrics.get("diversity_at_10", 0.0)
            if abs(cov_masked - official_cov) > 1e-9:
                raise ValueError(
                    f"Coverage mismatch for {m_key}: computed {cov_masked}, official {official_cov}"
                )
            if abs(div_masked - official_div) > 1e-9:
                raise ValueError(
                    f"Diversity mismatch for {m_key}: computed {div_masked}, official {official_div}"
                )

            # (b) Unmasked top-10 recommendations (for Coverage Mystery analysis)
            raw_scores = m_inst.score(hist_list).copy()
            top10_unmasked = get_top_k_indices(raw_scores, 10)
            cov_unmasked = catalog_coverage_at_k(top10_unmasked, data.n_items)
            div_unmasked = diversity_at_k(top10_unmasked, data.item_text_emb) if data.item_text_emb is not None else 0.0

            masked_unmasked_comparisons.append({
                "model_name": m_key,
                "masked_coverage": cov_masked,
                "masked_diversity": div_masked,
                "unmasked_coverage": cov_unmasked,
                "unmasked_diversity": div_unmasked,
            })

            if m_key == "recsys_two_tower":
                coverage_mystery_info = {
                    "official_two_tower_cov": f"{cov_masked:.4f}",
                    "official_two_tower_div": f"{div_masked:.4f}",
                    "unmasked_two_tower_cov": f"{cov_unmasked:.4f}",
                    "unmasked_two_tower_div": f"{div_unmasked:.4f}",
                }

            beyond_acc_entries.append({
                "model_name": m_key,
                "catalog_coverage_at_10": cov_masked,
                "novelty_at_10": nov_val,
                "diversity_at_10": div_masked,
                "head_share_at_10": head_sh,
                "gini_coefficient_at_10": gini_val,
            })

        # MMR Diversity Re-ranking on Ranker top-50
        self.logger.info("Computing MMR re-ranking trade-off (lambda in {1.0, 0.8, 0.6})...")

        # Score ranker with exact batching and seen masking
        ranker_test_scores = np.empty((len(eval_users), data.n_items), dtype=np.float32)
        for i in range(0, len(eval_users), 1024):
            b_slice = slice(i, min(i + 1024, len(eval_users)))
            b_hist = hist_list[b_slice]
            b_scores = ranker_model.score(b_hist).astype(np.float32, copy=False)
            for idx, g_idx in enumerate(range(b_slice.start, b_slice.stop)):
                s_items = list(seen_list[g_idx])
                if s_items:
                    b_scores[idx, s_items] = -np.inf
            ranker_test_scores[b_slice] = b_scores

        top50_idx = get_top_k_indices(ranker_test_scores, 50)
        row_grid = np.arange(len(eval_users))[:, None]
        top50_scores = ranker_test_scores[row_grid, top50_idx]

        test_targets = np.array([int(t_map[u]) for u in eval_users], dtype=np.int64)

        mmr_entries: list[dict[str, Any]] = []
        lambda1_ndcg_per_user: np.ndarray | None = None

        for lam in [1.0, 0.8, 0.6]:
            reranked_top10 = mmr_rerank(
                candidate_items=top50_idx,
                candidate_scores=top50_scores,
                item_embeddings=data.item_text_emb,
                lambda_param=lam,
                k=10,
            )

            ranks_mmr = np.full(len(eval_users), 999, dtype=np.int64)
            for u_i, t in enumerate(test_targets):
                top_items = reranked_top10[u_i]
                pos = np.where(top_items == t)[0]
                if len(pos) > 0:
                    ranks_mmr[u_i] = int(pos[0]) + 1

            ndcg_vals = ndcg_at_k(ranks_mmr, 10)
            nd_m, nd_lo, nd_hi = bootstrap_ci(ndcg_vals, n_boot=1000, seed=seed)
            div_val = diversity_at_k(reranked_top10, data.item_text_emb) if data.item_text_emb is not None else 0.0

            if lam == 1.0:
                lambda1_ndcg_per_user = ndcg_vals
                diff_m, diff_lo, diff_hi = 0.0, 0.0, 0.0
                official_ranker_test_ndcg = eval_test_results["recsys_two_tower_ranker"].metrics["ndcg_at_10"]
                if abs(nd_m - official_ranker_test_ndcg) > 1e-9:
                    raise ValueError(
                        f"MMR lambda=1.0 NDCG@10 ({nd_m:.8f}) does not equal official ranker test NDCG@10 ({official_ranker_test_ndcg:.8f})"
                    )
            else:
                diff_m, diff_lo, diff_hi = paired_bootstrap_diff(ndcg_vals, lambda1_ndcg_per_user, seed=seed)

            mmr_entries.append({
                "lambda_param": lam,
                "ndcg_at_10": nd_m,
                "ndcg_at_10_lo95": nd_lo,
                "ndcg_at_10_hi95": nd_hi,
                "diversity_at_10": div_val,
                "paired_ndcg_diff_vs_lambda1": diff_m,
                "paired_ndcg_diff_lo95": diff_lo,
                "paired_ndcg_diff_hi95": diff_hi,
            })

        # ── 8. Global SHAP Payload ──
        shap_file = ranker_dir / "global_shap.json"
        imp_file = ranker_dir / "feature_importance.json"

        global_shap_dict: dict[str, float] = {}
        gain_dict: dict[str, float] = {}
        split_dict: dict[str, float] = {}

        if shap_file.exists():
            with open(shap_file, encoding="utf-8") as f:
                global_shap_dict = json.load(f)
        if imp_file.exists():
            with open(imp_file, encoding="utf-8") as f:
                imp_raw = json.load(f)
                gain_dict = imp_raw.get("gain", {})
                split_dict = imp_raw.get("split", {})

        # ── 9. Build All Report Payloads ──
        self.logger.info("Building and validating all 10 report payloads...")

        # Query latest ModelRuns for leaderboard IDs
        model_runs_dict: dict[str, ModelRun] = {}
        if session is not None:
            stmt = select(ModelRun).where(ModelRun.model_name.like("recsys_%")).order_by(ModelRun.created_at.desc())
            res = await session.execute(stmt)
            for r in res.scalars().all():
                if r.model_name not in model_runs_dict:
                    model_runs_dict[r.model_name] = r

        leaderboard_models_list = []
        for m_key in all_models:
            m_inst = all_models[m_key]
            v_res = eval_val_results[m_key]
            t_res = eval_test_results[m_key]
            run_rec = model_runs_dict.get(m_key)

            leaderboard_models_list.append({
                "model_name": m_key,
                "family": m_inst.name.replace("recsys_", ""),
                "stage": run_rec.stage if run_rec else ("production" if m_key == "recsys_two_tower" else "candidate"),
                "model_run_id": str(run_rec.id) if run_rec else None,
                "hyperparameters": m_inst.hyperparameters(),
                "training_time_sec": None,
                "val_metrics": v_res.metrics,
                "test_metrics": t_res.metrics,
                "val_slices": v_res.slices,
                "test_slices": t_res.slices,
            })

        leaderboard_payload = build_leaderboard_payload(
            dataset_version=data.dataset_version,
            models_data=leaderboard_models_list,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        ablations_payload = build_ablations_payload(
            dataset_version=data.dataset_version,
            variants=ablation_payload_entries,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        ranker_ablations_payload = build_ranker_ablations_payload(
            dataset_version=data.dataset_version,
            full_ranker_features_count=len(FEATURE_COLUMNS),
            ablations=ranker_ablation_payload_entries,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        slices_payload = build_slices_payload(
            dataset_version=data.dataset_version,
            slice_definitions=slice_definitions,
            comparisons=slice_payload_entries,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        cold_start_payload = build_cold_start_payload(
            dataset_version=data.dataset_version,
            sparse_targets_study=cold_start_res["sparse_targets_study"],
            simulated_new_items_study=cold_start_res["simulated_new_items_study"],
            n_eval_users=len(eval_users),
            seed=seed,
            item_holdout_study=item_holdout_res,
        )

        beyond_accuracy_payload = build_beyond_accuracy_payload(
            dataset_version=data.dataset_version,
            metrics_at_10=beyond_acc_entries,
            mmr_reranking=mmr_entries,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        shap_global_payload = build_shap_global_payload(
            dataset_version=data.dataset_version,
            global_shap=global_shap_dict,
            gain=gain_dict,
            split=split_dict,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        training_curves_payload = build_training_curves_payload(
            dataset_version=data.dataset_version,
            two_tower_curves=tt_curves_history,
            ranker_curves=None,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        split_protocol_text = (
            "Per-user chronological split: For each user with >= min_user_positives (default 5), "
            "all interaction events are ordered by review timestamp. The latest positive interaction is assigned "
            "to TEST, the second latest to VAL, and all preceding interactions to TRAIN. Evaluation is strictly "
            "fold-in next-item recommendation over 5,000 fixed evaluation users."
        )

        dataset_payload = build_dataset_payload(
            data=data,
            split_protocol=split_protocol_text,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        limitations_payload = build_limitations_payload(
            dataset_version=data.dataset_version,
            coverage_mystery_info=coverage_mystery_info,
            n_eval_users=len(eval_users),
            seed=seed,
        )

        all_reports = {
            "leaderboard": leaderboard_payload,
            "ablations": ablations_payload,
            "ranker_ablations": ranker_ablations_payload,
            "slices": slices_payload,
            "cold_start": cold_start_payload,
            "beyond_accuracy": beyond_accuracy_payload,
            "shap_global": shap_global_payload,
            "training_curves": training_curves_payload,
            "dataset": dataset_payload,
            "limitations": limitations_payload,
        }

        # ── 10. Write Reports to Disk & Upsert to PostgreSQL ──
        reports_dir = resolved_data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        for rep_key, rep_payload in all_reports.items():
            save_report_json(reports_dir / f"{rep_key}.json", rep_payload)

            if session is not None and not self.dry_run:
                stmt = (
                    insert(ServingRecReport)
                    .values(
                        report_key=rep_key,
                        payload=rep_payload,
                        dataset_version=data.dataset_version,
                        model_run_id=model_runs_dict.get("recsys_two_tower").id if model_runs_dict.get("recsys_two_tower") else None,
                    )
                    .on_conflict_do_update(
                        index_elements=[ServingRecReport.report_key],
                        set_={
                            "payload": rep_payload,
                            "dataset_version": data.dataset_version,
                        },
                    )
                )
                await session.execute(stmt)

        if session is not None and not self.dry_run:
            await session.commit()
            self.logger.info("Upserted all %d report payloads into serving_rec_reports table.", len(all_reports))

        # Generate leaderboard.md
        md_content = generate_leaderboard_markdown(
            leaderboard_payload=leaderboard_payload,
            ablations_payload=ablations_payload,
            ranker_ablations_payload=ranker_ablations_payload,
            slices_payload=slices_payload,
            cold_start_payload=cold_start_payload,
            beyond_accuracy_payload=beyond_accuracy_payload,
        )

        with open(reports_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        with open(resolved_data_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info("Wrote leaderboard.md to %s and %s", reports_dir / "leaderboard.md", resolved_data_dir / "leaderboard.md")

        # ── 11. Print Summary Tables to Console ──
        print("\n" + "=" * 130)
        print("📊 (A) BEYOND-ACCURACY VERIFIED METRICS & MMR CONSISTENCY (1e-9 tolerance):")
        print("=" * 130)
        print(f"{'Model':<30} | {'Coverage@10':<15} | {'Diversity@10':<15} | {'Novelty@10':<15} | {'Gini@10':<15}")
        print("-" * 100)
        for ba in beyond_acc_entries:
            print(f"{ba['model_name']:<30} | {ba['catalog_coverage_at_10']:<15.4f} | {ba['diversity_at_10']:<15.4f} | {ba['novelty_at_10']:<15.4f} | {ba['gini_coefficient_at_10']:<15.4f}")

        print("\n" + "=" * 130)
        print("🔍 (B) COVERAGE MYSTERY: MASKED (OFFICIAL) VS UNMASKED TOP-10 CATALOG METRICS:")
        print("=" * 130)
        print(f"{'Model':<30} | {'Masked Cov@10':<15} | {'Masked Div@10':<15} | {'Unmasked Cov@10':<18} | {'Unmasked Div@10':<18}")
        print("-" * 105)
        for mu in masked_unmasked_comparisons:
            print(f"{mu['model_name']:<30} | {mu['masked_coverage']:<15.4f} | {mu['masked_diversity']:<15.4f} | {mu['unmasked_coverage']:<18.4f} | {mu['unmasked_diversity']:<18.4f}")

        print("\n" + "=" * 130)
        print("❄️ (C) COLD START: THREE FAIR SIMULATIONS (C1, C2, C3) & SPARSE TARGETS (PART A):")
        print("=" * 130)
        print(f"--- Part A: Sparse Targets (<= 10 train positives, n = {cold_start_res['sparse_targets_study']['n_eval_users']} users) ---")
        for m_k, m_v in cold_start_res["sparse_targets_study"]["models"].items():
            print(f"  {m_k:<20}: HR@10 = {m_v['hit_rate_at_10']:.4f} [{m_v['hit_rate_at_10_lo95']:.4f}, {m_v['hit_rate_at_10_hi95']:.4f}] | NDCG@10 = {m_v['ndcg_at_10']:.4f} [{m_v['ndcg_at_10_lo95']:.4f}, {m_v['ndcg_at_10_hi95']:.4f}]")

        for c_key, c_name in [("c1_old_setup", "C1 (Old Setup)"), ("c2_fair_random", "C2 (Fair Random 10%)"), ("c3_fair_bottom50", "C3 (Fair Bottom 50% Torso/Tail)")]:
            c_data = cold_start_res[c_key]
            print(f"\n--- {c_name} (n = {c_data['n_eval_users']} users, {c_data['n_new_items']} new items) ---")
            for m_k, m_v in c_data["models"].items():
                print(f"  {m_k:<20}: HR@10 = {m_v['hit_rate_at_10']:.4f} [{m_v['hit_rate_at_10_lo95']:.4f}, {m_v['hit_rate_at_10_hi95']:.4f}] | NDCG@10 = {m_v['ndcg_at_10']:.4f} [{m_v['ndcg_at_10_lo95']:.4f}, {m_v['ndcg_at_10_hi95']:.4f}]")

        print("\n" + "=" * 130)
        print("🛡️ (D) TRUE ITEM HOLDOUT STUDY (10% Torso/Tail Retrained from Scratch):")
        print("=" * 130)
        print(f"--- Evaluated on {item_holdout_res['n_eval_users']} test users with held-out targets ({item_holdout_res['n_held_out_items']} held-out items) ---")
        for m_k, m_v in item_holdout_res["models"].items():
            print(f"  {m_k:<20}: HR@10 = {m_v['hit_rate_at_10']:.4f} [{m_v['hit_rate_at_10_lo95']:.4f}, {m_v['hit_rate_at_10_hi95']:.4f}] | NDCG@10 = {m_v['ndcg_at_10']:.4f} [{m_v['ndcg_at_10_lo95']:.4f}, {m_v['ndcg_at_10_hi95']:.4f}]")

        print("\n" + "=" * 130)
        print("🌳 (E) LEARNING-TO-RANK SEED ROBUSTNESS (5 SEEDS) & FEATURE-GROUP ABLATIONS:")
        print("=" * 130)
        print(f"{'Ablation Group':<32} | {'Test NDCG@10':<15} | {'Test Diff vs Full (95% CI)':<30} | {'Status':<35}")
        print("-" * 120)
        for ra in ranker_ablation_payload_entries:
            diff_str = f"{ra['test_ndcg_diff_vs_full']:+.4f} [{ra['test_ndcg_diff_lo95']:+.4f}, {ra['test_ndcg_diff_hi95']:+.4f}]"
            print(f"{ra['ablated_group']:<32} | {ra['test_ndcg_at_10']:<15.4f} | {diff_str:<30} | {ra['analysis_sentence']}")

        print("\n" + md_content)

        return {
            "dataset_version": data.dataset_version,
            "reports_count": len(all_reports),
            "leaderboard": leaderboard_payload,
            "cold_start": cold_start_res,
            "item_holdout": item_holdout_res,
            "beyond_accuracy": beyond_acc_entries,
            "ranker_ablations": ranker_ablation_payload_entries,
        }

    def execute(self) -> dict[str, Any]:
        """Entry point for executing BaseJob."""
        if self.dry_run:
            return asyncio.run(self._run_async(session=None))

        async def _execute_with_engine() -> dict[str, Any]:
            engine = create_async_engine(settings.async_database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                res = await self._run_async(session=session)
            await engine.dispose()
            return res

        return asyncio.run(_execute_with_engine())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recommender system ablations and evidence generation")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs / smaller tuning subsets for fast run")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow or DB writes")
    args = parser.parse_args()

    job = RunRecsysExperimentsJob(
        quick=args.quick,
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
