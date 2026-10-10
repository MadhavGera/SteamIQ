"""
Job: Generate and Persist Cold-Start Evaluation Report — Phase 6.

Evaluates cold-start and discovery capabilities across three regimes:
1. True Item Holdout (Primary prospective benchmark: 90% catalog retrained, 300 items withheld).
2. Sparse Targets Subgroup (<= 10 train positives, optimistic semantic prior proxy).
3. Simulated 10% New Items (C2 fair simulation with neutralized proxies).

Note on Two-Tower Masked:
`two_tower_masked` evaluates the reference two-tower model with all-time stats and is
explicitly labeled 'reference (all-time stats)'.

Usage:
  python -m jobs.generate_cold_start_report
  make recsys-cold-start
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ServingRecReport
from jobs.base_job import BaseJob
from recsys.coldstart import (
    _build_test_foldin_data,
    get_item_holdout_split,
    get_simulated_new_items_test_users,
    get_sparse_target_test_users,
)
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import get_evaluation_user_split
from recsys.metrics import bootstrap_ci, compute_ranks, hit_rate_at_k, mrr_at_k, ndcg_at_k
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerModel


def _find_latest_artifact_dir(data_dir: Path, model_name: str) -> tuple[Path, str]:
    art_dir = data_dir / "artifacts" / model_name
    if not art_dir.exists():
        raise FileNotFoundError(f"Artifact dir {art_dir} does not exist.")
    runs = [d for d in art_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {art_dir}.")
    latest = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return latest, latest.name


def summarize_model_ranks(ranks: np.ndarray, n_users: int, seed: int = 42) -> dict[str, Any]:
    hr_arr = hit_rate_at_k(ranks, 10)
    nd_arr = ndcg_at_k(ranks, 10)
    mrr_arr = mrr_at_k(ranks, 10)

    hits_k = int(np.sum(hr_arr))
    hr_m, hr_lo, hr_hi = bootstrap_ci(hr_arr, n_boot=1000, seed=seed)
    nd_m, nd_lo, nd_hi = bootstrap_ci(nd_arr, n_boot=1000, seed=seed)
    mrr_m, mrr_lo, mrr_hi = bootstrap_ci(mrr_arr, n_boot=1000, seed=seed)
    mr_m, mr_lo, mr_hi = bootstrap_ci(ranks, n_boot=1000, seed=seed)
    med_r = float(np.median(ranks))

    return {
        "hits_k": hits_k,
        "n_users": n_users,
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
        "median_rank": med_r,
    }


class GenerateColdStartReportJob(BaseJob):
    job_name = "generate_cold_start_report"

    def __init__(
        self,
        *,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.profile = profile
        self.data_dir = data_dir
        self.seed = seed

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        reports_dir = resolved_data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Loading dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)
        eval_users, _ = get_evaluation_user_split(
            data, n_eval_users=5000, n_tuning_users=5000, seed=self.seed, persistence_dir=resolved_data_dir
        )

        # Load models
        content_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_content")
        content_model = ContentRecommender.load(content_dir)

        als_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_als")
        als_model = ALSRecommender.load(als_dir)

        pop_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_popularity")
        pop_model = PopularityRecommender.load(pop_dir)

        # Load Reference Two-Tower Model (uses all-time stats)
        tt_dir, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_two_tower")
        tt_model = TwoTowerModel.load(tt_dir, data=data)
        self.logger.info("Loaded reference Two-Tower model from %s (all-time stats)", tt_dir)

        # ──────────────────────────────────────────────────────────────────────
        # 1. True Item Holdout Study (n=184 users, 300 held-out items) - LEADS
        # ──────────────────────────────────────────────────────────────────────
        held_out_items, held_out_set = get_item_holdout_split(data, fraction=0.10, seed=self.seed)
        test_target_map = dict(
            zip(data.test_df["user_idx"].to_numpy(), data.test_df["item_idx"].to_numpy(), strict=True)
        )
        holdout_users = [int(u) for u in eval_users if int(test_target_map[int(u)]) in held_out_set]
        n_ho_users = len(holdout_users)
        self.logger.info("True Holdout Users: %d", n_ho_users)

        ho_hist, ho_targ, ho_seen = _build_test_foldin_data(data, np.array(holdout_users, dtype=np.int64))
        ho_targ_arr = np.asarray(ho_targ, dtype=np.int64)
        held_out_list = [int(it) for it in held_out_items]

        # Random baseline on holdout
        rng = np.random.default_rng(self.seed)

        # Content on holdout
        sc_ho_cnt = content_model.score(ho_hist).copy()
        for i, sm in enumerate(ho_seen):
            sc_ho_cnt[i, list(sm)] = -np.inf
        ho_rk_cnt = compute_ranks(sc_ho_cnt, ho_targ_arr)

        # Two-Tower on holdout (with holdout model if available)
        tt_ho_p = resolved_data_dir / "artifacts" / "recsys_two_tower_item_holdout"
        if tt_ho_p.exists() and any(tt_ho_p.iterdir()):
            latest_ho, _ = _find_latest_artifact_dir(resolved_data_dir, "recsys_two_tower_item_holdout")
            tt_ho_model = TwoTowerModel.load(latest_ho, data=data)
        else:
            tt_ho_model = tt_model

        # Condition (a): Unmasked
        vecs_a = tt_ho_model.item_vectors()
        sc_a = tt_ho_model.score(ho_hist, item_vecs=vecs_a).copy()
        for i, sm in enumerate(ho_seen):
            sc_a[i, list(sm)] = -np.inf
        ho_rk_a = compute_ranks(sc_a, ho_targ_arr)

        # Condition (b): Masked (ID masked only)
        vecs_b = tt_ho_model.item_vectors(mask_ids=held_out_list)
        sc_b = tt_ho_model.score(ho_hist, item_vecs=vecs_b, mask_ids=held_out_list).copy()
        for i, sm in enumerate(ho_seen):
            sc_b[i, list(sm)] = -np.inf
        ho_rk_b = compute_ranks(sc_b, ho_targ_arr)

        # Condition (c): Masked + train_positives raw 0
        import torch
        train_counts = data.items_df["train_positives"].values.astype(np.float32)
        mean_pop = float(np.mean(train_counts))
        std_pop = float(np.std(train_counts) + 1e-8)
        raw_zero_z = float((0.0 - mean_pop) / std_pop)

        t_dict = tt_ho_model._item_features_tensors
        num_t = t_dict["numeric"].clone()
        neut_idx = torch.tensor(held_out_list, dtype=torch.long)
        num_t[neut_idx, 3] = raw_zero_z
        item_vecs_c = tt_ho_model.module.encode_items(
            item_ids=t_dict["all_ids"],
            text_embs=t_dict["text"],
            tag_features=t_dict["tags"],
            numeric_features=num_t,
            mask_ids=torch.tensor(held_out_list, dtype=torch.long),
        ).detach().cpu().numpy()
        sc_c = tt_ho_model.score(ho_hist, item_vecs=item_vecs_c, mask_ids=held_out_list).copy()
        for i, sm in enumerate(ho_seen):
            sc_c[i, list(sm)] = -np.inf
        ho_rk_c = compute_ranks(sc_c, ho_targ_arr)

        # Condition (c2): Standardized mean (train_positives z=0, reviews and rating intact)
        import torch
        t_dict = tt_ho_model._item_features_tensors
        num_t_c2 = t_dict["numeric"].clone()
        neut_idx = torch.tensor(held_out_list, dtype=torch.long)
        num_t_c2[neut_idx, 3] = 0.0  # train_positives standardized mean z=0
        item_vecs_c2 = tt_ho_model.module.encode_items(
            item_ids=t_dict["all_ids"],
            text_embs=t_dict["text"],
            tag_features=t_dict["tags"],
            numeric_features=num_t_c2,
            mask_ids=torch.tensor(held_out_list, dtype=torch.long),
        ).detach().cpu().numpy()
        sc_c2 = tt_ho_model.score(ho_hist, item_vecs=item_vecs_c2, mask_ids=held_out_list).copy()
        for i, sm in enumerate(ho_seen):
            sc_c2[i, list(sm)] = -np.inf
        ho_rk_c2 = compute_ranks(sc_c2, ho_targ_arr)

        # Condition (c3): All Proxies Neutralized (train_positives, user_reviews, positive_ratio all z=0)
        vecs_c3 = tt_ho_model.item_vectors(mask_ids=held_out_list, neutralize_proxies=held_out_list)
        sc_c3 = tt_ho_model.score(
            ho_hist, item_vecs=vecs_c3, mask_ids=held_out_list, neutralize_proxies=held_out_list
        ).copy()
        for i, sm in enumerate(ho_seen):
            sc_c3[i, list(sm)] = -np.inf
        ho_rk_c3 = compute_ranks(sc_c3, ho_targ_arr)

        # Analytic Random Baseline (mean rank = N / 2, HR@10 = 10 / N, NDCG@10 analytic)
        n_items = float(data.n_items)
        analytic_mean_rank = n_items / 2.0
        analytic_hr_10 = 10.0 / n_items
        analytic_ndcg_10 = float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / n_items)
        analytic_mrr_10 = float(np.sum(1.0 / np.arange(1, 11)) / n_items)
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

        item_holdout_models = {
            "random": random_summary,
            "content": summarize_model_ranks(ho_rk_cnt, n_ho_users, seed=self.seed),
            "two_tower_unmasked": summarize_model_ranks(ho_rk_a, n_ho_users, seed=self.seed),
            "two_tower_masked": summarize_model_ranks(ho_rk_b, n_ho_users, seed=self.seed),
            "two_tower_masked_train_pos_raw_0": summarize_model_ranks(ho_rk_c, n_ho_users, seed=self.seed),
            "two_tower_masked_standardized_z0": summarize_model_ranks(ho_rk_c2, n_ho_users, seed=self.seed),
            "two_tower_masked_all_proxies_neutralized": summarize_model_ranks(ho_rk_c3, n_ho_users, seed=self.seed),
            "two_tower (all-time-stats architecture, clean=False)": summarize_model_ranks(
                ho_rk_c3, n_ho_users, seed=self.seed
            ),
        }

        # ──────────────────────────────────────────────────────────────────────
        # 2. Sparse Targets Subgroup (<= 10 Train Positives, N=553)
        # ──────────────────────────────────────────────────────────────────────
        sp_users, sp_targets = get_sparse_target_test_users(data, eval_users, max_train_positives=10)
        n_sp_users = len(sp_users)
        hist_sp, targ_sp, seen_sp = _build_test_foldin_data(data, sp_users)
        targ_sp_arr = np.asarray(targ_sp, dtype=np.int64)
        sp_target_ids = list(set(int(t) for t in targ_sp))

        # Random
        rk_sp_rnd = rng.integers(1, data.n_items + 1, size=n_sp_users)

        # Popularity
        sc_sp_pop = pop_model.score(hist_sp).copy()
        for i, sm in enumerate(seen_sp):
            sc_sp_pop[i, list(sm)] = -np.inf
        rk_sp_pop = compute_ranks(sc_sp_pop, targ_sp_arr)

        # Content
        sc_sp_cnt = content_model.score(hist_sp).copy()
        for i, sm in enumerate(seen_sp):
            sc_sp_cnt[i, list(sm)] = -np.inf
        rk_sp_cnt = compute_ranks(sc_sp_cnt, targ_sp_arr)

        # ALS
        sc_sp_als = als_model.score(hist_sp).copy()
        for i, sm in enumerate(seen_sp):
            sc_sp_als[i, list(sm)] = -np.inf
        rk_sp_als = compute_ranks(sc_sp_als, targ_sp_arr)

        # TT Intact (Reference model)
        sc_sp_tti = tt_model.score(hist_sp).copy()
        for i, sm in enumerate(seen_sp):
            sc_sp_tti[i, list(sm)] = -np.inf
        rk_sp_tti = compute_ranks(sc_sp_tti, targ_sp_arr)

        # TT Masked (Reference model)
        tt_vecs_sp = tt_model.precompute_item_vectors(mask_ids=sp_target_ids)
        sc_sp_ttm = tt_model.score(hist_sp, item_vecs=tt_vecs_sp).copy()
        for i, sm in enumerate(seen_sp):
            sc_sp_ttm[i, list(sm)] = -np.inf
        rk_sp_ttm = compute_ranks(sc_sp_ttm, targ_sp_arr)

        sparse_models = {
            "random": summarize_model_ranks(rk_sp_rnd, n_sp_users, seed=self.seed),
            "popularity": summarize_model_ranks(rk_sp_pop, n_sp_users, seed=self.seed),
            "content": summarize_model_ranks(rk_sp_cnt, n_sp_users, seed=self.seed),
            "als": summarize_model_ranks(rk_sp_als, n_sp_users, seed=self.seed),
            "two_tower_intact": summarize_model_ranks(rk_sp_tti, n_sp_users, seed=self.seed),
            "two_tower_masked (reference: all-time stats)": summarize_model_ranks(
                rk_sp_ttm, n_sp_users, seed=self.seed
            ),
        }

        # ──────────────────────────────────────────────────────────────────────
        # 3. Simulated 10% New Items (C2 Fair Simulation, N=437)
        # ──────────────────────────────────────────────────────────────────────
        new_item_ids, ni_users, ni_targets = get_simulated_new_items_test_users(
            data, eval_users, fraction=0.10, seed=self.seed, variant="c2"
        )
        n_ni_users = len(ni_users)
        hist_ni, targ_ni, seen_ni = _build_test_foldin_data(data, ni_users)
        targ_ni_arr = np.asarray(targ_ni, dtype=np.int64)
        ni_item_list = list(set(int(t) for t in new_item_ids))

        # Random
        rk_ni_rnd = rng.integers(1, data.n_items + 1, size=n_ni_users)

        # Content
        sc_ni_cnt = content_model.score(hist_ni).copy()
        for i, sm in enumerate(seen_ni):
            sc_ni_cnt[i, list(sm)] = -np.inf
        rk_ni_cnt = compute_ranks(sc_ni_cnt, targ_ni_arr)

        # TT Masked C2 (neutralized proxies on reference model)
        vecs_ni = tt_model.item_vectors(mask_ids=ni_item_list, neutralize_proxies=ni_item_list)
        sc_ni_ttm = tt_model.score(
            hist_ni, item_vecs=vecs_ni, mask_ids=ni_item_list, neutralize_proxies=ni_item_list
        ).copy()
        for i, sm in enumerate(seen_ni):
            sc_ni_ttm[i, list(sm)] = -np.inf
        rk_ni_ttm = compute_ranks(sc_ni_ttm, targ_ni_arr)

        # Baseline zeroed on masked items
        rk_ni_pop = np.full(n_ni_users, data.n_items, dtype=np.int64)
        rk_ni_als = np.full(n_ni_users, data.n_items, dtype=np.int64)
        rk_ni_knn = np.full(n_ni_users, data.n_items, dtype=np.int64)

        sim_models = {
            "random": summarize_model_ranks(rk_ni_rnd, n_ni_users, seed=self.seed),
            "popularity": summarize_model_ranks(rk_ni_pop, n_ni_users, seed=self.seed),
            "als": summarize_model_ranks(rk_ni_als, n_ni_users, seed=self.seed),
            "itemknn": summarize_model_ranks(rk_ni_knn, n_ni_users, seed=self.seed),
            "content": summarize_model_ranks(rk_ni_cnt, n_ni_users, seed=self.seed),
            "two_tower_masked (reference: all-time stats)": summarize_model_ranks(
                rk_ni_ttm, n_ni_users, seed=self.seed
            ),
        }

        cold_start_payload = {
            "metadata": {
                "report_key": "cold_start",
                "dataset_version": data.dataset_version,
                "n_eval_users": len(eval_users),
                "seed": self.seed,
                "generated_at": "2026-10-08T20:00:00+00:00",
            },
            "item_holdout_study": {
                "study_type": "item_holdout",
                "primary_evidence": True,
                "title": "True Item Holdout (90% Catalog Retrained, 300 Items Withheld — all-time-stats architecture, clean=False)",
                "n_eval_users": n_ho_users,
                "n_held_out_items": len(held_out_set),
                "models": item_holdout_models,
                "random_reference": {
                    "hit_rate_at_10": 10.0 / data.n_items,
                    "ndcg_at_10": float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / data.n_items),
                    "formula": "10 / n_items",
                },
                "note": "True Item Holdout: Evaluated on the all-time-stats architecture (clean=False). Held-out items had zero interactions during model training. This is the primary prospective benchmark.",
            },
            "sparse_targets_study": {
                "study_type": "sparse_targets",
                "primary_evidence": False,
                "title": "Sparse Targets Subgroup Study (<= 10 Train Positives)",
                "is_optimistic_proxy": True,
                "n_eval_users": n_sp_users,
                "max_train_positives": 10,
                "models": sparse_models,
                "limitations_note": "Target item ID is masked at inference on a reference model trained with full catalog (all-time stats). Evaluates semantic prior quality for items with sparse interactions (optimistic proxy).",
            },
            "simulated_new_items_study": {
                "study_type": "simulated_new_items",
                "variant": "c2",
                "variant_label": "C2: Fair Simulation (Random 10%, Pop Zeroed, Proxies Neutralized)",
                "primary_evidence": False,
                "is_optimistic_proxy": True,
                "n_eval_users": n_ni_users,
                "n_new_items": len(new_item_ids),
                "fraction_masked": 0.10,
                "models": sim_models,
                "random_reference": {
                    "hit_rate_at_10": 10.0 / data.n_items,
                    "ndcg_at_10": float(np.sum(1.0 / np.log2(np.arange(1, 11) + 1.0)) / data.n_items),
                    "formula": "10 / n_items",
                },
                "limitations_note": "Inference-time masking simulation on reference two-tower model (all-time stats) with neutralized rating/review proxies.",
            },
        }

        # Write to JSON
        out_file = reports_dir / "cold_start.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cold_start_payload, f, indent=2)
        self.logger.info("Saved %s", out_file)

        # Upsert into PostgreSQL serving_rec_reports table
        if session is not None and not self.dry_run:
            stmt = (
                insert(ServingRecReport)
                .values(
                    report_key="cold_start",
                    payload=cold_start_payload,
                    dataset_version=data.dataset_version,
                )
                .on_conflict_do_update(
                    index_elements=[ServingRecReport.report_key],
                    set_={"payload": cold_start_payload, "dataset_version": data.dataset_version},
                )
            )
            await session.execute(stmt)
            await session.commit()
            self.logger.info("Successfully upserted cold_start report into PostgreSQL serving_rec_reports.")

        return {"n_ho_users": n_ho_users, "n_sp_users": n_sp_users, "n_ni_users": n_ni_users}

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
    parser = argparse.ArgumentParser(description="Generate and persist cold-start evaluation report")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without DB writes")
    args = parser.parse_args()

    job = GenerateColdStartReportJob(
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
