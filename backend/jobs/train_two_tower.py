"""
Batch Job: Train and Evaluate PyTorch Two-Tower Hybrid Retrieval Model — Phase 6.

Subclasses BaseJob.
Trains TwoTowerModel with content projections, ID dropout, attention pooling,
and in-batch sampled softmax loss. Evaluates against all baseline models with paired bootstrap
significance testing, cold-start simulation, and slice breakdowns.

Usage:
  python -m jobs.train_two_tower --epochs 15 --evaluate-test
  python -m jobs.train_two_tower --epochs 8 --pool attn --id-dropout 0.25 --evaluate-test
  python -m jobs.train_two_tower --no-id --tag no_id --evaluate-test
  python -m jobs.train_two_tower --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from jobs.base_job import BaseJob
from recsys.config import RecsysConfig, get_recsys_config
from recsys.data import RecsysData
from recsys.evaluate import EvaluationResult, evaluate_model, get_evaluation_user_split
from recsys.metrics import paired_bootstrap_diff
from recsys.models.als import ALSRecommender
from recsys.models.content import ContentRecommender
from recsys.models.itemknn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.two_tower import TwoTowerConfig
from recsys.registry import log_recsys_run
from recsys.train_two_tower import train_two_tower_model


class TrainTwoTowerJob(BaseJob):
    job_name = "train_two_tower"

    def __init__(
        self,
        *,
        epochs: int = 15,
        batch_size: int = 2048,
        dim: int = 64,
        pool: str = "attn",
        id_dropout: float = 0.25,
        proxy_dropout: float = 0.0,
        no_pop_proxies: bool = False,
        use_id: bool = True,
        use_text: bool = True,
        use_tags: bool = True,
        use_numeric: bool = True,
        pairs_per_epoch: int | None = None,
        lr: float = 2e-3,
        tag: str | None = None,
        evaluate_test: bool = False,
        profile: str | None = None,
        data_dir: Path | str | None = None,
        seed: int | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dim = dim
        self.pool = pool
        self.id_dropout = id_dropout
        self.proxy_dropout = proxy_dropout
        self.no_pop_proxies = no_pop_proxies
        self.use_id = use_id
        self.use_text = use_text
        self.use_tags = use_tags
        self.use_numeric = use_numeric
        self.pairs_per_epoch = pairs_per_epoch
        self.lr = lr
        self.tag = tag
        self.evaluate_test = evaluate_test
        self.profile = profile
        self.data_dir = data_dir
        self.seed = seed

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        cfg: RecsysConfig = get_recsys_config(self.profile)
        seed = self.seed if self.seed is not None else cfg.seed

        resolved_data_dir = (
            Path(self.data_dir) if self.data_dir else (cfg.data_dir or settings.resolved_recsys_data_dir)
        )
        self.logger.info("Loading recommender dataset from %s...", resolved_data_dir)
        data = RecsysData.load(resolved_data_dir)

        # 1. Disjoint User Splits
        n_tuning = cfg.eval_users if cfg.profile == "dev" else 10000
        eval_users, tuning_users = get_evaluation_user_split(
            data,
            n_eval_users=cfg.eval_users,
            n_tuning_users=n_tuning,
            seed=seed,
            persistence_dir=resolved_data_dir,
        )

        model_name = f"recsys_two_tower_{self.tag.strip()}" if self.tag else "recsys_two_tower"
        patience = 5 if cfg.profile == "dev" else 3

        tt_cfg = TwoTowerConfig(
            n_items=data.n_items,
            text_dim=384,
            tag_dim=300,
            numeric_dim=6 if self.no_pop_proxies else 9,
            id_dim=64,
            text_proj_dim=64,
            tag_proj_dim=32,
            numeric_proj_dim=16,
            hidden_dim=256,
            output_dim=self.dim,
            pool=self.pool,
            id_dropout=self.id_dropout,
            proxy_dropout=self.proxy_dropout,
            no_pop_proxies=self.no_pop_proxies,
            use_id=self.use_id,
            use_text=self.use_text,
            use_tags=self.use_tags,
            use_numeric=self.use_numeric,
            max_history=cfg.max_history,
        )

        self.logger.info("═══════════════════════════════════════════════════════════")
        self.logger.info(
            "▶ Training Two-Tower Retrieval Model: %s (epochs=%d, batch_size=%d, pool=%s, id_drop=%.2f)...",
            model_name,
            self.epochs,
            self.batch_size,
            self.pool,
            self.id_dropout,
        )

        # 2. Train Two-Tower with Early Stopping on Tuning Users
        model, training_curves = train_two_tower_model(
            data=data,
            tuning_users=tuning_users,
            cfg=tt_cfg,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=patience,
            pairs_per_epoch=self.pairs_per_epoch,
            seed=seed,
        )
        model.name = model_name

        # 3. Final Evaluation on Fixed Evaluation Users
        self.logger.info("Evaluating %s on validation split (%d eval users)...", model_name, len(eval_users))
        val_res: EvaluationResult = evaluate_model(
            model=model,
            data=data,
            split="val",
            k_list=(10, 20),
            user_indices=eval_users,
            max_history=cfg.max_history,
            seed=seed,
        )

        test_res: EvaluationResult | None = None
        if self.evaluate_test:
            self.logger.info("Evaluating %s on test split (%d eval users)...", model_name, len(eval_users))
            test_res = evaluate_model(
                model=model,
                data=data,
                split="test",
                k_list=(10, 20),
                user_indices=eval_users,
                max_history=cfg.max_history,
                seed=seed,
            )

        # 4. Serialize Model Artifacts
        run_uuid = uuid.uuid4()
        artifact_dir = self.resolve_path(
            "..", "data", "recsys", "artifacts", model_name, str(run_uuid)
        ).resolve()

        if not self.dry_run:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model.save(artifact_dir)

            if "rank" in val_res.per_user:
                np.save(artifact_dir / "ranks_val.npy", val_res.per_user["rank"])
            if test_res is not None and "rank" in test_res.per_user:
                np.save(artifact_dir / "ranks_test.npy", test_res.per_user["rank"])

            self.logger.info("Saved Two-Tower artifacts to %s", artifact_dir)

            # 5. Log Run to MLflow & Database (stage='candidate', no promote_champion)
            await log_recsys_run(
                model_name=model_name,
                family="two_tower",
                dataset_version=data.dataset_version,
                hyperparameters=model.hyperparameters(),
                val_metrics=val_res,
                test_metrics=test_res,
                artifact_dir=artifact_dir,
                extra={
                    "seed": seed,
                    "profile": cfg.profile,
                    "pool": self.pool,
                    "id_dropout": self.id_dropout,
                    "best_epoch": training_curves["best_epoch"],
                    "epochs_completed": training_curves["epochs_completed"],
                },
                session=session,
            )

        # 6. Fit Quick Baselines for Side-by-Side Significance Testing
        self.logger.info("Evaluating reference baselines for paired bootstrap comparison...")
        pop_model = PopularityRecommender()
        pop_model.fit(data)
        pop_val = evaluate_model(pop_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=seed)
        pop_test = evaluate_model(pop_model, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=seed) if self.evaluate_test else None

        knn_model = ItemKNNRecommender(k_neighbors=100, shrink=10.0)
        knn_model.fit(data)
        knn_val = evaluate_model(knn_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=seed)
        knn_test = evaluate_model(knn_model, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=seed) if self.evaluate_test else None

        als_model = ALSRecommender(factors=64, regularization=0.35, alpha=5.0, iterations=15, seed=seed)
        als_model.fit(data)
        als_val = evaluate_model(als_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=seed)
        als_test = evaluate_model(als_model, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=seed) if self.evaluate_test else None

        content_model = ContentRecommender(weight_tags=0.5)
        content_model.fit(data)
        content_val = evaluate_model(content_model, data, split="val", k_list=(10, 20), user_indices=eval_users, seed=seed)
        content_test = evaluate_model(content_model, data, split="test", k_list=(10, 20), user_indices=eval_users, seed=seed) if self.evaluate_test else None

        # 7. Print Comparative Tables
        all_results = [
            {"name": model_name, "val": val_res, "test": test_res},
            {"name": "recsys_itemknn", "val": knn_val, "test": knn_test},
            {"name": "recsys_als", "val": als_val, "test": als_test},
            {"name": "recsys_popularity", "val": pop_val, "test": pop_test},
            {"name": "recsys_content", "val": content_val, "test": content_test},
        ]

        self._print_comparison_report(
            all_results=all_results,
            two_tower_val=val_res,
            two_tower_test=test_res,
            pop_val=pop_val,
            pop_test=pop_test,
            als_val=als_val,
            als_test=als_test,
            knn_val=knn_val,
            knn_test=knn_test,
            dataset_version=data.dataset_version,
            eval_users_count=len(eval_users),
            seed=seed,
            training_curves=training_curves,
        )

        return {
            "model_name": model_name,
            "best_epoch": training_curves["best_epoch"],
            "val_metrics": val_res.metrics,
            "test_metrics": test_res.metrics if test_res else None,
            "artifact_dir": str(artifact_dir),
        }

    def _print_comparison_report(
        self,
        all_results: list[dict[str, Any]],
        two_tower_val: EvaluationResult,
        two_tower_test: EvaluationResult | None,
        pop_val: EvaluationResult,
        pop_test: EvaluationResult | None,
        als_val: EvaluationResult,
        als_test: EvaluationResult | None,
        knn_val: EvaluationResult,
        knn_test: EvaluationResult | None,
        dataset_version: str,
        eval_users_count: int,
        seed: int,
        training_curves: dict[str, Any],
    ) -> None:
        div_main = "=" * 140
        sub_main = "-" * 140

        # ── EPOCH LOG SUMMARY ──
        print(f"\n{div_main}")
        print("📈 Two-Tower Epoch Training History (Loss & Tuning Validation NDCG@10)")
        print(div_main)
        print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Tau':<10} | {'Tuning Val NDCG@10':<22} | {'Tuning Val HR@10':<20} | {'Epoch Time (s)':<15}")
        print(sub_main)
        for h in training_curves.get("history", []):
            star = " ★" if h["epoch"] == training_curves["best_epoch"] else ""
            print(
                f"{h['epoch']:<8} | {h['train_loss']:<12.4f} | {h['tau']:<10.4f} | "
                f"{h['val_ndcg_at_10']:<22.4f}{star} | {h['val_hr_at_10']:<20.4f} | {h['epoch_sec']:<15.2f}"
            )
        print(f"{div_main}\n")

        # ── TABLE 1: MAIN LEADERBOARD ──
        sorted_results = sorted(
            all_results,
            key=lambda r: r["val"].metrics.get("ndcg_at_10", 0.0),
            reverse=True,
        )

        print(f"{div_main}")
        print(f"🏆 SteamIQ Comparative Recommender Leaderboard | Dataset: {dataset_version} | Eval Users: {eval_users_count} | Seed: {seed}")
        print(div_main)
        print(
            f"{'Model':<20} | {'Split':<5} | {'HR@10 (95% CI)':<23} | {'HR@20 (95% CI)':<23} | "
            f"{'NDCG@10 (95% CI)':<23} | {'NDCG@20 (95% CI)':<23} | {'MRR@10':<7} | {'Coverage':<8} | {'Diversity':<9}"
        )
        print(sub_main)

        for rec in sorted_results:
            name = rec["name"]
            vm = rec["val"].metrics
            v_hr10 = f"{vm.get('hit_rate_at_10', 0.0):.4f} [{vm.get('hit_rate_at_10_lo95', 0.0):.4f},{vm.get('hit_rate_at_10_hi95', 0.0):.4f}]"
            v_hr20 = f"{vm.get('hit_rate_at_20', 0.0):.4f} [{vm.get('hit_rate_at_20_lo95', 0.0):.4f},{vm.get('hit_rate_at_20_hi95', 0.0):.4f}]"
            v_ndcg10 = f"{vm.get('ndcg_at_10', 0.0):.4f} [{vm.get('ndcg_at_10_lo95', 0.0):.4f},{vm.get('ndcg_at_10_hi95', 0.0):.4f}]"
            v_ndcg20 = f"{vm.get('ndcg_at_20', 0.0):.4f} [{vm.get('ndcg_at_20_lo95', 0.0):.4f},{vm.get('ndcg_at_20_hi95', 0.0):.4f}]"
            v_mrr10 = f"{vm.get('mrr_at_10', 0.0):.4f}"
            v_cov = f"{vm.get('catalog_coverage_at_10', 0.0):.4f}"
            v_div = f"{vm.get('diversity_at_10', 0.0):.4f}"

            print(
                f"{name:<20} | {'val':<5} | {v_hr10:<23} | {v_hr20:<23} | "
                f"{v_ndcg10:<23} | {v_ndcg20:<23} | {v_mrr10:<7} | {v_cov:<8} | {v_div:<9}"
            )

            if rec["test"] is not None:
                tm = rec["test"].metrics
                t_hr10 = f"{tm.get('hit_rate_at_10', 0.0):.4f} [{tm.get('hit_rate_at_10_lo95', 0.0):.4f},{tm.get('hit_rate_at_10_hi95', 0.0):.4f}]"
                t_hr20 = f"{tm.get('hit_rate_at_20', 0.0):.4f} [{tm.get('hit_rate_at_20_lo95', 0.0):.4f},{tm.get('hit_rate_at_20_hi95', 0.0):.4f}]"
                t_ndcg10 = f"{tm.get('ndcg_at_10', 0.0):.4f} [{tm.get('ndcg_at_10_lo95', 0.0):.4f},{tm.get('ndcg_at_10_hi95', 0.0):.4f}]"
                t_ndcg20 = f"{tm.get('ndcg_at_20', 0.0):.4f} [{tm.get('ndcg_at_20_lo95', 0.0):.4f},{tm.get('ndcg_at_20_hi95', 0.0):.4f}]"
                t_mrr10 = f"{tm.get('mrr_at_10', 0.0):.4f}"
                t_cov = f"{tm.get('catalog_coverage_at_10', 0.0):.4f}"
                t_div = f"{tm.get('diversity_at_10', 0.0):.4f}"

                print(
                    f"{name:<20} | {'test':<5} | {t_hr10:<23} | {t_hr20:<23} | "
                    f"{t_ndcg10:<23} | {t_ndcg20:<23} | {t_mrr10:<7} | {t_cov:<8} | {t_div:<9}"
                )

        print(f"{div_main}\n")

        # ── TABLE 2: PAIRED BOOTSTRAP SIGNIFICANCE (TWO-TOWER VS BASELINES) ──
        div_paired = "=" * 115
        sub_paired = "-" * 115
        print(f"{div_paired}")
        print("📊 Paired Bootstrap Significance: Two-Tower vs Reference Baselines (Mean Diff & 95% CI)")
        print(f"{div_paired}")
        print(
            f"{'Comparison':<30} | {'Split':<5} | {'NDCG@10 Diff (95% CI)':<32} | "
            f"{'HR@10 Diff (95% CI)':<32} | {'Significant?'}"
        )
        print(f"{sub_paired}")

        comparisons = [
            ("Two-Tower vs Popularity", two_tower_val, pop_val, two_tower_test, pop_test),
            ("Two-Tower vs ALS", two_tower_val, als_val, two_tower_test, als_test),
            ("Two-Tower vs Item-kNN", two_tower_val, knn_val, two_tower_test, knn_test),
        ]

        for label, tt_v, base_v, tt_t, base_t in comparisons:
            # Val comparison
            tt_ndcg_v = tt_v.per_user["ndcg_at_10"]
            b_ndcg_v = base_v.per_user["ndcg_at_10"]
            tt_hr_v = tt_v.per_user["hit_rate_at_10"]
            b_hr_v = base_v.per_user["hit_rate_at_10"]

            d_ndcg, n_lo, n_hi = paired_bootstrap_diff(tt_ndcg_v, b_ndcg_v, n_boot=1000, seed=seed)
            d_hr, h_lo, h_hi = paired_bootstrap_diff(tt_hr_v, b_hr_v, n_boot=1000, seed=seed)

            sig = "YES (p<0.05)" if n_lo > 0.0 else ("NO" if n_hi < 0.0 else "TIE")
            print(
                f"{label:<30} | {'val':<5} | {d_ndcg:+.4f} [{n_lo:+.4f}, {n_hi:+.4f}]      | "
                f"{d_hr:+.4f} [{h_lo:+.4f}, {h_hi:+.4f}]      | {sig}"
            )

            # Test comparison
            if tt_t is not None and base_t is not None:
                tt_ndcg_t = tt_t.per_user["ndcg_at_10"]
                b_ndcg_t = base_t.per_user["ndcg_at_10"]
                tt_hr_t = tt_t.per_user["hit_rate_at_10"]
                b_hr_t = base_t.per_user["hit_rate_at_10"]

                td_ndcg, tn_lo, tn_hi = paired_bootstrap_diff(tt_ndcg_t, b_ndcg_t, n_boot=1000, seed=seed)
                td_hr, th_lo, th_hi = paired_bootstrap_diff(tt_hr_t, b_hr_t, n_boot=1000, seed=seed)

                t_sig = "YES (p<0.05)" if tn_lo > 0.0 else ("NO" if tn_hi < 0.0 else "TIE")
                print(
                    f"{label:<30} | {'test':<5} | {td_ndcg:+.4f} [{tn_lo:+.4f}, {tn_hi:+.4f}]      | "
                    f"{td_hr:+.4f} [{th_lo:+.4f}, {th_hi:+.4f}]      | {t_sig}"
                )

        print(f"{div_paired}\n")

        # ── TABLE 3: SLICE BREAKDOWN (NDCG@10) ──
        div_slice = "=" * 115
        sub_slice = "-" * 115
        print(f"{div_slice}")
        print("🔍 Sliced Evaluation Breakdown (NDCG@10 across Popularity & History Length Buckets)")
        print(f"{div_slice}")
        print(
            f"{'Model':<20} | {'Split':<5} | {'Pop Head (20%)':<14} | {'Pop Torso (30%)':<15} | "
            f"{'Pop Tail (50%)':<14} | {'Hist 3-4':<11} | {'Hist 5-9':<11} | {'Hist 10+':<11}"
        )
        print(f"{sub_slice}")

        for rec in sorted_results:
            name = rec["name"]
            vs = rec["val"].slices
            print(
                f"{name:<20} | {'val':<5} | {vs.get('pop_head', {}).get('ndcg_at_10', 0.0):<14.4f} | "
                f"{vs.get('pop_torso', {}).get('ndcg_at_10', 0.0):<15.4f} | {vs.get('pop_tail', {}).get('ndcg_at_10', 0.0):<14.4f} | "
                f"{vs.get('hist_3_4', {}).get('ndcg_at_10', 0.0):<11.4f} | {vs.get('hist_5_9', {}).get('ndcg_at_10', 0.0):<11.4f} | "
                f"{vs.get('hist_10_plus', {}).get('ndcg_at_10', 0.0):<11.4f}"
            )

            if rec["test"] is not None:
                ts = rec["test"].slices
                print(
                    f"{name:<20} | {'test':<5} | {ts.get('pop_head', {}).get('ndcg_at_10', 0.0):<14.4f} | "
                    f"{ts.get('pop_torso', {}).get('ndcg_at_10', 0.0):<15.4f} | {ts.get('pop_tail', {}).get('ndcg_at_10', 0.0):<14.4f} | "
                    f"{ts.get('hist_3_4', {}).get('ndcg_at_10', 0.0):<11.4f} | {ts.get('hist_5_9', {}).get('ndcg_at_10', 0.0):<11.4f} | "
                    f"{ts.get('hist_10_plus', {}).get('ndcg_at_10', 0.0):<11.4f}"
                )

        print(f"{div_slice}\n")

    def run(self) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(
        description="Train and evaluate PyTorch Two-Tower Hybrid Retrieval Model"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--pool", type=str, default="attn", choices=["attn", "mean"], help="History pooling strategy")
    parser.add_argument("--id-dropout", type=float, default=0.25, help="Probability of dropping item ID during training")
    parser.add_argument("--proxy-dropout", type=float, default=0.0, help="Probability of zeroing item ID and numeric features during training")
    parser.add_argument("--no-pop-proxies", action="store_true", help="Remove user_reviews and positive_ratio from numeric branch")
    parser.add_argument("--no-id", action="store_true", help="Disable item ID branch (ablation)")
    parser.add_argument("--no-text", action="store_true", help="Disable MiniLM text branch (ablation)")
    parser.add_argument("--no-tags", action="store_true", help="Disable tag multi-hot branch (ablation)")
    parser.add_argument("--no-numeric", action="store_true", help="Disable numeric features branch (ablation)")
    parser.add_argument("--pairs-per-epoch", type=int, default=None, help="Max pairs sampled per epoch")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--tag", type=str, default=None, help="Custom tag name for ablation runs")
    parser.add_argument("--evaluate-test", "--test", action="store_true", help="Evaluate final model on test split")
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "full"], help="Recsys profile override")
    parser.add_argument("--data-dir", type=str, default=None, help="Recsys precomputed data directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without MLflow or DB writes")
    args = parser.parse_args()

    job = TrainTwoTowerJob(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dim=args.dim,
        pool=args.pool,
        id_dropout=args.id_dropout,
        proxy_dropout=args.proxy_dropout,
        no_pop_proxies=args.no_pop_proxies,
        use_id=not args.no_id,
        use_text=not args.no_text,
        use_tags=not args.no_tags,
        use_numeric=not args.no_numeric,
        pairs_per_epoch=args.pairs_per_epoch,
        lr=args.lr,
        tag=args.tag,
        evaluate_test=args.evaluate_test,
        profile=args.profile,
        data_dir=args.data_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
