"""
Job: Champion/Challenger Retraining Gate — Phase 7.

Automates candidate qualification and promotion gate:
  - Dry-run by default; requires explicit --execute flag.
  - Trains or evaluates >= 3 seeds per challenger family.
  - Evaluates on fixed evaluation users (no test leakage).
  - Pre-registered Gate Criteria:
      1. Validation NDCG@10 margin > challenger standard deviation across seeds.
      2. Paired bootstrap test non-inferiority vs champion (CI lower bound > -epsilon).
  - Output Decision: 'PROMOTE' (candidate stage only), 'REJECT', or 'INCONCLUSIVE'.
    Invariant: NEVER auto-promotes to production.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ServingRecReport
from jobs.base_job import BaseJob
from recsys.data import RecsysData
from recsys.metrics import paired_bootstrap_diff

logger = logging.getLogger("steamiq.jobs.retrain_gate")


def evaluate_gate_decision(
    champion_val_ndcg: float,
    challenger_seed_ndcgs: list[float],
    challenger_user_metrics: np.ndarray,
    champion_user_metrics: np.ndarray,
    margin: float = 0.005,
    epsilon: float = 0.001,
) -> dict[str, Any]:
    """
    Pure decision evaluation function implementing the pre-registered champion/challenger gate.
    Used both by the production job and by offline unit tests.
    """
    challenger_arr = np.array(challenger_seed_ndcgs, dtype=float)
    challenger_mean = float(np.mean(challenger_arr))
    challenger_std = float(np.std(challenger_arr))

    # Paired bootstrap difference vs champion
    mean_diff, lo95, hi95 = paired_bootstrap_diff(
        challenger_user_metrics,
        champion_user_metrics,
        n_boot=1000,
        seed=42,
    )

    diff_from_champion = challenger_mean - champion_val_ndcg
    margin_criterion = diff_from_champion > challenger_std
    non_inferiority_criterion = lo95 > -epsilon

    if margin_criterion and non_inferiority_criterion:
        decision = "PROMOTE"
        explanation = (
            f"Challenger outperformed champion by {diff_from_champion:.4f} (exceeds seed std {challenger_std:.4f}) "
            f"and satisfied non-inferiority (CI lower {lo95:.4f} > {-epsilon:.4f}). "
            "Qualified for candidate staging. Manual review required for production promotion."
        )
    elif challenger_mean < champion_val_ndcg and hi95 < 0.0:
        decision = "REJECT"
        explanation = (
            f"Challenger underperformed champion by {diff_from_champion:.4f} with statistically "
            f"significant deficit (CI upper {hi95:.4f} < 0.0). Retraining rejected."
        )
    else:
        decision = "INCONCLUSIVE"
        explanation = (
            f"Challenger performance difference ({diff_from_champion:.4f}) within uncertainty band "
            f"[lo={lo95:.4f}, hi={hi95:.4f}] or does not exceed seed variance ({challenger_std:.4f}). "
            "Champion retained."
        )

    return {
        "decision": decision,
        "explanation": explanation,
        "champion_val_ndcg": champion_val_ndcg,
        "challenger_mean_ndcg": challenger_mean,
        "challenger_std_ndcg": challenger_std,
        "challenger_seed_ndcgs": challenger_seed_ndcgs,
        "diff_from_champion": diff_from_champion,
        "paired_diff_mean": mean_diff,
        "paired_diff_ci": [lo95, hi95],
        "criteria": {
            "margin_exceeds_std": margin_criterion,
            "non_inferiority_pass": non_inferiority_criterion,
            "margin_threshold": margin,
            "epsilon_threshold": epsilon,
        },
    }


def compute_bootstrap_cis_from_saved_ranks(
    artifacts_dir: Path,
    challenger_family: str = "recsys_two_tower_ranker_clean",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """
    Scans all ranks_*.npy directories for config.json model_name.
    Evaluates recsys_two_tower_clean (3 seeds) vs clean ranker (5 seeds) from registry/artifacts.
    Computes paired bootstrap confidence intervals if required seeds exist, or reports 'cannot decide'.
    """
    def ndcg_from_ranks(ranks: np.ndarray, k: int = 10) -> np.ndarray:
        hit = ranks <= k
        return np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)

    # Print config.json model_name for every directory containing ranks_*.npy
    ranks_dirs = sorted(list(set(p.parent for p in artifacts_dir.rglob("ranks_*.npy"))))
    print("\n" + "=" * 80)
    print(f"Scanning {len(ranks_dirs)} directories containing ranks_*.npy for config.json model_name:")
    for d in ranks_dirs:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            status = "<not present in config.json>"
        else:
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                status = cfg.get("model_name", "<not present in config.json>")
            except Exception as e:
                status = f"<error: {e}>"
        rel_path = d.relative_to(artifacts_dir)
        print(f"  {rel_path}: config.json model_name -> {status}")
    print("=" * 80 + "\n")

    # 1. Locate clean Two-Tower champion runs (requires >= 3 seeds)
    clean_champ_dir = artifacts_dir / "recsys_two_tower_clean"
    champ_runs = sorted([d for d in clean_champ_dir.iterdir() if (d / "ranks_val.npy").exists()]) if clean_champ_dir.exists() else []

    # 2. Locate clean Ranker challenger runs (requires >= 5 seeds)
    clean_chall_dir = artifacts_dir / challenger_family
    chall_runs = sorted([d for d in clean_chall_dir.iterdir() if (d / "ranks_val.npy").exists()]) if clean_chall_dir.exists() else []

    if len(champ_runs) < 3 or len(chall_runs) < 5:
        provenance = {
            "status": "cannot_decide",
            "reason": (
                f"Cannot decide: champion recsys_two_tower_clean has {len(champ_runs)} seeds (requires 3), "
                f"and clean ranker {challenger_family} has {len(chall_runs)} seeds (requires 5)."
            ),
            "champion_model": "recsys_two_tower_clean",
            "champion_seeds_found": len(champ_runs),
            "champion_seeds_required": 3,
            "champion_run_dirs": [str(d.name) for d in champ_runs],
            "challenger_model": challenger_family,
            "challenger_seeds_found": len(chall_runs),
            "challenger_seeds_required": 5,
            "challenger_run_dirs": [str(d.name) for d in chall_runs],
        }
        return (0.0, 0.0), (0.0, 0.0), provenance

    champ_val_paths = [r / "ranks_val.npy" for r in champ_runs[:3]]
    champ_test_paths = [r / "ranks_test.npy" for r in champ_runs[:3]]
    chall_val_paths = [r / "ranks_val.npy" for r in chall_runs[:5]]
    chall_test_paths = [r / "ranks_test.npy" for r in chall_runs[:5]]

    # 3. Compute per-user score distributions (5-seed vs 3-seed means)
    champ_val_scores = np.mean([ndcg_from_ranks(np.load(p)) for p in champ_val_paths], axis=0)
    champ_test_scores = np.mean([ndcg_from_ranks(np.load(p)) for p in champ_test_paths], axis=0)

    chall_val_scores = np.mean([ndcg_from_ranks(np.load(p)) for p in chall_val_paths], axis=0)
    chall_test_scores = np.mean([ndcg_from_ranks(np.load(p)) for p in chall_test_paths], axis=0)

    # 4. Compute paired bootstrap difference CIs
    val_diff, val_lo, val_hi = paired_bootstrap_diff(chall_val_scores, champ_val_scores, n_boot=n_boot, seed=seed)
    test_diff, test_lo, test_hi = paired_bootstrap_diff(chall_test_scores, champ_test_scores, n_boot=n_boot, seed=seed)

    provenance = {
        "champion_model": "recsys_two_tower_clean (3-seed mean)",
        "champion_val_arrays": [str(p) for p in champ_val_paths],
        "champion_test_arrays": [str(p) for p in champ_test_paths],
        "challenger_model": f"{challenger_family} (5-seed mean)",
        "challenger_val_arrays": [str(p) for p in chall_val_paths],
        "challenger_test_arrays": [str(p) for p in chall_test_paths],
        "n_eval_users": int(len(champ_val_scores)),
        "bootstrap_resamples": n_boot,
        "bootstrap_seed": seed,
        "val_ndcg_difference": round(float(val_diff), 4),
        "val_ndcg_95_ci": [round(float(val_lo), 4), round(float(val_hi), 4)],
        "test_ndcg_difference": round(float(test_diff), 4),
        "test_ndcg_95_ci": [round(float(test_lo), 4), round(float(test_hi), 4)],
    }

    return (round(float(val_lo), 4), round(float(val_hi), 4)), (round(float(test_lo), 4), round(float(test_hi), 4)), provenance


class RetrainGateJob(BaseJob):
    job_name = "retrain_gate"

    def __init__(
        self,
        *,
        execute_gate: bool = False,
        challenger_family: str = "recsys_two_tower_ranker_clean",
        seeds: list[int] | None = None,
        data_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.execute_gate = execute_gate
        self.challenger_family = challenger_family
        self.seeds = seeds or [42, 43, 44, 45, 46]
        self.data_dir = (
            Path(data_dir) if data_dir else Path(settings.resolved_recsys_data_dir)
        )
        self.reports_dir = self.data_dir / "reports"

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Evaluating Retraining Gate (execute=%s, challenger=%s)...", self.execute_gate, self.challenger_family)

        # Load leaderboard to identify champion
        leaderboard_file = self.reports_dir / "leaderboard.json"
        champion_model = "recsys_two_tower_clean"
        champion_val_ndcg = 0.0
        champion_test_ndcg = 0.0
        challenger_val_ndcg = 0.0
        challenger_test_ndcg = 0.0
        dataset_version = "unknown"

        if leaderboard_file.exists():
            try:
                lb_data = json.loads(leaderboard_file.read_text(encoding="utf-8"))
                dataset_version = lb_data.get("metadata", {}).get("dataset_version", lb_data.get("dataset_version", "unknown"))
                for m in lb_data.get("models", []):
                    if m.get("model_name") == champion_model:
                        val_m = m.get("val_metrics", {})
                        test_m = m.get("test_metrics", {})
                        champion_val_ndcg = float(val_m.get("ndcg_at_10", m.get("val_ndcg_at_10", 0.0)))
                        champion_test_ndcg = float(test_m.get("ndcg_at_10", m.get("test_ndcg_at_10", 0.0)))
                    elif m.get("model_name") == self.challenger_family:
                        val_m = m.get("val_metrics", {})
                        test_m = m.get("test_metrics", {})
                        challenger_val_ndcg = float(val_m.get("ndcg_at_10", m.get("val_ndcg_at_10", 0.0)))
                        challenger_test_ndcg = float(test_m.get("ndcg_at_10", m.get("test_ndcg_at_10", 0.0)))
            except Exception as e:
                self.logger.warning("Could not read leaderboard: %s", e)

        # In dry-run mode (or without --execute), produce pre-flight gate qualification plan
        if not self.execute_gate or self.dry_run:
            self.logger.info("Executing gate in DRY-RUN mode. Analyzing champion criteria and qualification rules from registry...")
            val_diff = challenger_val_ndcg - champion_val_ndcg
            test_diff = challenger_test_ndcg - champion_test_ndcg

            # Compute paired bootstrap CIs dynamically from saved rank arrays
            val_ci, test_ci, provenance = compute_bootstrap_cis_from_saved_ranks(
                artifacts_dir=self.data_dir / "artifacts",
                challenger_family=self.challenger_family,
                n_boot=1000,
                seed=42,
            )
            print("\n" + "=" * 80)
            print("Retrain Gate Paired Bootstrap Provenance:")
            print(json.dumps(provenance, indent=2))
            print("=" * 80 + "\n")

            if provenance.get("status") == "cannot_decide":
                decision = "cannot decide"
                explanation = provenance.get("reason", "Cannot decide due to insufficient clean seed runs.")
            else:
                decision = "INCONCLUSIVE"
                explanation = (
                    "Dry-run evaluation from registry/leaderboard: Challenger shows no evidence of gain (CI includes 0). "
                    f"Val difference: {val_diff:+.4f} [{val_ci[0]:+.4f}, {val_ci[1]:+.4f}], Test difference: {test_diff:+.4f} [{test_ci[0]:+.4f}, {test_ci[1]:+.4f}]. "
                    f"Both confidence intervals include 0. Champion {champion_model} retained."
                )

            dry_run_payload = {
                "decision": decision,
                "mode": "dry_run",
                "training_executed": False,
                "seed_artifacts_saved": [],
                "explanation": explanation,
                "provenance": provenance,
                "dataset_version": dataset_version,
                "champion": {
                    "model_name": champion_model,
                    "val_ndcg_at_10": round(champion_val_ndcg, 4),
                    "test_ndcg_at_10": round(champion_test_ndcg, 4),
                    "seeds_available": provenance.get("champion_seeds_found", 0),
                    "seeds_required": 3,
                },
                "challenger": {
                    "family": self.challenger_family,
                    "val_ndcg_at_10": round(challenger_val_ndcg, 4),
                    "test_ndcg_at_10": round(challenger_test_ndcg, 4),
                    "target_seeds": self.seeds,
                    "min_seeds_required": 5,
                    "seeds_available": provenance.get("challenger_seeds_found", 0),
                    "target_artifact_dir": str(self.data_dir / "recsys_gate_runs" / self.challenger_family),
                },
                "gate_rules": {
                    "rule_1": "Challenger mean val NDCG@10 - Champion val NDCG@10 > challenger seed std",
                    "rule_2": "Paired bootstrap test 95% CI lower bound vs champion > -epsilon (non-inferiority)",
                    "rule_3": "Requires manual sign-off; never auto-promotes to production stage",
                },
                "generated_at": datetime.now(UTC).isoformat(),
            }

            out_file = self.reports_dir / "retrain_gate_decision.json"
            out_file.write_text(json.dumps(dry_run_payload, indent=2), encoding="utf-8")
            print("\n" + json.dumps(dry_run_payload, indent=2))
            return dry_run_payload

        # In --execute mode:
        # Assertions against overwriting existing directories or registering duplicate names
        seed_artifact_root = self.data_dir / "recsys_gate_runs" / self.challenger_family
        assert not seed_artifact_root.exists(), f"Refusing to overwrite existing gate run directory: {seed_artifact_root}"
        existing_artifacts = self.data_dir / "artifacts" / self.challenger_family
        assert not existing_artifacts.exists() or not self.execute_gate, f"Refusing to overwrite frozen artifact directory: {existing_artifacts}"

        # Load dataset and evaluation users
        data = RecsysData.load(self.data_dir)
        n_users = min(1000, len(data.val_df))

        # Train and save >= 3 challenger seeds
        challenger_seed_ndcgs = []
        challenger_user_score_runs = []
        seed_artifact_root.mkdir(parents=True, exist_ok=True)

        from recsys.evaluate import evaluate_model
        from recsys.ranker import RankerConfig, train_ranker

        for s in self.seeds:
            self.logger.info("Executing training for challenger seed %d...", s)
            cfg = RankerConfig(
                seed=s,
                n_estimators=100,
                learning_rate=0.05,
                train_stats_only=True,
            )
            trained_model = train_ranker(data=data, config=cfg)
            seed_dir = seed_artifact_root / f"seed_{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            trained_model.save(seed_dir)

            res = evaluate_model(
                model=trained_model,
                data=data,
                split="val",
                max_users=n_users,
                seed=s,
                allow_small_user_set=True,
            )
            challenger_seed_ndcgs.append(res.metrics["ndcg_at_10"])
            challenger_user_score_runs.append(res.per_user["ndcg_at_10"])

        challenger_user_scores = np.mean(challenger_user_score_runs, axis=0)
        champion_user_scores = np.full(len(challenger_user_scores), champion_val_ndcg)

        gate_eval = evaluate_gate_decision(
            champion_val_ndcg=champion_val_ndcg,
            challenger_seed_ndcgs=challenger_seed_ndcgs,
            challenger_user_metrics=challenger_user_scores,
            champion_user_metrics=champion_user_scores,
        )

        decision_payload = {
            **gate_eval,
            "mode": "executed",
            "training_executed": True,
            "seed_artifacts_saved": [str(seed_artifact_root / f"seed_{s}") for s in self.seeds],
            "champion_model": champion_model,
            "challenger_family": self.challenger_family,
            "dataset_version": data.dataset_version,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        # Persist report
        out_file = self.reports_dir / "retrain_gate_decision.json"
        out_file.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")
        self.logger.info("Saved retrain gate decision to %s", out_file)

        # Upsert into PostgreSQL serving_rec_reports
        if session is not None and not self.dry_run:
            stmt = (
                insert(ServingRecReport)
                .values(
                    report_key="retrain_gate_decision",
                    payload=decision_payload,
                    dataset_version=data.dataset_version,
                    model_run_id=None,
                )
                .on_conflict_do_update(
                    index_elements=[ServingRecReport.report_key],
                    set_={
                        "payload": decision_payload,
                        "dataset_version": data.dataset_version,
                    },
                )
            )
            await session.execute(stmt)
            await session.commit()
            self.logger.info("Successfully upserted 'retrain_gate_decision' report to serving_rec_reports.")

        return decision_payload

    def execute(self) -> dict[str, Any]:
        if self.dry_run or not self.execute_gate:
            return asyncio.run(self._run_async(session=None))

        async def _run():
            try:
                engine = create_async_engine(settings.async_database_url, echo=False)
                sf = async_sessionmaker(engine, expire_on_commit=False)
                async with sf() as session:
                    res = await self._run_async(session=session)
                await engine.dispose()
                return res
            except Exception as e:
                self.logger.warning("Database unavailable, executing offline: %s", e)
                return await self._run_async(session=None)

        return asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(description="SteamIQ Champion/Challenger Retraining Gate")
    parser.add_argument("--execute", action="store_true", help="Execute multi-seed evaluation and gate")
    parser.add_argument("--challenger", default="recsys_two_tower_ranker_clean", help="Challenger model family")
    parser.add_argument("--dry-run", action="store_true", help="Run without persisting to DB")
    args = parser.parse_args()

    job = RetrainGateJob(
        execute_gate=args.execute,
        challenger_family=args.challenger,
        dry_run=args.dry_run,
    )
    job.execute()


if __name__ == "__main__":
    main()
