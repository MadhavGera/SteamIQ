"""
Statistical Drift Detector Validation and Power Simulation Job — Phase 7.

Validates:
  1. Injects synthetic shifts of known magnitude into reference samples:
     - Mean shifts: delta in [0.0, 0.1, 0.25, 0.5, 1.0] * std
     - Variance shifts: delta in [0.0, 0.2, 0.5, 1.0]
     - Categorical / popularity shifts: concentration skew
  2. Evaluates empirical False Positive Rate (FPR) on null resamples (delta = 0).
  3. Evaluates Statistical Power (Detection Rate) vs shift magnitude for PSI, KS, and JSD.
  4. Computes 95% bootstrap confidence intervals for detection rates.
  5. Persists report 'monitoring_detector_power' to disk and serving_rec_reports table.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from db.models import ServingRecReport
from jobs.base_job import BaseJob
from mlops.drift import calculate_js_distance, calculate_ks, calculate_psi
from recsys.metrics import bootstrap_ci

logger = logging.getLogger("steamiq.jobs.validate_drift_detectors")


class ValidateDriftDetectorsJob(BaseJob):
    job_name = "validate_drift_detectors"

    def __init__(
        self,
        *,
        n_trials: int = 100,
        sample_size: int = 1000,
        seed: int = 42,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.seed = seed
        self.reports_dir = Path(settings.resolved_recsys_data_dir) / "reports"

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Starting statistical drift detector validation (trials=%d, N=%d)...", self.n_trials, self.sample_size)

        # Baseline continuous reference: standard normal mixture simulating price/scores
        ref_population = np.random.normal(loc=20.0, scale=8.0, size=10000)
        pop_mean = float(np.mean(ref_population))
        pop_std = float(np.std(ref_population))

        # Baseline categorical reference: 10 discrete categories simulating intent shares
        ref_cat_probs = np.array([0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06])

        # ─── 1. Mean Shift Simulation (Power & FPR) ───────────────────────────
        mean_deltas = [0.0, 0.10, 0.25, 0.50, 1.0]
        psi_threshold = 0.10
        ks_threshold = 0.01

        mean_shift_results: list[dict[str, Any]] = []

        for delta in mean_deltas:
            psi_detections = []
            ks_detections = []
            psi_values = []
            ks_values = []

            for _trial in range(self.n_trials):
                ref_sample = np.random.choice(ref_population, size=self.sample_size, replace=True)
                curr_sample = np.random.choice(ref_population, size=self.sample_size, replace=True) + (delta * pop_std)

                psi = calculate_psi(ref_sample, curr_sample)
                ks_stat, ks_p = calculate_ks(ref_sample, curr_sample)

                psi_values.append(psi)
                ks_values.append(ks_stat)

                psi_detections.append(1.0 if psi >= psi_threshold else 0.0)
                ks_detections.append(1.0 if ks_p < ks_threshold else 0.0)

            psi_det_arr = np.array(psi_detections)
            ks_det_arr = np.array(ks_detections)

            psi_rate, psi_lo, psi_hi = bootstrap_ci(psi_det_arr, n_boot=500, seed=self.seed)
            ks_rate, ks_lo, ks_hi = bootstrap_ci(ks_det_arr, n_boot=500, seed=self.seed)

            mean_shift_results.append({
                "shift_magnitude_std": delta,
                "is_null_resample": (delta == 0.0),
                "psi_detection_rate": round(psi_rate, 4),
                "psi_ci": [round(psi_lo, 4), round(psi_hi, 4)],
                "mean_psi": round(float(np.mean(psi_values)), 4),
                "ks_detection_rate": round(ks_rate, 4),
                "ks_ci": [round(ks_lo, 4), round(ks_hi, 4)],
                "mean_ks_stat": round(float(np.mean(ks_values)), 4),
            })

        # ─── 2. Variance / Scale Shift Simulation ──────────────────────────────
        scale_deltas = [0.0, 0.20, 0.50, 1.0]
        scale_shift_results: list[dict[str, Any]] = []

        for delta in scale_deltas:
            psi_detections = []
            ks_detections = []

            for _trial in range(self.n_trials):
                ref_sample = np.random.choice(ref_population, size=self.sample_size, replace=True)
                # Scale variance around mean
                curr_sample = pop_mean + (np.random.choice(ref_population, size=self.sample_size, replace=True) - pop_mean) * (1.0 + delta)

                psi = calculate_psi(ref_sample, curr_sample)
                _, ks_p = calculate_ks(ref_sample, curr_sample)

                psi_detections.append(1.0 if psi >= psi_threshold else 0.0)
                ks_detections.append(1.0 if ks_p < ks_threshold else 0.0)

            psi_det_arr = np.array(psi_detections)
            ks_det_arr = np.array(ks_detections)
            psi_rate, psi_lo, psi_hi = bootstrap_ci(psi_det_arr, n_boot=500, seed=self.seed)
            ks_rate, ks_lo, ks_hi = bootstrap_ci(ks_det_arr, n_boot=500, seed=self.seed)

            scale_shift_results.append({
                "scale_factor_increase": delta,
                "is_null_resample": (delta == 0.0),
                "psi_detection_rate": round(psi_rate, 4),
                "psi_ci": [round(psi_lo, 4), round(psi_hi, 4)],
                "ks_detection_rate": round(ks_rate, 4),
                "ks_ci": [round(ks_lo, 4), round(ks_hi, 4)],
            })

        # ─── 3. Categorical Distribution Shift Simulation (JSD) ───────────────
        cat_shift_factors = [0.0, 0.10, 0.25, 0.50, 1.0]
        jsd_threshold = 0.15
        cat_shift_results: list[dict[str, Any]] = []

        for delta in cat_shift_factors:
            jsd_detections = []
            jsd_values = []

            for _trial in range(self.n_trials):
                ref_counts = np.random.multinomial(self.sample_size, ref_cat_probs)

                # Perturb probabilities by concentrating mass onto first category
                curr_probs = np.copy(ref_cat_probs)
                curr_probs[0] += delta * 0.40
                curr_probs[1:] = np.maximum(0.01, curr_probs[1:] - (delta * 0.40 / (len(curr_probs) - 1)))
                curr_probs = curr_probs / np.sum(curr_probs)

                curr_counts = np.random.multinomial(self.sample_size, curr_probs)

                jsd = calculate_js_distance(ref_counts, curr_counts)
                jsd_values.append(jsd)
                jsd_detections.append(1.0 if jsd >= jsd_threshold else 0.0)

            jsd_arr = np.array(jsd_detections)
            jsd_rate, jsd_lo, jsd_hi = bootstrap_ci(jsd_arr, n_boot=500, seed=self.seed)

            cat_shift_results.append({
                "category_shift_factor": delta,
                "is_null_resample": (delta == 0.0),
                "jsd_detection_rate": round(jsd_rate, 4),
                "jsd_ci": [round(jsd_lo, 4), round(jsd_hi, 4)],
                "mean_jsd": round(float(np.mean(jsd_values)), 4),
            })

        # Summary Metrics with Clopper-Pearson Exact CIs for Null FPR
        null_mean = mean_shift_results[0]
        null_cat = cat_shift_results[0]

        from assistant.calibration import clopper_pearson_ci
        psi_k = int(round(null_mean["psi_detection_rate"] * self.n_trials))
        ks_k = int(round(null_mean["ks_detection_rate"] * self.n_trials))
        jsd_k = int(round(null_cat["jsd_detection_rate"] * self.n_trials))

        report_payload = {
            "report_key": "monitoring_detector_power",
            "generated_at": datetime.now(UTC).isoformat(),
            "n_trials": self.n_trials,
            "sample_size": self.sample_size,
            "null_hypothesis_fpr": {
                "psi_null_fpr": null_mean["psi_detection_rate"],
                "psi_null_fpr_ci": list(clopper_pearson_ci(psi_k, self.n_trials)),
                "ks_null_fpr": null_mean["ks_detection_rate"],
                "ks_null_fpr_ci": list(clopper_pearson_ci(ks_k, self.n_trials)),
                "jsd_null_fpr": null_cat["jsd_detection_rate"],
                "jsd_null_fpr_ci": list(clopper_pearson_ci(jsd_k, self.n_trials)),
                "fpr_target_satisfied": bool(null_mean["psi_detection_rate"] <= 0.05 and null_mean["ks_detection_rate"] <= 0.05),
            },
            "mean_shift_power": mean_shift_results,
            "variance_shift_power": scale_shift_results,
            "categorical_shift_power": cat_shift_results,
        }

        # Write to reports dir
        report_path = self.reports_dir / "monitoring_detector_power.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        # Database persistence
        if session is not None and not self.dry_run:
            stmt = (
                insert(ServingRecReport)
                .values(
                    report_key="monitoring_detector_power",
                    payload=report_payload,
                    dataset_version="synthetic_validation_v1",
                )
                .on_conflict_do_update(
                    index_elements=[ServingRecReport.report_key],
                    set_={"payload": report_payload},
                )
            )
            await session.execute(stmt)
            await session.commit()
            self.logger.info("Saved monitoring_detector_power report to database.")

        # Print Markdown Table
        md_lines = [
            "# Statistical Drift Detector Power & False-Positive Rate Report",
            f"**Evaluation Setup:** {self.n_trials} trials, N={self.sample_size} per sample | Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## 1. Mean Shift Detection Power (Continuous)",
            "| Shift (Delta * Std) | Type | PSI Power (95% CI) | KS Power (95% CI) | Mean PSI |",
            "|---|:---:|:---:|:---:|:---:|",
        ]
        for row in mean_shift_results:
            typ = "Null (FPR)" if row["is_null_resample"] else "Shift"
            md_lines.append(
                f"| {row['shift_magnitude_std']:.2f} | {typ} | {row['psi_detection_rate']:.4f} [{row['psi_ci'][0]:.4f}, {row['psi_ci'][1]:.4f}] | {row['ks_detection_rate']:.4f} [{row['ks_ci'][0]:.4f}, {row['ks_ci'][1]:.4f}] | {row['mean_psi']:.4f} |"
            )

        md_lines.extend([
            "",
            "## 2. Categorical / Intent Shift Power (JSD)",
            "| Shift Factor | Type | JSD Power (95% CI) | Mean JSD |",
            "|---|:---:|:---:|:---:|",
        ])
        for row in cat_shift_results:
            typ = "Null (FPR)" if row["is_null_resample"] else "Shift"
            md_lines.append(
                f"| {row['category_shift_factor']:.2f} | {typ} | {row['jsd_detection_rate']:.4f} [{row['jsd_ci'][0]:.4f}, {row['jsd_ci'][1]:.4f}] | {row['mean_jsd']:.4f} |"
            )

        md_content = "\n".join(md_lines)
        (self.reports_dir / "monitoring_detector_power.md").write_text(md_content, encoding="utf-8")
        print("\n" + md_content)

        return report_payload

    def execute(self) -> dict[str, Any]:
        if self.dry_run:
            return asyncio.run(self._run_async(session=None))

        async def _run():
            engine = create_async_engine(settings.async_database_url, echo=False)
            sf = async_sessionmaker(engine, expire_on_commit=False)
            async with sf() as session:
                res = await self._run_async(session=session)
            await engine.dispose()
            return res

        return asyncio.run(_run())


def main():
    job = ValidateDriftDetectorsJob()
    job.execute()


if __name__ == "__main__":
    main()
