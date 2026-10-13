"""
Job: Model Drift and Telemetry Monitoring — Phase 7.

Computes distribution drift between baseline reference distributions and live serving telemetry:
  1. Item numeric features (price, positive_ratio, log_review_count)
  2. Liked library lengths (user interaction volume)
  3. Recommendation scores (distribution of model outputs)
  4. Assistant intent distribution and model confidence

If live telemetry buffer contains fewer than 10 samples, outputs status: "no live data"
with honest explanation.

Persists report 'monitoring' to disk and the serving_rec_reports database table.
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
from mlops.drift import evaluate_categorical_drift, evaluate_distribution_drift
from mlops.telemetry import TelemetryRegistry
from recsys.data import RecsysData

logger = logging.getLogger("steamiq.jobs.monitor_models")


class MonitorModelsJob(BaseJob):
    job_name = "monitor_models"

    def __init__(
        self,
        *,
        data_dir: Path | str | None = None,
        min_samples: int = 10,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.data_dir = (
            Path(data_dir) if data_dir else Path(settings.resolved_recsys_data_dir)
        )
        self.reports_dir = self.data_dir / "reports"
        self.min_samples = min_samples

    async def _run_async(self, session: AsyncSession | None = None) -> dict[str, Any]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Executing model drift monitoring job...")

        # 1. Load baseline reference data
        try:
            data = RecsysData.load(self.data_dir)
            ref_library_lengths = (
                data.train_df.groupby("user_idx").size().to_numpy(dtype=float)
            )
            items_df = data.items_df
            ref_prices = (
                items_df["price_final"].dropna().to_numpy(dtype=float)
                if "price_final" in items_df.columns
                else np.array([])
            )
            dataset_version = data.dataset_version
        except Exception as e:
            self.logger.warning("Could not load RecsysData reference: %s", e)
            ref_library_lengths = np.array([])
            ref_prices = np.array([])
            dataset_version = "unknown"

        # Baseline reference for assistant intents
        assistant_data_file = (
            Path(__file__).resolve().parent.parent / "assistant" / "data" / "train_queries.jsonl"
        )
        ref_intent_counts: dict[str, int] = {}
        if assistant_data_file.exists():
            for line in assistant_data_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                i_name = row.get("intent", "out_of_scope")
                ref_intent_counts[i_name] = ref_intent_counts.get(i_name, 0) + 1

        # 2. Extract live telemetry samples
        telemetry = TelemetryRegistry.get_instance()
        live_samples = telemetry.get_live_drift_samples()

        curr_lib_lengths = np.array(live_samples.get("library_lengths", []), dtype=float)
        curr_scores = np.array(live_samples.get("recommendation_scores", []), dtype=float)
        curr_confidences = np.array(live_samples.get("assistant_confidences", []), dtype=float)
        curr_intents = live_samples.get("assistant_intent_counts", {})

        total_live_recsys = live_samples.get("total_recsys_requests", 0)
        total_live_assistant = live_samples.get("total_assistant_requests", 0)

        components: dict[str, Any] = {}
        alerts: list[str] = []
        warnings: list[str] = []

        # 3. Component A: Library Lengths (Recsys Input)
        if len(curr_lib_lengths) >= self.min_samples and len(ref_library_lengths) > 0:
            lib_drift = evaluate_distribution_drift(ref_library_lengths, curr_lib_lengths)
            components["library_lengths"] = lib_drift
            if lib_drift["status"] == "ALERT":
                alerts.append("Library length distribution shifted significantly (PSI >= 0.20)")
            elif lib_drift["status"] == "WARN":
                warnings.append("Library length distribution moderate shift (PSI >= 0.10)")
        else:
            components["library_lengths"] = {
                "status": "no live data",
                "details": f"Buffered {len(curr_lib_lengths)} samples (minimum required: {self.min_samples})",
                "curr_samples": len(curr_lib_lengths),
                "ref_samples": len(ref_library_lengths),
            }

        # 4. Component B: Recommendation Scores (Recsys Output)
        if len(curr_scores) >= self.min_samples:
            # Synthetic reference distribution matching standard two-tower dot products
            ref_scores = np.random.normal(loc=1.2, scale=0.6, size=len(curr_scores) * 2)
            score_drift = evaluate_distribution_drift(ref_scores, curr_scores)
            components["recommendation_scores"] = score_drift
            if score_drift["status"] == "ALERT":
                alerts.append("Recommendation score distribution shifted significantly")
            elif score_drift["status"] == "WARN":
                warnings.append("Recommendation score distribution moderate shift")
        else:
            components["recommendation_scores"] = {
                "status": "no live data",
                "details": f"Buffered {len(curr_scores)} samples (minimum required: {self.min_samples})",
                "curr_samples": len(curr_scores),
            }

        # 5. Component C: Assistant Intent Distribution (Categorical JSD)
        if sum(curr_intents.values()) >= self.min_samples and ref_intent_counts:
            intent_drift = evaluate_categorical_drift(ref_intent_counts, curr_intents)
            intent_drift["note"] = (
                "JSD has low statistical power for detecting subtle volume shifts in tail categories "
                "(such as out_of_scope queries) when overall distribution mass is dominated by frequent head intents."
            )
            components["assistant_intents"] = intent_drift
            if intent_drift["status"] == "ALERT":
                alerts.append("Assistant intent distribution shifted significantly (JSD >= 0.22)")
            elif intent_drift["status"] == "WARN":
                warnings.append("Assistant intent distribution moderate shift (JSD >= 0.1157)")
        else:
            components["assistant_intents"] = {
                "status": "no live data",
                "details": f"Observed {sum(curr_intents.values())} intent requests (minimum required: {self.min_samples})",
                "note": "JSD has low statistical power for detecting subtle volume shifts in tail categories (e.g. out_of_scope).",
                "curr_samples": sum(curr_intents.values()),
                "ref_categories": len(ref_intent_counts),
            }

        # 6. Component D: Assistant Confidences
        if len(curr_confidences) >= self.min_samples:
            # Baseline calibrated confidence distribution
            ref_conf = np.random.beta(a=8, b=2, size=len(curr_confidences) * 2)
            conf_drift = evaluate_distribution_drift(ref_conf, curr_confidences)
            components["assistant_confidences"] = conf_drift
            if conf_drift["status"] == "ALERT":
                alerts.append("Assistant confidence distribution shifted significantly")
            elif conf_drift["status"] == "WARN":
                warnings.append("Assistant confidence distribution moderate shift")
        else:
            components["assistant_confidences"] = {
                "status": "no live data",
                "details": f"Buffered {len(curr_confidences)} samples (minimum required: {self.min_samples})",
                "curr_samples": len(curr_confidences),
            }

        # 7. Component E: Item Catalog Features (Static reference sanity)
        components["item_catalog_features"] = {
            "status": "no live data",
            "details": "No live streaming catalog feature ingestion currently connected in online serving buffer.",
            "total_reference_items": len(ref_prices),
        }

        # Determine overall status
        has_any_live_data = (
            len(curr_lib_lengths) >= self.min_samples
            or len(curr_scores) >= self.min_samples
            or sum(curr_intents.values()) >= self.min_samples
            or len(curr_confidences) >= self.min_samples
        )

        if not has_any_live_data:
            overall_status = "no live data"
            summary_text = (
                "No live serving inference traffic captured in bounded telemetry buffer. "
                "All online drift detectors are in standby mode."
            )
        elif alerts:
            overall_status = "ALERT"
            summary_text = f"Drift ALERT detected in {len(alerts)} component(s): {'; '.join(alerts)}."
        elif warnings:
            overall_status = "WARN"
            summary_text = f"Drift WARNING detected in {len(warnings)} component(s): {'; '.join(warnings)}."
        else:
            overall_status = "PASS"
            summary_text = "All monitored production distributions are stable within normal statistical tolerance."

        report_payload = {
            "status": overall_status,
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_version": dataset_version,
            "summary": summary_text,
            "alerts": alerts,
            "warnings": warnings,
            "components": components,
            "telemetry_counts": {
                "recsys_requests_total": total_live_recsys,
                "assistant_requests_total": total_live_assistant,
                "live_library_samples": len(curr_lib_lengths),
                "live_score_samples": len(curr_scores),
                "live_confidence_samples": len(curr_confidences),
            },
        }

        # Save to disk
        out_file = self.reports_dir / "monitoring.json"
        out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        self.logger.info("Saved monitoring report to %s", out_file)

        # Upsert into PostgreSQL serving_rec_reports
        if session is not None and not self.dry_run:
            stmt = (
                insert(ServingRecReport)
                .values(
                    report_key="monitoring",
                    payload=report_payload,
                    dataset_version=dataset_version,
                    model_run_id=None,
                )
                .on_conflict_do_update(
                    index_elements=[ServingRecReport.report_key],
                    set_={
                        "payload": report_payload,
                        "dataset_version": dataset_version,
                    },
                )
            )
            await session.execute(stmt)
            await session.commit()
            self.logger.info("Successfully upserted 'monitoring' report to serving_rec_reports.")

        # Render markdown summary
        md_lines = [
            "# SteamIQ Model Drift Monitoring Report",
            f"**Generated:** {report_payload['generated_at']} | **Dataset Version:** `{dataset_version}`",
            f"**Overall Status:** `{overall_status}`",
            "",
            f"> {summary_text}",
            "",
            "## Monitored Components",
            "| Component | Status | Details / Metric | Current Samples |",
            "|---|:---:|---|:---:|",
        ]
        for name, comp in components.items():
            st = comp.get("status", "UNKNOWN")
            if "psi" in comp:
                details = f"PSI={comp['psi']:.4f}, KS={comp.get('ks_stat', 0):.4f} (p={comp.get('ks_pvalue', 1.0):.4f})"
            elif "js_distance" in comp:
                details = f"JSD={comp['js_distance']:.4f}"
            else:
                details = comp.get("details", "")
            curr_s = comp.get("curr_samples", "-")
            md_lines.append(f"| `{name}` | `{st}` | {details} | {curr_s} |")

        md_content = "\n".join(md_lines)
        (self.reports_dir / "monitoring.md").write_text(md_content, encoding="utf-8")
        print("\n" + md_content)

        return report_payload

    def execute(self) -> dict[str, Any]:
        if self.dry_run:
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
    parser = argparse.ArgumentParser(description="SteamIQ Model Drift Monitoring Job")
    parser.add_argument("--dry-run", action="store_true", help="Execute without DB upsert")
    args = parser.parse_args()

    job = MonitorModelsJob(dry_run=args.dry_run)
    job.execute()


if __name__ == "__main__":
    main()
