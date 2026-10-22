"""
Operations & Telemetry API Router — Phase 7.

Endpoints:
  GET /api/v1/ops/metrics - Prometheus standard text exposition format
  GET /api/v1/ops/drift   - Statistical drift status and distribution telemetry
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.response import ApiResponse
from db.base import get_db
from mlops.telemetry import TelemetryRegistry

logger = logging.getLogger("steamiq.api.ops")
router = APIRouter(prefix="/ops", tags=["ops"])


@router.post(
    "/materialize-reports",
    summary="Materialize & Synchronize RecSys Evidence Reports",
    description="Loads precomputed reports from disk and synchronizes them into PostgreSQL serving_rec_reports.",
)
async def materialize_reports(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ApiResponse[dict[str, Any]]:
    if not getattr(settings, "enable_ops_write", True):
        raise HTTPException(
            status_code=403,
            detail="Operational writes are disabled by configuration (ENABLE_OPS_WRITE=False).",
        )
    start = time.perf_counter()
    from jobs.generate_reports import GenerateReportsJob
    job = GenerateReportsJob(allow_experiments=False)
    result = await job._run_async(session=db, allow_experiments=False)
    took_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=result, took_ms=round(took_ms, 2))


@router.get(
    "/metrics",
    summary="Get Prometheus metrics exposition",
    response_class=Response,
)
async def get_prometheus_metrics() -> Response:
    """
    Renders live telemetry counters, latency histograms, and cache hits in Prometheus format.
    """
    registry = TelemetryRegistry.get_instance()
    text_output = registry.render_prometheus_text()
    return Response(content=text_output, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get(
    "/drift",
    summary="Get model drift monitoring report",
)
async def get_drift_report() -> ApiResponse[dict[str, Any]]:
    """
    Returns latest drift monitoring evaluation report and live telemetry summary.
    """
    # 1. Attempt to read persisted monitoring report
    reports_dir = Path(settings.resolved_recsys_data_dir) / "reports"
    report_file = reports_dir / "monitoring.json"

    persisted_report = None
    if report_file.exists():
        try:
            persisted_report = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Could not read monitoring.json: %s", e)

    # 2. Extract current live telemetry distribution stats
    telemetry = TelemetryRegistry.get_instance()
    live_samples = telemetry.get_live_drift_samples()

    result = {
        "report": persisted_report or {"status": "no live data", "details": "No scheduled monitoring report generated yet"},
        "telemetry_samples": {
            "recsys_total_requests": live_samples["total_recsys_requests"],
            "assistant_total_requests": live_samples["total_assistant_requests"],
            "buffered_library_samples": len(live_samples["library_lengths"]),
            "buffered_score_samples": len(live_samples["recommendation_scores"]),
            "buffered_confidence_samples": len(live_samples["assistant_confidences"]),
        },
    }

    return ApiResponse.ok(data=result)


ALLOWED_EVIDENCE_NAMES = frozenset({
    "retrain_gate",
    "drift_detector_power",
    "monitor_models",
    "cold_latency_benchmark",
    "generate_model_cards",
})

EVIDENCE_JSON_ALIASES: dict[str, list[str]] = {
    "retrain_gate": ["retrain_gate.json", "retrain_gate_decision.json"],
    "drift_detector_power": ["drift_detector_power.json", "monitoring_detector_power.json"],
    "monitor_models": ["monitor_models.json", "monitoring.json"],
    "cold_latency_benchmark": ["cold_latency_benchmark.json", "latency_benchmark.json"],
    "generate_model_cards": ["generate_model_cards.json"],
}


@router.get(
    "/evidence/{name}",
    summary="Get MLOps verification evidence log and report data",
)
async def get_ops_evidence(name: str) -> ApiResponse[dict[str, Any]]:
    """
    Returns reproducible evidence log text and parsed JSON payload for a whitelisted evidence benchmark.
    """
    if name not in ALLOWED_EVIDENCE_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Evidence artifact '{name}' not found. Whitelisted: {sorted(ALLOWED_EVIDENCE_NAMES)}",
        )

    # 1. Search for evidence log
    evidence_dirs = [
        Path("docs/evidence"),
        Path("../docs/evidence"),
        Path(__file__).resolve().parent.parent.parent / "docs" / "evidence",
    ]
    log_text = ""
    for edir in evidence_dirs:
        log_candidate = edir / f"{name}.log"
        if log_candidate.exists():
            try:
                log_text = log_candidate.read_text(encoding="utf-8")
                break
            except Exception as exc:
                logger.warning("Could not read evidence log %s: %s", log_candidate, exc)

    # 2. Search for structured JSON data
    report_dirs = [
        Path(settings.resolved_recsys_data_dir) / "reports",
        Path("docs/results"),
        Path("../docs/results"),
        Path(__file__).resolve().parent.parent.parent / "docs" / "results",
    ]
    json_data = None
    target_filenames = EVIDENCE_JSON_ALIASES.get(name, [f"{name}.json"])
    for rdir in report_dirs:
        if json_data is not None:
            break
        for fname in target_filenames:
            candidate = rdir / fname
            if candidate.exists():
                try:
                    json_data = json.loads(candidate.read_text(encoding="utf-8"))
                    break
                except Exception as exc:
                    logger.warning("Could not read evidence JSON %s: %s", candidate, exc)

    return ApiResponse.ok(
        data={
            "name": name,
            "log": log_text,
            "data": json_data,
        }
    )


@router.get(
    "/model-cards",
    summary="Get list and contents of all registered model cards",
)
async def get_ops_model_cards() -> ApiResponse[list[dict[str, Any]]]:
    """
    Returns markdown contents and metadata for all model cards in docs/model-cards.
    """
    card_dirs = [
        Path("docs/model-cards"),
        Path("../docs/model-cards"),
        Path(__file__).resolve().parent.parent.parent / "docs" / "model-cards",
    ]

    cards: list[dict[str, Any]] = []
    found_dir: Path | None = None
    for cdir in card_dirs:
        if cdir.exists() and cdir.is_dir():
            found_dir = cdir
            break

    if found_dir:
        for card_path in sorted(found_dir.glob("*.md")):
            try:
                content = card_path.read_text(encoding="utf-8")
                # Extract first heading as title
                title = card_path.stem
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("# "):
                        title = stripped[2:].strip().strip("`")
                        break
                cards.append({
                    "filename": card_path.name,
                    "model_name": card_path.stem,
                    "title": title,
                    "content": content,
                })
            except Exception as exc:
                logger.warning("Could not read model card %s: %s", card_path, exc)

    return ApiResponse.ok(data=cards)
