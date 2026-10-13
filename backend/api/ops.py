"""
Operations & Telemetry API Router — Phase 7.

Endpoints:
  GET /api/v1/ops/metrics - Prometheus standard text exposition format
  GET /api/v1/ops/drift   - Statistical drift status and distribution telemetry
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
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
    start = time.perf_counter()
    from jobs.generate_reports import GenerateReportsJob
    job = GenerateReportsJob()
    result = await job._run_async(session=db)
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
