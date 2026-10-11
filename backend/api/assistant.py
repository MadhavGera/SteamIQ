"""
Ask SteamIQ API Router — Phase 6.

Endpoints:
  POST /api/v1/assistant/ask     - Ask natural language question, get structured mode-aware answer
  GET  /api/v1/assistant/health  - Health check & model readiness
  GET  /api/v1/assistant/model   - Active model metadata, tau, and provenance

ADR 0006 Compliance:
  - Bounded in-memory inference executed via AssistantService.
  - Handlers read strictly from serving_*/mart_* tables or pre-loaded in-memory snapshots.
  - Zero raw_* or feature_* table scans. Zero request-time training or DB writes.
  - Query character cap enforced (<= 300).
  - User queries are never persisted to database (privacy invariant).
"""
from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.assistant import (
    AssistantAskRequest,
    AssistantAskResponse,
    AssistantHealthResponse,
    AssistantModelInfoResponse,
)
from assistant.intents import ALL_INTENTS
from assistant.service import AssistantService
from core.errors import ModelNotReadyError, ValidationError
from core.response import ApiResponse
from db.base import get_db

logger = logging.getLogger("steamiq.api.assistant")
router = APIRouter(prefix="/assistant", tags=["assistant"])


def get_assistant_service() -> AssistantService:
    service = AssistantService.get_instance()
    if not service.is_ready():
        service.initialize()
    return service


@router.post(
    "/ask",
    response_model=ApiResponse[AssistantAskResponse],
    summary="Ask SteamIQ natural language question",
)
async def ask_assistant(
    request: AssistantAskRequest,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    service: Annotated[AssistantService, Depends(get_assistant_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[AssistantAskResponse]:
    """
    Submits a natural language query to the learned intent router and extractive retrieval pipeline.
    Returns deterministic mode-aware answers with complete provenance metadata.
    """
    start = time.perf_counter()

    if not request.query or not request.query.strip():
        raise ValidationError("Query cannot be empty.")
    if len(request.query) > 300:
        raise ValidationError("Query length exceeds maximum limit of 300 characters.")

    if not service.is_ready():
        raise ModelNotReadyError(
            "Ask SteamIQ model artifacts have not yet been initialized. "
            "Run 'make assistant-train' to prepare the model."
        )

    from core.cache import ResponseCache
    from mlops.telemetry import TelemetryRegistry

    cache = ResponseCache.get_instance()
    cache_key = cache.make_assistant_key(
        model_run_id=service.model_run_id,
        query=request.query,
        user_mode=request.user_mode,
    )

    cached_res = await cache.get(cache_key)
    if cached_res is not None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        TelemetryRegistry.get_instance().record_assistant_request(
            intent=cached_res.get("intent", "unknown"),
            confidence=float(cached_res.get("confidence", 1.0)),
            took_ms=elapsed_ms,
            cached=True,
        )
        return ApiResponse.ok(data=AssistantAskResponse(**cached_res), took_ms=elapsed_ms)

    res_dict = await service.ask(
        query=request.query,
        user_mode=request.user_mode,
        game_app_id=request.game_app_id,
        liked_app_ids=request.liked_app_ids,
        db_session=db,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    TelemetryRegistry.get_instance().record_assistant_request(
        intent=res_dict.get("intent", "unknown"),
        confidence=float(res_dict.get("confidence", 1.0)),
        took_ms=elapsed_ms,
        cached=False,
    )

    await cache.set(cache_key, res_dict)
    schema = AssistantAskResponse(**res_dict)
    return ApiResponse.ok(data=schema, took_ms=elapsed_ms)


@router.get(
    "/health",
    response_model=ApiResponse[AssistantHealthResponse],
    summary="Assistant service health and readiness check",
)
async def get_assistant_health(
    service: Annotated[AssistantService, Depends(get_assistant_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[AssistantHealthResponse]:
    """
    Returns service readiness and loaded model run ID.
    """
    start = time.perf_counter()
    ready = service.is_ready()
    status_str = "ready" if ready else "not_ready"

    data = AssistantHealthResponse(
        status=status_str,
        model_name=service.model_name,
        model_run_id=service.model_run_id,
        tau=service.tau,
        is_ready=ready,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=data, took_ms=elapsed_ms)


@router.get(
    "/model",
    response_model=ApiResponse[AssistantModelInfoResponse],
    summary="Get active assistant model provenance and metrics",
)
async def get_assistant_model(
    service: Annotated[AssistantService, Depends(get_assistant_service)] = None,  # type: ignore[assignment]
) -> ApiResponse[AssistantModelInfoResponse]:
    """
    Returns detailed metadata on the active intent classification router.
    """
    start = time.perf_counter()
    classes = service.router.classes_ if (service.router and service.router.classes_) else ALL_INTENTS

    data = AssistantModelInfoResponse(
        model_name=service.model_name,
        model_run_id=service.model_run_id,
        stage="candidate",
        tau=service.tau,
        classes=classes,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ApiResponse.ok(data=data, took_ms=elapsed_ms)
