"""
Exception hierarchy and global FastAPI exception handlers.

ADR 0001, Decision 5: All error responses use the ApiResponse envelope.
Every exception must flow through these handlers — no raw HTTPException
with a plain dict body anywhere in the codebase.

Exception classes carry:
  - status_code: HTTP status to return
  - code: machine-readable slug for the ApiResponse.error.code field
  - message: human-readable description
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.response import ApiResponse

logger = logging.getLogger(__name__)


# ─── Exception Hierarchy ──────────────────────────────────────────────────────

class SteamIQException(Exception):
    """Base exception for all SteamIQ domain errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.details = details
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class NotFoundError(SteamIQException):
    """Resource not found."""
    status_code = 404
    code = "not_found"


class GameNotFoundError(NotFoundError):
    """Specific: game not found by app_id or name."""
    code = "game_not_found"


class ValidationError(SteamIQException):
    """Client sent invalid input."""
    status_code = 422
    code = "validation_error"


class UpstreamError(SteamIQException):
    """Upstream API (Steam, SteamSpy) returned an error or is unavailable."""
    status_code = 502
    code = "upstream_error"


class NotYetComputedError(SteamIQException):
    """
    Data exists in raw_* / feature_* but hasn't been materialized to
    serving_* / mart_* yet. The client should retry after the next job run.
    """
    status_code = 202
    code = "not_yet_computed"


class RateLimitError(SteamIQException):
    """Rate limit hit (Steam API, SteamSpy, or internal)."""
    status_code = 429
    code = "rate_limited"


# ─── Handler Helpers ──────────────────────────────────────────────────────────

def _error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ApiResponse.fail(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump())


# ─── Global Handlers ──────────────────────────────────────────────────────────

async def steamiq_exception_handler(
    request: Request, exc: SteamIQException
) -> JSONResponse:
    """Handler for all SteamIQ domain exceptions."""
    if exc.status_code >= 500:
        logger.exception("Internal error: %s", exc.message, exc_info=exc)
    else:
        logger.warning("Client error %s [%s]: %s", exc.status_code, exc.code, exc.message)

    return _error_response(
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Converts Starlette HTTPException to ApiResponse envelope."""
    code_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "unprocessable_entity",
        429: "rate_limited",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }
    code = code_map.get(exc.status_code, "http_error")
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error_response(code=code, message=detail, status_code=exc.status_code)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Converts Pydantic RequestValidationError to ApiResponse envelope."""
    field_errors = {}
    for error in exc.errors():
        loc = " → ".join(str(l) for l in error["loc"] if l != "body")
        field_errors[loc] = error["msg"]

    return _error_response(
        code="validation_error",
        message="Request validation failed",
        status_code=422,
        details={"fields": field_errors},
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for any unhandled exception — never leak tracebacks."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url, exc_info=exc)
    return _error_response(
        code="internal_error",
        message="An unexpected error occurred. Please try again later.",
        status_code=500,
    )


# ─── Registration helper ──────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app. Called by create_app()."""
    app.add_exception_handler(SteamIQException, steamiq_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
