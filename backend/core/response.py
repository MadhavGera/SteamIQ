"""
ApiResponse envelope — wraps every API response in a consistent shape.

ADR 0001, Decision 5:
  Every response (success or error) uses this envelope. The 'code' field
  in ErrorDetail is a machine-readable slug so frontend can branch on
  specific error types without parsing message strings.

Shape:
  {
    "success": true,
    "data": { ... },
    "error": null,
    "meta": { "version": "0.1.0", "took_ms": 12 }
  }
"""
from __future__ import annotations

import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Machine-readable error info returned in ApiResponse.error."""
    code: str = Field(description="Machine-readable error slug, e.g. 'game_not_found'")
    message: str = Field(description="Human-readable error description")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra context (field errors, upstream info, etc.)",
    )


class ResponseMeta(BaseModel):
    """Optional metadata attached to every response."""
    version: str = Field(default="0.1.0")
    took_ms: float | None = Field(default=None, description="Request processing time in ms")
    page: int | None = None
    page_size: int | None = None
    total: int | None = None


class ApiResponse(BaseModel, Generic[T]):
    """
    Unified response envelope for all SteamIQ API endpoints.

    Usage (success):
        return ApiResponse.ok(data=my_object)

    Usage (error — prefer raising SteamIQException instead):
        return ApiResponse.fail(code="game_not_found", message="No game found")
    """
    success: bool
    data: T | None = None
    error: ErrorDetail | None = None
    meta: ResponseMeta | None = None

    @classmethod
    def ok(
        cls,
        data: T,
        *,
        took_ms: float | None = None,
        page: int | None = None,
        page_size: int | None = None,
        total: int | None = None,
        version: str = "0.1.0",
    ) -> ApiResponse[T]:
        """Build a success response."""
        meta = ResponseMeta(
            version=version,
            took_ms=took_ms,
            page=page,
            page_size=page_size,
            total=total,
        )
        return cls(success=True, data=data, meta=meta)

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ApiResponse[None]:
        """Build an error response (prefer raising SteamIQException in handlers)."""
        return cls(
            success=False,
            error=ErrorDetail(code=code, message=message, details=details),
        )


class TimingContext:
    """Context manager that measures elapsed ms for use in ApiResponse.meta."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimingContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
