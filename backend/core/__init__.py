"""
SteamIQ shared core library.

Exports:
  - create_app: FastAPI application factory
  - ApiResponse, ErrorDetail, ResponseMeta: response envelope types
  - SteamIQException and all domain exceptions
  - settings: application configuration singleton
"""
from core.app import create_app
from core.config import settings
from core.errors import (
    GameNotFoundError,
    NotFoundError,
    NotYetComputedError,
    RateLimitError,
    SteamIQException,
    UpstreamError,
    ValidationError,
)
from core.response import ApiResponse, ErrorDetail, ResponseMeta, TimingContext

__all__ = [
    "create_app",
    "settings",
    "ApiResponse",
    "ErrorDetail",
    "ResponseMeta",
    "TimingContext",
    "SteamIQException",
    "NotFoundError",
    "GameNotFoundError",
    "ValidationError",
    "UpstreamError",
    "NotYetComputedError",
    "RateLimitError",
]
