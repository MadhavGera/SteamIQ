"""
create_app() — FastAPI application factory.

ADR 0001, Decision 4: CORS configured with explicit origin allowlist from env.
ADR 0001, Decision 5: All responses use ApiResponse envelope via global handlers.

Usage:
    from core.app import create_app
    app = create_app()
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.errors import register_exception_handlers

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Registers:
    - CORS middleware with explicit origin allowlist (never *)
    - Global exception handlers (ApiResponse envelope on all errors)
    - /health router
    - /api/v1 routers
    """
    app = FastAPI(
        title="SteamIQ API",
        description=(
            "Game intelligence platform — NLP insights, competitor analysis, "
            "ML-powered success prediction, and AI assistant. "
            "All responses use the ApiResponse envelope."
        ),
        version=settings.app_version,
        docs_url="/docs" if settings.app_env == "development" else None,
        redoc_url="/redoc" if settings.app_env == "development" else None,
    )

    # ── CORS (ADR 0001, Decision 4 — explicit allowlist, never *) ─────────────
    origins = settings.cors_origins_list
    logger.info("CORS allowed origins: %s", origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Global exception handlers ──────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Health router ──────────────────────────────────────────────────────────
    from core.health import router as health_router
    app.include_router(health_router)

    # ── API routers ────────────────────────────────────────────────────────────
    from api.games import router as games_router
    app.include_router(games_router, prefix="/api/v1")

    logger.info("SteamIQ API v%s started in %s mode", settings.app_version, settings.app_env)
    return app
