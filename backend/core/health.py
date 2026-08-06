"""
/health endpoint — returns DB connectivity status and app version.

Used by Docker Compose healthcheck and the `make health` Makefile target.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.response import ApiResponse, ResponseMeta
from db.base import get_db

router = APIRouter(tags=["health"])


@router.get("/health", response_model=ApiResponse[dict], summary="Health check")
async def health(db: AsyncSession = Depends(get_db)) -> ApiResponse[dict]:
    """
    Returns:
      - status: "ok" | "degraded"
      - db: "connected" | "error"
      - version: app version string
    """
    start = time.perf_counter()

    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    took_ms = (time.perf_counter() - start) * 1000
    overall = "ok" if db_status == "connected" else "degraded"

    from core.config import settings

    return ApiResponse(
        success=overall == "ok",
        data={
            "status": overall,
            "db": db_status,
            "version": settings.app_version,
            "env": settings.app_env,
        },
        meta=ResponseMeta(version=settings.app_version, took_ms=round(took_ms, 2)),
    )
