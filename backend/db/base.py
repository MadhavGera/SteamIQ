"""
SQLAlchemy engine, session factory, and declarative Base.

All models import Base from here. The engine is configured async
(asyncpg driver) so FastAPI route handlers can use async sessions.
"""
from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# Build DATABASE_URL from individual env vars so Docker Compose and local
# dev both work without juggling two separate connection-string formats.
# Using absolute env-var resolution — never os.getcwd() or relative paths
# (ADR 0001, Decision 8).

def _build_database_url() -> str:
    # Allow a full URL override (useful in tests and CI)
    url = os.getenv("DATABASE_URL")
    if url:
        # Ensure the async driver is used even if someone passes a sync URL
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB",   "steamiq")
    user = os.getenv("POSTGRES_USER", "steamiq")
    pw   = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql+asyncpg://{user}:{pw}@{host}:{port}/{db}"


DATABASE_URL = _build_database_url()

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("APP_ENV", "development") == "development",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Dependency (FastAPI)
# ---------------------------------------------------------------------------

async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
