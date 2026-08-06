"""
Shared pytest fixtures.

Uses an in-memory SQLite DB for unit tests — no real Postgres needed.
Integration tests that require Postgres use the STEAMIQ_TEST_DATABASE_URL env var.
"""
from __future__ import annotations

import os
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.base import Base, get_db
from core.app import create_app

# Use SQLite for unit tests — fast and no Postgres needed
TEST_DATABASE_URL = os.getenv(
    "STEAMIQ_TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:",
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session


@pytest.fixture
async def client(db_session: AsyncSession):
    """AsyncClient wired to the FastAPI app with a test DB session."""
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
