"""
Alembic environment configuration.

Uses synchronous psycopg2 for migrations (Alembic doesn't support asyncpg
natively). The async SQLAlchemy engine used by the app is separate.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ─── Make sure 'backend/' is on sys.path so models can be imported ────────────
# This uses an absolute path derived from __file__ — never os.getcwd()
# (ADR 0001, Decision 8)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Import Base (which imports all models via db/__init__.py)
import db.models  # noqa: F401, E402 — registers all models with Base.metadata
from db.base import Base  # noqa: E402

# ─── Alembic Config ───────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ─── Build sync DATABASE_URL for Alembic ─────────────────────────────────────
def _build_sync_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url:
        # Strip async driver prefix
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB",   "steamiq")
    user = os.getenv("POSTGRES_USER", "steamiq")
    pw   = os.getenv("POSTGRES_PASSWORD", "changeme")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


# ─── Migration runners ────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL without connecting)."""
    url = _build_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to the real DB)."""
    cfg = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = _build_sync_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
