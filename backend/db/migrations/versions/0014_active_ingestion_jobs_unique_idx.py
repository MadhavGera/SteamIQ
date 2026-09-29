"""Update ingestion_jobs unique index to cover active (pending, running) jobs

Revision ID: 0014
Revises: 6edf05ec1a36
Create Date: 2026-09-23 01:05:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | None = "6edf05ec1a36"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop old index on running status only
    op.drop_index(
        "ix_ingestion_jobs_app_id_running",
        table_name="ingestion_jobs",
        postgresql_where=sa.text("status = 'running'"),
    )
    # Create new index covering all active statuses (pending, running)
    op.create_index(
        "ix_ingestion_jobs_app_id_active",
        "ingestion_jobs",
        ["app_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ingestion_jobs_app_id_active",
        table_name="ingestion_jobs",
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )
    op.create_index(
        "ix_ingestion_jobs_app_id_running",
        "ingestion_jobs",
        ["app_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )
