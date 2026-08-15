"""Create mart_review_intelligence table (Phase 5 Golden Rule Compliance)

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-16 00:50:00.000000

Phase 5 (Golden Rule):
  - Adds mart_review_intelligence to pre-materialize complete structured
    review intelligence bundles (sentiment overview, monthly timeline,
    NLP topic clusters, loved features, complaint breakdown, executive review summary).
  - Repoints api/reviews.py to read strictly from mart_review_intelligence
    with ZERO request-time queries to feature_* or raw_* tables.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql, sqlite

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Compatible JSON type
JSONB = postgresql.JSONB().with_variant(sqlite.JSON(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "mart_review_intelligence",
        sa.Column("app_id", sa.Integer(), sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("sentiment_overview", JSONB, nullable=False),
        sa.Column("timeline", JSONB, nullable=False),
        sa.Column("topics", JSONB, nullable=False),
        sa.Column("loved_features", JSONB, nullable=False),
        sa.Column("complaints", JSONB, nullable=False),
        sa.Column("summary", JSONB, nullable=False),
        sa.Column("is_processed", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("materialized_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("mart_review_intelligence")
