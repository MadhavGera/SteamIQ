"""phase 5 mart_update_impact table (Update Impact Tracker before/after window comparisons)

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-15 03:00:00.000000

Phase 5 — Update Impact Tracker:
  - Creates mart_update_impact table storing pre/post window metrics around patch dates.
  - Compares sentiment, CCU player activity, and complaint topic rates.
  - Results strictly labeled "observed/correlated" (never "caused").
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mart_update_impact",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("patch_name", sa.String(256), nullable=False),
        sa.Column("patch_version", sa.String(64), nullable=True),
        sa.Column("patch_date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("is_inferred", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("window_days", sa.Integer(), server_default=sa.text("14"), nullable=False),
        # Observed Sentiment Delta
        sa.Column("pre_sentiment_positive_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("post_sentiment_positive_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("sentiment_delta_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("observed_sentiment_verdict", sa.String(64), nullable=False),
        # Observed Player Activity Delta
        sa.Column("pre_avg_ccu", sa.Integer(), nullable=True),
        sa.Column("post_avg_ccu", sa.Integer(), nullable=True),
        sa.Column("ccu_change_pct", sa.Numeric(6, 2), nullable=True),
        # Observed Complaint Topic Shifts
        sa.Column("pre_complaint_distribution", JSONB, nullable=True),
        sa.Column("post_complaint_distribution", JSONB, nullable=True),
        sa.Column("top_resolved_complaints", JSONB, nullable=True),
        sa.Column("top_emerging_complaints", JSONB, nullable=True),
        # Non-Causal Correlation Summary
        sa.Column("correlation_summary", sa.Text(), nullable=False),
        sa.Column(
            "materialized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_mart_update_impact_app_date",
        "mart_update_impact",
        ["app_id", "patch_date"],
    )


def downgrade() -> None:
    op.drop_table("mart_update_impact")
