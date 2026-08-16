"""phase 5 serving_recommendations (hybrid rules + model output recommendation engine)

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-15 02:00:00.000000

Phase 5 — Recommendation Engine:
  - Creates serving_recommendations table with model_run_id foreign key (nullable for rules-only).
  - Stores prioritized, explainable recommendations with impact, difficulty, and evidence payloads.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "serving_recommendations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "model_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_runs.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("recommendation_type", sa.String(64), nullable=False, index=True),
        sa.Column("domain", sa.String(64), nullable=False, index=True),
        sa.Column("priority_rank", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("impact_level", sa.String(32), nullable=False),
        sa.Column("difficulty_level", sa.String(32), nullable=False),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence_type", sa.String(64), nullable=False),
        sa.Column("evidence_payload", JSONB, nullable=True),
        sa.Column("action_items", JSONB, nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_serving_recommendations_app_priority",
        "serving_recommendations",
        ["app_id", "priority_rank"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_serving_recommendations_domain",
        "serving_recommendations",
        ["domain"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("serving_recommendations")
