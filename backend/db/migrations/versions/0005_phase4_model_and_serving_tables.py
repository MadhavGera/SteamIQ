"""phase 4 model runs and serving predictions tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-14 00:00:00.000000

Phase 4 — Model Artifacts & Serving Predictions:
  - model_runs (UUID PK, stage enum: candidate -> staging -> production -> archived)
  - serving_predictions (FK model_run_id required on every row for 100% auditability)
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. model_runs ─────────────────────────────────────────────────────────
    op.create_table(
        "model_runs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("model_name", sa.String(128), nullable=False, index=True),
        sa.Column("stage", sa.String(32), nullable=False, index=True),  # candidate | staging | production | archived
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("hyperparameters", JSONB, nullable=True),
        sa.Column("metrics", JSONB, nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("promoted_by", sa.String(128), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_model_runs_model_name_stage",
        "model_runs",
        ["model_name", "stage"],
    )

    # ── 2. serving_predictions ────────────────────────────────────────────────
    op.create_table(
        "serving_predictions",
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
            sa.ForeignKey("model_runs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("prediction_type", sa.String(64), nullable=False, index=True),  # e.g. "success_score"
        sa.Column("score", sa.Numeric(6, 4), nullable=False),
        sa.Column("confidence_lower", sa.Numeric(6, 4), nullable=True),
        sa.Column("confidence_upper", sa.Numeric(6, 4), nullable=True),
        sa.Column("feature_importance", JSONB, nullable=True),
        sa.Column("shap_values", JSONB, nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "prediction_type", "model_run_id", name="uq_serving_predictions_app_type_run"),
    )
    op.create_index(
        "ix_serving_predictions_app_type",
        "serving_predictions",
        ["app_id", "prediction_type"],
    )


def downgrade() -> None:
    op.drop_table("serving_predictions")
    op.drop_table("model_runs")
