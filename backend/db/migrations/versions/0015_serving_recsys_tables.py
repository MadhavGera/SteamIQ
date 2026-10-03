"""create recommender serving tables (serving_rec_items, serving_rec_item_projection, serving_rec_reports)

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-03 12:00:00.000000

Creates:
  - serving_rec_items: Universe items with metadata, popularity, platform flags, and CDN image URLs.
  - serving_rec_item_projection: 2D coordinates and cluster assignments.
  - serving_rec_reports: Audit reports and evaluation payload storage.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. serving_rec_items
    op.create_table(
        "serving_rec_items",
        sa.Column("app_id", sa.Integer(), primary_key=True),
        sa.Column("item_idx", sa.Integer(), unique=True, nullable=False, index=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("date_release", sa.String(64), nullable=True),
        sa.Column("price_final", sa.Numeric(10, 2), nullable=True),
        sa.Column("positive_ratio", sa.Integer(), nullable=True),
        sa.Column("user_reviews", sa.Integer(), nullable=True),
        sa.Column("tags", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("description_short", sa.Text(), nullable=True),
        sa.Column("has_description", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("train_positives", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("popularity_rank", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("win", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("mac", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("linux", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("steam_deck", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("header_image_url", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # 2. serving_rec_item_projection
    op.create_table(
        "serving_rec_item_projection",
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("serving_rec_items.app_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column("cluster_label", sa.Text(), nullable=True),
    )

    # 3. serving_rec_reports
    op.create_table(
        "serving_rec_reports",
        sa.Column("report_key", sa.Text(), primary_key=True),
        sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column(
            "model_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("model_runs.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("serving_rec_reports")
    op.drop_table("serving_rec_item_projection")
    op.drop_table("serving_rec_items")
