"""create feature zone nlp tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-07 20:00:00.000000

Phase 2 — NLP Core: feature_review_* tables.
Zone convention (ADR 0001, Decision 1):
  feature_* → NLP and feature pipeline jobs write; ML and API handlers read
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. feature_review_sentiment ───────────────────────────────────────────
    op.create_table(
        "feature_review_sentiment",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("month", sa.String(16), nullable=False),
        sa.Column("positive_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("negative_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("net_positive_pct", sa.Numeric(5, 2), server_default="0.0", nullable=False),
        sa.Column("sentiment_score", sa.Numeric(5, 4), server_default="0.0", nullable=False),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "month", name="uq_game_sentiment_month"),
    )
    op.create_index("ix_feature_review_sentiment_app_id", "feature_review_sentiment", ["app_id"])
    op.create_index("ix_feature_review_sentiment_month", "feature_review_sentiment", ["month"])

    # ── 2. feature_review_topics ──────────────────────────────────────────────
    op.create_table(
        "feature_review_topics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("topic_label", sa.String(256), nullable=False),
        sa.Column("review_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("sentiment_score", sa.Numeric(5, 4), server_default="0.5", nullable=False),
        sa.Column("keywords", JSONB, nullable=True),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "topic_id", name="uq_game_topic"),
    )
    op.create_index("ix_feature_review_topics_app_id", "feature_review_topics", ["app_id"])

    # ── 3. feature_review_complaints ──────────────────────────────────────────
    op.create_table(
        "feature_review_complaints",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(128), nullable=False),
        sa.Column("volume_pct", sa.Numeric(5, 2), server_default="0.0", nullable=False),
        sa.Column("severity", sa.String(32), server_default="moderate", nullable=False),
        sa.Column("representative_snippets", JSONB, nullable=True),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "category", name="uq_game_complaint"),
    )
    op.create_index("ix_feature_review_complaints_app_id", "feature_review_complaints", ["app_id"])

    # ── 4. feature_review_features ────────────────────────────────────────────
    op.create_table(
        "feature_review_features",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature_name", sa.String(128), nullable=False),
        sa.Column("mention_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("praise_intensity", sa.Integer(), server_default="80", nullable=False),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "feature_name", name="uq_game_loved_feature"),
    )
    op.create_index("ix_feature_review_features_app_id", "feature_review_features", ["app_id"])

    # ── 5. feature_review_summary ─────────────────────────────────────────────
    op.create_table(
        "feature_review_summary",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("core_strengths", JSONB, nullable=True),
        sa.Column("pain_points", JSONB, nullable=True),
        sa.Column("feature_requests", JSONB, nullable=True),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_feature_review_summary_app_id", "feature_review_summary", ["app_id"])


def downgrade() -> None:
    op.drop_table("feature_review_summary")
    op.drop_table("feature_review_features")
    op.drop_table("feature_review_complaints")
    op.drop_table("feature_review_topics")
    op.drop_table("feature_review_sentiment")
