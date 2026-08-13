"""phase 4 feature tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-14 00:00:00.000000

Phase 4 — Predictive ML Feature Engineering Tables:
  - feature_game_features (price, genre, tags, dev/pub history, velocity, competitor density, sentiment)
  - feature_market_features (genre market statistics, saturation index)
  - Enforces strict feature_cutoff_date to eliminate temporal data leakage.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. feature_game_features ──────────────────────────────────────────────
    op.create_table(
        "feature_game_features",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("feature_cutoff_date", sa.DateTime(timezone=True), nullable=False, index=True),
        
        # Pricing features
        sa.Column("price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("discount_pct", sa.Integer(), server_default="0", nullable=False),
        sa.Column("is_free", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        
        # Genre & Tag features
        sa.Column("primary_genre", sa.String(128), nullable=True),
        sa.Column("genres", JSONB, nullable=True),
        sa.Column("top_tags", JSONB, nullable=True),
        
        # Developer & Publisher historical track record
        sa.Column("developer_game_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("publisher_game_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("developer_avg_review_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("publisher_avg_review_score", sa.Numeric(5, 2), nullable=True),
        
        # Review signals at cutoff
        sa.Column("total_reviews_at_cutoff", sa.Integer(), server_default="0", nullable=False),
        sa.Column("positive_reviews_at_cutoff", sa.Integer(), server_default="0", nullable=False),
        sa.Column("review_velocity_30d", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("positive_review_pct", sa.Numeric(5, 2), server_default="0.0", nullable=False),
        
        # Competitor density & market positioning
        sa.Column("competitor_density", sa.Integer(), server_default="0", nullable=False),
        sa.Column("price_vs_genre_median", sa.Numeric(10, 2), nullable=True),
        
        # Sentiment & NLP signals (from feature_review_* tables)
        sa.Column("sentiment_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("complaint_density", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("loved_feature_density", sa.Float(), server_default="0.0", nullable=False),
        
        # Target / engagement labels for training
        sa.Column("average_playtime_forever", sa.Integer(), server_default="0", nullable=False),
        sa.Column("target_success_score", sa.Float(), nullable=True),
        sa.Column("is_hit", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "feature_cutoff_date", name="uq_game_features_app_cutoff"),
    )
    op.create_index(
        "ix_feature_game_features_app_id_cutoff",
        "feature_game_features",
        ["app_id", "feature_cutoff_date"],
    )

    # ── 2. feature_market_features ────────────────────────────────────────────
    op.create_table(
        "feature_market_features",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("genre", sa.String(128), nullable=False, index=True),
        sa.Column("feature_cutoff_date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("game_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("median_price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("avg_review_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("total_positive_reviews", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("total_negative_reviews", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("median_playtime_forever", sa.Integer(), server_default="0", nullable=False),
        sa.Column("top_tags", JSONB, nullable=True),
        sa.Column("saturation_index", sa.Float(), server_default="0.0", nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("genre", "feature_cutoff_date", name="uq_market_features_genre_cutoff"),
    )
    op.create_index(
        "ix_feature_market_features_genre_cutoff",
        "feature_market_features",
        ["genre", "feature_cutoff_date"],
    )


def downgrade() -> None:
    op.drop_table("feature_market_features")
    op.drop_table("feature_game_features")
