"""phase 5 mart tables (mart_game_overview, mart_trends, mart_opportunity_scores)

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-15 00:00:00.000000

Phase 5 — Decision Intelligence & Data Mart Layer:
  - mart_game_overview (comprehensive pre-materialized game overview and dashboard KPIs)
  - mart_trends (patch impact and platform/genre trend time-series metrics)
  - mart_opportunity_scores (market opportunity analytics weighted scoring)
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. mart_game_overview ────────────────────────────────────────────────
    op.create_table(
        "mart_game_overview",
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            primary_key=True,
            index=True,
        ),
        sa.Column("name", sa.String(512), nullable=False, index=True),
        sa.Column("short_description", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("developer", sa.String(512), nullable=True),
        sa.Column("publisher", sa.String(512), nullable=True),
        sa.Column("release_date", sa.String(64), nullable=True),
        sa.Column("coming_soon", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("genres", JSONB, nullable=True),
        sa.Column("categories", JSONB, nullable=True),
        sa.Column("tags", JSONB, nullable=True),
        sa.Column("is_free", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("final_price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("discount_pct", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("platform_windows", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("platform_mac", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("platform_linux", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("positive_reviews", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("negative_reviews", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("review_score", sa.Integer(), nullable=True),
        sa.Column("review_score_desc", sa.String(128), nullable=True),
        sa.Column("owners_estimate", sa.String(64), nullable=True),
        sa.Column("average_playtime_forever", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("median_playtime_forever", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("metacritic_score", sa.Integer(), nullable=True),
        sa.Column("header_image", sa.String(512), nullable=True),
        sa.Column("website", sa.String(512), nullable=True),
        # Decision & Mart materializations
        sa.Column("primary_genre", sa.String(128), nullable=True, index=True),
        sa.Column("revenue_tier", sa.String(64), nullable=True),
        sa.Column("estimated_gross_revenue_usd", sa.Numeric(14, 2), nullable=True),
        sa.Column("success_score", sa.Numeric(5, 2), nullable=True, index=True),
        sa.Column("net_sentiment_pct", sa.Numeric(5, 2), nullable=True, index=True),
        sa.Column("peak_ccu_24h", sa.Integer(), nullable=True),
        sa.Column("executive_brief", JSONB, nullable=True),
        sa.Column("top_strengths", JSONB, nullable=True),
        sa.Column("top_complaints", JSONB, nullable=True),
        sa.Column("shap_values", JSONB, nullable=True),
        sa.Column("model_run_id", sa.String(64), nullable=True),
        sa.Column(
            "materialized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_mart_game_overview_genre_score",
        "mart_game_overview",
        ["primary_genre", "success_score"],
    )

    # ── 2. mart_trends ────────────────────────────────────────────────────────
    op.create_table(
        "mart_trends",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=True,
            index=True,
        ),
        sa.Column("trend_type", sa.String(64), nullable=False, index=True),
        sa.Column("category", sa.String(128), nullable=True, index=True),
        sa.Column("recorded_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric_name", sa.String(128), nullable=False),
        sa.Column("metric_value", sa.Numeric(12, 4), nullable=False),
        sa.Column("change_pct_7d", sa.Numeric(7, 2), nullable=True),
        sa.Column("change_pct_30d", sa.Numeric(7, 2), nullable=True),
        sa.Column("metadata_payload", JSONB, nullable=True),
        sa.Column(
            "materialized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_mart_trends_app_type",
        "mart_trends",
        ["app_id", "trend_type"],
    )
    op.create_index(
        "ix_mart_trends_type_cat",
        "mart_trends",
        ["trend_type", "category"],
    )

    # ── 3. mart_opportunity_scores ────────────────────────────────────────────
    op.create_table(
        "mart_opportunity_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("genre_or_tag", sa.String(128), nullable=False, index=True),
        sa.Column("entity_type", sa.String(32), server_default="genre", nullable=False),
        sa.Column("opportunity_score", sa.Numeric(5, 2), nullable=False, index=True),
        sa.Column("demand_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("saturation_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("sentiment_gap_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("monetization_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("game_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("median_price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("avg_review_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("top_complaint_themes", JSONB, nullable=True),
        sa.Column("top_loved_themes", JSONB, nullable=True),
        sa.Column("recommended_features", JSONB, nullable=True),
        sa.Column(
            "materialized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("genre_or_tag", "entity_type", name="uq_mart_opportunity_entity"),
    )


def downgrade() -> None:
    op.drop_table("mart_opportunity_scores")
    op.drop_table("mart_trends")
    op.drop_table("mart_game_overview")
