"""create raw zone tables

Revision ID: 0001
Revises: 
Create Date: 2026-08-06 07:00:00.000000

Phase 1 — Foundation: all raw_* tables.
Zone convention (ADR 0001, Decision 1):
  raw_*     → ingestion jobs write; API handlers may read only as a flagged stopgap
  feature_* → Phase 2+
  serving_* → Phase 3+
  mart_*    → Phase 5+
  model_*   → Phase 4+
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── raw_games ─────────────────────────────────────────────────────────────
    op.create_table(
        "raw_games",
        sa.Column("app_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("short_description", sa.Text(), nullable=True),
        sa.Column("developer", sa.String(512), nullable=True),
        sa.Column("publisher", sa.String(512), nullable=True),
        sa.Column("release_date", sa.String(64), nullable=True),
        sa.Column("coming_soon", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("genres", JSONB(), nullable=True),
        sa.Column("categories", JSONB(), nullable=True),
        sa.Column("tags", JSONB(), nullable=True),
        sa.Column("is_free", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("final_price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("discount_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("platform_windows", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("platform_mac", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("platform_linux", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("positive_reviews", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("negative_reviews", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("review_score", sa.Integer(), nullable=True),
        sa.Column("review_score_desc", sa.String(128), nullable=True),
        sa.Column("owners_estimate", sa.String(64), nullable=True),
        sa.Column("average_playtime_forever", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("median_playtime_forever", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metacritic_score", sa.Integer(), nullable=True),
        sa.Column("header_image", sa.String(512), nullable=True),
        sa.Column("website", sa.String(512), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_raw_games_name", "raw_games", ["name"])

    # ── raw_reviews ───────────────────────────────────────────────────────────
    op.create_table(
        "raw_reviews",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("review_id", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("author_steam_id", sa.String(64), nullable=True),
        sa.Column("author_playtime_forever", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("author_playtime_at_review", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("author_num_reviews", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("language", sa.String(16), nullable=False, server_default="'english'"),
        sa.Column("review_text", sa.Text(), nullable=True),
        sa.Column("voted_up", sa.Boolean(), nullable=False),
        sa.Column("votes_up", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("votes_funny", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("weighted_vote_score", sa.Numeric(6, 4), nullable=True),
        sa.Column("steam_purchase", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("received_for_free", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("written_during_early_access", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("review_created_at", sa.BigInteger(), nullable=True),
        sa.Column("review_updated_at", sa.BigInteger(), nullable=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_raw_reviews_app_id", "raw_reviews", ["app_id"])
    op.create_index("ix_raw_reviews_review_id", "raw_reviews", ["review_id"])
    op.create_index("ix_raw_reviews_author", "raw_reviews", ["author_steam_id"])

    # ── raw_player_snapshots ──────────────────────────────────────────────────
    op.create_table(
        "raw_player_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("player_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("peak_24h", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="'steam_api'"),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "snapshot_at", "source", name="uq_player_snapshot"),
    )
    op.create_index("ix_raw_player_snapshots_app_id", "raw_player_snapshots", ["app_id"])

    # ── raw_price_history ─────────────────────────────────────────────────────
    op.create_table(
        "raw_price_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("final_price_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column("discount_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(8), nullable=False, server_default="'USD'"),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "recorded_at", name="uq_price_history"),
    )
    op.create_index("ix_raw_price_history_app_id", "raw_price_history", ["app_id"])

    # ── raw_game_tags ─────────────────────────────────────────────────────────
    op.create_table(
        "raw_game_tags",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag_name", sa.String(128), nullable=False),
        sa.Column("votes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("app_id", "tag_name", name="uq_game_tag"),
    )
    op.create_index("ix_raw_game_tags_app_id", "raw_game_tags", ["app_id"])
    op.create_index("ix_raw_game_tags_tag_name", "raw_game_tags", ["tag_name"])


def downgrade() -> None:
    op.drop_table("raw_game_tags")
    op.drop_table("raw_price_history")
    op.drop_table("raw_player_snapshots")
    op.drop_table("raw_reviews")
    op.drop_table("raw_games")
