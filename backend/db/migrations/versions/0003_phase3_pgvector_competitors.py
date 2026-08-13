"""phase 3 pgvector and competitors tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-11 12:00:00.000000

Phase 3 — Similarity & Competitors:
  - Enable pgvector extension
  - model_game_embeddings (model_* zone)
  - serving_similar_games (serving_* zone)
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. Enable pgvector extension ──────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # ── 2. model_game_embeddings ──────────────────────────────────────────────
    op.create_table(
        "model_game_embeddings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(128), server_default="all-MiniLM-L6-v2", nullable=False),
        sa.Column("model_version", sa.String(64), server_default="1.0.0", nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
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
        sa.UniqueConstraint("app_id", "model_name", name="uq_game_embedding_model"),
    )
    op.create_index(
        "ix_model_game_embeddings_app_id",
        "model_game_embeddings",
        ["app_id"],
    )
    op.create_index(
        "ix_model_game_embeddings_text_hash",
        "model_game_embeddings",
        ["text_hash"],
    )

    # ── 3. serving_similar_games ──────────────────────────────────────────────
    op.create_table(
        "serving_similar_games",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "source_app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("similarity_score", sa.Numeric(5, 4), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("shared_tags", JSONB, nullable=True),
        sa.Column("price_delta_usd", sa.Numeric(10, 2), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("source_app_id", "target_app_id", name="uq_serving_similar_pair"),
    )
    op.create_index(
        "ix_serving_similar_games_source_app_id",
        "serving_similar_games",
        ["source_app_id"],
    )
    op.create_index(
        "ix_serving_similar_games_target_app_id",
        "serving_similar_games",
        ["target_app_id"],
    )


def downgrade() -> None:
    op.drop_table("serving_similar_games")
    op.drop_table("model_game_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector;")
