"""phase 5 mart_game_match_profile table (Game Match Profile)

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-16 00:05:00.000000

Phase 5 — Decision Intelligence / Game Match (Roadmap v2 §3):
  - Creates mart_game_match_profile table storing per-game intensity scores
    on a fixed 6-dimension set (difficulty, story_weight, exploration, combat,
    multiplayer, session_length) derived from NLP topics and store tags.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── mart_game_match_profile ───────────────────────────────────────────────
    op.create_table(
        "mart_game_match_profile",
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            primary_key=True,
            index=True,
        ),
        sa.Column("difficulty", sa.Numeric(4, 2), nullable=False, server_default="5.0"),
        sa.Column("story_weight", sa.Numeric(4, 2), nullable=False, server_default="5.0"),
        sa.Column("exploration", sa.Numeric(4, 2), nullable=False, server_default="5.0"),
        sa.Column("combat", sa.Numeric(4, 2), nullable=False, server_default="5.0"),
        sa.Column("multiplayer", sa.Numeric(4, 2), nullable=False, server_default="0.0"),
        sa.Column("session_length", sa.Numeric(4, 2), nullable=False, server_default="5.0"),
        sa.Column("confidence_score", sa.Numeric(4, 2), nullable=False, server_default="0.80"),
        sa.Column("profile_summary", JSONB, nullable=True),
        sa.Column(
            "materialized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("mart_game_match_profile")
