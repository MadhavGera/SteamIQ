"""Add market_presence column to serving_similar_games

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-16 00:15:00.000000

Phase 5 (Roadmap v2 §3):
  - Adds additive market_presence column to serving_similar_games
    (derived from raw_player_snapshots volume / review count) so the existing
    row serves both 'Similar Games' (player) and 'Competitors' (developer)
    without creating a second table.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "serving_similar_games",
        sa.Column(
            "market_presence",
            sa.Numeric(5, 2),
            nullable=True,
            server_default=None,
        ),
    )
    # Ensure existing rows are explicitly NULL (unscored)
    op.execute("UPDATE serving_similar_games SET market_presence = NULL")


def downgrade() -> None:
    op.drop_column("serving_similar_games", "market_presence")
