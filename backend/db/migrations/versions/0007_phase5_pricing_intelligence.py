"""phase 5 pricing intelligence (comparable price spectrum, historical low, and price history points)

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-15 01:00:00.000000

Phase 5 — Pricing Intelligence:
  - Adds historical low, genre price distribution benchmarks (min/p25/median/p75/max),
    pricing spectrum comparable object, and price history points to mart_game_overview.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mart_game_overview", sa.Column("price_tracking_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("mart_game_overview", sa.Column("historical_lowest_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("historical_lowest_discount_pct", sa.Integer(), server_default=sa.text("0"), nullable=True))
    op.add_column("mart_game_overview", sa.Column("genre_median_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("genre_min_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("genre_max_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("genre_p25_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("genre_p75_price_usd", sa.Numeric(10, 2), nullable=True))
    op.add_column("mart_game_overview", sa.Column("pricing_spectrum", JSONB, nullable=True))
    op.add_column("mart_game_overview", sa.Column("price_history_points", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("mart_game_overview", "price_history_points")
    op.drop_column("mart_game_overview", "pricing_spectrum")
    op.drop_column("mart_game_overview", "genre_p75_price_usd")
    op.drop_column("mart_game_overview", "genre_p25_price_usd")
    op.drop_column("mart_game_overview", "genre_max_price_usd")
    op.drop_column("mart_game_overview", "genre_min_price_usd")
    op.drop_column("mart_game_overview", "genre_median_price_usd")
    op.drop_column("mart_game_overview", "historical_lowest_discount_pct")
    op.drop_column("mart_game_overview", "historical_lowest_price_usd")
    op.drop_column("mart_game_overview", "price_tracking_started_at")
