"""phase 5 player activity telemetry (player activity payload on mart_game_overview)

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-19 14:00:00.000000

Phase 5 — Player Activity Telemetry:
  - Adds player_activity JSONB column to mart_game_overview to store pre-materialized
    CCU history, empirical playtime percentiles, and retention cohort distributions.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mart_game_overview", sa.Column("player_activity", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("mart_game_overview", "player_activity")
