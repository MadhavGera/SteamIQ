"""phase 5 raw_patch_notes table (Steam News / Patch Notes ingestion)

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-15 03:30:00.000000

Phase 5 — Steam News & Patch Notes:
  - Creates raw_patch_notes table storing authentic developer announcements & patch notes
    fetched from Steam News API (ISteamNews/GetNewsForApp/v2).
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_patch_notes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "app_id",
            sa.Integer(),
            sa.ForeignKey("raw_games.app_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("gid", sa.String(64), unique=True, nullable=False, index=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("author", sa.String(128), nullable=True),
        sa.Column("contents", sa.Text(), nullable=True),
        sa.Column("feedlabel", sa.String(128), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_raw_patch_notes_app_published",
        "raw_patch_notes",
        ["app_id", "published_at"],
    )


def downgrade() -> None:
    op.drop_table("raw_patch_notes")
