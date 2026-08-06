"""
Raw zone ORM models — Phase 1.

Naming convention enforced by ADR 0001, Decision 1:
  raw_*      → written by ingestion jobs, read by feature pipeline jobs
  feature_*  → Phase 2+
  serving_*  → Phase 3+
  mart_*     → Phase 5+
  model_*    → Phase 4+

All tables here are in the raw_* zone. Feature/serving/mart/model tables
are added in their respective phases.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


# ---------------------------------------------------------------------------
# raw_games
# ---------------------------------------------------------------------------

class RawGame(Base):
    """
    Game metadata ingested from the Steam Store API and SteamSpy.
    Written only by ingestion jobs.

    Phase 1 stopgap: api/games.py reads from this table directly.
    TODO(Phase5): repoint api/games.py to mart_game_overview once it exists.
    """
    __tablename__ = "raw_games"

    app_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    short_description: Mapped[Optional[str]] = mapped_column(Text)

    # Developer / publisher
    developer: Mapped[Optional[str]] = mapped_column(String(512))
    publisher: Mapped[Optional[str]] = mapped_column(String(512))

    # Release
    release_date: Mapped[Optional[str]] = mapped_column(String(64))
    coming_soon: Mapped[bool] = mapped_column(Boolean, default=False)

    # Genres / categories stored as JSON arrays (denormalised for simplicity at raw zone)
    genres: Mapped[Optional[dict]] = mapped_column(JSONB)          # [{"id": "1", "description": "Action"}, ...]
    categories: Mapped[Optional[dict]] = mapped_column(JSONB)      # Steam categories
    tags: Mapped[Optional[dict]] = mapped_column(JSONB)            # SteamSpy tags with vote counts

    # Pricing (USD, stored as cents to avoid float precision issues)
    is_free: Mapped[bool] = mapped_column(Boolean, default=False)
    price_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    final_price_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))   # after discount
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)

    # Platform support
    platform_windows: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_mac: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_linux: Mapped[bool] = mapped_column(Boolean, default=False)

    # Review aggregate (from Steam)
    positive_reviews: Mapped[int] = mapped_column(Integer, default=0)
    negative_reviews: Mapped[int] = mapped_column(Integer, default=0)
    review_score: Mapped[Optional[int]] = mapped_column(Integer)          # 0-9 Steam score
    review_score_desc: Mapped[Optional[str]] = mapped_column(String(128)) # "Overwhelmingly Positive"

    # SteamSpy estimates
    owners_estimate: Mapped[Optional[str]] = mapped_column(String(64))    # "2,000,000 .. 5,000,000"
    average_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    median_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)

    # Metacritic
    metacritic_score: Mapped[Optional[int]] = mapped_column(Integer)

    # Media
    header_image: Mapped[Optional[str]] = mapped_column(String(512))
    website: Mapped[Optional[str]] = mapped_column(String(512))

    # Ingestion metadata
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    reviews: Mapped[list["RawReview"]] = relationship(back_populates="game", lazy="noload")
    player_snapshots: Mapped[list["RawPlayerSnapshot"]] = relationship(back_populates="game", lazy="noload")
    price_history: Mapped[list["RawPriceHistory"]] = relationship(back_populates="game", lazy="noload")
    game_tags: Mapped[list["RawGameTag"]] = relationship(back_populates="game", lazy="noload")

    def __repr__(self) -> str:
        return f"<RawGame app_id={self.app_id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# raw_reviews
# ---------------------------------------------------------------------------

class RawReview(Base):
    """
    Individual Steam user reviews for a game.
    Written only by ingestion jobs.
    """
    __tablename__ = "raw_reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Author
    author_steam_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    author_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    author_playtime_at_review: Mapped[int] = mapped_column(Integer, default=0)
    author_num_reviews: Mapped[int] = mapped_column(Integer, default=0)

    # Review content
    language: Mapped[str] = mapped_column(String(16), default="english")
    review_text: Mapped[Optional[str]] = mapped_column(Text)
    voted_up: Mapped[bool] = mapped_column(Boolean, nullable=False)  # True = positive

    # Helpfulness signals
    votes_up: Mapped[int] = mapped_column(Integer, default=0)
    votes_funny: Mapped[int] = mapped_column(Integer, default=0)
    weighted_vote_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Purchase type
    steam_purchase: Mapped[bool] = mapped_column(Boolean, default=True)
    received_for_free: Mapped[bool] = mapped_column(Boolean, default=False)
    written_during_early_access: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps (Unix epoch from Steam)
    review_created_at: Mapped[Optional[int]] = mapped_column(BigInteger)
    review_updated_at: Mapped[Optional[int]] = mapped_column(BigInteger)

    # Ingestion metadata
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped["RawGame"] = relationship(back_populates="reviews")

    def __repr__(self) -> str:
        return f"<RawReview review_id={self.review_id} app_id={self.app_id} voted_up={self.voted_up}>"


# ---------------------------------------------------------------------------
# raw_player_snapshots
# ---------------------------------------------------------------------------

class RawPlayerSnapshot(Base):
    """
    Point-in-time player count snapshot from Steam API.
    Written only by ingestion jobs (run on a schedule).
    """
    __tablename__ = "raw_player_snapshots"
    __table_args__ = (
        UniqueConstraint("app_id", "snapshot_at", "source", name="uq_player_snapshot"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    player_count: Mapped[int] = mapped_column(Integer, default=0)
    peak_24h: Mapped[Optional[int]] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="steam_api")  # "steam_api" | "steamspy"

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped["RawGame"] = relationship(back_populates="player_snapshots")

    def __repr__(self) -> str:
        return f"<RawPlayerSnapshot app_id={self.app_id} players={self.player_count}>"


# ---------------------------------------------------------------------------
# raw_price_history
# ---------------------------------------------------------------------------

class RawPriceHistory(Base):
    """
    Price recorded at each ingestion run, enabling trend analysis in Phase 5.
    Written only by ingestion jobs.
    """
    __tablename__ = "raw_price_history"
    __table_args__ = (
        UniqueConstraint("app_id", "recorded_at", name="uq_price_history"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    final_price_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped["RawGame"] = relationship(back_populates="price_history")

    def __repr__(self) -> str:
        return f"<RawPriceHistory app_id={self.app_id} price={self.final_price_usd}>"


# ---------------------------------------------------------------------------
# raw_game_tags
# ---------------------------------------------------------------------------

class RawGameTag(Base):
    """
    Steam/SteamSpy tags for a game, with vote counts.
    Stored here (denormalised from raw_games.tags JSONB) for easier querying
    in feature pipelines without JSON unnesting.
    Written only by ingestion jobs.
    """
    __tablename__ = "raw_game_tags"
    __table_args__ = (
        UniqueConstraint("app_id", "tag_name", name="uq_game_tag"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    votes: Mapped[int] = mapped_column(Integer, default=0)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped["RawGame"] = relationship(back_populates="game_tags")

    def __repr__(self) -> str:
        return f"<RawGameTag app_id={self.app_id} tag={self.tag_name!r} votes={self.votes}>"
