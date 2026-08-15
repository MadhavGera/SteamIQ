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

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

# ---------------------------------------------------------------------------
# raw_games
# ---------------------------------------------------------------------------

class RawGame(Base):
    """
    Game metadata ingested from the Steam Store API and SteamSpy.
    Written only by ingestion jobs.
    """
    __tablename__ = "raw_games"

    app_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    short_description: Mapped[str | None] = mapped_column(Text)

    # Developer / publisher
    developer: Mapped[str | None] = mapped_column(String(512))
    publisher: Mapped[str | None] = mapped_column(String(512))

    # Release
    release_date: Mapped[str | None] = mapped_column(String(64))
    coming_soon: Mapped[bool] = mapped_column(Boolean, default=False)

    # Genres / categories stored as JSON arrays (denormalised for simplicity at raw zone)
    genres: Mapped[dict | None] = mapped_column(JSONB)          # [{"id": "1", "description": "Action"}, ...]
    categories: Mapped[dict | None] = mapped_column(JSONB)      # Steam categories
    tags: Mapped[dict | None] = mapped_column(JSONB)            # SteamSpy tags with vote counts

    # Pricing (USD, stored as cents to avoid float precision issues)
    is_free: Mapped[bool] = mapped_column(Boolean, default=False)
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    final_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))   # after discount
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)

    # Platform support
    platform_windows: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_mac: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_linux: Mapped[bool] = mapped_column(Boolean, default=False)

    # Review aggregate (from Steam)
    positive_reviews: Mapped[int] = mapped_column(Integer, default=0)
    negative_reviews: Mapped[int] = mapped_column(Integer, default=0)
    review_score: Mapped[int | None] = mapped_column(Integer)          # 0-9 Steam score
    review_score_desc: Mapped[str | None] = mapped_column(String(128)) # "Overwhelmingly Positive"

    # SteamSpy estimates
    owners_estimate: Mapped[str | None] = mapped_column(String(64))    # "2,000,000 .. 5,000,000"
    average_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    median_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)

    # Metacritic
    metacritic_score: Mapped[int | None] = mapped_column(Integer)

    # Media
    header_image: Mapped[str | None] = mapped_column(String(512))
    website: Mapped[str | None] = mapped_column(String(512))

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
    reviews: Mapped[list[RawReview]] = relationship(back_populates="game", lazy="noload")
    player_snapshots: Mapped[list[RawPlayerSnapshot]] = relationship(back_populates="game", lazy="noload")
    price_history: Mapped[list[RawPriceHistory]] = relationship(back_populates="game", lazy="noload")
    game_tags: Mapped[list[RawGameTag]] = relationship(back_populates="game", lazy="noload")
    patch_notes: Mapped[list[RawPatchNote]] = relationship(back_populates="game", lazy="noload")

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
    author_steam_id: Mapped[str | None] = mapped_column(String(64), index=True)
    author_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    author_playtime_at_review: Mapped[int] = mapped_column(Integer, default=0)
    author_num_reviews: Mapped[int] = mapped_column(Integer, default=0)

    # Review content
    language: Mapped[str] = mapped_column(String(16), default="english")
    review_text: Mapped[str | None] = mapped_column(Text)
    voted_up: Mapped[bool] = mapped_column(Boolean, nullable=False)  # True = positive

    # Helpfulness signals
    votes_up: Mapped[int] = mapped_column(Integer, default=0)
    votes_funny: Mapped[int] = mapped_column(Integer, default=0)
    weighted_vote_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))

    # Purchase type
    steam_purchase: Mapped[bool] = mapped_column(Boolean, default=True)
    received_for_free: Mapped[bool] = mapped_column(Boolean, default=False)
    written_during_early_access: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps (Unix epoch from Steam)
    review_created_at: Mapped[int | None] = mapped_column(BigInteger)
    review_updated_at: Mapped[int | None] = mapped_column(BigInteger)

    # Ingestion metadata
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped[RawGame] = relationship(back_populates="reviews")

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
    peak_24h: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="steam_api")  # "steam_api" | "steamspy"

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[RawGame] = relationship(back_populates="player_snapshots")

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
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    final_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[RawGame] = relationship(back_populates="price_history")

    def __repr__(self) -> str:
        return f"<RawPriceHistory app_id={self.app_id} price={self.final_price_usd}>"


# ---------------------------------------------------------------------------
# raw_patch_notes
# ---------------------------------------------------------------------------

class RawPatchNote(Base):
    """
    Authentic developer announcements and patch notes from Steam News API (ISteamNews/GetNewsForApp/v2).
    Written by jobs/ingest_news.py.
    """
    __tablename__ = "raw_patch_notes"
    __table_args__ = (
        Index("ix_raw_patch_notes_app_published", "app_id", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    gid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(128))
    contents: Mapped[str | None] = mapped_column(Text)
    feedlabel: Mapped[str | None] = mapped_column(String(128))
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    game: Mapped[RawGame] = relationship(back_populates="patch_notes")

    def __repr__(self) -> str:
        return f"<RawPatchNote app_id={self.app_id} title={self.title!r} date={self.published_at}>"


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

    game: Mapped[RawGame] = relationship(back_populates="game_tags")

    def __repr__(self) -> str:
        return f"<RawGameTag app_id={self.app_id} tag={self.tag_name!r} votes={self.votes}>"


# ===========================================================================
# FEATURE ZONE (Phase 2 — NLP Core)
# Written ONLY by NLP/feature pipeline jobs, read by serving layer & API handlers.
# ===========================================================================

# ---------------------------------------------------------------------------
# feature_review_sentiment
# ---------------------------------------------------------------------------

class FeatureReviewSentiment(Base):
    """
    Aggregated sentiment metrics per game, both overall and broken down monthly.
    Written by NLP processing pipeline jobs.
    """
    __tablename__ = "feature_review_sentiment"
    __table_args__ = (
        UniqueConstraint("app_id", "month", name="uq_game_sentiment_month"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    month: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # e.g. "2024-05" or "ALL_TIME"
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    net_positive_pct: Mapped[float] = mapped_column(Numeric(5, 2), default=0.0)  # 0.00 - 100.00%
    sentiment_score: Mapped[float] = mapped_column(Numeric(5, 4), default=0.0)   # 0.0000 - 1.0000

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureReviewSentiment app_id={self.app_id} month={self.month} net_pos={self.net_positive_pct}%>"


# ---------------------------------------------------------------------------
# feature_review_topics
# ---------------------------------------------------------------------------

class FeatureReviewTopic(Base):
    """
    Semantic topic clusters discovered from player reviews via BERTopic (or TF-IDF fallback).
    Written by NLP topic discovery pipeline.
    """
    __tablename__ = "feature_review_topics"
    __table_args__ = (
        UniqueConstraint("app_id", "topic_id", name="uq_game_topic"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_id: Mapped[int] = mapped_column(Integer, nullable=False)
    topic_label: Mapped[str] = mapped_column(String(256), nullable=False)        # e.g. "Combat & Movement Mechanics"
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    sentiment_score: Mapped[float] = mapped_column(Numeric(5, 4), default=0.5)  # 0.0000 - 1.0000
    keywords: Mapped[dict | None] = mapped_column(JSONB)                     # ["sword", "dodge", "fluid", "parry"]

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureReviewTopic app_id={self.app_id} label={self.topic_label!r} count={self.review_count}>"


# ---------------------------------------------------------------------------
# feature_review_complaints
# ---------------------------------------------------------------------------

class FeatureReviewComplaint(Base):
    """
    Classified complaint categories and representative quotes from zero-shot classification.
    Written by NLP complaint classification pipeline.
    """
    __tablename__ = "feature_review_complaints"
    __table_args__ = (
        UniqueConstraint("app_id", "category", name="uq_game_complaint"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(128), nullable=False)          # e.g. "Performance / FPS Drops"
    volume_pct: Mapped[float] = mapped_column(Numeric(5, 2), default=0.0)       # share of negative reviews
    severity: Mapped[str] = mapped_column(String(32), default="moderate")       # "high", "moderate", "low"
    representative_snippets: Mapped[dict | None] = mapped_column(JSONB)      # [quote1, quote2, quote3]

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureReviewComplaint app_id={self.app_id} cat={self.category!r} vol={self.volume_pct}%>"


# ---------------------------------------------------------------------------
# feature_review_features
# ---------------------------------------------------------------------------

class FeatureReviewFeature(Base):
    """
    Appreciated / loved game features extracted via embeddings & HDBSCAN.
    Written by NLP feature appreciation pipeline.
    """
    __tablename__ = "feature_review_features"
    __table_args__ = (
        UniqueConstraint("app_id", "feature_name", name="uq_game_loved_feature"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    feature_name: Mapped[str] = mapped_column(String(128), nullable=False)       # e.g. "Soundtrack & Audio"
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    praise_intensity: Mapped[int] = mapped_column(Integer, default=80)          # 0 - 100 score

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureReviewFeature app_id={self.app_id} feature={self.feature_name!r}>"


# ---------------------------------------------------------------------------
# feature_review_summary
# ---------------------------------------------------------------------------

class FeatureReviewSummary(Base):
    """
    Hierarchical review summary divided into Strengths, Pain Points, and Player Requests.
    Written by NLP summarization pipeline.
    """
    __tablename__ = "feature_review_summary"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    core_strengths: Mapped[dict | None] = mapped_column(JSONB)   # ["Pristine audio-visual execution", ...]
    pain_points: Mapped[dict | None] = mapped_column(JSONB)      # ["Early difficulty spike", ...]
    feature_requests: Mapped[dict | None] = mapped_column(JSONB) # ["Boss rush mode", ...]

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureReviewSummary app_id={self.app_id}>"


# ---------------------------------------------------------------------------
# model_game_embeddings (Phase 3)
# ---------------------------------------------------------------------------

class ModelGameEmbedding(Base):
    """
    Dense semantic vector representation of game metadata, tags, and review themes.
    Written by jobs/process_embeddings.py (Phase 3).
    """
    __tablename__ = "model_game_embeddings"
    __table_args__ = (
        UniqueConstraint("app_id", "model_name", name="uq_game_embedding_model"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="all-MiniLM-L6-v2")
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="1.0.0")
    embedding = mapped_column(Vector(384), nullable=False)
    text_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<ModelGameEmbedding app_id={self.app_id} model={self.model_name}>"


# ---------------------------------------------------------------------------
# serving_similar_games (Phase 3)
# ---------------------------------------------------------------------------

class ServingSimilarGame(Base):
    """
    Precomputed top-N closest competitor games by vector cosine similarity.
    Written by jobs/process_embeddings.py (Phase 3).
    Read by api/competitors.py (ADR 0001, Decision 2).
    """
    __tablename__ = "serving_similar_games"
    __table_args__ = (
        UniqueConstraint("source_app_id", "target_app_id", name="uq_serving_similar_pair"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)  # 0.0000 - 1.0000
    rank: Mapped[int] = mapped_column(Integer, nullable=False)                      # 1, 2, 3...
    shared_tags: Mapped[dict | None] = mapped_column(JSONB)                      # ["Metroidvania", "Difficult", ...]
    price_delta_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))      # target price - source price

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    source_game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[source_app_id], lazy="noload")
    target_game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[target_app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<ServingSimilarGame {self.source_app_id} -> {self.target_app_id} sim={self.similarity_score}>"


# ===========================================================================
# FEATURE ZONE (Phase 4 — Tabular ML Features)
# Written ONLY by feature pipeline jobs, read by training jobs.
# ===========================================================================

# ---------------------------------------------------------------------------
# feature_game_features
# ---------------------------------------------------------------------------

class FeatureGameFeature(Base):
    """
    Consolidated tabular feature vector for predictive models at a specific cutoff date.
    Written by jobs/build_features.py (Phase 4).
    Enforces feature_cutoff_date to strictly prevent temporal leakage during training.
    """
    __tablename__ = "feature_game_features"
    __table_args__ = (
        UniqueConstraint("app_id", "feature_cutoff_date", name="uq_game_features_app_cutoff"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    feature_cutoff_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Pricing features
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)
    is_free: Mapped[bool] = mapped_column(Boolean, default=False)

    # Genre & Tag features
    primary_genre: Mapped[str | None] = mapped_column(String(128))
    genres: Mapped[dict | None] = mapped_column(JSONB)
    top_tags: Mapped[dict | None] = mapped_column(JSONB)

    # Developer & Publisher historical track record
    developer_game_count: Mapped[int] = mapped_column(Integer, default=1)
    publisher_game_count: Mapped[int] = mapped_column(Integer, default=1)
    developer_avg_review_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    publisher_avg_review_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))

    # Review signals at cutoff
    total_reviews_at_cutoff: Mapped[int] = mapped_column(Integer, default=0)
    positive_reviews_at_cutoff: Mapped[int] = mapped_column(Integer, default=0)
    review_velocity_30d: Mapped[float] = mapped_column(Float, default=0.0)
    positive_review_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=0.0)

    # Competitor density & market positioning
    competitor_density: Mapped[int] = mapped_column(Integer, default=0)
    price_vs_genre_median: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))

    # Sentiment & NLP signals (from feature_review_* tables)
    sentiment_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    complaint_density: Mapped[float] = mapped_column(Float, default=0.0)
    loved_feature_density: Mapped[float] = mapped_column(Float, default=0.0)

    # Target / engagement labels for training
    average_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    target_success_score: Mapped[float | None] = mapped_column(Float)
    is_hit: Mapped[bool] = mapped_column(Boolean, default=False)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<FeatureGameFeature app_id={self.app_id} cutoff={self.feature_cutoff_date}>"


# ---------------------------------------------------------------------------
# feature_market_features
# ---------------------------------------------------------------------------

class FeatureMarketFeature(Base):
    """
    Market and genre-level aggregations computed at specific cutoff dates.
    Written by jobs/build_features.py (Phase 4).
    """
    __tablename__ = "feature_market_features"
    __table_args__ = (
        UniqueConstraint("genre", "feature_cutoff_date", name="uq_market_features_genre_cutoff"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    genre: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    feature_cutoff_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    game_count: Mapped[int] = mapped_column(Integer, default=0)
    median_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    avg_review_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    total_positive_reviews: Mapped[int] = mapped_column(BigInteger, default=0)
    total_negative_reviews: Mapped[int] = mapped_column(BigInteger, default=0)
    median_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    top_tags: Mapped[dict | None] = mapped_column(JSONB)
    saturation_index: Mapped[float] = mapped_column(Float, default=0.0)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FeatureMarketFeature genre={self.genre!r} games={self.game_count}>"


# ===========================================================================
# MODEL ZONE (Phase 4 — Predictive ML Runs & Artifacts)
# Written ONLY by training & promotion jobs. ADR 0001, Decision 6.
# ===========================================================================

# ---------------------------------------------------------------------------
# model_runs
# ---------------------------------------------------------------------------

class ModelRun(Base):
    """
    Tracks every ML model training run, metadata, hyperparameters, and evaluation metrics.
    Stages: candidate -> staging -> production -> archived.
    Written by jobs/train_success_model.py (Phase 4).
    """
    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # "candidate" | "staging" | "production" | "archived"
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    promoted_by: Mapped[str | None] = mapped_column(String(128))
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    predictions: Mapped[list[ServingPrediction]] = relationship(
        back_populates="model_run", lazy="noload"
    )

    def __repr__(self) -> str:
        return f"<ModelRun id={self.id} model={self.model_name!r} stage={self.stage!r}>"


# ===========================================================================
# SERVING ZONE (Phase 4 — Model Predictions & SHAP Explainability)
# Written ONLY by scoring/training pipeline jobs, read by API handlers in Phase 5.
# ===========================================================================

# ---------------------------------------------------------------------------
# serving_predictions
# ---------------------------------------------------------------------------

class ServingPrediction(Base):
    """
    Precomputed inference predictions with explainable SHAP feature values.
    Every row requires a model_run_id foreign key for 100% auditability.
    Written by jobs/train_success_model.py.
    """
    __tablename__ = "serving_predictions"
    __table_args__ = (
        UniqueConstraint(
            "app_id", "prediction_type", "model_run_id", name="uq_serving_predictions_app_type_run"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prediction_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # e.g. "success_score"
    score: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    confidence_lower: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    confidence_upper: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    feature_importance: Mapped[dict | None] = mapped_column(JSONB)
    shap_values: Mapped[dict | None] = mapped_column(JSONB)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")
    model_run: Mapped[ModelRun] = relationship("ModelRun", back_populates="predictions", lazy="noload")

    def __repr__(self) -> str:
        return f"<ServingPrediction app_id={self.app_id} type={self.prediction_type!r} score={self.score}>"


# ---------------------------------------------------------------------------
# serving_recommendations
# ---------------------------------------------------------------------------

class ServingRecommendation(Base):
    """
    Actionable recommendations synthesized by jobs/generate_recommendations.py.
    Carries model_run_id when influenced by ML / SHAP predictions, nullable when rules-only.
    Read by api/recommendations.py.
    """
    __tablename__ = "serving_recommendations"
    __table_args__ = (
        Index("ix_serving_recommendations_app_priority", "app_id", "priority_rank"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("model_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    recommendation_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    priority_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    impact_level: Mapped[str] = mapped_column(String(32), nullable=False)
    difficulty_level: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_payload: Mapped[dict | None] = mapped_column(JSONB)
    action_items: Mapped[list | None] = mapped_column(JSONB)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")
    model_run: Mapped[ModelRun | None] = relationship("ModelRun", foreign_keys=[model_run_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<ServingRecommendation app_id={self.app_id} rank={self.priority_rank} title={self.title!r}>"


# ===========================================================================
# MART ZONE (Phase 5 — Decision Intelligence & Data Marts)
# Written ONLY by materialization jobs, read by API handlers & dashboards.
# ADR 0001, Decision 1 & 2.
# ===========================================================================

# ---------------------------------------------------------------------------
# mart_game_overview
# ---------------------------------------------------------------------------

class MartGameOverview(Base):
    """
    Pre-materialized comprehensive game overview and dashboard KPI mart.
    Written by jobs/materialize_marts.py.
    Read by api/games.py and dashboard handlers.
    """
    __tablename__ = "mart_game_overview"
    __table_args__ = (
        Index("ix_mart_game_overview_genre_score", "primary_genre", "success_score"),
    )

    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    short_description: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    # Developer / publisher
    developer: Mapped[str | None] = mapped_column(String(512))
    publisher: Mapped[str | None] = mapped_column(String(512))

    # Release
    release_date: Mapped[str | None] = mapped_column(String(64))
    coming_soon: Mapped[bool] = mapped_column(Boolean, default=False)

    # Genres / categories / tags
    genres: Mapped[dict | list | None] = mapped_column(JSONB)
    categories: Mapped[dict | list | None] = mapped_column(JSONB)
    tags: Mapped[dict | None] = mapped_column(JSONB)

    # Pricing
    is_free: Mapped[bool] = mapped_column(Boolean, default=False)
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    final_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    discount_pct: Mapped[int] = mapped_column(Integer, default=0)

    # Platform support
    platform_windows: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_mac: Mapped[bool] = mapped_column(Boolean, default=False)
    platform_linux: Mapped[bool] = mapped_column(Boolean, default=False)

    # Review aggregate
    positive_reviews: Mapped[int] = mapped_column(Integer, default=0)
    negative_reviews: Mapped[int] = mapped_column(Integer, default=0)
    review_score: Mapped[int | None] = mapped_column(Integer)
    review_score_desc: Mapped[str | None] = mapped_column(String(128))

    # SteamSpy estimates
    owners_estimate: Mapped[str | None] = mapped_column(String(64))
    average_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)
    median_playtime_forever: Mapped[int] = mapped_column(Integer, default=0)

    # Metacritic & Media
    metacritic_score: Mapped[int | None] = mapped_column(Integer)
    header_image: Mapped[str | None] = mapped_column(String(512))
    website: Mapped[str | None] = mapped_column(String(512))

    # Decision / Mart Materialized Insights
    primary_genre: Mapped[str | None] = mapped_column(String(128), index=True)
    # Directional estimate based on public SteamSpy owner estimate range & final price (not verified financial data)
    revenue_tier: Mapped[str | None] = mapped_column(String(64))
    estimated_gross_revenue_usd: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    success_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), index=True)
    net_sentiment_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), index=True)
    peak_ccu_24h: Mapped[int | None] = mapped_column(Integer)
    executive_brief: Mapped[dict | None] = mapped_column(JSONB)
    top_strengths: Mapped[dict | list | None] = mapped_column(JSONB)
    top_complaints: Mapped[dict | list | None] = mapped_column(JSONB)

    # Pricing Intelligence (comparable-range output, historical low, no causal elasticity claims)
    price_tracking_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    historical_lowest_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    historical_lowest_discount_pct: Mapped[int | None] = mapped_column(Integer, default=0)
    genre_median_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    genre_min_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    genre_max_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    genre_p25_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    genre_p75_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    pricing_spectrum: Mapped[dict | None] = mapped_column(JSONB)
    price_history_points: Mapped[list | dict | None] = mapped_column(JSONB)

    materialized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<MartGameOverview app_id={self.app_id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# mart_trends
# ---------------------------------------------------------------------------

class MartTrend(Base):
    """
    Time-series and patch impact trends across games and market segments.
    Written by jobs/materialize_marts.py.
    """
    __tablename__ = "mart_trends"
    __table_args__ = (
        Index("ix_mart_trends_app_type", "app_id", "trend_type"),
        Index("ix_mart_trends_type_cat", "trend_type", "category"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=True, index=True
    )
    trend_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    recorded_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    change_pct_7d: Mapped[Decimal | None] = mapped_column(Numeric(7, 2))
    change_pct_30d: Mapped[Decimal | None] = mapped_column(Numeric(7, 2))
    metadata_payload: Mapped[dict | None] = mapped_column(JSONB)

    materialized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped[RawGame | None] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<MartTrend id={self.id} type={self.trend_type!r} metric={self.metric_name!r}>"


# ---------------------------------------------------------------------------
# mart_opportunity_scores
# ---------------------------------------------------------------------------

class MartOpportunityScore(Base):
    """
    Market opportunity analytics scores by genre and tag clusters.
    Computed via weighted scoring formula (analytics only, not ML).
    Written by jobs/materialize_marts.py.
    """
    __tablename__ = "mart_opportunity_scores"
    __table_args__ = (
        UniqueConstraint("genre_or_tag", "entity_type", name="uq_mart_opportunity_entity"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    genre_or_tag: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), default="genre", nullable=False)
    opportunity_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, index=True)
    demand_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    saturation_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    sentiment_gap_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    monetization_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    game_count: Mapped[int] = mapped_column(Integer, default=0)
    median_price_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    avg_review_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    top_complaint_themes: Mapped[dict | list | None] = mapped_column(JSONB)
    top_loved_themes: Mapped[dict | list | None] = mapped_column(JSONB)
    recommended_features: Mapped[dict | list | None] = mapped_column(JSONB)

    materialized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<MartOpportunityScore entity={self.genre_or_tag!r} type={self.entity_type!r} score={self.opportunity_score}>"


# ---------------------------------------------------------------------------
# mart_update_impact
# ---------------------------------------------------------------------------

class MartUpdateImpact(Base):
    """
    Pre-materialized update/patch impact before-and-after window metrics.
    Compares sentiment, player CCU, and complaint shifts around detected patch dates.
    Results are strictly labeled 'observed/correlated' (never 'caused').
    Written by jobs/materialize_update_impact.py.
    Read by api/updates.py.
    """
    __tablename__ = "mart_update_impact"
    __table_args__ = (
        Index("ix_mart_update_impact_app_date", "app_id", "patch_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("raw_games.app_id", ondelete="CASCADE"), nullable=False, index=True
    )
    patch_name: Mapped[str] = mapped_column(String(256), nullable=False)
    patch_version: Mapped[str | None] = mapped_column(String(64))
    patch_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    is_inferred: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    window_days: Mapped[int] = mapped_column(Integer, default=14, nullable=False)

    # Observed Sentiment Delta
    pre_sentiment_positive_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    post_sentiment_positive_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    sentiment_delta_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    observed_sentiment_verdict: Mapped[str] = mapped_column(String(64), nullable=False)

    # Observed Player Activity Delta
    pre_avg_ccu: Mapped[int | None] = mapped_column(Integer)
    post_avg_ccu: Mapped[int | None] = mapped_column(Integer)
    ccu_change_pct: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))

    # Observed Complaint Topic Shifts
    pre_complaint_distribution: Mapped[dict | None] = mapped_column(JSONB)
    post_complaint_distribution: Mapped[dict | None] = mapped_column(JSONB)
    top_resolved_complaints: Mapped[list | None] = mapped_column(JSONB)
    top_emerging_complaints: Mapped[list | None] = mapped_column(JSONB)

    # Non-Causal Summary
    correlation_summary: Mapped[str] = mapped_column(Text, nullable=False)

    materialized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    game: Mapped[RawGame] = relationship("RawGame", foreign_keys=[app_id], lazy="noload")

    def __repr__(self) -> str:
        return f"<MartUpdateImpact app_id={self.app_id} patch={self.patch_name!r} verdict={self.observed_sentiment_verdict!r}>"



