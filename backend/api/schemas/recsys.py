"""
Pydantic v2 schemas for Recommender System Serving API — Phase 6.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecsysCatalogItemSchema(BaseModel):
    """Item representation in the recommender serving catalog."""
    model_config = ConfigDict(from_attributes=True)

    app_id: int = Field(..., description="Steam App ID")
    item_idx: int = Field(..., description="0-indexed dense integer matrix index")
    title: str = Field(..., description="Game title")
    date_release: str | None = Field(default=None, description="Release date string")
    price_final: float | None = Field(default=None, description="Final price in USD")
    positive_ratio: int | None = Field(default=None, description="Positive review percentage (0..100)")
    user_reviews: int | None = Field(default=None, description="Total user reviews count")
    tags: list[str] = Field(default_factory=list, description="Top Steam community tags")
    description_short: str | None = Field(default=None, description="Short store description")
    has_description: bool = Field(default=False, description="Flag indicating non-empty description")
    train_positives: int = Field(default=0, description="Number of positive user interactions in train split")
    popularity_rank: int = Field(default=0, description="Global popularity rank (1 = most played)")
    win: bool = Field(default=False, description="Windows support flag")
    mac: bool = Field(default=False, description="macOS support flag")
    linux: bool = Field(default=False, description="Linux support flag")
    steam_deck: bool = Field(default=False, description="Steam Deck verified/playable flag")
    header_image_url: str = Field(default="", description="CDN header image URL")


class RecsysSearchResponseSchema(BaseModel):
    """Response payload for catalog item search."""
    items: list[RecsysCatalogItemSchema] = Field(default_factory=list, description="Matching catalog games")
    total: int = Field(..., description="Total matching items found")
    universe_size: int = Field(..., description="Total universe catalog size")
    query: str = Field(default="", description="Search query string")


class RecsysForYouRequest(BaseModel):
    """Request payload for personalized recommendations."""
    liked_app_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of Steam App IDs the user played or liked (1..50 items)",
    )
    model: str | None = Field(
        default=None,
        description="Deployable model identifier or alias (e.g. 'two_tower_clean', 'ranker_clean', 'itemknn', 'als', 'content', 'popularity')",
    )
    k: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of recommendations to return (1..50)",
    )
    explain: bool = Field(
        default=True,
        description="Whether to generate explainability reasons and anchor games",
    )


class RecsysAnchorGame(BaseModel):
    """History item anchor for explainability."""
    app_id: int = Field(..., description="Steam App ID")
    title: str = Field(..., description="Game title")
    similarity: float = Field(..., description="Cosine similarity or weight in latent space")


class RecsysRecommendationItemSchema(BaseModel):
    """Individual recommended game card with score and explanations."""
    model_config = ConfigDict(from_attributes=True)

    rank: int = Field(..., description="Recommendation rank position (1..k)")
    app_id: int = Field(..., description="Steam App ID")
    item_idx: int = Field(..., description="Dense catalog index")
    title: str = Field(..., description="Game title")
    header_image_url: str = Field(default="", description="CDN header image URL")
    score: float = Field(..., description="Recommender relevance score")
    price_final: float | None = Field(default=None, description="Final price in USD")
    positive_ratio: int | None = Field(default=None, description="Positive review percentage")
    user_reviews: int | None = Field(default=None, description="Total user reviews count")
    tags: list[str] = Field(default_factory=list, description="Top Steam community tags")
    description_short: str | None = Field(default=None, description="Short store description")
    popularity_rank: int = Field(..., description="Global popularity rank")
    win: bool = Field(default=False, description="Windows support flag")
    mac: bool = Field(default=False, description="macOS support flag")
    linux: bool = Field(default=False, description="Linux support flag")
    steam_deck: bool = Field(default=False, description="Steam Deck support flag")
    because_you_liked: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History games that contributed most to this recommendation",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Natural language explanation statements",
    )


class RecsysForYouResponse(BaseModel):
    """Personalized recommendation response payload."""
    items: list[RecsysRecommendationItemSchema] = Field(
        default_factory=list,
        description="Rank-ordered personalized recommendations",
    )
    total_items: int = Field(..., description="Number of items returned")
    model_name: str = Field(..., description="Canonical model identifier used for inference")
    model_run_id: uuid.UUID | str | None = Field(default=None, description="ModelRun provenance UUID")
    val_ndcg_at_10: float | None = Field(default=None, description="Model validation NDCG@10 metric")
    dataset_version: str = Field(..., description="Training dataset version fingerprint")
    universe_size: int = Field(..., description="Catalog universe size")
    unknown_app_ids: list[int] = Field(
        default_factory=list,
        description="Input app_ids that were not recognized in the universe",
    )
    warning: str | None = Field(default=None, description="Personalization or library size warning")
    preview: bool = Field(default=False, description="Preview flag for experimental models")
    took_ms: float = Field(..., description="Inference compute time in milliseconds")


class RecsysModelInfoSchema(BaseModel):
    """Metadata and benchmark metrics for a deployable recommender model."""
    model_name: str = Field(..., description="Canonical model name")
    display_name: str = Field(..., description="Human-friendly model title")
    deployable: bool = Field(default=True, description="Whether this model is approved for online serving")
    stage: str = Field(..., description="Registry stage (production, candidate, reference, staging)")
    model_run_id: uuid.UUID | str | None = Field(default=None, description="Latest model_run ID")
    val_ndcg_at_10: float | None = Field(default=None, description="Validation NDCG@10")
    test_ndcg_at_10_mean: float | None = Field(default=None, description="Benchmark test NDCG@10 mean")
    test_ndcg_at_10_std: float | None = Field(default=None, description="Benchmark test NDCG@10 std across seeds")
    test_hr_at_10_mean: float | None = Field(default=None, description="Benchmark test HR@10 mean")
    test_hr_at_10_std: float | None = Field(default=None, description="Benchmark test HR@10 std across seeds")
    n_seeds: int = Field(default=1, description="Number of evaluation seeds")
    is_ready: bool = Field(default=True, description="Whether the model artifact is loaded in memory")
    is_preview: bool = Field(default=False, description="Preview status flag")
    leak_free: bool = Field(default=False, description="Whether training interactions and retrieval are leak-free")
    clean: bool = Field(default=False, description="Whether the model uses clean stage-1 retrieval")


class RecsysReportSchema(BaseModel):
    """Recommender offline audit / evaluation report payload."""
    model_config = ConfigDict(from_attributes=True)

    report_key: str = Field(..., description="Report identifier key")
    dataset_version: str = Field(..., description="Dataset fingerprint version")
    model_run_id: uuid.UUID | None = Field(default=None, description="Associated ModelRun UUID")
    payload: dict[str, Any] = Field(..., description="Full structured JSON report content")
    updated_at: datetime | None = Field(default=None, description="Last updated timestamp")
    source: str = Field(default="database", description="'database' or 'file_fallback'")


class RecsysMapPointSchema(BaseModel):
    """2D UMAP coordinate projection and semantic cluster for an item."""
    model_config = ConfigDict(from_attributes=True)

    app_id: int = Field(..., description="Steam App ID")
    title: str = Field(..., description="Game title")
    x: float = Field(..., description="2D UMAP X coordinate")
    y: float = Field(..., description="2D UMAP Y coordinate")
    cluster_id: int | None = Field(default=None, description="KMeans cluster ID (0..19)")
    cluster_label: str | None = Field(default=None, description="Top distinctive tags describing cluster")
    popularity_rank: int | None = Field(default=None, description="Catalog popularity rank")
    header_image_url: str | None = Field(default=None, description="CDN header image URL")
    tags: list[str] = Field(default_factory=list, description="Top community tags")


class RecsysMapResponse(BaseModel):
    """Full 2D item map response."""
    points: list[RecsysMapPointSchema] = Field(default_factory=list, description="Projected catalog points")
    total_points: int = Field(..., description="Total points returned")
    universe_size: int = Field(..., description="Total universe size")


class RecsysNeighborItemSchema(BaseModel):
    """Nearest neighbor item in Two-Tower embedding space."""
    model_config = ConfigDict(from_attributes=True)

    rank: int = Field(..., description="Neighbor rank position (1..k)")
    app_id: int = Field(..., description="Steam App ID")
    title: str = Field(..., description="Game title")
    header_image_url: str = Field(default="", description="CDN header image URL")
    similarity: float = Field(..., description="Cosine similarity in Two-Tower 64-d space")
    price_final: float | None = Field(default=None, description="Final price in USD")
    positive_ratio: int | None = Field(default=None, description="Positive review percentage")
    user_reviews: int | None = Field(default=None, description="Total user reviews count")
    tags: list[str] = Field(default_factory=list, description="Top Steam community tags")
    popularity_rank: int = Field(..., description="Global popularity rank")


class RecsysNeighborsResponse(BaseModel):
    """Nearest neighbors response payload."""
    query_app_id: int = Field(..., description="Target query Steam App ID")
    neighbors: list[RecsysNeighborItemSchema] = Field(default_factory=list, description="Nearest neighbor games")
    total: int = Field(..., description="Number of neighbors returned")


class RecsysHealthResponse(BaseModel):
    """Recommender service health and readiness status."""
    status: str = Field(..., description="'ready' or 'not_ready'")
    is_ready: bool = Field(..., description="Readiness boolean")
    universe_size: int = Field(..., description="Number of loaded catalog games")
    loaded_models: list[str] = Field(default_factory=list, description="List of loaded deployable model keys")
