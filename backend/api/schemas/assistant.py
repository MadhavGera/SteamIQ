"""
Pydantic schemas for Ask SteamIQ API — Phase 6.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssistantAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, description="Natural language query (max 300 chars)")
    user_mode: str = Field(default="player", description="'player' or 'developer'")
    game_app_id: int | None = Field(default=None, description="Optional Steam App ID context")
    liked_app_ids: list[int] | None = Field(default=None, description="Optional list of library App IDs")


class AssistantEntitySchema(BaseModel):
    type: str = Field(default="game")
    app_id: int | None = None
    title: str | None = None
    confidence: float = 1.0
    method: str = "exact"


class AssistantPassageSchema(BaseModel):
    passage_id: int | None = None
    app_id: int | None = None
    game_title: str | None = None
    field: str | None = None
    text: str = ""
    score: float = 0.0


class GameCandidateSchema(BaseModel):
    app_id: int
    title: str
    header_image: str | None = None
    score: float = 0.0
    score_type: str = "similarity"  # "hybrid", "collaborative", "vector_cosine"
    rank: int = 1
    price_usd: str | None = None
    positive_ratio: int | None = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    match_reason: str = ""
    explanation: dict[str, Any] | None = None


class AssistantProvenanceSchema(BaseModel):
    model_name: str
    model_run_id: str | None = None
    data_tables_read: list[str] = Field(default_factory=list)
    materialized_at: str | None = None
    took_ms: float = 0.0


class MarketOverviewMetricsSchema(BaseModel):
    games_analyzed: int = 0
    price_range_usd: str = "--"
    median_price_usd: str = "--"
    sentiment_pct: float | None = None
    review_count: int = 0
    sources: list[str] = Field(default_factory=list)
    scope_notes: str = ""


class MLEvaluationMetricsSchema(BaseModel):
    intent_model_name: str = "assistant_intent_main"
    intent_accuracy: float = 0.964
    intent_macro_f1: float = 0.958
    intent_test_samples: int = 350
    recsys_model_name: str = "recsys_two_tower_clean"
    recsys_ndcg_at_10: float = 0.082
    recsys_hr_at_10: float = 0.146
    recsys_test_users: int = 5000
    architecture_summary: str = (
        "Dual-Stage Architecture: MiniLM-L6-v2 Intent Router + Two-Tower Neural Collaborative Filtering (R^64) "
        "+ Pre-Materialized Analytical Data Marts."
    )


class AssistantAskResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    abstained: bool
    tau: float
    user_mode: str
    entities: list[AssistantEntitySchema] = Field(default_factory=list)
    resolved_game: dict[str, Any] | None = None
    candidates: list[GameCandidateSchema] = Field(default_factory=list)
    market_summary: MarketOverviewMetricsSchema | None = None
    ml_evaluation: MLEvaluationMetricsSchema | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_passages: list[AssistantPassageSchema] = Field(default_factory=list)
    provenance: AssistantProvenanceSchema
    latency_breakdown: dict[str, float] | None = None


class AssistantHealthResponse(BaseModel):
    status: str
    model_name: str
    model_run_id: str | None = None
    tau: float = 0.50
    is_ready: bool = False


class AssistantModelInfoResponse(BaseModel):
    model_name: str
    model_run_id: str | None = None
    stage: str = "candidate"
    tau: float = 0.50
    accuracy: float | None = None
    macro_f1: float | None = None
    ece: float | None = None
    classes: list[str] = Field(default_factory=list)
