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


class AssistantProvenanceSchema(BaseModel):
    model_name: str
    model_run_id: str | None = None
    data_tables_read: list[str] = Field(default_factory=list)
    materialized_at: str | None = None
    took_ms: float = 0.0


class AssistantAskResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    abstained: bool
    tau: float
    user_mode: str
    entities: list[AssistantEntitySchema] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_passages: list[AssistantPassageSchema] = Field(default_factory=list)
    provenance: AssistantProvenanceSchema


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
