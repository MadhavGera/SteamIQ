"""
Pydantic schemas for Player Activity & Telemetry Intelligence (Phase 5).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlaytimeCohortSchema(BaseModel):
    """Playtime retention cohort distribution bucket."""
    cohort: str = Field(..., description="Cohort bracket identifier (e.g. Under 10h)")
    label: str = Field(..., description="Descriptive label (e.g. Casual / Early Drop)")
    hours_range: str = Field(..., description="Human-readable hours range")
    player_count: int = Field(..., description="Number of sampled players in this cohort")
    share_pct: float = Field(..., description="Percentage share of total player sample")


class PlaytimePercentilesSchema(BaseModel):
    """Playtime distribution percentiles in hours."""
    p25_hours: float | None = Field(default=None, description="25th percentile lifetime playtime in hours")
    median_hours: float | None = Field(default=None, description="Median lifetime playtime in hours")
    p75_hours: float | None = Field(default=None, description="75th percentile lifetime playtime in hours")
    p90_hours: float | None = Field(default=None, description="90th percentile lifetime playtime in hours")


class CCUPointSchema(BaseModel):
    """Historical concurrent players (CCU) snapshot point."""
    snapshot_at: str | None = Field(default=None, description="ISO timestamp of snapshot")
    player_count: int = Field(..., description="Concurrent player count at snapshot")
    source: str = Field(default="steam_api", description="Telemetry ingestion source")


class EventImpactSummarySchema(BaseModel):
    """Correlated update/patch event impact summary."""
    patch_name: str = Field(..., description="Name or milestone of the update")
    patch_date: str | None = Field(default=None, description="ISO timestamp of update")
    pre_avg_ccu: int | None = Field(default=None, description="Average CCU before update window")
    post_avg_ccu: int | None = Field(default=None, description="Average CCU after update window")
    ccu_change_pct: float | None = Field(default=None, description="Percentage change in CCU")
    verdict: str | None = Field(default=None, description="Automated impact verdict")


class PlayerActivitySchema(BaseModel):
    """
    Consolidated Player Activity & Telemetry intelligence schema.
    Strictly reads from pre-materialized mart_game_overview.player_activity.
    """
    app_id: int = Field(..., description="Steam App ID")
    game_name: str = Field(..., description="Game title")
    current_ccu: int | None = Field(default=None, description="Latest observed concurrent player count")
    peak_ccu_24h: int | None = Field(default=None, description="24-hour peak concurrent players")
    all_time_peak_ccu: int | None = Field(default=None, description="All-time peak concurrent players observed")
    owners_estimate: str | None = Field(default=None, description="Owner bracket from SteamSpy")
    average_playtime_forever_hours: float | None = Field(default=None, description="Lifetime average playtime in hours")
    median_playtime_forever_hours: float | None = Field(default=None, description="Lifetime median playtime in hours")
    average_playtime_at_review_hours: float | None = Field(default=None, description="Average playtime when review was submitted")
    post_review_engagement_delta_hours: float | None = Field(default=None, description="Additional playtime accumulated after reviewing")
    sample_size_reviews: int = Field(default=0, description="Sample size of verified player reviews analyzed")
    playtime_percentiles: PlaytimePercentilesSchema = Field(default_factory=PlaytimePercentilesSchema, description="Empirical playtime percentiles")
    playtime_cohorts: list[PlaytimeCohortSchema] = Field(default_factory=list, description="Retention and engagement cohort distribution")
    ccu_history: list[CCUPointSchema] = Field(default_factory=list, description="Historical CCU telemetry snapshots")
    recent_event_impacts: list[EventImpactSummarySchema] = Field(default_factory=list, description="Recent update impact events")
    telemetry_mode: str = Field(default="snapshot_polled", description="Telemetry ingestion mode (e.g. snapshot_polled)")
    updated_at: str | None = Field(default=None, description="ISO timestamp when telemetry was materialized")
