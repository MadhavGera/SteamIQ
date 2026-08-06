"""
Settings — single source of truth for all configuration.

Uses pydantic-settings to validate and coerce env vars.
Import `settings` everywhere instead of calling os.getenv() directly.

ADR 0001, Decision 7: NEXT_PUBLIC_API_URL is a frontend concern — not here.
ADR 0001, Decision 4: CORS_ALLOWED_ORIGINS is always an explicit list, never "*".
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import AnyUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = "development"
    app_version: str = "0.1.0"

    # ── Database ──────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "steamiq"
    postgres_user: str = "steamiq"
    postgres_password: str = "changeme"

    # Optional full URL override (takes precedence)
    database_url: str | None = None

    @property
    def async_database_url(self) -> str:
        if self.database_url:
            url = self.database_url
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── CORS (ADR 0001, Decision 4 — never "*") ───────────────────────────────
    cors_allowed_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS_ALLOWED_ORIGINS into a list."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # ── Steam API ─────────────────────────────────────────────────────────────
    steam_api_key: str = ""

    # ── Redis (Phase 2+ job queue) ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — call this everywhere."""
    return Settings()


# Convenience alias used in most modules
settings = get_settings()
