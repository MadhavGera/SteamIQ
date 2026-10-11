"""
Redis Response Cache with Graceful Degradation — Phase 7.

Invariants:
  - Key incorporates model_run_id ensuring immediate invalidation upon model promotion.
  - Byte-for-byte fidelity with uncached responses.
  - Graceful degradation: logs connection failure once and falls back to live uncached serving.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from core.config import settings

logger = logging.getLogger("steamiq.core.cache")


class ResponseCache:
    """
    Async Redis response cache.
    Degrades gracefully to no-op when Redis is unreachable.
    """

    _instance: ResponseCache | None = None

    def __init__(self, redis_url: str | None = None, default_ttl: int = 3600) -> None:
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl
        self._redis_client = None
        self._warned_unreachable: bool = False

    @classmethod
    def get_instance(cls) -> ResponseCache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def client(self):
        if self._redis_client is None:
            try:
                import redis.asyncio as aioredis
                self._redis_client = aioredis.from_url(
                    self.redis_url,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    decode_responses=True,
                )
            except Exception as e:
                if not self._warned_unreachable:
                    logger.warning("Redis client initialization failed: %s. Cache disabled.", e)
                    self._warned_unreachable = True
                self._redis_client = None
        return self._redis_client

    @staticmethod
    def make_recsys_key(
        model_run_id: str | None,
        liked_app_ids: list[int],
        k: int = 20,
        explain: bool = False,
    ) -> str:
        """Constructs deterministic hash key for recommender request."""
        sorted_ids = sorted(int(x) for x in liked_app_ids)
        key_raw = f"{model_run_id or 'none'}:{','.join(map(str, sorted_ids))}:{k}:{int(explain)}"
        digest = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
        return f"recsys:{digest}"

    @staticmethod
    def make_assistant_key(
        model_run_id: str | None,
        query: str,
        user_mode: str = "player",
    ) -> str:
        """Constructs deterministic hash key for assistant request."""
        clean_q = query.strip().lower()
        key_raw = f"{model_run_id or 'none'}:{clean_q}:{user_mode}"
        digest = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
        return f"assistant:{digest}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieves and deserializes JSON response from cache."""
        c = self.client
        if c is None:
            return None
        try:
            val = await c.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as e:
            if not self._warned_unreachable:
                logger.warning("Redis get failed (%s). Degrading to uncached.", e)
                self._warned_unreachable = True
            return None
        return None

    async def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> None:
        """Serializes and persists JSON response to cache."""
        c = self.client
        if c is None:
            return
        try:
            serialized = json.dumps(value)
            await c.set(key, serialized, ex=ttl or self.default_ttl)
        except Exception as e:
            if not self._warned_unreachable:
                logger.warning("Redis set failed (%s). Degrading to uncached.", e)
                self._warned_unreachable = True
