"""
Bounded in-memory aggregate telemetry and Prometheus metrics — Phase 7.

Privacy Invariant:
  - Stores ONLY aggregate counts, histograms, and latency buckets.
  - NEVER persists, logs, or buffers user liked-app lists or query strings.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger("steamiq.mlops.telemetry")


class TelemetryRegistry:
    """
    Thread-safe, bounded in-memory telemetry collector.
    Exposes metrics in Prometheus text exposition format.
    """

    _instance: TelemetryRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Request Counters
        self.recsys_requests: dict[str, int] = defaultdict(int)
        self.assistant_requests: dict[str, int] = defaultdict(int)

        # Cache Counters
        self.recsys_cache_hits: int = 0
        self.recsys_cache_misses: int = 0
        self.assistant_cache_hits: int = 0
        self.assistant_cache_misses: int = 0

        # Latency Buckets (ms): [10, 25, 50, 100, 250, 500, 1000, 2500, inf]
        self.latency_buckets = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]
        self.recsys_latency_counts: dict[float, int] = defaultdict(int)
        self.assistant_latency_counts: dict[float, int] = defaultdict(int)

        # Bounded Aggregate Samples for Drift Monitoring (maximum 2000 points retained in-memory)
        self.max_sample_buffer = 2000
        self.live_library_lengths: list[int] = []
        self.live_recommendation_scores: list[float] = []
        self.live_assistant_confidences: list[float] = []
        self.live_assistant_intents: dict[str, int] = defaultdict(int)

        self.start_time = time.time()

    @classmethod
    def get_instance(cls) -> TelemetryRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_recsys_request(
        self,
        model_name: str,
        took_ms: float,
        library_len: int = 0,
        scores: list[float] | None = None,
        cached: bool = False,
    ) -> None:
        with self.lock:
            self.recsys_requests[model_name] += 1
            if cached:
                self.recsys_cache_hits += 1
            else:
                self.recsys_cache_misses += 1

            for b in self.latency_buckets:
                if took_ms <= b:
                    self.recsys_latency_counts[b] += 1

            if library_len > 0:
                if len(self.live_library_lengths) >= self.max_sample_buffer:
                    self.live_library_lengths.pop(0)
                self.live_library_lengths.append(library_len)

            if scores:
                for s in scores[:5]:
                    if len(self.live_recommendation_scores) >= self.max_sample_buffer:
                        self.live_recommendation_scores.pop(0)
                    self.live_recommendation_scores.append(float(s))

    def record_assistant_request(
        self,
        intent: str,
        confidence: float,
        took_ms: float,
        cached: bool = False,
    ) -> None:
        with self.lock:
            self.assistant_requests[intent] += 1
            self.live_assistant_intents[intent] += 1
            if cached:
                self.assistant_cache_hits += 1
            else:
                self.assistant_cache_misses += 1

            for b in self.latency_buckets:
                if took_ms <= b:
                    self.assistant_latency_counts[b] += 1

            if len(self.live_assistant_confidences) >= self.max_sample_buffer:
                self.live_assistant_confidences.pop(0)
            self.live_assistant_confidences.append(float(confidence))

    def get_live_drift_samples(self) -> dict[str, Any]:
        """Returns snapshot of live telemetry aggregate distributions for drift monitoring."""
        with self.lock:
            return {
                "library_lengths": list(self.live_library_lengths),
                "recommendation_scores": list(self.live_recommendation_scores),
                "assistant_confidences": list(self.live_assistant_confidences),
                "assistant_intent_counts": dict(self.live_assistant_intents),
                "total_recsys_requests": sum(self.recsys_requests.values()),
                "total_assistant_requests": sum(self.assistant_requests.values()),
            }

    def render_prometheus_text(self) -> str:
        """Renders Prometheus standard exposition text."""
        with self.lock:
            lines = [
                "# HELP steamiq_uptime_seconds Total uptime of the service in seconds",
                "# TYPE steamiq_uptime_seconds gauge",
                f"steamiq_uptime_seconds {round(time.time() - self.start_time, 2)}",
                "",
                "# HELP steamiq_recsys_requests_total Total number of recsys recommendations served",
                "# TYPE steamiq_recsys_requests_total counter",
            ]
            for m_name, count in self.recsys_requests.items():
                lines.append(f'steamiq_recsys_requests_total{{model="{m_name}"}} {count}')

            lines.extend([
                "",
                "# HELP steamiq_assistant_requests_total Total number of assistant queries answered",
                "# TYPE steamiq_assistant_requests_total counter",
            ])
            for i_name, count in self.assistant_requests.items():
                lines.append(f'steamiq_assistant_requests_total{{intent="{i_name}"}} {count}')

            lines.extend([
                "",
                "# HELP steamiq_cache_hits_total Total cache hits across services",
                "# TYPE steamiq_cache_hits_total counter",
                f'steamiq_cache_hits_total{{service="recsys"}} {self.recsys_cache_hits}',
                f'steamiq_cache_hits_total{{service="assistant"}} {self.assistant_cache_hits}',
                "",
                "# HELP steamiq_cache_misses_total Total cache misses across services",
                "# TYPE steamiq_cache_misses_total counter",
                f'steamiq_cache_misses_total{{service="recsys"}} {self.recsys_cache_misses}',
                f'steamiq_cache_misses_total{{service="assistant"}} {self.assistant_cache_misses}',
            ])

            # Recsys Latency Histogram
            lines.extend([
                "",
                "# HELP steamiq_recsys_latency_ms Recommender latency distribution in milliseconds",
                "# TYPE steamiq_recsys_latency_ms histogram",
            ])
            for b in self.latency_buckets:
                c = self.recsys_latency_counts[b]
                lines.append(f'steamiq_recsys_latency_ms_bucket{{le="{b}"}} {c}')
            lines.append(f'steamiq_recsys_latency_ms_bucket{{le="+Inf"}} {sum(self.recsys_requests.values())}')

            # Assistant Latency Histogram
            lines.extend([
                "",
                "# HELP steamiq_assistant_latency_ms Assistant latency distribution in milliseconds",
                "# TYPE steamiq_assistant_latency_ms histogram",
            ])
            for b in self.latency_buckets:
                c = self.assistant_latency_counts[b]
                lines.append(f'steamiq_assistant_latency_ms_bucket{{le="{b}"}} {c}')
            lines.append(f'steamiq_assistant_latency_ms_bucket{{le="+Inf"}} {sum(self.assistant_requests.values())}')

            lines.append("")
            return "\n".join(lines)
