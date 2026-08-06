"""
Base job class — shared infrastructure for all ingestion and pipeline jobs.

Provides:
  - Structured logging with job name + context
  - Dry-run mode (parse + validate without writing to DB)
  - Error wrapping that logs before re-raising
  - Absolute path resolution (ADR 0001, Decision 8 — never os.getcwd())
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any


class BaseJob:
    """
    All jobs inherit from this class.

    Subclasses must implement `run()`.
    Invoke via `job.execute()` which wraps run() with timing + error handling.
    """

    #: Override in subclass with a descriptive name
    job_name: str = "base_job"

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.logger = logging.getLogger(f"steamiq.jobs.{self.job_name}")
        self._setup_logging()

    def _setup_logging(self) -> None:
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
                stream=sys.stdout,
            )

    def resolve_path(self, *parts: str) -> Path:
        """
        Resolve a path relative to this file's directory — never os.getcwd().
        ADR 0001, Decision 8.
        """
        return Path(__file__).resolve().parent.joinpath(*parts)

    def log_start(self, **context: Any) -> None:
        ctx = "  ".join(f"{k}={v!r}" for k, v in context.items())
        self.logger.info("▶ Starting %s  %s%s", self.job_name, ctx, "  [DRY RUN]" if self.dry_run else "")

    def log_finish(self, elapsed_s: float, **stats: Any) -> None:
        stats_str = "  ".join(f"{k}={v}" for k, v in stats.items())
        self.logger.info("✅ Finished %s in %.2fs  %s", self.job_name, elapsed_s, stats_str)

    def log_skip(self, reason: str) -> None:
        self.logger.info("⏭ Skipped %s: %s", self.job_name, reason)

    def run(self) -> dict[str, Any]:
        """Override in subclass. Return a stats dict."""
        raise NotImplementedError

    def execute(self, **context: Any) -> dict[str, Any]:
        """
        Wraps run() with timing, logging, and error handling.
        Use this as the entry point rather than calling run() directly.
        """
        self.log_start(**context)
        start = time.perf_counter()
        try:
            result = self.run()
            elapsed = time.perf_counter() - start
            self.log_finish(elapsed, **(result or {}))
            return result or {}
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self.logger.exception(
                "❌ %s failed after %.2fs: %s", self.job_name, elapsed, exc
            )
            raise
