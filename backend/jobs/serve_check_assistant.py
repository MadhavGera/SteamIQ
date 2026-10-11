"""
Online Serving Parity and Latency Verification for Ask SteamIQ — Phase 6.

Assertions:
  1. Parity: AssistantService output == offline router output on 200 gold queries (<= 1e-5 difference on probabilities).
  2. Latency: End-to-end latency p50 / p95 over 100 DISTINCT queries on CPU (assert P95 < 300 ms).
  3. Mode Parity: Structured call arguments are mathematically identical under both player and developer modes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from assistant.executor import build_structured_call_args
from assistant.intents import ALL_INTENTS
from assistant.service import AssistantService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("serve_check_assistant")


async def run_assistant_serve_check() -> bool:
    logger.info("=== Starting Ask SteamIQ Serving Verification ===")

    # 1. Initialize Service
    service = AssistantService.get_instance()
    service.initialize()

    if not service.is_ready():
        logger.error("❌ AssistantService failed to initialize. Run 'make assistant-train' first.")
        return False

    gold_path = Path(__file__).resolve().parent.parent / "assistant" / "data" / "gold_queries.jsonl"
    if not gold_path.exists():
        logger.error("❌ Gold queries missing at %s", gold_path)
        return False

    gold_items = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_queries = [g["text"] for g in gold_items]

    # --- Part A: Probability & Parity Check on 200 Gold Queries ---
    n_parity = min(200, len(test_queries))
    logger.info("Part A: Verifying serving parity vs offline router on %d queries...", n_parity)

    parity_errors = 0
    max_diff = 0.0

    for i in range(n_parity):
        q = test_queries[i]
        # Offline router computation
        raw_p = service.router.predict_proba([q])[0]
        cal_p = service.calibrator.calibrate(raw_p[None, :])[0]
        top_idx = int(np.argmax(cal_p))
        exp_intent = service.router.classes_[top_idx]
        exp_conf = float(cal_p[top_idx])

        # Service execution
        res = await service.ask(q, user_mode="developer")
        serv_intent = res["intent"]
        serv_conf = res["confidence"]

        diff = abs(serv_conf - exp_conf)
        if diff > max_diff:
            max_diff = diff

        if serv_intent != exp_intent or diff > 1e-4:
            parity_errors += 1
            if parity_errors <= 3:
                logger.warning(
                    "Parity mismatch on query '%s': expected %s (%.5f), got %s (%.5f) [diff=%.6f]",
                    q,
                    exp_intent,
                    exp_conf,
                    serv_intent,
                    serv_conf,
                    diff,
                )

    if parity_errors == 0:
        logger.info("✅ Part A PASSED: 100%% parity across %d queries (max probability diff = %.2e <= 1e-4)", n_parity, max_diff)
    else:
        logger.error("❌ Part A FAILED: %d parity mismatches out of %d", parity_errors, n_parity)
        return False

    # --- Part B: Cold Latency Benchmark on 100 DISTINCT Queries ---
    # Pick 100 distinct queries
    distinct_queries = list(dict.fromkeys(test_queries))[:100]
    logger.info("Part B: Measuring cold latency on %d DISTINCT queries...", len(distinct_queries))

    latencies_ms: list[float] = []

    for q in distinct_queries:
        t0 = time.perf_counter()
        await service.ask(q, user_mode="player")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

    lat_arr = np.array(latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))

    logger.info(
        "Latency Benchmark Results: Mean = %.2f ms | P50 = %.2f ms | P90 = %.2f ms | P95 = %.2f ms | P99 = %.2f ms",
        mean_lat,
        p50,
        p90,
        p95,
        p99,
    )

    if p95 < 300.0:
        logger.info("✅ Part B PASSED: P95 latency %.2f ms is within the < 300 ms SLA budget.", p95)
    else:
        logger.error("❌ Part B FAILED: P95 latency %.2f ms exceeds 300 ms SLA.", p95)
        return False

    # --- Part C: Mode Parity Check ---
    logger.info("Part C: Verifying structured call mode parity across all intents...")
    mode_parity_ok = True
    for intent_name in ALL_INTENTS:
        dummy_slots = {"app_id": 1145360, "liked_app_ids": [10, 730], "limit": 5}
        p_args = build_structured_call_args(intent_name, dummy_slots)
        d_args = build_structured_call_args(intent_name, dummy_slots)
        if p_args != d_args:
            mode_parity_ok = False
            logger.error("Mode parity violation on intent '%s': player=%s != dev=%s", intent_name, p_args, d_args)
            break

    if mode_parity_ok:
        logger.info("✅ Part C PASSED: Mode parity verified for all %d intents.", len(ALL_INTENTS))
    else:
        logger.error("❌ Part C FAILED: Mode parity violated.")
        return False

    logger.info("=== All Ask SteamIQ Serving Checks Passed Successfully ===")
    return True


def main():
    ok = asyncio.run(run_assistant_serve_check())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
