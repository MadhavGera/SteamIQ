# SteamIQ Evidence Logs

This directory stores reproducible evidence logs from verification benchmarks, serving parity checks, explainability audits, and test suites.

## File Manifest

| File | Description |
|---|---|
| **`cold_latency_benchmark.log`** | Cold-cache serving latency profiling (`explain=False`) over 100 distinct user histories with candidate caching (`_SCORE_CAND_CACHE`) explicitly cleared before each request. Compares the old ranker (`recsys_two_tower_ranker_clean`), ranker v2 (`recsys_two_tower_ranker_clean_v2`), and trainstats ranker (`recsys_two_tower_ranker_v2_trainstats`). |
| **`serve_check.log`** | Comprehensive serving check output covering **Part A** (500-user online vs. offline score parity within $10^{-5}$), **Part B** (5,000-user full-split anonymous serving path evaluation with paired bootstrap tests), and **Part C** (request latency profiling across models). |
| **`explain_3_users.log`** | TreeSHAP feature attribution audit on 3 test user histories, verifying SHAP value computation, anchor similarity ($\ge 0.5$), and ensuring review/popularity leakage concepts are excluded from natural language reasons. |
| **`pytest.log`** | Full backend test suite execution record (169 passing tests, 0 failures) validating recommender models, API routes, data pipelines, and anti-leakage invariants. |

> [!NOTE]
> **Cache-Warm vs. Cold Latency Distinction:**
> In `serve_check.log` **Part C**, the `explain=False` ranker rows represent **cache-warm** inference (candidate pools reused across evaluation passes). For production cold-request latency where candidate generation must run end-to-end without cache hits, refer to **`cold_latency_benchmark.log`**.
