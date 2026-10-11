# SteamIQ Evidence Logs

This directory stores reproducible evidence logs from verification benchmarks, serving parity checks, explainability audits, and test suites.

## File Manifest

| File | Description |
|---|---|
| **`cold_latency_benchmark.log`** | Cold-cache serving latency profiling (`explain=False`) over 100 distinct user histories with candidate caching (`_SCORE_CAND_CACHE`) explicitly cleared before each request. Compares the old ranker (`recsys_two_tower_ranker_clean`), ranker v2 (`recsys_two_tower_ranker_clean_v2`), and trainstats ranker (`recsys_two_tower_ranker_v2_trainstats`). |
| **`serve_check.log`** | Comprehensive serving check output (see interpretation notes below). |
| **`explain_3_users.log`** | TreeSHAP feature attribution audit on 3 test user histories, verifying SHAP value computation, anchor similarity ($\ge 0.5$), and ensuring review/popularity leakage concepts are excluded from natural language reasons. |
| **`pytest.log`** | Full backend test suite execution record (169 passing tests, 0 failures) validating recommender models, API routes, data pipelines, and anti-leakage invariants. |

## `serve_check.log` Interpretation Notes

**Part A (500-user score parity)** tests only the five single-stage models (Popularity, Item-kNN, ALS, Content, Two-Tower Clean). A float-array identity check is not applicable to the multi-stage ranker pipeline, which involves candidate generation, feature extraction, and LightGBM inference. The ranker is instead verified by:

- **Part A.2 (3-user top-10 parity):** Confirms `service.recommend(ranker) == ranker.score()` (identical top-10 order) and that both differ from the two-tower order, proving active re-ranking occurs.
- **Part B (5,000-user statistical evaluation):** Compares the ranker's anonymous serving path against fresh offline evaluation on both val and test splits with paired bootstrap CIs.

**Part B log label:** The `evaluate_model` logger prints the `RankerModel` class-level `name` attribute (`"recsys_two_tower_ranker"`), not the registry family name. The actual model under test is identified by the artifact resolution header at the top of the log, which shows `recsys_two_tower_ranker_v2_trainstats` with run ID `ee6d93db-6e51-4548-918f-d5133b214990`.

> [!NOTE]
> **Cache-Warm vs. Cold Latency Distinction:**
> In `serve_check.log` **Part C**, the `explain=False` ranker rows represent **cache-warm** inference (candidate pools reused across evaluation passes). For production cold-request latency where candidate generation must run end-to-end without cache hits, refer to **`cold_latency_benchmark.log`**.
