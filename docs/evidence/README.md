# SteamIQ Evidence Logs

This directory stores reproducible evidence logs from verification benchmarks, serving parity checks, explainability audits, and test suites.

## File Manifest

| File | Description |
|---|---|
| **`cold_latency_benchmark.log`** | Cold-cache serving latency profiling (`explain=False`) over 100 distinct user histories with candidate caching (`_SCORE_CAND_CACHE`) explicitly cleared before each request. Compares the old ranker (`recsys_two_tower_ranker_clean`), ranker v2 (`recsys_two_tower_ranker_clean_v2`), and trainstats ranker (`recsys_two_tower_ranker_v2_trainstats`). |
| **`serve_check.log`** | Comprehensive serving check output (see interpretation notes below). |
| **`explain_3_users.log`** | TreeSHAP feature attribution audit on 3 test user histories, verifying SHAP value computation, anchor similarity ($\ge 0.5$), and ensuring review/popularity leakage concepts are excluded from natural language reasons. |
| **`pytest.log`** | Full backend test suite execution record (169 passing tests, 0 failures) validating recommender models, API routes, data pipelines, and anti-leakage invariants. |
| **`pytest_full.log`** | Comprehensive test suite execution record (202 passing tests, 0 failures) including Phase 6 Ask SteamIQ and Phase 7 MLOps/Drift/Golden offline test suites. |
| **`assistant_train.log`** | Ask SteamIQ Phase 6 training & evaluation record: 4 router variants, temperature calibration ($T=0.210$, $ECE=0.0400$), selective abstention, dense extractive retrieval fallback vs TF-IDF, and entity resolution evaluation. |
| **`assistant_serve_check.log`** | Ask SteamIQ Phase 6 online serving verification: 100% serving parity across 200 gold queries (max diff $\le 10^{-4}$), cold latency benchmark ($P_{95}=18.74\text{ ms} < 300\text{ ms}$ SLA budget), and mode parity check across all 13 intents. |
| **`drift_detector_power.log`** | Phase 7 statistical drift detector validation: power curves and false-positive rates (FPR on null resamples) for PSI, two-sample Kolmogorov-Smirnov, and Jensen-Shannon distance under mean, variance, and categorical shifts. |
| **`retrain_gate.log`** | Phase 7 champion/challenger retraining gate evaluation: dry-run analysis and decision logic under pre-registered non-inferiority and margin criteria. |
| **`monitor_models.log`** | Phase 7 telemetry monitoring job output: evaluates live serving buffers against reference distributions with graceful handling of low-traffic cold states. |
| **`generate_model_cards.log`** | Standardized markdown model card generator execution for production and candidate model families. |

## `serve_check.log` Interpretation Notes

**Part A (500-user score parity)** tests only the five single-stage models (Popularity, Item-kNN, ALS, Content, Two-Tower Clean). A float-array identity check is not applicable to the multi-stage ranker pipeline, which involves candidate generation, feature extraction, and LightGBM inference. The ranker is instead verified by:

- **Part A.2 (3-user top-10 parity):** Confirms `service.recommend(ranker) == ranker.score()` (identical top-10 order) and that both differ from the two-tower order, proving active re-ranking occurs.
- **Part B (5,000-user statistical evaluation):** Compares the ranker's anonymous serving path against fresh offline evaluation on both val and test splits with paired bootstrap CIs.

**Part B log label:** The `evaluate_model` logger prints the `RankerModel` class-level `name` attribute (`"recsys_two_tower_ranker"`), not the registry family name. The actual model under test is identified by the artifact resolution header at the top of the log, which shows `recsys_two_tower_ranker_v2_trainstats` with run ID `ee6d93db-6e51-4548-918f-d5133b214990`.

> [!NOTE]
> **Cache-Warm vs. Cold Latency Distinction:**
> In `serve_check.log` **Part C**, the `explain=False` ranker rows represent **cache-warm** inference (candidate pools reused across evaluation passes). For production cold-request latency where candidate generation must run end-to-end without cache hits, refer to **`cold_latency_benchmark.log`**.
