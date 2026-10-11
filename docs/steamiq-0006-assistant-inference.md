# ADR 0006 — Online Assistant Inference & Golden Rule Exception

**Status:** Accepted  
**Date:** 2026-10-12  
**Deciders:** Project lead, Machine Learning Engineering team  
**Applies To:** `backend/api/assistant.py`, `backend/assistant/service.py`, `backend/core/app.py`  
**Related ADRs:** [ADR 0001 (Architecture Blueprint)](./design-docs/steamiq/0001-architecture-blueprint.md), [ADR 0004 (Honest Fallbacks)](./steamiq-0004-honest-fallbacks-convention.md), [ADR 0005 (Online Recommender Inference)](./steamiq-0005-online-recommender-inference.md)

---

## 1. Context

SteamIQ delivers "Ask SteamIQ" (Roadmap Phase 6), a learned, evaluated, mode-aware game intelligence assistant. Users can submit arbitrary natural language queries (up to 300 characters) in either Player or Developer mode to receive instant answers about game overviews, review sentiment, top complaints, pricing advice, match profiles, competitor positioning, or personalized recommendations.

Under **ADR 0001 Decision 2 (The Golden Rule)**:
> *"No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only."*

While offline-materialized mart tables contain the factual answers and review summaries, classifying natural language user intent and performing extractive passage retrieval over arbitrary queries cannot be precomputed offline.

---

## 2. Decision

We establish **the second authorized architectural exception** to ADR 0001 Decision 2 (following ADR 0005):
An in-memory, thread-safe `AssistantService` is authorized to execute fixed-size, bounded sentence embedding (via pre-cached `all-MiniLM-L6-v2` on CPU), linear intent classification scoring, calibrated selective abstention, and extractive dot-product passage retrieval at request time.

Deterministic templates then format data sourced exclusively from `mart_*` and `serving_*` tables or in-memory snapshots. No generative LLM free-text generation is permitted.

Every other endpoint in the API continues to read strictly from `serving_*` and `mart_*` tables with zero request-time computation.

---

## 3. Strict Constraints & Invariants

The `AssistantService` exception is bounded by eight strict architectural guardrails:

1. **No Training at Request Time:** Handlers and services never call `.fit()`, calibrate parameters, or update embeddings at runtime.
2. **Zero Database Writes:** Inference is purely stateless and in-memory; no row insertions, state persistence, or session mutations occur during assistant requests.
3. **Zero Raw or Feature Table Reads:** The API handler and inference service never scan `raw_*` or `feature_*` tables. The service loads metadata from `mart_game_overview`, `mart_review_intelligence`, and static model artifacts persisted on disk.
4. **Strict Computational Budget:** Natural language query length is capped at 300 characters. Single-query MiniLM CPU embedding + linear classifier scoring + dot-product passage retrieval operates in $< 50\text{ ms}$, comfortably within the $P_{95} < 300\text{ ms}$ budget.
5. **Single-Load Immutability:** Pre-trained model weights (MiniLM embedder, logistic weights, temperature scaling parameter, passage index) are loaded lazily once into memory.
6. **Complete Provenance & Honest Metadata:** Every response payload includes `model_name`, `model_run_id`, `data_tables_read`, `confidence`, `materialized_at`, and processing latency in `took_ms`.
7. **Honest Readiness (HTTP 503 `ASSISTANT_NOT_READY` / `MODEL_NOT_READY`):** If requested assistant artifacts are missing or uninitialized, the service returns an explicit HTTP 503 with machine-readable error envelope.
8. **User Privacy Invariant:** User queries and library contents are evaluated ephemerally and are NEVER stored, persisted, or logged to disk or database beyond bounded aggregate telemetry counters (request counts, latency histograms, intent distributions).

---

## 4. Performance Budget

- **Throughput & Latency Target:** $P_{95} < 300\text{ ms}$ CPU inference latency for cold and warm requests.
- **Query Embedding:** SentenceTransformers MiniLM encoding takes $< 30\text{ ms}$ on CPU for queries $\le 300$ chars.
- **Linear Scoring & Calibration:** Logistic classification and temperature scaling take $< 2\text{ ms}$ on CPU.
- **Extractive Retrieval:** Dot-product matrix multiplication over candidate passages takes $< 5\text{ ms}$ on CPU.

---

## 5. Alternatives Considered & Rejected

### Alternative A: Generative LLM API (e.g. OpenAI / Claude / Gemini API)
- **Reason Rejected:** Violates predictability, determinism, evaluation rigor, latency budgets, and cost predictability. Hallucinates non-existent features or stats. SteamIQ answers must strictly originate from verified data.

### Alternative B: Keyword-Only Hardcoded Rules
- **Reason Rejected:** Extremely brittle to typos, synonyms, conversational queries, and user phrasing variations. Empirical evaluation demonstrates learned semantic routers significantly outperform dictionary keyword matching.

---

## 6. Consequences

- **Positive:** Robust natural language interface with deterministic, audited domain answers.
- **Positive:** Rigorous offline evaluation, calibration, and selective abstention.
- **Negative:** Holds additional sentence-transformer weights and passage vectors in FastAPI memory (~90 MB RAM).
