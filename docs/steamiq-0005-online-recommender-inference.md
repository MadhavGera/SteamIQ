# ADR 0005 — Online Recommender Inference & Golden Rule Exception

**Status:** Accepted  
**Date:** 2026-10-07  
**Deciders:** Project lead, Recommender System Architecture team  
**Applies To:** `backend/api/recsys.py`, `backend/recsys/inference.py`, `backend/core/app.py`  
**Related ADRs:** [ADR 0001 (Architecture Blueprint)](./design-docs/steamiq/0001-architecture-blueprint.md), [ADR 0004 (Honest Fallbacks)](./steamiq-0004-honest-fallbacks-convention.md)

---

## 1. Context

SteamIQ is pivoting to a hybrid game recommender system with a transparent, visible ML core. Users can select up to 50 games from their Steam library on the "For You" page, choose a recommender model (Popularity, Content-Only, Item-kNN, ALS, Two-Tower Clean, or Ranker Clean), and receive instant personalized recommendations with explainability signals (SHAP feature attributions and "because you liked" nearest-neighbor anchors).

Under **ADR 0001 Decision 2 (The Golden Rule)**:
> *"No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only."*

While batch personalization tables work for fixed registered users, interactive recommendation based on arbitrary user-provided game selections cannot be fully precomputed offline across all $2^{3000}$ possible library subsets.

---

## 2. Decision

We establish **exactly one documented architectural exception** to ADR 0001 Decision 2:
An in-memory, thread-safe `RecommenderService` is authorized to execute fixed-size, bounded vector scoring and candidate reranking at request time in response to user-submitted library inputs.

Every other endpoint in the API continues to read strictly from `serving_*` and `mart_*` tables with zero request-time computation.

---

## 3. Strict Constraints & Invariants

The `RecommenderService` exception is bounded by seven strict architectural guardrails:

1. **No Training at Request Time:** Handlers and services never call `.fit()`, optimize parameters, or update embeddings at runtime.
2. **Zero Database Writes:** Inference is purely stateless and in-memory; no row insertions, state persistence, or session mutations occur during recommendation requests.
3. **Zero Raw or Feature Table Reads:** The API handler and inference service never scan `raw_*` or `feature_*` tables. The service loads metadata from `serving_rec_items` and static model artifacts persisted on disk.
4. **Strict Computational Budget $O(n_{\text{items}} \times d)$:** History fold-in scoring operates via precomputed 64-dimensional Two-Tower item embeddings ($3000 \times 64$ matrix multiplication) and LightGBM reranking over at most 300 candidates. Input liked libraries are hard-capped at 50 games.
5. **Single-Load Immutability:** Pre-trained model weights are loaded lazily once at startup/first request from the artifact paths registered in `model_runs` for the production champion (or candidate) runs.
6. **Complete Provenance & Honest Metadata:** Every recommendation payload includes `model_name`, `model_run_id`, the model's official validation `ndcg_at_10`, `dataset_version`, `universe_size`, and inference latency in `took_ms`.
7. **Honest Readiness (HTTP 503 `MODEL_NOT_READY`):** If a requested model artifact is missing or un-promoted, the service returns an explicit HTTP 503 with machine-readable code `MODEL_NOT_READY`. No fabricated scores or fake recommendations are ever generated.

---

## 4. Performance Budget

- **Throughput & Latency Target:** $P_{95} < 1.0\text{ s}$ CPU inference latency for cold and warm requests with 5 liked items.
- **Vector Retrieval:** Two-Tower 64-d dot-product fold-in takes $< 5\text{ ms}$ on CPU for 3,000 items.
- **Stage-2 Reranking:** LightGBM LambdaMART evaluation over 300 candidates takes $< 40\text{ ms}$ on CPU.

---

## 5. Alternatives Considered & Rejected

### Alternative A: Precompute Recommendations for Known Users Only
- **Approach:** Generate `serving_user_recommendations` table offline for all users in the dataset.
- **Reason Rejected:** Fails to support arbitrary web visitors, anonymous users, cold-start exploratory sessions, or real-time library updates where a user adds or removes a game interactively.

### Alternative B: Separate External Model Serving Server (e.g., Triton, TorchServe, vLLM)
- **Approach:** Host a dedicated microservice container for model inference.
- **Reason Rejected:** Excessive architectural and operational overhead for a 3,000-item catalog where in-memory vectorized CPU matrix multiplications execute in $< 50\text{ ms}$ within the existing FastAPI worker process.

---

## 6. Consequences

- **Positive:** Interactive, sub-second recommendation experience with full model provenance and explainability.
- **Positive:** Full compatibility with ADR 0001 and ADR 0004 (Honest Fallbacks).
- **Negative:** FastAPI worker processes hold model artifacts in memory (~120 MB RAM total across all models).
