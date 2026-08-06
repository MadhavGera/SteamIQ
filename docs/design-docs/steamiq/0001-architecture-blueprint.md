# ADR 0001 — SteamIQ Architecture Blueprint

**Status:** Accepted  
**Date:** 2026-08-06  
**Deciders:** Project lead  
**Supersedes:** N/A (first ADR)

---

## Context

SteamIQ is a game intelligence platform that ingests Steam game data, runs NLP and ML pipelines offline, and exposes pre-materialized analytics through a REST API and web frontend.

The decisions in this document are the ones that are **expensive to retrofit** once downstream code exists. Everything here is decided once; downstream phases point to this ADR instead of re-deciding.

---

## Decision 1 — Data Zone Table Naming Convention

### Decision
All database tables use a **zone prefix** that encodes who writes to them and who is allowed to read from them:

| Zone | Prefix | Written by | Read by |
|---|---|---|---|
| Raw ingestion | `raw_*` | Ingestion jobs only | Feature pipeline jobs; Phase 1 API stopgap (flagged) |
| Feature engineering | `feature_*` | NLP/feature pipeline jobs only | ML training jobs; Phase 2 UI (via API, but only for feature tables) |
| Model artifacts | `model_*` | Training jobs only | Evaluation and promotion scripts |
| Serving layer | `serving_*` | Pipeline/scoring jobs only | **API handlers** ✅ |
| Mart (dashboard) | `mart_*` | Materialization jobs only | **API handlers** ✅ |

### Rationale
Without enforced zone naming, queries end up in handlers (Martech-AI's `customer_profiling/main.py` grew to 93KB doing this). The prefix makes violations immediately visible in code review.

### Consequences
- Every new table must be assigned a zone prefix before the migration is written
- API handlers that read `raw_*` or `feature_*` must be marked `# TODO(PhaseN): replace with serving_*/mart_*` and tracked
- No exceptions to this convention without a new ADR

---

## Decision 2 — API Handler Contract (The Golden Rule)

### Decision
> **No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only. If the data isn't there yet, add or fix a job — never add a bigger query to the handler.**

Phase 1 has one explicitly-allowed stopgap: `api/games.py` reads from `raw_games` until `mart_game_overview` is built in Phase 5. This stopgap is marked with `# TODO(Phase5)` and will be removed when `mart_game_overview` exists.

### Rationale
Every project in the SteamIQ learning-notes library violated this in some form. Enforcing it from Phase 1 prevents the pattern from ever starting.

### Consequences
- NLP inference, model scoring, and aggregations must be done in job workers, not handlers
- A handler that cannot find data in `serving_*`/`mart_*` tables returns a 404 or a "not yet computed" response — it does not compute the answer inline

---

## Decision 3 — Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Backend | FastAPI (async) | Performance, Pydantic-native, OpenAPI generation |
| Database | PostgreSQL 16 | Single store for relational + vector (pgvector, Phase 3) — avoids dual-store maintenance |
| ORM + migrations | SQLAlchemy 2.x + Alembic | Industry standard; async support; migration baseline from day one |
| Vector search | pgvector (Phase 3+) | Keeps vector store inside existing Postgres — avoids Lexflow-AI's dual-vector-store burden |
| Job runner | RQ + Redis | Simple, inspectable, matches `make process-reviews` / `make train` DX pattern |
| ML experiment tracking | MLflow | Wired from first training run (Phase 4), not added retroactively |
| Frontend | Next.js 14 (App Router) | SSR, file-system routing, React ecosystem |
| Containerization | Docker Compose (local) | Single `docker compose up` — Postgres + backend + frontend |
| Linter | ruff | Fast, zero-config, used in CI |
| Testing | pytest + httpx (async) | Standard; golden-output test added in Phase 7 |

---

## Decision 4 — CORS Policy

### Decision
The FastAPI app is created via `create_app()` which configures `CORSMiddleware` with:
- `allow_origins` = value of `CORS_ALLOWED_ORIGINS` env var (comma-separated list)
- **Never** `["*"]`
- Default for local dev: `["http://localhost:3000"]`

### Rationale
`CORS_ALLOWED_ORIGINS = ["*"]` is the most common security mistake in every learning-notes project. Explicitly allowlisting origins from day one costs nothing and prevents a class of bugs entirely.

### Consequences
- `CORS_ALLOWED_ORIGINS` must be set in every deployment environment
- Staging and production values must be explicit origin URLs, not wildcards

---

## Decision 5 — API Response Envelope

### Decision
Every API response (success or error) is wrapped in the `ApiResponse` envelope:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "version": "1.0",
    "took_ms": 12
  }
}
```

Error responses:
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "game_not_found",
    "message": "No game found with app_id 99999",
    "details": null
  },
  "meta": null
}
```

### Rationale
Consistent envelope means frontend code never needs to special-case response shape. Error `code` field (machine-readable slug) enables frontend to branch on specific error types without parsing message strings.

### Consequences
- All route handlers must return `ApiResponse[T]` — raw Pydantic models are not returned directly
- All exceptions must flow through global handlers that produce `ApiResponse(success=False, ...)`

---

## Decision 6 — model_runs Schema

### Decision
Every ML training run writes a row to `model_runs`:

```
model_runs (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name       TEXT NOT NULL,          -- e.g. "success_predictor"
  stage            TEXT NOT NULL,          -- see enum below
  dataset_version  TEXT NOT NULL,          -- e.g. "2026-08-01" cutoff date
  hyperparameters  JSONB,
  metrics          JSONB,                  -- {"auc": 0.87, "f1": 0.82, ...}
  artifact_path    TEXT,                   -- MLflow artifact URI
  promoted_by      TEXT,                   -- script/user that promoted
  promoted_at      TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT now()
)
```

Stage enum (enforced at application level):
```
candidate → staging → production → archived
```

Promotion rules (ported from AutoModelX `registry.py` pattern):
- Only one model per `model_name` may be in `production` at a time
- Promotion from `candidate → staging` requires passing evaluation thresholds
- Promotion from `staging → production` requires explicit `evaluate_champion()` call
- Demoted models move to `archived` (never deleted)

Every row in `serving_predictions` carries `model_run_id` (FK to `model_runs`) — any score is traceable to the exact model version that produced it.

---

## Decision 7 — Frontend URL Convention

### Decision
The backend API base URL is always read from `process.env.NEXT_PUBLIC_API_URL`. This env var is set in `.env.local` (local dev) and in the container environment (Docker/production). It is **never** hardcoded in source files.

### Rationale
MC4-Final's hardcoded `http://localhost:8000` cost a debugging session months after initial development. One line in week one prevented permanently.

### Consequences
- No string literal containing a port number (`:8000`, `:3000`, etc.) may appear in frontend source files
- A central `lib/api.ts` client file reads `NEXT_PUBLIC_API_URL` and exports typed fetch helpers

---

## Decision 8 — Ingestion Paths

### Decision
All ingestion jobs use absolute paths derived from env vars or `pathlib.Path(__file__).parent` — never `os.getcwd()` or relative path strings.

### Rationale
Lexflow-AI's CWD-dependent data paths broke silently when the working directory changed. Absolute paths are always correct regardless of where the process is invoked from.

---

## MVP vs Final vs Stretch Scope Table

| Phase | MVP | Strong Final | Stretch |
|---|---|---|---|
| 0 — Architecture & ADR | ✅ required | | |
| 1 — Foundation | ✅ game search + stored data end-to-end | | |
| 2 — NLP Core | ✅ sentiment, topics, complaints, loved features, summary | | |
| 3 — Similarity + Competitors | ✅ pgvector similarity, competitor UI | | |
| 4 — Predictive ML | ✅ XGBoost+LightGBM ensemble, MLflow, SHAP | Revenue category + activity forecast | |
| 5 — Decision Intelligence | | ✅ Opportunity Finder, Update Impact, Recommendation Engine, mart_* | Pricing simulator refinements |
| 6 — AI Assistant | | ✅ Deterministic router + RAG fallback | Multi-game portfolio queries |
| 7 — Productionization | | ✅ Redis, CI/CD, golden-output tests, Docker final | Cloud deployment (Azure/AWS), automatic retraining, What-If Simulator |

---

## Consequences of This ADR

- Every downstream phase references this document instead of re-deciding these points
- A violation of any decision above requires either a new ADR or a code review block
- The `core/README.md` pins Decision 2 (the golden rule) at the top as a daily reminder
