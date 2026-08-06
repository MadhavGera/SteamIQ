# SteamIQ

**Game intelligence platform.** Ingests Steam data, runs NLP and ML pipelines offline, and serves precomputed analytics — sentiment, topic clusters, competitor similarity, success prediction, and an AI assistant — all served from pre-materialized tables, never computed per request.

---

## Architecture rule (pinned)

> **No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only. If the data isn't there yet, add or fix a job — never add a bigger query to the handler.**

See [`core/README.md`](backend/core/README.md) and [`docs/design-docs/steamiq/0001-architecture-blueprint.md`](docs/design-docs/steamiq/0001-architecture-blueprint.md).

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| Docker + Docker Compose | v2+ |
| PostgreSQL (local, optional) | 16 |

You need a [Steam Web API key](https://steamcommunity.com/dev/apikey).

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd SteamIQ

# 2. Copy env template and fill in your Steam API key + DB password
cp .env.example .env

# 3. Install dependencies and run migrations
make setup

# 4. Ingest a game (Hollow Knight = 1145360)
make ingest APPID=1145360

# 5. Start everything
docker compose up
```

Open `http://localhost:3000` — search for "Hollow Knight".

---

## Makefile CLI

| Command | What it does |
|---|---|
| `make setup` | Create venv, install deps, run Alembic migrations |
| `make ingest APPID=<id>` | Ingest a single game + its reviews from Steam/SteamSpy |
| `make seed` | Ingest a default set of well-known games for local dev |
| `make health` | Hit the backend `/health` endpoint |
| `make dev` | Start backend (uvicorn reload) + frontend (next dev) locally |
| `make test` | Run full pytest suite |
| `make lint` | Run ruff linter |
| `make migrate` | Run pending Alembic migrations |
| `make migrate-create MSG="description"` | Create a new Alembic migration |

---

## Project Structure

```
SteamIQ/
├── frontend/          # Next.js 14 App Router
├── backend/
│   ├── api/           # Route handlers (read serving_*/mart_* only)
│   ├── core/          # App factory, ApiResponse envelope, CORS, errors, /health
│   ├── db/            # SQLAlchemy models + Alembic migrations
│   ├── jobs/          # Ingestion and pipeline jobs (write to raw_*/feature_*)
│   └── tests/
├── ml/                # Training scripts (Phase 4+)
├── pipelines/         # NLP pipelines (Phase 2+)
├── docs/
│   └── design-docs/steamiq/   # ADRs
├── .env.example
├── docker-compose.yml
└── Makefile
```

---

## Data Zone Convention

| Zone | Prefix | Written by | Read by |
|---|---|---|---|
| Raw ingestion | `raw_*` | Ingestion jobs | Feature jobs (Phase 1: API stopgap — see TODO) |
| Feature engineering | `feature_*` | NLP/feature pipeline jobs | Training, serving |
| Model artifacts | `model_*` | Training jobs | Evaluation, promotion |
| Serving layer | `serving_*` | Pipeline jobs | API handlers ✅ |
| Mart (dashboards) | `mart_*` | Materialization jobs | API handlers ✅ |

---

## Build Phases

| Phase | Goal | Status |
|---|---|---|
| 0 — Architecture & ADR | Lock decisions, repo skeleton | ✅ |
| 1 — Foundation | Search game → stored data → rendered end-to-end | ✅ |
| 2 — NLP Core | Reviews → sentiment, topics, complaints, summary | 🔲 |
| 3 — Similarity + Competitors | Vector similarity, competitor discovery | 🔲 |
| 4 — Predictive ML | XGBoost+LightGBM ensemble, MLflow, SHAP | 🔲 |
| 5 — Decision Intelligence | mart_* tables, recommendations, opportunity finder | 🔲 |
| 6 — AI Assistant | Deterministic router + RAG fallback | 🔲 |
| 7 — Productionization | CI/CD, caching, golden-output tests, Docker | 🔲 |
