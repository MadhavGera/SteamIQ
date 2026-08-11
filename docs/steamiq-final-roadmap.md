# SteamIQ — Final Project Roadmap

This supersedes the phasing in the original spec (§37–38). Same seven build phases, now with Phase 0 added and every phase carrying its production revisions baked in from the start rather than bolted on later. Each phase lists: goal, deliverables, exit criteria (what "done" means before moving on), and the specific anti-pattern from your own past projects to watch for at that stage.

---

## Phase 0 — Architecture & ADR
**Goal:** Lock in the decisions that are expensive to retrofit, before any code is written.

**Deliverables**
- `design-docs/steamiq/0001-architecture-blueprint.md` written using your `ai-mill` ADR template (skeleton already drafted in the prior review — fill it in, don't restart it)
- Data-zone table map finalized: every table in §13 assigned to `raw_*` / `feature_*` / `serving_*` / `mart_*` / `model_*`
- `model_runs` schema defined (id, model_name, stage, dataset_version, hyperparameters, metrics, artifact_path) with the stage enum: `candidate → staging → production → archived`
- Repo skeleton created: `frontend/`, `backend/`, `ml/`, `pipelines/`, `core/` (shared FastAPI app factory + `ApiResponse` envelope + error handlers), `docs/`
- One-page "MVP vs Final vs Stretch" scope table committed to the repo (from §38) so scope creep has a written reference to check against

**Exit criteria:** ADR merged; every downstream phase can point to it instead of re-deciding. No API or job code written yet.

**Watch for:** skipping this because it "feels like paperwork." This is the exact step MartechAI-Deployment_Friendly had and its prototype predecessor (Martech-AI) didn't — it's the difference between a planned migration and an ad hoc one.

---

## Phase 1 — Foundation
**Goal:** Search a game, see reliably stored data, on a schema that won't need renaming later.

**Deliverables**
- PostgreSQL schema created directly with zone-prefixed table names (`raw_games`, `raw_reviews`, `raw_player_snapshots`, `raw_price_history`, ...) via SQLAlchemy + Alembic migrations from the first table
- Steam API + SteamSpy ingestion jobs, writing only to `raw_*`
- `core/` shared library: `create_app()` wiring CORS with an **explicit origin allowlist** (never `["*"]`), the `ApiResponse` envelope, generic error handlers, `/health`
- FastAPI skeleton with `api/games.py` as the only router so far, reading straight from `raw_*` for this phase only (acceptable stopgap — flagged for removal once `mart_game_overview` exists in Phase 5)
- Next.js skeleton, `NEXT_PUBLIC_API_URL` env var wired from the first component (never a hardcoded localhost URL)
- Docker Compose: Postgres + backend + frontend

**Exit criteria:** search a game by name, see stored metadata rendered end to end through Docker Compose.

**Watch for:** Lexflow-AI's CWD-dependent data paths and MC4-Final's hardcoded backend URL. Both are one-line mistakes made in week one that cost a debugging session months later.

---

## Phase 2 — NLP Core
**Goal:** SteamIQ becomes a viable project — reviews turn into structured intelligence.

**Deliverables**
- `feature_review_sentiment`, `feature_review_topics`, `feature_review_complaints`, `feature_review_features` tables, written only by NLP pipeline jobs (never an API handler)
- Sentiment: RoBERTa/DistilBERT via Hugging Face Transformers, batched in a job, not inline
- Topic Discovery: BERTopic pipeline, wrapped in a `BERTOPIC_AVAILABLE` try/except with a TF-IDF+KMeans fallback so a broken install doesn't block the container
- Complaint detection: zero-shot classification bootstrap, multi-label
- Feature appreciation (loved features): embeddings + HDBSCAN, same graceful-degradation wrapper as BERTopic
- Review summarization: hierarchical (representative-review selection → topic grouping → LLM/BART summarization), queued as a job, never synchronous in a request
- Job runner in place: RQ + Redis, invoked via a Makefile target (`make process-reviews`) mirroring Martech-AI's `make train-all` DX pattern
- Review Intelligence UI (§22) reads only from `feature_*` tables

**Exit criteria:** for a seeded game with real review data, sentiment/topics/complaints/loved-features/summary all render from precomputed tables with no live NLP call in the request path.

**Watch for:** letting BERTopic or embedding generation run inside an API handler "just for now" — this is precisely how Martech-AI's `customer_profiling/main.py` grew to 93KB.

---

## Phase 3 — Similarity + Competitors
**Goal:** Competitor discovery, the feature that most differentiates SteamIQ from a generic review analyzer.

**Deliverables**
- `model_game_embeddings` table (sentence-transformer vectors over description + genres/tags + review topics)
- Vector index: pgvector inside the existing Postgres instance (preferred over adding FAISS as a second system unless query volume later proves it's needed — avoids Lexflow-AI's dual-vector-store maintenance burden)
- Similarity API reading from a `serving_similar_games` table populated by a scheduled job, not computed per request
- Competitor Analysis UI (§23) wired to real data

**Exit criteria:** given any seeded game, return its top-N competitors with similarity scores, served from a precomputed table.

---

## Phase 4 — Predictive ML
**Goal:** One rigorously-designed predictive model, done properly — this is the non-negotiable ML core, not a checklist item.

**Deliverables**
- `feature_game_features`, `feature_market_features` tables (price, genre, tags, developer/publisher history, review velocity, competitor density, sentiment — as in §32) built with **enforced temporal leakage boundaries**: a `feature_cutoff_date` parameter that makes it structurally impossible to feed post-cutoff data into training
- Success Prediction: **XGBoost + LightGBM ensemble, tuned with Optuna**, not a single model — this is the default now, not a stretch upgrade
- Baseline comparison table populated for real (Logistic Regression → Random Forest → XGBoost → XGBoost+LightGBM ensemble), matching the format in §18
- MLflow wired from the *first* training run, not added retroactively
- `model_runs` table populated on every training run; `evaluate_champion()`-style promotion logic (ported from AutoModelX's `registry.py`) auto-promotes the best eligible run per metric direction
- Every row written to `serving_predictions` carries `model_run_id`, so any score is traceable to the exact model version that produced it
- SHAP wired in for this model as soon as it's trained, not deferred to Phase 5, since explainability and the model should be developed together
- Revenue Category and Activity/Retention forecasting follow the same ensemble+registry pattern once Success Prediction validates the pipeline

**Exit criteria:** a success score for a seeded game, generated by a promoted `production`-stage model, explainable via SHAP, with the model run traceable in MLflow and `model_runs`.

**Watch for:** treating the ensemble+HPO step or the registry as later polish. Both are the same amount of work now as retrofitting them into a system that already has predictions flowing — cheaper to do once, correctly, here.

---

## Phase 5 — Decision Intelligence
**Goal:** Turn stored intelligence into materialized, dashboard-ready insight — and close the loop from data to recommendation.

**Deliverables**
- `mart_game_overview`, `mart_trends`, `mart_opportunity_scores` populated by a scheduled materialization job — the Game Intelligence Page (§21) and Main Dashboard (§20) are repointed from `raw_*`/`feature_*` (Phase 1 stopgap) to these `mart_*` tables
- Opportunity Finder: weighted-scoring formula as originally specified (§8) — kept as analytics, not upgraded to ML until there's real historical performance data to learn weights from
- Pricing Intelligence: market-based comparable-range output (§5.3), no causal price-elasticity claims
- Update Impact Tracker: before/after windows, explicitly labeled "observed/correlated," same discipline extended to the What-If Simulator's scenario framing (§39) if built
- Recommendation Engine: hybrid rules + model-output system (§10), reading from `serving_predictions`/`feature_*`, writing to `serving_recommendations`
- Recommendation Center UI (§26) built as its own dedicated view, not folded into individual charts

**Exit criteria:** the dashboard and game page load entirely from `mart_*`/`serving_*` tables — zero live aggregation or model inference triggered by a page render.

---

## Phase 6 — AI Assistant
**Goal:** "Ask SteamIQ" — grounded, auditable, and only built once there's real intelligence underneath it to query.

**Deliverables**
- Deterministic intent router first: keyword/pattern rules (compare, why/caused, competitor+sentiment, etc.) mapped directly to structured API/analytics calls — LLM intent classification only as fallback for unmatched queries, per the HR-BOT planner pattern
- `schema_metadata.json` describing `mart_*`/`serving_*` tables in business terms, injected into any prompt that needs structured retrieval — prevents the hallucinated-query problem both HR-BOT and MC4-FORECASTING hit
- RAG path (vector search over review-topic summaries and evidence text) reserved for genuinely unstructured questions; structured questions go straight to APIs, never through embedding + LLM guesswork
- Assistant calls queued through the existing RQ worker, not synchronous in the request handler, since LLM latency is user-facing
- Citations/evidence links (§12) returned alongside every answer

**Exit criteria:** the example query set from §12 ("Why are players complaining," "How does this compare with Hades," "What caused sentiment to fall") all resolve via the deterministic path where a rule applies, with correct evidence citations.

---

## Phase 7 — Productionization
**Goal:** Everything that turns a working prototype into something you'd trust to keep running.

**Deliverables**
- Redis caching finalized across the endpoints in §15
- All jobs (ingestion, feature engineering, NLP, training, materialization) running through RQ with a real scheduler (cron or RQ-scheduler) — no in-process APScheduler thread anywhere, per Lexflow-AI's lesson
- Model registry promotion process fully operational and documented — this closes the exact gap MartechAI-DF flagged as unsolved
- `.gitignore` audit: no committed `.pkl`/`.joblib` artifacts, no committed SQLite/Postgres dumps, no committed API keys — checked against the anti-pattern table before first public push
- CI/CD: GitHub Actions running tests + lint (ruff) + a build check on every PR
- Testing: standard pytest suite **plus** a golden-output behavioral test — run the full pipeline for a fixed seeded game, diff the JSON output against a stored baseline, catch silent regressions in scoring/recommendations the way unit tests alone won't (Martech-AI's `audit_comparison.md` pattern)
- Docker images finalized for backend, worker, and frontend; docker-compose covers full local stack
- README: setup instructions, Makefile CLI (`make seed`, `make ingest`, `make process-reviews`, `make train`, `make health`) mirroring the DX pattern from Martech-AI, and — if cloud deployment is pursued — a migration-mapping doc listing exactly where each local component goes in the cloud target, written *before* touching infra code

**Exit criteria:** fresh clone → `make setup && docker compose up` → working platform, with CI green and the golden-output test passing.

---

## Roadmap-level scope map

| Phase | MVP | Strong Final Version | Stretch |
|---|---|---|---|
| 0 — Architecture & ADR | ✅ required | | |
| 1 — Foundation | ✅ | | |
| 2 — NLP Core | ✅ (sentiment, topics, complaints, loved features, summary) | | |
| 3 — Similarity + Competitors | ✅ | | |
| 4 — Predictive ML | ✅ (one model, ensemble+registry) | Revenue category, activity forecast added | |
| 5 — Decision Intelligence | | ✅ Opportunity Finder, Update Impact, Recommendation Engine, SHAP | Pricing simulator refinements |
| 6 — AI Assistant | | ✅ RAG assistant | |
| 7 — Productionization | | ✅ Redis, MLflow, registry, Docker, CI/CD, golden-output tests | Cloud deployment (Azure/AWS), publisher-portfolio dashboard, automatic retraining, What-If Simulator |

This keeps your original MVP/Final/Stretch split (§38) intact — the only change is that Phase 0's decisions and Phase 4's ensemble+registry work move from "stretch" into the non-negotiable core, since both are cheaper to build correctly the first time than to retrofit once predictions are already flowing through the system.

---

## One rule to keep visible throughout

> No API handler computes an aggregate, runs a model, or scans `raw_*`/`feature_*` tables. It reads from `serving_*` or `mart_*` only. If the data isn't there yet, add or fix a job — never add a bigger query to the handler.

Pin this at the top of `core/README.md` or your AGENTS.md-equivalent. It's the single rule that, if followed from Phase 1 onward, prevents SteamIQ from repeating the one mistake nearly every past project in your learning-notes library made in some form.
