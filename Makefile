.PHONY: setup migrate migrate-create ingest seed dev health test lint build-backend build-frontend build-features train-model train

# ─── Config ───────────────────────────────────────────────────────────────────
PYTHON    := python
PIP       := pip
UVICORN   := uvicorn
ALEMBIC   := alembic
BACKEND   := backend
FRONTEND  := frontend

# Load .env if it exists (for APPID overrides etc.)
-include .env
export

# ─── Setup ────────────────────────────────────────────────────────────────────
setup: ## Install all dependencies and run DB migrations
	@echo ">>> Installing backend dependencies..."
	cd $(BACKEND) && $(PIP) install -e ".[dev]"
	@echo ">>> Running Alembic migrations..."
	cd $(BACKEND) && $(ALEMBIC) upgrade head
	@echo ">>> Installing frontend dependencies..."
	cd $(FRONTEND) && npm install
	@echo "✅  Setup complete. Copy .env.example -> .env and fill in your values."

# ─── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run pending Alembic migrations
	cd $(BACKEND) && $(ALEMBIC) upgrade head

migrate-create: ## Create new Alembic migration: make migrate-create MSG="add foo table"
	cd $(BACKEND) && $(ALEMBIC) revision --autogenerate -m "$(MSG)"

# ─── Data ingestion ───────────────────────────────────────────────────────────
ingest: ## Ingest a single game: make ingest APPID=1145360
	@if [ -z "$(APPID)" ]; then echo "❌  Usage: make ingest APPID=<steam_app_id>"; exit 1; fi
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid $(APPID)

seed: ## Ingest a curated set of known games for local dev
	@echo ">>> Seeding benchmark catalog (fast mode: 10 games)..."
	cd $(BACKEND) && $(PYTHON) -m jobs.seed_catalog --limit 10
	@echo "✅  Seed complete."

seed-catalog: ## Ingest full 30-game benchmark catalog for Phase 3 embeddings
	@echo ">>> Seeding full benchmark catalog (30 games)..."
	cd $(BACKEND) && $(PYTHON) -m jobs.seed_catalog --all
	@echo "✅  Full catalog seed complete."

process-reviews: ## Process NLP reviews for a game: make process-reviews APPID=1145360
	@if [ -z "$(APPID)" ]; then echo "❌  Usage: make process-reviews APPID=<steam_app_id>"; exit 1; fi
	cd $(BACKEND) && $(PYTHON) -m jobs.process_reviews --app-id $(APPID)

process-reviews-all: ## Process NLP reviews for all ingested games
	cd $(BACKEND) && $(PYTHON) -m jobs.process_reviews --all

ingest-news: ## Ingest Steam news and patch notes: make ingest-news
	@echo ">>> Ingesting Steam news and patch notes..."
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_news --all
	@echo "✅  Patch notes ingestion complete."

process-embeddings: ## Generate pgvector embeddings for a single game: make process-embeddings APPID=367520
	@if [ -z "$(APPID)" ]; then echo "❌  Usage: make process-embeddings APPID=<steam_app_id>"; exit 1; fi
	cd $(BACKEND) && $(PYTHON) -m jobs.process_embeddings --app-id $(APPID)

process-embeddings-all: ## Generate pgvector embeddings and competitor rankings for all games
	@echo ">>> Processing pgvector embeddings for all games..."
	cd $(BACKEND) && $(PYTHON) -m jobs.process_embeddings --all
	@echo "✅  Competitor embeddings complete."

# ─── Predictive ML (Phase 4) ──────────────────────────────────────────────────
build-features: ## Build tabular game and market features: make build-features
	@echo ">>> Building tabular ML features..."
	cd $(BACKEND) && $(PYTHON) -m jobs.build_features --all
	@echo "✅  Feature build complete."

train-model: ## Train predictive models & promote champion: make train-model TRIALS=15
	@echo ">>> Training baseline models and tuning ensemble..."
	cd $(BACKEND) && $(PYTHON) -m jobs.train_success_model --trials $(or $(TRIALS),15)
	@echo "✅  Model training and promotion complete."

train: build-features train-model ## Full Phase 4 pipeline: build features and train models

# ─── Decision Intelligence (Phase 5) ──────────────────────────────────────────
materialize-marts: ## Materialize decision marts (overview, trends, opportunity): make materialize-marts
	@echo ">>> Materializing Phase 5 decision marts..."
	cd $(BACKEND) && $(PYTHON) -m jobs.materialize_marts --all
	@echo "✅  Decision marts materialized."

materialize-updates: ## Materialize update impact tracker: make materialize-updates
	@echo ">>> Materializing Update Impact Tracker windows..."
	cd $(BACKEND) && $(PYTHON) -m jobs.materialize_update_impact --all
	@echo "✅  Update impact materialized."

generate-recommendations: ## Generate hybrid recommendations: make generate-recommendations
	@echo ">>> Generating hybrid recommendations (SHAP + domain rules)..."
	cd $(BACKEND) && $(PYTHON) -m jobs.generate_recommendations --all
	@echo "✅  Recommendations generation complete."

decision-pipeline: materialize-marts materialize-updates generate-recommendations ## Full Phase 5 pipeline


# ─── Development ──────────────────────────────────────────────────────────────
dev: ## Start backend (uvicorn --reload) and frontend (next dev) locally
	@echo ">>> Starting backend..."
	cd $(BACKEND) && $(UVICORN) main:app --reload --host 0.0.0.0 --port 8000 &
	@echo ">>> Starting frontend..."
	cd $(FRONTEND) && npm run dev

health: ## Check backend health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

# ─── Testing & Quality ────────────────────────────────────────────────────────
test: ## Run full pytest suite
	cd $(BACKEND) && pytest tests/ -v --tb=short

lint: ## Run ruff linter
	cd $(BACKEND) && ruff check . --fix

# ─── Docker ───────────────────────────────────────────────────────────────────
build-backend: ## Build backend Docker image
	docker build -t steamiq-backend ./$(BACKEND)

build-frontend: ## Build frontend Docker image
	docker build -t steamiq-frontend ./$(FRONTEND)

up: ## Start full stack via Docker Compose
	docker compose up --build

down: ## Stop Docker Compose stack
	docker compose down

# ─── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
