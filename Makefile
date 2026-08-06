.PHONY: setup migrate migrate-create ingest seed dev health test lint build-backend build-frontend

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
	cd $(BACKEND) && $(PIP) install -r requirements.txt
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
	@echo ">>> Seeding known games..."
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid 1145360  # Hollow Knight
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid 1091500  # Cyberpunk 2077
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid 292030   # The Witcher 3
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid 413150   # Stardew Valley
	cd $(BACKEND) && $(PYTHON) -m jobs.ingest_games --appid 570      # Dota 2
	@echo "✅  Seed complete."

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
