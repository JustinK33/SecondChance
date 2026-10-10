# SecondChance - developer entry points.
# Every target here is a real command. Targets that depend on code from a later
# phase are wired to the command they will run once that code lands.
#
# Backend runs on uv-managed CPython 3.12 in backend/.venv. The system python3 is
# 3.14 with no pip and must never be used, so targets call the venv binaries
# directly instead of going through `uv run` (which would re-resolve the pins).
# Frontend needs Node 22; the default node on PATH is v14 and breaks Next.js, so
# every web target sources nvm first.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ROOT    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BACKEND := $(ROOT)/backend
WEB     := $(ROOT)/frontend
BIN     := $(BACKEND)/.venv/bin
COMPOSE := docker compose -f $(ROOT)/docker-compose.yml

UV   := PATH="$$HOME/.local/bin:$$PATH" uv
API  := cd $(BACKEND) &&
NODE := . "$$HOME/.nvm/nvm.sh" && nvm use 22 >/dev/null && cd $(WEB)
# Export backend/.env into the recipe shell. Used where a command needs a var
# that pydantic-settings would otherwise load for itself.
ENVLOAD := set -a && . $(BACKEND)/.env && set +a

DB_USER := secondchance
DB_NAME := secondchance
DB_TEST := secondchance_test
HAS_WEB := [ -f $(WEB)/package.json ]

.DEFAULT_GOAL := help
.PHONY: help install db-up db-down db-reset db-shell migrate revision seed lint fmt \
        typecheck test test-unit test-int e2e smoke bench types dev-api dev-web \
        build-web verify

help: ## List available targets
	@echo "SecondChance targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup

install: ## Create backend/.venv and install backend + frontend dependencies
	cd $(BACKEND) && $(UV) venv --python 3.12
	cd $(BACKEND) && $(UV) pip install -e ".[dev]"
	@if $(HAS_WEB); then $(NODE) && npm install; \
	else echo "skip: $(WEB)/package.json does not exist yet"; fi

# ---------------------------------------------------------------- database

db-up: ## Start Postgres 16 on localhost:5433 and ensure both databases exist
	$(COMPOSE) up -d --wait
	@if [ "$$($(COMPOSE) exec -T db psql -qtAX -U $(DB_USER) -d $(DB_NAME) \
		-c "SELECT 1 FROM pg_database WHERE datname = '$(DB_TEST)'")" != "1" ]; then \
		$(COMPOSE) exec -T db createdb -U $(DB_USER) -O $(DB_USER) $(DB_TEST); \
	fi
	@echo "databases ready on localhost:5433 -> $(DB_NAME), $(DB_TEST)"

db-down: ## Stop Postgres, keeping the data volume
	$(COMPOSE) down

db-reset: ## Destroy the data volume, recreate both databases, migrate and seed
	$(COMPOSE) down -v
	$(MAKE) db-up
	$(MAKE) migrate
	$(MAKE) seed

db-shell: ## Open a psql shell on the app database
	$(COMPOSE) exec db psql -U $(DB_USER) -d $(DB_NAME)

migrate: ## Apply alembic migrations to head, on both the app and test databases
	$(API) $(BIN)/alembic upgrade head
	$(API) $(ENVLOAD) && DATABASE_URL="$$TEST_DATABASE_URL" $(BIN)/alembic upgrade head

revision: ## Autogenerate a migration. Usage: make revision m="add programs table"
	@test -n "$(m)" || { echo 'usage: make revision m="message"'; exit 2; }
	$(API) $(BIN)/alembic revision --autogenerate -m "$(m)"

seed: ## Load the curated program catalog into the database
	$(API) $(BIN)/python -m scripts.seed

# ---------------------------------------------------------------- quality

lint: ## ruff check + format check (backend), eslint (frontend)
	$(API) $(BIN)/ruff check .
	$(API) $(BIN)/ruff format --check .
	@if $(HAS_WEB); then $(NODE) && npm run lint; \
	else echo "skip: frontend not scaffolded yet"; fi

fmt: ## Autoformat and autofix
	$(API) $(BIN)/ruff format .
	$(API) $(BIN)/ruff check --fix .
	@if $(HAS_WEB); then $(NODE) && npx --yes prettier --write .; \
	else echo "skip: frontend not scaffolded yet"; fi

typecheck: ## mypy --strict (backend), tsc --noEmit (frontend)
	$(API) $(BIN)/mypy app
	@if [ -f $(WEB)/tsconfig.json ]; then $(NODE) && npx --yes tsc --noEmit; \
	else echo "skip: frontend not scaffolded yet"; fi

test: ## Full backend test suite with coverage
	$(MAKE) db-up
	$(API) $(BIN)/pytest --cov=app --cov-report=term-missing

test-unit: ## Unit tests only, no database and no live Gemini
	$(API) $(BIN)/pytest -m "not db and not live"

test-int: ## Integration tests that require Postgres
	$(MAKE) db-up
	$(API) $(BIN)/pytest -m db

e2e: ## Playwright end to end run against a live api and web
	@if $(HAS_WEB); then $(NODE) && npx --yes playwright test; \
	else echo "skip: frontend not scaffolded yet"; fi

smoke: ## Drive the core flow against a running API and assert it responds
	$(API) $(BIN)/python -m scripts.smoke

bench: ## Latency and cost benchmark over the golden scenarios
	$(API) $(BIN)/python -m benchmarks.run

types: ## Regenerate frontend TypeScript types from the backend OpenAPI schema
	$(API) $(BIN)/python -m scripts.export_openapi > $(WEB)/openapi.json
	@if $(HAS_WEB); then \
		$(NODE) && npx --yes openapi-typescript ./openapi.json -o ./src/lib/api-types.ts; \
	else echo "skip: frontend not scaffolded yet"; fi

# ---------------------------------------------------------------- run

dev-api: ## Run the API with autoreload on :8000
	$(API) $(BIN)/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-web: ## Run the Next.js dev server on :3000
	$(NODE) && npm run dev

build-web: ## Production build of the frontend
	$(NODE) && npm run build

verify: lint typecheck test ## Everything CI runs
	@echo "verify: OK"
