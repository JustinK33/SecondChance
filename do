#!/usr/bin/env bash
# SecondChance - developer entry points.
#
# This file, not the Makefile, holds the recipes. `make` is not installable on
# every machine this project has to run on (a WSL image with no make binary, no
# compiler to build one and no passwordless sudo is the case that forced this),
# and a documented workflow that cannot be executed is not a workflow. The
# Makefile is kept as a thin wrapper so `make <target>` still works where make
# exists; both entry points run exactly the same code because there is only one
# copy of it.
#
#   ./do                  list the targets
#   ./do <target> [args]  run one; trailing args are forwarded to the command
#
# Backend runs on uv-managed CPython 3.12 in backend/.venv. The system python3
# is 3.14 with no pip and must never be used, so targets call the venv binaries
# directly instead of going through `uv run` (which would re-resolve the pins).
# Frontend needs Node 22; the default node on PATH is v14 and breaks Next.js, so
# every web target sources nvm first.
#
# Semantics match the Makefile's `SHELL := /bin/bash` with
# `.SHELLFLAGS := -eu -o pipefail -c`: unset variables and failed commands -
# including inside a pipeline - stop the target.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND="$ROOT/backend"
WEB="$ROOT/frontend"
BIN="$BACKEND/.venv/bin"

DB_USER=secondchance
DB_NAME=secondchance
DB_TEST=secondchance_test

# ---------------------------------------------------------------- helpers

compose() { docker compose -f "$ROOT/docker-compose.yml" "$@"; }

uv_() { PATH="$HOME/.local/bin:$PATH" uv "$@"; }

has_web() { [ -f "$WEB/package.json" ]; }

# nvm.sh reads variables it has not set yet, so it cannot be sourced under
# `set -u`. Relax nounset for the source itself and nothing else.
#
# Without nvm, fall back to the node already on PATH, but only when it is new
# enough. CI runners install Node directly and have no ~/.nvm; the development
# machine has nvm and a v14 default. Both need to work, and neither should be
# able to start a build on a Node that breaks Next.js.
use_node() {
	if [ -s "$HOME/.nvm/nvm.sh" ]; then
		set +u
		# shellcheck disable=SC1091
		. "$HOME/.nvm/nvm.sh"
		nvm use 22 >/dev/null
		set -u
		return
	fi

	local major=""
	if command -v node >/dev/null 2>&1; then
		major="$(node --version | sed 's/^v//; s/[^0-9].*//')"
	fi
	if [ -n "$major" ] && [ "$major" -ge 22 ]; then
		return
	fi

	die "Node 22 is required (the default node breaks Next.js): no nvm at ~/.nvm, and node on PATH is ${major:-not installed}"
}

# Export backend/.env into this shell. Used where a command needs a variable
# that pydantic-settings would otherwise load for itself.
load_backend_env() {
	[ -f "$BACKEND/.env" ] || die "$BACKEND/.env does not exist - copy .env.example and fill it in"
	set -a
	# shellcheck disable=SC1091
	. "$BACKEND/.env"
	set +a
}

need_venv() {
	[ -x "$BIN/python" ] || die "backend/.venv is missing or incomplete - run './do install' first"
}

die() {
	echo "error: $*" >&2
	exit 2
}

# ---------------------------------------------------------------- setup

describe_install="Create backend/.venv and install backend + frontend dependencies"
target_install() {
	(cd "$BACKEND" && uv_ venv --python 3.12)
	(cd "$BACKEND" && uv_ pip install -e ".[dev]")
	if has_web; then
		(use_node && cd "$WEB" && npm install)
	else
		echo "skip: $WEB/package.json does not exist yet"
	fi
}

# ---------------------------------------------------------------- database

describe_db_up="Start Postgres 16 on localhost:5433 and ensure both databases exist"
target_db_up() {
	compose up -d --wait
	local exists
	exists="$(compose exec -T db psql -qtAX -U "$DB_USER" -d "$DB_NAME" \
		-c "SELECT 1 FROM pg_database WHERE datname = '$DB_TEST'")"
	if [ "$exists" != "1" ]; then
		compose exec -T db createdb -U "$DB_USER" -O "$DB_USER" "$DB_TEST"
	fi
	echo "databases ready on localhost:5433 -> $DB_NAME, $DB_TEST"
}

describe_db_down="Stop Postgres, keeping the data volume"
target_db_down() {
	compose down
}

describe_db_reset="Destroy the data volume, recreate both databases, migrate and seed"
target_db_reset() {
	compose down -v
	target_db_up
	target_migrate
	target_seed
}

describe_db_shell="Open a psql shell on the app database"
target_db_shell() {
	compose exec db psql -U "$DB_USER" -d "$DB_NAME"
}

describe_migrate="Apply alembic migrations to head, on both the app and test databases"
target_migrate() {
	need_venv
	(cd "$BACKEND" && "$BIN/alembic" upgrade head)
	(
		cd "$BACKEND"
		load_backend_env
		DATABASE_URL="$TEST_DATABASE_URL" "$BIN/alembic" upgrade head
	)
}

describe_revision='Autogenerate a migration. Usage: ./do revision "add programs table"'
target_revision() {
	need_venv
	local message="${1-}"
	# `m=...` is accepted too, so the make-style invocation keeps working.
	case "$message" in
	m=*) message="${message#m=}" ;;
	esac
	[ -n "$message" ] || die 'usage: ./do revision "message"'
	(cd "$BACKEND" && "$BIN/alembic" revision --autogenerate -m "$message")
}

describe_seed="Load the curated program catalog into the database"
target_seed() {
	need_venv
	(cd "$BACKEND" && "$BIN/python" -m scripts.seed "$@")
}

# ---------------------------------------------------------------- quality

describe_lint="ruff check + format check (backend, benchmarks), eslint (frontend)"
target_lint() {
	need_venv
	(cd "$BACKEND" && "$BIN/ruff" check .)
	(cd "$BACKEND" && "$BIN/ruff" format --check .)
	# benchmarks/ sits outside backend/, so ruff has to be pointed at the
	# project's config explicitly or it would silently fall back to defaults.
	(cd "$ROOT" && "$BIN/ruff" check --config "$BACKEND/pyproject.toml" benchmarks)
	(cd "$ROOT" && "$BIN/ruff" format --check --config "$BACKEND/pyproject.toml" benchmarks)
	if has_web; then
		(use_node && cd "$WEB" && npm run lint)
	else
		echo "skip: frontend not scaffolded yet"
	fi
}

describe_fmt="Autoformat and autofix"
target_fmt() {
	need_venv
	(cd "$BACKEND" && "$BIN/ruff" format .)
	(cd "$BACKEND" && "$BIN/ruff" check --fix .)
	(cd "$ROOT" && "$BIN/ruff" format --config "$BACKEND/pyproject.toml" benchmarks)
	(cd "$ROOT" && "$BIN/ruff" check --fix --config "$BACKEND/pyproject.toml" benchmarks)
	if has_web; then
		(use_node && cd "$WEB" && npx --yes prettier --write .)
	else
		echo "skip: frontend not scaffolded yet"
	fi
}

describe_typecheck="mypy --strict (backend, benchmarks), tsc --noEmit (frontend)"
target_typecheck() {
	need_venv
	(cd "$BACKEND" && "$BIN/mypy" app)
	# MYPYPATH rather than PYTHONPATH: the editable install of `app` has no
	# py.typed marker, so mypy has to find it as source or it skips it.
	(cd "$ROOT" && MYPYPATH="$BACKEND${MYPYPATH:+:$MYPYPATH}" \
		"$BIN/mypy" --config-file "$BACKEND/pyproject.toml" benchmarks)
	if [ -f "$WEB/tsconfig.json" ]; then
		(use_node && cd "$WEB" && npx --yes tsc --noEmit)
	else
		echo "skip: frontend not scaffolded yet"
	fi
}

describe_test="Full backend test suite with coverage"
target_test() {
	need_venv
	target_db_up
	(cd "$BACKEND" && "$BIN/pytest" --cov=app --cov-report=term-missing "$@")
}

describe_test_unit="Unit tests only, no database and no live Gemini"
target_test_unit() {
	need_venv
	(cd "$BACKEND" && "$BIN/pytest" -m "not db and not live" "$@")
}

describe_test_int="Integration tests that require Postgres"
target_test_int() {
	need_venv
	target_db_up
	(cd "$BACKEND" && "$BIN/pytest" -m db "$@")
}

describe_e2e="Drive the whole product end to end over HTTP and assert every transition"
target_e2e() {
	need_venv
	target_db_up
	target_migrate
	(cd "$BACKEND" && "$BIN/python" -m scripts.e2e_journey "$@")
}

describe_e2e_web="Browser end to end run, once a Playwright suite exists"
target_e2e_web() {
	if [ -f "$WEB/playwright.config.ts" ]; then
		(use_node && cd "$WEB" && npx --yes playwright test "$@")
	else
		echo "skip: no $WEB/playwright.config.ts yet - run './do e2e' for the API journey"
	fi
}

describe_smoke="Drive the core flow against a running API and assert it responds"
target_smoke() {
	need_venv
	(cd "$BACKEND" && "$BIN/python" -m scripts.smoke "$@")
}

describe_bench="Latency and cost benchmark over the golden scenarios"
target_bench() {
	need_venv
	# benchmarks/ lives at the repository root (docs/INTERFACES.md), so it is
	# imported from here with the backend on PYTHONPATH rather than from
	# inside backend/.
	(cd "$ROOT" && PYTHONPATH="$BACKEND${PYTHONPATH:+:$PYTHONPATH}" \
		"$BIN/python" -m benchmarks.run "$@")
}

describe_types="Regenerate frontend/openapi.json and the TypeScript types built from it"
target_types() {
	need_venv
	(cd "$BACKEND" && "$BIN/python" -m scripts.export_openapi --output "$WEB/openapi.json")
	if has_web; then
		(use_node && cd "$WEB" && npm run types)
	else
		echo "skip: frontend not scaffolded yet"
	fi
}

# ---------------------------------------------------------------- run

describe_dev_api="Run the API with autoreload on :8000"
target_dev_api() {
	need_venv
	(cd "$BACKEND" && "$BIN/uvicorn" app.main:app --reload --host 0.0.0.0 --port 8000 "$@")
}

describe_dev_web="Run the Next.js dev server on :3000"
target_dev_web() {
	(use_node && cd "$WEB" && npm run dev "$@")
}

describe_build_web="Production build of the frontend"
target_build_web() {
	(use_node && cd "$WEB" && npm run build "$@")
}

describe_verify="Everything CI runs"
target_verify() {
	target_lint
	target_typecheck
	target_test
	echo "verify: OK"
}

# ---------------------------------------------------------------- dispatch

# Every target, in the order `./do help` prints them.
TARGETS=(
	help install
	db-up db-down db-reset db-shell migrate revision seed
	lint fmt typecheck test test-unit test-int e2e e2e-web smoke bench types
	dev-api dev-web build-web verify
)

describe() {
	local name="describe_${1//-/_}"
	echo "${!name-}"
}

describe_help="List available targets"
target_help() {
	echo "SecondChance targets - run them as ./do <target>:"
	local name
	for name in "${TARGETS[@]}"; do
		printf '  \033[36m%-11s\033[0m %s\n' "$name" "$(describe "$name")"
	done
}

main() {
	local requested="${1-help}"
	if [ "$#" -gt 0 ]; then
		shift
	fi

	local name
	for name in "${TARGETS[@]}"; do
		if [ "$name" = "$requested" ]; then
			"target_${requested//-/_}" "$@"
			return
		fi
	done

	echo "error: no such target: $requested" >&2
	echo >&2
	target_help >&2
	exit 2
}

main "$@"
