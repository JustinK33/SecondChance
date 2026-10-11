# SecondChance - developer entry points.
#
# The recipes are NOT here. They live in ./do, a POSIX-shell entry point, and
# every target below is a one-line delegation to it.
#
# Why: make is not available on every machine this project has to run on (the
# case that forced this was a WSL image with no make binary, no compiler to
# build one and no passwordless sudo to install one), and a documented workflow
# that cannot be executed is not a workflow. Keeping one copy of each recipe in
# ./do means the two entry points cannot drift apart:
#
#   ./do <target> [args]      works everywhere, no make needed
#   make <target> [ARGS=...]  works wherever make exists
#
# `./do help` is the list of targets with descriptions - including this file's.
# Pass extra arguments to the underlying command with ARGS, e.g.
# `make smoke ARGS="--base-url http://127.0.0.1:8000"`.
#
# ./do sets `set -euo pipefail`, which is the same shell contract these
# variables used to give the recipes directly.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DO   := $(ROOT)/do

ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help install db-up db-down db-reset db-shell migrate revision seed lint fmt \
        typecheck test test-unit test-int e2e e2e-web smoke bench types dev-api dev-web \
        build-web verify

help:
	@$(DO) help

# ---------------------------------------------------------------- setup

install:
	@$(DO) install $(ARGS)

# ---------------------------------------------------------------- database

db-up:
	@$(DO) db-up

db-down:
	@$(DO) db-down

db-reset:
	@$(DO) db-reset

db-shell:
	@$(DO) db-shell

migrate:
	@$(DO) migrate

# Usage: make revision m="add programs table"
revision:
	@$(DO) revision "$(m)"

seed:
	@$(DO) seed $(ARGS)

# ---------------------------------------------------------------- quality

lint:
	@$(DO) lint

fmt:
	@$(DO) fmt

typecheck:
	@$(DO) typecheck

test:
	@$(DO) test $(ARGS)

test-unit:
	@$(DO) test-unit $(ARGS)

test-int:
	@$(DO) test-int $(ARGS)

e2e:
	@$(DO) e2e $(ARGS)

e2e-web:
	@$(DO) e2e-web $(ARGS)

smoke:
	@$(DO) smoke $(ARGS)

bench:
	@$(DO) bench $(ARGS)

types:
	@$(DO) types

# ---------------------------------------------------------------- run

dev-api:
	@$(DO) dev-api $(ARGS)

dev-web:
	@$(DO) dev-web $(ARGS)

build-web:
	@$(DO) build-web $(ARGS)

verify:
	@$(DO) verify
