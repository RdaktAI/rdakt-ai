.PHONY: help
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help
help:
	@echo 'Usage:'
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' |  sed -e 's/^/ /'

## init: Initialize the project
.PHONY: init
init:
	uv venv
	uv sync --all-extras --all-groups
	uv tool install pre-commit
	pre-commit install

## test: Run the test suite
.PHONY: test
test:
	uv run pytest tests/ -v --tb=short

## test/cov: Run tests with coverage
.PHONY: test/cov
test/cov:
	uv run pytest tests/ -v --tb=short --cov=rdakt_ai --cov-report=term-missing

## pre_commit: Run the pre-commit pipeline
.PHONY: pre_commit
pre_commit:
	uv run pre-commit run --all

## mypy: Run mypy type checking
.PHONY: mypy
mypy:
	uv run mypy --install-types --non-interactive

## audit: Run the full audit pipeline (pre-commit + mypy)
.PHONY: audit
audit: pre_commit mypy

## lint: Run ruff check and format
.PHONY: lint
lint:
	uv run ruff check rdakt_ai tests
	uv run ruff format --check rdakt_ai tests

## build: Build the package
.PHONY: build
build:
	uv build
