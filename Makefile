.PHONY: install dev test coverage lint format typecheck check clean

ifeq ($(wildcard .venv/bin/python),)
PY ?= python3
else
PY ?= .venv/bin/python
endif

install: ## Install as editable package with provider support
	$(PY) -m pip install -e '.[native]'

dev: ## Install with development tooling
	$(PY) -m pip install -e '.[dev,native]'

test: ## Run the test suite
	$(PY) -m pytest

coverage: ## Run tests with coverage enforcement
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Ruff lint + format check
	$(PY) -m ruff check fastevals tests
	$(PY) -m ruff format --check fastevals tests

format: ## Auto-format the codebase
	$(PY) -m ruff check --fix fastevals tests
	$(PY) -m ruff format fastevals tests

typecheck: ## mypy strict
	$(PY) -m mypy

check: lint typecheck coverage ## All quality gates

clean: ## Remove caches and run artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov runs
	find . -name "__pycache__" -type d -exec rm -rf {} +
