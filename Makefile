.PHONY: install dev test coverage lint format typecheck check demo clean

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
	$(PY) -m ruff check fasteval tests
	$(PY) -m ruff format --check fasteval tests

format: ## Auto-format the codebase
	$(PY) -m ruff check --fix fasteval tests
	$(PY) -m ruff format fasteval tests

typecheck: ## mypy strict
	$(PY) -m mypy

check: lint typecheck coverage ## All quality gates

demo: ## Run the no-API-key mock demo
	$(PY) -m fasteval.cli --prompt "Explain evaluation in one sentence" --providers mock --out runs/demo
	@echo "Open runs/demo/*/report.html"

clean: ## Remove caches and run artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov runs
	find . -name "__pycache__" -type d -exec rm -rf {} +
