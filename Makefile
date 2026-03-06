# Makefile for fraud-risk-system
# Usage:
#   make setup
#   make ingest
#   make train
#   make report
#   make serve
#   make test
#   make clean

SHELL := /bin/bash

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
PIP := pip

PROJECT := home_credit_risk
SRC_DIR := src
PIPE_DIR := pipelines

DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
PROCESSED_DIR := $(DATA_DIR)/processed
DUCKDB_DIR := $(DATA_DIR)/duckdb

REPORTS_DIR := reports
FIGURES_DIR := $(REPORTS_DIR)/figures

MODELS_DIR := models
ARTIFACTS_DIR := $(MODELS_DIR)/artifacts
METRICS_DIR := $(MODELS_DIR)/metrics

CONFIG_DIR := configs
BASE_CONFIG := $(CONFIG_DIR)/base.yaml
MODEL_CONFIG := $(CONFIG_DIR)/model_lgbm.yaml
COST_CONFIG := $(CONFIG_DIR)/costs.yaml

VENV := .venv
VENV_BIN := $(VENV)/bin
PY := $(VENV_BIN)/python
PYTEST := $(VENV_BIN)/pytest
UVICORN := $(VENV_BIN)/uvicorn

.PHONY: help setup venv dirs install fmt lint ingest train score report serve test clean

help:
	@echo "Targets:"
	@echo "  make setup    - create venv + install deps + create dirs"
	@echo "  make ingest   - run data ingest (raw -> duckdb/parquet)"
	@echo "  make train    - train + evaluate models, save artifacts/metrics"
	@echo "  make score    - batch scoring (later)"
	@echo "  make report   - regenerate reports/figures (if separated)"
	@echo "  make serve    - run FastAPI service locally"
	@echo "  make test     - run tests"
	@echo "  make clean    - remove caches and build artifacts"

setup: venv install dirs

venv:
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "Created venv at $(VENV)"; \
	fi
	@$(PY) -m pip install -U pip setuptools wheel

install:
	@if [ -f "pyproject.toml" ]; then \
		$(PY) -m pip install -e ".[dev]"; \
	elif [ -f "requirements.txt" ]; then \
		$(PY) -m pip install -r requirements.txt; \
	else \
		echo "No pyproject.toml or requirements.txt found."; \
		exit 1; \
	fi

dirs:
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR) $(DUCKDB_DIR)
	@mkdir -p $(REPORTS_DIR) $(FIGURES_DIR)
	@mkdir -p $(ARTIFACTS_DIR) $(METRICS_DIR)
	@mkdir -p $(CONFIG_DIR)
	@mkdir -p tests
	@echo "Created data/, reports/, models/, configs/, tests/ directories"

# Optional: formatting/linting if you add tools (ruff/black)
fmt:
	@echo "Formatting (optional) ..."
	@$(PY) -m ruff format . || true

lint:
	@echo "Linting (optional) ..."
	@$(PY) -m ruff check . || true

# Pipelines (these assume you have pipelines/ingest.py and pipelines/train.py)
ingest:
	@$(PY) -m $(PIPE_DIR).ingest --config $(BASE_CONFIG)

train:
	@$(PY) -m $(PIPE_DIR).train --config $(BASE_CONFIG) --model-config $(MODEL_CONFIG) --cost-config $(COST_CONFIG)

score:
	@$(PY) -m $(PIPE_DIR).score_batch --config $(BASE_CONFIG) --model-config $(MODEL_CONFIG)

report:
	@$(PY) -m $(PIPE_DIR).report --config $(BASE_CONFIG)

serve:
	@$(UVICORN) $(PROJECT).serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	@$(PYTEST) -q

clean:
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleaned caches"
