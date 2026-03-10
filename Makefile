# Makefile for home-credit-risk-system
# Production-ready ML pipeline

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

VENV := .venv
VENV_BIN := $(VENV)/bin
PY := $(VENV_BIN)/python
PYTEST := $(VENV_BIN)/pytest
UVICORN := $(VENV_BIN)/uvicorn

.PHONY: help setup venv install fmt lint ingest aggregate train score serve test clean dirs

help:
	@echo "Home Credit Risk System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Create venv + install deps + create dirs"
	@echo "  make install      - Install package in editable mode"
	@echo ""
	@echo "Pipeline:"
	@echo "  make ingest       - Load raw CSV data into DuckDB/Parquet"
	@echo "  make aggregate    - Create aggregated features"
	@echo "  make train        - Train XGBoost model"
	@echo "  make score        - Batch score new data"
	@echo "  make pipeline     - Run full pipeline (ingest -> aggregate -> train)"
	@echo ""
	@echo "Serving:"
	@echo "  make serve        - Start FastAPI server on port 8000"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make fmt          - Format code with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make clean        - Remove caches and build artifacts"
	@echo ""
	@echo "Utilities:"
	@echo "  make dirs         - Create directory structure"

# Setup
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
	else \
		echo "No pyproject.toml found."; \
		exit 1; \
	fi

dirs:
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR) $(DUCKDB_DIR)
	@mkdir -p $(REPORTS_DIR) $(FIGURES_DIR)
	@mkdir -p $(ARTIFACTS_DIR) $(METRICS_DIR)
	@mkdir -p $(CONFIG_DIR)
	@mkdir -p tests
	@echo "Created directory structure"

# Pipeline
ingest:
	@$(PY) -m $(PIPE_DIR).ingest --config $(BASE_CONFIG)

aggregate:
	@$(PY) -m $(PIPE_DIR).aggregate --config $(BASE_CONFIG)

train:
	@$(PY) -m $(PIPE_DIR).train --config $(BASE_CONFIG)

score:
	@$(PY) -m $(PIPE_DIR).score --config $(BASE_CONFIG)

pipeline: ingest aggregate train

# Serving
serve:
	@$(UVICORN) src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Development
test:
	@$(PYTEST) tests/ -v --tb=short

test-cov:
	@$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

fmt:
	@$(PY) -m ruff format .

lint:
	@$(PY) -m ruff check . --fix

# Cleanup
clean:
	@rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned caches"

clean-data:
	@rm -rf $(PROCESSED_DIR)/*.parquet
	@rm -rf $(DUCKDB_DIR)/*.duckdb
	@rm -rf $(DUCKDB_DIR)/*.wal
	@echo "Cleaned processed data"

clean-models:
	@rm -rf $(ARTIFACTS_DIR)/*.joblib
	@rm -rf $(ARTIFACTS_DIR)/*.json
	@rm -rf $(METRICS_DIR)/*.json
	@echo "Cleaned model artifacts"
