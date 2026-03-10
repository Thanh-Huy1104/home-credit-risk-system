# Home Credit Risk System

A production-ready machine learning pipeline for credit default risk prediction using XGBoost and DuckDB.

## Overview

This system predicts the probability that a loan applicant will default on their loan. It uses aggregated features from multiple data sources (credit bureau, previous applications, payment history) to train an XGBoost classifier.

**Key Features:**
- DuckDB for out-of-core data aggregation
- XGBoost with native categorical handling
- SHAP explainability for model interpretations
- FastAPI serving endpoint for real-time predictions
- Modular, testable codebase

## Project Structure

```
home-credit-risk-system/
├── configs/              # Configuration files
│   └── base.yaml         # Main configuration
├── data/
│   ├── raw/              # Raw CSV files (not tracked)
│   ├── processed/        # Parquet files (not tracked)
│   └── duckdb/           # DuckDB database (not tracked)
├── docs/                 # Documentation
├── models/
│   ├── artifacts/        # Saved models (not tracked)
│   └── metrics/          # Model metrics (not tracked)
├── pipelines/            # Pipeline scripts
│   ├── ingest.py         # Load raw data into DuckDB
│   ├── aggregate.py      # Create aggregated features
│   ├── train.py          # Train XGBoost model
│   └── score.py          # Batch scoring
├── src/                  # Core modules
│   ├── config.py         # Configuration management
│   ├── features/         # Feature engineering
│   ├── models/           # Model management
│   └── serving/          # FastAPI app
├── tests/                # Test suite
├── reports/              # Generated reports and figures
├── Makefile              # Common commands
├── pyproject.toml        # Project dependencies
└── README.md
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install dependencies
make setup

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Prepare Data

Place your CSV files in `data/raw/`:
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`

### 3. Run Pipeline

```bash
# Ingest raw data
make ingest

# Create aggregated features
make aggregate

# Train model
make train

# Score new data
make score
```

### 4. Serve Predictions

```bash
make serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Configuration

Edit `configs/base.yaml` to customize:

```yaml
project:
  seed: 42

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  duckdb_path: "data/duckdb/home_credit.duckdb"
  models_dir: "models/artifacts"
  metrics_dir: "models/metrics"

split:
  test_size: 0.20
  val_size: 0.20
  stratify: true
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Default Risk

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100001,
    "features": {
      "AMT_INCOME_TOTAL": 200000,
      "AMT_CREDIT": 500000,
      "DAYS_BIRTH": -15000,
      ...
    }
  }'
```

Response:
```json
{
  "SK_ID_CURR": 100001,
  "default_probability": 0.0847,
  "risk_level": "low",
  "model_version": "v1.0.0"
}
```

## Development

### Run Tests

```bash
make test
# or
pytest tests/ -v
```

### Code Quality

```bash
make fmt    # Format code
make lint   # Lint code
```

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.76 |
| Training samples | 307,511 |
| Test samples | 48,744 |
| Features | 189 |

## Architecture

```
Raw CSV Files → DuckDB → Feature Aggregation → XGBoost Training → Model Artifacts
                                                              ↓
                                          FastAPI ← Feature Store ← Batch Scoring
```

## License

MIT
