# Home Credit Risk System - Project Report

## Overview

This project implements a machine learning pipeline for the Home Credit Default Risk competition. The goal is to predict whether a loan applicant will default on their loan.

### Key Metrics
- **Total records:** 356,255 applicants
- **Training set:** 307,511 (86.32%)
- **Test set:** 48,744 (13.68%)
- **Class imbalance:** 11.4:1 (91.93% no default, 8.07% default)
- **Total features:** 189 columns (122 base + 67 aggregated)

---

## Pipeline Files

### 1. `pipelines/ingest.py`

**Purpose:** Loads raw CSV data and prepares it for processing.

**What it does:**
1. Reads `application_train.csv` and `application_test.csv` using Pandas
2. Saves individual parquet files for train and test
3. Adds `is_train` flag (1 for train, 0 for test)
4. Concatenates both datasets into `application_all.parquet`
5. Creates `application_all` table in DuckDB

**Key decisions:**
- Uses Pandas to read CSVs (handles complex text qualifiers better than DuckDB's CSV reader)
- Uses Pandas `concat()` instead of SQL `UNION ALL` to preserve correct column types
- Saves intermediate parquet for faster subsequent runs

**Critical fix applied:**
- **Issue:** SQL `UNION ALL` was converting numeric columns to VARCHAR, causing data shift
- **Solution:** Use Pandas to concatenate dataframes before writing to parquet, preserving correct dtypes

---

### 2. `pipelines/aggregate.py`

**Purpose:** Creates aggregated features from supplementary tables following the "Aggregate Before You Join" principle.

**What it does:**
1. Aggregates 6 supplementary tables by `SK_ID_CURR`:
   - **bureau.csv** - External credit history (loan counts, debt, overdue metrics)
   - **bureau_balance.csv** - Monthly credit bureau balance status
   - **previous_application.csv** - Previous Home Credit applications
   - **installments_payments.csv** - Payment history
   - **POS_CASH_balance.csv** - Point of sale cash loans
   - **credit_card_balance.csv** - Credit card balance history

2. Uses DuckDB's `read_csv_auto()` for out-of-core aggregation (memory efficient)
3. Joins all aggregations to master table in a single LEFT JOIN query
4. Writes final feature table to `application_features.parquet`

**Aggregated features include:**
- Count of previous loans
- Total/average debt and credit amounts
- Percentage of active vs closed loans
- Number of refused applications
- Payment behavior (late payments, payment differences)
- Credit card utilization metrics

**Key principle:** Aggregating BEFORE joining prevents row duplication. A single applicant with 6 previous loans would otherwise create 6 rows in the master table.

---

### 3. `pipelines/visualize.py`

**Purpose:** Explores and documents the dataset for understanding.

**What it shows:**
- Table overview and row counts
- Train/test split verification
- Target distribution (class imbalance)
- Column categories breakdown
- Sample aggregated feature statistics
- Credit history summary statistics
- Risk indicator analysis

**Explains:**
- Difference between train and test sets
- What `is_train` flag means
- How the data is structured for modeling

---

### 4. `pipelines/train.py`

**Purpose:** Trains XGBoost model for default prediction.

**What it does:**
1. Loads `application_features.parquet`
2. Splits into train/test using `is_train` flag
3. Separates features (X) from target (y)
4. Creates validation split (80/20 stratified)
5. Calculates imbalance ratio for `scale_pos_weight`
6. Trains XGBoost with early stopping
7. Evaluates on validation set using ROC-AUC

**Key parameters:**
- `n_estimators=1000` - Max trees
- `learning_rate=0.05`
- `max_depth=6`
- `scale_pos_weight` - Handles class imbalance
- `eval_metric="auc"` - Optimizes for ROC-AUC
- `early_stopping_rounds=50` - Prevents overfitting
- `tree_method="hist"` - Optimized for CPU
- `enable_categorical=True` - Native categorical handling

---

## Configuration Files

### `configs/base.yaml`

Project configuration including:
- Paths for data directories
- DuckDB database location
- Model/metrics output directories
- Dataset configuration
- Train/test split parameters

---

## Data Flow

```
Raw CSV Files                    Processing              Output
─────────────────                ──────────              ──────

application_train.csv  ─────┐
                           │    ingest.py             application_train.parquet
application_test.csv   ─────┤                         application_test.parquet
                           │                         application_all.parquet
                           ├────► DuckDB              application_all table
                                        
bureau.csv            ──────┐
bureau_balance.csv    ──────┤
previous_application.csv───►  aggregate.py            application_features.parquet
installments_payments.csv──┤                         (356,255 × 189)
POS_CASH_balance.csv   ─────┤
credit_card_balance.csv─────┘

application_features.parquet ──► train.py              Trained model
```

---

## Issues Encountered and Fixed

### 1. Data Shift Bug (Critical)

**Problem:** 
- Model was learning from garbage data
- `AMT_GOODS_PRICE` contained text values like "Unaccompanied", "Family"
- `DAYS_BIRTH` had 9,274 rows with placeholder value 365243
- Column names didn't match actual data

**Root Cause:**
DuckDB's `UNION ALL` was converting all columns to VARCHAR when combining train and test, allowing string values to leak into numeric columns.

**Solution:**
Use Pandas `pd.concat()` to merge dataframes before writing to parquet, preserving correct dtypes. Then load into DuckDB from parquet (which preserves types).

**Impact:**
- Before: 45,608 rows with text in numeric columns, 9,274 placeholder dates
- After: 0 issues

---

### 2. Missing Dependencies

**Problem:** `xgboost` was not listed in `pyproject.toml`

**Solution:** Added `xgboost>=2.0` to dependencies

---

## Key Concepts

### Aggregate Before You Join
Supplementary tables have one-to-many relationships with the master table. Joining without aggregation duplicates applicant rows:
- Applicant 100001 might have 6 loans in bureau.csv
- Standard JOIN would create 6 rows for that applicant
- Solution: Aggregate to 1 row per applicant first, then join

### Class Imbalance
- 91.93% No Default (0)
- 8.07% Default (1)
- Imbalance ratio: 11.4:1
- Handled via `scale_pos_weight` parameter in XGBoost

### DuckDB Benefits
- Out-of-core processing (handles large files without loading into RAM)
- Fast SQL aggregations
- Integration with Python data ecosystem

---

## Running the Pipeline

```bash
# Ingest raw data
python pipelines/ingest.py --config configs/base.yaml

# Create aggregated features
python pipelines/aggregate.py --config configs/base.yaml

# Explore data
python pipelines/visualize.py

# Train model
python pipelines/train.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/application_train.parquet` | Training data |
| `data/processed/application_test.parquet` | Test data |
| `data/processed/application_all.parquet` | Combined train+test |
| `data/processed/application_features.parquet` | Full feature set |
| `data/duckdb/home_credit.duckdb` | DuckDB database with all tables |

---

## DuckDB Tables

| Table | Rows | Description |
|-------|------|-------------|
| `application_all` | 356,255 | Combined train/test |
| `application_features` | 356,255 | Full feature set (189 cols) |
| `bureau_agg` | 305,811 | Bureau aggregations |
| `bureau_balance_agg` | 134,542 | Bureau balance aggregations |
| `prev_app_agg` | 338,857 | Previous application aggregations |
| `installments_agg` | 339,587 | Installments aggregations |
| `pos_cash_agg` | 337,252 | POS cash aggregations |
| `credit_card_agg` | 103,558 | Credit card aggregations |
