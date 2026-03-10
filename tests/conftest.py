"""Shared pytest fixtures."""


import duckdb
import pandas as pd
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_application_data():
    """Create sample application data."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003, 100004, 100005],
            "TARGET": [0, 1, 0, 0, 1],
            "AMT_INCOME_TOTAL": [200000, 150000, 300000, 250000, 180000],
            "AMT_CREDIT": [500000, 400000, 600000, 550000, 450000],
            "AMT_ANNUITY": [25000, 20000, 30000, 27500, 22500],
            "DAYS_BIRTH": [-15000, -12000, -18000, -16000, -14000],
            "DAYS_EMPLOYED": [-1000, -500, -2000, -1500, -800],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Revolving loans",
                "Cash loans",
                "Cash loans",
                "Revolving loans",
            ],
            "CODE_GENDER": ["M", "F", "M", "F", "M"],
        }
    )


@pytest.fixture
def sample_bureau_data():
    """Create sample bureau data."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001, 100002, 100003, 100003],
            "SK_ID_BUREAU": [1, 2, 3, 4, 5],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Closed"],
            "AMT_CREDIT_SUM_DEBT": [50000, 0, 30000, 40000, 0],
            "AMT_CREDIT_SUM": [100000, 50000, 80000, 90000, 60000],
            "AMT_CREDIT_SUM_OVERDUE": [0, 0, 1000, 0, 0],
            "CREDIT_DAY_OVERDUE": [0, 0, 30, 0, 0],
        }
    )


@pytest.fixture
def sample_duckdb(temp_dir, sample_application_data, sample_bureau_data):
    """Create a sample DuckDB database."""
    db_path = temp_dir / "test.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute("CREATE TABLE application_all AS SELECT * FROM sample_application_data")

    con.close()
    return db_path


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file."""
    config_content = """
project:
  seed: 42

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  duckdb_path: "data/duckdb/test.duckdb"
  models_dir: "models/test"
  metrics_dir: "metrics/test"

split:
  test_size: 0.2
  val_size: 0.2
  stratify: true

model:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 4
"""
    config_path = temp_dir / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_features_df():
    """Create sample features DataFrame for testing."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002, 100003, 100004, 100005],
            "is_train": [1, 1, 1, 0, 0],
            "TARGET": [0, 1, 0, None, None],
            "AMT_INCOME_TOTAL": [200000, 150000, 300000, 250000, 180000],
            "AMT_CREDIT": [500000, 400000, 600000, 550000, 450000],
            "DAYS_BIRTH": [-15000, -12000, -18000, -16000, -14000],
            "bureau_loan_count": [2, 1, 2, 0, 1],
            "bureau_total_debt": [50000, 30000, 40000, 0, 20000],
            "prev_app_count": [3, 2, 1, 0, 2],
        }
    )
