"""Integration tests for the ML pipeline."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def integration_config(tmp_path):
    """Create integration test configuration."""
    config_content = """
project:
  seed: 42

paths:
  data_raw: "{data_raw}"
  data_processed: "{data_processed}"
  duckdb_path: "{duckdb_path}"
  models_dir: "{models_dir}"
  metrics_dir: "{metrics_dir}"

split:
  test_size: 0.2
  val_size: 0.2
  stratify: true

model:
  n_estimators: 10
  learning_rate: 0.1
  max_depth: 2
  min_child_weight: 1
  colsample_bytree: 0.8
  subsample: 0.8
  early_stopping_rounds: 5
  tree_method: "hist"
  enable_categorical: true
  random_state: 42
"""
    config_path = tmp_path / "config.yaml"
    config_content = config_content.format(
        data_raw=str(tmp_path / "raw"),
        data_processed=str(tmp_path / "processed"),
        duckdb_path=str(tmp_path / "db" / "test.duckdb"),
        models_dir=str(tmp_path / "models"),
        metrics_dir=str(tmp_path / "metrics"),
    )
    config_path.write_text(config_content)
    return Config.from_yaml(config_path)


@pytest.fixture
def sample_raw_data(integration_config):
    """Create sample raw data files."""
    raw_dir = Path(integration_config.paths.data_raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_train = 100
    n_test = 20

    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n_train + 1),
            "TARGET": [0] * (n_train - 10) + [1] * 10,
            "AMT_INCOME_TOTAL": [200000 + i * 1000 for i in range(n_train)],
            "AMT_CREDIT": [500000 + i * 2000 for i in range(n_train)],
            "AMT_ANNUITY": [25000 + i * 100 for i in range(n_train)],
            "DAYS_BIRTH": [-15000 - i * 10 for i in range(n_train)],
            "DAYS_EMPLOYED": [-1000 - i * 5 for i in range(n_train)],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * n_train,
            "CODE_GENDER": ["M", "F"] * (n_train // 2),
        }
    )

    test_df = train_df.drop(columns=["TARGET"]).head(n_test).copy()
    test_df["SK_ID_CURR"] = range(n_train + 1, n_train + n_test + 1)

    train_path = raw_dir / "application_train.csv"
    test_path = raw_dir / "application_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    bureau_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3, 4],
            "SK_ID_BUREAU": [101, 102, 103, 104, 105],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Closed"],
            "AMT_CREDIT_SUM_DEBT": [50000, 0, 30000, 40000, 0],
            "AMT_CREDIT_SUM": [100000, 50000, 80000, 90000, 60000],
            "AMT_CREDIT_SUM_OVERDUE": [0, 0, 1000, 0, 0],
            "CREDIT_DAY_OVERDUE": [0, 0, 30, 0, 0],
            "AMT_CREDIT_MAX_OVERDUE": [0, 0, 500, 0, 0],
            "AMT_ANNUITY": [5000, 0, 4000, 3000, 0],
            "DAYS_CREDIT": [-100, -200, -150, -300, -400],
        }
    )
    bureau_df.to_csv(raw_dir / "bureau.csv", index=False)

    prev_app_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3],
            "SK_ID_PREV": [201, 202, 203, 204],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Canceled"],
            "AMT_APPLICATION": [200000, 150000, 300000, 100000],
            "AMT_CREDIT": [200000, 0, 300000, 0],
            "AMT_ANNUITY": [10000, 0, 15000, 0],
            "DAYS_DECISION": [-100, -200, -150, -250],
            "CNT_PAYMENT": [24, 0, 36, 0],
        }
    )
    prev_app_df.to_csv(raw_dir / "previous_application.csv", index=False)

    bureau_balance_df = pd.DataFrame(
        {
            "SK_ID_BUREAU": [101, 102, 103],
            "SK_ID_CURR": [1, 1, 2],
            "MONTHS_BALANCE": [-1, -2, -1],
            "STATUS": ["0", "1", "C"],
        }
    )
    bureau_balance_df.to_csv(raw_dir / "bureau_balance.csv", index=False)

    for name in [
        "installments_payments",
        "POS_CASH_balance",
        "credit_card_balance",
    ]:
        pd.DataFrame({"SK_ID_CURR": []}).to_csv(raw_dir / f"{name}.csv", index=False)

    return raw_dir


class TestIngestPipeline:
    """Integration tests for data ingestion."""

    def test_ingest_creates_parquet(self, integration_config, sample_raw_data):
        """Test that ingestion creates parquet files."""
        from pipelines.ingest import ingest_application_data

        result = ingest_application_data(
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
            duckdb_path=integration_config.paths.duckdb_path,
        )

        assert result["train_rows"] == 100
        assert result["test_rows"] == 20
        assert result["total_rows"] == 120

        processed_dir = Path(integration_config.paths.data_processed)
        assert (processed_dir / "application_train.parquet").exists()
        assert (processed_dir / "application_test.parquet").exists()
        assert (processed_dir / "application_all.parquet").exists()

    def test_ingest_creates_duckdb_table(self, integration_config, sample_raw_data):
        """Test that ingestion creates DuckDB table."""
        from pipelines.ingest import ingest_application_data

        ingest_application_data(
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
            duckdb_path=integration_config.paths.duckdb_path,
        )

        con = duckdb.connect(integration_config.paths.duckdb_path)
        count = con.execute("SELECT COUNT(*) FROM application_all").fetchone()[0]
        con.close()

        assert count == 120


class TestAggregatePipeline:
    """Integration tests for feature aggregation."""

    def test_aggregate_creates_features(self, integration_config, sample_raw_data):
        """Test that aggregation creates feature table."""
        from pipelines.aggregate import run_aggregation_pipeline
        from pipelines.ingest import ingest_application_data

        ingest_application_data(
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
            duckdb_path=integration_config.paths.duckdb_path,
        )

        result = run_aggregation_pipeline(
            duckdb_path=integration_config.paths.duckdb_path,
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
        )

        assert "join" in result
        assert result["join"]["rows"] == 120

        features_path = (
            Path(integration_config.paths.data_processed) / "application_features.parquet"
        )
        assert features_path.exists()


class TestTrainPipeline:
    """Integration tests for model training."""

    @pytest.mark.slow
    def test_train_creates_model(self, integration_config, sample_raw_data):
        """Test that training creates model artifacts."""
        from pipelines.aggregate import run_aggregation_pipeline
        from pipelines.ingest import ingest_application_data
        from pipelines.train import train_model

        ingest_application_data(
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
            duckdb_path=integration_config.paths.duckdb_path,
        )

        run_aggregation_pipeline(
            duckdb_path=integration_config.paths.duckdb_path,
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
        )

        features_path = (
            Path(integration_config.paths.data_processed) / "application_features.parquet"
        )
        df = pd.read_parquet(features_path)

        metrics = train_model(
            df=df,
            config=integration_config,
            output_dir=integration_config.paths.models_dir,
        )

        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

        models_dir = Path(integration_config.paths.models_dir)
        model_files = list(models_dir.glob("model_*.joblib"))
        assert len(model_files) == 1


class TestScorePipeline:
    """Integration tests for batch scoring."""

    @pytest.mark.slow
    def test_score_creates_predictions(self, integration_config, sample_raw_data):
        """Test that scoring creates predictions."""
        from pipelines.aggregate import run_aggregation_pipeline
        from pipelines.ingest import ingest_application_data
        from pipelines.score import score_batch
        from pipelines.train import train_model

        ingest_application_data(
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
            duckdb_path=integration_config.paths.duckdb_path,
        )

        run_aggregation_pipeline(
            duckdb_path=integration_config.paths.duckdb_path,
            data_raw=integration_config.paths.data_raw,
            processed_dir=integration_config.paths.data_processed,
        )

        features_path = (
            Path(integration_config.paths.data_processed) / "application_features.parquet"
        )
        df = pd.read_parquet(features_path)

        train_model(
            df=df,
            config=integration_config,
            output_dir=integration_config.paths.models_dir,
        )

        stats = score_batch(
            input_path=str(features_path),
            output_path=str(Path(integration_config.paths.data_processed) / "predictions.csv"),
            model_dir=integration_config.paths.models_dir,
        )

        assert stats["n_predictions"] == 120
        assert 0 <= stats["mean_probability"] <= 1

        predictions_path = Path(integration_config.paths.data_processed) / "predictions.csv"
        assert predictions_path.exists()

        predictions_df = pd.read_csv(predictions_path)
        assert "SK_ID_CURR" in predictions_df.columns
        assert "default_probability" in predictions_df.columns
        assert "risk_level" in predictions_df.columns
