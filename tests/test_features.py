"""Tests for feature aggregations."""

import tempfile

import duckdb
import pandas as pd
import pytest

from src.features.aggregations import (
    aggregate_bureau,
    aggregate_previous_application,
)


@pytest.fixture
def temp_db():
    """Create temporary DuckDB connection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.duckdb"
        con = duckdb.connect(db_path)
        yield con, tmpdir
        con.close()


@pytest.fixture
def sample_bureau_csv(tmp_path):
    """Create sample bureau CSV file."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001, 100002, 100003],
            "SK_ID_BUREAU": [1, 2, 3, 4],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Closed"],
            "AMT_CREDIT_SUM_DEBT": [50000, 0, 30000, 0],
            "AMT_CREDIT_SUM": [100000, 50000, 80000, 60000],
            "AMT_CREDIT_SUM_OVERDUE": [0, 0, 1000, 0],
            "CREDIT_DAY_OVERDUE": [0, 0, 30, 0],
            "AMT_CREDIT_MAX_OVERDUE": [0, 0, 500, 0],
            "AMT_ANNUITY": [5000, 0, 4000, 0],
            "DAYS_CREDIT": [-100, -200, -150, -300],
        }
    )
    csv_path = tmp_path / "bureau.csv"
    df.to_csv(csv_path, index=False)
    return str(tmp_path)


@pytest.fixture
def sample_prev_app_csv(tmp_path):
    """Create sample previous_application CSV file."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100001, 100002],
            "SK_ID_PREV": [1, 2, 3],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "AMT_APPLICATION": [200000, 150000, 300000],
            "AMT_CREDIT": [200000, 0, 300000],
            "AMT_ANNUITY": [10000, 0, 15000],
            "DAYS_DECISION": [-100, -200, -150],
            "CNT_PAYMENT": [24, 0, 36],
        }
    )
    csv_path = tmp_path / "previous_application.csv"
    df.to_csv(csv_path, index=False)
    return str(tmp_path)


class TestAggregateBureau:
    """Tests for bureau aggregation."""

    def test_aggregate_bureau(self, sample_bureau_csv):
        """Test bureau aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.duckdb"
            con = duckdb.connect(db_path)

            result = aggregate_bureau(con, sample_bureau_csv)

            assert "table" in result
            assert "rows" in result

            table_check = con.execute("SELECT COUNT(*) FROM bureau_agg").fetchone()[0]
            assert table_check > 0

            sample_row = con.execute("SELECT * FROM bureau_agg LIMIT 1").fetchdf()
            assert "bureau_loan_count" in sample_row.columns
            assert "bureau_total_debt" in sample_row.columns

            con.close()


class TestAggregatePreviousApplication:
    """Tests for previous application aggregation."""

    def test_aggregate_prev_app(self, sample_prev_app_csv):
        """Test previous application aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.duckdb"
            con = duckdb.connect(db_path)

            result = aggregate_previous_application(con, sample_prev_app_csv)

            assert "rows" in result

            table_check = con.execute("SELECT COUNT(*) FROM prev_app_agg").fetchone()[0]
            assert table_check > 0

            sample_row = con.execute("SELECT * FROM prev_app_agg LIMIT 1").fetchdf()
            assert "prev_app_count" in sample_row.columns
            assert "prev_app_refused" in sample_row.columns

            con.close()


class TestAggregationLogic:
    """Tests for aggregation logic."""

    def test_count_aggregation(self, sample_bureau_csv):
        """Test that counts are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.duckdb"
            con = duckdb.connect(db_path)

            aggregate_bureau(con, sample_bureau_csv)

            result = con.execute("""
                SELECT SK_ID_CURR, bureau_loan_count
                FROM bureau_agg
                WHERE SK_ID_CURR = 100001
            """).fetchone()

            assert result is not None
            assert result[1] == 2

            con.close()

    def test_null_handling(self):
        """Test that aggregations handle nulls properly."""
        df = pd.DataFrame(
            {
                "SK_ID_CURR": [100001, 100001],
                "value": [10, None],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/test.csv"
            df.to_csv(csv_path, index=False)

            db_path = f"{tmpdir}/test.duckdb"
            con = duckdb.connect(db_path)

            con.execute(f"""
                CREATE TABLE test_agg AS
                SELECT
                    SK_ID_CURR,
                    SUM(value) as total,
                    AVG(value) as avg_value
                FROM read_csv_auto('{csv_path}')
                GROUP BY SK_ID_CURR
            """)

            result = con.execute("SELECT total, avg_value FROM test_agg").fetchone()

            assert result[0] == 10
            assert result[1] == 10

            con.close()
