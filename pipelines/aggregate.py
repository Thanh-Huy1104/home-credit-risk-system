"""Feature aggregation pipeline: Create aggregated features from raw tables."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import duckdb

from src.config import Config
from src.features.aggregations import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_credit_card,
    aggregate_installments_payments,
    aggregate_pos_cash,
    aggregate_previous_application,
    join_all_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_aggregation_pipeline(
    duckdb_path: str,
    data_raw: str,
    processed_dir: str,
) -> dict[str, int]:
    """Run the complete feature aggregation pipeline.

    Args:
        duckdb_path: Path to DuckDB database.
        data_raw: Path to raw data directory.
        processed_dir: Path to processed data directory.

    Returns:
        Dictionary with aggregation results.

    Raises:
        FileNotFoundError: If required tables not found.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(duckdb_path)).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(duckdb_path)

    try:
        tables_check = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name='application_all'"
        ).fetchone()
        if not tables_check:
            raise FileNotFoundError("application_all table not found. Run ingest.py first.")

        results = {}

        results["bureau"] = aggregate_bureau(con, data_raw)
        results["bureau_balance"] = aggregate_bureau_balance(con, data_raw)
        results["previous_application"] = aggregate_previous_application(con, data_raw)
        results["installments"] = aggregate_installments_payments(con, data_raw)
        results["pos_cash"] = aggregate_pos_cash(con, data_raw)
        results["credit_card"] = aggregate_credit_card(con, data_raw)

        results["join"] = join_all_features(con, processed_dir)

        logger.info("\nWriting feature table to parquet...")
        con.execute(
            f"COPY (SELECT * FROM application_features) TO '{os.path.join(processed_dir, 'application_features.parquet')}' (FORMAT PARQUET)"
        )

        logger.info("Done!")
        return results

    finally:
        con.close()


def main() -> int:
    """Main entry point for aggregation pipeline."""
    parser = argparse.ArgumentParser(description="Aggregate features from raw tables")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)

        result = run_aggregation_pipeline(
            duckdb_path=config.paths.duckdb_path,
            data_raw=config.paths.data_raw,
            processed_dir=config.paths.data_processed,
        )

        logger.info(f"Aggregation complete: {result['join']}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Aggregation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
