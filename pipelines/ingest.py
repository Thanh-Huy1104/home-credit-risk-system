"""Data ingestion pipeline: Load raw CSV files into DuckDB."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_application_data(
    data_raw: str,
    processed_dir: str,
    duckdb_path: str,
) -> dict[str, int]:
    """Ingest application train/test CSV files into DuckDB.

    Args:
        data_raw: Path to raw data directory.
        processed_dir: Path to processed data directory.
        duckdb_path: Path to DuckDB database.

    Returns:
        Dictionary with row counts.

    Raises:
        FileNotFoundError: If required CSV files not found.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(duckdb_path)).mkdir(parents=True, exist_ok=True)

    train_csv = os.path.join(data_raw, "application_train.csv")
    test_csv = os.path.join(data_raw, "application_test.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training data not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test data not found: {test_csv}")

    logger.info("Ingesting application_train...")
    df_train = pd.read_csv(train_csv)
    train_parquet = os.path.join(processed_dir, "application_train.parquet")
    df_train.to_parquet(train_parquet, index=False)
    logger.info(f"   {len(df_train):,} rows")

    logger.info("Ingesting application_test...")
    df_test = pd.read_csv(test_csv)
    test_parquet = os.path.join(processed_dir, "application_test.parquet")
    df_test.to_parquet(test_parquet, index=False)
    logger.info(f"   {len(df_test):,} rows")

    logger.info("Creating combined parquet...")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    df_train["is_train"] = 1
    df_test["is_train"] = 0
    df_test["TARGET"] = None

    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined_parquet = os.path.join(processed_dir, "application_all.parquet")
    combined.to_parquet(combined_parquet, index=False)
    logger.info(f"   {len(combined):,} rows")

    con = duckdb.connect(duckdb_path)

    logger.info("Creating DuckDB table...")
    con.execute(f"CREATE TABLE application_all AS SELECT * FROM '{combined_parquet}'")

    total = con.execute("SELECT COUNT(*) FROM application_all").fetchone()[0]
    logger.info(f"   Combined: {total:,} rows -> application_all")

    con.close()
    logger.info(f"DuckDB: {duckdb_path}")

    return {
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "total_rows": total,
    }


def main() -> int:
    """Main entry point for ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest raw data into DuckDB")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)

        result = ingest_application_data(
            data_raw=config.paths.data_raw,
            processed_dir=config.paths.data_processed,
            duckdb_path=config.paths.duckdb_path,
        )

        logger.info(f"Ingestion complete: {result}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
