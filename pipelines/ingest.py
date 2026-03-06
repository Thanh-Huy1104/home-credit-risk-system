import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    duckdb_path = cfg["paths"]["duckdb_path"]
    data_raw = cfg["paths"]["data_raw"]
    processed_dir = cfg["paths"]["data_processed"]

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(duckdb_path)).mkdir(parents=True, exist_ok=True)

    train_csv = os.path.join(data_raw, "application_train.csv")
    test_csv = os.path.join(data_raw, "application_test.csv")

    print("Ingesting application_train...")
    df_train = pd.read_csv(train_csv)
    train_parquet = os.path.join(processed_dir, "application_train.parquet")
    df_train.to_parquet(train_parquet, index=False)
    print(f"   {len(df_train):,} rows")

    print("Ingesting application_test...")
    df_test = pd.read_csv(test_csv)
    test_parquet = os.path.join(processed_dir, "application_test.parquet")
    df_test.to_parquet(test_parquet, index=False)
    print(f"   {len(df_test):,} rows")

    print("Creating combined parquet...")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)

    df_train["is_train"] = 1
    df_test["is_train"] = 0
    df_test["TARGET"] = None

    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined_parquet = os.path.join(processed_dir, "application_all.parquet")
    combined.to_parquet(combined_parquet, index=False)
    print(f"   {len(combined):,} rows")

    con = duckdb.connect(duckdb_path)

    print("Creating DuckDB table...")
    con.execute(f"CREATE TABLE application_all AS SELECT * FROM '{combined_parquet}'")

    total = con.execute("SELECT COUNT(*) FROM application_all").fetchone()[0]
    print(f"   Combined: {total:,} rows -> application_all")

    con.close()
    print(f"DuckDB: {duckdb_path}")


if __name__ == "__main__":
    main()
