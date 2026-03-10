"""Batch scoring pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.config import Config
from src.models.manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def score_batch(
    input_path: str,
    output_path: str,
    model_dir: str,
    id_col: str = "SK_ID_CURR",
) -> dict[str, int | float]:
    """Score a batch of applications.

    Args:
        input_path: Path to input parquet file.
        output_path: Path to save predictions.
        model_dir: Directory containing model artifacts.
        id_col: Name of ID column.

    Returns:
        Dictionary with scoring statistics.

    Raises:
        FileNotFoundError: If input or model not found.
        ValueError: If data validation fails.
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows")

    manager = ModelManager(model_dir)
    model, metadata = manager.load()
    logger.info(f"Loaded model version: {metadata.version}")

    drop_cols = [id_col]
    if "TARGET" in df.columns:
        drop_cols.append("TARGET")
    if "is_train" in df.columns:
        drop_cols.append("is_train")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in X.select_dtypes(include=["object", "str"]).columns:
        X[col] = X[col].astype("category")

    manager.validate_features(X)
    X = X[metadata.features]

    logger.info("Generating predictions...")
    probabilities = manager.predict_proba(X)

    result = pd.DataFrame(
        {
            id_col: df[id_col].astype(int),
            "default_probability": probabilities,
        }
    )

    result["risk_level"] = pd.cut(
        probabilities,
        bins=[0, 0.2, 0.5, 1.0],
        labels=["low", "medium", "high"],
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    stats = {
        "n_predictions": int(len(result)),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(pd.Series(probabilities).median()),
        "n_high_risk": int((probabilities >= 0.5).sum()),
        "n_medium_risk": int(((probabilities >= 0.2) & (probabilities < 0.5)).sum()),
        "n_low_risk": int((probabilities < 0.2).sum()),
    }

    return stats


def main() -> int:
    """Main entry point for scoring pipeline."""
    parser = argparse.ArgumentParser(description="Batch scoring for credit risk")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--input", default=None, help="Path to input parquet")
    parser.add_argument("--output", default=None, help="Path to output CSV")
    parser.add_argument("--model-version", default=None, help="Model version to use")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)

        input_path = args.input
        if input_path is None:
            input_path = os.path.join(config.paths.data_processed, "application_features.parquet")

        output_path = args.output
        if output_path is None:
            output_path = os.path.join(config.paths.data_processed, "predictions.csv")

        stats = score_batch(
            input_path=input_path,
            output_path=output_path,
            model_dir=config.paths.models_dir,
        )

        logger.info(f"Scoring complete: {stats}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Scoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
