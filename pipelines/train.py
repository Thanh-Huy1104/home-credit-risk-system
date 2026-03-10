"""Model training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import Config
from src.models.manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for training.

    Args:
        df: Raw feature DataFrame.

    Returns:
        Prepared DataFrame with proper types.
    """
    df = df.copy()

    df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype("category")

    return df


def train_model(
    df: pd.DataFrame,
    config: Config,
    output_dir: str,
) -> dict[str, float]:
    """Train XGBoost model for credit default prediction.

    Args:
        df: Feature DataFrame.
        config: Configuration object.
        output_dir: Directory to save model artifacts.

    Returns:
        Dictionary of evaluation metrics.

    Raises:
        ValueError: If data is invalid.
    """
    logger.info("Preparing data...")
    df = prepare_features(df)

    logger.info("Splitting data...")
    train_data = df[df["is_train"] == 1].copy()
    test_data = df[df["is_train"] == 0].copy()

    if len(train_data) == 0:
        raise ValueError("No training data found")
    if len(test_data) == 0:
        logger.warning("No test data found")

    cols_to_drop = ["SK_ID_CURR", "TARGET", "is_train"]

    X = train_data.drop(columns=cols_to_drop)
    y = train_data["TARGET"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.split.val_size,
        random_state=config.project.seed,
        stratify=y if config.split.stratify else None,
    )

    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    logger.info("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=config.model.n_estimators,
        learning_rate=config.model.learning_rate,
        max_depth=config.model.max_depth,
        min_child_weight=config.model.min_child_weight,
        colsample_bytree=config.model.colsample_bytree,
        subsample=config.model.subsample,
        scale_pos_weight=imbalance_ratio,
        enable_categorical=config.model.enable_categorical,
        eval_metric="auc",
        early_stopping_rounds=config.model.early_stopping_rounds,
        tree_method=config.model.tree_method,
        n_jobs=-1,
        random_state=config.model.random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    logger.info("Evaluating model...")
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)

    metrics = {
        "roc_auc": float(auc_score),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "imbalance_ratio": float(imbalance_ratio),
        "best_iteration": int(model.best_iteration),
    }

    logger.info(f"Validation ROC-AUC: {auc_score:.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manager = ModelManager(output_dir)

    params = {
        "n_estimators": config.model.n_estimators,
        "learning_rate": config.model.learning_rate,
        "max_depth": config.model.max_depth,
        "min_child_weight": config.model.min_child_weight,
        "colsample_bytree": config.model.colsample_bytree,
        "subsample": config.model.subsample,
        "tree_method": config.model.tree_method,
    }

    manager.save(
        model=model,
        features=list(X.columns),
        metrics=metrics,
        params=params,
        n_samples_train=len(X_train),
    )

    if len(test_data) > 0:
        logger.info("Generating predictions for test set...")
        X_test = test_data.drop(columns=cols_to_drop)
        test_preds = model.predict_proba(X_test)[:, 1]

        submission = pd.DataFrame(
            {
                "SK_ID_CURR": test_data["SK_ID_CURR"].astype(int),
                "TARGET": test_preds,
            }
        )

        submission_path = os.path.join(output_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        logger.info(f"Saved predictions to {submission_path}")

    return metrics


def main() -> int:
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--features", default=None, help="Path to features parquet")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(args.config)

        features_path = args.features
        if features_path is None:
            features_path = os.path.join(
                config.paths.data_processed, "application_features.parquet"
            )

        logger.info(f"Loading features from {features_path}...")
        df = pd.read_parquet(features_path)
        logger.info(f"Loaded {len(df):,} rows")

        metrics = train_model(
            df=df,
            config=config,
            output_dir=config.paths.models_dir,
        )

        metrics_path = os.path.join(config.paths.metrics_dir, "train_metrics.json")
        Path(config.paths.metrics_dir).mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        logger.info(f"Training complete: {metrics}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
