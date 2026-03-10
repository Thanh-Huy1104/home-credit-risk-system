"""Model management for training, saving, and loading models."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""

    version: str
    created_at: str
    model_type: str
    features: list[str]
    target: str
    metrics: dict[str, float]
    params: dict[str, Any]
    n_features: int
    n_samples_train: int


class ModelManager:
    """Manages model lifecycle: training, saving, loading, versioning."""

    def __init__(self, models_dir: str | Path = "models/artifacts"):
        """Initialize model manager.

        Args:
            models_dir: Directory to store model artifacts.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model: Any = None
        self.metadata: ModelMetadata | None = None

    def save(
        self,
        model: Any,
        features: list[str],
        metrics: dict[str, float],
        params: dict[str, Any],
        version: str | None = None,
        target: str = "TARGET",
        n_samples_train: int = 0,
    ) -> Path:
        """Save model with metadata.

        Args:
            model: Trained model object.
            features: List of feature names.
            metrics: Dictionary of evaluation metrics.
            params: Model hyperparameters.
            version: Model version string. Auto-generated if None.
            target: Target column name.
            n_samples_train: Number of training samples.

        Returns:
            Path to saved model artifact.
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        self.metadata = ModelMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=type(model).__name__,
            features=features,
            target=target,
            metrics=metrics,
            params=params,
            n_features=len(features),
            n_samples_train=n_samples_train,
        )

        model_path = self.models_dir / f"model_{version}.joblib"
        metadata_path = self.models_dir / f"model_{version}_metadata.json"

        joblib.dump(model, model_path)
        with open(metadata_path, "w") as f:
            f.write(self.metadata.model_dump_json(indent=2))

        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved metadata to {metadata_path}")

        self.model = model
        return model_path

    def load(self, version: str | None = None) -> tuple[Any, ModelMetadata]:
        """Load model and metadata.

        Args:
            version: Model version to load. Loads latest if None.

        Returns:
            Tuple of (model, metadata).

        Raises:
            FileNotFoundError: If no model found.
        """
        if version is None:
            version = self._get_latest_version()
            if version is None:
                raise FileNotFoundError(f"No models found in {self.models_dir}")

        model_path = self.models_dir / f"model_{version}.joblib"
        metadata_path = self.models_dir / f"model_{version}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata = ModelMetadata(**json.load(f))

        logger.info(f"Loaded model {version} from {model_path}")
        return self.model, self.metadata

    def _get_latest_version(self) -> str | None:
        """Get the latest model version.

        Returns:
            Latest version string or None if no models found.
        """
        model_files = sorted(self.models_dir.glob("model_v*.joblib"), reverse=True)
        if not model_files:
            return None

        filename = model_files[0].stem
        return filename.replace("model_", "")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for input features.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of probabilities.

        Raises:
            ValueError: If model not loaded.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes for input features.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model not loaded.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        return self.model.predict(X)

    def validate_features(self, X: pd.DataFrame) -> bool:
        """Validate that input features match expected features.

        Args:
            X: Feature DataFrame.

        Returns:
            True if valid.

        Raises:
            ValueError: If features don't match.
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load() first.")

        missing = set(self.metadata.features) - set(X.columns)
        extra = set(X.columns) - set(self.metadata.features)

        if missing:
            raise ValueError(f"Missing features: {missing}")
        if extra:
            logger.warning(f"Extra features will be ignored: {extra}")

        return True


def save_model(
    model: Any,
    path: str | Path,
    features: list[str],
    metrics: dict[str, float],
    params: dict[str, Any],
    **kwargs: Any,
) -> Path:
    """Convenience function to save a model.

    Args:
        model: Trained model.
        path: Base path for saving.
        features: Feature names.
        metrics: Evaluation metrics.
        params: Model parameters.
        **kwargs: Additional arguments for ModelManager.save().

    Returns:
        Path to saved model.
    """
    manager = ModelManager(Path(path).parent)
    return manager.save(model, features, metrics, params, **kwargs)


def load_model(path: str | Path) -> tuple[Any, ModelMetadata]:
    """Convenience function to load a model.

    Args:
        path: Path to model directory.

    Returns:
        Tuple of (model, metadata).
    """
    manager = ModelManager(path)
    return manager.load()
