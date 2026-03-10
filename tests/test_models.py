"""Tests for model management."""


import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.manager import ModelManager, ModelMetadata, load_model, save_model


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory."""
    return tmp_path / "models"


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    model = DummyClassifier(strategy="prior")
    model.fit(X, y)
    return model


@pytest.fixture
def sample_features():
    """Sample feature names."""
    return ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {"accuracy": 0.85, "roc_auc": 0.90}


@pytest.fixture
def sample_params():
    """Sample model parameters."""
    return {"strategy": "prior"}


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = ModelMetadata(
            version="v1.0.0",
            created_at="2024-01-01T00:00:00",
            model_type="DummyClassifier",
            features=["a", "b"],
            target="TARGET",
            metrics={"auc": 0.9},
            params={},
            n_features=2,
            n_samples_train=1000,
        )
        assert metadata.version == "v1.0.0"
        assert metadata.n_features == 2


class TestModelManager:
    """Tests for ModelManager."""

    def test_init(self, temp_models_dir):
        """Test manager initialization."""
        manager = ModelManager(temp_models_dir)
        assert manager.models_dir == temp_models_dir
        assert manager.model is None
        assert manager.metadata is None

    def test_save_model(
        self,
        temp_models_dir,
        sample_model,
        sample_features,
        sample_metrics,
        sample_params,
    ):
        """Test saving a model."""
        manager = ModelManager(temp_models_dir)

        model_path = manager.save(
            model=sample_model,
            features=sample_features,
            metrics=sample_metrics,
            params=sample_params,
            version="v1.0.0",
        )

        assert model_path.exists()
        assert "v1.0.0" in str(model_path)
        assert manager.metadata is not None
        assert manager.metadata.version == "v1.0.0"

    def test_load_model(self, temp_models_dir, sample_model, sample_features):
        """Test loading a model."""
        manager = ModelManager(temp_models_dir)

        manager.save(
            model=sample_model,
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
            version="v1.0.0",
        )

        loaded_model, metadata = manager.load("v1.0.0")

        assert loaded_model is not None
        assert metadata.version == "v1.0.0"
        assert metadata.features == sample_features

    def test_load_latest(self, temp_models_dir, sample_model, sample_features):
        """Test loading latest model."""
        manager = ModelManager(temp_models_dir)

        manager.save(
            model=sample_model,
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
            version="v1.0.0",
        )

        manager.save(
            model=sample_model,
            features=sample_features,
            metrics={"auc": 0.91},
            params={},
            version="v2.0.0",
        )

        model, metadata = manager.load()
        assert metadata.version == "v2.0.0"

    def test_load_not_found(self, temp_models_dir):
        """Test loading non-existent model."""
        manager = ModelManager(temp_models_dir)

        with pytest.raises(FileNotFoundError):
            manager.load("v999.0.0")

    def test_predict(self, temp_models_dir, sample_model, sample_features):
        """Test model prediction."""
        manager = ModelManager(temp_models_dir)

        manager.save(
            model=sample_model,
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
        )

        X = pd.DataFrame(np.random.randn(10, 5), columns=sample_features)

        predictions = manager.predict(X)
        assert len(predictions) == 10

        probabilities = manager.predict_proba(X)
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)

    def test_validate_features(self, temp_models_dir, sample_model, sample_features):
        """Test feature validation."""
        manager = ModelManager(temp_models_dir)

        manager.save(
            model=sample_model,
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
        )

        X_valid = pd.DataFrame(np.random.randn(5, 5), columns=sample_features)
        assert manager.validate_features(X_valid) is True

        X_missing = pd.DataFrame(
            np.random.randn(5, 3),
            columns=["a", "b", "c"],
        )
        with pytest.raises(ValueError, match="Missing features"):
            manager.validate_features(X_missing)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_save_model(self, temp_models_dir, sample_model, sample_features):
        """Test save_model convenience function."""
        model_path = save_model(
            model=sample_model,
            path=temp_models_dir / "test.joblib",
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
        )
        assert model_path.exists()

    def test_load_model(self, temp_models_dir, sample_model, sample_features):
        """Test load_model convenience function."""
        save_model(
            model=sample_model,
            path=temp_models_dir / "test.joblib",
            features=sample_features,
            metrics={"auc": 0.9},
            params={},
            version="v1.0.0",
        )

        model, metadata = load_model(temp_models_dir)
        assert metadata.version == "v1.0.0"
