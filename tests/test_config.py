"""Tests for configuration management."""

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    Config,
    ModelConfig,
    PathsConfig,
    ProjectConfig,
    load_config,
)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "project": {"seed": 42},
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "duckdb_path": "data/duckdb/test.duckdb",
            "models_dir": "models/test",
            "metrics_dir": "metrics/test",
        },
        "dataset": {
            "tables": {"application_train": "csv"},
            "label_col": "TARGET",
        },
        "split": {"test_size": 0.2, "val_size": 0.2, "stratify": True},
        "model": {"n_estimators": 100, "learning_rate": 0.1},
    }


@pytest.fixture
def config_file(sample_config_dict, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProjectConfig()
        assert config.seed == 42

    def test_custom_seed(self):
        """Test custom seed value."""
        config = ProjectConfig(seed=123)
        assert config.seed == 123

    def test_invalid_seed(self):
        """Test validation of negative seed."""
        with pytest.raises(ValidationError):
            ProjectConfig(seed=-1)


class TestPathsConfig:
    """Tests for PathsConfig."""

    def test_default_values(self):
        """Test default paths."""
        config = PathsConfig()
        assert config.data_raw == "data/raw"
        assert config.data_processed == "data/processed"

    def test_env_expansion(self, monkeypatch):
        """Test environment variable expansion."""
        monkeypatch.setenv("PROJECT_ROOT", "/tmp/test")
        config = PathsConfig(data_raw="$PROJECT_ROOT/data")
        assert "/tmp/test" in config.data_raw


class TestConfig:
    """Tests for main Config class."""

    def test_from_yaml(self, config_file):
        """Test loading config from YAML file."""
        config = Config.from_yaml(config_file)
        assert config.project.seed == 42
        assert config.paths.data_raw == "data/raw"

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/config.yaml")

    def test_get_absolute_path(self, config_file):
        """Test getting absolute path."""
        config = Config.from_yaml(config_file)
        path = config.get_absolute_path("data_raw")
        assert path.is_absolute()

    def test_defaults(self):
        """Test default configuration values."""
        config = Config()
        assert config.project.seed == 42
        assert config.split.test_size == 0.2
        assert config.model.n_estimators == 1000


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_with_path(self, config_file):
        """Test loading with explicit path."""
        config = load_config(config_file)
        assert config is not None

    def test_load_with_env_var(self, config_file, monkeypatch):
        """Test loading via environment variable."""
        monkeypatch.setenv("CONFIG_PATH", str(config_file))
        config = load_config()
        assert config is not None


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.n_estimators == 1000
        assert config.learning_rate == 0.05
        assert config.max_depth == 4

    def test_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValidationError):
            ModelConfig(learning_rate=1.5)

        with pytest.raises(ValidationError):
            ModelConfig(max_depth=20)
