"""Configuration management with Pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    """Project-level configuration."""

    seed: int = Field(default=42, ge=0)


class PathsConfig(BaseModel):
    """Path configuration for data, models, and outputs."""

    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    duckdb_path: str = "data/duckdb/home_credit.duckdb"
    models_dir: str = "models/artifacts"
    metrics_dir: str = "models/metrics"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand environment variables in paths."""
        return os.path.expandvars(v)


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    tables: dict[str, str] = Field(
        default_factory=lambda: {
            "application_train": "csv",
            "application_test": "csv",
        }
    )
    label_col: str = "TARGET"


class SplitConfig(BaseModel):
    """Train/test split configuration."""

    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.2, gt=0, lt=1)
    stratify: bool = True


class ModelConfig(BaseModel):
    """Model hyperparameters."""

    n_estimators: int = Field(default=1000, ge=1)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_depth: int = Field(default=4, ge=1, le=12)
    min_child_weight: int = Field(default=30, ge=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    early_stopping_rounds: int = Field(default=50, ge=1)
    tree_method: str = Field(default="hist")
    enable_categorical: bool = True
    random_state: int = Field(default=42)


class Config(BaseModel):
    """Main configuration model."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Validated Config instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config validation fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def get_absolute_path(self, key: str) -> Path:
        """Get absolute path for a given path key.

        Args:
            key: Key from paths configuration.

        Returns:
            Absolute Path object.
        """
        relative_path = getattr(self.paths, key)
        return Path(relative_path).resolve()


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration with optional override path.

    Args:
        config_path: Optional path to config file.
                    Defaults to configs/base.yaml or env var CONFIG_PATH.

    Returns:
        Validated Config instance.
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "configs/base.yaml")

    return Config.from_yaml(config_path)
