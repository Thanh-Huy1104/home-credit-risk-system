"""FastAPI application for credit risk prediction."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Config, load_config
from src.models.manager import ModelManager

logger = logging.getLogger(__name__)

model_manager: ModelManager | None = None
config: Config | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_manager, config

    config = load_config()
    model_manager = ModelManager(config.paths.models_dir)

    try:
        model_manager.load()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("No model found. Train a model first using 'make train'")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Home Credit Risk API",
    description="API for predicting credit default risk",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Request schema for prediction."""

    SK_ID_CURR: int = Field(..., description="Application ID")
    features: dict[str, Any] = Field(..., description="Feature values")


class PredictResponse(BaseModel):
    """Response schema for prediction."""

    SK_ID_CURR: int
    default_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="low, medium, or high")
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str | None


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status."""
    model_version = None
    if model_manager and model_manager.metadata:
        model_version = model_manager.metadata.version

    return HealthResponse(
        status="healthy" if model_manager and model_manager.model else "degraded",
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_version=model_version,
    )


@app.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Check if service is ready to accept requests."""
    if model_manager is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict default probability for an application.

    Args:
        request: Prediction request with application ID and features.

    Returns:
        Prediction response with probability and risk level.

    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model_manager is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = pd.DataFrame([request.features])

        model_manager.validate_features(X)

        X = X[model_manager.metadata.features]

        proba = model_manager.predict_proba(X)[0]

        if proba < 0.2:
            risk_level = "low"
        elif proba < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        return PredictResponse(
            SK_ID_CURR=request.SK_ID_CURR,
            default_probability=round(float(proba), 4),
            risk_level=risk_level,
            model_version=model_manager.metadata.version,
            timestamp=datetime.now().isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    """Batch prediction for multiple applications.

    Args:
        requests: List of prediction requests.

    Returns:
        List of prediction responses.

    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model_manager is None or model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Maximum batch size is 1000")

    results = []
    for request in requests:
        result = await predict(request)
        results.append(result)

    return results


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    """Get model information."""
    if model_manager is None or model_manager.metadata is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    return {
        "version": model_manager.metadata.version,
        "created_at": model_manager.metadata.created_at,
        "model_type": model_manager.metadata.model_type,
        "n_features": model_manager.metadata.n_features,
        "metrics": model_manager.metadata.metrics,
        "features": model_manager.metadata.features[:10],
    }


def create_app(config_path: str | None = None) -> FastAPI:
    """Create FastAPI application with custom config.

    Args:
        config_path: Path to configuration file.

    Returns:
        FastAPI application instance.
    """
    if config_path:
        os.environ["CONFIG_PATH"] = config_path

    return app
