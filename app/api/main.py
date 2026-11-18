from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Base, Prediction
from app.database.schema import ensure_monitoring_schema
from app.database.session import engine, get_db
from app.model_registry import (
    ChampionUnavailable,
    ServingModel,
    load_champion,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "bike_demand_model.pkl"

FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
MODEL_NAME = "bike_demand_hist_gradient_boosting_v1"
LOCAL_MODEL_VERSION = "local-baseline"

serving: ServingModel | None = None


class BikeDemandFeatures(BaseModel):
    season: int = Field(..., ge=1, le=4)
    yr: int = Field(..., ge=0, le=1)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0, le=1)
    atemp: float = Field(..., ge=0, le=1)
    hum: float = Field(..., ge=0, le=1)
    windspeed: float = Field(..., ge=0, le=1)


class PredictionRequest(BaseModel):
    features: list[BikeDemandFeatures] = Field(..., min_length=1)
    source: str = Field(default="webapp", pattern="^(webapp|scheduled)$")


class PredictionItem(BaseModel):
    prediction: float
    features: BikeDemandFeatures


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[PredictionItem]


class PastPredictionItem(BaseModel):
    id: int
    prediction: float
    source: str
    model_name: str
    model_version: str | None
    features: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class PastPredictionsResponse(BaseModel):
    predictions: list[PastPredictionItem]


class ModelStatusResponse(BaseModel):
    model_name: str
    model_version: str
    source: str


def load_local_model() -> ServingModel:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    return ServingModel(
        model=joblib.load(MODEL_PATH),
        name=MODEL_NAME,
        version=LOCAL_MODEL_VERSION,
        source="local",
    )


def load_startup_model() -> ServingModel:
    try:
        return load_champion()
    except ChampionUnavailable as exc:
        # Keeps Defense 1 working on a fresh stack with an empty registry.
        logger.warning("Serving the local model: %s", exc)
        return load_local_model()


def get_serving_model() -> ServingModel:
    if serving is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return serving


def describe_model(current: ServingModel) -> ModelStatusResponse:
    return ModelStatusResponse(
        model_name=current.name,
        model_version=current.version,
        source=current.source,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global serving
    serving = load_startup_model()
    Base.metadata.create_all(bind=engine)
    ensure_monitoring_schema(engine)
    yield


app = FastAPI(
    title="Bike Sharing Demand API",
    description=(
        "Predict hourly bike rental demand from weather and calendar data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/model-status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    return describe_model(get_serving_model())


@app.post("/reload-model", response_model=ModelStatusResponse)
def reload_model() -> ModelStatusResponse:
    global serving
    try:
        loaded = load_champion()
    except ChampionUnavailable as exc:
        # The model already in memory keeps serving predictions.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    serving = loaded
    return describe_model(loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    # A single reference keeps each batch on one model during a reload.
    current = get_serving_model()

    rows = [item.model_dump() for item in request.features]
    data = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    try:
        raw_predictions = current.model.predict(data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        ) from exc

    predictions = []
    for prediction, features in zip(raw_predictions, request.features):
        prediction_value = float(prediction)
        predictions.append(
            PredictionItem(prediction=prediction_value, features=features)
        )
        db.add(
            Prediction(
                prediction=prediction_value,
                source=request.source,
                model_name=current.name,
                model_version=current.version,
                features=features.model_dump(),
            )
        )

    db.commit()

    return PredictionResponse(
        model_name=current.name,
        model_version=current.version,
        predictions=predictions,
    )


@app.get("/past-predictions", response_model=PastPredictionsResponse)
def past_predictions(
    source: str = Query(default="all", pattern="^(webapp|scheduled|all)$"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> PastPredictionsResponse:
    query = select(Prediction)

    if source != "all":
        query = query.where(Prediction.source == source)
    if start_date is not None:
        query = query.where(Prediction.created_at >= start_date)
    if end_date is not None:
        query = query.where(Prediction.created_at <= end_date)

    query = query.order_by(Prediction.created_at.desc()).limit(limit)
    rows = db.execute(query).scalars().all()

    return PastPredictionsResponse(
        predictions=[PastPredictionItem.model_validate(row) for row in rows]
    )
