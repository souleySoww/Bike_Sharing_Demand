from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


DEFAULT_REGISTRY_NAME = "bike-sharing-demand"
CHAMPION_ALIAS = "champion"

# MLflow retries for minutes by default, longer than the API healthcheck allows
# when the tracking server is down at startup.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


class ChampionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ServingModel:
    model: Any
    name: str
    version: str
    source: str


def registry_name() -> str:
    return os.getenv("MODEL_REGISTRY_NAME", DEFAULT_REGISTRY_NAME)


def load_champion() -> ServingModel:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ChampionUnavailable("MLFLOW_TRACKING_URI is not configured")

    name = registry_name()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = MlflowClient()
        version = client.get_model_version_by_alias(
            name,
            CHAMPION_ALIAS,
        ).version
        model = mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
        # A promotion between the lookup and the download would label the
        # predictions with the wrong version.
        if client.get_model_version_by_alias(
            name,
            CHAMPION_ALIAS,
        ).version != version:
            raise ChampionUnavailable("@champion moved during loading")
    except Exception as exc:
        raise ChampionUnavailable(
            f"Cannot load models:/{name}@{CHAMPION_ALIAS}: {exc}"
        ) from exc
    return ServingModel(
        model=model,
        name=name,
        version=str(version),
        source="mlflow",
    )
