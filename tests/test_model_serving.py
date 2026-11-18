from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import model_registry
from app.api import main
from app.model_registry import ChampionUnavailable, ServingModel


PAYLOAD = {
    "features": [
        {
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": 8,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.24,
            "atemp": 0.28,
            "hum": 0.6,
            "windspeed": 0.2,
        }
    ],
    "source": "webapp",
}


class FixedModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, data: pd.DataFrame) -> list[float]:
        return [self.value] * len(data)


def champion(version: str, value: float = 50.0) -> ServingModel:
    return ServingModel(
        model=FixedModel(value),
        name="BikeDemand",
        version=version,
        source="mlflow",
    )


def champion_is(
    monkeypatch: pytest.MonkeyPatch,
    loaded: ServingModel,
) -> None:
    monkeypatch.setattr(main, "load_champion", lambda: loaded)


def champion_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> ServingModel:
        raise ChampionUnavailable("registry is empty")

    monkeypatch.setattr(main, "load_champion", fail)


def test_api_starts_on_the_champion(monkeypatch: pytest.MonkeyPatch) -> None:
    champion_is(monkeypatch, champion("7"))

    with TestClient(main.app) as client:
        response = client.get("/model-status")

    assert response.json() == {
        "model_name": "BikeDemand",
        "model_version": "7",
        "source": "mlflow",
    }


def test_api_falls_back_to_local_model_without_a_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_is_unavailable(monkeypatch)

    with TestClient(main.app) as client:
        status_body = client.get("/model-status").json()
        prediction = client.post("/predict", json=PAYLOAD)

    assert status_body["model_version"] == "local-baseline"
    assert status_body["source"] == "local"
    assert prediction.status_code == 200
    assert prediction.json()["predictions"][0]["prediction"] > 0


def test_each_prediction_stores_the_serving_model_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_is(monkeypatch, champion("7", value=123.5))

    with TestClient(main.app) as client:
        prediction = client.post("/predict", json=PAYLOAD).json()
        history = client.get("/past-predictions", params={"limit": 1}).json()

    assert prediction["model_version"] == "7"
    assert prediction["predictions"][0]["prediction"] == 123.5
    assert history["predictions"][0]["model_version"] == "7"


def test_reload_switches_to_the_new_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_is(monkeypatch, champion("7", value=10.0))

    with TestClient(main.app) as client:
        champion_is(monkeypatch, champion("8", value=20.0))
        reload_response = client.post("/reload-model")
        prediction = client.post("/predict", json=PAYLOAD).json()

    assert reload_response.status_code == 200
    assert reload_response.json()["model_version"] == "8"
    assert prediction["model_version"] == "8"
    assert prediction["predictions"][0]["prediction"] == 20.0


def test_failed_reload_keeps_the_current_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    champion_is(monkeypatch, champion("7"))

    with TestClient(main.app) as client:
        champion_is_unavailable(monkeypatch)
        reload_response = client.post("/reload-model")
        status_body = client.get("/model-status").json()

    assert reload_response.status_code == 503
    assert status_body["model_version"] == "7"


def test_load_champion_needs_a_tracking_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ChampionUnavailable):
        model_registry.load_champion()


def test_load_champion_uses_the_champion_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_uris: list[str] = []

    class FakeClient:
        def get_model_version_by_alias(
            self,
            name: str,
            alias: str,
        ) -> Any:
            assert (name, alias) == ("bike-sharing-demand", "champion")
            return type("Version", (), {"version": "3"})()

    def fake_load_model(uri: str) -> FixedModel:
        loaded_uris.append(uri)
        return FixedModel(1.0)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.delenv("MODEL_REGISTRY_NAME", raising=False)
    monkeypatch.setattr(model_registry, "MlflowClient", FakeClient)
    monkeypatch.setattr(model_registry.mlflow, "set_tracking_uri", print)
    monkeypatch.setattr(model_registry.mlflow, "set_registry_uri", print)
    monkeypatch.setattr(
        model_registry.mlflow.sklearn,
        "load_model",
        fake_load_model,
    )

    loaded = model_registry.load_champion()

    assert loaded_uris == ["models:/bike-sharing-demand@champion"]
    assert loaded.version == "3"
    assert loaded.source == "mlflow"


def test_load_champion_reports_registry_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> Any:
            raise ConnectionError("tracking server is down")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(model_registry, "MlflowClient", BrokenClient)
    monkeypatch.setattr(model_registry.mlflow, "set_tracking_uri", print)
    monkeypatch.setattr(model_registry.mlflow, "set_registry_uri", print)

    with pytest.raises(ChampionUnavailable, match="tracking server is down"):
        model_registry.load_champion()
