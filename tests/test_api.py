from __future__ import annotations
from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import patch
from app.api import main
from app.model_registry import ChampionUnavailable, ServingModel


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_returns_prediction() -> None:
    payload = {
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

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "bike_demand_hist_gradient_boosting_v1"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["prediction"] > 0


def test_model_status_reports_local_model_without_registry() -> None:
    with TestClient(app) as client:
        response = client.get("/model-status")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "bike_demand_hist_gradient_boosting_v1",
        "model_version": "local-baseline",
        "source": "local",
    }


def test_reload_model_switches_to_new_champion() -> None:
    with TestClient(app) as client:
        champion = ServingModel(
            model=main.serving.model,
            name="bike-sharing-demand",
            version="2",
            source="mlflow",
        )
        with patch.object(main, "load_champion", return_value=champion):
            response = client.post("/reload-model")

        assert response.status_code == 200
        assert response.json() == {
            "model_name": "bike-sharing-demand",
            "model_version": "2",
            "source": "mlflow",
        }
        assert client.get("/model-status").json()["model_version"] == "2"


def test_reload_model_keeps_current_model_on_failure() -> None:
    with TestClient(app) as client:
        before = client.get("/model-status").json()
        with patch.object(
            main,
            "load_champion",
            side_effect=ChampionUnavailable("registry down"),
        ):
            response = client.post("/reload-model")

        assert response.status_code == 503
        assert "registry down" in response.json()["detail"]
        assert client.get("/model-status").json() == before


def test_predict_returns_and_stores_model_version() -> None:
    payload = {
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

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        history = client.get("/past-predictions").json()["predictions"]

    assert response.json()["model_version"] == "local-baseline"
    assert history[0]["model_version"] == "local-baseline"
