from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import model_registry


def test_load_champion_loads_through_the_alias(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MODEL_REGISTRY_NAME", "bike-sharing-demand")
    client = MagicMock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="3"
    )

    with patch.object(model_registry, "MlflowClient", return_value=client), \
            patch.object(
                model_registry.mlflow.sklearn,
                "load_model",
                return_value="fake-model",
            ) as load_model:
        serving = model_registry.load_champion()

    load_model.assert_called_once_with("models:/bike-sharing-demand@champion")
    assert serving.version == "3"
    assert serving.name == "bike-sharing-demand"
    assert serving.source == "mlflow"


def test_load_champion_rejects_an_alias_that_moved(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = [
        SimpleNamespace(version="3"),
        SimpleNamespace(version="4"),
    ]

    with patch.object(model_registry, "MlflowClient", return_value=client), \
            patch.object(model_registry.mlflow.sklearn, "load_model"):
        with pytest.raises(model_registry.ChampionUnavailable, match="moved"):
            model_registry.load_champion()


def test_load_champion_requires_tracking_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(model_registry.ChampionUnavailable):
        model_registry.load_champion()


def test_load_champion_wraps_registry_errors(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("no alias")

    with patch.object(model_registry, "MlflowClient", return_value=client):
        with pytest.raises(model_registry.ChampionUnavailable):
            model_registry.load_champion()
