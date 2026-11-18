from __future__ import annotations

from pathlib import Path

from app.api.main import BikeDemandFeatures
from app.web import prediction_data
from app.web.prediction_data import (
    MAX_FEELS_LIKE_C,
    MAX_HUMIDITY_PERCENT,
    MAX_TEMPERATURE_C,
    MAX_WIND_SPEED_KMH,
)


HOME_PAGE = Path(__file__).resolve().parents[1] / "app" / "web" / "Home.py"


def features(**weather: float) -> dict:
    return {
        "season": 3,
        "yr": 1,
        "mnth": 7,
        "hr": 17,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        **weather,
    }


def test_slider_maximums_give_features_the_api_accepts() -> None:
    row = features(
        temp=prediction_data.normalize_temperature(MAX_TEMPERATURE_C),
        atemp=prediction_data.normalize_feels_like_temperature(
            MAX_FEELS_LIKE_C
        ),
        hum=prediction_data.normalize_humidity(MAX_HUMIDITY_PERCENT),
        windspeed=prediction_data.normalize_wind_speed(MAX_WIND_SPEED_KMH),
    )

    validated = BikeDemandFeatures(**row)

    assert validated.temp == 1.0
    assert validated.atemp == 1.0
    assert validated.hum == 1.0
    assert validated.windspeed == 1.0


def test_temperature_uses_the_dataset_scale() -> None:
    # Readme of the dataset: temp is divided by 41, atemp by 50.
    assert prediction_data.normalize_temperature(20.5) == 0.5
    assert prediction_data.normalize_feels_like_temperature(25) == 0.5


def test_home_page_sliders_stop_at_the_dataset_maximums() -> None:
    source = HOME_PAGE.read_text(encoding="utf-8")

    for constant in (
        "MAX_TEMPERATURE_C",
        "MAX_FEELS_LIKE_C",
        "MAX_HUMIDITY_PERCENT",
        "MAX_WIND_SPEED_KMH",
    ):
        assert f"max_value={constant}" in source
    assert "max_value=50.0" not in source
