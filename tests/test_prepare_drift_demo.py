from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.training.pipeline import load_history, split_history
from app.validation.bike_data import (
    REQUIRED_COLUMNS,
    collect_invalid_row_indices,
)
from scripts.prepare_drift_demo import prepare_drift_files


def source_dataset(row_count: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instant": index + 1,
                "dteday": "2012-01-01",
                "season": 1 + index % 4,
                "yr": 1,
                "mnth": 1 + index % 12,
                "hr": index % 24,
                "holiday": 0,
                "weekday": index % 7,
                "workingday": 1,
                "weathersit": 1 + index % 2,
                "temp": round(0.2 + (index % 50) / 100, 2),
                "atemp": round(0.2 + (index % 50) / 100, 2),
                "hum": round(0.4 + (index % 50) / 100, 2),
                "windspeed": 0.2,
                "cnt": 100 + index % 300,
            }
            for index in range(row_count)
        ]
    )


def generate(tmp_path: Path) -> tuple[Path, list[Path], pd.DataFrame]:
    dataset_path = tmp_path / "hour.csv"
    source = source_dataset()
    source.to_csv(dataset_path, index=False)
    files = prepare_drift_files(
        dataset_path,
        tmp_path / "drift",
        files=4,
        rows_per_file=40,
    )
    return dataset_path, files, source


def test_drift_rows_come_only_from_the_test_window(tmp_path: Path) -> None:
    dataset_path, files, _ = generate(tmp_path)
    _, test = split_history(load_history(dataset_path))
    rows = pd.concat([pd.read_csv(path) for path in files])

    assert len(files) == 4
    assert set(rows["instant"]) <= set(test["instant"])
    assert rows["instant"].is_unique


def test_drift_rows_pass_validation(tmp_path: Path) -> None:
    _, files, _ = generate(tmp_path)

    for path in files:
        frame = pd.read_csv(path)
        assert list(frame.columns) == REQUIRED_COLUMNS
        invalid, _ = collect_invalid_row_indices(frame, [])
        assert invalid == set()


def test_drift_rows_are_shifted(tmp_path: Path) -> None:
    _, files, source = generate(tmp_path)
    rows = pd.concat([pd.read_csv(path) for path in files])
    original = source[source["instant"].isin(rows["instant"])]

    assert rows["temp"].mean() > original["temp"].mean() + 0.1
    assert rows["hum"].mean() < original["hum"].mean() - 0.1
    assert (rows["weathersit"] == 3).mean() > 0.5


def test_manifest_reports_psi_above_retrain_threshold(tmp_path: Path) -> None:
    _, files, _ = generate(tmp_path)
    manifest = json.loads(
        (files[0].parent / "manifest.json").read_text(encoding="utf-8")
    )

    assert set(manifest["files"]) == {path.name for path in files}
    for scores in manifest["psi_against_baseline"].values():
        assert scores["after_shift"] > 0.25
