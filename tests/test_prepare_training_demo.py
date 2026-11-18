from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.training.pipeline import (
    baseline_train_rows,
    load_history,
    split_history,
)
from app.validation.bike_data import (
    REQUIRED_COLUMNS,
    collect_invalid_row_indices,
)
from scripts.prepare_training_demo import prepare_training_files


def write_dataset(tmp_path: Path, row_count: int = 1000) -> Path:
    dataset_path = tmp_path / "hour.csv"
    pd.DataFrame(
        [
            {
                "instant": index + 1,
                "dteday": "2012-01-01",
                "season": 1,
                "yr": 1,
                "mnth": 1,
                "hr": index % 24,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.5,
                "atemp": 0.5,
                "hum": 0.5,
                "windspeed": 0.2,
                "cnt": 100,
            }
            # Shuffled on purpose: the window follows instant order.
            for index in reversed(range(row_count))
        ]
    ).to_csv(dataset_path, index=False)
    return dataset_path


def read_rows(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in files])


def test_rows_are_outside_baseline_and_test_set(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    history = load_history(dataset_path)
    _, test = split_history(history)
    base = baseline_train_rows(history)

    files = prepare_training_files(dataset_path, tmp_path / "out")
    rows = read_rows(files)

    assert len(files) == 3
    assert len(rows) == 60
    assert rows["instant"].is_unique
    assert not set(rows["instant"]) & set(base["instant"])
    assert not set(rows["instant"]) & set(test["instant"])
    assert rows["instant"].min() == base["instant"].max() + 1


def test_files_pass_validation(tmp_path: Path) -> None:
    files = prepare_training_files(write_dataset(tmp_path), tmp_path / "out")

    for path in files:
        frame = pd.read_csv(path)
        assert list(frame.columns) == REQUIRED_COLUMNS
        invalid, _ = collect_invalid_row_indices(frame, [])
        assert invalid == set()


def test_start_row_moves_to_fresh_rows(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    first = read_rows(prepare_training_files(dataset_path, tmp_path / "a"))
    second = read_rows(
        prepare_training_files(dataset_path, tmp_path / "b", start_row=60)
    )

    assert not set(first["instant"]) & set(second["instant"])


def test_manifest_points_to_the_next_batch(tmp_path: Path) -> None:
    out = tmp_path / "out"
    prepare_training_files(write_dataset(tmp_path), out, rows=50)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["next_start_row"] == 50
    assert manifest["training_triggered"] is True


def test_rows_beyond_the_window_are_rejected(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)

    # 1000 rows: 600 baseline, 200 test, 200 in the window.
    with pytest.raises(ValueError, match="training window"):
        prepare_training_files(
            dataset_path, tmp_path / "out", start_row=150, rows=60
        )
