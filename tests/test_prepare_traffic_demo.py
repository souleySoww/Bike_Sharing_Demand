from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.training.pipeline import (
    baseline_train_rows,
    load_history,
    prepare_training_data,
)
from app.validation.bike_data import (
    REQUIRED_COLUMNS,
    collect_invalid_row_indices,
)
from scripts.prepare_drift_demo import feature_psi
from scripts.prepare_traffic_demo import prepare_traffic_files


DATASET = Path(__file__).resolve().parents[1] / "data" / "raw" / "hour.csv"


def write_dataset(tmp_path: Path, row_count: int = 1000) -> Path:
    dataset_path = tmp_path / "hour.csv"
    pd.DataFrame(
        [
            {
                "instant": index + 1,
                "dteday": "2012-01-01",
                "season": 1 + index % 4,
                "yr": 1,
                "mnth": 1,
                "hr": index % 24,
                "holiday": 0,
                "weekday": 1,
                "workingday": 1,
                "weathersit": 1 + index % 3,
                "temp": (index % 50) / 50,
                "atemp": 0.5,
                "hum": 0.5,
                "windspeed": 0.2,
                "cnt": 100,
            }
            for index in range(row_count)
        ]
    ).to_csv(dataset_path, index=False)
    return dataset_path


def read_rows(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path) for path in files])


def test_rows_come_from_the_baseline_only(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    base = baseline_train_rows(load_history(dataset_path))

    files = prepare_traffic_files(
        dataset_path, tmp_path / "out", files=5, rows_per_file=10
    )
    rows = read_rows(files)

    assert len(files) == 5
    assert rows["instant"].is_unique
    assert set(rows["instant"]) <= set(base["instant"])
    assert (tmp_path / "out" / "manifest.json").exists()


def test_files_pass_validation(tmp_path: Path) -> None:
    files = prepare_traffic_files(
        write_dataset(tmp_path), tmp_path / "out", files=3
    )

    for path in files:
        frame = pd.read_csv(path)
        assert list(frame.columns) == REQUIRED_COLUMNS
        assert collect_invalid_row_indices(frame, [])[0] == set()


def test_traffic_never_triggers_training(tmp_path: Path) -> None:
    dataset_path = write_dataset(tmp_path)
    base = baseline_train_rows(load_history(dataset_path))
    good_dir = tmp_path / "good_data"
    prepare_traffic_files(dataset_path, good_dir, files=10)

    with pytest.raises(Exception, match="0 new rows"):
        prepare_training_data(
            dataset_path,
            good_dir,
            tmp_path / "run",
            trained_ids=set(base["instant"]),
        )


def test_too_many_rows_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline window"):
        prepare_traffic_files(
            write_dataset(tmp_path), tmp_path / "out", files=100
        )


@pytest.mark.skipif(not DATASET.exists(), reason="hour.csv not downloaded")
def test_real_traffic_stays_below_the_drift_threshold(tmp_path: Path) -> None:
    files = prepare_traffic_files(DATASET, tmp_path / "out")
    traffic = read_rows(files)
    train = baseline_train_rows(load_history(DATASET))

    for column in ("temp", "hum", "weathersit"):
        assert feature_psi(column, train, traffic) < 0.1
