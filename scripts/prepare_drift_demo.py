"""Drift demo batches: valid rows whose distribution moved.

Rows come from the fixed test window (newest 20% of hour.csv), which the
training DAG always excludes, so the shifted features are never trained on.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from app.training.pipeline import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    baseline_train_rows,
    bin_edges,
    histogram,
    load_history,
    split_history,
)
from app.validation.bike_data import REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_data" / "defense2" / "drift"
DEFAULT_FILES = 4
DEFAULT_ROWS_PER_FILE = 50
TEMPERATURE_SHIFT = 0.25
HUMIDITY_SHIFT = -0.3
STORM_SHARE = 0.7
STORM_WEATHERSIT = 3
MONITORED_FEATURES = ["temp", "hum", "weathersit"]
PSI_FLOOR = 0.0001


def shift_features(
    rows: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Move temperature, humidity and weather, keeping every rule valid."""
    shifted = rows.copy()
    for column in ("temp", "atemp"):
        shifted[column] = (shifted[column] + TEMPERATURE_SHIFT).clip(0, 1)
    shifted["hum"] = (shifted["hum"] + HUMIDITY_SHIFT).clip(0, 1)
    storms = np.random.default_rng(seed).random(len(shifted)) < STORM_SHARE
    shifted.loc[storms, "weathersit"] = STORM_WEATHERSIT
    return shifted.round(4)


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for expected_share, actual_share in zip(expected, actual):
        expected_share = max(expected_share, PSI_FLOOR)
        actual_share = max(actual_share, PSI_FLOOR)
        total += (actual_share - expected_share) * math.log(
            actual_share / expected_share
        )
    return round(total, 3)


def feature_psi(
    column: str,
    train: pd.DataFrame,
    batch: pd.DataFrame,
) -> float:
    """Same binning as the training baseline and feature_psi() in SQL."""
    if column in CATEGORICAL_COLUMNS:
        keys = sorted(set(train[column]) | set(batch[column]))
        expected = train[column].value_counts(normalize=True)
        actual = batch[column].value_counts(normalize=True)
        return psi(
            [float(expected.get(key, 0.0)) for key in keys],
            [float(actual.get(key, 0.0)) for key in keys],
        )
    edges = bin_edges(column, train[column])
    return psi(
        histogram(train[column], edges),
        histogram(batch[column], edges),
    )


def prepare_drift_files(
    dataset_path: Path,
    output_dir: Path,
    files: int = DEFAULT_FILES,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
    seed: int = 42,
) -> list[Path]:
    source = pd.read_csv(dataset_path)
    history = load_history(dataset_path)
    _, test = split_history(history)
    train = baseline_train_rows(history)

    window = source[source[ID_COLUMN].isin(test[ID_COLUMN])]
    required_rows = files * rows_per_file
    if len(window) < required_rows:
        raise ValueError(
            f"The test window has {len(window)} rows, "
            f"{required_rows} are required"
        )

    selected = (
        window[REQUIRED_COLUMNS]
        .sample(n=required_rows, random_state=seed)
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )
    shifted = shift_features(selected, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for number in range(files):
        batch = shifted.iloc[
            number * rows_per_file:(number + 1) * rows_per_file
        ]
        output_path = output_dir / f"drift_{number:04d}.csv"
        batch.to_csv(output_path, index=False)
        generated_files.append(output_path)

    manifest = {
        "source": "newest 20% of hour.csv (fixed test set, never trained on)",
        "instants": [
            int(selected[ID_COLUMN].min()),
            int(selected[ID_COLUMN].max()),
        ],
        "rows_per_file": rows_per_file,
        "shifts": {
            "temp": f"+{TEMPERATURE_SHIFT}, clipped to [0, 1]",
            "atemp": f"+{TEMPERATURE_SHIFT}, clipped to [0, 1]",
            "hum": f"{HUMIDITY_SHIFT}, clipped to [0, 1]",
            "weathersit": f"{STORM_WEATHERSIT} for about "
                          f"{int(STORM_SHARE * 100)}% of rows",
        },
        "psi_against_baseline": {
            column: {
                "before_shift": feature_psi(column, train, selected),
                "after_shift": feature_psi(column, train, shifted),
            }
            for column in MONITORED_FEATURES
        },
        "files": {
            path.name: {
                "expected_criticality": "none",
                "expected_good_rows": rows_per_file,
                "expected_bad_rows": 0,
                "expected_alert": "ML Alerts: PSI above 0.25",
            }
            for path in generated_files
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare valid batches that drift from the baseline."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--files", type=int, default=DEFAULT_FILES)
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=DEFAULT_ROWS_PER_FILE,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = prepare_drift_files(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        files=args.files,
        rows_per_file=args.rows_per_file,
        seed=args.seed,
    )
    print(f"Prepared {len(generated)} drift files in {args.output_dir}")


if __name__ == "__main__":
    main()
