"""Clean ingestion files that give the training DAG enough new rows.

Rows come from the window between the 60% baseline and the fixed test set:
the champion never trained on them and they carry labels in hour.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.training.pipeline import (
    DEFAULT_MIN_NEW_ROWS,
    ID_COLUMN,
    baseline_train_rows,
    load_history,
    split_history,
)
from app.validation.bike_data import REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_data" / "defense2" / "training"
DEFAULT_ROWS = 60
DEFAULT_ROWS_PER_FILE = 20
DEFAULT_PREFIX = "train"


def training_window(dataset_path: Path) -> pd.DataFrame:
    """Labelled rows that are neither in the baseline nor in the test set."""
    source = pd.read_csv(dataset_path).sort_values(ID_COLUMN)
    history = load_history(dataset_path)
    _, test = split_history(history)
    base = baseline_train_rows(history)
    excluded = set(base[ID_COLUMN]) | set(test[ID_COLUMN])
    window = source[~source[ID_COLUMN].isin(excluded)]
    return window[REQUIRED_COLUMNS].reset_index(drop=True)


def prepare_training_files(
    dataset_path: Path,
    output_dir: Path,
    start_row: int = 0,
    rows: int = DEFAULT_ROWS,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
    prefix: str = DEFAULT_PREFIX,
) -> list[Path]:
    if start_row < 0 or rows < 1 or rows_per_file < 1:
        raise ValueError("start_row must be >= 0, rows and rows_per_file > 0")

    window = training_window(dataset_path)
    if start_row + rows > len(window):
        raise ValueError(
            f"start_row + rows = {start_row + rows}, but the training window "
            f"only has {len(window)} rows"
        )

    chunk = window.iloc[start_row:start_row + rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for number, offset in enumerate(range(0, len(chunk), rows_per_file)):
        output_path = output_dir / f"{prefix}_{number:04d}.csv"
        chunk.iloc[offset:offset + rows_per_file].to_csv(
            output_path,
            index=False,
        )
        generated_files.append(output_path)

    manifest = {
        "source": "hour.csv between the 60% baseline and the 20% test set",
        "start_row": start_row,
        "next_start_row": start_row + rows,
        "instants": [
            int(chunk[ID_COLUMN].min()),
            int(chunk[ID_COLUMN].max()),
        ],
        "rows": rows,
        "training_triggered": rows >= DEFAULT_MIN_NEW_ROWS,
        "files": {
            path.name: {
                "expected_criticality": "none",
                "expected_good_rows": len(pd.read_csv(path)),
                "expected_bad_rows": 0,
                "expected_alert": False,
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
        description="Prepare clean files that trigger the training DAG."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=DEFAULT_ROWS_PER_FILE,
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = prepare_training_files(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        start_row=args.start_row,
        rows=args.rows,
        rows_per_file=args.rows_per_file,
        prefix=args.prefix,
    )
    print(f"Prepared {len(generated)} training files in {args.output_dir}")


if __name__ == "__main__":
    main()
