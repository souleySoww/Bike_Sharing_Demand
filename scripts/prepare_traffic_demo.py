"""Background traffic for the demo: clean files the champion already knows.

Rows come from the 60% baseline that v1 was trained on, so they never count
as new training data, and their distribution matches the drift baseline:
the dashboards fill with history and stay green until the drift step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.training.pipeline import (
    ID_COLUMN,
    baseline_train_rows,
    load_history,
)
from app.validation.bike_data import REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hour.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_data" / "defense2" / "traffic"
DEFAULT_FILES = 30
DEFAULT_ROWS_PER_FILE = 10
DEFAULT_PREFIX = "traffic"


def prepare_traffic_files(
    dataset_path: Path,
    output_dir: Path,
    files: int = DEFAULT_FILES,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
    seed: int = 42,
    prefix: str = DEFAULT_PREFIX,
) -> list[Path]:
    if files < 1 or rows_per_file < 1:
        raise ValueError("files and rows_per_file must be > 0")

    source = pd.read_csv(dataset_path)
    base = baseline_train_rows(load_history(dataset_path))
    window = source[source[ID_COLUMN].isin(base[ID_COLUMN])]
    required_rows = files * rows_per_file
    if required_rows > len(window):
        raise ValueError(
            f"{required_rows} rows requested, the baseline window only has "
            f"{len(window)}"
        )

    # Random rows of the whole window: same distribution as the baseline.
    selected = window[REQUIRED_COLUMNS].sample(
        n=required_rows, random_state=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for number in range(files):
        output_path = output_dir / f"{prefix}_{number:04d}.csv"
        selected.iloc[
            number * rows_per_file:(number + 1) * rows_per_file
        ].to_csv(output_path, index=False)
        generated_files.append(output_path)

    manifest = {
        "source": "oldest 60% of hour.csv (already learned by v1)",
        "rows_per_file": rows_per_file,
        "training_triggered": False,
        "expected_drift": "PSI below 0.1",
        "files": {
            path.name: {
                "expected_criticality": "none",
                "expected_good_rows": rows_per_file,
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
        description="Prepare clean background traffic for the demo."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--files", type=int, default=DEFAULT_FILES)
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=DEFAULT_ROWS_PER_FILE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = prepare_traffic_files(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        files=args.files,
        rows_per_file=args.rows_per_file,
        seed=args.seed,
        prefix=args.prefix,
    )
    print(f"Prepared {len(generated)} traffic files in {args.output_dir}")


if __name__ == "__main__":
    main()
