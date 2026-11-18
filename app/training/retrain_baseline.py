from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.training import bootstrap
from app.training.pipeline import (
    baseline_train_rows,
    load_history,
    set_champion_alias,
    split_history,
    train_and_register,
)


def main() -> None:
    hour_path = os.getenv("HOUR_CSV_PATH", "/opt/airflow/data/raw/hour.csv")
    history = load_history(hour_path)
    _, test = split_history(history)
    train = baseline_train_rows(history)
    with tempfile.TemporaryDirectory() as tmp:
        train_path = Path(tmp) / "train.csv"
        test_path = Path(tmp) / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        result = train_and_register(
            train_path, test_path, run_name="baseline-60pct", new_rows=0
        )

    set_champion_alias(result["registry_name"], result["model_version"])
    print(f"Baseline v{result['model_version']} is now @champion")
    bootstrap.main()


if __name__ == '__main__':
    main()
