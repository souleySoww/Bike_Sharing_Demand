from __future__ import annotations

import json
import os

import mlflow
import psycopg2
from mlflow import MlflowClient

from app.training.pipeline import (
    DEFAULT_REGISTRY_NAME,
    ID_COLUMN,
    compute_training_statistics,
    load_history,
    baseline_train_rows,
)


def main() -> None:
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    name = os.getenv("MODEL_REGISTRY_NAME", DEFAULT_REGISTRY_NAME)
    champion = MlflowClient().get_model_version_by_alias(name, "champion")
    version = str(champion.version)

    hour_path = os.getenv("HOUR_CSV_PATH", "/opt/airflow/data/raw/hour.csv")
    train = baseline_train_rows(load_history(hour_path))
    statistics = compute_training_statistics(train)
    trained_ids = [int(i) for i in train[ID_COLUMN]]

    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE training_stats SET is_champion = FALSE, "
                "lifecycle_status = 'retired' "
                "WHERE is_champion AND model_version <> %s",
                (version,),
            )
            cursor.execute(
                """
                INSERT INTO training_stats
                    (model_version, run_id, statistics, trained_ids,
                     input_files, is_champion, lifecycle_status)
                VALUES (%s, %s, %s, %s, %s, TRUE, 'champion')
                ON CONFLICT (model_version) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    statistics = EXCLUDED.statistics,
                    trained_ids = EXCLUDED.trained_ids,
                    is_champion = TRUE,
                    lifecycle_status = 'champion'
                """,
                (
                    version,
                    champion.run_id,
                    json.dumps(statistics),
                    json.dumps(trained_ids),
                    json.dumps({"files": ["hour.csv (oldest 60%)"]}),
                ),
            )
    print(f"Bootstrapped training_stats for {name} v{version}")


if __name__ == "__main__":
    main()
