from __future__ import annotations

import json
import os
import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from app.training.pipeline import train_and_register

from app.training.pipeline import (
    DEFAULT_MIN_NEW_ROWS,
    NotEnoughNewData,
    prepare_training_data,
    compute_training_statistics,
    DEFAULT_MAX_INFERENCE_MS,
    DEFAULT_REGISTRY_NAME,
    evaluate_against_champion,
    archive_files,
    set_champion_alias,
)

try:
    from airflow.exceptions import AirflowSkipException  # type:ignore
    from airflow.sdk import DAG, task                    # type:ignore
except ImportError:  # pragma: no cover - Airflow is installed in Docker.
    DAG = None
    task = None

    class AirflowSkipException(Exception):
        pass


try:
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # pragma: no cover - Airflow is installed in Docker.
    PostgresHook = None


DAG_ID = "bike_demand_training"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_training_config() -> dict[str, Any]:
    data_dir = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))
    return {
        "hour_path": os.getenv(
            "HOUR_CSV_PATH", f"{data_dir}/raw/hour.csv"
        ),
        "good_data_dir": os.getenv("GOOD_DATA_DIR", f"{data_dir}/good_data"),
        "run_dir": os.getenv(
            "TRAINING_RUN_DIR", f"{data_dir}/training_runs"
        ),
        "min_new_rows": int(
            os.getenv("TRAINING_MIN_NEW_ROWS", DEFAULT_MIN_NEW_ROWS)
        ),
        "postgres_conn_id": os.getenv(
            "AIRFLOW_POSTGRES_CONN_ID",
            "dsp_postgres"
        ),
        "registry_name": os.getenv(
            "MODEL_REGISTRY_NAME",
            DEFAULT_REGISTRY_NAME
            ),
        "max_inference_ms": float(
            os.getenv("TRAINING_MAX_INFERENCE_MS", DEFAULT_MAX_INFERENCE_MS)
        ),
        "archived_data_dir": os.getenv(
            "ARCHIVED_DATA_DIR", f"{data_dir}/archived_data"
        ),
        "api_url": os.getenv("PREDICTION_API_URL", "http://api:8000"),
    }


COMPLETED_STATUSES = ("champion", "retired", "rejected")


def parse_ids(value: Any) -> set[int]:
    if not value:
        return set()
    if isinstance(value, str):
        value = json.loads(value)
    return {int(item) for item in value}


def read_trained_ids(conn_id: str) -> tuple[set[int], set[int]]:
    """Rows of the champion, and rows of every completed run.

    Rejected runs count as completed, so their rows are not new data next
    time. Runs that crashed before evaluation stay 'candidate' and do not
    block their rows.
    """
    hook = PostgresHook(postgres_conn_id=conn_id)
    rows = hook.get_records(
        "SELECT trained_ids, is_champion FROM training_stats "
        "WHERE lifecycle_status IN %s",
        parameters=(COMPLETED_STATUSES,),
    )
    champion_ids: set[int] = set()
    seen_ids: set[int] = set()
    for trained_ids, is_champion in rows:
        ids = parse_ids(trained_ids)
        seen_ids |= ids
        if is_champion:
            champion_ids = ids
    return champion_ids, seen_ids


def load_data(config: dict[str, Any]) -> dict[str, Any]:
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    champion_ids, seen_ids = read_trained_ids(config["postgres_conn_id"])
    try:
        result = prepare_training_data(
            hour_path=config["hour_path"],
            good_data_dir=config["good_data_dir"],
            output_dir=Path(config["run_dir"]) / run_name,
            trained_ids=champion_ids,
            min_new_rows=config["min_new_rows"],
            seen_ids=seen_ids,
        )
    except NotEnoughNewData as exc:
        raise AirflowSkipException(str(exc)) from exc
    return {**result, "run_name": run_name}


def train_model(payload: dict[str, Any]) -> dict[str, Any]:
    result = train_and_register(
        train_path=payload["train_path"],
        test_path=payload["test_path"],
        run_name=payload["run_name"],
        new_rows=payload["new_rows"]
    )
    return {**payload, **result}


def save_training_stats(payload: dict[str, Any], config: dict[str, Any]):
    train = pd.read_csv(payload["train_path"])
    statistics = compute_training_statistics(train)
    hook = PostgresHook(postgres_conn_id=config["postgres_conn_id"])
    hook.run(
        """
        INSERT INTO training_stats
            (model_version, run_id, statistics, trained_ids,
             input_files, is_champion, lifecycle_status)
        VALUES (%s, %s, %s, %s, %s, FALSE, 'candidate')
        ON CONFLICT (model_version) DO UPDATE SET
            run_id = EXCLUDED.run_id,
            statistics = EXCLUDED.statistics,
            trained_ids = EXCLUDED.trained_ids,
            input_files = EXCLUDED.input_files
        """,
        parameters=(
            payload["model_version"],
            payload["run_id"],
            json.dumps(statistics),
            json.dumps(payload["trained_ids"]),
            json.dumps({"files": payload["good_files"]}),
        ),
    )
    return payload


def send_ml_alert(message: str) -> None:
    url = os.getenv("TEAMS_ML_ALERTS_WEBHOOK_URL")
    if not url:
        print("No Teams webhook configured; alert not sent")
        return
    card = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {"type": "TextBlock", "text": message, "wrap": True}
                    ],
                },
            }
        ],
    }
    try:
        requests.post(url, json=card, timeout=15).raise_for_status()
    except requests.RequestException as exc:
        print(f"Teams alert could not be sent: {exc}")


def evaluate_candidate(
        payload: dict[str, Any],
        config: dict[str, Any],
) -> dict[str, any]:
    evaluation = evaluate_against_champion(
        test_path=payload["test_path"],
        registry_name=config["registry_name"],
        candidate_version=payload["model_version"],
        candidate_run_id=payload["run_id"],
        max_inference_ms=config["max_inference_ms"],
    )
    if not evaluation["passed"]:
        hook = PostgresHook(postgres_conn_id=config["postgres_conn_id"])
        hook.run(
            "UPDATE training_stats SET lifecycle_status = 'rejected' "
            "WHERE model_version = %s",
            parameters=(payload["model_version"],),
        )
        send_ml_alert(
            "ML ALERT - Candidate model NOT promoted\n\n"
            f"Model: {config['registry_name']} v{payload['model_version']}\n"
            f"Current champion: v{evaluation['champion_version']}\n"
            f"Candidate RMSE: {evaluation['candidate_metrics']['rmse']:.2f}\n"
            "Champion RMSE: "
            f"{(evaluation['champion_metrics'] or {}).get('rmse', 'n/a')}\n"
            "Inference time: "
            f"{evaluation['candidate_metrics']['inference_ms_per_row']:.3f}"
            " ms/row\n"
            f"Reason: {'; '.join(evaluation['reasons'])}\n"
            f"MLflow run: {payload['run_id']}\n\n"
            "The current champion keeps serving predictions."
        )
    return {**payload, "evaluation": evaluation}


def promote_to_champion(
    payload: dict[str, Any],
    config: dict[str, Any]
) -> dict[str, Any]:
    evaluation = payload["evaluation"]
    if not evaluation["passed"]:
        raise AirflowSkipException(
            "Candidate rejected: " + "; ".join(evaluation["reasons"])
        )
    version = payload["model_version"]
    set_champion_alias(config["registry_name"], version)

    hook = PostgresHook(postgres_conn_id=config["postgres_conn_id"])
    conn = hook.get_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE training_stats SET is_champion = FALSE, "
                "lifecycle_status = 'retired' WHERE is_champion"
            )
            cursor.execute(
                "UPDATE training_stats SET is_champion = TRUE, "
                "lifecycle_status = 'champion' WHERE model_version = %s",
                (version,),
            )
        conn.commit()
    finally:
        conn.close()
    return payload


def notify_api_reload(
    payload: dict[str, Any],
    config: dict[str, Any]
) -> dict[str, Any]:
    response = requests.post(
        f"{config['api_url'].rstrip('/')}/reload-model", timeout=60
    )
    response.raise_for_status()
    loaded = response.json()
    if loaded["model_version"] != payload["model_version"]:
        raise RuntimeError(
            f"API loaded v{loaded['model_version']}, "
            f"expected v{payload['model_version']}"
        )
    return loaded


def archive_data(
    payload: dict[str, Any],
    config: dict[str, Any]
) -> list[str]:
    return archive_files(
        config["good_data_dir"],
        config["archived_data_dir"],
        payload["good_files"]
    )


if DAG is not None and task is not None:
    with DAG(
        dag_id=DAG_ID,
        description="Retrain, evaluate and promote the demand model.",
        schedule="*/10 * * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["bike-sharing", "training"],
    ) as training_dag:

        @task(task_id="load_data")
        def load_data_task() -> dict[str, Any]:
            return load_data(get_training_config())

        @task(task_id="train_model")
        def train_model_task(payload: dict[str, Any]) -> dict[str, Any]:
            return train_model(payload)

        @task(task_id="save_training_stats")
        def save_training_stats_task(payload: dict[str, Any]):
            return save_training_stats(payload, get_training_config())

        @task(task_id="evaluate_candidate")
        def evaluate_candidate_task(payload: dict[str, Any]) -> dict[str, Any]:
            return evaluate_candidate(payload, get_training_config())

        @task(task_id="promote_to_champion")
        def promote_task(payload: dict[str, Any]) -> dict[str, Any]:
            return promote_to_champion(payload, get_training_config())

        @task(task_id="notify_api_reload")
        def notify_api_reload_task(payload: dict[str, Any]):
            return notify_api_reload(payload, get_training_config())

        @task(task_id="archive_data")
        def archive_data_task(payload: dict[str, Any]):
            return archive_data(payload, get_training_config())

        evaluated = evaluate_candidate_task(
            save_training_stats_task(train_model_task(load_data_task()))
        )

        promoted = promote_task(evaluated)
        notify_api_reload_task(promoted)
        archive_data_task(promoted)

else:
    training_dag = None
