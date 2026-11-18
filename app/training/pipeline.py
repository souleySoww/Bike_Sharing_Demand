from __future__ import annotations
from pathlib import Path
from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import json
import math
import os
import time
import numpy as np
import pandas as pd
import shutil


FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

ID_COLUMN = "instant"
TARGET_COLUMN = "cnt"
TEST_FRACTION = 0.2
DEFAULT_MIN_NEW_ROWS = 50
CATEGORICAL_COLUMNS = FEATURE_COLUMNS[:8]
EXPERIMENT_NAME = "bike-sharing-training"
DEFAULT_REGISTRY_NAME = "bike-sharing-demand"
DEFAULT_MAX_INFERENCE_MS = 5.0
BASELINE_TRAIN_FRACTION = 0.6
NUMERICAL_COLUMNS = FEATURE_COLUMNS[8:]
PSI_BINS = 10
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


class NotEnoughNewData(Exception):
    """Raised when too few new rows accumulated since the last training."""


def load_history(hour_path: str | Path) -> pd.DataFrame:
    """Labelled reference data, ordered chronologically."""
    frame = pd.read_csv(hour_path)
    required = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"hour.csv is missing: {', '.join(missing)}")
    return frame[required].sort_values(ID_COLUMN).reset_index(drop=True)


def split_history(
    history: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split, identical to the Defense 1 baseline."""
    test_size = math.ceil(len(history) * test_fraction)
    return history.iloc[:-test_size].copy(), history.iloc[-test_size:].copy()


def baseline_train_rows(history: pd.DataFrame) -> pd.DataFrame:
    """Oldest 60% of the history : what is baseline champion is trained on."""
    size = int(len(history) * BASELINE_TRAIN_FRACTION)
    return history.iloc[:size].copy()


def read_good_rows(good_data_dir: str | Path) -> tuple[pd.DataFrame,
                                                       list[str]]:
    """Concatenate good_data files (features only), deduplicated by id."""
    files = sorted(Path(good_data_dir).glob("*.csv"))
    columns = [ID_COLUMN, *FEATURE_COLUMNS]
    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(
                f"{file_path.name} is missing: {', '.join(missing)}"
            )
        frames.append(frame[columns])

    if not frames:
        return pd.DataFrame(columns=columns), []

    good = pd.concat(frames, ignore_index=True)
    good = good.dropna(subset=[ID_COLUMN])
    good[ID_COLUMN] = good[ID_COLUMN].astype(int)
    return good.drop_duplicates(ID_COLUMN), [path.name for path in files]


def select_new_rows(
    good_rows: pd.DataFrame,
    history: pd.DataFrame,
    test_ids: set[int],
    trained_ids: set[int],
) -> pd.DataFrame:
    """New labelled rows: not in the test set, not already trained on."""
    excluded = test_ids | trained_ids
    fresh = good_rows[~good_rows[ID_COLUMN].isin(excluded)]
    labels = history[[ID_COLUMN, TARGET_COLUMN]]
    return fresh.merge(labels, on=ID_COLUMN, how="inner")


def prepare_training_data(
    hour_path: str | Path,
    good_data_dir: str | Path,
    output_dir: str | Path,
    trained_ids: set[int],
    min_new_rows: int = DEFAULT_MIN_NEW_ROWS,
    seen_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Build the fixed test set and the candidate training set.

    trained_ids are the rows the champion learned from, and the candidate
    keeps them. seen_ids are rows of earlier completed runs, including
    rejected ones: they are trained on again but do not count as new, so a
    rejection does not trigger the same retraining on every schedule.

    Raises NotEnoughNewData when fewer than min_new_rows new rows exist.
    """
    history = load_history(hour_path)
    _, test = split_history(history)
    base_train = baseline_train_rows(history)
    test_ids = set(test[ID_COLUMN].astype(int))

    good_rows, good_files = read_good_rows(good_data_dir)
    labelled_good = select_new_rows(good_rows, history, test_ids, set())
    already_used = trained_ids | (seen_ids or set())
    new_rows = labelled_good[~labelled_good[ID_COLUMN].isin(already_used)]
    if len(new_rows) < min_new_rows:
        raise NotEnoughNewData(
            f"{len(new_rows)} new rows available, {min_new_rows} required"
        )

    champion_rows = history[
        history[ID_COLUMN].isin(trained_ids)
        & ~history[ID_COLUMN].isin(test_ids)
    ]
    train = (
        pd.concat([base_train, champion_rows, labelled_good],
                  ignore_index=True)
        .drop_duplicates(ID_COLUMN)
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.csv"
    test_path = output / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    result = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "new_rows": int(len(new_rows)),
        "trained_ids": sorted(int(i) for i in train[ID_COLUMN]),
        "good_files": good_files,
    }
    (output / "manifest.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "trained_ids"}),
        encoding="utf-8",
    )
    return result


def build_model() -> Pipeline:
    """Same pipeline as scripts/train_baseline.py (Defense 1)."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=float("nan")
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough"
    )
    regressor = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL_COLUMNS))),
        learning_rate=0.1,
        max_iter=600,
        max_leaf_nodes=64,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", regressor)]
    )


def score_model(model: Any, test: pd.DataFrame) -> dict[str, float]:
    """Metrics and mean inference time (ms per row) on a test frame."""
    features = test[FEATURE_COLUMNS]
    start = time.perf_counter()
    predictions = model.predict(features)
    elapsed = time.perf_counter() - start
    target = test[TARGET_COLUMN]
    return {
        "rmse": float(mean_squared_error(target, predictions) ** 0.5),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
        "inference_ms_per_row": float(elapsed * 1000 / len(features))
    }


def fit_candidate(
        train: pd.DataFrame,
        test: pd.DataFrame
) -> tuple[Pipeline, dict[str, float]]:
    model = build_model()
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model, score_model(model, test)


def train_and_register(
        train_path: str | Path,
        test_path: str | Path,
        run_name: str,
        new_rows: int,
) -> dict[str, Any]:
    """ Train the candidate, log it to MLflow, register a new version.
        The @champion alias is never touched here
    """
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    model, metrics = fit_candidate(train, test)
    registry_name = os.getenv("MODEL_REGISTRY_NAME", DEFAULT_REGISTRY_NAME)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)
    sample = train[FEATURE_COLUMNS].head(5)
    with mlflow.start_run(run_name=f"candidate-{run_name}") as run:
        mlflow.log_params(
            {
                "model-type": "HistGradientBoostingRegressor",
                "split": "chronological",
                "training_rows": len(train),
                "test_rows": len(test),
                "new_rows": new_rows,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            registered_model_name=registry_name
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{registry_name}'")
        candidate = next(v for v in versions if v.run_id == run.info.run_id)
        client.set_model_version_tag(
            registry_name, candidate.version, "lifecycle_status", "candidate"
        )
        # The model under test against @champion, as in the course.
        client.set_registered_model_alias(
            registry_name, CHALLENGER_ALIAS, candidate.version
        )
        return {
            "run_id": run.info.run_id,
            "model_version": str(candidate.version),
            "registry_name": registry_name,
            "metrics": metrics,
        }


def bin_edges(column: str, values: pd.Series) -> list[float]:
    """Fixed edges so serving data can be binned the same way later.

    Weather features are normalised to [0, 1]. The target has no fixed
    ceiling, so its edges stop at the training maximum and predictions above
    it fall into the last bin.
    """
    upper = 1.0 if column != TARGET_COLUMN else max(float(values.max()), 1.0)
    return [float(edge) for edge in np.linspace(0.0, upper, PSI_BINS + 1)]


def histogram(values: pd.Series, edges: list[float]) -> list[float]:
    clipped = values.astype(float).clip(edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return [float(count) / len(values) for count in counts]


def compute_training_statistics(train: pd.DataFrame) -> dict[str, Any]:
    """Baseline for drift: format expected by the v_baseline_* views.

    The 10-bin histograms feed the PSI described in course 5.
    """
    columns = list(NUMERICAL_COLUMNS)
    if TARGET_COLUMN in train:
        columns.append(TARGET_COLUMN)
    numerical = {
        column: {
            "mean": float(train[column].mean()),
            "std": float(train[column].std()),
            "min": float(train[column].min()),
            "max": float(train[column].max()),
        }
        for column in columns
    }
    bins = {}
    for column in columns:
        edges = bin_edges(column, train[column])
        bins[column] = {
            "edges": edges,
            "proportions": histogram(train[column], edges),
        }
    categorical = {
        column: {
            str(int(value)): float(share)
            for value, share in train[column]
            .value_counts(normalize=True)
            .items()
        }
        for column in CATEGORICAL_COLUMNS
    }
    return {
        "rows": int(len(train)),
        "numerical": numerical,
        "categorical": categorical,
        "bins": bins,
    }


def decide_promotion(
        candidate: dict[str, float],
        champion: dict[str, float] | None,
        max_inference_ms: float,
) -> dict[str, Any]:
    """Promote if rmse(candidate) <= rmse(champion) and inference is fast."""
    reasons = []
    if candidate["inference_ms_per_row"] >= max_inference_ms:
        reasons.append(
            f"inference {candidate['inference_ms_per_row']:.3f} ms/row "
            f">= threshold {max_inference_ms} ms/row"
        )
    if champion is not None and candidate["rmse"] > champion["rmse"]:
        reasons.append(
            f"candidate RMSE {candidate['rmse']:.2f} is worse than "
            f"champion RMSE {champion['rmse']:.2f}"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "has_champion": champion is not None,
    }


def is_missing_champion(exc: Exception) -> bool:
    """MLflow reports an unset alias as INVALID_PARAMETER_VALUE, and a model
    that was never registered as RESOURCE_DOES_NOT_EXIST."""
    code = getattr(exc, "error_code", None)
    message = str(exc).lower()
    return code == "RESOURCE_DOES_NOT_EXIST" or (
        code == "INVALID_PARAMETER_VALUE"
        and "alias" in message
        and "not found" in message
    )


def find_champion_version(client: Any, registry_name: str) -> str | None:
    from mlflow.exceptions import MlflowException

    try:
        return str(
            client.get_model_version_by_alias(
                registry_name, CHAMPION_ALIAS
            ).version
        )
    except MlflowException as exc:
        if is_missing_champion(exc):
            return None
        raise


def evaluate_against_champion(
    test_path: str | Path,
    registry_name: str,
    candidate_version: str,
    candidate_run_id: str,
    max_inference_ms: float,
) -> dict[str, Any]:

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    test = pd.read_csv(test_path)

    candidate = score_model(
        mlflow.sklearn.load_model(
            f"models:/{registry_name}/{candidate_version}"
        ),
        test,
    )

    champion = None
    champion_version = find_champion_version(client, registry_name)
    if champion_version is not None:
        champion = score_model(
            mlflow.sklearn.load_model(
                f"models:/{registry_name}/{champion_version}"
            ),
            test,
        )
    decision = decide_promotion(candidate, champion, max_inference_ms)

    # Keep the comparison on the candidate run, whatever the outcome.
    client.log_metric(candidate_run_id, "eval_rmse", candidate["rmse"])
    if champion is not None:
        client.log_metric(
            candidate_run_id,
            "champion_rmse", champion["rmse"])
    client.set_tag(
        candidate_run_id,
        "promotion_passed", str(decision["passed"]))
    client.set_tag(
        candidate_run_id,
        "promotion_reasons",
        "; ".join(decision["reasons"]) or "all criteria met",
    )
    client.set_tag(
        candidate_run_id,
        "compared_with_champion_version",
        str(champion_version)
    )
    return {
        **decision,
        "candidate_metrics": candidate,
        "champion_metrics": champion,
        "champion_version": champion_version
    }


def set_champion_alias(registry_name: str, version: str) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
    MlflowClient().set_registered_model_alias(
        registry_name, CHAMPION_ALIAS, version
    )


def archive_files(
    good_data_dir: str | Path,
    archive_dir: str | Path,
    file_names: list[str],
) -> list[str]:
    """Move the files used for training out of good_data."""
    source = Path(good_data_dir)
    target = Path(archive_dir)
    target.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in file_names:
        path = source / name
        if path.exists():
            shutil.move(str(path), str(target / name))
            moved.append(name)
    return moved
