from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
import re
from dags.training_dag import get_training_config
from app.training.pipeline import (
    FEATURE_COLUMNS,
    NotEnoughNewData,
    prepare_training_data,
    split_history,
    fit_candidate,
    compute_training_statistics,
    decide_promotion,
    archive_files,
    baseline_train_rows,
    histogram,
    is_missing_champion,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)

FAST = {"rmse": 50.0, "inference_ms_per_row": 0.1}
DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "training_dag.py"


def make_history(rows: int = 100) -> pd.DataFrame:
    frame = pd.DataFrame({"instant": range(1, rows + 1)})
    for column in FEATURE_COLUMNS:
        frame[column] = 1
    frame["cnt"] = frame["instant"] * 2
    return frame


def write_good(directory: Path, history: pd.DataFrame, ids: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = history[history["instant"].isin(ids)]
    rows.drop(columns="cnt").to_csv(directory / "batch_good.csv", index=False)


def test_split_is_chronological() -> None:
    train, test = split_history(make_history(100))

    assert len(train) == 80 and len(test) == 20
    assert train["instant"].max() < test["instant"].min()


def test_prepare_excludes_test_rows_and_returns_labels(
    tmp_path: Path,
) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    write_good(tmp_path / "good", history, list(range(60, 100)))

    result = prepare_training_data(
        hour_path, tmp_path / "good", tmp_path / "run", set(), min_new_rows=5
    )

    train = pd.read_csv(result["train_path"])
    test = pd.read_csv(result["test_path"])
    assert set(train["instant"]).isdisjoint(set(test["instant"]))
    assert result["new_rows"] == 21  # instants 60-80, test (81-100) excluded
    assert (train["cnt"] == train["instant"] * 2).all()


def test_prepare_skips_rows_already_trained(tmp_path: Path) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    write_good(tmp_path / "good", history, list(range(60, 80)))

    with pytest.raises(NotEnoughNewData):
        prepare_training_data(
            hour_path,
            tmp_path / "good",
            tmp_path / "run",
            trained_ids=set(range(1, 101)),
            min_new_rows=5,
        )


def test_prepare_requires_minimum_new_rows(tmp_path: Path) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    write_good(tmp_path / "good", history, [10, 11])

    with pytest.raises(NotEnoughNewData):
        prepare_training_data(
            hour_path, tmp_path / "good", tmp_path / "run", set(), 50
        )


def test_fit_candidate_returns_metrics() -> None:
    rows = 400
    frame = pd.DataFrame(
        {
            "instant": range(rows),
            "season": [i % 4 + 1 for i in range(rows)],
            "yr": [i % 2 for i in range(rows)],
            "mnth": [i % 12 + 1 for i in range(rows)],
            "hr": [i % 24 for i in range(rows)],
            "holiday": 0,
            "weekday": [i % 7 for i in range(rows)],
            "workingday": 1,
            "weathersit": [i % 3 + 1 for i in range(rows)],
            "temp": [(i % 10) / 10 for i in range(rows)],
            "atemp": [(i % 10) / 10 for i in range(rows)],
            "hum": 0.5,
            "windspeed": 0.2,
        }
    )
    frame["cnt"] = frame["hr"] * 5 + 10

    model, metrics = fit_candidate(frame.iloc[:300], frame.iloc[300:])

    assert set(metrics) == {"rmse", "mae", "r2", "inference_ms_per_row"}
    assert metrics["rmse"] >= 0
    assert hasattr(model, "predict")


def test_training_statistics_match_view_format() -> None:
    frame = make_history(100)
    frame["temp"] = [i / 100 for i in range(100)]

    stats = compute_training_statistics(frame)

    assert set(stats["numerical"]) == {
        "temp", "atemp", "hum", "windspeed", "cnt"
    }
    assert set(stats["numerical"]["temp"]) == {"mean", "std", "min", "max"}
    assert "season" in stats["categorical"]
    assert abs(sum(stats["categorical"]["season"].values()) - 1) < 1e-9


def test_training_statistics_include_ten_bin_histograms() -> None:
    frame = make_history(100)
    frame["temp"] = [(i + 0.5) / 100 for i in range(100)]

    bins = compute_training_statistics(frame)["bins"]

    assert set(bins) == {"temp", "atemp", "hum", "windspeed", "cnt"}
    assert bins["temp"]["edges"][0] == 0.0
    assert bins["temp"]["edges"][-1] == 1.0
    assert len(bins["temp"]["edges"]) == 11
    assert bins["temp"]["proportions"] == pytest.approx([0.1] * 10)
    assert bins["cnt"]["edges"][-1] == 200.0


def test_histogram_keeps_out_of_range_values_in_edge_bins() -> None:
    edges = [float(i) / 10 for i in range(11)]

    proportions = histogram(pd.Series([-0.5, 0.05, 1.2, 1.0]), edges)

    assert proportions[0] == pytest.approx(0.5)
    assert proportions[-1] == pytest.approx(0.5)
    assert sum(proportions) == pytest.approx(1.0)


def test_promotes_when_candidate_matches_champion() -> None:
    decision = decide_promotion(FAST, {"rmse": 50.0}, 5.0)
    assert decision["passed"] is True


def test_rejects_worse_candidate() -> None:
    decision = decide_promotion(FAST, {"rmse": 40.0}, 5.0)
    assert decision["passed"] is False
    assert "worse" in decision["reasons"][0]


def test_rejects_slow_candidate() -> None:
    slow = {"rmse": 30.0, "inference_ms_per_row": 9.0}
    assert decide_promotion(slow, {"rmse": 40.0}, 5.0)["passed"] is False


def test_first_run_without_champion_is_promoted() -> None:
    decision = decide_promotion(FAST, None, 5.0)
    assert decision["passed"] and not decision["has_champion"]


def test_archive_moves_only_listed_files(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "a.csv").write_text("x")
    (good / "b.csv").write_text("x")

    moved = archive_files(good, tmp_path / "archive", ["a.csv", "missing.csv"])

    assert moved == ["a.csv"]
    assert (tmp_path / "archive" / "a.csv").exists()
    assert (good / "b.csv").exists()
    assert not (good / "a.csv").exists()


def test_baseline_uses_oldest_sixty_percent() -> None:
    base = baseline_train_rows(make_history(100))
    assert len(base) == 60
    assert base["instant"].max() == 60


def test_rows_seen_by_baseline_are_not_new(tmp_path: Path) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    write_good(tmp_path / "good", history, list(range(40, 100)))

    result = prepare_training_data(
        hour_path,
        tmp_path / "good",
        tmp_path / "run",
        trained_ids=set(range(1, 61)),
        min_new_rows=5,
    )

    assert result["new_rows"] == 20  # instants 61-80 only


def test_rows_of_a_rejected_run_are_not_new_again(tmp_path: Path) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    write_good(tmp_path / "good", history, list(range(61, 81)))

    with pytest.raises(NotEnoughNewData):
        prepare_training_data(
            hour_path,
            tmp_path / "good",
            tmp_path / "run",
            trained_ids=set(range(1, 61)),
            min_new_rows=5,
            seen_ids=set(range(1, 81)),
        )


def test_candidate_keeps_rows_the_champion_learned(tmp_path: Path) -> None:
    history = make_history(100)
    hour_path = tmp_path / "hour.csv"
    history.to_csv(hour_path, index=False)
    # Rows 61-70 were archived by an earlier promotion; 71-80 are new.
    write_good(tmp_path / "good", history, list(range(71, 81)))

    result = prepare_training_data(
        hour_path,
        tmp_path / "good",
        tmp_path / "run",
        trained_ids=set(range(1, 71)),
        min_new_rows=5,
    )

    train_ids = set(pd.read_csv(result["train_path"])["instant"])
    assert result["new_rows"] == 10
    assert train_ids == set(range(1, 81))


def test_missing_champion_alias_is_recognised() -> None:
    unset_alias = MlflowException(
        "Registered model alias champion not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    unknown_model = MlflowException(
        "Registered Model with name=x not found",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    other = MlflowException(
        "Invalid run id", error_code=INVALID_PARAMETER_VALUE
    )

    assert is_missing_champion(unset_alias)
    assert is_missing_champion(unknown_model)
    assert not is_missing_champion(other)


def test_every_config_key_used_in_the_dag_exists() -> None:
    source = DAG_FILE.read_text(encoding="utf-8")
    used = set(re.findall(r'config\["([a-z_]+)"\]', source))
    assert used <= set(get_training_config())


def test_evaluation_key_is_spelled_consistently() -> None:
    source = DAG_FILE.read_text(encoding="utf-8")
    assert "evalution" not in source
