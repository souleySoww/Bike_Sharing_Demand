from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import Base, TrainingStat
from app.database.schema import (
    ERROR_CATEGORIES,
    VIEWS,
    ensure_monitoring_schema,
)
from app.validation.bike_data import collect_invalid_row_indices


LEGACY_PREDICTIONS_TABLE = """
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY,
        prediction FLOAT NOT NULL,
        source VARCHAR(32) NOT NULL,
        model_name VARCHAR(128) NOT NULL,
        features JSON NOT NULL,
        created_at DATETIME NOT NULL
    )
"""


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> Engine:
    return create_engine(f"sqlite:///{tmp_path / 'schema.db'}")


def training_stat(version: str, is_champion: bool) -> TrainingStat:
    return TrainingStat(
        model_version=version,
        run_id=f"run-{version}",
        statistics={"numerical": {}, "categorical": {}},
        trained_ids=[1, 2],
        input_files={},
        is_champion=is_champion,
    )


def test_upgrade_adds_model_version_and_keeps_existing_rows(
    sqlite_engine: Engine,
) -> None:
    with sqlite_engine.begin() as connection:
        connection.execute(text(LEGACY_PREDICTIONS_TABLE))
        connection.execute(
            text(
                "INSERT INTO predictions "
                "VALUES (1, 12.5, 'webapp', 'old', '{}', '2026-01-01')"
            )
        )

    ensure_monitoring_schema(sqlite_engine)

    inspector = inspect(sqlite_engine)
    columns = {item["name"] for item in inspector.get_columns("predictions")}
    indexes = {item["name"] for item in inspector.get_indexes("predictions")}
    assert "model_version" in columns
    assert "ix_predictions_model_version" in indexes
    with sqlite_engine.connect() as connection:
        row = connection.execute(
            text("SELECT prediction, model_version FROM predictions")
        ).one()
    assert row == (12.5, None)


def test_upgrade_can_run_on_every_startup(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(bind=sqlite_engine)

    ensure_monitoring_schema(sqlite_engine)
    ensure_monitoring_schema(sqlite_engine)

    assert "model_version" in {
        item["name"]
        for item in inspect(sqlite_engine).get_columns("predictions")
    }


def test_only_one_model_can_be_the_drift_baseline(
    sqlite_engine: Engine,
) -> None:
    Base.metadata.create_all(bind=sqlite_engine)

    with Session(sqlite_engine) as session:
        session.add(training_stat("1", True))
        session.commit()
        session.add(training_stat("2", True))
        with pytest.raises(IntegrityError):
            session.commit()


def test_rejected_candidates_can_coexist_with_the_champion(
    sqlite_engine: Engine,
) -> None:
    Base.metadata.create_all(bind=sqlite_engine)

    with Session(sqlite_engine) as session:
        session.add_all(
            [
                training_stat("1", True),
                training_stat("2", False),
                training_stat("3", False),
            ]
        )
        session.commit()
        champions = session.scalars(
            select(TrainingStat.model_version).where(TrainingStat.is_champion)
        ).all()

    assert champions == ["1"]


def test_every_validation_counter_belongs_to_one_category() -> None:
    _, error_counts = collect_invalid_row_indices(
        pd.DataFrame({"instant": [1]})
    )
    mapped = [
        counter
        for counters in ERROR_CATEGORIES.values()
        for counter in counters
    ]

    assert sorted(mapped) == sorted(error_counts)


def test_all_five_required_error_categories_exist() -> None:
    assert set(ERROR_CATEGORIES) == {
        "completeness",
        "validity",
        "consistency",
        "schema",
        "type",
    }


def test_monitoring_views_are_only_created_on_postgres(
    sqlite_engine: Engine,
) -> None:
    Base.metadata.create_all(bind=sqlite_engine)

    ensure_monitoring_schema(sqlite_engine)

    assert inspect(sqlite_engine).get_view_names() == []
    assert set(VIEWS) == {
        "v_ingestion_errors",
        "v_ingestion_error_types",
        "v_baseline_numeric",
        "v_baseline_categorical",
        "v_prediction_features",
        "v_baseline_bins",
    }
