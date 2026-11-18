from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.validation.bike_data import DATA_DOCS_SITE_DIR, validate_bike_data


def make_valid_rows(count: int = 10) -> list[dict]:
    return [
        {
            "instant": index + 1,
            "dteday": "2012-01-01",
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": index % 24,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
        }
        for index in range(count)
    ]


def test_validate_bike_data_marks_clean_file_as_valid(tmp_path) -> None:
    file_path = tmp_path / "clean.csv"
    pd.DataFrame(make_valid_rows()).to_csv(file_path, index=False)

    summary, good_df, bad_df = validate_bike_data(file_path)

    assert summary.criticality == "none"
    assert summary.invalid_rows == 0
    assert summary.valid_rows == 10
    assert summary.gx_success is True
    assert summary.gx_report_path.startswith("validations/")
    assert (
        Path("data/gx") / DATA_DOCS_SITE_DIR / summary.gx_report_path
    ).exists()
    assert len(good_df) == 10
    assert bad_df.empty


def test_validate_bike_data_detects_corrupted_rows_and_missing_column(
    tmp_path,
) -> None:
    rows = make_valid_rows()
    rows[0]["hum"] = 2.5
    rows[1]["weathersit"] = 99
    rows[2]["windspeed"] = "not_a_number"
    rows[3]["holiday"] = 1
    rows[3]["workingday"] = 1
    rows[4]["instant"] = rows[5]["instant"]
    rows[6]["atemp"] = None
    file_path = tmp_path / "bad.csv"
    pd.DataFrame(rows).drop(columns=["dteday"]).to_csv(file_path, index=False)

    summary, good_df, bad_df = validate_bike_data(file_path)

    assert summary.criticality == "high"
    assert "dteday" in summary.missing_columns
    assert summary.invalid_rows == 10
    assert summary.valid_rows == 0
    assert summary.error_counts["missing_required_column"] == 10
    assert summary.gx_success is False
    assert good_df.empty
    assert len(bad_df) == 10


def test_each_validation_links_to_its_own_report(tmp_path) -> None:
    clean_path = tmp_path / "clean.csv"
    broken_path = tmp_path / "broken.csv"
    pd.DataFrame(make_valid_rows()).to_csv(clean_path, index=False)
    broken = pd.DataFrame(make_valid_rows())
    broken["temp"] = 5.0
    broken.to_csv(broken_path, index=False)

    clean_summary, _, _ = validate_bike_data(clean_path)
    broken_summary, _, _ = validate_bike_data(broken_path)

    assert clean_summary.gx_report_path != broken_summary.gx_report_path
    report = Path("data/gx") / DATA_DOCS_SITE_DIR
    assert "temp" in (report / broken_summary.gx_report_path).read_text(
        encoding="utf-8"
    )
