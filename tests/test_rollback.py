from __future__ import annotations

import pytest

from app.training import rollback


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, parameters=None):
        self.statements.append((sql, parameters))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_ = FakeCursor(rows)
        self.committed = False

    def cursor(self):
        return self.cursor_

    def commit(self):
        self.committed = True


HISTORY = [
    ("1", False, "retired"),
    ("2", False, "rejected"),
    ("3", False, "retired"),
    ("4", True, "champion"),
]


class FakeResponse:
    def __init__(self, version):
        self.version = version

    def raise_for_status(self):
        return None

    def json(self):
        return {"model_version": self.version}


@pytest.fixture
def calls(monkeypatch):
    recorded = {"alias": [], "post": []}

    def fake_alias(name, version):
        recorded["alias"].append((name, version))

    def fake_post(url, timeout):
        recorded["post"].append(url)
        return FakeResponse(recorded["alias"][-1][1])

    monkeypatch.setattr(rollback, "set_champion_alias", fake_alias)
    monkeypatch.setattr(rollback.requests, "post", fake_post)
    return recorded


def test_default_target_is_the_latest_earlier_champion() -> None:
    champion, retired = rollback.read_versions(FakeCursor(HISTORY))

    assert champion == "4"
    assert retired == ["1", "3"]
    assert rollback.choose_target(champion, retired) == "3"


def test_rejected_versions_are_never_restored() -> None:
    with pytest.raises(rollback.RollbackError, match="never served"):
        rollback.choose_target("4", ["1", "3"], requested="2")


def test_explicit_version_must_differ_from_champion() -> None:
    with pytest.raises(rollback.RollbackError, match="already"):
        rollback.choose_target("4", ["1", "3"], requested="4")


def test_nothing_older_than_the_first_model() -> None:
    with pytest.raises(rollback.RollbackError, match="before v1"):
        rollback.choose_target("1", ["3"])


def test_rollback_moves_alias_database_and_api(calls) -> None:
    connection = FakeConnection(HISTORY)

    target = rollback.rollback(
        connection, "bike-sharing-demand", "http://api:8000/"
    )

    assert target == "3"
    assert calls["alias"] == [("bike-sharing-demand", "3")]
    assert connection.committed
    updates = connection.cursor_.statements[1:]
    assert "is_champion = FALSE" in updates[0][0]
    assert updates[1][1] == ("3",)
    assert calls["post"] == ["http://api:8000/reload-model"]


def test_dry_run_changes_nothing(calls) -> None:
    connection = FakeConnection(HISTORY)

    target = rollback.rollback(
        connection, "bike-sharing-demand", "http://api:8000",
        requested="1", dry_run=True,
    )

    assert target == "1"
    assert calls == {"alias": [], "post": []}
    assert not connection.committed


def test_failed_alias_change_is_not_committed(monkeypatch) -> None:
    def broken_alias(name, version):
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(rollback, "set_champion_alias", broken_alias)
    connection = FakeConnection(HISTORY)

    with pytest.raises(RuntimeError):
        rollback.rollback(connection, "bike-sharing-demand", "http://api")

    assert not connection.committed


def test_api_serving_another_version_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(rollback, "set_champion_alias", lambda n, v: None)
    monkeypatch.setattr(
        rollback.requests, "post", lambda url, timeout: FakeResponse("4")
    )

    with pytest.raises(rollback.RollbackError, match="expected v3"):
        rollback.rollback(
            FakeConnection(HISTORY), "bike-sharing-demand", "http://api"
        )
