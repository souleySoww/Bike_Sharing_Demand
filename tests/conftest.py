from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api import main
from app.database.session import get_db


@pytest.fixture(autouse=True)
def isolated_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep API tests away from the real database and MLflow server."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    engine = create_engine(
        f"sqlite:///{tmp_path / 'api.db'}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    def get_test_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(main, "engine", engine)
    main.app.dependency_overrides[get_db] = get_test_db
    yield
    main.app.dependency_overrides.clear()
    engine.dispose()
