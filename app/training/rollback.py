"""Put an older model back in service.

Moves the @champion alias to an earlier registered version, makes its
training statistics the drift baseline again, then asks the API to load it.

    docker compose exec airflow-scheduler python -m app.training.rollback
    docker compose exec airflow-scheduler python -m app.training.rollback \
        --version 1 --dry-run
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import requests

from app.training.pipeline import DEFAULT_REGISTRY_NAME, set_champion_alias


class RollbackError(RuntimeError):
    pass


def read_versions(cursor: Any) -> tuple[str | None, list[str]]:
    """Current champion and the versions that served before, oldest first."""
    cursor.execute(
        "SELECT model_version, is_champion, lifecycle_status "
        "FROM training_stats"
    )
    champion = None
    retired = []
    for version, is_champion, status in cursor.fetchall():
        if is_champion:
            champion = str(version)
        elif status == "retired":
            retired.append(str(version))
    return champion, sorted(retired, key=version_number)


def version_number(version: str) -> int:
    return int(version) if version.isdigit() else -1


def choose_target(
    champion: str | None,
    retired: list[str],
    requested: str | None = None,
) -> str:
    """The requested version, or the latest one that served before."""
    if champion is None:
        raise RollbackError("No champion in training_stats: nothing to undo")
    if requested is not None:
        if requested == champion:
            raise RollbackError(f"v{requested} is already the champion")
        if requested not in retired:
            raise RollbackError(
                f"v{requested} never served as champion; choose one of "
                f"{', '.join(retired) or 'none'}"
            )
        return requested
    older = [
        version
        for version in retired
        if version_number(version) < version_number(champion)
    ]
    if not older:
        raise RollbackError(f"No version served before v{champion}")
    return older[-1]


def swap_champion(cursor: Any, target: str) -> None:
    # Unset first: a partial unique index allows one champion at a time.
    cursor.execute(
        "UPDATE training_stats SET is_champion = FALSE, "
        "lifecycle_status = 'retired' WHERE is_champion"
    )
    cursor.execute(
        "UPDATE training_stats SET is_champion = TRUE, "
        "lifecycle_status = 'champion' WHERE model_version = %s",
        (target,),
    )


def reload_api(api_url: str, expected_version: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/reload-model", timeout=60
    )
    response.raise_for_status()
    loaded = response.json()
    if str(loaded["model_version"]) != expected_version:
        raise RollbackError(
            f"API loaded v{loaded['model_version']}, "
            f"expected v{expected_version}"
        )
    return loaded


def rollback(
    connection: Any,
    registry_name: str,
    api_url: str,
    requested: str | None = None,
    dry_run: bool = False,
) -> str:
    with connection.cursor() as cursor:
        champion, retired = read_versions(cursor)
        target = choose_target(champion, retired, requested)
        print(f"Rollback {registry_name}: v{champion} -> v{target}")
        if dry_run:
            print("Dry run: nothing changed")
            return target
        # The alias moves inside the transaction: if it fails, the
        # database change is never committed.
        swap_champion(cursor, target)
        set_champion_alias(registry_name, target)
    connection.commit()
    loaded = reload_api(api_url, target)
    print(f"API now serves v{loaded['model_version']}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        help="Version to restore (default: the one that served before)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the change without making it",
    )
    return parser.parse_args()


def main() -> None:
    import psycopg2

    args = parse_args()
    connection = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        rollback(
            connection,
            registry_name=os.getenv(
                "MODEL_REGISTRY_NAME", DEFAULT_REGISTRY_NAME
            ),
            api_url=os.getenv("PREDICTION_API_URL", "http://api:8000"),
            requested=args.version,
            dry_run=args.dry_run,
        )
    except RollbackError as exc:
        raise SystemExit(f"Rollback refused: {exc}") from exc
    finally:
        connection.close()


if __name__ == "__main__":
    main()
