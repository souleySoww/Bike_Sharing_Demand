"""Create the MLflow database when the Postgres volume predates it.

docker/postgres/init-databases.sh only runs on a brand-new volume, so a
volume created for Defense 1 has no 'mlflow' database.
"""
from __future__ import annotations

import os

import psycopg2
from psycopg2 import sql


DATABASE = "mlflow"


def main() -> None:
    connection = psycopg2.connect(
        host="postgres",
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    # CREATE DATABASE cannot run inside a transaction.
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DATABASE,),
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(DATABASE)
                    )
                )
                print(f"Created database {DATABASE}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
