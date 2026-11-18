FROM ghcr.io/mlflow/mlflow:v3.1.4

RUN pip install --no-cache-dir psycopg2-binary==2.9.12

COPY docker/mlflow_create_db.py /opt/mlflow_create_db.py
