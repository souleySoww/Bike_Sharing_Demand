#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 server for the stack.
#
#   bash setup_vm.sh            (as the user that GitHub Actions logs in with)
#
# Installs Docker, creates /opt/dsp with its data folders, a .env with random
# secrets, and the dataset. Deployments then only replace the code.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/dsp}
DATASET_URL="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"
sudo apt-get install -y unzip

sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"
cd "$APP_DIR"
mkdir -p data/raw data/raw_data data/good_data data/bad_data \
  data/archived_data data/prediction_state data/processing_data mlflow-artifacts

if [ ! -f data/raw/hour.csv ]; then
  curl -fsSL -o data/raw/dataset.zip "$DATASET_URL"
  unzip -o -q data/raw/dataset.zip -d data/raw
  rm data/raw/dataset.zip
fi

if [ ! -f .env ]; then
  secret() { openssl rand -hex 24; }
  db_password=$(secret)
  cat > .env <<EOF
POSTGRES_USER=dsp
POSTGRES_PASSWORD=$db_password
POSTGRES_DB=dsp
DATABASE_URL=postgresql://dsp:$db_password@postgres:5432/dsp
API_BASE_URL=http://api:8000
PREDICTION_API_URL=http://api:8000
RAW_DATA_DIR=/opt/airflow/data/raw_data
GOOD_DATA_DIR=/opt/airflow/data/good_data
BAD_DATA_DIR=/opt/airflow/data/bad_data
PREDICTION_STATE_FILE=/opt/airflow/data/prediction_state/processed_files.json
AIRFLOW_POSTGRES_CONN_ID=dsp_postgres
AIRFLOW_API_SECRET_KEY=$(secret)
AIRFLOW_JWT_SECRET=$(secret)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(secret)
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_REGISTRY_NAME=bike-sharing-demand
TRAINING_MIN_NEW_ROWS=50
TRAINING_MAX_INFERENCE_MS=5.0
TEAMS_WEBHOOK_URL=
TEAMS_ML_ALERTS_WEBHOOK_URL=
DATA_DOCS_PUBLIC_URL=http://localhost:8081
EOF
  chmod 600 .env
  echo "Created $APP_DIR/.env: add the Teams webhooks there if needed."
fi

# Airflow (uid 50000) and MLflow write into these bind mounts.
chmod -R a+rwX data mlflow-artifacts

echo "Server ready. Log out and back in once so the docker group applies."
