#!/usr/bin/env bash
# Deploy one release on the server. GitHub Actions uploads the code as
# /tmp/dsp-release.tar.gz and runs this script.
#
# The code is replaced; .env, data/, mlflow-artifacts/ and the Docker volumes
# (PostgreSQL, Grafana) are kept, so models and history survive releases.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/dsp}
ARCHIVE=${1:-/tmp/dsp-release.tar.gz}
DAGS="bike_data_ingestion bike_demand_prediction bike_demand_training"

cd "$APP_DIR"
test -f .env || { echo "Run deploy/setup_vm.sh first"; exit 1; }

find . -mindepth 1 -maxdepth 1 \
  ! -name .env ! -name data ! -name mlflow-artifacts -exec rm -rf {} +
tar -xzf "$ARCHIVE" -C "$APP_DIR"
rm -f "$ARCHIVE"
chmod -R a+rwX data mlflow-artifacts

docker compose up -d --build --remove-orphans

echo "Waiting for the API and Airflow"
for _ in $(seq 60); do
  if curl -fs http://localhost:8000/health >/dev/null \
    && curl -fs http://localhost:8080/api/v2/monitor/health >/dev/null; then
    break
  fi
  sleep 10
done
curl -fs http://localhost:8000/health >/dev/null

# First deployment: create v1 as @champion, then let the API load it.
if curl -fs http://localhost:8000/model-status | grep -q local-baseline; then
  docker compose exec -T airflow-scheduler \
    python -m app.training.retrain_baseline
  curl -fs -X POST http://localhost:8000/reload-model
  echo
fi

for dag in $DAGS; do
  docker compose exec -T airflow-scheduler airflow dags unpause "$dag" \
    >/dev/null
done

docker image prune -f >/dev/null
docker compose ps
echo "Serving: $(curl -fs http://localhost:8000/model-status)"
