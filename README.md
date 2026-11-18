# Hourly Bike Sharing Demand Prediction

This EPITA Data Science in Production project predicts hourly bike-rental
demand. Defense 1 delivered the data pipeline and model serving. Defense 2
adds the model lifecycle: automated retraining with an MLflow registry,
monitoring dashboards, alerts, and continuous delivery.

## Components

Defense 2:

- MLflow tracking server and model registry, with `@champion` and
  `@challenger` aliases
- Airflow 3 training DAG every 10 minutes: retrains on new labelled data,
  compares the candidate with the champion on a fixed test set, promotes it
  and reloads the API without a restart
- Model version stored with every prediction
- Grafana dashboards for data quality and for drift and predictions, with
  alerts routed to two Teams channels
- Clickable links from Teams alerts to the Great Expectations report of the
  ingested file
- Rollback of `@champion` to an earlier version in one command
- GitHub Release created on every merge into `main`

Defense 1:

- HistGradientBoosting regression model, served by FastAPI for single and
  batch predictions
- PostgreSQL storage for predictions and ingestion statistics
- Streamlit multipage application for predictions and prediction history
- Great Expectations validation, Checkpoint, and Data Docs
- Airflow 3 ingestion DAG every minute and prediction DAG every two minutes
- Teams notifications for medium- and high-criticality data problems
- Docker Compose orchestration and GitHub Actions CI

## Dataset and Model

The project uses the
[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
and its `hour.csv` file. The target is `cnt`, the hourly number of bike
rentals. `hour.csv` is split in time order:

| Rows | `instant` | Role |
| --- | --- | --- |
| Oldest 60% | 1 to 10427 | Training set of the first model (v1) |
| 60% to 80% | 10428 to 13903 | Data that arrives later and triggers retraining |
| Newest 20% | 13904 to 17379 | Fixed test set, never used for training |

Every candidate model and the current champion are scored on the same fixed
test set, so the comparison is fair. Ingested files carry no `cnt`: the
training DAG joins the real demand from `hour.csv` by `instant`, as labels
that arrive later in a real system.

## Architecture

The Streamlit application and scheduled prediction DAG send feature data to
FastAPI. FastAPI loads the `@champion` model from the MLflow registry (or the
local model when the registry is empty), makes predictions, and saves them
with their features and model version. The ingestion DAG reads one file from
`data/raw_data`, validates it with Great Expectations, saves statistics, sends
alerts when required, and routes rows to `good_data` or `bad_data`. The
prediction DAG sends all new good rows to FastAPI in one batch request.

The training DAG collects the good rows that the champion has never seen.
With 50 or more, it trains a candidate, registers it in MLflow, saves its
training statistics, and promotes it only when its RMSE is not worse than the
champion's and its inference time is below 5 ms per row. A rejected candidate
raises a Teams alert and the champion keeps serving. Grafana reads PostgreSQL
and compares live data and predictions with the champion's training data.

## Prerequisites

- Git
- Docker Desktop with Docker Compose v2
- Python 3.12 for dataset and test scripts
- A Microsoft Teams workflow webhook for alert delivery

Confirm the required versions:

```bash
docker --version
docker compose version
python3.12 --version
```

## Clone and Configure

```bash
git clone https://github.com/kranos212/dsp-bike-sharing-mlops.git
cd dsp-bike-sharing-mlops
cp .env.docker.example .env
```

The final defense must run from the default `main` branch. Team members testing
work before the final release should explicitly switch to `develop`.

Edit `.env` and replace both `change-me` values with local passwords. Keep the
password in `DATABASE_URL` identical to `POSTGRES_PASSWORD`. Add the Teams
workflow URL locally:

```env
POSTGRES_PASSWORD=your-local-postgres-password
DATABASE_URL=postgresql://dsp:your-local-postgres-password@postgres:5432/dsp
AIRFLOW_API_SECRET_KEY=your-long-random-airflow-api-secret
AIRFLOW_JWT_SECRET=your-long-random-airflow-jwt-secret
GRAFANA_ADMIN_PASSWORD=your-local-grafana-password
TEAMS_WEBHOOK_URL=https://your-private-teams-workflow-url
TEAMS_ML_ALERTS_WEBHOOK_URL=https://your-private-ml-alerts-workflow-url
```

`TEAMS_WEBHOOK_URL` feeds the "Data Quality Alerts" channel and
`TEAMS_ML_ALERTS_WEBHOOK_URL` the "ML Alerts" channel. Without them the stack
still starts and the alerts stay inside Grafana. `DATA_DOCS_PUBLIC_URL`
(default `http://localhost:8081`) is the start of the report links in Teams.

The two Airflow secrets must be shared by every Airflow container. Without
them, Airflow 3 rejects task-execution tokens and DAG tasks fail before their
Python functions start.

Never commit `.env` or the real webhook URL.

## Start the Complete Application

Make sure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
docker compose ps
```

Wait until PostgreSQL, FastAPI, Streamlit, and the Airflow API server report
healthy. The first build can take several minutes.

## Service URLs

| Service | URL | Login |
| --- | --- | --- |
| Streamlit | http://127.0.0.1:8501 | None |
| FastAPI docs | http://127.0.0.1:8000/docs | None |
| Airflow 3 | http://127.0.0.1:8080 | User: `admin` |
| Data Docs | http://127.0.0.1:8081 | None |
| Grafana | http://127.0.0.1:3000 | Values from `.env` |
| MLflow | http://127.0.0.1:5000 | None |

Airflow's Simple Auth Manager generates the administrator password at startup.
Read it with:

```bash
docker compose exec airflow-api-server \
  cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

On macOS, port 5000 is often taken by AirPlay Receiver: turn it off or change
the MLflow host port in `docker-compose.yml`.

Create the first champion once, on a new stack. It trains v1 on the oldest
60% of `hour.csv`, registers it as `@champion`, and stores its training
statistics as the drift baseline:

```bash
docker compose exec airflow-scheduler python -m app.training.retrain_baseline
```

Until then the API serves the local model as `local-baseline`.

On completely new Docker volumes, Data Docs can return HTTP 403 until the
first ingestion run generates a Great Expectations report. Open it after that
run; the report directory is intentionally empty before validation.

## Local Python Environment

The Docker application does not require a host Python environment. Create one
when running dataset scripts or tests locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset Scripts

Download and extract the UCI dataset:

```bash
python -m scripts.download_dataset
```

Create 30 ingestion files with exactly 10 rows each:

```bash
python -m scripts.split_dataset data/raw/hour.csv data/raw_data 30
```

Create a reproducible corrupted file:

```bash
python -m scripts.generate_data_errors \
  data/raw_data/batch_001.csv \
  data/raw_data/corrupted_batch.csv \
  --null-probability 0.2 \
  --range-probability 0.2 \
  --category-probability 0.2 \
  --type-probability 0.2 \
  --duplicate-probability 0.2 \
  --inconsistency-probability 0.2 \
  --outlier-probability 0.2 \
  --schema-probability 0 \
  --seed 42
```

Use `--drop-column` to force a high-criticality missing-column scenario.

## Defense Demo Files

Five prepared files are stored in `demo_data/defense1`. Every CSV contains
exactly 10 rows. Regenerate them from `hour.csv` with:

```bash
python -m scripts.prepare_defense_demo
```

Keep the files outside `data/raw_data` until their demo step. Copy only one
file at a time:

```bash
cp demo_data/defense1/01_all_invalid.csv data/raw_data/
```

The expected outcomes are recorded in
`demo_data/defense1/manifest.json`:

- `01_all_invalid.csv`: high criticality, 10 bad rows, Teams alert
- `02_clean.csv`: 10 good rows, no alert
- `03_clean.csv`: 10 good rows, no alert
- `04_clean.csv`: 10 good rows, no alert
- `05_mixed.csv`: medium criticality, 7 good and 3 bad rows, Teams alert

Defense 2 adds three sets, each with a `manifest.json`:

- `demo_data/defense2/traffic`: 30 clean files of 10 rows from the data v1
  was trained on. Background traffic that fills the dashboards without
  triggering retraining or drift. Regenerate with
  `python -m scripts.prepare_traffic_demo`.

- `demo_data/defense2/training`: 3 clean files of 20 rows from the 60-80%
  window, enough new labelled rows to trigger retraining. Regenerate with
  `python -m scripts.prepare_training_demo`; use `--start-row 60` for fresh
  rows in a second demo.
- `demo_data/defense2/drift`: 4 valid files of 50 rows from the fixed test
  set, with warmer, drier and rainier weather. They make the drift panels and
  the drift alert go red, and are never trained on. Regenerate with
  `python -m scripts.prepare_drift_demo`.

## Airflow DAGs

The DAGs are paused after a fresh installation. Unpause them in the Airflow UI
or run:

```bash
docker compose exec airflow-scheduler \
  airflow dags unpause bike_data_ingestion
docker compose exec airflow-scheduler \
  airflow dags unpause bike_demand_prediction
docker compose exec airflow-scheduler \
  airflow dags unpause bike_demand_training
```

The ingestion DAG runs every minute and contains:

1. `read_data`
2. `validate_data`
3. `save_statistics`
4. `send_alerts`
5. `split_and_save_data`

The last three tasks run in parallel after validation. Teams notifications are
sent only for medium or high criticality.

The prediction DAG runs every two minutes and contains:

1. `check_for_new_data`
2. `make_predictions`

It combines all new good-data rows into one FastAPI batch request. It does not
move or delete files from `good_data`. It remembers each file by path,
modification time and size, so a file name ingested again with new rows is
predicted again.

The training DAG runs every 10 minutes and contains:

1. `load_data`: skips the run with fewer than 50 new labelled rows
   (`TRAINING_MIN_NEW_ROWS`)
2. `train_model`: trains and registers a candidate tagged `@challenger`
3. `save_training_stats`
4. `evaluate_candidate`: same fixed test set for candidate and champion
5. `promote_to_champion`
6. `notify_api_reload`: calls `/reload-model` and checks the loaded version
7. `archive_data`: moves the used files to `data/archived_data`

Ingest new files, then let one prediction cycle run before training: after a
promotion the training DAG archives the files it used.

## Monitoring and Alerts

Grafana loads its datasource, dashboards and alert rules from `grafana/`. They
cannot be edited in the UI: change the files and restart Grafana.

| Dashboard | For | Panels | Alerts (Teams channel) |
| --- | --- | --- | --- |
| Data Quality | Data engineers | Invalid rows %, criticality, errors by category, top error types, failed expectations, valid and invalid rows | File with 100% invalid rows; schema error or more than 50% invalid rows (Data Quality Alerts) |
| Drift and Predictions | ML engineers | PSI of temp, atemp, hum and weathersit; mean prediction per model version; PSI of predictions; prediction spread | PSI above 0.25; constant or unrealistic predictions (ML Alerts) |

Drift uses the Population Stability Index with 10 bins against the
champion's training data: below 0.1 stable, 0.1 to 0.25 monitor, above 0.25
retrain. Buckets with fewer than 30 rows are hidden because PSI is noise on
small samples.

## Rollback

Put the model that served before the current champion back in service:

```bash
docker compose exec airflow-scheduler python -m app.training.rollback --dry-run
docker compose exec airflow-scheduler python -m app.training.rollback
```

It moves `@champion` in MLflow, makes that version's training statistics the
drift baseline again, and asks the API to load it. Use `--version N` for a
specific version; rejected candidates are refused.

## Defense 1 Demonstration Order

1. Show all Docker containers running with `docker compose ps`.
2. Make a single prediction in Streamlit.
3. Upload a feature-only CSV for a batch prediction.
4. Show both records on the Past Predictions page.
5. Copy `01_all_invalid.csv` into `raw_data` and trigger ingestion.
6. Show high criticality, the Teams alert, database statistics, rendered Data
   Docs, and all rows in `bad_data`.
7. Copy `05_mixed.csv` and show the seven/three good/bad split.
8. Copy the three clean files and run ingestion for each one.
9. Trigger the prediction DAG and show scheduled predictions in Streamlit.
10. Trigger it again without new files and show that no data is reprocessed.

Data Docs are generated by Great Expectations, not handwritten HTML. Open the
report through http://127.0.0.1:8081 after an ingestion run.

## Tests and Continuous Integration

Run the same checks used by GitHub Actions:

```bash
python -m flake8 .
python -m pytest -q
docker compose config --quiet
```

GitHub Actions runs linting and tests as separate steps for pull requests into
`develop` and `main`.

Continuous delivery (`.github/workflows/release.yml`): every merge into `main`
runs the same checks plus `docker compose config`, then publishes a GitHub
Release tagged `vYYYY.MM.DD-<run number>` with generated notes. Starting the
workflow by hand on another branch publishes a pre-release instead, which
tests the pipeline without touching `main`.

When the `DEPLOY_HOST`, `DEPLOY_USER` and `DEPLOY_SSH_KEY` secrets exist, the
same workflow deploys each release to a server over SSH. Setting up a server
on Google Cloud is described in `deploy/GOOGLE_CLOUD.md`.

The Defense 2 rehearsal runbook is `defense/REHEARSAL_D2.md`.

## Git Workflow

- `main`: stable defense releases only
- `develop`: integrated and tested development code
- `feature/*`: one feature created from the latest `develop`

Do not commit directly to `main` or `develop`. Open a pull request into
`develop`, wait for CI, squash-merge, and delete the completed feature branch.
Create the `develop` to `main` pull request only after the final defense
rehearsal.

## Stop or Reset

Stop containers while preserving database and service volumes:

```bash
docker compose down
```

Completely reset all containers and named volumes:

```bash
docker compose down -v --remove-orphans
```

The reset command deletes PostgreSQL, Airflow, Grafana, and MLflow state. Use it
only when a clean environment is intended. If PostgreSQL credentials change,
perform a volume reset before restarting.

## Troubleshooting

Check service status and logs:

```bash
docker compose ps
docker compose logs --tail=100 api
docker compose logs --tail=100 airflow-api-server
docker compose logs --tail=100 airflow-scheduler
```

Confirm FastAPI readiness:

```bash
curl http://127.0.0.1:8000/health
```

List loaded DAGs:

```bash
docker compose exec airflow-scheduler airflow dags list
```

If a port is already occupied, stop the application using that port before
starting this Compose project. If a container was built before a code change,
run `docker compose up -d --build` again.
