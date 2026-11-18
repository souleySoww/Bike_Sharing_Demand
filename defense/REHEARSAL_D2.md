# Defense 2 Rehearsal Runbook

Use `develop` for rehearsals. Use `main` only after the team approves the final
release pull request. The Defense 1 runbook (`defense/REHEARSAL.md`) still
applies: Defense 1 must keep working.

## Team Speaking Order

1. Yassine: architecture, model lifecycle, data quality dashboard, rollback,
   CI/CD.
2. Yahya: MLflow registry and the training DAG.
3. Ebelechkwu: drift and prediction dashboard and its alerts.
4. Vishal: training demo data and the clickable Data Docs links.

Every member should be able to explain the complete flow:

```text
CSV -> ingestion -> validation -> good/bad split -> prediction DAG -> API
    -> predictions with model_version -> Grafana
good data + labels -> training DAG -> candidate vs @champion on the fixed
    test set -> promote -> /reload-model -> new version in Grafana
```

## Preflight

Run this at least 30 minutes before the defense, then leave the stack running
so the dashboards have history:

```bash
git switch develop
git pull --ff-only origin develop
docker compose up -d --build
docker compose ps
docker compose exec airflow-scheduler python -m app.training.retrain_baseline
```

The last command creates v1 as `@champion`. Skip it if MLflow already has a
champion. Then unpause the three DAGs:

```bash
for dag in bike_data_ingestion bike_demand_prediction bike_demand_training; do
  docker compose exec airflow-scheduler airflow dags unpause "$dag"
done
```

Confirm these pages open:

- Streamlit http://127.0.0.1:8501, FastAPI http://127.0.0.1:8000/docs
- Airflow http://127.0.0.1:8080, MLflow http://127.0.0.1:5000
- Grafana http://127.0.0.1:3000: folders `data-quality` and `ml-monitoring`,
  and Alerting > Alert rules lists 4 rules without errors
- Data Docs http://127.0.0.1:8081 after the first ingestion

Check that `curl http://127.0.0.1:8000/model-status` reports version 1.

Send one test notification to each Teams channel from Grafana (Alerting >
Contact points > Test). An HTTP 200 is not enough: check that the message
appears in Teams.

Feed normal traffic for at least 30 minutes before the demo, so the PSI and
prediction panels have green history:

```bash
cp demo_data/defense2/traffic/*.csv data/raw_data/
```

These 30 files come from the data v1 was trained on: they never trigger
retraining and their PSI stays below 0.1. Do not use
`scripts.split_dataset` here: it samples the whole dataset, so its files
would trigger retraining and drift before the scripted steps.

Before each demo step, wait until `data/raw_data` is empty: ingestion picks
one random file per minute, so leftover traffic files delay the demo file.

## Demo Sequence

### 1. Architecture - Yassine

Show `docker compose ps` and the architecture slide: three serving
strategies (model as a service, model loaded from the registry, predictions
exposed in a database).

### 2. Data quality - Vishal and Yassine

```bash
cp demo_data/defense1/01_all_invalid.csv data/raw_data/
```

Within about two minutes:

- the Teams card in "Data Quality Alerts" with the "Open validation report"
  button, which opens the report of that file;
- the Data Quality dashboard: a red point at 100%, a red block on the
  criticality timeline, errors stacked by category;
- the two Grafana data quality rules fire (both are expected for this file).

### 3. Retraining and promotion - Yahya and Vishal

```bash
cp demo_data/defense2/training/*.csv data/raw_data/
```

Wait until the three files are ingested and one prediction cycle has run,
then trigger `bike_demand_training` in Airflow. Show:

- `load_data` reports 60 new rows;
- MLflow: the new version, its metrics, and the `@champion` alias moved;
- `curl http://127.0.0.1:8000/model-status` reports the new version;
- the Drift and Predictions dashboard: a new line on "Mean prediction per
  model version".

If the candidate is rejected, show the ML Alerts card with the reason: the
champion keeps serving. That is also a valid outcome to explain.

### 4. Drift - Ebelechkwu

```bash
cp demo_data/defense2/drift/*.csv data/raw_data/
```

Explain while the DAGs run: these rows come from the fixed test set, so the
training DAG never learns them. After two or three minutes the PSI panels
turn red, then the "Feature drift above PSI 0.25" card reaches ML Alerts with
the name of the feature.

### 5. Rollback - Yassine

```bash
docker compose exec airflow-scheduler python -m app.training.rollback --dry-run
docker compose exec airflow-scheduler python -m app.training.rollback
curl http://127.0.0.1:8000/model-status
```

Show the alias back on the previous version in MLflow and the API serving it.

### 6. CI/CD - Yassine

Show the last CI run on a pull request and the GitHub Release created by the
merge into `main`.

## Likely Questions

**Why compare on a fixed test set?** Both models are scored on the same data,
which neither has seen (course 5). A fresh test set would change at every run
and make the scores incomparable.

**When is a candidate promoted?** RMSE not worse than the champion's and
inference below 5 ms per row. Otherwise it is rejected, an ML alert explains
why, and the champion keeps serving.

**Why PSI, and why 0.1 and 0.25?** Same metric for numerical and categorical
features with the thresholds from course 5. Buckets under 30 rows are hidden
because PSI is noise on small samples.

**Why does temp drift before any demo file?** The baseline covers other
months: seasonality is real drift.

**Why does "Constant or unrealistic predictions" not fire in the demo?** It
looks at all predictions of the last 30 minutes, so it only fires when the
model itself is broken (constant, negative or far above any training
demand), not on a few identical requests among normal traffic. Show the rule
and its state Normal in Grafana.

**How do you roll back?** Move `@champion` to the previous version; the API
reloads without a restart and the drift baseline follows.

**What is the deployment strategy?** Recreate: the API swaps the model in
place. A shadow or canary deployment with `@challenger` would be the next
step.

## Fast Recovery

- A Grafana panel shows "No data": check the time range and that the DAGs
  are unpaused; PSI panels need 30 rows in a bucket.
- Grafana did not load a file: `docker compose logs grafana | grep -i error`,
  fix the file, then `docker compose restart grafana`.
- The API serves `local-baseline`: run `retrain_baseline`, then
  `curl -X POST http://127.0.0.1:8000/reload-model`.
- Training skips: fewer than 50 new rows; copy more training files with
  `python -m scripts.prepare_training_demo --start-row 60`.
- Ingestion seems idle: it takes one random file per minute; wait or trigger
  `bike_data_ingestion` by hand.
