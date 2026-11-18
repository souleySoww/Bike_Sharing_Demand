# Deploying on Google Cloud

The stack runs about 10 containers and needs 8 GB of RAM, so the Always Free
`e2-micro` (1 GB) is too small. Use the Google Cloud free trial credit
instead: an `e2-standard-2` (2 vCPU, 8 GB) with a 40 GB disk uses roughly a
third of the credit over the 90 days. Without an upgrade to a paid account,
nothing is charged when the credit ends; the server just stops.

Once set up, every merge into `main` is deployed automatically: the release
workflow uploads the released code over SSH and runs `deploy/deploy.sh`.
`.env`, `data/`, `mlflow-artifacts/` and the database volumes stay on the
server, so models, predictions and dashboards keep their history.

Only SSH is open to the internet. Airflow (`admin:admin`), MLflow (no login)
and the other UIs are reached through an SSH tunnel.

## 1. Create the server (once)

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install),
start the free trial in the console, then:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud compute instances create dsp-mlops \
  --zone=europe-west1-b \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=40GB
```

Set a budget alert (Billing > Budgets & alerts) at a few euros so any
unexpected cost is visible.

## 2. Prepare it (once)

```bash
gcloud compute scp deploy/setup_vm.sh dsp-mlops:~ --zone=europe-west1-b
gcloud compute ssh dsp-mlops --zone=europe-west1-b -- bash setup_vm.sh
```

This installs Docker and creates `/opt/dsp` with the dataset and a `.env`
holding random passwords. To receive Teams alerts, add the two webhooks to
`/opt/dsp/.env` on the server (never in Git).

## 3. Give GitHub Actions access (once)

Create a key used only for deployments:

```bash
ssh-keygen -t ed25519 -f dsp_deploy_key -N "" -C github-deploy
gcloud compute ssh dsp-mlops --zone=europe-west1-b -- \
  "echo '$(cat dsp_deploy_key.pub)' >> ~/.ssh/authorized_keys && whoami"
gcloud compute instances describe dsp-mlops --zone=europe-west1-b \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

Add three repository secrets (Settings > Secrets and variables > Actions),
then delete the local key files:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | The external IP printed above |
| `DEPLOY_USER` | The user name printed by `whoami` |
| `DEPLOY_SSH_KEY` | The whole content of `dsp_deploy_key` |

The IP stays the same while the server runs. If you stop the server, reserve
a static IP (VPC network > IP addresses) or update `DEPLOY_HOST`.

Until `DEPLOY_HOST` exists, the deploy job is skipped and releases still
work.

## 4. Deploy

Merge into `main`. The Release workflow runs lint and tests, publishes the
release, then deploys it. The first deployment builds the images (about 10
minutes), creates v1 as `@champion` and unpauses the three DAGs.

To deploy a branch by hand before merging (for example to test `develop`):

```bash
git archive --format=tar.gz -o release.tar.gz HEAD
gcloud compute scp release.tar.gz dsp-mlops:/tmp/dsp-release.tar.gz \
  --zone=europe-west1-b
gcloud compute ssh dsp-mlops --zone=europe-west1-b -- \
  "tar -xzf /tmp/dsp-release.tar.gz -C /tmp deploy/deploy.sh \
   && bash /tmp/deploy/deploy.sh"
```

## 5. Use it

Open a tunnel, keep it running, and use the usual local URLs:

```bash
gcloud compute ssh dsp-mlops --zone=europe-west1-b -- -N \
  -L 8501:localhost:8501 -L 8000:localhost:8000 -L 8080:localhost:8080 \
  -L 8081:localhost:8081 -L 3000:localhost:3000 -L 5000:localhost:5000
```

Streamlit http://localhost:8501, Grafana http://localhost:3000, Airflow
http://localhost:8080, MLflow http://localhost:5000. The Data Docs buttons in
Teams open http://localhost:8081, so they work while the tunnel is open.

Run commands on the server with
`gcloud compute ssh dsp-mlops --zone=europe-west1-b`, then `cd /opt/dsp`. The
Grafana password is `GRAFANA_ADMIN_PASSWORD` in `/opt/dsp/.env`.

To feed demo files, copy them into `/opt/dsp/data/raw_data/`, for example:

```bash
gcloud compute scp demo_data/defense1/01_all_invalid.csv \
  dsp-mlops:/opt/dsp/data/raw_data/ --zone=europe-west1-b
```

## Stop paying

```bash
gcloud compute instances stop dsp-mlops --zone=europe-west1-b    # pause
gcloud compute instances delete dsp-mlops --zone=europe-west1-b  # remove
```

A stopped server only costs its disk.
