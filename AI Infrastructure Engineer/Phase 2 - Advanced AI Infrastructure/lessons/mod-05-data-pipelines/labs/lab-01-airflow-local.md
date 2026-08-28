# Lab 01: Run Airflow Locally with docker-compose

**Duration:** 60 min  **Prerequisites:** Docker + Compose

## Objective
Stand up a working Airflow stack (scheduler + webserver + worker + Postgres) locally with docker-compose. Confirm you can author and execute a trivial DAG.

## Steps

### 1. Fetch the official quick-start compose file
```bash
mkdir -p ~/airflow && cd ~/airflow
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.1/docker-compose.yaml'
echo "AIRFLOW_UID=$(id -u)" > .env
mkdir -p ./dags ./logs ./plugins ./config
```

### 2. Initialize the metadata database
```bash
docker compose up airflow-init
```
First-time init takes ~1-2 min; expect "Admin user airflow created" at the end.

### 3. Bring up the full stack
```bash
docker compose up -d
docker compose ps
```
All 6+ services should be healthy.

### 4. Open the UI
http://localhost:8080  user `airflow` / pass `airflow`. Several example DAGs are pre-loaded.

### 5. Drop in a hello DAG
```python
# dags/hello.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("hello", start_date=datetime(2026, 1, 1), schedule=None, catchup=False) as dag:
    BashOperator(task_id="say_hi", bash_command="echo hello from airflow")
```
Refresh UI → `hello` appears. Trigger it manually; watch logs.

### 6. CLI usage
```bash
docker compose exec airflow-worker airflow dags list
docker compose exec airflow-worker airflow dags trigger hello
docker compose exec airflow-worker airflow tasks list hello
```

## Validation
- [ ] `docker compose ps` shows scheduler, webserver, worker, triggerer, postgres all healthy.
- [ ] `hello` DAG runs to success.
- [ ] Logs visible in UI for each task.

## Cleanup
```bash
docker compose down -v
```

## Troubleshooting
- **`Permission denied` on logs/** — fix UID: `echo "AIRFLOW_UID=$(id -u)" > .env` and `chown -R $(id -u):0 logs/`.
- **Webserver returns 502** — Scheduler hasn't finished init yet; wait 30s.
- **No example DAGs in UI** — Set `AIRFLOW__CORE__LOAD_EXAMPLES=true` (default in the official compose).
