# Exercise 02: Airflow Fundamentals — Build a Real DAG

**Duration:** 3 hours
**Difficulty:** Beginner+
**Prerequisites:** Exercise 01 + lab 01 (airflow running locally)

## Objective

Build an Airflow DAG that runs an end-to-end ML training pipeline: ingest from S3, validate, transform, train, evaluate, conditionally deploy. Cover TaskFlow API, XCom, branching, error handling, retries, SLAs, and dynamic task mapping.

## Why this matters

Airflow is the single most-used workflow engine in ML infra. Engineers who know its idioms cold (TaskFlow vs PythonOperator, XCom limits, branching tricks) ship pipelines that other engineers can extend without rewrites.

## Requirements

Build `recs_training_dag.py` with:

1. **6+ tasks** via TaskFlow API (`@task` decorator).
2. **Conditional branching** based on data quality results (deploy vs skip).
3. **Dynamic task mapping**: parallel feature engineering jobs per category.
4. **XCom passing** of intermediate artifacts (file paths, not data).
5. **Retries** with exponential backoff for flaky steps.
6. **SLA** of 4 hours per DAG run; SLA miss notifies Slack.
7. **Task groups** for visual organization.
8. **Sensors** that wait for upstream data drop on S3.

## Step-by-step

### Step 1 — Skeleton (15 min)
```python
# dags/recs_training.py
from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task, task_group
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.s3 import S3KeySensor

DEFAULT_ARGS = {
    "owner": "recs-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "sla": timedelta(hours=4),
    "email_on_failure": False,
    "on_failure_callback": notify_slack,
    "on_sla_miss_callback": notify_slack_sla,
}

with DAG(
    "recs_training",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "recs"],
) as dag:
    ...
```

### Step 2 — Wait for upstream data (15 min)
```python
wait_events = S3KeySensor(
    task_id="wait_events",
    bucket_key="events/{{ ds }}/_SUCCESS",
    bucket_name="recs-raw",
    poke_interval=300,
    timeout=3600,
    mode="reschedule",      # frees worker between pokes
)
```

### Step 3 — Ingest + validate (30 min)
```python
@task
def ingest(date: str) -> str:
    import pandas as pd
    df = pd.read_parquet(f"s3://recs-raw/events/{date}/")
    out = f"/tmp/events_{date}.parquet"
    df.to_parquet(out)
    return out

@task
def validate(path: str) -> dict:
    # Quick checks; return summary
    import pandas as pd
    df = pd.read_parquet(path)
    return {
        "rows": len(df),
        "schema_ok": set(df.columns) >= {"user_id","item_id","ts","event_type"},
        "freshness_ok": pd.Timestamp.utcnow() - df["ts"].max() < pd.Timedelta("3h"),
    }
```

### Step 4 — Branching on validation (20 min)
```python
@task.branch
def decide(validation: dict) -> str:
    if not (validation["schema_ok"] and validation["freshness_ok"] and validation["rows"] > 10_000):
        return "skip"
    return "transform"

@task
def skip() -> None:
    print("data quality insufficient — skipping training")
```

### Step 5 — Dynamic task mapping for feature engineering (30 min)
```python
@task
def list_categories(path: str) -> list[str]:
    import pandas as pd
    return list(pd.read_parquet(path)["category"].unique())

@task
def features_for_category(category: str, events_path: str) -> str:
    # produce per-category feature parquet
    ...
    return f"/tmp/features_{category}.parquet"

# Dynamic mapping:
categories = list_categories(events_path)
features = features_for_category.partial(events_path=events_path).expand(category=categories)
```

### Step 6 — Train + evaluate (30 min)
```python
@task
def train(feature_paths: list[str]) -> str:
    import joblib
    # combine, train
    model_path = f"/tmp/model_{ds}.joblib"
    joblib.dump(model, model_path)
    return model_path

@task
def evaluate(model_path: str) -> dict:
    return {"auc": 0.85, "ndcg": 0.42}
```

### Step 7 — Conditional deploy (15 min)
```python
@task.branch
def deploy_decision(metrics: dict) -> str:
    return "deploy" if metrics["auc"] >= 0.80 else "alert_regression"

@task
def deploy(model_path: str) -> None: ...

@task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
def alert_regression() -> None: ...
```

### Step 8 — Task groups (10 min)
```python
@task_group(group_id="quality_gate")
def quality_gate(events_path):
    v = validate(events_path)
    return decide(v)
```

### Step 9 — Wire it all (15 min)
```python
events = ingest(date="{{ ds }}")
decision = quality_gate(events)
features = features_for_category.partial(events_path=events).expand(category=list_categories(events))
model = train(features) >> decision
metrics = evaluate(model)
deploy_decision(metrics) >> [deploy(model), alert_regression()]
```

## Deliverables

1. `recs_training.py` DAG.
2. Working Airflow run via `airflow dags trigger recs_training`.
3. Screenshots of: graph view, gantt view, dynamic task mapping expansion.
4. `RUNBOOK.md`: how to backfill, how to clear a failed run, how to interpret SLA misses.

## Validation

- [ ] DAG appears in UI without import errors.
- [ ] Manual trigger succeeds end-to-end.
- [ ] Feature engineering generates 1 task per category (verify dynamic mapping in Graph view).
- [ ] Failing the validation step deliberately routes to `skip` not `transform`.
- [ ] Backfilling 5 days produces 5 successful runs.

## Stretch goals

- Add **Datasets** so the DAG triggers on the upstream events landing (event-driven, not time-driven).
- Add a **DAG-level test**: `pytest` that imports the DAG, validates structure, dry-runs.
- Add **task-level instrumentation**: emit Prometheus metrics from each task via `prometheus_client.start_http_server`.

## Common pitfalls

- **XCom for large data** — XCom backend (often DB) caps payloads. Pass file paths or S3 URIs.
- **`mode="poke"` for sensors** — Occupies a worker slot the whole time. Use `mode="reschedule"` to free it between pokes.
- **`task.branch` returns wrong task ID** — Must match exactly. Typos silently break the branch.
- **Backfill creates infinite runs** — Bound with `catchup=False` and `max_active_runs=1` to avoid concurrency issues.
