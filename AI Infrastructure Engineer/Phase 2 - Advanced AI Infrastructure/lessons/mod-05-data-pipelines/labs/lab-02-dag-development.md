# Lab 02: Author Your First DAG

**Duration:** 75 min  **Prerequisites:** Lab 01 complete; Airflow running

## Objective
Build an end-to-end DAG that mimics a real ML pipeline: download data, transform it, train a small model, and write metrics back. Use TaskFlow API and demonstrate dependencies, retries, and XCom.

## Steps

### 1. Project DAG
```python
# dags/iris_pipeline.py
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task

DEFAULTS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=15),
}

with DAG(
    dag_id="iris_pipeline",
    description="download → transform → train → report",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=DEFAULTS,
    tags=["lab", "iris"],
) as dag:

    @task
    def download() -> str:
        from sklearn.datasets import load_iris
        import pandas as pd
        df = pd.DataFrame(load_iris(as_frame=True).frame)
        out = "/tmp/iris.parquet"
        df.to_parquet(out)
        return out

    @task
    def transform(path: str) -> str:
        import pandas as pd
        df = pd.read_parquet(path)
        df = df[df["sepal length (cm)"] > 4.5]
        out = "/tmp/iris_clean.parquet"
        df.to_parquet(out)
        return out

    @task
    def train(path: str) -> dict:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(path)
        X = df.drop(columns=["target"])
        y = df["target"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
        return {"accuracy": float(accuracy_score(y_te, model.predict(X_te)))}

    @task
    def report(metrics: dict) -> None:
        print(f"final metrics: {json.dumps(metrics)}")

    report(train(transform(download())))
```

### 2. Drop into the dags/ folder and watch it appear
The scheduler scans every 30s.

### 3. Trigger and watch
UI → iris_pipeline → Trigger DAG with logical date today.

### 4. Observe XCom flow
Click into the `train` task → XCom tab → see the dict passed to `report`.

### 5. Force a failure to test retries
Edit `transform` to `raise RuntimeError("synthetic")`. Trigger. Watch it retry twice before final fail.

### 6. Convert to use TaskGroups for clarity
```python
from airflow.utils.task_group import TaskGroup
with TaskGroup("data") as data_grp:
    raw = download(); clean = transform(raw)
report(train(clean))
```

## Validation
- [ ] DAG appears in UI.
- [ ] All 4 tasks run successfully in order.
- [ ] XCom from `train` is visible.
- [ ] Forced failure produces 2 retries.

## Cleanup
```bash
rm ~/airflow/dags/iris_pipeline.py
```

## Troubleshooting
- **DAG doesn't appear** — Syntax error: `airflow dags list-import-errors` shows them.
- **`@task` not recognized** — Older Airflow version; `from airflow.decorators import task` requires 2.0+.
- **XCom too large** — Default backend limits to 48KB; use S3/GCS backend or write file paths through XCom and data to object storage.
