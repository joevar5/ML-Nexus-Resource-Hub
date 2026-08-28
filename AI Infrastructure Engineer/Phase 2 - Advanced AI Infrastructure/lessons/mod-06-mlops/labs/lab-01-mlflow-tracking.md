# Lab 01: MLflow Tracking Server End-to-End

**Duration:** 60 min  **Prerequisites:** Docker

## Objective
Run an MLflow tracking server backed by Postgres metadata and S3-compatible artifact storage (Minio). Log experiments from a training script and explore them in the UI.

## Steps

### 1. compose.yaml
```yaml
services:
  pg:
    image: postgres:15
    environment: { POSTGRES_USER: mlflow, POSTGRES_PASSWORD: mlflow, POSTGRES_DB: mlflow }
    volumes: [pg:/var/lib/postgresql/data]
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: minioadmin, MINIO_ROOT_PASSWORD: minioadmin }
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio:/data]
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    depends_on: [pg, minio]
    ports: ["5000:5000"]
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    command: >
      mlflow server --host 0.0.0.0 --port 5000
        --backend-store-uri postgresql://mlflow:mlflow@pg:5432/mlflow
        --default-artifact-root s3://mlflow/
volumes: { pg: {}, minio: {} }
```

### 2. Start and create the bucket
```bash
docker compose up -d
sleep 5
# Create bucket via Minio CLI:
docker run --rm --network host minio/mc alias set local http://localhost:9000 minioadmin minioadmin
docker run --rm --network host minio/mc mb local/mlflow
```

### 3. Log from a script
```python
# train.py
import mlflow, mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-baseline")

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

for n in (10, 50, 200):
    with mlflow.start_run(run_name=f"rf-n{n}"):
        mlflow.log_param("n_estimators", n)
        model = RandomForestClassifier(n_estimators=n, random_state=42).fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")
```

### 4. Open UI
http://localhost:5000 → "iris-baseline" → 3 runs.

### 5. Compare runs
Select all 3 → "Compare" → see metric and param table.

### 6. Load a logged model back
```python
import mlflow.pyfunc
best = mlflow.search_runs(experiment_names=["iris-baseline"], order_by=["metrics.accuracy DESC"]).iloc[0]
m = mlflow.pyfunc.load_model(f"runs:/{best.run_id}/model")
print(m.predict([[5.1, 3.5, 1.4, 0.2]]))
```

## Validation
- [ ] All 3 runs visible in UI with accuracies.
- [ ] Artifacts (model.pkl) browsable in UI.
- [ ] Loading a logged model produces a valid prediction.

## Cleanup
```bash
docker compose down -v
```

## Troubleshooting
- **`OperationalError: connection refused`** — Postgres not ready. Wait 10s after `compose up`.
- **`NoSuchBucket: mlflow`** — Skipped the `mc mb local/mlflow` step.
- **Artifacts upload but UI shows download error** — Browser can't reach minio:9000 (the URL inside the artifact path). Use a local-host alias or run `mlflow ui` with `--artifacts-destination`.
