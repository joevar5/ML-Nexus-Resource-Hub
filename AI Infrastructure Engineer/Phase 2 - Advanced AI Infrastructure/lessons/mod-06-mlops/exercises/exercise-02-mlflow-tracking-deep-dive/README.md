# Exercise 02: MLflow Tracking Deep Dive

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Python + MLflow installed

## Objective

Use MLflow at the depth a production team requires: backend-aware experiment naming, autologging, custom metrics, model signatures, model packaging (`mlflow.pyfunc`), and a tracking server backed by Postgres + S3.

## Why this matters

Most teams stop at "we log to MLflow." Teams that use MLflow well treat it as the canonical source of truth for "what experiments ran, what artifacts were produced, who can serve them." This exercise pushes from casual use to professional use.

## Requirements

1. Run a backed MLflow tracking server (Postgres + S3/Minio).
2. Log a model with `autolog` + custom metrics + custom artifacts + model signature.
3. Use `mlflow.pyfunc` to package a non-standard model.
4. Query MLflow programmatically: find best run, list runs by tag, compare runs.
5. Demonstrate parent-child run relationships (HPO sweep).
6. Demonstrate **MLflow Recipes** or a custom orchestrator wrapping MLflow.

## Step-by-step

### Step 1 — Backed tracking server (per mod-06 lab 01) (15 min)
Refer to mod-06 lab 01 docker-compose for the full setup.

### Step 2 — Autolog (15 min)
```python
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("recs-baseline")
mlflow.sklearn.autolog(log_models=False)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="autolog-demo"):
    model = GradientBoostingClassifier(n_estimators=100).fit(Xtr, ytr)
```
Open UI — all hyperparameters + metrics + signature auto-captured.

### Step 3 — Custom metrics + artifacts (30 min)
```python
import json
with mlflow.start_run(run_name="custom-metrics"):
    model = ...
    mlflow.log_metric("custom_business_metric", 0.42)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_dict({"thresholds": [0.3, 0.5, 0.7]}, "thresholds.json")
    mlflow.set_tag("team", "recs-platform")
    mlflow.set_tag("data_version", "2026-05-01")
```

### Step 4 — Model signature + log_model (30 min)
```python
from mlflow.models.signature import infer_signature
sig = infer_signature(Xtr, model.predict(Xtr))

mlflow.sklearn.log_model(model, "model", signature=sig, input_example=Xtr[:5])
```
Signature serves two purposes: documentation in UI, and runtime input validation when served.

### Step 5 — Pyfunc wrapper for non-standard model (45 min)
For a model that isn't directly supported by `mlflow.<flavor>` (e.g., XGBoost + custom preprocessing):
```python
import mlflow.pyfunc

class WrappedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.preproc = joblib.load(context.artifacts["preproc"])
        self.model   = joblib.load(context.artifacts["model"])
    def predict(self, context, model_input):
        return self.model.predict(self.preproc.transform(model_input))

mlflow.pyfunc.log_model(
    artifact_path="combined",
    python_model=WrappedModel(),
    artifacts={"preproc": "/tmp/preproc.joblib", "model": "/tmp/model.joblib"},
    conda_env={...},
    signature=sig,
)

# Loading back:
loaded = mlflow.pyfunc.load_model("runs:/<id>/combined")
loaded.predict(Xte)
```

### Step 6 — Programmatic queries (30 min)
```python
client = mlflow.MlflowClient()
runs = client.search_runs(experiment_ids=[exp_id],
                          filter_string="metrics.auc > 0.9 AND tags.team = 'recs-platform'",
                          order_by=["metrics.auc DESC"], max_results=10)
for r in runs: print(r.info.run_id, r.data.metrics.get("auc"))
```

### Step 7 — Parent-child runs for HPO (30 min)
```python
with mlflow.start_run(run_name="hpo-parent") as parent:
    for lr in (0.01, 0.03, 0.1):
        for d in (3, 6, 10):
            with mlflow.start_run(run_name=f"trial-lr{lr}-d{d}", nested=True):
                mlflow.log_params({"lr": lr, "depth": d})
                # train + log metrics
                mlflow.log_metric("auc", auc)
```
UI shows parent + collapsible children.

### Step 8 — Workflow integration (15 min)
Wrap the training script in an Airflow DAG that:
- Reads previous best AUC from MLflow.
- Runs training.
- If new AUC > previous best + 0.005, registers new model version. Else logs and exits.

## Deliverables

1. Tracking server config + at least 3 experiments populated.
2. Pyfunc-wrapped model with preprocessing.
3. Programmatic query script that finds + summarizes best runs.
4. Airflow DAG with auto-registration on improvement.
5. `MLFLOW_PLAYBOOK.md`: how to use MLflow on your team (tags, run naming, experiment lifecycle).

## Validation

- [ ] All experiments visible in UI with metrics + params + artifacts.
- [ ] Pyfunc model loads and predicts correctly.
- [ ] Query returns runs in expected order.
- [ ] HPO parent-child structure visible in UI.

## Stretch goals

- Add **Optuna** for principled HPO with MLflow integration.
- Use **MLflow Recipes** to standardize the regression / classification workflow.
- Build a **model card generator** from MLflow metadata.

## Common pitfalls

- **No experiment names** — Default "Default" experiment becomes a graveyard. Always `set_experiment`.
- **Logging full datasets as artifacts** — Bloats MLflow storage. Log a sample + reference to S3 instead.
- **Tag drift** — Without conventions, every run has different tag schemas. Document standard tags in `MLFLOW_PLAYBOOK.md`.
- **Tracking server SQLite in production** — Backend store should be Postgres for concurrent writes.
