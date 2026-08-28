# Lab 02: Model Registry and Stage Promotion

**Duration:** 60 min  **Prerequisites:** Lab 01; MLflow stack running

## Objective
Register a model in the MLflow Model Registry, transition between stages (Staging → Production), and load by stage in a serving script.

## Steps

### 1. Register a model from a run
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

best = mlflow.search_runs(experiment_names=["iris-baseline"], order_by=["metrics.accuracy DESC"]).iloc[0]
mv = mlflow.register_model(f"runs:/{best.run_id}/model", name="iris-classifier")
print(mv.version)
```

### 2. Transition stages via client
```python
from mlflow.tracking import MlflowClient
c = MlflowClient()
c.transition_model_version_stage("iris-classifier", version=mv.version, stage="Staging",
                                  archive_existing_versions=True)
```

### 3. Load by stage in a serving script
```python
import mlflow.pyfunc
m = mlflow.pyfunc.load_model("models:/iris-classifier/Staging")
print(m.predict([[5.1, 3.5, 1.4, 0.2]]))
```
This is the production pattern: app code references a stage, not a fixed version.

### 4. Promote to Production
```python
c.transition_model_version_stage("iris-classifier", mv.version, stage="Production",
                                  archive_existing_versions=True)
```

### 5. Webhook / event on transition (optional)
MLflow doesn't have first-party webhooks, but you can poll via the API or trigger via your CI on a tag.

### 6. UI tour
http://localhost:5000 → "Models" tab → "iris-classifier" → see version history and stage assignments.

## Validation
- [ ] Model appears in registry with the right version.
- [ ] Stage transitions work in UI and via client.
- [ ] `models:/iris-classifier/Production` URI resolves and predicts.

## Cleanup
```bash
# delete the registered model via UI or:
python -c "from mlflow.tracking import MlflowClient; MlflowClient().delete_registered_model('iris-classifier')"
```

## Troubleshooting
- **`Registered model not found`** — Spelling mismatch between register and load steps.
- **Stage transition silently rolls back** — Permissions issue when using Databricks-hosted MLflow; OSS MLflow allows any client.
- **`archive_existing_versions` deprecated** — In MLflow 2.9+ use the new "aliases" API instead of stages.
