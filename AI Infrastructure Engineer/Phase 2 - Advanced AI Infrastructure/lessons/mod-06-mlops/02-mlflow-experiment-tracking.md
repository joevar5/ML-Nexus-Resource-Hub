# Lesson 02: MLflow Experiment Tracking

## Overview
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. This lesson focuses on MLflow Tracking, which enables you to log parameters, metrics, and artifacts from your ML experiments, making them reproducible and comparable.

**Duration:** 6-7 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, basic ML knowledge, understanding of ML lifecycle

## Learning Objectives
By the end of this lesson, you will be able to:
- Set up and configure MLflow Tracking
- Log experiments, parameters, and metrics
- Track models and artifacts
- Query and compare experiments
- Use MLflow UI for visualization
- Integrate MLflow with popular frameworks
- Implement best practices for experiment tracking

---

## 1. MLflow Architecture

### 1.1 MLflow Components

```
┌─────────────────────────────────────────────────────┐
│                 MLflow Platform                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐               │
│  │   Tracking   │  │    Models    │               │
│  │  • Params    │  │  • Packaging │               │
│  │  • Metrics   │  │  • Serving   │               │
│  │  • Artifacts │  │  • Registry  │               │
│  └──────────────┘  └──────────────┘               │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐               │
│  │   Projects   │  │   Plugins    │               │
│  │  • Packaging │  │  • Custom    │               │
│  │  • Repro     │  │  • Extensions│               │
│  └──────────────┘  └──────────────┘               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 1.2 MLflow Tracking Server Architecture

```
┌──────────────┐         ┌──────────────────┐
│  ML Scripts  │────────▶│ Tracking Server  │
│  (Training)  │         │  • REST API      │
└──────────────┘         │  • File Store    │
                         └──────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │                       │
              ┌─────▼─────┐         ┌──────▼────┐
              │  Backend  │         │  Artifact │
              │   Store   │         │   Store   │
              │(PostgreSQL│         │    (S3)   │
              │  /SQLite) │         └───────────┘
              └───────────┘
```

---

## 2. Setting Up MLflow

### 2.1 Installation

```bash
# Basic installation
pip install mlflow==2.9.0

# With extras
pip install mlflow[extras]  # includes sklearn, tensorflow integrations

# Verify installation
mlflow --version
```

### 2.2 Local Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Set tracking URI (local file store)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
mlflow.set_experiment("fraud_detection")

# Start a run
with mlflow.start_run(run_name="baseline_rf"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "random_forest")

    # Train model
    X_train, X_test, y_train, y_test = load_and_split_data()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### 2.3 Remote Tracking Server

```bash
# Start tracking server with PostgreSQL backend and S3 artifacts
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

# Or with SQLite and local artifacts (development)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

```python
# Connect to remote server
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("my_experiment")

# Everything else stays the same
with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", 0.8)
```

---

## 3. Logging Experiments

### 3.1 Parameters, Metrics, and Tags

```python
import mlflow
import numpy as np

with mlflow.start_run(run_name="experiment_001"):
    # Log single values
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("loss", 0.5)

    # Log multiple parameters at once
    params = {
        "optimizer": "adam",
        "dropout_rate": 0.2,
        "hidden_units": 128
    }
    mlflow.log_params(params)

    # Log multiple metrics
    metrics = {
        "train_accuracy": 0.95,
        "val_accuracy": 0.92,
        "test_accuracy": 0.91
    }
    mlflow.log_metrics(metrics)

    # Log metrics at multiple steps (for training curves)
    for epoch in range(10):
        train_loss = np.random.random()
        val_loss = np.random.random()

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Log tags for organization
    mlflow.set_tag("model_family", "neural_network")
    mlflow.set_tag("dataset_version", "v1.2.0")
    mlflow.set_tag("engineer", "alice@example.com")
```

### 3.2 Logging Artifacts

```python
import mlflow
import matplotlib.pyplot as plt
import json
import pandas as pd

with mlflow.start_run():
    # Log a file
    with open("config.json", "w") as f:
        json.dump({"param1": "value1"}, f)
    mlflow.log_artifact("config.json")

    # Log a directory
    mlflow.log_artifacts("output_dir", artifact_path="outputs")

    # Log a figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("plot.png")
    mlflow.log_artifact("plot.png")
    plt.close()

    # Log a DataFrame as CSV
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv("data.csv", index=False)
    mlflow.log_artifact("data.csv")

    # Log text content directly
    mlflow.log_text("Model training completed successfully", "status.txt")

    # Log dictionary as JSON
    metadata = {"model_version": "1.0", "framework": "sklearn"}
    mlflow.log_dict(metadata, "metadata.json")
```

### 3.3 Logging Models

```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow

# Scikit-learn model
with mlflow.start_run():
    model = train_sklearn_model()

    # Log model with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )

# PyTorch model
with mlflow.start_run():
    model = train_pytorch_model()

    mlflow.pytorch.log_model(
        model,
        "model",
        conda_env="conda.yaml",
        code_paths=["model_code.py"]
    )

# Custom Python model
class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        # Custom prediction logic
        return self.model.predict(model_input)

with mlflow.start_run():
    custom_model = CustomModel(sklearn_model)
    mlflow.pyfunc.log_model(
        artifact_path="custom_model",
        python_model=custom_model
    )
```

---

## 4. Organizing Experiments

### 4.1 Hierarchical Organization

```python
import mlflow

# Create experiments programmatically
experiment_id = mlflow.create_experiment(
    "deep_learning_experiments",
    artifact_location="s3://my-bucket/experiments",
    tags={"team": "ml-research", "project": "vision"}
)

# Use nested runs for hyperparameter sweeps
with mlflow.start_run(run_name="grid_search") as parent_run:
    mlflow.log_param("search_type", "grid")

    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [16, 32, 64]:
            with mlflow.start_run(nested=True) as child_run:
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("batch_size", batch_size)

                # Train model
                metrics = train_model(lr, batch_size)
                mlflow.log_metrics(metrics)

# Query parent run's children
from mlflow.tracking import MlflowClient

client = MlflowClient()
child_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
)
```

### 4.2 Tagging Strategy

```python
import mlflow
from datetime import datetime

def log_training_run(model, data_version, experiment_type):
    """Standardized training run with consistent tagging"""

    with mlflow.start_run():
        # Standard tags
        mlflow.set_tag("mlflow.runName", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("experiment_type", experiment_type)  # baseline, tuning, production
        mlflow.set_tag("model_architecture", model.__class__.__name__)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("engineer", get_current_user())

        # Environment tags
        mlflow.set_tag("environment", "development")
        mlflow.set_tag("python_version", sys.version)

        # Business tags
        mlflow.set_tag("use_case", "fraud_detection")
        mlflow.set_tag("priority", "high")

        # Train and log model
        # ... training code ...
```

---

## 5. Querying and Comparing Runs

### 5.1 Searching Runs

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("fraud_detection")

# Search runs with filters
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9 and params.model_type = 'random_forest'",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Convert to DataFrame
runs_df = pd.DataFrame([{
    "run_id": run.info.run_id,
    "accuracy": run.data.metrics.get("accuracy"),
    "f1_score": run.data.metrics.get("f1_score"),
    "n_estimators": run.data.params.get("n_estimators"),
    "start_time": run.info.start_time
} for run in runs])

print(runs_df)

# Get best run
best_run = runs[0]
print(f"Best run ID: {best_run.info.run_id}")
print(f"Best accuracy: {best_run.data.metrics['accuracy']}")
```

### 5.2 Comparing Runs

```python
def compare_runs(run_ids: list) -> pd.DataFrame:
    """Compare multiple runs"""
    client = MlflowClient()

    comparison = []
    for run_id in run_ids:
        run = client.get_run(run_id)

        comparison.append({
            "run_id": run_id[:8],
            "run_name": run.data.tags.get("mlflow.runName", ""),
            **run.data.params,
            **run.data.metrics
        })

    return pd.DataFrame(comparison)

# Usage
run_ids = ["abc123", "def456", "ghi789"]
comparison_df = compare_runs(run_ids)
print(comparison_df)

# Find runs with similar parameters
def find_similar_runs(reference_run_id, param_names):
    """Find runs with similar parameter values"""
    client = MlflowClient()

    ref_run = client.get_run(reference_run_id)
    ref_params = {p: ref_run.data.params[p] for p in param_names}

    # Build filter string
    filters = [f"params.{k} = '{v}'" for k, v in ref_params.items()]
    filter_string = " and ".join(filters)

    similar_runs = client.search_runs(
        experiment_ids=[ref_run.info.experiment_id],
        filter_string=filter_string
    )

    return similar_runs
```

---

## 6. MLflow UI

### 6.1 Starting the UI

```bash
# Start UI pointing to tracking server
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or if using remote tracking server
mlflow ui --backend-store-uri postgresql://user:pass@host/db

# Custom port
mlflow ui --port 5001
```

### 6.2 UI Features

**Key Features:**
- Compare runs side-by-side
- Visualize metrics over time
- Download artifacts
- Register models
- Search and filter runs
- View experiment hierarchy

---

## 7. Framework Integrations

### 7.1 Auto-logging

```python
import mlflow

# TensorFlow/Keras auto-logging
mlflow.tensorflow.autolog()

import tensorflow as tf
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)  # Automatically logged!

# Scikit-learn auto-logging
mlflow.sklearn.autolog()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Automatically logged!

# PyTorch Lightning auto-logging
import mlflow.pytorch
mlflow.pytorch.autolog()

import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader)  # Automatically logged!

# XGBoost auto-logging
mlflow.xgboost.autolog()

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)  # Automatically logged!
```

### 7.2 Custom Integration

```python
class MLflowCallback:
    """Custom callback for logging during training"""

    def __init__(self):
        self.run = None

    def on_train_begin(self, config):
        """Start MLflow run"""
        self.run = mlflow.start_run()
        mlflow.log_params(config)

    def on_epoch_end(self, epoch, metrics):
        """Log metrics after each epoch"""
        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, model):
        """Log model and end run"""
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

# Usage
callback = MLflowCallback()
train_model(callbacks=[callback])
```

---

## 8. Production Patterns

### 8.1 Experiment Tracking Wrapper

```python
import mlflow
from functools import wraps
import traceback

def track_experiment(experiment_name: str):
    """Decorator for automatic experiment tracking"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                try:
                    # Log function arguments
                    mlflow.log_params({
                        f"arg_{i}": str(arg)
                        for i, arg in enumerate(args)
                    })
                    mlflow.log_params(kwargs)

                    # Execute function
                    result = func(*args, **kwargs)

                    # Log success
                    mlflow.set_tag("status", "success")

                    return result

                except Exception as e:
                    # Log failure
                    mlflow.set_tag("status", "failed")
                    mlflow.log_text(traceback.format_exc(), "error.txt")
                    raise

        return wrapper
    return decorator

# Usage
@track_experiment("model_training")
def train_model(learning_rate, batch_size):
    # Training code
    model = ...
    metrics = {"accuracy": 0.95}
    mlflow.log_metrics(metrics)
    return model
```

### 8.2 Model Lineage Tracking

```python
class ModelLineageTracker:
    """Track complete model lineage"""

    def __init__(self):
        self.client = MlflowClient()

    def log_data_lineage(self, data_sources: dict):
        """Log data sources and versions"""
        for source, version in data_sources.items():
            mlflow.log_param(f"data_source_{source}", version)
            mlflow.set_tag(f"data_{source}_hash", compute_hash(version))

    def log_code_version(self):
        """Log code version information"""
        import subprocess

        # Git information
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            mlflow.set_tag("git_commit", commit)

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode().strip()
            mlflow.set_tag("git_branch", branch)
        except:
            pass

    def log_dependencies(self):
        """Log Python package versions"""
        import pkg_resources

        dependencies = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        }
        mlflow.log_dict(dependencies, "dependencies.json")

    def log_model_lineage(self, parent_model_uri: str = None):
        """Log parent model if retraining"""
        if parent_model_uri:
            mlflow.set_tag("parent_model", parent_model_uri)

    def track_complete_lineage(self, data_sources, parent_model=None):
        """Track all lineage information"""
        self.log_data_lineage(data_sources)
        self.log_code_version()
        self.log_dependencies()
        if parent_model:
            self.log_model_lineage(parent_model)

# Usage
tracker = ModelLineageTracker()

with mlflow.start_run():
    tracker.track_complete_lineage(
        data_sources={
            "training_data": "v2.1.0",
            "validation_data": "v2.1.0"
        },
        parent_model="models:/fraud_model/3"
    )

    # Train model...
```

---

## 9. Best Practices

### 9.1 Experiment Organization

✅ **DO:**
- Use descriptive experiment names
- Create separate experiments for different projects
- Use consistent naming conventions
- Tag runs with metadata
- Use nested runs for hyperparameter sweeps
- Clean up old experiments periodically

❌ **DON'T:**
- Mix different projects in one experiment
- Use generic names like "experiment1"
- Log sensitive data
- Leave experiments running indefinitely

### 9.2 What to Log

```python
# Comprehensive logging example
with mlflow.start_run():
    # 1. Configuration
    mlflow.log_params(config)
    mlflow.log_dict(full_config, "config.json")

    # 2. Data information
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", X_train.shape[1])

    # 3. Model architecture
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_params(model.get_params())

    # 4. Training metrics
    for epoch in range(epochs):
        mlflow.log_metrics(epoch_metrics, step=epoch)

    # 5. Final metrics
    mlflow.log_metrics(final_metrics)

    # 6. Model artifacts
    mlflow.sklearn.log_model(model, "model")

    # 7. Visualizations
    mlflow.log_figure(confusion_matrix_plot, "confusion_matrix.png")
    mlflow.log_figure(roc_curve_plot, "roc_curve.png")

    # 8. Additional artifacts
    mlflow.log_artifact("feature_importance.csv")
    mlflow.log_text(model_summary, "model_summary.txt")
```

---

## 10. Summary

Key takeaways:
- ✅ MLflow Tracking enables reproducible experiments
- ✅ Log parameters, metrics, models, and artifacts
- ✅ Use tags for organization and searchability
- ✅ Leverage auto-logging for quick setup
- ✅ Query and compare runs programmatically
- ✅ Track complete model lineage
- ✅ Follow consistent naming and tagging conventions

**Next Lesson:** [03 - Model Registry & Versioning](./03-model-registry-versioning.md)
