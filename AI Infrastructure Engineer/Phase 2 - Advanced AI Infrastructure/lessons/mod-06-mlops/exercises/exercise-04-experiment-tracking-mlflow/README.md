# Exercise 04: Experiment Tracking and Model Registry with MLflow

**Estimated Time**: 30-38 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, MLflow, PostgreSQL, S3/MinIO, Docker

## Overview

Build a production-grade ML experiment tracking and model registry system using MLflow. Implement centralized experiment tracking, model versioning, A/B testing framework, model performance monitoring, and automated model lifecycle management (staging → production → archived). This exercise teaches MLOps best practices for managing the complete ML model lifecycle.

In production ML platforms, experiment tracking and model registry are critical for:
- **Reproducibility**: Recreate any model from any experiment
- **Collaboration**: Share experiments across data science teams
- **Governance**: Track who deployed which model when
- **Comparison**: Compare 100+ experiments to find best model
- **Lifecycle Management**: Promote models through stages safely

## Learning Objectives

By completing this exercise, you will:

1. **Set up MLflow tracking server** with PostgreSQL backend and S3 artifact storage
2. **Implement experiment tracking** for model training runs
3. **Build model registry** with versioning and stage transitions
4. **Create A/B testing framework** for model comparison
5. **Implement automated model promotion** based on metrics
6. **Build model serving** with version management
7. **Monitor model performance** with drift detection

## Business Context

**Real-World Scenario**: Your data science team runs 500+ ML experiments per month across 10 models. Current problems:

- **Lost experiments**: No tracking, can't reproduce model from 2 months ago
- **Scattered artifacts**: Models saved to laptops, S3, random servers
- **Manual promotion**: Data scientists manually copy models to production
- **No comparison**: Can't easily compare experiment results
- **Version conflicts**: Multiple versions in production, confusion about which is active
- **No rollback**: Can't quickly rollback to previous model version

Your task: Build MLflow-based system that:
- Tracks all experiments with parameters, metrics, artifacts
- Stores models in centralized registry with versioning
- Automates promotion: challenger models automatically promoted if metrics better
- Provides A/B testing framework for comparing model versions
- Enables one-click rollback to previous version
- Tracks model lineage (data, code, hyperparameters used)

## Project Structure

```
exercise-04-experiment-tracking-mlflow/
├── README.md
├── requirements.txt
├── docker-compose.yaml              # MLflow server, Postgres, MinIO
├── config/
│   ├── mlflow-server.conf
│   └── model-configs.yaml
├── src/
│   └── mlflow_platform/
│       ├── __init__.py
│       ├── tracking/
│       │   ├── __init__.py
│       │   ├── experiment_tracker.py    # Experiment tracking wrapper
│       │   └── auto_logger.py           # Automatic logging decorators
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── model_registry.py        # Model registry operations
│       │   ├── version_manager.py       # Version lifecycle management
│       │   └── promotion_policy.py      # Auto-promotion logic
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── model_server.py          # FastAPI model serving
│       │   └── ab_testing.py            # A/B testing framework
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── performance_tracker.py   # Track model metrics
│       │   └── drift_detector.py        # Detect data/concept drift
│       └── cli/
│           ├── __init__.py
│           └── mlflow_cli.py            # CLI for model operations
├── examples/
│   ├── training/
│   │   ├── train_sklearn_model.py
│   │   ├── train_pytorch_model.py
│   │   └── train_with_hyperopt.py
│   ├── serving/
│   │   ├── serve_model.py
│   │   └── ab_test_models.py
│   └── monitoring/
│       └── track_production_metrics.py
├── tests/
│   ├── test_tracking.py
│   ├── test_registry.py
│   ├── test_serving.py
│   └── test_promotion.py
└── docs/
    ├── DESIGN.md
    ├── MODEL_LIFECYCLE.md
    └── API_REFERENCE.md
```

## Requirements

### Functional Requirements

1. **Experiment Tracking**:
   - Log parameters, metrics, tags
   - Store artifacts (models, plots, data samples)
   - Track code version (Git commit hash)
   - Support nested runs (hyperparameter tuning)

2. **Model Registry**:
   - Register models with versions
   - Stage management (None → Staging → Production → Archived)
   - Model metadata (description, tags, performance metrics)
   - Model lineage (which experiment produced this model)

3. **Model Serving**:
   - Load models by name and version/stage
   - A/B testing between versions
   - Traffic splitting (90% v1, 10% v2)
   - Fallback to previous version on errors

4. **Automated Promotion**:
   - Compare challenger vs champion metrics
   - Auto-promote if challenger better by >5%
   - Require manual approval for production
   - Rollback on production errors

5. **Monitoring**:
   - Track prediction latency
   - Monitor prediction distribution (drift detection)
   - Alert on performance degradation
   - Compare production metrics vs training metrics

### Non-Functional Requirements

- **Scalability**: Support 1000+ experiments, 100+ models
- **Performance**: Model loading <2s, inference <100ms
- **Reliability**: Model registry 99.9% available
- **Auditability**: Full audit trail of model changes

## Implementation Tasks

### Task 1: MLflow Server Setup (5-6 hours)

Set up MLflow tracking server with backend storage.

```yaml
# docker-compose.yaml

version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  mlflow-server:
    image: python:3.9-slim
    command: >
      bash -c "pip install mlflow boto3 psycopg2-binary &&
               mlflow server
               --backend-store-uri postgresql://mlflow:mlflow123@postgres/mlflow
               --default-artifact-root s3://mlflow/artifacts
               --host 0.0.0.0
               --port 5000"
    environment:
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio123
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - minio

volumes:
  postgres_data:
  minio_data:
```

**Startup script**:

```bash
#!/bin/bash
# setup.sh

set -e

echo "Starting MLflow infrastructure..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

echo "Creating MinIO bucket..."
docker-compose exec -T minio mc alias set myminio http://localhost:9000 minio minio123
docker-compose exec -T minio mc mb myminio/mlflow

echo "✅ MLflow server ready at http://localhost:5000"
echo "✅ MinIO console at http://localhost:9001 (minio/minio123)"
```

**Acceptance Criteria**:
- ✅ MLflow server accessible at http://localhost:5000
- ✅ PostgreSQL backend for metadata
- ✅ S3 (MinIO) for artifacts
- ✅ UI shows experiments and models

---

### Task 2: Experiment Tracking Wrapper (6-8 hours)

Build high-level experiment tracking wrapper.

```python
# src/mlflow_platform/tracking/experiment_tracker.py

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import os
import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """
    High-level wrapper for MLflow experiment tracking

    Simplifies common tracking operations and enforces best practices
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "default"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_name = experiment_name

        # Create experiment if doesn't exist
        try:
            self.experiment = self.client.get_experiment_by_name(experiment_name)
            self.experiment_id = self.experiment.experiment_id
        except:
            self.experiment_id = self.client.create_experiment(experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Start MLflow run

        Auto-logs:
        - Git commit hash
        - User
        - Timestamp
        - Python version

        Args:
            run_name: Human-readable run name
            tags: Additional tags
            description: Run description

        Returns:
            Run ID
        """
        # TODO: Auto-detect git commit
        git_commit = self._get_git_commit()

        # TODO: Build default tags
        default_tags = {
            "mlflow.user": os.environ.get("USER", "unknown"),
            "mlflow.source.git.commit": git_commit,
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        }

        if tags:
            default_tags.update(tags)

        # TODO: Start run
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags
        )

        if description:
            self.client.set_tag(run.info.run_id, "mlflow.note.content", description)

        return run.info.run_id

    def log_params(self, run_id: str, params: Dict[str, Any]):
        """
        Log parameters

        Handles nested dicts by flattening
        """
        # TODO: Flatten nested params
        flat_params = self._flatten_dict(params)

        # TODO: Log to MLflow
        for key, value in flat_params.items():
            self.client.log_param(run_id, key, value)

    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics

        Args:
            metrics: {"accuracy": 0.95, "loss": 0.1}
            step: Training step/epoch (for time-series metrics)
        """
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value, step=step)

    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: Optional[str] = None
    ):
        """
        Log artifact (file or directory)

        Args:
            local_path: Path to local file/directory
            artifact_path: Path in artifact store (optional)
        """
        self.client.log_artifact(run_id, local_path, artifact_path)

    def log_model(
        self,
        run_id: str,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        conda_env: Optional[Dict] = None,
        signature: Optional[Any] = None
    ):
        """
        Log model artifact

        Args:
            model: Model object (sklearn, pytorch, etc.)
            registered_model_name: Register to model registry with this name
            conda_env: Conda environment for model
            signature: Model signature (input/output schema)
        """
        # TODO: Detect model type and use appropriate log function
        if hasattr(model, 'predict'):  # sklearn-like
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                conda_env=conda_env,
                signature=signature
            )
        # Add support for PyTorch, TensorFlow, etc.

    def log_figure(self, run_id: str, figure, artifact_file: str):
        """
        Log matplotlib/plotly figure

        Args:
            figure: Matplotlib Figure or Plotly Figure
            artifact_file: Filename (e.g., "confusion_matrix.png")
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / artifact_file
            figure.savefig(filepath)
            self.log_artifact(run_id, str(filepath))

    def end_run(self, run_id: str, status: str = "FINISHED"):
        """
        End run

        Args:
            status: FINISHED, FAILED, KILLED
        """
        self.client.set_terminated(run_id, status)

    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare multiple runs

        Returns:
            {
                "run_1": {"accuracy": 0.95, "f1": 0.92},
                "run_2": {"accuracy": 0.93, "f1": 0.90}
            }
        """
        comparison = {}

        for run_id in run_ids:
            run = self.client.get_run(run_id)
            metrics = run.data.metrics

            if metric_keys:
                metrics = {k: v for k, v in metrics.items() if k in metric_keys}

            comparison[run_id] = metrics

        return comparison

    def get_best_run(
        self,
        metric_name: str,
        order_by: str = "DESC",
        filter_string: Optional[str] = None
    ) -> Dict:
        """
        Get best run based on metric

        Args:
            metric_name: Metric to optimize (e.g., "accuracy")
            order_by: DESC (maximize) or ASC (minimize)
            filter_string: MLflow filter query

        Returns:
            Best run info and metrics
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric_name} {order_by}"],
            max_results=1
        )

        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
        return None

    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        import subprocess
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            return commit
        except:
            return "unknown"

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict:
        """
        Flatten nested dictionary

        {"model": {"lr": 0.01}} → {"model.lr": 0.01}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

```python
# src/mlflow_platform/tracking/auto_logger.py

from functools import wraps
import mlflow
from typing import Callable, Dict, Any

def mlflow_track(
    experiment_name: str,
    log_params: bool = True,
    log_metrics: bool = True,
    log_model: bool = False
):
    """
    Decorator to automatically track function as MLflow run

    Usage:
        @mlflow_track(experiment_name="fraud_detection")
        def train_model(X, y, lr=0.01, epochs=10):
            model = train(X, y, lr, epochs)
            metrics = {"accuracy": 0.95}
            return model, metrics
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: Start run
            with mlflow.start_run(run_name=func.__name__):
                mlflow.set_experiment(experiment_name)

                # TODO: Log function parameters as params
                if log_params:
                    mlflow.log_params(kwargs)

                # TODO: Execute function
                result = func(*args, **kwargs)

                # TODO: Auto-log metrics if result is tuple (model, metrics)
                if log_metrics and isinstance(result, tuple) and len(result) == 2:
                    model, metrics = result
                    if isinstance(metrics, dict):
                        mlflow.log_metrics(metrics)

                    if log_model:
                        mlflow.sklearn.log_model(model, "model")

                return result

        return wrapper
    return decorator
```

**Acceptance Criteria**:
- ✅ Start/end runs with auto-logging
- ✅ Log params, metrics, artifacts
- ✅ Compare multiple runs
- ✅ Find best run by metric
- ✅ Auto-track git commit

---

### Task 3: Model Registry (7-9 hours)

Implement model registry with version management.

```python
# src/mlflow_platform/registry/model_registry.py

from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class ModelStage(Enum):
    """Model lifecycle stages"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class ModelRegistry:
    """
    High-level wrapper for MLflow Model Registry

    Manages model versioning and lifecycle
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Register model from run

        Args:
            model_uri: URI to model artifact (runs:/<run_id>/model)
            model_name: Name in registry
            description: Model description
            tags: Additional tags

        Returns:
            Model version info
        """
        # TODO: Create registered model if doesn't exist
        try:
            self.client.get_registered_model(model_name)
        except:
            self.client.create_registered_model(
                model_name,
                description=description,
                tags=tags
            )

        # TODO: Create model version
        model_version = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=self._extract_run_id(model_uri),
            tags=tags or {}
        )

        # TODO: Update description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )

        return {
            "name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "status": model_version.status
        }

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing_versions: bool = True
    ):
        """
        Transition model version to new stage

        Args:
            archive_existing_versions: Archive old versions in target stage
        """
        # TODO: Archive existing versions in target stage
        if archive_existing_versions and stage != ModelStage.ARCHIVED:
            current_versions = self.client.get_latest_versions(
                model_name,
                stages=[stage.value]
            )
            for v in current_versions:
                self.client.transition_model_version_stage(
                    model_name,
                    v.version,
                    ModelStage.ARCHIVED.value
                )

        # TODO: Transition to new stage
        self.client.transition_model_version_stage(
            model_name,
            version,
            stage.value
        )

    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Dict:
        """
        Get specific model version or latest in stage

        Args:
            version: Specific version number
            stage: Get latest version in this stage

        Returns:
            Model version metadata
        """
        if version:
            # Get specific version
            mv = self.client.get_model_version(model_name, version)
        elif stage:
            # Get latest version in stage
            versions = self.client.get_latest_versions(
                model_name,
                stages=[stage.value]
            )
            if not versions:
                raise ValueError(f"No model in {stage.value} stage")
            mv = versions[0]
        else:
            raise ValueError("Must specify version or stage")

        return {
            "name": mv.name,
            "version": mv.version,
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "source": mv.source,
            "status": mv.status,
            "description": mv.description,
            "tags": mv.tags
        }

    def list_model_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None
    ) -> List[Dict]:
        """List all versions of model"""
        filter_string = None
        if stage:
            filter_string = f"current_stage='{stage.value}'"

        versions = self.client.search_model_versions(
            filter_string=f"name='{model_name}'" +
                         (f" AND {filter_string}" if filter_string else "")
        )

        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000)
            }
            for v in versions
        ]

    def delete_model_version(self, model_name: str, version: str):
        """Delete specific model version"""
        self.client.delete_model_version(model_name, version)

    def set_model_version_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str
    ):
        """Set tag on model version"""
        self.client.set_model_version_tag(model_name, version, key, value)

    def _extract_run_id(self, model_uri: str) -> str:
        """Extract run ID from model URI"""
        # URI format: runs:/<run_id>/model
        if model_uri.startswith("runs:/"):
            return model_uri.split("/")[1]
        return None
```

**Acceptance Criteria**:
- ✅ Register models from runs
- ✅ Transition between stages
- ✅ Get model by version or stage
- ✅ List all versions
- ✅ Archive old versions

---

### Task 4: Automated Model Promotion (6-7 hours)

Build automated promotion logic.

```python
# src/mlflow_platform/registry/promotion_policy.py

from dataclasses import dataclass
from typing import Dict, Optional, List
from .model_registry import ModelRegistry, ModelStage
import logging

logger = logging.getLogger(__name__)

@dataclass
class PromotionCriteria:
    """Criteria for auto-promotion"""
    metric_name: str
    min_value: Optional[float] = None  # Minimum metric value
    improvement_threshold: float = 0.05  # Must be 5% better than champion
    max_latency_ms: Optional[float] = None  # Max inference latency
    require_approval: bool = False  # Require manual approval

class ModelPromoter:
    """
    Automate model promotion based on policies

    Workflow:
    1. New model registered → goes to "None" stage
    2. If passes criteria → promote to "Staging"
    3. Manual/auto validation in staging
    4. If better than production → promote to "Production"
    5. Old production → "Archived"
    """

    def __init__(
        self,
        registry: ModelRegistry,
        promotion_criteria: Dict[str, PromotionCriteria]
    ):
        self.registry = registry
        self.criteria = promotion_criteria

    def evaluate_promotion(
        self,
        model_name: str,
        challenger_version: str,
        metrics: Dict[str, float]
    ) -> Dict:
        """
        Evaluate if challenger should be promoted

        Args:
            challenger_version: New model version
            metrics: Performance metrics from validation

        Returns:
            {
                "should_promote": True/False,
                "target_stage": "Staging"/"Production",
                "reason": "...",
                "comparison": {...}
            }
        """
        criteria = self.criteria.get(model_name)
        if not criteria:
            return {
                "should_promote": False,
                "reason": "No promotion criteria defined"
            }

        # TODO: Check minimum metric value
        metric_value = metrics.get(criteria.metric_name)
        if metric_value is None:
            return {
                "should_promote": False,
                "reason": f"Metric {criteria.metric_name} not found"
            }

        if criteria.min_value and metric_value < criteria.min_value:
            return {
                "should_promote": False,
                "reason": f"{criteria.metric_name} {metric_value} < {criteria.min_value}"
            }

        # TODO: Compare to current production model
        try:
            production_model = self.registry.get_model_version(
                model_name,
                stage=ModelStage.PRODUCTION
            )

            # Get production metrics (from model tags/run)
            production_metrics = self._get_model_metrics(
                model_name,
                production_model['version']
            )

            production_value = production_metrics.get(criteria.metric_name)

            if production_value:
                # Calculate improvement
                improvement = (metric_value - production_value) / production_value

                if improvement < criteria.improvement_threshold:
                    return {
                        "should_promote": False,
                        "reason": f"Improvement {improvement:.2%} < {criteria.improvement_threshold:.2%}",
                        "comparison": {
                            "challenger": metric_value,
                            "champion": production_value,
                            "improvement": improvement
                        }
                    }

        except ValueError:
            # No production model yet
            pass

        # TODO: Check latency
        if criteria.max_latency_ms:
            latency = metrics.get('latency_ms')
            if latency and latency > criteria.max_latency_ms:
                return {
                    "should_promote": False,
                    "reason": f"Latency {latency}ms > {criteria.max_latency_ms}ms"
                }

        # TODO: Determine target stage
        target_stage = ModelStage.PRODUCTION
        if criteria.require_approval:
            target_stage = ModelStage.STAGING

        return {
            "should_promote": True,
            "target_stage": target_stage.value,
            "reason": f"{criteria.metric_name} meets criteria",
            "metrics": metrics
        }

    def promote_if_better(
        self,
        model_name: str,
        challenger_version: str,
        metrics: Dict[str, float],
        dry_run: bool = False
    ) -> Dict:
        """
        Promote challenger if better than champion

        Args:
            dry_run: Only evaluate, don't actually promote

        Returns:
            Promotion result
        """
        # TODO: Evaluate promotion
        evaluation = self.evaluate_promotion(model_name, challenger_version, metrics)

        if not evaluation['should_promote']:
            logger.info(f"Not promoting {model_name} v{challenger_version}: {evaluation['reason']}")
            return evaluation

        # TODO: Promote if not dry run
        if not dry_run:
            target_stage = ModelStage[evaluation['target_stage'].upper()]
            self.registry.transition_stage(
                model_name,
                challenger_version,
                target_stage,
                archive_existing_versions=True
            )

            # Tag with promotion info
            self.registry.set_model_version_tag(
                model_name,
                challenger_version,
                "promoted_at",
                datetime.utcnow().isoformat()
            )

            logger.info(f"✅ Promoted {model_name} v{challenger_version} to {target_stage.value}")

        evaluation['promoted'] = not dry_run
        return evaluation

    def _get_model_metrics(self, model_name: str, version: str) -> Dict[str, float]:
        """Get metrics for model version from run"""
        model_version = self.registry.get_model_version(model_name, version=version)
        run_id = model_version['run_id']

        # Get run metrics
        run = self.registry.client.get_run(run_id)
        return run.data.metrics
```

**Acceptance Criteria**:
- ✅ Define promotion criteria
- ✅ Compare challenger vs champion
- ✅ Auto-promote if better
- ✅ Require approval for production
- ✅ Archive old versions

---

### Task 5: Model Serving with A/B Testing (5-6 hours)

Build model serving API with A/B testing.

```python
# src/mlflow_platform/serving/model_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import time
from ..registry.model_registry import ModelRegistry, ModelStage
from ..monitoring.performance_tracker import PerformanceTracker

app = FastAPI(title="MLflow Model Server")

# Initialize components
registry = ModelRegistry()
performance_tracker = PerformanceTracker()

# Model cache
loaded_models = {}

class PredictionRequest(BaseModel):
    """Request for model prediction"""
    features: List[float]
    model_name: str
    version: Optional[str] = None  # If None, uses production version
    ab_test_variant: Optional[str] = None  # For A/B testing

class PredictionResponse(BaseModel):
    """Response from model"""
    prediction: float
    model_name: str
    model_version: str
    latency_ms: float
    variant: Optional[str] = None

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make prediction

    If version not specified, uses production version
    """
    start_time = time.time()

    try:
        # TODO: Load model
        if request.version:
            model_version = request.version
        else:
            # Get production version
            model_info = registry.get_model_version(
                request.model_name,
                stage=ModelStage.PRODUCTION
            )
            model_version = model_info['version']

        # TODO: Get model from cache or load
        model_key = f"{request.model_name}:{model_version}"
        if model_key not in loaded_models:
            model_uri = f"models:/{request.model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            loaded_models[model_key] = model
        else:
            model = loaded_models[model_key]

        # TODO: Make prediction
        features_array = np.array([request.features])
        prediction = model.predict(features_array)[0]

        # TODO: Track metrics
        latency_ms = (time.time() - start_time) * 1000
        performance_tracker.track_prediction(
            model_name=request.model_name,
            model_version=model_version,
            latency_ms=latency_ms,
            prediction=prediction,
            features=request.features
        )

        return PredictionResponse(
            prediction=float(prediction),
            model_name=request.model_name,
            model_version=model_version,
            latency_ms=latency_ms,
            variant=request.ab_test_variant
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ab_test")
def predict_ab_test(request: PredictionRequest, traffic_split: Dict[str, float]):
    """
    A/B test between model versions

    Args:
        traffic_split: {"v1": 0.9, "v2": 0.1}
    """
    import random

    # TODO: Select version based on traffic split
    rand = random.random()
    cumulative = 0
    selected_version = None

    for version, weight in traffic_split.items():
        cumulative += weight
        if rand < cumulative:
            selected_version = version
            break

    # TODO: Make prediction with selected version
    request.version = selected_version
    request.ab_test_variant = selected_version
    return predict(request)

@app.get("/models/{model_name}/versions")
def list_model_versions(model_name: str):
    """List all versions of model"""
    return registry.list_model_versions(model_name)

@app.post("/models/{model_name}/promote/{version}")
def promote_model(model_name: str, version: str, stage: str):
    """Promote model to stage"""
    stage_enum = ModelStage[stage.upper()]
    registry.transition_stage(model_name, version, stage_enum)
    return {"status": "success"}

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "loaded_models": len(loaded_models)}
```

**Acceptance Criteria**:
- ✅ Serve models by version/stage
- ✅ A/B testing with traffic splits
- ✅ Track prediction metrics
- ✅ Model caching for performance
- ✅ Health check endpoint

---

### Task 6: Model Performance Monitoring (4-5 hours)

Monitor model performance in production.

```python
# src/mlflow_platform/monitoring/performance_tracker.py

from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class PredictionLog:
    """Log entry for prediction"""
    timestamp: datetime
    model_name: str
    model_version: str
    features: List[float]
    prediction: float
    latency_ms: float
    actual: float = None  # Ground truth (if available)

class PerformanceTracker:
    """
    Track model performance metrics

    Metrics:
    - Prediction latency (p50, p95, p99)
    - Prediction distribution (detect drift)
    - Accuracy (if ground truth available)
    - Throughput (predictions per second)
    """

    def __init__(self):
        self.predictions: List[PredictionLog] = []

    def track_prediction(
        self,
        model_name: str,
        model_version: str,
        latency_ms: float,
        prediction: float,
        features: List[float],
        actual: float = None
    ):
        """Log prediction"""
        log = PredictionLog(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            model_version=model_version,
            features=features,
            prediction=prediction,
            latency_ms=latency_ms,
            actual=actual
        )
        self.predictions.append(log)

        # TODO: Write to persistent storage (database/file)
        # For production, use time-series database like InfluxDB

    def get_metrics(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        window: timedelta = timedelta(hours=1)
    ) -> Dict:
        """
        Get metrics for time window

        Returns:
            {
                "latency_p50": 45.2,
                "latency_p95": 98.5,
                "latency_p99": 145.3,
                "throughput": 100.5,  # predictions/sec
                "prediction_mean": 0.25,
                "prediction_std": 0.15,
                "accuracy": 0.87  # if ground truth available
            }
        """
        # TODO: Filter predictions
        cutoff = datetime.utcnow() - window
        filtered = [
            p for p in self.predictions
            if p.timestamp >= cutoff
            and p.model_name == model_name
            and (model_version is None or p.model_version == model_version)
        ]

        if not filtered:
            return {}

        # TODO: Calculate latency percentiles
        latencies = [p.latency_ms for p in filtered]
        metrics = {
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
        }

        # TODO: Calculate throughput
        duration_seconds = window.total_seconds()
        metrics["throughput"] = len(filtered) / duration_seconds

        # TODO: Prediction distribution
        predictions = [p.prediction for p in filtered]
        metrics["prediction_mean"] = np.mean(predictions)
        metrics["prediction_std"] = np.std(predictions)

        # TODO: Accuracy (if ground truth available)
        with_actual = [p for p in filtered if p.actual is not None]
        if with_actual:
            errors = [abs(p.prediction - p.actual) for p in with_actual]
            metrics["mae"] = np.mean(errors)

        return metrics

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        window: timedelta = timedelta(hours=1)
    ) -> Dict:
        """
        Compare metrics between two versions (for A/B testing)

        Returns:
            {
                "version_a": {...metrics...},
                "version_b": {...metrics...},
                "winner": "version_a",
                "confidence": 0.95
            }
        """
        metrics_a = self.get_metrics(model_name, version_a, window)
        metrics_b = self.get_metrics(model_name, version_b, window)

        # TODO: Statistical test to determine winner
        # For now, simple comparison
        winner = None
        if metrics_a.get('latency_p95', float('inf')) < metrics_b.get('latency_p95', float('inf')):
            winner = version_a
        else:
            winner = version_b

        return {
            "version_a": metrics_a,
            "version_b": metrics_b,
            "winner": winner
        }
```

**Acceptance Criteria**:
- ✅ Track predictions with metrics
- ✅ Calculate latency percentiles
- ✅ Monitor prediction distribution
- ✅ Compare A/B test versions
- ✅ Detect performance degradation

---

### Task 7: CLI Tool (3-4 hours)

Build CLI for common operations.

```python
# src/mlflow_platform/cli/mlflow_cli.py

import click
from ..registry.model_registry import ModelRegistry, ModelStage
from ..registry.promotion_policy import ModelPromoter, PromotionCriteria

registry = ModelRegistry()

@click.group()
def cli():
    """MLflow Platform CLI"""
    pass

@cli.command()
@click.argument('model_name')
def list_versions(model_name: str):
    """List all versions of model"""
    versions = registry.list_model_versions(model_name)
    for v in versions:
        click.echo(f"Version {v['version']}: {v['stage']} (created {v['created_at']})")

@cli.command()
@click.argument('model_name')
@click.argument('version')
@click.argument('stage', type=click.Choice(['Staging', 'Production', 'Archived']))
def promote(model_name: str, version: str, stage: str):
    """Promote model to stage"""
    stage_enum = ModelStage[stage.upper()]
    registry.transition_stage(model_name, version, stage_enum)
    click.echo(f"✅ Promoted {model_name} v{version} to {stage}")

@cli.command()
@click.argument('model_name')
@click.option('--stage', default='Production')
def get_model(model_name: str, stage: str):
    """Get model info"""
    stage_enum = ModelStage[stage.upper()]
    model = registry.get_model_version(model_name, stage=stage_enum)
    click.echo(json.dumps(model, indent=2))

@cli.command()
@click.argument('model_name')
@click.argument('version')
@click.option('--metric', required=True)
@click.option('--value', type=float, required=True)
def auto_promote(model_name: str, version: str, metric: str, value: float):
    """Auto-promote model if meets criteria"""
    criteria = PromotionCriteria(
        metric_name=metric,
        min_value=value,
        improvement_threshold=0.05
    )

    promoter = ModelPromoter(registry, {model_name: criteria})
    result = promoter.promote_if_better(
        model_name,
        version,
        {metric: value}
    )

    if result['should_promote']:
        click.echo(f"✅ {result['reason']}")
    else:
        click.echo(f"❌ {result['reason']}")

if __name__ == '__main__':
    cli()
```

**Acceptance Criteria**:
- ✅ List model versions
- ✅ Promote models
- ✅ Get model info
- ✅ Auto-promote based on criteria
- ✅ User-friendly output

---

## Testing Requirements

```python
def test_experiment_tracking():
    """Test experiment tracking"""
    tracker = ExperimentTracker(experiment_name="test")

    run_id = tracker.start_run("test_run")
    tracker.log_params(run_id, {"lr": 0.01})
    tracker.log_metrics(run_id, {"accuracy": 0.95})
    tracker.end_run(run_id)

    # Verify run exists
    run = tracker.client.get_run(run_id)
    assert run.data.params['lr'] == '0.01'
    assert run.data.metrics['accuracy'] == 0.95

def test_model_promotion():
    """Test automated promotion"""
    # Register model
    # Promote if meets criteria
    # Verify stage transition
    pass
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Model Load Time** | <2s | ________s |
| **Inference Latency (P95)** | <100ms | ________ms |
| **Registry Availability** | 99.9% | ________% |

## Validation

Submit:
1. MLflow server setup
2. Experiment tracking implementation
3. Model registry with promotion
4. A/B testing framework
5. Monitoring system
6. CLI tool
7. Documentation

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

**Estimated Completion Time**: 30-38 hours

**Skills Practiced**:
- MLflow experiment tracking
- Model registry and versioning
- Automated model promotion
- A/B testing
- Model monitoring
- MLOps best practices
