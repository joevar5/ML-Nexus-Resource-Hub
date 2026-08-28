# Lesson 05: CI/CD for ML Models

## Overview
Continuous Integration and Continuous Deployment (CI/CD) for machine learning extends traditional software engineering practices to handle the unique challenges of ML systems. This lesson covers building automated pipelines for testing, validating, and deploying ML models.

**Duration:** 3-4 hours
**Prerequisites:** Understanding of CI/CD concepts, Git, Docker, ML model training
**Learning Objectives:**
- Understand CI/CD principles for ML systems
- Implement automated model testing and validation
- Build deployment pipelines for ML models
- Handle model versioning and rollbacks
- Integrate monitoring and alerting

---

## 1. CI/CD for ML vs Traditional Software

### 1.1 Key Differences

| Aspect | Traditional CI/CD | ML CI/CD |
|--------|------------------|----------|
| **Artifacts** | Code only | Code + Data + Models + Configs |
| **Testing** | Unit/Integration tests | + Data validation + Model performance |
| **Deployment** | Binary/Container | + Model serving + Feature pipelines |
| **Versioning** | Code versions | + Data versions + Model versions |
| **Rollback** | Previous code version | + Previous model + Feature schema |
| **Monitoring** | Uptime, latency | + Model accuracy + Data drift |

### 1.2 ML CI/CD Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML CI/CD Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐ │
│  │   Code   │    │   Data   │    │  Model   │    │  Feature  │ │
│  │  Change  │───▶│Validation│───▶│ Training │───▶│Engineering│ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────┘ │
│       │               │                │               │         │
│       ▼               ▼                ▼               ▼         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐ │
│  │   Code   │    │   Data   │    │  Model   │    │  Feature  │ │
│  │   Tests  │    │  Quality │    │Validation│    │   Tests   │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────┘ │
│       │               │                │               │         │
│       └───────────────┴────────────────┴───────────────┘         │
│                             │                                     │
│                             ▼                                     │
│                  ┌──────────────────┐                            │
│                  │ Integration Tests │                            │
│                  └──────────────────┘                            │
│                             │                                     │
│                             ▼                                     │
│       ┌─────────────────────────────────────┐                   │
│       │      Deployment Decision Gate        │                   │
│       │  • Performance thresholds met?       │                   │
│       │  • Data quality acceptable?          │                   │
│       │  • Model bias within limits?         │                   │
│       └─────────────────────────────────────┘                   │
│                             │                                     │
│                   ┌─────────┴─────────┐                         │
│                   ▼                   ▼                         │
│          ┌─────────────┐      ┌─────────────┐                  │
│          │   Staging   │      │   Canary    │                  │
│          │ Deployment  │      │ Deployment  │                  │
│          └─────────────┘      └─────────────┘                  │
│                   │                   │                         │
│                   └─────────┬─────────┘                         │
│                             ▼                                     │
│                  ┌──────────────────┐                            │
│                  │    Production     │                            │
│                  │    Deployment     │                            │
│                  └──────────────────┘                            │
│                             │                                     │
│                             ▼                                     │
│                  ┌──────────────────┐                            │
│                  │    Monitoring     │                            │
│                  │    & Alerting     │                            │
│                  └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Setting Up ML CI/CD Pipeline

### 2.1 GitHub Actions for ML

```yaml
# .github/workflows/ml-ci-cd.yml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_REGION: us-west-2

jobs:
  # Job 1: Code Quality and Unit Tests
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov black flake8 mypy

      - name: Code formatting check
        run: black --check src/

      - name: Linting
        run: flake8 src/ --max-line-length=100

      - name: Type checking
        run: mypy src/

      - name: Run unit tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # Job 2: Data Validation
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install great-expectations pandas-profiling

      - name: Validate data schema
        run: |
          python scripts/validate_data_schema.py

      - name: Check data quality
        run: |
          python scripts/check_data_quality.py

      - name: Generate data profile
        run: |
          python scripts/generate_data_profile.py

      - name: Upload data quality report
        uses: actions/upload-artifact@v3
        with:
          name: data-quality-report
          path: reports/data_profile.html

  # Job 3: Model Training and Validation
  model-training:
    runs-on: ubuntu-latest
    needs: [code-quality, data-validation]
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: |
          python src/train.py \
            --experiment-name "ci-cd-pipeline" \
            --run-name "build-${{ github.run_number }}"

      - name: Validate model performance
        run: |
          python scripts/validate_model_performance.py \
            --min-accuracy 0.85 \
            --max-inference-time-ms 100

      - name: Test model fairness
        run: |
          python scripts/test_model_fairness.py

      - name: Export model metrics
        run: |
          python scripts/export_model_metrics.py

      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: models/

  # Job 4: Integration Tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: model-training
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v3

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: model-artifacts
          path: models/

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Test API endpoints
        run: |
          python scripts/test_api_endpoints.py

  # Job 5: Build and Push Docker Image
  build-image:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: model-artifacts
          path: models/

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1

      - name: Build and push image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/ml-model:$IMAGE_TAG .
          docker push $ECR_REGISTRY/ml-model:$IMAGE_TAG
          docker tag $ECR_REGISTRY/ml-model:$IMAGE_TAG $ECR_REGISTRY/ml-model:latest
          docker push $ECR_REGISTRY/ml-model:latest

  # Job 6: Deploy to Staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-image
    environment: staging
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ml-staging \
            --service ml-model-service \
            --force-new-deployment

      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster ml-staging \
            --services ml-model-service

      - name: Run smoke tests
        run: |
          python scripts/smoke_tests.py \
            --endpoint https://staging.ml-api.example.com

  # Job 7: Deploy to Production (Manual Approval)
  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Canary deployment logic
          python scripts/canary_deployment.py \
            --cluster ml-production \
            --service ml-model-service \
            --traffic-percentage 10

      - name: Monitor canary metrics
        run: |
          python scripts/monitor_canary.py \
            --duration-minutes 30 \
            --error-threshold 0.01

      - name: Promote canary to full deployment
        run: |
          python scripts/promote_canary.py \
            --cluster ml-production \
            --service ml-model-service
```

### 2.2 GitLab CI for ML

```yaml
# .gitlab-ci.yml
stages:
  - test
  - validate
  - train
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.11"
  DOCKER_DRIVER: overlay2

# Code quality and testing
code-quality:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - pip install pytest black flake8
    - black --check src/
    - flake8 src/
    - pytest tests/unit/
  artifacts:
    reports:
      junit: test-results.xml

# Data validation
data-validation:
  stage: validate
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - python scripts/validate_data.py
  artifacts:
    paths:
      - reports/data_validation/

# Model training
train-model:
  stage: train
  image: python:${PYTHON_VERSION}
  only:
    - main
    - develop
  script:
    - pip install -r requirements.txt
    - python src/train.py --experiment-name "gitlab-ci"
    - python scripts/validate_model.py --min-accuracy 0.85
  artifacts:
    paths:
      - models/
      - metrics/

# Build Docker image
build-image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  only:
    - main
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

# Deploy to staging
deploy-staging:
  stage: deploy
  image: alpine:latest
  environment:
    name: staging
  only:
    - main
  script:
    - apk add --no-cache curl
    - curl -X POST $STAGING_DEPLOY_WEBHOOK

# Deploy to production
deploy-production:
  stage: deploy
  image: alpine:latest
  environment:
    name: production
  only:
    - main
  when: manual
  script:
    - apk add --no-cache curl
    - curl -X POST $PRODUCTION_DEPLOY_WEBHOOK
```

---

## 3. Automated Testing for ML Models

### 3.1 Data Validation Tests

```python
# scripts/validate_data_schema.py
import pandas as pd
import great_expectations as ge
from pathlib import Path

def validate_data_schema(data_path: str) -> bool:
    """
    Validate data schema and constraints
    """
    df = pd.read_parquet(data_path)
    ge_df = ge.from_pandas(df)

    # Schema validation
    expectations = [
        # Column existence
        ge_df.expect_column_to_exist("user_id"),
        ge_df.expect_column_to_exist("features"),
        ge_df.expect_column_to_exist("label"),

        # Data types
        ge_df.expect_column_values_to_be_of_type("user_id", "string"),
        ge_df.expect_column_values_to_be_of_type("label", "int"),

        # Value constraints
        ge_df.expect_column_values_to_be_between("label", min_value=0, max_value=1),
        ge_df.expect_column_values_to_not_be_null("user_id"),

        # Uniqueness
        ge_df.expect_column_values_to_be_unique("user_id"),

        # Statistical checks
        ge_df.expect_column_mean_to_be_between("age", min_value=18, max_value=100),
        ge_df.expect_column_stdev_to_be_between("purchase_amount", min_value=0, max_value=1000),
    ]

    # Check all expectations
    all_passed = all([exp.success for exp in expectations])

    if not all_passed:
        failed = [exp for exp in expectations if not exp.success]
        print("❌ Data validation failed:")
        for exp in failed:
            print(f"  - {exp.expectation_config.expectation_type}")
        return False

    print("✅ All data validation checks passed")
    return True

# scripts/check_data_quality.py
import pandas as pd
import numpy as np

def check_data_quality(data_path: str, thresholds: dict) -> bool:
    """
    Check data quality metrics
    """
    df = pd.read_parquet(data_path)

    quality_checks = {
        "missing_values": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
        "duplicate_rows": df.duplicated().sum() / len(df) * 100,
        "label_imbalance": abs(df['label'].value_counts(normalize=True).values[0] - 0.5) * 100,
        "outlier_percentage": detect_outliers(df).sum() / len(df) * 100,
    }

    failures = []
    for check, value in quality_checks.items():
        threshold = thresholds.get(check, float('inf'))
        if value > threshold:
            failures.append(f"{check}: {value:.2f}% (threshold: {threshold}%)")

    if failures:
        print("❌ Data quality checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        return False

    print("✅ All data quality checks passed")
    return True

def detect_outliers(df: pd.DataFrame) -> pd.Series:
    """Detect outliers using IQR method"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.Series(False, index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers |= (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))

    return outliers
```

### 3.2 Model Performance Validation

```python
# scripts/validate_model_performance.py
import mlflow
import argparse
from typing import Dict
import sys

def validate_model_performance(
    model_uri: str,
    min_accuracy: float = 0.85,
    max_inference_time_ms: float = 100,
    min_recall: float = 0.80,
    max_false_positive_rate: float = 0.05
) -> bool:
    """
    Validate model meets performance thresholds
    """
    # Load model from MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    # Get logged metrics
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(model_uri.split("/")[-1])
    metrics = run.data.metrics

    # Performance checks
    checks = {
        "Accuracy": (metrics.get("accuracy", 0), min_accuracy, ">="),
        "Recall": (metrics.get("recall", 0), min_recall, ">="),
        "False Positive Rate": (metrics.get("fpr", 1), max_false_positive_rate, "<="),
        "Inference Time (ms)": (metrics.get("inference_time_ms", float('inf')), max_inference_time_ms, "<="),
    }

    failures = []
    for check_name, (actual, threshold, operator) in checks.items():
        if operator == ">=" and actual < threshold:
            failures.append(f"{check_name}: {actual:.4f} < {threshold}")
        elif operator == "<=" and actual > threshold:
            failures.append(f"{check_name}: {actual:.4f} > {threshold}")

    if failures:
        print("❌ Model performance validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return False

    print("✅ All model performance checks passed")
    for check_name, (actual, threshold, operator) in checks.items():
        print(f"  ✓ {check_name}: {actual:.4f} {operator} {threshold}")

    return True

# scripts/test_model_fairness.py
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from sklearn.metrics import accuracy_score

def test_model_fairness(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.DataFrame,
    max_disparity: float = 0.1
) -> bool:
    """
    Test model for fairness across sensitive features
    """
    # Calculate metrics by group
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Check demographic parity
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    print(f"\nFairness Metrics:")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"\nAccuracy by group:")
    print(metric_frame.by_group["accuracy"])

    if abs(dp_diff) > max_disparity:
        print(f"❌ Fairness check failed: DP difference {dp_diff:.4f} > {max_disparity}")
        return False

    print("✅ Fairness checks passed")
    return True
```

### 3.3 Integration Tests

```python
# tests/integration/test_model_serving.py
import pytest
import requests
import time
from typing import Dict

@pytest.fixture
def api_endpoint():
    return "http://localhost:8000"

def test_model_prediction_endpoint(api_endpoint):
    """Test model prediction API"""
    payload = {
        "user_id": "user_123",
        "features": {
            "age": 35,
            "purchase_count": 10,
            "avg_purchase_amount": 75.50
        }
    }

    response = requests.post(
        f"{api_endpoint}/predict",
        json=payload
    )

    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1

def test_model_latency(api_endpoint):
    """Test model inference latency"""
    payload = {"user_id": "user_123", "features": {...}}

    latencies = []
    for _ in range(100):
        start = time.time()
        response = requests.post(f"{api_endpoint}/predict", json=payload)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)

    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms > 100ms"

def test_model_throughput(api_endpoint):
    """Test model serving throughput"""
    import concurrent.futures

    payload = {"user_id": "user_123", "features": {...}}

    def make_request():
        return requests.post(f"{api_endpoint}/predict", json=payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        start = time.time()
        futures = [executor.submit(make_request) for _ in range(1000)]
        results = [f.result() for f in futures]
        duration = time.time() - start

    throughput = len(results) / duration
    assert throughput > 100, f"Throughput {throughput:.2f} req/s < 100 req/s"

def test_feature_store_integration(api_endpoint):
    """Test integration with feature store"""
    response = requests.post(
        f"{api_endpoint}/predict",
        json={"user_id": "user_123"}  # Only user_id, features from store
    )

    assert response.status_code == 200
    result = response.json()
    assert "features_from_store" in result
    assert result["features_from_store"] is True

def test_model_version_endpoint(api_endpoint):
    """Test model version information"""
    response = requests.get(f"{api_endpoint}/model/version")

    assert response.status_code == 200
    version_info = response.json()
    assert "model_version" in version_info
    assert "mlflow_run_id" in version_info
    assert "training_date" in version_info
```

---

## 4. Model Versioning and Registry

### 4.1 MLflow Model Registry Integration

```python
# src/model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional

class ModelRegistry:
    """Manage ML models in MLflow Registry"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[dict] = None
    ) -> str:
        """Register a new model version"""
        # Register model
        model_version = mlflow.register_model(model_uri, name)

        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=name,
                    version=model_version.version,
                    key=key,
                    value=value
                )

        print(f"✅ Registered {name} version {model_version.version}")
        return model_version.version

    def promote_model(
        self,
        name: str,
        version: str,
        stage: str  # "Staging", "Production", "Archived"
    ):
        """Promote model to a deployment stage"""
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=True  # Archive old versions
        )
        print(f"✅ Promoted {name} v{version} to {stage}")

    def get_latest_model(self, name: str, stage: str = "Production"):
        """Get latest model in a stage"""
        versions = self.client.get_latest_versions(name, stages=[stage])
        if not versions:
            raise ValueError(f"No {stage} model found for {name}")
        return versions[0]

    def compare_models(self, name: str, version1: str, version2: str):
        """Compare two model versions"""
        v1 = self.client.get_model_version(name, version1)
        v2 = self.client.get_model_version(name, version2)

        # Get metrics for both versions
        run1 = self.client.get_run(v1.run_id)
        run2 = self.client.get_run(v2.run_id)

        comparison = {
            "version_1": {
                "version": version1,
                "metrics": run1.data.metrics,
                "created": v1.creation_timestamp
            },
            "version_2": {
                "version": version2,
                "metrics": run2.data.metrics,
                "created": v2.creation_timestamp
            }
        }

        return comparison

# Usage in CI/CD pipeline
def cicd_model_registration():
    """Register and promote model in CI/CD"""
    registry = ModelRegistry(tracking_uri="https://mlflow.example.com")

    # Register new model version
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    version = registry.register_model(
        model_uri=model_uri,
        name="recommendation_model",
        tags={
            "git_commit": os.getenv("GITHUB_SHA"),
            "build_number": os.getenv("GITHUB_RUN_NUMBER"),
            "trained_by": "ci-cd-pipeline"
        }
    )

    # Promote to staging for testing
    registry.promote_model(
        name="recommendation_model",
        version=version,
        stage="Staging"
    )
```

### 4.2 Automated Model Deployment Decision

```python
# scripts/deployment_gate.py
import mlflow
from model_registry import ModelRegistry
from typing import Dict

def should_deploy_model(
    model_name: str,
    new_version: str,
    production_version: str,
    thresholds: Dict[str, float]
) -> bool:
    """
    Decide if new model should be deployed to production
    """
    registry = ModelRegistry(mlflow.get_tracking_uri())

    # Compare new model with production
    comparison = registry.compare_models(
        name=model_name,
        version1=production_version,
        version2=new_version
    )

    new_metrics = comparison["version_2"]["metrics"]
    prod_metrics = comparison["version_1"]["metrics"]

    # Decision criteria
    checks = [
        # Performance must improve or stay within tolerance
        ("accuracy", new_metrics["accuracy"] >= prod_metrics["accuracy"] - 0.01),

        # Recall must not decrease significantly
        ("recall", new_metrics["recall"] >= prod_metrics["recall"] - 0.02),

        # Latency must not increase
        ("inference_time_ms", new_metrics["inference_time_ms"] <= prod_metrics["inference_time_ms"] * 1.1),

        # Must meet absolute thresholds
        ("min_accuracy", new_metrics["accuracy"] >= thresholds["min_accuracy"]),
        ("max_latency", new_metrics["inference_time_ms"] <= thresholds["max_latency_ms"]),
    ]

    failures = [check for check, passed in checks if not passed]

    if failures:
        print("❌ Deployment gate failed:")
        for check in failures:
            print(f"  - {check}")
        return False

    print("✅ Deployment gate passed")
    print(f"  Accuracy: {prod_metrics['accuracy']:.4f} → {new_metrics['accuracy']:.4f}")
    print(f"  Latency: {prod_metrics['inference_time_ms']:.2f}ms → {new_metrics['inference_time_ms']:.2f}ms")
    return True

# Use in CI/CD
if __name__ == "__main__":
    import sys

    should_deploy = should_deploy_model(
        model_name="recommendation_model",
        new_version=sys.argv[1],
        production_version=sys.argv[2],
        thresholds={
            "min_accuracy": 0.85,
            "max_latency_ms": 100
        }
    )

    sys.exit(0 if should_deploy else 1)
```

---

## 5. Deployment Strategies

### 5.1 Canary Deployment

```python
# scripts/canary_deployment.py
import boto3
import time
from typing import Optional

class CanaryDeployment:
    """Manage canary deployments in AWS ECS"""

    def __init__(self, cluster: str, service: str):
        self.ecs = boto3.client('ecs')
        self.cloudwatch = boto3.client('cloudwatch')
        self.cluster = cluster
        self.service = service

    def deploy_canary(self, traffic_percentage: int = 10):
        """Deploy new version as canary with specified traffic"""

        # Update task definition with new image
        response = self.ecs.describe_services(
            cluster=self.cluster,
            services=[self.service]
        )
        current_task_def = response['services'][0]['taskDefinition']

        # Create new task definition with updated image
        # ... (update logic)

        # Update service with canary configuration
        self.ecs.update_service(
            cluster=self.cluster,
            service=self.service,
            taskDefinition=new_task_def,
            deploymentConfiguration={
                'deploymentCircuitBreaker': {
                    'enable': True,
                    'rollback': True
                },
                'maximumPercent': 200,
                'minimumHealthyPercent': 100
            }
        )

        # Set canary traffic weight using ALB
        self._configure_traffic_split(traffic_percentage)

        print(f"✅ Canary deployed with {traffic_percentage}% traffic")

    def monitor_canary(
        self,
        duration_minutes: int = 30,
        error_threshold: float = 0.01
    ) -> bool:
        """Monitor canary metrics and decide to promote or rollback"""

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            metrics = self._get_canary_metrics()

            # Check error rate
            if metrics['error_rate'] > error_threshold:
                print(f"❌ Canary failing: error rate {metrics['error_rate']:.4f}")
                self.rollback_canary()
                return False

            # Check latency
            if metrics['p99_latency'] > metrics['baseline_p99_latency'] * 1.5:
                print(f"❌ Canary failing: high latency {metrics['p99_latency']:.2f}ms")
                self.rollback_canary()
                return False

            print(f"✅ Canary healthy: {(time.time() - start_time) / 60:.1f}min elapsed")
            time.sleep(60)  # Check every minute

        print("✅ Canary monitoring completed successfully")
        return True

    def promote_canary(self):
        """Promote canary to 100% traffic"""
        self._configure_traffic_split(100)
        print("✅ Canary promoted to 100% traffic")

    def rollback_canary(self):
        """Rollback to previous version"""
        # Revert to previous task definition
        self.ecs.update_service(
            cluster=self.cluster,
            service=self.service,
            taskDefinition=self.previous_task_def
        )
        print("⚠️ Rolled back to previous version")

    def _configure_traffic_split(self, canary_percentage: int):
        """Configure ALB target group weights"""
        # Implementation depends on your load balancer setup
        pass

    def _get_canary_metrics(self) -> dict:
        """Get current canary metrics from CloudWatch"""
        # Query CloudWatch for metrics
        response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName='TargetResponseTime',
            Dimensions=[
                {'Name': 'ServiceName', 'Value': self.service}
            ],
            StartTime=time.time() - 300,  # Last 5 minutes
            EndTime=time.time(),
            Period=60,
            Statistics=['Average', 'Maximum']
        )

        return {
            'error_rate': 0.001,  # Calculate from actual metrics
            'p99_latency': 85,
            'baseline_p99_latency': 80
        }
```

### 5.2 Blue-Green Deployment

```python
# scripts/blue_green_deployment.py
import boto3
from typing import Tuple

class BlueGreenDeployment:
    """Manage blue-green deployments"""

    def __init__(self, cluster: str):
        self.ecs = boto3.client('ecs')
        self.elbv2 = boto3.client('elbv2')
        self.cluster = cluster

    def deploy_green(self, task_definition: str) -> str:
        """Deploy green environment"""

        # Create new service (green)
        green_service = f"{self.service_name}-green"

        self.ecs.create_service(
            cluster=self.cluster,
            serviceName=green_service,
            taskDefinition=task_definition,
            desiredCount=self.desired_count,
            loadBalancers=[{
                'targetGroupArn': self.green_target_group_arn,
                'containerName': 'ml-model',
                'containerPort': 8000
            }]
        )

        # Wait for green to be healthy
        self._wait_for_healthy(green_service)

        return green_service

    def switch_traffic(self, blue_service: str, green_service: str):
        """Switch traffic from blue to green"""

        # Update ALB listener rules
        self.elbv2.modify_listener(
            ListenerArn=self.listener_arn,
            DefaultActions=[{
                'Type': 'forward',
                'TargetGroupArn': self.green_target_group_arn
            }]
        )

        print("✅ Traffic switched to green environment")

        # Monitor for issues
        time.sleep(300)  # Monitor for 5 minutes

        # If successful, decommission blue
        self.ecs.update_service(
            cluster=self.cluster,
            service=blue_service,
            desiredCount=0
        )

        print("✅ Blue environment decommissioned")
```

---

## 6. Rollback Strategies

### 6.1 Automated Rollback

```python
# scripts/automated_rollback.py
import mlflow
from model_registry import ModelRegistry

class AutomatedRollback:
    """Handle automated model rollbacks"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.registry = ModelRegistry(mlflow.get_tracking_uri())
        self.cloudwatch = boto3.client('cloudwatch')

    def monitor_and_rollback(
        self,
        current_version: str,
        monitoring_duration_minutes: int = 60
    ):
        """Monitor deployment and rollback if needed"""

        start_time = time.time()
        end_time = start_time + (monitoring_duration_minutes * 60)

        while time.time() < end_time:
            # Check key metrics
            metrics = self._get_production_metrics()

            if self._should_rollback(metrics):
                print("❌ Metrics degraded, initiating rollback")
                self.rollback()
                return False

            time.sleep(60)  # Check every minute

        return True

    def rollback(self):
        """Rollback to previous production version"""

        # Get current and previous versions
        versions = self.registry.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )

        if len(versions) < 2:
            raise ValueError("No previous version to rollback to")

        current = versions[0]
        previous = versions[1]

        # Demote current to staging
        self.registry.promote_model(
            name=self.model_name,
            version=current.version,
            stage="Staging"
        )

        # Promote previous to production
        self.registry.promote_model(
            name=self.model_name,
            version=previous.version,
            stage="Production"
        )

        # Trigger deployment
        self._trigger_deployment(previous.version)

        print(f"✅ Rolled back from v{current.version} to v{previous.version}")

    def _should_rollback(self, metrics: dict) -> bool:
        """Decide if rollback is needed"""
        return (
            metrics['error_rate'] > 0.05 or
            metrics['p99_latency'] > 500 or
            metrics['throughput'] < 50
        )

    def _get_production_metrics(self) -> dict:
        """Get current production metrics"""
        # Query from CloudWatch/Prometheus
        return {
            'error_rate': 0.01,
            'p99_latency': 120,
            'throughput': 150
        }

    def _trigger_deployment(self, version: str):
        """Trigger deployment of specific version"""
        # Implementation depends on your deployment system
        pass
```

---

## 7. Continuous Monitoring

### 7.1 Post-Deployment Monitoring

```python
# scripts/post_deployment_monitoring.py
import boto3
import time
from typing import Dict, List

class PostDeploymentMonitor:
    """Monitor model after deployment"""

    def __init__(self, model_name: str, environment: str):
        self.model_name = model_name
        self.environment = environment
        self.cloudwatch = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')

    def monitor(self, duration_minutes: int = 60) -> List[str]:
        """Monitor deployment and return alerts"""

        alerts = []
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            checks = {
                'error_rate': self._check_error_rate(),
                'latency': self._check_latency(),
                'throughput': self._check_throughput(),
                'prediction_distribution': self._check_prediction_distribution(),
                'feature_drift': self._check_feature_drift()
            }

            for check_name, (passed, message) in checks.items():
                if not passed:
                    alert = f"⚠️ {check_name}: {message}"
                    alerts.append(alert)
                    self._send_alert(alert)

            time.sleep(300)  # Check every 5 minutes

        return alerts

    def _check_error_rate(self) -> Tuple[bool, str]:
        """Check if error rate is within threshold"""
        error_rate = self._get_metric('ErrorRate')
        threshold = 0.01
        passed = error_rate < threshold
        message = f"{error_rate:.4f} (threshold: {threshold})"
        return (passed, message)

    def _check_prediction_distribution(self) -> Tuple[bool, str]:
        """Check if prediction distribution changed unexpectedly"""
        current_dist = self._get_prediction_distribution()
        baseline_dist = self._get_baseline_distribution()

        # KL divergence
        kl_div = self._calculate_kl_divergence(current_dist, baseline_dist)
        threshold = 0.1

        passed = kl_div < threshold
        message = f"KL divergence: {kl_div:.4f} (threshold: {threshold})"
        return (passed, message)

    def _send_alert(self, message: str):
        """Send alert via SNS"""
        self.sns.publish(
            TopicArn=f"arn:aws:sns:us-west-2:123456789:ml-alerts",
            Subject=f"Model Deployment Alert: {self.model_name}",
            Message=message
        )
```

---

## Summary

In this lesson, you learned:

✅ **ML CI/CD Pipelines:**
- Unique aspects of ML CI/CD vs traditional
- Automated testing for models and data
- GitHub Actions and GitLab CI implementations

✅ **Model Management:**
- Model versioning with MLflow Registry
- Deployment gates and decision logic
- Comparison and validation

✅ **Deployment Strategies:**
- Canary deployments
- Blue-green deployments
- Automated rollbacks

✅ **Monitoring:**
- Post-deployment monitoring
- Automated alerting
- Performance tracking

---

## Additional Resources

- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [GitHub Actions for ML](https://github.com/machine-learning-apps/actions-ml-cicd)
- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)

---

## Next Lesson

**Lesson 06: Model Deployment Strategies** - Deep dive into various deployment patterns for ML models at scale.
