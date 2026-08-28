# Exercise 06: CI/CD for ML Pipelines

**Estimated Time**: 26-34 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, GitHub Actions, Docker, pytest, MLflow

## Overview

Build production-grade CI/CD pipelines for ML projects using GitHub Actions. Implement automated testing (unit tests, integration tests, model validation), model training in CI, automated deployment, and rollback mechanisms. This exercise teaches MLOps best practices for continuous integration and delivery of ML models.

In production ML platforms, CI/CD is critical for:
- **Automated Testing**: Catch bugs before deployment
- **Model Validation**: Ensure models meet quality standards
- **Reproducibility**: Build/deploy from any commit
- **Fast Iteration**: Deploy models in minutes, not days
- **Rollback Safety**: Quick rollback on failures

## Learning Objectives

By completing this exercise, you will:

1. **Build GitHub Actions workflows** for ML projects
2. **Implement automated testing** (unit, integration, model tests)
3. **Create model training pipeline** in CI
4. **Build Docker images** automatically
5. **Deploy models** to staging/production
6. **Implement blue-green deployment** for zero-downtime
7. **Add automated rollback** on deployment failures

## Business Context

**Real-World Scenario**: Your data science team deploys models manually:

- **Slow deployments**: Takes 2-3 hours to deploy (manual steps)
- **Broken deployments**: 30% of deployments fail due to missing dependencies
- **No testing**: Models deployed without validation (causes production incidents)
- **Can't rollback**: No automated rollback, takes hours to fix failures
- **No reproducibility**: Can't rebuild model from 2 weeks ago

Your task: Build CI/CD pipeline that:
- Runs all tests automatically on every commit
- Trains model in CI and validates performance
- Builds Docker image with exact dependencies
- Deploys to staging automatically, production with approval
- Implements blue-green deployment (zero downtime)
- Auto-rollbacks on failures within 5 minutes

## Project Structure

```
exercise-06-ci-cd-ml-pipelines/
├── README.md
├── requirements.txt
├── .github/
│   └── workflows/
│       ├── test.yml                 # Run tests on PR
│       ├── train.yml                # Train model in CI
│       ├── build.yml                # Build Docker image
│       ├── deploy-staging.yml       # Deploy to staging
│       └── deploy-production.yml    # Deploy to production
├── src/
│   └── ml_project/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── predictor.py
│       └── utils/
│           ├── __init__.py
│           └── validation.py
├── tests/
│   ├── unit/
│   │   ├── test_data_loader.py
│   │   ├── test_features.py
│   │   └── test_model.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── model/
│       ├── test_model_performance.py
│       └── test_model_bias.py
├── scripts/
│   ├── train.py                     # Training script
│   ├── validate_model.py            # Model validation
│   └── deploy.py                    # Deployment script
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
│       ├── deployment.yaml
│       └── service.yaml
├── config/
│   ├── model_config.yaml
│   └── deployment_config.yaml
└── docs/
    ├── CICD.md
    └── DEPLOYMENT.md
```

## Requirements

### Functional Requirements

1. **Automated Testing**:
   - Unit tests for all functions
   - Integration tests for pipeline
   - Model performance tests (accuracy >threshold)
   - Model bias tests (fairness metrics)
   - Data quality tests

2. **Model Training in CI**:
   - Train model on PR
   - Cache training data
   - Log metrics to MLflow
   - Fail if model performance drops

3. **Docker Image Build**:
   - Build on every merge to main
   - Tag with git commit SHA
   - Push to container registry
   - Multi-stage builds for size optimization

4. **Deployment**:
   - Deploy to staging automatically
   - Deploy to production with approval
   - Blue-green deployment pattern
   - Health checks before traffic switch

5. **Rollback**:
   - Auto-rollback on errors
   - Keep last 5 versions deployable
   - One-command manual rollback

### Non-Functional Requirements

- **Build Time**: <10 minutes for full pipeline
- **Test Coverage**: >80%
- **Deployment Time**: <5 minutes
- **Rollback Time**: <2 minutes

## Implementation Tasks

### Task 1: Test Workflow (5-6 hours)

Set up automated testing on pull requests.

```yaml
# .github/workflows/test.yml

name: Test ML Pipeline

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install flake8 black mypy
          pip install -r requirements.txt

      - name: Lint with flake8
        run: flake8 src/ tests/ --max-line-length=120

      - name: Check formatting with black
        run: black --check src/ tests/

      - name: Type check with mypy
        run: mypy src/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test

  model-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download test data
        run: |
          # Download sample test dataset
          mkdir -p data/test
          # TODO: Download from S3 or artifact storage

      - name: Run model performance tests
        run: pytest tests/model/test_model_performance.py -v

      - name: Run model bias tests
        run: pytest tests/model/test_model_bias.py -v
```

**Example model test**:

```python
# tests/model/test_model_performance.py

import pytest
from src.ml_project.models.trainer import train_model
from src.ml_project.models.predictor import ModelPredictor

def test_model_accuracy_threshold():
    """Test model meets minimum accuracy threshold"""
    # Load test data
    X_test, y_test = load_test_data()

    # Train model (or load pre-trained)
    model = train_model(X_train, y_train)

    # Predict
    predictor = ModelPredictor(model)
    y_pred = predictor.predict(X_test)

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()

    # Assert minimum threshold
    assert accuracy >= 0.85, f"Model accuracy {accuracy} below threshold 0.85"

def test_model_precision_recall():
    """Test precision and recall meet thresholds"""
    from sklearn.metrics import precision_score, recall_score

    # Load data and model
    X_test, y_test = load_test_data()
    model = load_trained_model()

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Assert thresholds
    assert precision >= 0.80, f"Precision {precision} below 0.80"
    assert recall >= 0.75, f"Recall {recall} below 0.75"
```

**Acceptance Criteria**:
- ✅ Lint, format, type checks pass
- ✅ Unit tests run with coverage
- ✅ Integration tests with services
- ✅ Model performance validated
- ✅ Tests run on every PR

---

### Task 2: Model Training Workflow (6-7 hours)

Train model in CI and track with MLflow.

```yaml
# .github/workflows/train.yml

name: Train Model

on:
  workflow_dispatch:  # Manual trigger
    inputs:
      experiment_name:
        description: 'MLflow experiment name'
        required: true
        default: 'fraud_detection'
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download training data
        run: |
          # Download from S3
          aws s3 cp s3://ml-data/training/latest.parquet data/training.parquet
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Train model
        run: python scripts/train.py --experiment-name ${{ github.event.inputs.experiment_name }}
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}

      - name: Validate model
        run: python scripts/validate_model.py --min-accuracy 0.85

      - name: Register model
        if: success()
        run: |
          python -c "
          import mlflow
          mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')

          # Get best run from experiment
          experiment = mlflow.get_experiment_by_name('${{ github.event.inputs.experiment_name }}')
          runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=['metrics.accuracy DESC'], max_results=1)

          if len(runs) > 0:
              run_id = runs.iloc[0]['run_id']
              # Register model
              mlflow.register_model(
                  f'runs:/{run_id}/model',
                  name='fraud_detection',
                  tags={'git_sha': '${{ github.sha }}', 'trained_by': 'github_actions'}
              )
          "

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: |
            models/
            metrics/
```

**Training script**:

```python
# scripts/train.py

import argparse
import mlflow
import mlflow.sklearn
from src.ml_project.data.loader import load_data
from src.ml_project.features.engineering import engineer_features
from src.ml_project.models.trainer import train_model
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Log git commit
        mlflow.set_tag('git_sha', os.environ.get('GITHUB_SHA', 'local'))

        # Load data
        df = load_data('data/training.parquet')
        mlflow.log_param('data_rows', len(df))

        # Engineer features
        X, y = engineer_features(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, 'model')

        print(f"Accuracy: {metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()
```

**Acceptance Criteria**:
- ✅ Train model in CI
- ✅ Log to MLflow
- ✅ Validate performance
- ✅ Register model
- ✅ Upload artifacts

---

### Task 3: Docker Build Workflow (5-6 hours)

Build and push Docker images automatically.

```yaml
# .github/workflows/build.yml

name: Build Docker Image

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: deployment/Dockerfile
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run container security scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

**Dockerfile** (multi-stage build):

```dockerfile
# deployment/Dockerfile

# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Download model from MLflow (or bake into image)
# In production, typically download at runtime
ENV MLFLOW_TRACKING_URI=https://mlflow.example.com

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run model server
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
```

**Acceptance Criteria**:
- ✅ Multi-stage Docker build
- ✅ Push to registry with tags
- ✅ Security scan with Trivy
- ✅ Build cache optimization
- ✅ Metadata labels

---

### Task 4: Deployment Workflows (6-7 hours)

Deploy to staging and production with approval.

```yaml
# .github/workflows/deploy-staging.yml

name: Deploy to Staging

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - uses: actions/checkout@v3

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG_STAGING }}" | base64 -d > kubeconfig
          export KUBECONFIG=./kubeconfig

      - name: Update deployment
        run: |
          # Update image tag in deployment
          kubectl set image deployment/model-server \
            model-server=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} \
            -n staging

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/model-server -n staging --timeout=5m

      - name: Run smoke tests
        run: |
          # Get service URL
          SERVICE_URL=$(kubectl get svc model-server -n staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

          # Test health endpoint
          curl -f http://$SERVICE_URL/health || exit 1

          # Test prediction endpoint
          curl -X POST http://$SERVICE_URL/predict \
            -H 'Content-Type: application/json' \
            -d '{"features": [1.0, 2.0, 3.0]}' || exit 1

      - name: Rollback on failure
        if: failure()
        run: |
          kubectl rollout undo deployment/model-server -n staging
```

```yaml
# .github/workflows/deploy-production.yml

name: Deploy to Production

on:
  workflow_dispatch:
    inputs:
      image_tag:
        description: 'Image tag to deploy'
        required: true

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production  # Requires approval

    steps:
      - uses: actions/checkout@v3

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG_PROD }}" | base64 -d > kubeconfig
          export KUBECONFIG=./kubeconfig

      - name: Blue-Green Deployment
        run: |
          # Deploy new version (green)
          kubectl apply -f deployment/kubernetes/deployment-green.yaml

          # Update green deployment image
          kubectl set image deployment/model-server-green \
            model-server=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.event.inputs.image_tag }} \
            -n production

          # Wait for green deployment
          kubectl rollout status deployment/model-server-green -n production --timeout=10m

      - name: Run production smoke tests
        run: |
          # Test green deployment
          GREEN_URL=$(kubectl get svc model-server-green -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

          # Health check
          curl -f http://$GREEN_URL/health || exit 1

          # Prediction test
          curl -X POST http://$GREEN_URL/predict \
            -H 'Content-Type: application/json' \
            -d '{"features": [1.0, 2.0, 3.0]}' || exit 1

      - name: Switch traffic to green
        if: success()
        run: |
          # Update service selector to point to green
          kubectl patch service model-server -n production \
            -p '{"spec":{"selector":{"version":"green"}}}'

          echo "Traffic switched to green deployment"

      - name: Keep blue deployment as rollback
        run: |
          # Scale down blue but keep for quick rollback
          kubectl scale deployment model-server-blue -n production --replicas=1

      - name: Rollback on failure
        if: failure()
        run: |
          # Delete green deployment
          kubectl delete deployment model-server-green -n production

          echo "Deployment failed, keeping blue active"
```

**Kubernetes deployment** (blue-green):

```yaml
# deployment/kubernetes/deployment-green.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-green
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
      version: green
  template:
    metadata:
      labels:
        app: model-server
        version: green
    spec:
      containers:
      - name: model-server
        image: ghcr.io/org/model:placeholder  # Updated by CI
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-credentials
              key: tracking-uri
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**Acceptance Criteria**:
- ✅ Deploy to staging automatically
- ✅ Deploy to production with approval
- ✅ Blue-green deployment pattern
- ✅ Smoke tests before traffic switch
- ✅ Auto-rollback on failures

---

### Task 5: Rollback Mechanism (4-5 hours)

Implement quick rollback capability.

```yaml
# .github/workflows/rollback.yml

name: Rollback Production

on:
  workflow_dispatch:
    inputs:
      target_version:
        description: 'Version to rollback to (or "previous")'
        required: true
        default: 'previous'

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v3

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG_PROD }}" | base64 -d > kubeconfig
          export KUBECONFIG=./kubeconfig

      - name: Rollback deployment
        run: |
          if [ "${{ github.event.inputs.target_version }}" == "previous" ]; then
            # Rollback to previous revision
            kubectl rollout undo deployment/model-server -n production
          else
            # Rollback to specific version
            # Switch traffic back to blue
            kubectl patch service model-server -n production \
              -p '{"spec":{"selector":{"version":"blue"}}}'

            # Scale up blue
            kubectl scale deployment model-server-blue -n production --replicas=3

            # Delete green
            kubectl delete deployment model-server-green -n production || true
          fi

      - name: Wait for rollback
        run: |
          kubectl rollout status deployment/model-server -n production --timeout=5m

      - name: Verify rollback
        run: |
          SERVICE_URL=$(kubectl get svc model-server -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          curl -f http://$SERVICE_URL/health

      - name: Notify team
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              text: "Production rollback completed to ${{ github.event.inputs.target_version }}",
              username: 'GitHub Actions',
              icon_emoji: ':warning:'
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

**Acceptance Criteria**:
- ✅ One-click rollback
- ✅ Rollback to previous or specific version
- ✅ Verification after rollback
- ✅ Team notification
- ✅ <2 minute rollback time

---

## Testing Requirements

```python
# tests/integration/test_deployment.py

def test_deployment_health_check():
    """Test deployment responds to health check"""
    import requests

    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_prediction_endpoint():
    """Test prediction endpoint works"""
    import requests

    payload = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    response = requests.post("http://localhost:8000/predict", json=payload)

    assert response.status_code == 200
    assert 'prediction' in response.json()
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Build Time** | <10min | ________min |
| **Test Coverage** | >80% | ________% |
| **Deployment Time** | <5min | ________min |
| **Rollback Time** | <2min | ________min |

## Validation

Submit:
1. GitHub Actions workflows
2. Automated tests (unit, integration, model)
3. Docker build configuration
4. Kubernetes deployment manifests
5. Blue-green deployment implementation
6. Rollback workflow
7. Documentation

## Resources

- [GitHub Actions](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Blue-Green Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html)

---

**Estimated Completion Time**: 26-34 hours

**Skills Practiced**:
- GitHub Actions workflows
- Automated ML testing
- Docker image building
- Kubernetes deployment
- Blue-green deployment
- CI/CD for ML
- Automated rollback
