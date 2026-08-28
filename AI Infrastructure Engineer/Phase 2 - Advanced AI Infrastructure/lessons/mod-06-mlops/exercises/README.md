# Module 06: MLOps & Experiment Tracking - Hands-On Exercises

## Overview
This directory contains 5 comprehensive hands-on labs that will give you practical experience with MLOps tools and practices. Each lab builds on the previous ones and takes approximately 3-4 hours to complete.

**Total Time:** 15-20 hours
**Prerequisites:** Python, basic ML knowledge, Docker basics

---

## Lab Setup

### Prerequisites Installation

```bash
# Create virtual environment
python -m venv mlops-labs
source mlops-labs/bin/activate  # On Windows: mlops-labs\Scripts\activate

# Install required packages
pip install mlflow==2.9.2
pip install feast==0.35.0
pip install great-expectations==0.18.8
pip install fastapi==0.109.0
pip install uvicorn==0.27.0
pip install scikit-learn==1.4.0
pip install pandas==2.2.0
pip install redis==5.0.1
pip install docker==7.0.0

# Install Docker (if not already installed)
# Follow instructions at: https://docs.docker.com/get-docker/
```

### Project Structure

```
exercises/
├── README.md (this file)
├── lab-01-experiment-tracking/
├── lab-02-feature-store/
├── lab-03-cicd-pipeline/
├── lab-04-model-deployment/
└── lab-05-production-monitoring/
```

---

## Lab 01: Experiment Tracking with MLflow

**Duration:** 3-4 hours
**Difficulty:** Beginner
**Topics:** MLflow, experiment tracking, model registry

### Learning Objectives
- Set up MLflow tracking server
- Log experiments, parameters, and metrics
- Compare multiple model runs
- Register and version models
- Load models from registry

### Tasks

#### Task 1.1: MLflow Setup (30 minutes)

```bash
# Start MLflow tracking server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

**TODO:**
- Access MLflow UI at http://localhost:5000
- Create a new experiment called "customer-churn-prediction"
- Document the components of MLflow (tracking, projects, models, registry)

#### Task 1.2: Training with Experiment Tracking (60 minutes)

Create `train.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# TODO 1: Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# TODO 2: Set experiment
mlflow.set_experiment("customer-churn-prediction")

def load_data():
    """Load and preprocess data"""
    # TODO 3: Load your dataset
    # For this exercise, generate synthetic data
    np.random.seed(42)
    n_samples = 10000

    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'num_support_tickets': np.random.randint(0, 10, n_samples),
        'churn': np.random.randint(0, 2, n_samples)
    })

    X = data.drop('churn', axis=1)
    y = data['churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(n_estimators, max_depth, min_samples_split):
    """Train model with MLflow tracking"""

    # TODO 4: Start MLflow run
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()

        # TODO 5: Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("model_type", "RandomForest")

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        # TODO 6: Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # TODO 7: Log model
        mlflow.sklearn.log_model(model, "model")

        # TODO 8: Log additional artifacts
        # Create and log feature importance plot
        import matplotlib.pyplot as plt
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')

        print(f"✅ Model trained with accuracy: {accuracy:.4f}")

        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    # TODO 9: Train multiple models with different hyperparameters
    configs = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 150, "max_depth": 15, "min_samples_split": 10},
    ]

    for config in configs:
        run_id = train_model(**config)
        print(f"Run ID: {run_id}")
```

**TODO:**
- Complete all TODO items in the code
- Run training with 3 different hyperparameter sets
- Compare runs in MLflow UI
- Identify the best performing model

#### Task 1.3: Model Registry (45 minutes)

Create `register_model.py`:

```python
import mlflow
from mlflow.tracking import MlflowClient

# TODO 1: Initialize MLflow client
client = MlflowClient("http://localhost:5000")

def register_best_model():
    """Find best model and register it"""

    # TODO 2: Get experiment
    experiment = client.get_experiment_by_name("customer-churn-prediction")

    # TODO 3: Search for best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    best_run = runs[0]

    print(f"Best run ID: {best_run.info.run_id}")
    print(f"F1 Score: {best_run.data.metrics['f1_score']:.4f}")

    # TODO 4: Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="churn-prediction-model"
    )

    print("✅ Model registered successfully")

def promote_to_production(model_name, version):
    """Promote model version to production"""

    # TODO 5: Transition model to production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✅ Model version {version} promoted to Production")

if __name__ == "__main__":
    register_best_model()
    # Promote version 1 to production
    promote_to_production("churn-prediction-model", "1")
```

**Deliverables:**
- Screenshot of MLflow UI showing all experiment runs
- Screenshot of registered model in Model Registry
- Document comparing the three model configurations
- Best model promoted to "Production" stage

---

## Lab 02: Feature Store Implementation

**Duration:** 4-5 hours
**Difficulty:** Intermediate
**Topics:** Feast, feature engineering, online/offline serving

### Learning Objectives
- Set up Feast feature store
- Define entities and feature views
- Implement feature engineering pipelines
- Serve features for training and inference
- Handle point-in-time correctness

### Tasks

#### Task 2.1: Feast Setup (45 minutes)

```bash
# Initialize Feast project
feast init customer_features
cd customer_features
```

**TODO:**
- Understand Feast project structure
- Configure `feature_store.yaml` for local development
- Set up Redis for online store (use Docker)

```bash
# Start Redis
docker run -d -p 6379:6379 redis:latest
```

#### Task 2.2: Define Features (90 minutes)

Edit `features.py`:

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# TODO 1: Define entities
customer = Entity(
    name="customer_id",
    value_type=String,
    description="Customer unique identifier"
)

# TODO 2: Create data source
customer_stats_source = FileSource(
    path="data/customer_statistics.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# TODO 3: Define feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="avg_support_tickets_30d", dtype=Float32),
        Field(name="payment_failures_30d", dtype=Int64),
    ],
    source=customer_stats_source,
    tags={"team": "data-science", "version": "v1"},
)

# TODO 4: Create on-demand feature view for derived features
from feast import on_demand_feature_view, RequestSource
import pandas as pd

request_source = RequestSource(
    name="request_data",
    schema=[
        Field(name="current_month_usage", dtype=Float32),
    ]
)

@on_demand_feature_view(
    sources=[customer_features, request_source],
    schema=[
        Field(name="usage_vs_avg", dtype=Float32),
        Field(name="high_risk_score", dtype=Float32),
    ]
)
def customer_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived features on-demand"""
    output = pd.DataFrame()

    # TODO 5: Implement derived feature logic
    output["usage_vs_avg"] = (
        inputs["current_month_usage"] / inputs["monthly_charges"]
    )

    output["high_risk_score"] = (
        (inputs["payment_failures_30d"] > 2).astype(float) * 0.3 +
        (inputs["avg_support_tickets_30d"] > 3).astype(float) * 0.3 +
        (output["usage_vs_avg"] < 0.5).astype(float) * 0.4
    )

    return output
```

Create `generate_data.py`:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_customers=1000):
    """Generate synthetic customer data"""

    # TODO: Generate synthetic data
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=90),
        end=datetime.now(),
        freq='D'
    )

    data = []
    for customer_id in range(n_customers):
        for date in dates:
            data.append({
                'customer_id': f"cust_{customer_id:04d}",
                'event_timestamp': date,
                'created_timestamp': datetime.now(),
                'age': np.random.randint(18, 80),
                'tenure_months': np.random.randint(1, 72),
                'monthly_charges': np.random.uniform(20, 120),
                'total_charges': np.random.uniform(100, 5000),
                'avg_support_tickets_30d': np.random.uniform(0, 5),
                'payment_failures_30d': np.random.randint(0, 5),
            })

    df = pd.DataFrame(data)
    df.to_parquet('data/customer_statistics.parquet')
    print(f"✅ Generated {len(df)} rows of customer data")

if __name__ == "__main__":
    generate_customer_data()
```

**TODO:**
- Generate synthetic data
- Apply feature definitions: `feast apply`
- Verify features registered: `feast feature-views list`

#### Task 2.3: Feature Serving (90 minutes)

Create `serve_features.py`:

```python
from feast import FeatureStore
import pandas as pd
from datetime import datetime

# TODO 1: Initialize feature store
fs = FeatureStore(repo_path=".")

def get_training_data():
    """Get historical features for training"""

    # TODO 2: Create entity dataframe with timestamps
    entity_df = pd.DataFrame({
        'customer_id': [f"cust_{i:04d}" for i in range(100)],
        'event_timestamp': [datetime(2024, 1, 15)] * 100
    })

    # TODO 3: Get historical features
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_features:age",
            "customer_features:tenure_months",
            "customer_features:monthly_charges",
            "customer_features:avg_support_tickets_30d",
            "customer_features:payment_failures_30d",
        ],
    ).to_df()

    print(f"✅ Retrieved {len(training_df)} rows for training")
    return training_df

def materialize_features():
    """Materialize features to online store"""

    # TODO 4: Materialize features
    from datetime import timedelta

    fs.materialize(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        feature_views=["customer_features"]
    )

    print("✅ Features materialized to online store")

def get_online_features(customer_ids):
    """Get features for real-time inference"""

    # TODO 5: Get online features
    online_features = fs.get_online_features(
        features=[
            "customer_features:age",
            "customer_features:monthly_charges",
            "customer_features:payment_failures_30d",
        ],
        entity_rows=[
            {"customer_id": cust_id} for cust_id in customer_ids
        ]
    ).to_dict()

    print(f"✅ Retrieved online features for {len(customer_ids)} customers")
    return online_features

if __name__ == "__main__":
    # Get training data
    training_df = get_training_data()
    print(training_df.head())

    # Materialize for online serving
    materialize_features()

    # Get online features
    online_features = get_online_features(["cust_0001", "cust_0002", "cust_0003"])
    print(online_features)
```

**Deliverables:**
- Feast feature definitions in `features.py`
- Generated synthetic data in Parquet format
- Screenshot of successful feature retrieval
- Document explaining offline vs online serving

---

## Lab 03: CI/CD Pipeline for ML Models

**Duration:** 3-4 hours
**Difficulty:** Intermediate
**Topics:** GitHub Actions, automated testing, deployment

### Learning Objectives
- Set up CI/CD for ML projects
- Implement automated testing (data, model, code)
- Create deployment pipelines
- Handle model versioning

### Tasks

#### Task 3.1: Project Setup (30 minutes)

```bash
# Create GitHub repository
git init
git remote add origin <your-repo-url>

# Create project structure
mkdir -p .github/workflows
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p src
```

**TODO:**
- Initialize Git repository
- Create `.gitignore` for Python ML projects
- Set up project structure

#### Task 3.2: Automated Tests (90 minutes)

Create `tests/unit/test_data_validation.py`:

```python
import pytest
import pandas as pd
import great_expectations as ge

def test_data_schema():
    """Test data has expected schema"""
    # TODO 1: Load test data
    df = pd.read_parquet("data/customer_statistics.parquet")
    ge_df = ge.from_pandas(df)

    # TODO 2: Schema validations
    assert ge_df.expect_column_to_exist("customer_id").success
    assert ge_df.expect_column_to_exist("age").success
    assert ge_df.expect_column_to_exist("monthly_charges").success

def test_data_quality():
    """Test data quality"""
    df = pd.read_parquet("data/customer_statistics.parquet")
    ge_df = ge.from_pandas(df)

    # TODO 3: Quality checks
    assert ge_df.expect_column_values_to_not_be_null("customer_id").success
    assert ge_df.expect_column_values_to_be_between("age", 18, 100).success
    assert ge_df.expect_column_values_to_be_between("monthly_charges", 0, 500).success

def test_no_data_leakage():
    """Ensure no future data in training set"""
    # TODO 4: Implement data leakage test
    df = pd.read_parquet("data/customer_statistics.parquet")
    # Check timestamps
    assert df['event_timestamp'].max() <= pd.Timestamp.now()
```

Create `tests/unit/test_model.py`:

```python
import pytest
from src.train import train_model, load_data
import mlflow

def test_model_trains_successfully():
    """Test model can be trained"""
    # TODO 1: Test model training
    X_train, X_test, y_train, y_test = load_data()
    assert len(X_train) > 0
    assert len(y_train) > 0

def test_model_performance_threshold():
    """Test model meets minimum performance"""
    # TODO 2: Train model and check performance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Model must be better than random guessing
    assert accuracy > 0.6

def test_model_prediction_shape():
    """Test model outputs correct shape"""
    # TODO 3: Test prediction shape
    pass
```

#### Task 3.3: CI/CD Pipeline (90 minutes)

Create `.github/workflows/ml-ci-cd.yml`:

```yaml
name: ML CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # TODO 1: Add code quality job
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install black flake8 pytest
          pip install -r requirements.txt

      - name: Code formatting
        run: black --check src/

      - name: Linting
        run: flake8 src/ --max-line-length=100

  # TODO 2: Add data validation job
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run data tests
        run: pytest tests/unit/test_data_validation.py

  # TODO 3: Add model training job
  model-training:
    runs-on: ubuntu-latest
    needs: [code-quality, data-validation]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python src/train.py

      - name: Run model tests
        run: pytest tests/unit/test_model.py

  # TODO 4: Add deployment job (only on main branch)
  deploy:
    runs-on: ubuntu-latest
    needs: model-training
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy model
        run: echo "Deploying model..."
        # Add actual deployment logic here
```

**Deliverables:**
- Complete test suite with >80% coverage
- Working CI/CD pipeline
- Screenshot of successful pipeline run
- Document describing each pipeline stage

---

## Lab 04: Model Deployment & Serving

**Duration:** 3-4 hours
**Difficulty:** Intermediate
**Topics:** FastAPI, Docker, model serving, monitoring

### Learning Objectives
- Build REST API for model serving
- Containerize ML application
- Implement health checks and monitoring
- Handle prediction requests

### Tasks

#### Task 4.1: FastAPI Model Server (120 minutes)

Create `src/serve.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from typing import Dict, List
import logging

# TODO 1: Initialize FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0.0")

# TODO 2: Define request/response models
class PredictionRequest(BaseModel):
    customer_id: str
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    num_support_tickets: int

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    model_version: str

# TODO 3: Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    """Load ML model on startup"""
    global model
    try:
        # Load from MLflow
        model_uri = "models:/churn-prediction-model/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"✅ Model loaded from {model_uri}")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        raise

# TODO 4: Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# TODO 5: Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make churn prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        features = pd.DataFrame([{
            'age': request.age,
            'tenure_months': request.tenure_months,
            'monthly_charges': request.monthly_charges,
            'total_charges': request.total_charges,
            'num_support_tickets': request.num_support_tickets
        }])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = prediction  # Assuming probability output

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(probability),
            churn_prediction=int(probability > 0.5),
            model_version="1.0"
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TODO 6: Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    # Implement batch prediction
    pass

# TODO 7: Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model metadata"""
    return {
        "model_name": "churn-prediction-model",
        "version": "1.0",
        "input_features": [
            "age", "tenure_months", "monthly_charges",
            "total_charges", "num_support_tickets"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Task 4.2: Dockerization (60 minutes)

Create `Dockerfile`:

```dockerfile
# TODO 1: Use appropriate base image
FROM python:3.11-slim

# TODO 2: Set working directory
WORKDIR /app

# TODO 3: Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TODO 4: Copy application code
COPY src/ ./src/
COPY models/ ./models/

# TODO 5: Expose port
EXPOSE 8000

# TODO 6: Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# TODO 7: Run application
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # TODO: Define model serving service
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

**TODO:**
- Build Docker image: `docker build -t churn-prediction-api .`
- Run container: `docker-compose up`
- Test API endpoints
- Verify health check works

**Deliverables:**
- Working FastAPI application
- Dockerfile and docker-compose.yml
- API documentation (auto-generated at /docs)
- Test results from Postman or curl

---

## Lab 05: Production Monitoring & A/B Testing

**Duration:** 4-5 hours
**Difficulty:** Advanced
**Topics:** Monitoring, A/B testing, drift detection

### Learning Objectives
- Implement model monitoring
- Set up A/B testing infrastructure
- Detect data and concept drift
- Create alerting system

### Tasks

#### Task 5.1: Monitoring Setup (120 minutes)

Create `src/monitoring.py`:

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import json
from datetime import datetime

class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.baseline_stats = self._calculate_stats(baseline_data)

    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution statistics"""
        # TODO 1: Calculate baseline statistics
        stats = {}
        for column in df.columns:
            stats[column] = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'percentiles': df[column].quantile([0.25, 0.50, 0.75]).to_dict()
            }
        return stats

    def detect_data_drift(
        self,
        new_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict:
        """Detect if data distribution has drifted"""
        # TODO 2: Implement drift detection using KS test
        drift_results = {}

        for column in new_data.columns:
            if column in self.baseline_data.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    self.baseline_data[column].dropna(),
                    new_data[column].dropna()
                )

                drift_results[column] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drifted': p_value < threshold
                }

        return drift_results

    def monitor_predictions(
        self,
        predictions: List[float],
        actuals: List[int] = None
    ) -> Dict:
        """Monitor prediction distribution and accuracy"""
        # TODO 3: Monitor predictions
        pred_stats = {
            'mean_prediction': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'prediction_min': np.min(predictions),
            'prediction_max': np.max(predictions),
        }

        if actuals:
            from sklearn.metrics import accuracy_score, roc_auc_score
            pred_labels = [1 if p > 0.5 else 0 for p in predictions]

            pred_stats.update({
                'accuracy': accuracy_score(actuals, pred_labels),
                'auc_roc': roc_auc_score(actuals, predictions)
            })

        return pred_stats

    def generate_alert(self, drift_results: Dict) -> List[str]:
        """Generate alerts for drifted features"""
        # TODO 4: Generate alerts
        alerts = []

        for feature, result in drift_results.items():
            if result['drifted']:
                alerts.append(
                    f"⚠️ Data drift detected in '{feature}': "
                    f"KS statistic={result['ks_statistic']:.4f}, "
                    f"p-value={result['p_value']:.4f}"
                )

        return alerts

# TODO 5: Integrate monitoring into prediction endpoint
```

#### Task 5.2: A/B Testing (120 minutes)

Create `src/ab_testing.py`:

```python
import hashlib
from typing import Dict
import redis
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    experiment_id: str
    control_model: str
    treatment_model: str
    traffic_split: float  # % for treatment

class ABTestRouter:
    """Route users to model variants"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.redis_client = redis.Redis(host='localhost', port=6379)

    def assign_variant(self, user_id: str) -> str:
        """Assign user to variant using consistent hashing"""
        # TODO 1: Implement consistent hashing

        # Check cache
        cache_key = f"ab_test:{self.config.experiment_id}:{user_id}"
        cached = self.redis_client.get(cache_key)

        if cached:
            return cached.decode('utf-8')

        # Hash user_id
        hash_value = int(hashlib.md5(
            f"{self.config.experiment_id}:{user_id}".encode()
        ).hexdigest(), 16)

        # Determine variant
        assignment_value = (hash_value % 10000) / 10000.0

        if assignment_value < self.config.traffic_split:
            variant = self.config.treatment_model
        else:
            variant = self.config.control_model

        # Cache assignment
        self.redis_client.setex(cache_key, 86400 * 30, variant)  # 30 days

        return variant

    def log_event(self, user_id: str, variant: str, prediction: float, outcome: int = None):
        """Log experiment event"""
        # TODO 2: Log to analytics system
        event = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.config.experiment_id,
            'user_id': user_id,
            'variant': variant,
            'prediction': prediction,
            'outcome': outcome
        }

        # Store in Redis (in practice, send to analytics pipeline)
        event_key = f"events:{self.config.experiment_id}:{datetime.now().date()}"
        self.redis_client.rpush(event_key, json.dumps(event))

# TODO 3: Integrate A/B testing into prediction endpoint
```

Create `src/experiment_analysis.py`:

```python
import pandas as pd
import numpy as np
from scipy import stats

class ExperimentAnalyzer:
    """Analyze A/B test results"""

    def proportion_test(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> Dict:
        """Two-proportion z-test"""
        # TODO: Implement statistical test

        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total

        # Pooled proportion
        p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))

        # Z-score
        z = (p2 - p1) / se

        # P-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Confidence interval
        se_diff = np.sqrt(p1*(1-p1)/control_total + p2*(1-p2)/treatment_total)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff

        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'relative_lift': (p2 - p1) / p1 if p1 > 0 else 0,
            'z_score': z,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': (ci_lower, ci_upper)
        }
```

**Deliverables:**
- Working monitoring system
- A/B testing infrastructure
- Dashboard showing monitoring metrics
- Document analyzing A/B test results

---

## Submission Guidelines

### For Each Lab

1. **Code**: Submit all code files with TODOs completed
2. **Documentation**: Include README explaining your implementation
3. **Screenshots**: Capture key results and UIs
4. **Analysis**: Write brief analysis of findings
5. **Learnings**: Document what you learned and challenges faced

### Submission Format

```
submission/
├── lab-01/
│   ├── code/
│   ├── screenshots/
│   ├── README.md
│   └── analysis.md
├── lab-02/
│   └── ...
├── lab-03/
│   └── ...
├── lab-04/
│   └── ...
└── lab-05/
    └── ...
```

---

## Grading Rubric

Each lab is graded out of 20 points:

- **Functionality (10 points)**: Code runs without errors, all TODOs completed
- **Code Quality (4 points)**: Clean, readable, well-commented code
- **Documentation (3 points)**: Clear README and analysis
- **Innovation (3 points)**: Going beyond requirements, creative solutions

**Total: 100 points**
**Passing Score: 70 points**

---

## Tips for Success

1. **Start Early**: Labs build on each other
2. **Test Often**: Run tests after each section
3. **Document As You Go**: Don't wait until the end
4. **Ask Questions**: Use office hours or forums
5. **Review Lessons**: Refer back to module lessons
6. **Experiment**: Try variations beyond requirements

---

## Troubleshooting

### Common Issues

**MLflow Connection Issues:**
```bash
# Check MLflow server is running
curl http://localhost:5000/health
```

**Redis Connection Issues:**
```bash
# Test Redis connection
redis-cli ping
```

**Docker Build Failures:**
```bash
# Clean Docker cache
docker system prune -a
```

**Import Errors:**
```bash
# Verify all packages installed
pip list
```

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feast Documentation](https://docs.feast.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Happy Learning!** 🚀

If you complete all 5 labs, you'll have hands-on experience with the entire MLOps lifecycle!
