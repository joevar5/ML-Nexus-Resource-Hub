# Exercise 04: ML Workflow Orchestration with Airflow

**Estimated Time**: 32-40 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, Apache Airflow, Docker, AWS/GCP SDK

## Overview

Build production-grade ML workflow orchestration using Apache Airflow. Implement DAGs for model training pipelines, data validation, feature engineering, model deployment, and monitoring. Handle dependencies, retries, alerts, and dynamic task generation. This exercise teaches end-to-end ML pipeline orchestration patterns essential for productionizing machine learning at scale.

In production ML platforms, workflow orchestration is critical for:
- **Dependency Management**: Ensure data ready before training starts
- **Scheduling**: Run training daily, feature engineering hourly
- **Retry Logic**: Automatically retry failed tasks with backoff
- **Monitoring**: Track pipeline health, alert on failures
- **Reproducibility**: Version code, data, and models together

## Learning Objectives

By completing this exercise, you will:

1. **Design complex ML DAGs** with proper task dependencies
2. **Implement data validation** before training
3. **Build dynamic DAGs** (generate tasks programmatically)
4. **Handle failures** with retries, alerts, and rollbacks
5. **Integrate with ML tools** (MLflow, DVC, feature stores)
6. **Implement sensor patterns** for external dependencies
7. **Build custom operators** for ML-specific tasks

## Business Context

**Real-World Scenario**: Your ML platform trains 20 models daily with manual coordination. Current problems:

- **Manual orchestration**: Data scientists SSH to servers, run scripts manually
- **No dependency tracking**: Training starts before features ready (fails 30% of time)
- **No retries**: Transient AWS errors require manual re-runs
- **No monitoring**: Failed pipelines discovered hours later
- **Reproducibility issues**: Can't recreate models from 2 weeks ago
- **Resource conflicts**: Multiple training jobs compete for same GPU

Your task: Build Airflow-based orchestration that:
- Automates end-to-end ML pipeline (data → features → train → validate → deploy)
- Validates data quality before training
- Retries failed tasks 3x with exponential backoff
- Alerts on failures within 5 minutes (Slack/PagerDuty)
- Tracks lineage (which data/code version produced which model)
- Schedules 20 models without resource conflicts

## Project Structure

```
exercise-04-workflow-orchestration-airflow/
├── README.md
├── requirements.txt
├── docker-compose.yaml          # Airflow + Postgres + Redis
├── airflow/
│   ├── airflow.cfg              # Airflow configuration
│   └── dags/
│       ├── ml_training_dag.py           # Main training DAG
│       ├── feature_engineering_dag.py   # Feature pipeline DAG
│       ├── model_deployment_dag.py      # Deployment DAG
│       ├── data_validation_dag.py       # Data quality checks
│       └── monitoring_dag.py            # Model monitoring DAG
├── plugins/
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── data_validation_operator.py
│   │   ├── model_training_operator.py
│   │   ├── model_registry_operator.py
│   │   └── slack_alert_operator.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── s3_data_sensor.py
│   │   └── model_performance_sensor.py
│   └── hooks/
│       ├── __init__.py
│       ├── mlflow_hook.py
│       └── feature_store_hook.py
├── src/
│   └── ml_pipeline/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py             # Data loading
│       │   └── validator.py          # Data validation
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py        # Feature engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── trainer.py            # Model training
│       │   └── evaluator.py          # Model evaluation
│       └── deployment/
│           ├── __init__.py
│           └── deployer.py           # Model deployment
├── tests/
│   ├── test_dags.py                  # Test DAG integrity
│   ├── test_operators.py
│   └── test_tasks.py
├── config/
│   ├── model_configs/
│   │   ├── fraud_detection.yaml
│   │   └── recommendation.yaml
│   └── data_validation_rules.yaml
└── docs/
    ├── DESIGN.md
    ├── DAG_REFERENCE.md
    └── TROUBLESHOOTING.md
```

## Requirements

### Functional Requirements

1. **ML Training DAG**:
   - Extract data from S3/GCS
   - Validate data quality
   - Engineer features
   - Train model
   - Evaluate model
   - Register model (if metrics pass)
   - Deploy model (if champion)

2. **Data Validation**:
   - Schema validation (column types, names)
   - Statistical validation (distributions, outliers)
   - Freshness checks (data not older than X hours)
   - Completeness checks (no missing values in critical columns)

3. **Dynamic DAGs**:
   - Generate DAG per model from config
   - Parameterized training (hyperparameters from config)
   - Dynamic task generation (one task per data partition)

4. **Failure Handling**:
   - Retry failed tasks (3 attempts, exponential backoff)
   - Send alerts on failures (Slack, email)
   - Rollback on deployment failures
   - Dead letter queue for permanently failed tasks

5. **Monitoring**:
   - Track DAG run duration
   - Monitor task success rates
   - Alert on SLA violations (DAG should complete in <2 hours)
   - Track model performance drift

### Non-Functional Requirements

- **Scalability**: Handle 50+ concurrent DAG runs
- **Reliability**: 99.5% DAG success rate
- **Observability**: Detailed logs, metrics, lineage tracking
- **Performance**: DAG scheduling latency <30s

## Implementation Tasks

### Task 1: Airflow Setup (5-6 hours)

Set up Airflow with proper configuration.

```yaml
# docker-compose.yaml

version: '3.8'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.8.0-python3.9
  environment:
    - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
    - AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow
    - AIRFLOW__CELERY__BROKER_URL=redis://:@redis:6379/0
    - AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}
    - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
    - AIRFLOW__CORE__LOAD_EXAMPLES=false
    - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
    - AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=30
  volumes:
    - ./airflow/dags:/opt/airflow/dags
    - ./plugins:/opt/airflow/plugins
    - ./logs:/opt/airflow/logs
    - ./src:/opt/airflow/src
  depends_on:
    - postgres
    - redis

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    healthcheck:
      test: ["CMD-SHELL", 'celery --app airflow.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com \
          --password admin
    restart: on-failure
```

**Acceptance Criteria**:
- ✅ Airflow webserver accessible
- ✅ Scheduler running
- ✅ Celery workers healthy
- ✅ DAGs folder monitored

---

### Task 2: ML Training DAG (8-10 hours)

Build end-to-end training DAG.

```python
# airflow/dags/ml_training_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import timedelta
import sys
sys.path.append('/opt/airflow/src')

from ml_pipeline.data.loader import load_data
from ml_pipeline.data.validator import validate_data
from ml_pipeline.features.engineering import engineer_features
from ml_pipeline.models.trainer import train_model
from ml_pipeline.models.evaluator import evaluate_model
from ml_pipeline.deployment.deployer import deploy_model

# DAG configuration
default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'email': ['ml-alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),  # Fail if task runs >2 hours
}

# Model configuration (loaded from Airflow Variables or config file)
MODEL_CONFIG = {
    'model_name': 'fraud_detection',
    'model_type': 'xgboost',
    'data_path': 's3://ml-data/transactions/{{ ds }}/',  # Templated with execution_date
    'model_output_path': 's3://ml-models/fraud_detection/{{ ds }}/',
    'feature_store_table': 'fraud_features',
    'target_metric': 'auc',
    'min_auc_threshold': 0.85,  # Deploy only if AUC >= 0.85
}

with DAG(
    dag_id='ml_training_fraud_detection',
    default_args=default_args,
    description='Train and deploy fraud detection model',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=days_ago(1),
    catchup=False,  # Don't backfill historical runs
    max_active_runs=1,  # Only one run at a time
    tags=['ml', 'training', 'fraud'],
) as dag:

    # Task 1: Wait for upstream feature engineering DAG
    wait_for_features = ExternalTaskSensor(
        task_id='wait_for_features',
        external_dag_id='feature_engineering_fraud',
        external_task_id='write_to_feature_store',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='poke',
        poke_interval=60,  # Check every 60 seconds
        timeout=3600,  # Wait max 1 hour
    )

    # Task 2: Load data from S3
    def load_training_data(**context):
        """
        Load data from S3

        Uses Airflow context to get execution_date for templating
        """
        execution_date = context['ds']  # YYYY-MM-DD format
        data_path = MODEL_CONFIG['data_path'].replace('{{ ds }}', execution_date)

        # TODO: Load data
        df = load_data(data_path)

        # TODO: Push data statistics to XCom
        context['task_instance'].xcom_push(
            key='data_stats',
            value={
                'row_count': len(df),
                'column_count': len(df.columns),
                'fraud_rate': df['is_fraud'].mean()
            }
        )

        # TODO: Save data to temp location for next tasks
        temp_path = f"/tmp/{execution_date}_data.parquet"
        df.to_parquet(temp_path)
        context['task_instance'].xcom_push(key='data_path', value=temp_path)

        return temp_path

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_training_data,
        provide_context=True
    )

    # Task 3: Validate data quality
    def validate_data_quality(**context):
        """
        Validate data before training

        Checks:
        - Schema matches expected
        - No missing values in critical columns
        - Distributions within expected ranges
        - Sufficient data volume
        """
        data_path = context['task_instance'].xcom_pull(
            task_ids='load_data',
            key='data_path'
        )

        # TODO: Run validation
        validation_result = validate_data(
            data_path,
            schema_path='config/schemas/fraud_detection.json',
            validation_rules='config/data_validation_rules.yaml'
        )

        # TODO: Fail task if validation fails
        if not validation_result['passed']:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")

        print(f"✅ Data validation passed: {validation_result}")
        return validation_result

    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_quality,
        provide_context=True
    )

    # Task 4: Engineer features
    def engineer_training_features(**context):
        """
        Feature engineering

        In practice, might use feature store instead of computing here
        """
        data_path = context['task_instance'].xcom_pull(
            task_ids='load_data',
            key='data_path'
        )

        # TODO: Compute features
        features_df = engineer_features(data_path, MODEL_CONFIG)

        # TODO: Save features
        features_path = f"/tmp/{context['ds']}_features.parquet"
        features_df.to_parquet(features_path)
        context['task_instance'].xcom_push(key='features_path', value=features_path)

        return features_path

    engineer_features_task = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_training_features,
        provide_context=True
    )

    # Task 5: Train model
    def train_ml_model(**context):
        """
        Train ML model

        Logs to MLflow for experiment tracking
        """
        features_path = context['task_instance'].xcom_pull(
            task_ids='engineer_features',
            key='features_path'
        )

        # TODO: Train model
        model, metrics, artifacts = train_model(
            features_path,
            model_config=MODEL_CONFIG,
            mlflow_tracking_uri='http://mlflow:5000',
            experiment_name='fraud_detection'
        )

        # TODO: Save model path to XCom
        model_path = artifacts['model_path']
        context['task_instance'].xcom_push(key='model_path', value=model_path)
        context['task_instance'].xcom_push(key='metrics', value=metrics)

        return model_path

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_ml_model,
        provide_context=True,
        pool='gpu_pool',  # Limit concurrent GPU tasks
        priority_weight=10  # Higher priority than other tasks
    )

    # Task 6: Evaluate model
    def evaluate_ml_model(**context):
        """
        Evaluate model on hold-out set

        Compare to champion model (current production model)
        """
        model_path = context['task_instance'].xcom_pull(
            task_ids='train_model',
            key='model_path'
        )

        # TODO: Evaluate model
        eval_results = evaluate_model(
            model_path,
            test_data_path=MODEL_CONFIG['data_path'],
            metrics=['auc', 'precision', 'recall', 'f1']
        )

        # TODO: Compare to champion model
        champion_auc = Variable.get('fraud_detection_champion_auc', default_var=0.80)
        challenger_auc = eval_results['auc']

        print(f"Champion AUC: {champion_auc}")
        print(f"Challenger AUC: {challenger_auc}")

        # TODO: Decide if should deploy
        should_deploy = challenger_auc >= MODEL_CONFIG['min_auc_threshold']

        context['task_instance'].xcom_push(key='should_deploy', value=should_deploy)
        context['task_instance'].xcom_push(key='eval_results', value=eval_results)

        return should_deploy

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_ml_model,
        provide_context=True
    )

    # Task 7: Register model (if evaluation passed)
    def register_ml_model(**context):
        """
        Register model in MLflow Model Registry

        Tag as "staging" for review before production
        """
        should_deploy = context['task_instance'].xcom_pull(
            task_ids='evaluate_model',
            key='should_deploy'
        )

        if not should_deploy:
            print("⚠️ Model did not pass evaluation criteria. Skipping registration.")
            return None

        model_path = context['task_instance'].xcom_pull(
            task_ids='train_model',
            key='model_path'
        )
        metrics = context['task_instance'].xcom_pull(
            task_ids='train_model',
            key='metrics'
        )

        # TODO: Register in MLflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri='http://mlflow:5000')

        # Create model version
        model_version = client.create_model_version(
            name=MODEL_CONFIG['model_name'],
            source=model_path,
            run_id=context['task_instance'].xcom_pull(task_ids='train_model', key='run_id'),
            tags={
                'execution_date': context['ds'],
                'auc': str(metrics['auc'])
            }
        )

        # Transition to staging
        client.transition_model_version_stage(
            name=MODEL_CONFIG['model_name'],
            version=model_version.version,
            stage='Staging'
        )

        context['task_instance'].xcom_push(key='model_version', value=model_version.version)
        return model_version.version

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_ml_model,
        provide_context=True
    )

    # Task 8: Send success notification
    send_notification = BashOperator(
        task_id='send_notification',
        bash_command='''
        curl -X POST {{ var.value.slack_webhook_url }} \
        -H 'Content-Type: application/json' \
        -d '{
            "text": "✅ Fraud detection model training completed successfully!\\nAUC: {{ task_instance.xcom_pull(task_ids=\"evaluate_model\", key=\"eval_results\")[\"auc\"] }}\\nModel version: {{ task_instance.xcom_pull(task_ids=\"register_model\", key=\"model_version\") }}"
        }'
        '''
    )

    # Define dependencies
    wait_for_features >> load_data_task >> validate_data_task >> engineer_features_task
    engineer_features_task >> train_model_task >> evaluate_model_task >> register_model_task
    register_model_task >> send_notification
```

**Acceptance Criteria**:
- ✅ DAG executes end-to-end
- ✅ Proper task dependencies
- ✅ XCom for data passing
- ✅ Retry logic works
- ✅ Alerts on failure

---

### Task 3: Custom Operators (6-7 hours)

Build custom operators for ML tasks.

```python
# plugins/operators/data_validation_operator.py

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import pandas as pd
import json
from typing import Dict, List

class DataValidationOperator(BaseOperator):
    """
    Custom operator for data validation

    Validates:
    - Schema (column names, types)
    - Statistical properties (mean, std, percentiles)
    - Data quality (missing values, duplicates)
    - Freshness (timestamp within acceptable range)
    """

    @apply_defaults
    def __init__(
        self,
        data_path: str,
        schema_path: str,
        validation_rules_path: str,
        fail_on_error: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.schema_path = schema_path
        self.validation_rules_path = validation_rules_path
        self.fail_on_error = fail_on_error

    def execute(self, context):
        """Execute validation"""
        self.log.info(f"Validating data at {self.data_path}")

        # TODO: Load data
        df = pd.read_parquet(self.data_path)

        # TODO: Load expected schema
        with open(self.schema_path) as f:
            expected_schema = json.load(f)

        # TODO: Load validation rules
        with open(self.validation_rules_path) as f:
            import yaml
            rules = yaml.safe_load(f)

        # TODO: Run validations
        errors = []

        # Schema validation
        schema_errors = self._validate_schema(df, expected_schema)
        errors.extend(schema_errors)

        # Statistical validation
        stats_errors = self._validate_statistics(df, rules.get('statistics', {}))
        errors.extend(stats_errors)

        # Quality validation
        quality_errors = self._validate_quality(df, rules.get('quality', {}))
        errors.extend(quality_errors)

        # TODO: Return results
        if errors:
            error_msg = f"Validation failed with {len(errors)} errors: {errors}"
            self.log.error(error_msg)
            if self.fail_on_error:
                raise ValueError(error_msg)

        self.log.info("✅ Data validation passed")
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'row_count': len(df),
            'column_count': len(df.columns)
        }

    def _validate_schema(self, df: pd.DataFrame, expected_schema: Dict) -> List[str]:
        """Validate schema matches expected"""
        errors = []

        # Check columns exist
        expected_columns = set(expected_schema['columns'].keys())
        actual_columns = set(df.columns)

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        if missing:
            errors.append(f"Missing columns: {missing}")
        if extra:
            errors.append(f"Extra columns: {extra}")

        # Check data types
        for col, expected_type in expected_schema['columns'].items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    errors.append(f"Column {col}: expected {expected_type}, got {actual_type}")

        return errors

    def _validate_statistics(self, df: pd.DataFrame, rules: Dict) -> List[str]:
        """Validate statistical properties"""
        errors = []

        for col, col_rules in rules.items():
            if col not in df.columns:
                continue

            # Check min/max
            if 'min' in col_rules and df[col].min() < col_rules['min']:
                errors.append(f"{col}: min {df[col].min()} < {col_rules['min']}")
            if 'max' in col_rules and df[col].max() > col_rules['max']:
                errors.append(f"{col}: max {df[col].max()} > {col_rules['max']}")

        return errors

    def _validate_quality(self, df: pd.DataFrame, rules: Dict) -> List[str]:
        """Validate data quality"""
        errors = []

        # Check missing values
        max_missing_pct = rules.get('max_missing_percent', 0.05)
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > max_missing_pct:
                errors.append(f"{col}: {missing_pct:.2%} missing > {max_missing_pct:.2%}")

        # Check duplicates
        max_duplicate_pct = rules.get('max_duplicate_percent', 0.01)
        duplicate_pct = df.duplicated().mean()
        if duplicate_pct > max_duplicate_pct:
            errors.append(f"Duplicates: {duplicate_pct:.2%} > {max_duplicate_pct:.2%}")

        return errors
```

```python
# plugins/operators/model_training_operator.py

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import mlflow
import mlflow.sklearn
from typing import Dict, Any

class ModelTrainingOperator(BaseOperator):
    """
    Custom operator for model training

    Integrates with MLflow for experiment tracking
    """

    @apply_defaults
    def __init__(
        self,
        model_config: Dict[str, Any],
        data_path: str,
        mlflow_tracking_uri: str,
        experiment_name: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        self.data_path = data_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

    def execute(self, context):
        """Train model"""
        self.log.info(f"Training {self.model_config['model_type']} model")

        # TODO: Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"run_{context['ds']}"):
            # TODO: Log parameters
            mlflow.log_params(self.model_config)

            # TODO: Load data and train
            # (Implementation depends on model type)

            # TODO: Log metrics
            metrics = {'auc': 0.87, 'precision': 0.82, 'recall': 0.91}
            mlflow.log_metrics(metrics)

            # TODO: Log model
            # mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

        self.log.info(f"Training complete. MLflow run: {run_id}")
        return {'run_id': run_id, 'metrics': metrics}
```

**Acceptance Criteria**:
- ✅ Custom operators work in DAGs
- ✅ Proper logging
- ✅ MLflow integration
- ✅ Configurable parameters
- ✅ Error handling

---

### Task 4: Dynamic DAG Generation (5-6 hours)

Generate DAGs dynamically from config.

```python
# airflow/dags/dynamic_model_dags.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import yaml
import glob

# Load all model configs
model_configs = []
for config_file in glob.glob('/opt/airflow/config/model_configs/*.yaml'):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        model_configs.append(config)

# Generate one DAG per model
for model_config in model_configs:
    dag_id = f"ml_training_{model_config['model_name']}"

    default_args = {
        'owner': 'ml-platform',
        'retries': model_config.get('retries', 3),
        'retry_delay': timedelta(minutes=5),
    }

    # Create DAG
    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        description=f"Train {model_config['model_name']} model",
        schedule_interval=model_config.get('schedule', '0 2 * * *'),
        start_date=days_ago(1),
        catchup=False,
        tags=['ml', 'training', model_config['model_name']],
    )

    # Add tasks (same structure as manual DAG)
    with dag:
        # TODO: Add tasks dynamically
        pass

    # Register DAG in globals (Airflow discovers it)
    globals()[dag_id] = dag
```

**Example model config**:

```yaml
# config/model_configs/fraud_detection.yaml

model_name: fraud_detection
model_type: xgboost
schedule: "0 2 * * *"  # Daily at 2 AM
retries: 3

data:
  source: s3://ml-data/transactions/
  target_column: is_fraud
  features:
    - amount
    - merchant_category
    - transaction_count_1h
    - transaction_avg_24h

training:
  hyperparameters:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
  validation_split: 0.2

deployment:
  min_auc_threshold: 0.85
  endpoint_name: fraud-detection-api
```

**Acceptance Criteria**:
- ✅ Generate DAGs from config files
- ✅ One DAG per model
- ✅ Parameterized tasks
- ✅ Config validation
- ✅ Easy to add new models

---

### Task 5: Sensors and Hooks (4-5 hours)

Build sensors for external dependencies.

```python
# plugins/sensors/s3_data_sensor.py

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import boto3
from datetime import datetime, timedelta

class S3DataSensor(BaseSensorOperator):
    """
    Sensor that waits for data to appear in S3

    Checks for file existence and optionally validates:
    - File size > min_size_mb
    - File modified within last X hours (freshness)
    """

    @apply_defaults
    def __init__(
        self,
        bucket: str,
        key: str,
        min_size_mb: float = 0,
        max_age_hours: int = 24,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bucket = bucket
        self.key = key
        self.min_size_mb = min_size_mb
        self.max_age_hours = max_age_hours

    def poke(self, context):
        """
        Check if data exists and meets criteria

        Returns True when data ready, False otherwise
        """
        self.log.info(f"Checking for s3://{self.bucket}/{self.key}")

        s3 = boto3.client('s3')

        try:
            # TODO: Check if object exists
            response = s3.head_object(Bucket=self.bucket, Key=self.key)

            # TODO: Validate size
            size_mb = response['ContentLength'] / (1024 * 1024)
            if size_mb < self.min_size_mb:
                self.log.info(f"File too small: {size_mb:.2f} MB < {self.min_size_mb} MB")
                return False

            # TODO: Validate freshness
            last_modified = response['LastModified']
            age = datetime.now(last_modified.tzinfo) - last_modified
            if age > timedelta(hours=self.max_age_hours):
                self.log.warning(f"File too old: {age} > {self.max_age_hours} hours")
                return False

            self.log.info(f"✅ Data ready: {size_mb:.2f} MB, modified {age} ago")
            return True

        except s3.exceptions.NoSuchKey:
            self.log.info("File not found yet")
            return False
```

```python
# plugins/hooks/mlflow_hook.py

from airflow.hooks.base import BaseHook
import mlflow
from mlflow.tracking import MlflowClient

class MLflowHook(BaseHook):
    """
    Hook for MLflow integration

    Provides helper methods for MLflow operations
    """

    def __init__(self, mlflow_conn_id: str = 'mlflow_default'):
        self.conn_id = mlflow_conn_id
        self.connection = self.get_connection(mlflow_conn_id)
        self.tracking_uri = self.connection.host
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def get_best_model(self, experiment_name: str, metric: str = 'auc') -> str:
        """Get best model from experiment"""
        experiment = self.client.get_experiment_by_name(experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        if runs:
            return runs[0].info.run_id
        return None

    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to stage (Staging, Production)"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
```

**Acceptance Criteria**:
- ✅ S3 sensor waits for data
- ✅ Validate file size and freshness
- ✅ MLflow hook for model operations
- ✅ Proper poke interval
- ✅ Timeout handling

---

### Task 6: Monitoring and Alerting (4-5 hours)

Implement monitoring and alerting.

```python
# airflow/dags/monitoring_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import timedelta

def check_dag_sla(**context):
    """
    Check if DAGs are meeting SLAs

    Alert if:
    - DAG run duration > 2 hours
    - DAG failure rate > 5%
    - Tasks stuck in queued state > 30 min
    """
    from airflow.models import DagRun, TaskInstance
    from airflow.utils.state import State
    from sqlalchemy import func

    # TODO: Query Airflow metadata database
    # TODO: Calculate SLA violations
    # TODO: Return violations for alerting

    violations = []

    # Check recent DAG runs
    recent_runs = DagRun.find(
        dag_id='ml_training_fraud_detection',
        state=State.SUCCESS,
        execution_start_date=context['execution_date'] - timedelta(days=7)
    )

    for run in recent_runs:
        duration = (run.end_date - run.start_date).total_seconds() / 3600
        if duration > 2:  # SLA: 2 hours
            violations.append(f"DAG {run.dag_id} run {run.execution_date} took {duration:.1f}h")

    return violations

with DAG(
    dag_id='monitoring_ml_pipelines',
    schedule_interval='*/30 * * * *',  # Every 30 minutes
    start_date=days_ago(1),
    catchup=False,
) as dag:

    check_sla_task = PythonOperator(
        task_id='check_sla',
        python_callable=check_dag_sla,
        provide_context=True
    )

    # Send alert if violations found
    send_alert = SlackWebhookOperator(
        task_id='send_slack_alert',
        slack_webhook_conn_id='slack_alerts',
        message="""
        ⚠️ ML Pipeline SLA Violations:
        {{ task_instance.xcom_pull(task_ids='check_sla') }}
        """,
        username='Airflow Monitoring'
    )

    check_sla_task >> send_alert
```

**Acceptance Criteria**:
- ✅ Monitor DAG SLAs
- ✅ Track task failure rates
- ✅ Alert on violations
- ✅ Slack integration
- ✅ Configurable thresholds

---

## Testing Requirements

```python
# tests/test_dags.py

from airflow.models import DagBag
import pytest

def test_dag_loaded():
    """Test that DAG loads without errors"""
    dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_has_tags():
    """Test DAG has proper tags"""
    dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
    dag = dagbag.get_dag('ml_training_fraud_detection')
    assert 'ml' in dag.tags
    assert 'training' in dag.tags

def test_task_dependencies():
    """Test task dependencies are correct"""
    dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
    dag = dagbag.get_dag('ml_training_fraud_detection')

    # Check load_data -> validate_data dependency
    load_task = dag.get_task('load_data')
    validate_task = dag.get_task('validate_data')
    assert validate_task in load_task.downstream_list
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **DAG Success Rate** | >99% | ________% |
| **Scheduling Latency** | <30s | ________s |
| **SLA Violations** | <5% | ________% |
| **Alert Latency** | <5min | ________min |

## Validation

Submit:
1. Complete DAG implementations
2. Custom operators and sensors
3. Dynamic DAG generation
4. Test suite
5. Monitoring dashboards
6. Documentation

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [MLflow Integration](https://mlflow.org/)
- [Airflow Providers](https://airflow.apache.org/docs/apache-airflow-providers/)

---

**Estimated Completion Time**: 32-40 hours

**Skills Practiced**:
- Workflow orchestration
- Apache Airflow DAGs
- Custom operators
- Dynamic DAG generation
- ML pipeline automation
- Monitoring and alerting
