"""
Project 02: End-to-End MLOps Pipeline
Training Pipeline DAG - Handles model training, evaluation, and registration

This DAG demonstrates:
- Loading versioned data from DVC
- Model training with MLflow experiment tracking
- Hyperparameter tuning
- Model evaluation and validation
- Model registration in MLflow Model Registry
- Automatic promotion to Staging/Production based on metrics
- Triggering deployment pipeline on model promotion

Author: AI Infrastructure Learning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
import os

# TODO: Import your custom modules
# from src.training.train import train_model
# from src.training.evaluate import evaluate_model
# from src.training.register import register_model

# ==================== DAG Configuration ====================

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=4),
}

# ==================== Helper Functions ====================

def load_versioned_data(**context):
    """
    TODO: Load data versioned with DVC

    Steps:
    1. Pull latest data from DVC remote
    2. Load training, validation, and test datasets
    3. Verify data integrity
    4. Return data paths

    DVC commands:
    - dvc pull  # Pull latest data from remote

    Returns:
        dict: Paths to train/val/test data
    """
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    logger.info("Loading versioned data from DVC...")

    # TODO: Pull data from DVC remote
    # subprocess.run(['dvc', 'pull'], check=True)

    processed_dir = os.getenv('PROCESSED_DATA_DIR', './data/processed')

    # Get latest data files
    # In production, you'd use specific DVC version
    data_paths = {
        'train': os.path.join(processed_dir, 'train_latest.csv'),
        'val': os.path.join(processed_dir, 'val_latest.csv'),
        'test': os.path.join(processed_dir, 'test_latest.csv'),
    }

    logger.info(f"Data loaded: {data_paths}")
    return data_paths


def train_model_with_mlflow(**context):
    """
    TODO: Train ML model with MLflow experiment tracking

    Steps:
    1. Initialize MLflow experiment
    2. Load training data
    3. Set hyperparameters
    4. Train model
    5. Log parameters, metrics, and artifacts to MLflow
    6. Save model checkpoint

    MLflow logging:
    - mlflow.log_param() for hyperparameters
    - mlflow.log_metric() for metrics (accuracy, loss, etc.)
    - mlflow.log_artifact() for plots, configs
    - mlflow.sklearn.log_model() or mlflow.pytorch.log_model()

    Returns:
        dict: Training results including run_id, metrics, model_path
    """
    import logging
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logger = logging.getLogger(__name__)
    logger.info("Starting model training with MLflow...")

    # Get data paths from previous task
    ti = context['ti']
    data_paths = ti.xcom_pull(task_ids='load_data')

    # TODO: Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment('model_training_experiment')

    with mlflow.start_run(run_name=f"training_{context['ds']}") as run:
        # TODO: Load data
        # train_df = pd.read_csv(data_paths['train'])
        # val_df = pd.read_csv(data_paths['val'])
        # X_train, y_train = train_df.drop('target', axis=1), train_df['target']
        # X_val, y_val = val_df.drop('target', axis=1), val_df['target']

        # Placeholder data
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]
        X_val = [[2, 3]]
        y_val = [1]

        # TODO: Define hyperparameters
        params = {
            'n_estimators': int(os.getenv('N_ESTIMATORS', 100)),
            'max_depth': int(os.getenv('MAX_DEPTH', 10)),
            'min_samples_split': int(os.getenv('MIN_SAMPLES_SPLIT', 2)),
            'random_state': 42
        }

        # Log parameters
        mlflow.log_params(params)

        # TODO: Train model
        logger.info("Training model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # TODO: Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions, average='weighted', zero_division=0)

        # Log metrics
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.log_metric('val_f1_score', val_f1)
        mlflow.log_metric('train_size', len(y_train))

        logger.info(f"Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        # TODO: Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally as well
        model_dir = os.getenv('MODELS_DIR', './models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{context['ds']}.pkl")

        import joblib
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        run_id = run.info.run_id

    results = {
        'run_id': run_id,
        'val_accuracy': val_accuracy,
        'val_f1_score': val_f1,
        'model_path': model_path,
        'mlflow_tracking_uri': mlflow.get_tracking_uri()
    }

    logger.info(f"Training complete: {results}")
    return results


def evaluate_model_performance(**context):
    """
    TODO: Comprehensive model evaluation

    Steps:
    1. Load trained model
    2. Evaluate on test set
    3. Calculate multiple metrics
    4. Generate evaluation plots (ROC curve, confusion matrix, etc.)
    5. Compare with baseline and previous models
    6. Determine if model meets promotion criteria

    Evaluation metrics to implement:
    - Accuracy, Precision, Recall, F1-score
    - ROC AUC, PR AUC
    - Confusion matrix
    - Feature importance
    - Model calibration plots

    Returns:
        dict: Comprehensive evaluation results
    """
    import logging
    import mlflow

    logger = logging.getLogger(__name__)
    logger.info("Evaluating model performance...")

    # Get training results
    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_model')

    run_id = training_results['run_id']
    val_accuracy = training_results['val_accuracy']

    # TODO: Load model and test data
    # TODO: Calculate comprehensive metrics
    # TODO: Generate plots
    # TODO: Compare with baseline

    # Placeholder evaluation
    test_accuracy = val_accuracy * 0.98  # Simulating slightly lower test accuracy
    meets_criteria = test_accuracy >= float(os.getenv('MODEL_PROMOTION_THRESHOLD', 0.85))

    results = {
        'test_accuracy': test_accuracy,
        'test_f1_score': training_results['val_f1_score'] * 0.98,
        'meets_promotion_criteria': meets_criteria,
        'run_id': run_id
    }

    logger.info(f"Evaluation complete: {results}")
    return results


def decide_model_promotion(**context):
    """
    TODO: Decide whether to promote model to Staging/Production

    Decision logic:
    1. Check if model meets minimum performance threshold
    2. Compare with current Production model
    3. Check for data drift
    4. Verify model bias and fairness metrics
    5. Return decision (promote_to_staging, promote_to_production, reject)

    Returns:
        str: Task ID to execute next ('register_model_staging', 'register_model_production', 'skip_registration')
    """
    import logging

    logger = logging.getLogger(__name__)

    ti = context['ti']
    evaluation = ti.xcom_pull(task_ids='evaluate_model')

    if evaluation['meets_promotion_criteria']:
        logger.info("Model meets criteria - promoting to Staging")
        return 'register_model_staging'
    else:
        logger.warning("Model does not meet criteria - skipping registration")
        return 'skip_registration'


def register_model_in_mlflow(stage='Staging', **context):
    """
    TODO: Register model in MLflow Model Registry

    Steps:
    1. Connect to MLflow Model Registry
    2. Register model with version
    3. Add model metadata (metrics, tags, description)
    4. Transition model to specified stage (Staging/Production)
    5. Archive old models

    MLflow Model Registry workflow:
    - mlflow.register_model() to register
    - client.transition_model_version_stage() to promote
    - Add tags with client.set_model_version_tag()

    Args:
        stage: Model stage ('Staging', 'Production', 'Archived')

    Returns:
        dict: Registration details (model_name, version, stage)
    """
    import logging
    import mlflow
    from mlflow.tracking import MlflowClient

    logger = logging.getLogger(__name__)
    logger.info(f"Registering model in MLflow Model Registry as '{stage}'...")

    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_model')
    run_id = training_results['run_id']

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    client = MlflowClient()

    # Model name
    model_name = os.getenv('MODEL_NAME', 'mlops-classifier')

    # TODO: Register model
    """
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri, model_name)
    version = model_details.version

    # Transition to stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )

    # Add tags
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="training_date",
        value=context['ds']
    )
    """

    # Placeholder
    version = 1
    logger.info(f"Model '{model_name}' version {version} registered in '{stage}'")

    return {
        'model_name': model_name,
        'version': version,
        'stage': stage,
        'run_id': run_id
    }


def skip_model_registration(**context):
    """
    Skip model registration and log reason
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Model registration skipped - performance below threshold")
    return {'registered': False}


# ==================== DAG Definition ====================

with DAG(
    dag_id='training_pipeline',
    default_args=default_args,
    description='Train, evaluate, and register ML models',
    schedule_interval='0 4 * * *',  # Run daily at 4 AM (after data pipeline)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'training', 'machine-learning'],
    max_active_runs=1,
) as dag:

    # Task 1: Load versioned data
    task_load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_versioned_data,
        provide_context=True,
    )

    # Task 2: Train model with MLflow
    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_with_mlflow,
        provide_context=True,
    )

    # Task 3: Evaluate model
    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_performance,
        provide_context=True,
    )

    # Task 4: Decide on promotion
    task_decide = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=decide_model_promotion,
        provide_context=True,
    )

    # Task 5a: Register model in Staging
    task_register_staging = PythonOperator(
        task_id='register_model_staging',
        python_callable=lambda **context: register_model_in_mlflow('Staging', **context),
        provide_context=True,
    )

    # Task 5b: Skip registration
    task_skip = PythonOperator(
        task_id='skip_registration',
        python_callable=skip_model_registration,
        provide_context=True,
    )

    # Task 6: Trigger deployment pipeline (if model registered)
    task_trigger_deployment = TriggerDagRunOperator(
        task_id='trigger_deployment',
        trigger_dag_id='deployment_pipeline',
        wait_for_completion=False,
        trigger_rule='all_success',
    )

    # Define task dependencies
    task_load_data >> task_train >> task_evaluate >> task_decide
    task_decide >> [task_register_staging, task_skip]
    task_register_staging >> task_trigger_deployment

# TODO: Add hyperparameter tuning task group with Ray Tune or Optuna
# TODO: Implement model comparison with A/B testing
# TODO: Add distributed training for large models
# TODO: Implement early stopping based on validation metrics
# TODO: Add model explainability (SHAP values)
