"""
Project 02: End-to-End MLOps Pipeline
Data Pipeline DAG - Handles data ingestion, validation, and preprocessing

This DAG demonstrates:
- Data ingestion from multiple sources
- Data validation with Great Expectations
- Data preprocessing and feature engineering
- Data versioning with DVC
- Error handling and retries
- Slack/Email notifications on failure

Author: AI Infrastructure Learning
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
import os

# TODO: Import your custom data processing modules
# from src.data.ingestion import ingest_data
# from src.data.validation import validate_data
# from src.data.preprocessing import preprocess_data

# ==================== DAG Configuration ====================

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# ==================== Helper Functions ====================

def ingest_raw_data(**context):
    """
    TODO: Implement data ingestion logic

    Steps:
    1. Connect to data source (API, database, file system, etc.)
    2. Download/fetch raw data
    3. Save to raw data directory
    4. Log metadata (size, source, timestamp)
    5. Push metadata to XCom for downstream tasks

    Example data sources:
    - CSV files from S3/GCS
    - Database tables (PostgreSQL, MySQL)
    - REST APIs
    - Streaming data (Kafka)

    Returns:
        dict: Metadata about ingested data (file_path, size, row_count, etc.)
    """
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion...")

    # TODO: Replace with your actual data source
    # Example: Download from URL, read from database, etc.

    # Placeholder implementation
    data_path = os.path.join(os.getenv('RAW_DATA_DIR', './data/raw'),
                             f"data_{context['ds']}.csv")

    # TODO: Implement actual data fetching
    # Example:
    # df = pd.read_csv('https://example.com/dataset.csv')
    # df = pd.read_sql('SELECT * FROM table', connection)

    # For now, create a dummy dataset
    logger.warning("Using dummy data - Replace with actual data source!")
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })

    # Save raw data
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info(f"Data saved to {data_path}")

    # Return metadata for downstream tasks
    metadata = {
        'file_path': data_path,
        'row_count': len(df),
        'column_count': len(df.columns),
        'file_size_mb': os.path.getsize(data_path) / (1024 * 1024),
        'ingestion_timestamp': datetime.now().isoformat()
    }

    logger.info(f"Ingestion complete: {metadata}")
    return metadata


def validate_data_quality(**context):
    """
    TODO: Implement data validation using Great Expectations or Pandera

    Steps:
    1. Load raw data
    2. Define expectations (schema, ranges, distributions, etc.)
    3. Run validation suite
    4. Generate validation report
    5. Raise exception if critical validations fail
    6. Log warnings for non-critical failures

    Validation checks to implement:
    - Schema validation (column names, data types)
    - Missing value checks
    - Range validation (min/max for numerical features)
    - Categorical value validation
    - Distribution checks (mean, std, percentiles)
    - Duplicate detection
    - Outlier detection

    Returns:
        dict: Validation results summary
    """
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting data validation...")

    # Get metadata from previous task
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='ingest_raw_data')
    data_path = metadata['file_path']

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")

    # TODO: Implement Great Expectations validation
    # Example using Great Expectations:
    """
    import great_expectations as ge

    # Convert to GE DataFrame
    ge_df = ge.from_pandas(df)

    # Define expectations
    ge_df.expect_column_to_exist('feature1')
    ge_df.expect_column_values_to_not_be_null('target')
    ge_df.expect_column_values_to_be_between('feature1', min_value=0, max_value=100)

    # Run validation
    validation_result = ge_df.validate()

    if not validation_result['success']:
        logger.error("Data validation failed!")
        raise ValueError("Data quality check failed")
    """

    # Placeholder validation
    validation_results = {
        'total_checks': 5,
        'passed_checks': 5,
        'failed_checks': 0,
        'missing_values': df.isnull().sum().to_dict(),
        'validation_passed': True
    }

    logger.info(f"Validation complete: {validation_results}")
    return validation_results


def preprocess_and_engineer_features(**context):
    """
    TODO: Implement data preprocessing and feature engineering

    Steps:
    1. Load validated raw data
    2. Handle missing values (imputation, removal)
    3. Encode categorical variables
    4. Scale/normalize numerical features
    5. Create derived features
    6. Split data (train/validation/test)
    7. Save processed data

    Feature engineering examples:
    - Create interaction features
    - Binning continuous variables
    - Time-based features (day, month, seasonality)
    - Aggregations (rolling mean, cumulative sum)
    - Text features (TF-IDF, embeddings)

    Returns:
        dict: Metadata about processed data
    """
    import pandas as pd
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing...")

    # Get metadata from ingestion task
    ti = context['ti']
    metadata = ti.xcom_pull(task_ids='ingest_raw_data')
    data_path = metadata['file_path']

    # Load raw data
    df = pd.read_csv(data_path)

    # TODO: Implement actual preprocessing
    # Example preprocessing steps:

    # 1. Handle missing values
    # df = df.fillna(df.mean())

    # 2. Feature engineering
    # df['feature3'] = df['feature1'] * df['feature2']

    # 3. Encoding (if needed)
    # df = pd.get_dummies(df, columns=['categorical_col'])

    # 4. Scaling
    # scaler = StandardScaler()
    # df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 5. Train/validation/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Save processed data
    processed_dir = os.getenv('PROCESSED_DATA_DIR', './data/processed')
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(processed_dir, f"train_{context['ds']}.csv")
    val_path = os.path.join(processed_dir, f"val_{context['ds']}.csv")
    test_path = os.path.join(processed_dir, f"test_{context['ds']}.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    metadata = {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'feature_count': len(df.columns) - 1,  # excluding target
    }

    logger.info(f"Preprocessing complete: {metadata}")
    return metadata


def version_data_with_dvc(**context):
    """
    TODO: Implement DVC data versioning

    Steps:
    1. Initialize DVC (if not already done)
    2. Add processed data to DVC tracking
    3. Configure remote storage (S3, GCS, Azure, etc.)
    4. Push data to remote storage
    5. Commit .dvc files to Git

    DVC Commands to implement:
    - dvc init (if first time)
    - dvc add data/processed/*.csv
    - dvc remote add -d myremote s3://bucket/path
    - dvc push
    - git add data/processed/*.dvc .dvc/config
    - git commit -m "Update data version"

    Returns:
        dict: DVC versioning metadata
    """
    import logging
    import subprocess

    logger = logging.getLogger(__name__)
    logger.info("Starting DVC versioning...")

    # TODO: Implement DVC versioning
    # Example:
    """
    # Add data to DVC
    result = subprocess.run(['dvc', 'add', 'data/processed/train.csv'],
                          capture_output=True, text=True)
    logger.info(f"DVC add output: {result.stdout}")

    # Push to remote
    result = subprocess.run(['dvc', 'push'],
                          capture_output=True, text=True)
    logger.info(f"DVC push output: {result.stdout}")

    # Get DVC file hash
    with open('data/processed/train.csv.dvc', 'r') as f:
        dvc_info = yaml.safe_load(f)
    data_version = dvc_info['md5']
    """

    # Placeholder
    data_version = "abc123def456"  # Replace with actual DVC hash

    logger.info(f"Data versioned with DVC: {data_version}")
    return {'dvc_version': data_version, 'versioning_success': True}


def send_success_notification(**context):
    """
    TODO: Implement success notification (Slack, Email, etc.)

    Steps:
    1. Gather pipeline execution summary
    2. Format notification message
    3. Send via configured channel (Slack webhook, SMTP, etc.)

    Information to include:
    - Pipeline name and execution date
    - Success status
    - Data statistics (row counts, file sizes)
    - Execution duration
    - Link to Airflow UI

    Returns:
        bool: True if notification sent successfully
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Sending success notification...")

    # TODO: Implement notification logic
    # Example Slack notification:
    """
    import requests

    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    message = {
        'text': f"✅ Data Pipeline Success for {context['ds']}",
        'blocks': [
            {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': f"*Data Pipeline Completed*\\nDate: {context['ds']}"
                }
            }
        ]
    }
    requests.post(webhook_url, json=message)
    """

    logger.info("Notification sent successfully")
    return True


# ==================== DAG Definition ====================

with DAG(
    dag_id='data_pipeline',
    default_args=default_args,
    description='Ingest, validate, and preprocess data for ML pipeline',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'data-engineering', 'etl'],
    max_active_runs=1,
) as dag:

    # Task 1: Ingest raw data
    task_ingest = PythonOperator(
        task_id='ingest_raw_data',
        python_callable=ingest_raw_data,
        provide_context=True,
    )

    # Task 2: Validate data quality
    task_validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        provide_context=True,
    )

    # Task 3: Preprocess and engineer features
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_and_engineer_features,
        provide_context=True,
    )

    # Task 4: Version data with DVC
    task_version = PythonOperator(
        task_id='version_data_dvc',
        python_callable=version_data_with_dvc,
        provide_context=True,
    )

    # Task 5: Send success notification
    task_notify = PythonOperator(
        task_id='send_notification',
        python_callable=send_success_notification,
        provide_context=True,
        trigger_rule='all_success',
    )

    # TODO: Add additional tasks
    # - Data quality report generation
    # - Data drift detection
    # - Feature store update (if using Feast/Tecton)
    # - Trigger training pipeline

    # Define task dependencies
    task_ingest >> task_validate >> task_preprocess >> task_version >> task_notify

# TODO: Add sensor tasks for external data dependencies
# TODO: Implement branch operator for conditional logic (e.g., skip training if data quality is poor)
# TODO: Add task groups for better organization
# TODO: Implement SLA monitoring
