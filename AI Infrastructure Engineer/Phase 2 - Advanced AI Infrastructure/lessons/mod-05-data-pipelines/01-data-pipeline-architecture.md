# Lesson 01: Data Pipeline Architecture for ML

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand the components and patterns of ML data pipelines
- Design scalable data pipeline architectures
- Choose between batch and streaming processing approaches
- Apply data pipeline design principles and best practices
- Identify common anti-patterns and pitfalls
- Plan data pipeline infrastructure for ML projects

## Prerequisites
- Understanding of Python and SQL
- Basic knowledge of databases and data storage
- Familiarity with ML training workflows
- Completed Module 01-04

## Introduction

**Why data pipelines matter for ML:**
- **Data is the foundation**: ML models are only as good as their training data
- **Automation**: Manual data preparation doesn't scale
- **Reproducibility**: Pipelines ensure consistent data transformations
- **Monitoring**: Track data quality and detect issues early
- **Efficiency**: Orchestrated pipelines save time and reduce errors

**Real-world impact:**
- **Uber**: Processes petabytes of data daily through pipelines for ML models (pricing, routing, fraud detection)
- **Netflix**: Data pipelines feed recommendation models trained on 200M+ user interactions/day
- **Spotify**: Pipelines generate features for personalization models in near-real-time
- **Airbnb**: 1000+ Airflow DAGs orchestrate data pipelines for pricing, search, and recommendations

## 1. ML Data Pipeline Components

### 1.1 Pipeline Stages

**Typical ML data pipeline:**

```
┌─────────────┐
│ Data Sources│
│ APIs, DBs,  │
│ Files, etc  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Ingestion  │  ← Collect raw data
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Validation  │  ← Check data quality
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Transformation│  ← Clean, normalize, transform
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Feature Eng. │  ← Generate ML features
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Storage    │  ← Store for training
│ (S3, DB)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ML Training  │  ← Train models
│             │
└─────────────┘
```

### 1.2 Component Breakdown

**1. Data Sources**

Common sources for ML pipelines:
- **Databases**: PostgreSQL, MySQL, MongoDB (user data, transactions)
- **APIs**: REST/GraphQL endpoints (external data, real-time feeds)
- **Object Storage**: S3, GCS, Azure Blob (large datasets, logs, images)
- **Streaming**: Kafka, Kinesis (real-time events, clickstreams)
- **Files**: CSV, Parquet, JSON (batch uploads, exports)

**2. Data Ingestion**

Extract data from sources:
```python
# Example: Ingest from multiple sources
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests

# Database ingestion
def ingest_from_db():
    hook = PostgresHook(postgres_conn_id='analytics_db')
    df = hook.get_pandas_df("SELECT * FROM user_events WHERE date >= %(start_date)s")
    return df

# API ingestion
def ingest_from_api():
    response = requests.get("https://api.example.com/data", headers={"Authorization": f"Bearer {token}"})
    return response.json()

# S3 ingestion
def ingest_from_s3(bucket, key):
    hook = S3Hook(aws_conn_id='aws_default')
    obj = hook.read_key(key, bucket)
    return pd.read_parquet(BytesIO(obj))
```

**3. Data Validation**

Check data quality before processing:
```python
import great_expectations as ge

def validate_data(df):
    """Validate data quality with Great Expectations"""
    # Convert to GE DataFrame
    ge_df = ge.from_pandas(df)

    # Define expectations
    ge_df.expect_column_to_exist("user_id")
    ge_df.expect_column_values_to_not_be_null("user_id")
    ge_df.expect_column_values_to_be_unique("user_id")
    ge_df.expect_column_values_to_be_between("age", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_in_set("country", ["US", "UK", "CA", "AU"])

    # Run validation
    result = ge_df.validate()

    if not result["success"]:
        raise ValueError(f"Data validation failed: {result}")

    return df
```

**4. Data Transformation**

Clean and transform data:
```python
def transform_data(df):
    """Apply transformations"""
    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id', 'timestamp'])

    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)

    # Normalize text
    df['name'] = df['name'].str.lower().str.strip()

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter outliers
    df = df[df['purchase_amount'] < df['purchase_amount'].quantile(0.99)]

    return df
```

**5. Feature Engineering**

Generate ML features:
```python
def engineer_features(df):
    """Create features for ML model"""
    # Time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # Aggregation features
    user_stats = df.groupby('user_id').agg({
        'purchase_amount': ['sum', 'mean', 'count'],
        'timestamp': 'max'
    })
    df = df.merge(user_stats, on='user_id', how='left')

    # Categorical encoding
    df = pd.get_dummies(df, columns=['country', 'device_type'])

    # Feature scaling (for specific columns)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['purchase_amount', 'user_lifetime_value']] = scaler.fit_transform(
        df[['purchase_amount', 'user_lifetime_value']]
    )

    return df
```

**6. Storage**

Store processed data:
```python
def store_data(df, output_path):
    """Store data in partitioned Parquet format"""
    # Partition by date for efficient querying
    df['date'] = df['timestamp'].dt.date

    # Write to S3 in Parquet format (compressed, columnar)
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        partition_cols=['date'],
        index=False
    )
```

## 2. Pipeline Patterns

### 2.1 Batch Processing

**Characteristics:**
- Processes data in scheduled batches (hourly, daily, weekly)
- High throughput, high latency
- Good for historical analysis, model training

**Example use case:** Daily model retraining with previous day's data

```python
# Airflow DAG for batch processing
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_model_training_pipeline',
    default_args=default_args,
    description='Daily batch pipeline for model training',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_from_db,
    op_kwargs={'date': '{{ ds }}'},  # Airflow execution date
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

feature_eng_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=engineer_features,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_data',
    python_callable=store_data,
    op_kwargs={'output_path': 's3://ml-data/processed/{{ ds }}/'},
    dag=dag,
)

# Define dependencies
ingest_task >> validate_task >> transform_task >> feature_eng_task >> store_task
```

### 2.2 Streaming Processing

**Characteristics:**
- Processes data in real-time or near-real-time
- Low latency, lower throughput per event
- Good for real-time features, online learning

**Example use case:** Real-time fraud detection features

```python
# Kafka consumer for streaming processing
from kafka import KafkaConsumer
from json import loads
import pandas as pd

consumer = KafkaConsumer(
    'user_transactions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='ml-feature-engineering',
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

def process_stream():
    """Process streaming transactions in micro-batches"""
    batch = []
    batch_size = 100

    for message in consumer:
        transaction = message.value

        # Add to batch
        batch.append(transaction)

        # Process micro-batch
        if len(batch) >= batch_size:
            df = pd.DataFrame(batch)

            # Validate
            validated_df = validate_data(df)

            # Transform
            transformed_df = transform_data(validated_df)

            # Engineer features
            features_df = engineer_features(transformed_df)

            # Write to feature store (Redis, for real-time serving)
            write_to_feature_store(features_df)

            # Clear batch
            batch = []
```

### 2.3 Lambda Architecture

**Combines batch and streaming:**

```
        ┌─────────────────┐
        │   Data Source   │
        └────────┬────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼────┐    ┌────▼────┐
    │ Batch   │    │ Stream  │  ← Speed layer
    │ Layer   │    │ Layer   │
    └────┬────┘    └────┬────┘
         │               │
         │         ┌─────▼─────┐
         │         │Real-time  │
         │         │Views      │
    ┌────▼─────────▼──────────┐
    │    Serving Layer         │
    │ (Combines batch + stream)│
    └──────────────────────────┘
```

**When to use:**
- Need both historical analysis (batch) and real-time features (stream)
- E.g., recommendation systems (batch: user history, stream: current session)

## 3. Data Pipeline Design Principles

### 3.1 Idempotency

**Principle:** Running the same pipeline multiple times produces the same result.

**Why it matters:**
- Enables safe retries on failure
- Allows backfilling historical data
- Prevents duplicate data

**Example (non-idempotent):**

```python
# ❌ BAD: Appends data, not idempotent
def save_data(df):
    df.to_csv('output.csv', mode='append')  # Duplicates if re-run!
```

**Example (idempotent):**

```python
# ✅ GOOD: Overwrites partition, idempotent
def save_data(df, date):
    df.to_parquet(f's3://ml-data/processed/date={date}/data.parquet', mode='overwrite')
```

### 3.2 Incremental Processing

**Principle:** Process only new/changed data, not entire dataset.

**Why it matters:**
- Reduces processing time and cost
- Enables faster pipeline runs
- Scales to large datasets

**Example:**

```python
def incremental_pipeline(execution_date):
    """Process only data for execution_date"""
    # Read only new data
    df = read_data(start_date=execution_date, end_date=execution_date + timedelta(days=1))

    # Process
    df = transform_data(df)

    # Write to date-partitioned storage
    df.to_parquet(f's3://ml-data/processed/date={execution_date}/data.parquet')
```

### 3.3 Data Partitioning

**Principle:** Organize data into logical partitions for efficient querying.

**Common partitioning strategies:**
- **Time-based**: `date=2023-10-15/` (most common for ML)
- **Categorical**: `country=US/`, `model_version=v1.2/`
- **Hash-based**: `user_id_hash=abc123/` (for distributed processing)

**Example:**

```python
# Write partitioned data
df.to_parquet(
    's3://ml-data/training/',
    partition_cols=['date', 'model_version'],  # Creates date=.../model_version=.../
    engine='pyarrow'
)

# Read specific partition (much faster than scanning all data)
df = pd.read_parquet('s3://ml-data/training/date=2023-10-15/model_version=v1.2/')
```

### 3.4 Schema Evolution

**Principle:** Handle schema changes gracefully without breaking pipelines.

**Strategies:**

```python
# Strategy 1: Default values for new columns
def read_data_with_defaults(path):
    df = pd.read_parquet(path)

    # Add new column if it doesn't exist
    if 'new_feature' not in df.columns:
        df['new_feature'] = None  # Or default value

    return df

# Strategy 2: Schema validation and versioning
from pandera import DataFrameSchema, Column, Check

schema_v1 = DataFrameSchema({
    "user_id": Column(int),
    "purchase_amount": Column(float, Check.greater_than(0)),
})

schema_v2 = schema_v1.add_columns({
    "loyalty_tier": Column(str, nullable=True)
})

def validate_with_schema_version(df, version='v2'):
    schemas = {'v1': schema_v1, 'v2': schema_v2}
    return schemas[version].validate(df)
```

### 3.5 Error Handling and Retries

**Principle:** Handle transient failures gracefully, fail fast on data issues.

```python
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

def process_with_error_handling(**context):
    """Process data with proper error handling"""
    try:
        # Attempt processing
        df = ingest_data()
        df = validate_data(df)
        df = transform_data(df)
        store_data(df)

    except ValidationError as e:
        # Data quality issue - don't retry, alert immediately
        send_alert(f"Data validation failed: {e}")
        raise  # Fail the task

    except ConnectionError as e:
        # Transient network issue - retry (Airflow will handle)
        raise

    except EmptyDataError as e:
        # Expected scenario - skip this run
        raise AirflowSkipException("No data available for this date")

task = PythonOperator(
    task_id='process_data',
    python_callable=process_with_error_handling,
    retries=3,  # Retry on failure
    retry_delay=timedelta(minutes=5),
    retry_exponential_backoff=True,  # 5min, 10min, 20min
)
```

## 4. Pipeline Architecture Patterns

### 4.1 ETL vs ELT

**ETL (Extract, Transform, Load):**
```
Extract → Transform → Load
```
- Transform data before loading to warehouse
- Good when: limited storage, need data cleaning before storage
- Tools: Airflow, Luigi

**ELT (Extract, Load, Transform):**
```
Extract → Load → Transform
```
- Load raw data, transform in warehouse
- Good when: cheap storage, powerful warehouse (Snowflake, BigQuery)
- Tools: dbt, Airflow + BigQuery

**Example (ETL):**
```python
# ETL: Transform in pipeline before loading
df = extract_from_api()
df = transform_data(df)  # Clean, normalize
df = engineer_features(df)  # Feature engineering
load_to_warehouse(df)  # Load clean data
```

**Example (ELT):**
```python
# ELT: Load raw data, transform in warehouse
df = extract_from_api()
load_to_data_lake(df)  # Load raw data to S3

# Transform with SQL in BigQuery
query = """
  CREATE OR REPLACE TABLE ml_features AS
  SELECT
    user_id,
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    AVG(purchase_amount) OVER (PARTITION BY user_id) as avg_purchase
  FROM raw_data.transactions
  WHERE date = CURRENT_DATE()
"""
run_bigquery_transform(query)
```

### 4.2 Medallion Architecture (Bronze/Silver/Gold)

**Layered data processing:**

```
┌───────────┐
│  Bronze   │  ← Raw data (as-is from source)
│ (Raw Lake)│
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Silver   │  ← Cleaned, validated data
│ (Curated) │
└─────┬─────┘
      │
      ▼
┌───────────┐
│   Gold    │  ← Business-level aggregates, ML features
│ (Refined) │
└───────────┘
```

**Implementation:**

```python
# Bronze layer: Ingest raw data
def bronze_layer(execution_date):
    raw_df = extract_from_source()
    raw_df.to_parquet(f's3://ml-data/bronze/date={execution_date}/')

# Silver layer: Clean and validate
def silver_layer(execution_date):
    raw_df = pd.read_parquet(f's3://ml-data/bronze/date={execution_date}/')
    clean_df = validate_data(raw_df)
    clean_df = transform_data(clean_df)
    clean_df.to_parquet(f's3://ml-data/silver/date={execution_date}/')

# Gold layer: ML features
def gold_layer(execution_date):
    clean_df = pd.read_parquet(f's3://ml-data/silver/date={execution_date}/')
    features_df = engineer_features(clean_df)
    features_df.to_parquet(f's3://ml-data/gold/date={execution_date}/')
```

**Benefits:**
- Clear separation of concerns
- Can replay transformations from bronze if needed
- Easy to debug (inspect each layer)

### 4.3 Microservices Pipeline Architecture

**Decouple pipeline stages as independent services:**

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Ingestion   │      │ Transformation│     │   Storage    │
│  Service     │─────▶│   Service     │────▶│   Service    │
│ (REST API)   │      │ (Kafka Stream)│     │  (S3 Writer) │
└──────────────┘      └──────────────┘      └──────────────┘
```

**Benefits:**
- Independent scaling
- Language flexibility (Python, Scala, Go)
- Fault isolation

**Example:**

```python
# Ingestion microservice (FastAPI)
from fastapi import FastAPI

app = FastAPI()

@app.post("/ingest")
async def ingest_data(data: dict):
    # Validate input
    validated = validate_schema(data)

    # Publish to Kafka
    producer.send('raw-data-topic', value=validated)

    return {"status": "ingested", "id": data['id']}

# Transformation microservice (Kafka Streams consumer)
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('raw-data-topic', ...)
producer = KafkaProducer('transformed-data-topic', ...)

for message in consumer:
    raw_data = message.value
    transformed = transform_data(raw_data)
    producer.send('transformed-data-topic', value=transformed)
```

## 5. Anti-Patterns to Avoid

### 5.1 Tightly Coupled Pipelines

**❌ BAD: Monolithic pipeline (all-or-nothing)**

```python
def monolithic_pipeline():
    """If any step fails, entire pipeline fails"""
    data1 = fetch_from_source_1()  # If this fails, everything fails
    data2 = fetch_from_source_2()
    data3 = fetch_from_source_3()

    combined = merge_all_data(data1, data2, data3)
    train_model(combined)
```

**✅ GOOD: Decoupled, independent tasks**

```python
# Each source is independent task
task_source_1 = ingest_source_1()
task_source_2 = ingest_source_2()
task_source_3 = ingest_source_3()

# Merge only if all succeeded
merge_task = merge_data([task_source_1, task_source_2, task_source_3])
```

### 5.2 No Data Versioning

**❌ BAD: Overwrite data without versioning**

```python
df.to_csv('training_data.csv')  # Lost previous version!
```

**✅ GOOD: Version data with timestamps/hashes**

```python
import hashlib

# Version by date
df.to_parquet(f's3://ml-data/training/date={today}/data.parquet')

# Version by content hash (for reproducibility)
data_hash = hashlib.md5(df.to_csv().encode()).hexdigest()
df.to_parquet(f's3://ml-data/training/hash={data_hash}/data.parquet')
```

### 5.3 Ignoring Data Quality

**❌ BAD: No validation, garbage in = garbage out**

```python
df = fetch_data()
train_model(df)  # Hope it's good data!
```

**✅ GOOD: Validate before processing**

```python
df = fetch_data()

# Check for nulls
if df.isnull().sum().sum() > len(df) * 0.1:  # > 10% nulls
    raise ValueError("Too many nulls in data")

# Check for schema drift
expected_columns = ['user_id', 'purchase_amount', 'timestamp']
if not set(expected_columns).issubset(df.columns):
    raise ValueError(f"Missing columns: {set(expected_columns) - set(df.columns)}")

# Validate ranges
assert df['purchase_amount'].min() >= 0, "Negative purchase amount detected"

train_model(df)
```

### 5.4 Not Monitoring Pipelines

**❌ BAD: No visibility into pipeline health**

**✅ GOOD: Monitor pipeline metrics**

```python
from prometheus_client import Counter, Histogram

# Metrics
rows_processed = Counter('pipeline_rows_processed_total', 'Rows processed')
processing_duration = Histogram('pipeline_duration_seconds', 'Pipeline duration')

@processing_duration.time()
def run_pipeline():
    df = fetch_data()
    df = transform_data(df)

    rows_processed.inc(len(df))

    store_data(df)

# Alert on metrics
# - Row count drop > 20% from historical average
# - Processing duration > 2x normal
# - Error rate > 1%
```

## 6. Summary

### Key Takeaways

✅ **ML data pipelines have 6 core stages:**
- Ingestion → Validation → Transformation → Feature Engineering → Storage → Training

✅ **Pipeline patterns:**
- **Batch**: High throughput, scheduled processing
- **Streaming**: Real-time, low latency
- **Lambda**: Combines batch + stream

✅ **Design principles:**
- **Idempotency**: Safe to re-run
- **Incremental processing**: Only new data
- **Partitioning**: Efficient querying
- **Error handling**: Retry transient, fail fast on data issues

✅ **Architecture patterns:**
- **ETL vs ELT**: Transform before vs after loading
- **Medallion (Bronze/Silver/Gold)**: Layered refinement
- **Microservices**: Decoupled, scalable services

✅ **Avoid anti-patterns:**
- Tightly coupled pipelines
- No data versioning
- No data quality checks
- No monitoring

### Real-World Impact

Companies with strong data pipeline practices:
- **Uber**: 100,000+ tables, 10PB+ data processed daily
- **Airbnb**: 1000+ Airflow DAGs, powers search and pricing
- **Spotify**: Near-real-time features for 400M+ users
- **Netflix**: Processes 500B+ events/day for recommendations

## Self-Check Questions

1. What are the 6 core stages of an ML data pipeline?
2. What's the difference between batch and streaming processing?
3. Why is idempotency important in data pipelines?
4. What is the Medallion architecture (Bronze/Silver/Gold)?
5. What's the difference between ETL and ELT?
6. How would you handle schema evolution in a production pipeline?
7. What metrics would you monitor for a data pipeline?
8. When would you use Lambda architecture?

## Additional Resources

- [Data Engineering Design Patterns](https://www.dedp.online/)
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [AWS Data Pipeline Patterns](https://aws.amazon.com/blogs/big-data/)

---

**Next lesson:** Apache Airflow Fundamentals - Learn how to implement these patterns with the industry-standard orchestration tool!
