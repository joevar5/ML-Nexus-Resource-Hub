# Lesson 02: Apache Airflow Fundamentals

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand Airflow architecture and core concepts
- Install and configure Airflow on Kubernetes
- Write and deploy DAGs (Directed Acyclic Graphs)
- Use Airflow operators for common tasks
- Schedule and trigger workflows
- Monitor DAG execution and debug failures
- Apply Airflow best practices for ML pipelines

## Prerequisites
- Completed Lesson 01 (Data Pipeline Architecture)
- Python programming skills
- Understanding of Kubernetes basics
- Familiarity with SQL and databases

## Introduction

**Apache Airflow** is the industry-standard workflow orchestration platform for data pipelines.

**Why Airflow matters for ML:**
- **Workflow orchestration**: Manage complex DAGs with dependencies
- **Scheduling**: Run pipelines on schedule or trigger-based
- **Monitoring**: Built-in UI for tracking pipeline execution
- **Scalability**: Distributed execution with Kubernetes
- **Extensibility**: 100+ built-in operators + custom operators

**Real-world usage:**
- **Airbnb**: Created Airflow, runs 1000+ DAGs for data pipelines
- **Uber**: 1800+ DAGs, processes 10PB+ data daily
- **Lyft**: Orchestrates ML training pipelines with Airflow
- **Twitter**: Uses Airflow for batch data processing
- **Spotify**: 5000+ DAGs for personalization and recommendations

## 1. Airflow Architecture

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────┐
│                  Airflow Architecture               │
│                                                     │
│  ┌──────────────┐         ┌──────────────┐        │
│  │   Web Server │◄────────┤   Scheduler  │        │
│  │   (Flask)    │         │  (DAG Parser)│        │
│  └──────┬───────┘         └──────┬───────┘        │
│         │                        │                │
│         │                        │ Creates tasks  │
│         ▼                        ▼                │
│  ┌──────────────────────────────────────┐        │
│  │         Metadata Database            │        │
│  │         (PostgreSQL/MySQL)           │        │
│  └──────────────┬───────────────────────┘        │
│                 │                                 │
│                 │ Reads task queue                │
│                 ▼                                 │
│  ┌──────────────────────────────────────┐        │
│  │           Executor                   │        │
│  │  (Kubernetes/Celery/Local)           │        │
│  └──────────────┬───────────────────────┘        │
│                 │                                 │
│                 │ Spawns workers                  │
│                 ▼                                 │
│  ┌──────────────────────────────────────┐        │
│  │          Workers                     │        │
│  │  (Execute tasks in pods/processes)   │        │
│  └──────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

**Component roles:**

1. **Web Server**: UI for monitoring, managing DAGs
2. **Scheduler**: Parses DAGs, schedules tasks, monitors execution
3. **Metadata Database**: Stores DAG definitions, task states, logs
4. **Executor**: Determines how tasks are executed (Kubernetes pods, Celery workers, etc.)
5. **Workers**: Execute the actual tasks

### 1.2 DAG (Directed Acyclic Graph)

**DAG structure:**

```
       ┌─────────────┐
       │  ingest_data│
       └──────┬──────┘
              │
              ▼
       ┌──────────────┐
       │validate_data │
       └──────┬───────┘
              │
         ┌────┴────┐
         ▼         ▼
   ┌─────────┐ ┌─────────┐
   │transform│ │ feature │
   │  _data  │ │   _eng  │
   └────┬────┘ └────┬────┘
        │           │
        └─────┬─────┘
              ▼
        ┌──────────┐
        │store_data│
        └──────────┘
```

**Key properties:**
- **Directed**: Tasks have a defined order (A → B → C)
- **Acyclic**: No loops (can't go A → B → A)
- **Graph**: Network of tasks with dependencies

## 2. Installing Airflow on Kubernetes

### 2.1 Install with Helm

**Add Airflow Helm repository:**

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
```

**Create values file for production:**

```yaml
# airflow-values.yaml
# Executor: KubernetesExecutor (scales automatically)
executor: "KubernetesExecutor"

# Airflow version
defaultAirflowTag: "2.7.2-python3.10"

# Database (PostgreSQL)
postgresql:
  enabled: true
  auth:
    username: airflow
    password: airflow
    database: airflow
  primary:
    persistence:
      size: 50Gi

# Redis (for CeleryExecutor, optional)
redis:
  enabled: false  # Not needed for KubernetesExecutor

# Webserver configuration
webserver:
  replicas: 2
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

# Scheduler configuration
scheduler:
  replicas: 2  # HA setup
  resources:
    limits:
      cpu: 2000m
      memory: 4Gi
    requests:
      cpu: 1000m
      memory: 2Gi

# DAGs configuration
dags:
  persistence:
    enabled: true
    size: 10Gi
    storageClassName: gp3-balanced

# Logs persistence
logs:
  persistence:
    enabled: true
    size: 50Gi

# Git-sync for DAG deployment (recommended)
gitSync:
  enabled: true
  repo: https://github.com/ai-infra-curriculum/airflow-dags.git
  branch: main
  rev: HEAD
  depth: 1
  maxFailures: 3
  subPath: "dags"
  wait: 60  # Sync every 60 seconds
  credentialsSecret: git-credentials

# Environment variables
env:
  - name: AIRFLOW__CORE__LOAD_EXAMPLES
    value: "False"
  - name: AIRFLOW__WEBSERVER__EXPOSE_CONFIG
    value: "True"
  - name: AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL
    value: "60"

# Service (LoadBalancer for external access)
service:
  type: LoadBalancer
```

**Install Airflow:**

```bash
# Create namespace
kubectl create namespace airflow

# Install with Helm
helm install airflow apache-airflow/airflow \
  --namespace airflow \
  --values airflow-values.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=airflow -n airflow --timeout=600s

# Get webserver URL
kubectl get service airflow-webserver -n airflow
```

**Access Airflow UI:**

```bash
# Port-forward to access locally
kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow

# Open browser: http://localhost:8080
# Default credentials: admin / admin
```

### 2.2 Configure Connections

**Airflow Connections** store credentials for external systems (databases, APIs, cloud storage).

**Add connections via UI:**
1. Navigate to Admin → Connections
2. Click "+" to add new connection

**Common connections for ML:**

```python
# AWS S3 connection
Conn Id: aws_default
Conn Type: Amazon Web Services
AWS Access Key ID: <your-key>
AWS Secret Access Key: <your-secret>
Extra: {"region_name": "us-west-2"}

# PostgreSQL connection
Conn Id: analytics_db
Conn Type: Postgres
Host: analytics.db.company.com
Schema: ml_data
Login: airflow
Password: <password>
Port: 5432

# Kubernetes connection (for KubernetesPodOperator)
Conn Id: kubernetes_default
Conn Type: Kubernetes
In Cluster Configuration: True  # If running in K8s
```

**Add connections via CLI:**

```bash
kubectl exec -it airflow-scheduler-0 -n airflow -- \
  airflow connections add 'aws_default' \
    --conn-type 'aws' \
    --conn-extra '{"region_name": "us-west-2", "aws_access_key_id": "xxx", "aws_secret_access_key": "xxx"}'
```

## 3. Writing Your First DAG

### 3.1 Basic DAG Structure

```python
# dags/my_first_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for all tasks
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,  # Don't wait for previous run to complete
    'email': ['ml-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),  # Kill task if runs too long
}

# Define the DAG
dag = DAG(
    'my_first_ml_pipeline',
    default_args=default_args,
    description='A simple ML data pipeline',
    schedule_interval='0 2 * * *',  # Cron: 2 AM daily
    start_date=datetime(2023, 1, 1),
    catchup=False,  # Don't backfill historical runs
    tags=['ml', 'training'],
)

# Task 1: Extract data
def extract_data(**context):
    """Extract data from source"""
    print(f"Extracting data for {context['ds']}")  # ds = execution date
    # Your extraction logic here
    return {'rows_extracted': 1000}

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

# Task 2: Validate data
def validate_data(**context):
    """Validate extracted data"""
    print("Validating data...")
    # Your validation logic
    return True

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Task 3: Transform data (using Bash)
transform_task = BashOperator(
    task_id='transform_data',
    bash_command='python /opt/scripts/transform.py --date {{ ds }}',
    dag=dag,
)

# Task 4: Train model
def train_model(**context):
    """Train ML model"""
    print("Training model...")
    # Your training logic
    return {'accuracy': 0.95}

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Define task dependencies
extract_task >> validate_task >> transform_task >> train_task
```

**Deploy DAG:**

```bash
# Copy DAG to Airflow DAGs folder
kubectl cp my_first_dag.py airflow-scheduler-0:/opt/airflow/dags/ -n airflow

# Or commit to Git (if using git-sync)
git add dags/my_first_dag.py
git commit -m "Add first ML pipeline DAG"
git push origin main

# DAG will appear in UI within ~60 seconds (git-sync interval)
```

### 3.2 DAG with XCom (Cross-Communication)

**XCom** allows tasks to share data.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG('xcom_example', start_date=datetime(2023, 1, 1), schedule_interval=None)

def extract_data(**context):
    """Extract data and push to XCom"""
    data = {'users': 1000, 'purchases': 5000}
    # Push to XCom
    context['task_instance'].xcom_push(key='data_stats', value=data)
    return data  # Also automatically pushed to XCom with key 'return_value'

def process_data(**context):
    """Pull data from XCom"""
    # Pull from XCom
    data_stats = context['task_instance'].xcom_pull(task_ids='extract_data', key='data_stats')
    print(f"Processing {data_stats['users']} users and {data_stats['purchases']} purchases")

extract_task = PythonOperator(task_id='extract_data', python_callable=extract_data, dag=dag)
process_task = PythonOperator(task_id='process_data', python_callable=process_data, dag=dag)

extract_task >> process_task
```

**⚠️ XCom limitations:**
- Stored in metadata database (limited size: ~48KB)
- Don't use for large datasets! Use S3/GCS instead

**Better approach for large data:**

```python
import boto3

def extract_data(**context):
    """Save large data to S3, pass S3 path via XCom"""
    df = fetch_large_dataset()  # 10GB dataset

    # Save to S3
    s3_path = f"s3://ml-data/processed/{context['ds']}/data.parquet"
    df.to_parquet(s3_path)

    # Push S3 path (small) to XCom
    return s3_path

def process_data(**context):
    """Load data from S3 using path from XCom"""
    s3_path = context['task_instance'].xcom_pull(task_ids='extract_data')
    df = pd.read_parquet(s3_path)
    # Process df...
```

## 4. Common Airflow Operators

### 4.1 PythonOperator

**Execute Python functions:**

```python
from airflow.operators.python import PythonOperator

def my_function(param1, param2, **context):
    print(f"Param1: {param1}, Param2: {param2}")
    print(f"Execution date: {context['ds']}")
    return "success"

task = PythonOperator(
    task_id='run_python_function',
    python_callable=my_function,
    op_kwargs={'param1': 'value1', 'param2': 'value2'},
    dag=dag,
)
```

### 4.2 BashOperator

**Execute bash commands:**

```python
from airflow.operators.bash import BashOperator

task = BashOperator(
    task_id='run_bash_script',
    bash_command='python /opt/scripts/train.py --date {{ ds }}',
    env={'MODEL_VERSION': 'v1.2'},  # Environment variables
    dag=dag,
)
```

### 4.3 KubernetesPodOperator

**Run tasks in Kubernetes pods (highly recommended for ML):**

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

train_task = KubernetesPodOperator(
    task_id='train_model_gpu',
    name='model-training-pod',
    namespace='ml-jobs',
    image='myregistry.io/ml-training:v1.0',
    cmds=["python", "train.py"],
    arguments=["--data-dir=/data", "--epochs=100"],

    # GPU resources
    resources={
        'request_memory': '16Gi',
        'request_cpu': '8',
        'limit_memory': '16Gi',
        'limit_cpu': '8',
        'limit_nvidia.com/gpu': '1',
    },

    # Volumes
    volumes=[
        k8s.V1Volume(
            name='training-data',
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name='ml-training-data-pvc'
            ),
        ),
    ],
    volume_mounts=[
        k8s.V1VolumeMount(
            name='training-data',
            mount_path='/data',
            read_only=True,
        ),
    ],

    # Auto-delete pod after completion
    is_delete_operator_pod=True,
    get_logs=True,

    dag=dag,
)
```

### 4.4 S3Operators

**Work with S3:**

```python
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator, S3DeleteObjectsOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

# Wait for file to appear in S3
wait_for_data = S3KeySensor(
    task_id='wait_for_new_data',
    bucket_name='ml-data',
    bucket_key='raw/{{ ds }}/data.parquet',
    aws_conn_id='aws_default',
    timeout=3600,  # Wait up to 1 hour
    poke_interval=60,  # Check every 60 seconds
    dag=dag,
)

# Copy object
copy_data = S3CopyObjectOperator(
    task_id='copy_data_to_processed',
    source_bucket_name='ml-data',
    source_bucket_key='raw/{{ ds }}/data.parquet',
    dest_bucket_name='ml-data',
    dest_bucket_key='processed/{{ ds }}/data.parquet',
    aws_conn_id='aws_default',
    dag=dag,
)

wait_for_data >> copy_data
```

### 4.5 SQLOperators

**Execute SQL queries:**

```python
from airflow.providers.postgres.operators.postgres import PostgresOperator

create_table_task = PostgresOperator(
    task_id='create_features_table',
    postgres_conn_id='analytics_db',
    sql="""
        CREATE TABLE IF NOT EXISTS ml_features (
            user_id BIGINT,
            feature_1 FLOAT,
            feature_2 FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """,
    dag=dag,
)

insert_data_task = PostgresOperator(
    task_id='insert_features',
    postgres_conn_id='analytics_db',
    sql="""
        INSERT INTO ml_features (user_id, feature_1, feature_2)
        SELECT user_id, AVG(purchase_amount), COUNT(*)
        FROM transactions
        WHERE date = '{{ ds }}'
        GROUP BY user_id;
    """,
    dag=dag,
)

create_table_task >> insert_data_task
```

## 5. Scheduling and Triggering

### 5.1 Schedule Intervals

**Cron expressions:**

```python
# Every day at 2 AM
schedule_interval='0 2 * * *'

# Every hour
schedule_interval='0 * * * *'

# Every Monday at 3 AM
schedule_interval='0 3 * * 1'

# First day of month at midnight
schedule_interval='0 0 1 * *'

# Or use Airflow presets
from airflow.timetables.datasets import DatasetOrTimeSchedule
schedule_interval='@daily'   # Midnight daily
schedule_interval='@hourly'  # Top of every hour
schedule_interval='@weekly'  # Sunday midnight
schedule_interval='@monthly' # First of month midnight

# Manual trigger only (no schedule)
schedule_interval=None
```

### 5.2 Backfilling

**Run historical DAG executions:**

```bash
# Backfill DAG for date range
airflow dags backfill \
  --start-date 2023-10-01 \
  --end-date 2023-10-15 \
  my_ml_pipeline

# Backfill specific task
airflow tasks run my_ml_pipeline extract_data 2023-10-15
```

### 5.3 Triggering DAGs

**Trigger from UI:**
- Click DAG → Click "Trigger DAG" button
- Can pass parameters via UI

**Trigger via CLI:**

```bash
kubectl exec -it airflow-scheduler-0 -n airflow -- \
  airflow dags trigger my_ml_pipeline --conf '{"model_version": "v2.0"}'
```

**Trigger via API:**

```python
import requests

response = requests.post(
    'http://airflow-webserver:8080/api/v1/dags/my_ml_pipeline/dagRuns',
    json={"conf": {"model_version": "v2.0"}},
    auth=('admin', 'admin')
)
```

**Trigger from another DAG:**

```python
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger_task = TriggerDagRunOperator(
    task_id='trigger_training_dag',
    trigger_dag_id='model_training_pipeline',
    conf={"model_version": "v2.0"},
    wait_for_completion=True,  # Wait for triggered DAG to complete
    dag=dag,
)
```

## 6. Monitoring and Debugging

### 6.1 Airflow UI

**Key views:**
1. **DAGs view**: List of all DAGs, status, last run
2. **Graph view**: Visual DAG structure with task states
3. **Tree view**: Historical runs timeline
4. **Gantt view**: Task duration and overlaps
5. **Task logs**: stdout/stderr from task execution

**Task states:**
- ⚪ **None**: Not yet scheduled
- 🟡 **Scheduled**: Queued for execution
- 🟢 **Running**: Currently executing
- ✅ **Success**: Completed successfully
- ❌ **Failed**: Task failed
- ⏸️ **Skipped**: Task was skipped
- ⏱️ **Up for retry**: Will retry after delay
- 🔄 **Upstream failed**: Parent task failed

### 6.2 Debugging Failed Tasks

**View logs:**

```bash
# Via UI: Click task → View Logs

# Via CLI:
kubectl exec -it airflow-scheduler-0 -n airflow -- \
  airflow tasks logs my_ml_pipeline extract_data 2023-10-15
```

**Common failure causes:**

1. **Import errors** (Python module not found)
   ```
   Solution: Add to requirements.txt, rebuild image
   ```

2. **Connection failures** (DB/API unreachable)
   ```
   Solution: Check connection in Admin → Connections
   ```

3. **Timeout** (task runs too long)
   ```
   Solution: Increase execution_timeout in task
   ```

4. **Out of memory**
   ```
   Solution: Increase resources in KubernetesPodOperator
   ```

### 6.3 Clearing and Retrying Tasks

**Clear task state (re-run):**

```bash
# Via UI: Click task → Clear

# Via CLI:
airflow tasks clear my_ml_pipeline extract_data --start-date 2023-10-15
```

**Mark task as success (skip re-run):**

```bash
airflow tasks state my_ml_pipeline extract_data 2023-10-15 --state success
```

## 7. Best Practices for ML Pipelines

### 7.1 Idempotent Tasks

**Ensure tasks can be re-run safely:**

```python
def process_data(**context):
    """Idempotent task - safe to re-run"""
    date = context['ds']

    # Overwrite output (not append)
    output_path = f's3://ml-data/processed/date={date}/data.parquet'
    df.to_parquet(output_path, mode='overwrite')  # ✅ Idempotent

    # Don't append to tables (creates duplicates if re-run)
    # df.to_sql('features', con=engine, if_exists='append')  # ❌ Not idempotent

    # Instead, use upsert or truncate-insert
    df.to_sql('features', con=engine, if_exists='replace')  # ✅ Idempotent
```

### 7.2 Use TaskFlow API (Airflow 2.0+)

**Cleaner DAG syntax:**

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule_interval='@daily', start_date=datetime(2023, 1, 1), catchup=False)
def ml_pipeline():
    @task
    def extract_data():
        return {'rows': 1000}

    @task
    def transform_data(data):
        rows = data['rows']
        return {'rows': rows, 'transformed': True}

    @task
    def load_data(data):
        print(f"Loading {data['rows']} rows")

    # Automatically creates dependencies
    data = extract_data()
    transformed = transform_data(data)
    load_data(transformed)

ml_pipeline_dag = ml_pipeline()
```

### 7.3 Parameterize DAGs

**Make DAGs configurable:**

```python
from airflow.models import Variable

# Store config in Airflow Variables (Admin → Variables)
MODEL_VERSION = Variable.get("model_version", default_var="v1.0")
BATCH_SIZE = Variable.get("batch_size", default_var=32, deserialize_json=False)

# Or pass via DAG config when triggering
def train_model(**context):
    config = context['dag_run'].conf or {}
    model_version = config.get('model_version', 'v1.0')
    print(f"Training model {model_version}")
```

### 7.4 Separate Concerns

**Don't put heavy logic in DAG files:**

```python
# ❌ BAD: All logic in DAG file
def train_model(**context):
    # 1000 lines of training logic here...
    df = pd.read_csv(...)
    model = RandomForest(...)
    model.fit(...)
    # ...

# ✅ GOOD: Import from separate module
from ml.training import train_model as train_fn

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_fn,
    dag=dag,
)
```

### 7.5 Use Sensors Wisely

**Sensors block a worker slot while waiting:**

```python
# ✅ GOOD: Use mode='reschedule' to free worker slot
sensor = S3KeySensor(
    task_id='wait_for_data',
    bucket_key='data.csv',
    mode='reschedule',  # Free worker slot while waiting
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,
    dag=dag,
)

# ❌ BAD: mode='poke' blocks worker
sensor = S3KeySensor(
    task_id='wait_for_data',
    bucket_key='data.csv',
    mode='poke',  # Blocks worker slot!
    dag=dag,
)
```

## 8. Summary

### Key Takeaways

✅ **Airflow is the standard orchestration tool for ML pipelines**
- Used by Airbnb, Uber, Lyft, Netflix, Spotify

✅ **Core concepts:**
- **DAG**: Directed Acyclic Graph of tasks
- **Operators**: Python, Bash, Kubernetes, SQL, etc.
- **XCom**: Share small data between tasks
- **Connections**: Store credentials

✅ **Best practices:**
- Make tasks idempotent
- Use KubernetesPodOperator for ML jobs
- Separate heavy logic from DAG files
- Use TaskFlow API for cleaner code
- Monitor with Airflow UI

✅ **Deployment:**
- Use Helm for Kubernetes installation
- GitSync for DAG deployment
- KubernetesExecutor for auto-scaling

## Self-Check Questions

1. What are the 5 core components of Airflow architecture?
2. What does DAG stand for? What are the key properties?
3. How do you share data between tasks in Airflow?
4. What's the difference between PythonOperator and KubernetesPodOperator?
5. How do you schedule a DAG to run daily at 3 AM?
6. What does `catchup=False` do in a DAG definition?
7. Why should tasks be idempotent?
8. When should you use `mode='reschedule'` for sensors?

## Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)
- [Airflow Summit Videos](https://airflowsummit.org/)

---

**Next lesson:** Advanced Airflow for ML - Complex DAG patterns, dynamic DAGs, and production optimization!
