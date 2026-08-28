# Lesson 03: Advanced Airflow for ML

## Learning Objectives
By the end of this lesson, you will be able to:
- Implement complex DAG patterns for ML workflows
- Create dynamic DAGs that generate tasks programmatically
- Use TaskFlow API effectively for data passing
- Optimize Airflow performance for large-scale ML pipelines
- Implement branching and conditional logic in DAGs
- Build distributed training pipelines with Airflow
- Apply production-ready patterns and anti-patterns

## Prerequisites
- Completed Lesson 02 (Airflow Fundamentals)
- Python programming (decorators, comprehensions)
- Understanding of ML training workflows
- Familiarity with Kubernetes

## Introduction

This lesson covers advanced Airflow patterns specifically for ML infrastructure:
- **Dynamic DAGs**: Generate tasks for multiple models/experiments
- **Complex dependencies**: Fan-out/fan-in, conditional branching
- **Distributed training**: Orchestrate multi-GPU/multi-node training
- **Performance optimization**: Handle 100+ concurrent DAGs

**Real-world applications:**
- **Uber**: Dynamic DAGs for 100+ ML models trained daily
- **Lyft**: Branching DAGs for A/B testing model variants
- **Netflix**: Fan-out pattern for personalization across 190+ countries

## 1. Dynamic DAGs

### 1.1 Why Dynamic DAGs?

**Problem**: Need to train 50 models with similar pipelines

**❌ Bad solution**: Copy-paste 50 DAG files

**✅ Good solution**: Generate DAGs programmatically

### 1.2 Dynamic DAG with Loop

```python
# dags/dynamic_model_training.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# List of models to train
MODELS = ['bert', 'gpt2', 'roberta', 'albert', 'xlnet']

def train_model(model_name, **context):
    """Train a specific model"""
    print(f"Training {model_name} for {context['ds']}")
    # Training logic here
    return f"{model_name}_trained"

# Generate one DAG per model
for model in MODELS:
    dag_id = f'train_{model}_daily'

    dag = DAG(
        dag_id,
        schedule_interval='@daily',
        start_date=datetime(2023, 1, 1),
        catchup=False,
        tags=['ml', 'training', model],
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={'model_name': model},
        dag=dag,
    )

    # Register DAG in globals() so Airflow can find it
    globals()[dag_id] = dag

# Result: 5 DAGs created (train_bert_daily, train_gpt2_daily, etc.)
```

### 1.3 Dynamic DAG from Config File

**More maintainable: define models in YAML**

```yaml
# config/models.yaml
models:
  - name: sentiment_classifier
    architecture: bert
    dataset: imdb
    epochs: 10
    gpu_count: 2

  - name: ner_model
    architecture: roberta
    dataset: conll2003
    epochs: 20
    gpu_count: 4

  - name: qa_model
    architecture: albert
    dataset: squad
    epochs: 15
    gpu_count: 2
```

```python
# dags/dynamic_from_config.py
import yaml
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime
from kubernetes.client import models as k8s

# Load config
with open('/opt/airflow/config/models.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_training_dag(model_config):
    """Create DAG for a model"""
    model_name = model_config['name']

    dag = DAG(
        f'train_{model_name}',
        schedule_interval='@daily',
        start_date=datetime(2023, 1, 1),
        catchup=False,
        tags=['ml', 'dynamic', model_config['architecture']],
    )

    train_task = KubernetesPodOperator(
        task_id='train_model',
        name=f'{model_name}-training',
        namespace='ml-jobs',
        image=f'myregistry.io/{model_config["architecture"]}:latest',
        cmds=["python", "train.py"],
        arguments=[
            "--model-name", model_name,
            "--dataset", model_config['dataset'],
            "--epochs", str(model_config['epochs']),
        ],
        resources={
            'limit_nvidia.com/gpu': str(model_config['gpu_count']),
            'limit_memory': '32Gi',
            'limit_cpu': '16',
        },
        dag=dag,
    )

    return dag

# Generate DAGs from config
for model in config['models']:
    dag_id = f"train_{model['name']}"
    globals()[dag_id] = create_training_dag(model)

# Result: 3 DAGs created from config
```

### 1.4 Dynamic Tasks within a DAG

**Fan-out pattern: process multiple items in parallel**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG('dynamic_tasks', start_date=datetime(2023, 1, 1), schedule_interval=None)

# Generate tasks dynamically
COUNTRIES = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP']

def process_country(country, **context):
    print(f"Processing data for {country}")
    # Country-specific logic

# Create one task per country (fan-out)
country_tasks = []
for country in COUNTRIES:
    task = PythonOperator(
        task_id=f'process_{country.lower()}',
        python_callable=process_country,
        op_kwargs={'country': country},
        dag=dag,
    )
    country_tasks.append(task)

# Aggregate results (fan-in)
def aggregate_results(**context):
    print("Aggregating results from all countries")

aggregate_task = PythonOperator(
    task_id='aggregate_results',
    python_callable=aggregate_results,
    dag=dag,
)

# Dependencies: all country tasks → aggregate
for task in country_tasks:
    task >> aggregate_task
```

## 2. Complex DAG Patterns

### 2.1 Branching (Conditional Execution)

**Execute different tasks based on conditions:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

dag = DAG('branching_example', start_date=datetime(2023, 1, 1), schedule_interval=None)

def check_data_quality(**context):
    """Decide which branch to take based on data quality"""
    # Check data quality
    data_quality_score = 0.95  # Example: compute actual score

    if data_quality_score >= 0.9:
        return 'high_quality_pipeline'  # Task ID to execute
    elif data_quality_score >= 0.7:
        return 'medium_quality_pipeline'
    else:
        return 'low_quality_pipeline'

branch_task = BranchPythonOperator(
    task_id='check_quality',
    python_callable=check_data_quality,
    dag=dag,
)

# Branch 1: High quality data
high_quality_task = PythonOperator(
    task_id='high_quality_pipeline',
    python_callable=lambda: print("Running high-quality pipeline"),
    dag=dag,
)

# Branch 2: Medium quality data
medium_quality_task = PythonOperator(
    task_id='medium_quality_pipeline',
    python_callable=lambda: print("Running medium-quality pipeline with extra validation"),
    dag=dag,
)

# Branch 3: Low quality data
low_quality_task = PythonOperator(
    task_id='low_quality_pipeline',
    python_callable=lambda: print("Alerting team about low quality data"),
    dag=dag,
)

# Join point (runs after any branch)
join_task = EmptyOperator(
    task_id='join',
    trigger_rule='none_failed_min_one_success',  # Run if at least one branch succeeded
    dag=dag,
)

# Dependencies
branch_task >> [high_quality_task, medium_quality_task, low_quality_task]
[high_quality_task, medium_quality_task, low_quality_task] >> join_task
```

### 2.2 SubDAGs (Deprecated, use TaskGroups instead)

**TaskGroups: Organize tasks visually**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime

dag = DAG('taskgroup_example', start_date=datetime(2023, 1, 1), schedule_interval=None)

# TaskGroup for data preprocessing
with TaskGroup('preprocessing', tooltip='Data preprocessing tasks', dag=dag) as preprocessing_group:
    extract_task = PythonOperator(
        task_id='extract',
        python_callable=lambda: print("Extracting data"),
    )

    validate_task = PythonOperator(
        task_id='validate',
        python_callable=lambda: print("Validating data"),
    )

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=lambda: print("Transforming data"),
    )

    extract_task >> validate_task >> transform_task

# TaskGroup for model training
with TaskGroup('training', tooltip='Model training tasks', dag=dag) as training_group:
    train_task = PythonOperator(
        task_id='train',
        python_callable=lambda: print("Training model"),
    )

    evaluate_task = PythonOperator(
        task_id='evaluate',
        python_callable=lambda: print("Evaluating model"),
    )

    train_task >> evaluate_task

# Dependencies between groups
preprocessing_group >> training_group
```

### 2.3 Trigger Rules

**Control when a task runs based on upstream task states:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG('trigger_rules', start_date=datetime(2023, 1, 1), schedule_interval=None)

task_a = PythonOperator(task_id='task_a', python_callable=lambda: print("A"), dag=dag)
task_b = PythonOperator(task_id='task_b', python_callable=lambda: print("B"), dag=dag)
task_c = PythonOperator(task_id='task_c', python_callable=lambda: print("C"), dag=dag)

# Task D runs only if ALL upstream tasks succeed (default)
task_d = PythonOperator(
    task_id='task_d_all_success',
    python_callable=lambda: print("D: All succeeded"),
    trigger_rule='all_success',  # Default
    dag=dag,
)

# Task E runs if AT LEAST ONE upstream task succeeds
task_e = PythonOperator(
    task_id='task_e_one_success',
    python_callable=lambda: print("E: At least one succeeded"),
    trigger_rule='one_success',
    dag=dag,
)

# Task F runs if ALL upstream tasks are done (regardless of success/failure)
task_f = PythonOperator(
    task_id='task_f_all_done',
    python_callable=lambda: print("F: All done"),
    trigger_rule='all_done',
    dag=dag,
)

# Task G runs if AT LEAST ONE upstream task failed
task_g = PythonOperator(
    task_id='task_g_one_failed',
    python_callable=lambda: print("G: Cleanup after failure"),
    trigger_rule='one_failed',
    dag=dag,
)

[task_a, task_b, task_c] >> task_d
[task_a, task_b, task_c] >> task_e
[task_a, task_b, task_c] >> task_f
[task_a, task_b, task_c] >> task_g
```

**Common trigger rules:**
- `all_success` (default): All upstream tasks succeeded
- `all_failed`: All upstream tasks failed
- `all_done`: All upstream tasks completed (success or failed)
- `one_success`: At least one upstream task succeeded
- `one_failed`: At least one upstream task failed
- `none_failed`: No upstream tasks failed (some may have been skipped)
- `none_failed_min_one_success`: No failures and at least one success
- `none_skipped`: No upstream tasks were skipped
- `always`: Always run (ignore upstream states)

## 3. Distributed Training Orchestration

### 3.1 Multi-GPU Training on Single Node

```python
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime
from kubernetes.client import models as k8s

dag = DAG('distributed_training_single_node', start_date=datetime(2023, 1, 1), schedule_interval='@daily')

train_task = KubernetesPodOperator(
    task_id='train_multi_gpu',
    name='multi-gpu-training',
    namespace='ml-jobs',
    image='pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime',
    cmds=["torchrun"],
    arguments=[
        "--nproc_per_node=4",  # 4 GPUs on this node
        "train_distributed.py",
        "--data-dir=/data",
        "--epochs=100",
    ],

    # Request 4 GPUs
    resources={
        'limit_nvidia.com/gpu': '4',
        'limit_memory': '64Gi',
        'limit_cpu': '32',
        'request_nvidia.com/gpu': '4',
        'request_memory': '64Gi',
        'request_cpu': '32',
    },

    # Mount data volume
    volumes=[
        k8s.V1Volume(
            name='training-data',
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name='imagenet-pvc'
            ),
        ),
    ],
    volume_mounts=[
        k8s.V1VolumeMount(name='training-data', mount_path='/data', read_only=True),
    ],

    dag=dag,
)
```

### 3.2 Multi-Node Distributed Training

**Use PyTorchJob (Kubeflow Training Operator):**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from kubernetes import client, config
import yaml

dag = DAG('distributed_training_multi_node', start_date=datetime(2023, 1, 1), schedule_interval='@daily')

def submit_pytorch_job(**context):
    """Submit PyTorchJob for distributed training"""
    config.load_incluster_config()  # Load K8s config from pod

    pytorch_job_manifest = {
        'apiVersion': 'kubeflow.org/v1',
        'kind': 'PyTorchJob',
        'metadata': {
            'name': f"resnet-training-{context['ds'].replace('-', '')}",
            'namespace': 'ml-jobs',
        },
        'spec': {
            'pytorchReplicaSpecs': {
                # Master replica
                'Master': {
                    'replicas': 1,
                    'restartPolicy': 'OnFailure',
                    'template': {
                        'spec': {
                            'containers': [{
                                'name': 'pytorch',
                                'image': 'pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime',
                                'command': ['python', 'train_distributed.py'],
                                'args': ['--backend=nccl', '--epochs=100'],
                                'resources': {
                                    'limits': {'nvidia.com/gpu': '2'},
                                },
                            }],
                        },
                    },
                },
                # Worker replicas
                'Worker': {
                    'replicas': 3,  # 3 worker nodes
                    'restartPolicy': 'OnFailure',
                    'template': {
                        'spec': {
                            'containers': [{
                                'name': 'pytorch',
                                'image': 'pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime',
                                'command': ['python', 'train_distributed.py'],
                                'args': ['--backend=nccl', '--epochs=100'],
                                'resources': {
                                    'limits': {'nvidia.com/gpu': '2'},
                                },
                            }],
                        },
                    },
                },
            },
        },
    }

    # Create PyTorchJob
    api = client.CustomObjectsApi()
    api.create_namespaced_custom_object(
        group='kubeflow.org',
        version='v1',
        namespace='ml-jobs',
        plural='pytorchjobs',
        body=pytorch_job_manifest,
    )

    print(f"Submitted PyTorchJob: {pytorch_job_manifest['metadata']['name']}")

submit_task = PythonOperator(
    task_id='submit_distributed_training',
    python_callable=submit_pytorch_job,
    dag=dag,
)
```

### 3.3 Hyperparameter Tuning with Parallel Training

**Train multiple model variants in parallel:**

```python
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import itertools

dag = DAG('hyperparameter_tuning', start_date=datetime(2023, 1, 1), schedule_interval=None)

# Define hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
optimizers = ['adam', 'sgd']

# Generate all combinations
hyperparam_combinations = list(itertools.product(learning_rates, batch_sizes, optimizers))

training_tasks = []

for lr, batch_size, optimizer in hyperparam_combinations:
    task_id = f'train_lr{lr}_bs{batch_size}_{optimizer}'.replace('.', '_')

    train_task = KubernetesPodOperator(
        task_id=task_id,
        name=f'hp-tuning-{task_id}',
        namespace='ml-jobs',
        image='myregistry.io/model-training:v1.0',
        cmds=["python", "train.py"],
        arguments=[
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--optimizer", optimizer,
            "--output-dir", f"s3://ml-experiments/{{ ds }}/{task_id}/",
        ],
        resources={'limit_nvidia.com/gpu': '1'},
        dag=dag,
    )

    training_tasks.append(train_task)

# Aggregate results and select best model
def select_best_model(**context):
    """Compare all experiment results and select best model"""
    print("Selecting best model from hyperparameter tuning...")
    # Logic to compare metrics from S3 and select winner

select_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    dag=dag,
)

# All training tasks run in parallel, then select best
training_tasks >> select_task

# Result: 18 parallel training tasks (3 LRs × 3 batch sizes × 2 optimizers)
```

## 4. TaskFlow API Advanced Patterns

### 4.1 Multiple Outputs

```python
from airflow.decorators import dag, task
from datetime import datetime
from typing import Dict

@dag(schedule_interval='@daily', start_date=datetime(2023, 1, 1), catchup=False)
def ml_pipeline_taskflow():

    @task(multiple_outputs=True)
    def extract_and_validate() -> Dict[str, any]:
        """Extract data and return multiple outputs"""
        return {
            'data_path': 's3://ml-data/processed/2023-10-15/data.parquet',
            'row_count': 1000000,
            'quality_score': 0.95,
        }

    @task
    def train_model(data_path: str, row_count: int):
        print(f"Training on {data_path} with {row_count} rows")
        return {'model_path': 's3://models/model.pth', 'accuracy': 0.92}

    @task
    def deploy_model(model_path: str, accuracy: float):
        if accuracy >= 0.9:
            print(f"Deploying model from {model_path}")
        else:
            print("Model accuracy too low, not deploying")

    # Unpack multiple outputs
    extraction = extract_and_validate()
    training = train_model(
        data_path=extraction['data_path'],
        row_count=extraction['row_count']
    )
    deploy_model(model_path=training['model_path'], accuracy=training['accuracy'])

ml_pipeline_taskflow_dag = ml_pipeline_taskflow()
```

### 4.2 Task Mapping (Airflow 2.3+)

**Dynamic task mapping for parallel processing:**

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule_interval='@daily', start_date=datetime(2023, 1, 1), catchup=False)
def dynamic_task_mapping():

    @task
    def get_countries():
        """Return list of countries to process"""
        return ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN', 'BR', 'CN']

    @task
    def process_country(country: str):
        """Process data for a single country"""
        print(f"Processing {country}")
        return {'country': country, 'rows': 10000}

    @task
    def aggregate_results(results: list):
        """Aggregate results from all countries"""
        total_rows = sum(r['rows'] for r in results)
        print(f"Processed {total_rows} total rows across {len(results)} countries")

    # Dynamic mapping: creates one process_country task per country
    countries = get_countries()
    results = process_country.expand(country=countries)  # Magic happens here!
    aggregate_results(results)

dynamic_task_mapping_dag = dynamic_task_mapping()
```

## 5. Performance Optimization

### 5.1 Reduce DAG Parsing Time

**Problem**: DAGs with heavy imports slow down scheduler

**❌ Bad:**
```python
import pandas as pd  # Heavy import
import tensorflow as tf  # Very heavy!

dag = DAG('slow_dag', ...)
```

**✅ Good:**
```python
dag = DAG('fast_dag', ...)

def train_model(**context):
    import tensorflow as tf  # Import only when task runs
    # Training logic
```

### 5.2 Pool Management

**Limit concurrent tasks to avoid resource exhaustion:**

```bash
# Create pool via UI: Admin → Pools → Create
# Or via CLI:
airflow pools set ml_training_pool 10 "Pool for ML training tasks (max 10 concurrent)"
```

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG('pooled_tasks', start_date=datetime(2023, 1, 1))

# Tasks will use pool (max 10 concurrent across all DAGs)
for i in range(50):
    task = PythonOperator(
        task_id=f'train_model_{i}',
        python_callable=lambda: print("Training model"),
        pool='ml_training_pool',  # Use pool
        pool_slots=1,  # Takes 1 slot
        dag=dag,
    )
```

### 5.3 Smart Sensors (Airflow 2.2+)

**Reduce scheduler load from sensors:**

```python
from airflow.sensors.s3 import S3KeySensor

# ✅ Smart sensor: offloads to separate sensor worker
sensor = S3KeySensor(
    task_id='wait_for_data',
    bucket_key='data/{{ ds }}/file.parquet',
    aws_conn_id='aws_default',
    timeout=3600,
    poke_interval=60,
    mode='reschedule',  # Important for smart sensors
    deferrable=True,  # Enable smart sensor (Airflow 2.2+)
    dag=dag,
)
```

### 5.4 Parallelism Configuration

**Tune Airflow for high throughput:**

```yaml
# airflow.cfg or Helm values
core:
  parallelism: 128  # Max tasks across all DAGs
  dag_concurrency: 32  # Max tasks per DAG
  max_active_runs_per_dag: 4  # Max concurrent DAG runs

scheduler:
  max_dagruns_per_loop_to_schedule: 10
  scheduler_heartbeat_sec: 5

kubernetes_executor:
  worker_pods_creation_batch_size: 8  # Create 8 pods at a time
```

## 6. Production Anti-Patterns

### 6.1 ❌ Storing Large Data in XCom

```python
# ❌ BAD: XCom for large datasets
def extract_data():
    df = pd.read_csv('large_file.csv')  # 10 GB
    return df  # Stored in XCom (database) - WILL FAIL!

# ✅ GOOD: Store in S3, pass path via XCom
def extract_data():
    df = pd.read_csv('large_file.csv')
    s3_path = 's3://bucket/data.parquet'
    df.to_parquet(s3_path)
    return s3_path  # Small string, safe for XCom
```

### 6.2 ❌ Not Setting execution_timeout

```python
# ❌ BAD: Task runs forever if stuck
task = PythonOperator(
    task_id='train',
    python_callable=train_model,
)

# ✅ GOOD: Kill if exceeds timeout
task = PythonOperator(
    task_id='train',
    python_callable=train_model,
    execution_timeout=timedelta(hours=6),  # Kill after 6 hours
)
```

### 6.3 ❌ Too Many Small Tasks

```python
# ❌ BAD: 1000 tasks for 1000 rows (scheduler overhead!)
for row in range(1000):
    task = PythonOperator(task_id=f'process_row_{row}', ...)

# ✅ GOOD: Batch into fewer tasks
for batch in range(10):  # 10 tasks processing 100 rows each
    task = PythonOperator(
        task_id=f'process_batch_{batch}',
        op_kwargs={'start_row': batch*100, 'end_row': (batch+1)*100},
        ...
    )
```

### 6.4 ❌ Tight Coupling Between DAGs

```python
# ❌ BAD: DAG B depends on specific task in DAG A
# If DAG A changes structure, DAG B breaks

# ✅ GOOD: Use TriggerDagRunOperator or Datasets (Airflow 2.4+)
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

trigger = TriggerDagRunOperator(
    task_id='trigger_model_deployment',
    trigger_dag_id='deploy_model',
    conf={'model_version': '{{ ti.xcom_pull("train_model") }}'},
)
```

## 7. Summary

### Key Takeaways

✅ **Dynamic DAGs**:
- Generate DAGs from config files (YAML/JSON)
- Use loops to create multiple similar DAGs
- Dynamic tasks for fan-out/fan-in patterns

✅ **Advanced patterns**:
- Branching for conditional execution
- TaskGroups for organization
- Trigger rules for complex dependencies

✅ **Distributed training**:
- KubernetesPodOperator for multi-GPU
- PyTorchJob for multi-node training
- Parallel hyperparameter tuning

✅ **TaskFlow API**:
- Cleaner syntax with decorators
- Multiple outputs and task mapping
- Better type safety

✅ **Performance**:
- Lazy imports in DAG files
- Pools to limit concurrency
- Smart sensors to reduce load
- Tune parallelism settings

✅ **Avoid anti-patterns**:
- Don't use XCom for large data
- Always set execution_timeout
- Batch small tasks
- Decouple DAGs

## Self-Check Questions

1. How do you create multiple DAGs from a config file?
2. What's the difference between fan-out and fan-in patterns?
3. When would you use BranchPythonOperator?
4. What trigger rule runs a task if at least one upstream task succeeds?
5. How do you limit concurrent training tasks to 10?
6. Why should you avoid storing large datasets in XCom?
7. How do you implement hyperparameter tuning with Airflow?
8. What's the benefit of deferrable sensors?

## Additional Resources

- [Dynamic DAGs](https://www.astronomer.io/guides/dynamically-generating-dags/)
- [TaskFlow API Guide](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [Airflow Performance Tuning](https://www.astronomer.io/guides/airflow-scaling-workers/)

---

**Next lesson:** Data Versioning with DVC - Track datasets like code!
