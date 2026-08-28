# Lesson 08: Pipeline Monitoring and Error Handling

## Overview
Production data pipelines require robust monitoring and error handling to ensure reliability. This lesson covers observability, alerting, debugging, and recovery strategies for ML data pipelines.

**Duration:** 6-8 hours
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Python, Airflow, understanding of distributed systems

## Learning Objectives
By the end of this lesson, you will be able to:
- Implement comprehensive pipeline monitoring
- Set up effective alerting systems
- Debug pipeline failures efficiently
- Handle errors gracefully with retries and fallbacks
- Implement dead letter queues and circuit breakers
- Create runbooks for incident response

---

## 1. Pipeline Observability Fundamentals

### 1.1 Three Pillars of Observability

```
┌──────────────────────────────────────────────────────────┐
│              Pipeline Observability                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. METRICS       - Time-series numerical data           │
│     • Pipeline execution time                            │
│     • Records processed per second                       │
│     • Error rates                                        │
│     • Resource utilization                               │
│                                                           │
│  2. LOGS          - Structured event records             │
│     • Task start/completion events                       │
│     • Error messages and stack traces                    │
│     • Data validation results                            │
│     • Debug information                                  │
│                                                           │
│  3. TRACES        - Request flow through system          │
│     • End-to-end pipeline execution                      │
│     • Task dependencies and timing                       │
│     • Bottleneck identification                          │
│     • Cross-service communication                        │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Key Metrics for ML Pipelines

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class PipelineMetrics:
    """Core metrics for pipeline monitoring"""

    # Performance metrics
    execution_time_seconds: float
    records_processed: int
    records_per_second: float
    task_duration_seconds: dict

    # Quality metrics
    validation_success_rate: float
    data_drift_score: float
    null_percentage: float

    # Reliability metrics
    error_count: int
    retry_count: int
    success: bool

    # Resource metrics
    memory_usage_mb: float
    cpu_utilization_percent: float

    # Metadata
    pipeline_name: str
    execution_date: datetime
    version: str

    def to_dict(self):
        """Convert to dictionary for export"""
        return {
            'pipeline_name': self.pipeline_name,
            'execution_date': self.execution_date.isoformat(),
            'version': self.version,
            'success': self.success,
            'execution_time_seconds': self.execution_time_seconds,
            'records_processed': self.records_processed,
            'records_per_second': self.records_per_second,
            'validation_success_rate': self.validation_success_rate,
            'error_count': self.error_count,
            'memory_usage_mb': self.memory_usage_mb
        }
```

---

## 2. Instrumentation and Metrics Collection

### 2.1 Prometheus Metrics

```python
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, CollectorRegistry
)
import time
from functools import wraps

# Create metrics registry
registry = CollectorRegistry()

# Define metrics
pipeline_executions_total = Counter(
    'pipeline_executions_total',
    'Total pipeline executions',
    ['pipeline_name', 'status'],
    registry=registry
)

pipeline_duration_seconds = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_name'],
    buckets=[10, 30, 60, 300, 600, 1800, 3600],
    registry=registry
)

records_processed_total = Counter(
    'records_processed_total',
    'Total records processed',
    ['pipeline_name', 'stage'],
    registry=registry
)

pipeline_errors_total = Counter(
    'pipeline_errors_total',
    'Total pipeline errors',
    ['pipeline_name', 'error_type'],
    registry=registry
)

active_pipelines = Gauge(
    'active_pipelines',
    'Number of currently running pipelines',
    ['pipeline_name'],
    registry=registry
)

data_quality_score = Gauge(
    'data_quality_score',
    'Current data quality score',
    ['pipeline_name'],
    registry=registry
)

class MetricsCollector:
    """Collect and export metrics for ML pipelines"""

    def __init__(self, pipeline_name: str, port: int = 8000):
        self.pipeline_name = pipeline_name
        self.port = port

    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        start_http_server(self.port, registry=registry)
        print(f"Metrics server started on port {self.port}")

    def track_execution(self, func):
        """Decorator to track pipeline execution metrics"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Increment active pipelines
            active_pipelines.labels(pipeline_name=self.pipeline_name).inc()

            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                status = 'failure'
                pipeline_errors_total.labels(
                    pipeline_name=self.pipeline_name,
                    error_type=type(e).__name__
                ).inc()
                raise

            finally:
                # Record duration
                duration = time.time() - start_time
                pipeline_duration_seconds.labels(
                    pipeline_name=self.pipeline_name
                ).observe(duration)

                # Record execution
                pipeline_executions_total.labels(
                    pipeline_name=self.pipeline_name,
                    status=status
                ).inc()

                # Decrement active pipelines
                active_pipelines.labels(pipeline_name=self.pipeline_name).dec()

        return wrapper

    def record_records_processed(self, stage: str, count: int):
        """Record number of records processed"""
        records_processed_total.labels(
            pipeline_name=self.pipeline_name,
            stage=stage
        ).inc(count)

    def update_quality_score(self, score: float):
        """Update data quality score"""
        data_quality_score.labels(
            pipeline_name=self.pipeline_name
        ).set(score)

# Usage
metrics = MetricsCollector('ml_training_pipeline')
metrics.start_metrics_server()

@metrics.track_execution
def run_training_pipeline():
    """ML training pipeline with metrics"""
    # Load data
    df = load_data()
    metrics.record_records_processed('load', len(df))

    # Validate
    validation_score = validate_data(df)
    metrics.update_quality_score(validation_score)

    # Transform
    features = transform_features(df)
    metrics.record_records_processed('transform', len(features))

    # Train
    model = train_model(features)

    return model

# Run pipeline
model = run_training_pipeline()
```

### 2.2 Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict
import traceback

class StructuredLogger:
    """Structured logger for ML pipelines"""

    def __init__(self, pipeline_name: str, log_file: str = None):
        self.pipeline_name = pipeline_name
        self.logger = logging.getLogger(pipeline_name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._json_formatter())
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._json_formatter())
            self.logger.addHandler(file_handler)

    def _json_formatter(self):
        """Create JSON formatter"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'logger': record.name,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                # Add extra fields
                if hasattr(record, 'extra'):
                    log_data.update(record.extra)

                # Add exception info
                if record.exc_info:
                    log_data['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }

                return json.dumps(log_data)

        return JSONFormatter()

    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log error message with exception"""
        extra = {'extra': kwargs} if kwargs else {}
        self.logger.error(message, exc_info=exception, extra=extra)

    def log_pipeline_start(self, run_id: str, config: Dict[str, Any]):
        """Log pipeline start"""
        self.info(
            "Pipeline started",
            run_id=run_id,
            pipeline_name=self.pipeline_name,
            config=config
        )

    def log_pipeline_end(self, run_id: str, duration: float, status: str, metrics: Dict):
        """Log pipeline completion"""
        self.info(
            "Pipeline completed",
            run_id=run_id,
            pipeline_name=self.pipeline_name,
            duration_seconds=duration,
            status=status,
            metrics=metrics
        )

    def log_task_execution(self, task_name: str, duration: float, records: int):
        """Log task execution"""
        self.info(
            f"Task '{task_name}' completed",
            task_name=task_name,
            duration_seconds=duration,
            records_processed=records
        )

# Usage
logger = StructuredLogger('ml_training_pipeline', 'pipeline.log')

run_id = 'run_20250115_120000'
logger.log_pipeline_start(run_id, {'model_type': 'xgboost', 'features': 100})

try:
    # Execute pipeline tasks
    start_time = time.time()
    result = process_data()
    logger.log_task_execution('process_data', time.time() - start_time, len(result))

    logger.log_pipeline_end(
        run_id,
        duration=time.time() - start_time,
        status='success',
        metrics={'records': len(result), 'quality': 0.95}
    )

except Exception as e:
    logger.error("Pipeline failed", exception=e, run_id=run_id)
    raise
```

---

## 3. Error Handling Strategies

### 3.1 Retry Logic

```python
import time
from typing import Callable, Type, Tuple
from functools import wraps

class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

def retry_with_backoff(
    config: RetryConfig,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: StructuredLogger = None
):
    """Retry decorator with exponential backoff"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0

            while attempt < config.max_attempts:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    attempt += 1

                    if attempt >= config.max_attempts:
                        if logger:
                            logger.error(
                                f"Max retries ({config.max_attempts}) exceeded",
                                exception=e,
                                function=func.__name__
                            )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )

                    # Add jitter
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random())

                    if logger:
                        logger.warning(
                            f"Retry attempt {attempt}/{config.max_attempts}",
                            function=func.__name__,
                            delay_seconds=delay,
                            error=str(e)
                        )

                    time.sleep(delay)

        return wrapper
    return decorator

# Usage
retry_config = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    exponential_base=2.0,
    jitter=True
)

@retry_with_backoff(
    config=retry_config,
    exceptions=(ConnectionError, TimeoutError),
    logger=logger
)
def fetch_data_from_api(url: str):
    """Fetch data with automatic retries"""
    import requests
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

# Will retry up to 5 times with exponential backoff
data = fetch_data_from_api('https://api.example.com/data')
```

### 3.2 Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta
import threading

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for external dependencies"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if datetime.now() - self.opened_at > timedelta(seconds=self.timeout_duration):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info("Circuit breaker recovered to CLOSED state")

            return result

        except Exception as e:
            with self._lock:
                self.failure_count += 1

                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
                    self.opened_at = datetime.now()
                    logger.warning("Circuit breaker re-opened from HALF_OPEN")

                elif self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.opened_at = datetime.now()
                    logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")

            raise

# Usage
api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout_duration=60,
    success_threshold=2
)

def call_external_api():
    """Call external API with circuit breaker protection"""
    return api_circuit_breaker.call(
        fetch_data_from_api,
        'https://api.example.com/data'
    )

# Will open circuit after 5 failures
# Will reject requests for 60 seconds
# Will test recovery after timeout
try:
    data = call_external_api()
except Exception as e:
    logger.error("API call failed", exception=e)
```

### 3.3 Dead Letter Queue

```python
from kafka import KafkaProducer, KafkaConsumer
import json
from typing import Any, Dict
from datetime import datetime

class DeadLetterQueue:
    """Dead letter queue for failed messages"""

    def __init__(
        self,
        bootstrap_servers: list,
        dlq_topic: str = 'pipeline-dlq'
    ):
        self.dlq_topic = dlq_topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_to_dlq(
        self,
        message: Dict[str, Any],
        error: Exception,
        source_topic: str,
        metadata: Dict[str, Any] = None
    ):
        """Send failed message to DLQ"""
        dlq_message = {
            'original_message': message,
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            },
            'source_topic': source_topic,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }

        self.producer.send(self.dlq_topic, value=dlq_message)
        self.producer.flush()

        logger.warning(
            f"Message sent to DLQ",
            source_topic=source_topic,
            error_type=type(error).__name__
        )

class ResilientConsumer:
    """Kafka consumer with DLQ support"""

    def __init__(
        self,
        input_topic: str,
        dlq: DeadLetterQueue,
        max_retries: int = 3
    ):
        self.input_topic = input_topic
        self.dlq = dlq
        self.max_retries = max_retries

        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=False
        )

    def process_message(self, message: Dict) -> bool:
        """Process message (override in subclass)"""
        raise NotImplementedError

    def run(self):
        """Run consumer with DLQ support"""
        for kafka_message in self.consumer:
            message = kafka_message.value
            retry_count = 0
            processed = False

            while retry_count < self.max_retries and not processed:
                try:
                    self.process_message(message)
                    self.consumer.commit()
                    processed = True

                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Processing failed (attempt {retry_count}/{self.max_retries})",
                        error=str(e)
                    )

                    if retry_count >= self.max_retries:
                        # Send to DLQ
                        self.dlq.send_to_dlq(
                            message=message,
                            error=e,
                            source_topic=self.input_topic,
                            metadata={'retry_count': retry_count}
                        )
                        self.consumer.commit()
                    else:
                        time.sleep(2 ** retry_count)  # Exponential backoff

# Usage
dlq = DeadLetterQueue(['localhost:9092'])
consumer = ResilientConsumer('user-events', dlq, max_retries=3)
consumer.run()
```

---

## 4. Airflow Monitoring

### 4.1 Airflow Callbacks

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

def send_alert(context):
    """Send alert on task failure"""
    task_instance = context['task_instance']
    exception = context.get('exception')

    alert_message = f"""
    Task Failed: {task_instance.task_id}
    DAG: {task_instance.dag_id}
    Execution Date: {context['execution_date']}
    Log URL: {task_instance.log_url}
    Error: {exception}
    """

    # Send to Slack/PagerDuty/etc
    logger.error(
        "Task failure alert",
        dag_id=task_instance.dag_id,
        task_id=task_instance.task_id,
        error=str(exception)
    )

def task_success_callback(context):
    """Callback on task success"""
    task_instance = context['task_instance']
    duration = (task_instance.end_date - task_instance.start_date).total_seconds()

    logger.info(
        "Task succeeded",
        task_id=task_instance.task_id,
        duration_seconds=duration
    )

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_alert,
    'on_success_callback': task_success_callback
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training']
)
```

### 4.2 Custom Airflow Sensors

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import requests

class DataQualitySensor(BaseSensorOperator):
    """Sensor to check data quality before processing"""

    @apply_defaults
    def __init__(
        self,
        data_source: str,
        quality_threshold: float = 0.95,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_source = data_source
        self.quality_threshold = quality_threshold

    def poke(self, context):
        """Check if data quality meets threshold"""
        # Load and validate data
        df = pd.read_csv(self.data_source)
        validator = DataValidator()

        # Run checks
        validator.check_not_null(df, df.columns.tolist(), threshold=0.05)
        report = validator.get_report()

        quality_score = report['success_rate']
        self.log.info(f"Data quality score: {quality_score:.2%}")

        if quality_score >= self.quality_threshold:
            return True
        else:
            self.log.warning(
                f"Data quality below threshold: {quality_score:.2%} < {self.quality_threshold:.2%}"
            )
            return False

# Usage in DAG
quality_check = DataQualitySensor(
    task_id='check_data_quality',
    data_source='/data/training_data.csv',
    quality_threshold=0.95,
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Timeout after 1 hour
    dag=dag
)
```

---

## 5. Alerting and Incident Response

### 5.1 Alert Manager Integration

```python
import requests
from typing import List, Dict

class AlertManager:
    """Send alerts to various channels"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def send_slack_alert(
        self,
        message: str,
        channel: str,
        severity: str = 'warning'
    ):
        """Send alert to Slack"""
        webhook_url = self.config['slack_webhook']

        color_map = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'critical': '#ff0000'
        }

        payload = {
            'channel': channel,
            'attachments': [{
                'color': color_map.get(severity, '#808080'),
                'title': f'Pipeline Alert [{severity.upper()}]',
                'text': message,
                'ts': int(time.time())
            }]
        }

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

    def send_pagerduty_alert(
        self,
        summary: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Send alert to PagerDuty"""
        api_key = self.config['pagerduty_api_key']
        routing_key = self.config['pagerduty_routing_key']

        payload = {
            'routing_key': routing_key,
            'event_action': 'trigger',
            'payload': {
                'summary': summary,
                'severity': severity,
                'source': 'ml-pipeline',
                'custom_details': details
            }
        }

        headers = {
            'Authorization': f'Token token={api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json=payload,
            headers=headers
        )
        response.raise_for_status()

    def check_and_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        comparison: str = 'greater_than'
    ):
        """Check metric and send alert if threshold exceeded"""
        should_alert = False

        if comparison == 'greater_than':
            should_alert = current_value > threshold
        elif comparison == 'less_than':
            should_alert = current_value < threshold

        if should_alert:
            message = f"""
            Alert: {metric_name} threshold exceeded
            Current value: {current_value}
            Threshold: {threshold}
            Comparison: {comparison}
            """

            self.send_slack_alert(
                message=message,
                channel='#ml-alerts',
                severity='warning'
            )

# Usage
alert_manager = AlertManager({
    'slack_webhook': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
    'pagerduty_api_key': 'your-api-key',
    'pagerduty_routing_key': 'your-routing-key'
})

# Check pipeline metrics
pipeline_duration = 3600  # seconds
alert_manager.check_and_alert(
    metric_name='pipeline_duration',
    current_value=pipeline_duration,
    threshold=1800,
    comparison='greater_than'
)
```

### 5.2 Runbook Template

```markdown
# Pipeline Incident Runbook

## Incident: Training Pipeline Failure

### Symptoms
- Airflow DAG shows failed status
- No new model artifacts in S3
- Slack alert: "Training pipeline failed"

### Investigation Steps

1. **Check Airflow UI**
   - Navigate to DAG: `ml_training_pipeline`
   - Identify failed task
   - Review task logs

2. **Check Pipeline Logs**
   ```bash
   kubectl logs -n ml-pipelines <pod-name>
   grep ERROR pipeline.log | tail -50
   ```

3. **Check Data Quality**
   ```python
   df = pd.read_csv('/data/training_data.csv')
   print(df.info())
   print(df.describe())
   ```

4. **Check Resource Usage**
   ```bash
   kubectl top pods -n ml-pipelines
   df -h  # Check disk space
   ```

### Common Issues & Fixes

#### Issue: Out of Memory
**Symptoms:** OOMKilled in pod logs
**Fix:**
```bash
# Increase memory limit
kubectl edit deployment training-pipeline
# Change: memory: "8Gi" -> "16Gi"
```

#### Issue: Data Quality Failure
**Symptoms:** ValidationError in logs
**Fix:**
1. Check data source freshness
2. Validate against schema
3. Run data quality report
4. Contact data team if persistent

#### Issue: External API Timeout
**Symptoms:** ConnectionTimeout errors
**Fix:**
1. Check API status page
2. Verify network connectivity
3. Review circuit breaker state
4. Use backup data source if available

### Escalation
- Level 1: ML Engineer (check logs, retry)
- Level 2: Senior ML Engineer (debug code, fix data)
- Level 3: ML Platform Lead (infrastructure issues)

### Communication
- Post updates in #ml-incidents Slack channel
- Update status page
- Notify stakeholders if SLA at risk
```

---

## 6. Debugging Techniques

### 6.1 Pipeline Debugging Utilities

```python
import pandas as pd
from typing import Callable

class PipelineDebugger:
    """Utilities for debugging data pipelines"""

    @staticmethod
    def checkpoint_data(
        df: pd.DataFrame,
        checkpoint_name: str,
        save_path: str = '/tmp/checkpoints'
    ):
        """Save intermediate data for debugging"""
        import os
        os.makedirs(save_path, exist_ok=True)

        filepath = f"{save_path}/{checkpoint_name}.parquet"
        df.to_parquet(filepath)

        logger.info(
            f"Checkpoint saved: {checkpoint_name}",
            rows=len(df),
            columns=df.columns.tolist(),
            filepath=filepath
        )

    @staticmethod
    def profile_transformation(func: Callable):
        """Profile execution time and memory usage"""
        import memory_profiler

        @wraps(func)
        def wrapper(df):
            import psutil
            process = psutil.Process()

            # Before
            start_time = time.time()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Execute
            result = func(df)

            # After
            duration = time.time() - start_time
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_delta = mem_after - mem_before

            logger.info(
                f"Transformation profiling: {func.__name__}",
                duration_seconds=duration,
                input_rows=len(df),
                output_rows=len(result),
                memory_delta_mb=mem_delta
            )

            return result

        return wrapper

    @staticmethod
    def compare_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str = 'df1',
        name2: str = 'df2'
    ):
        """Compare two dataframes for debugging"""
        print(f"\n=== DataFrame Comparison: {name1} vs {name2} ===\n")

        # Shape
        print(f"Shape: {df1.shape} vs {df2.shape}")

        # Columns
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        print(f"\nColumns only in {name1}: {cols1 - cols2}")
        print(f"Columns only in {name2}: {cols2 - cols1}")

        # Common columns stats
        common_cols = cols1 & cols2
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df1[col]):
                print(f"\n{col}:")
                print(f"  {name1} mean: {df1[col].mean():.2f}")
                print(f"  {name2} mean: {df2[col].mean():.2f}")

# Usage
debugger = PipelineDebugger()

# Save checkpoints
debugger.checkpoint_data(raw_df, 'raw_data')
debugger.checkpoint_data(processed_df, 'processed_data')

# Profile transformation
@debugger.profile_transformation
def expensive_transformation(df):
    # Complex transformation
    return df.groupby('user_id').agg({'spend': 'sum'})

result = expensive_transformation(df)

# Compare versions
debugger.compare_dataframes(old_data, new_data, 'old', 'new')
```

---

## 7. Best Practices

### 7.1 Monitoring Checklist

✅ **DO:**
- Monitor pipeline end-to-end latency
- Track data quality metrics over time
- Set up alerts for critical failures
- Log structured data for analysis
- Use distributed tracing for complex pipelines
- Monitor resource utilization
- Track business metrics (e.g., model accuracy)

❌ **DON'T:**
- Monitor too many metrics (alert fatigue)
- Set overly sensitive thresholds
- Ignore warning signs
- Log sensitive data (PII, passwords)
- Skip testing of alerting logic

### 7.2 Error Handling Principles

```python
# GOOD: Specific exception handling
try:
    data = fetch_from_api()
except requests.Timeout as e:
    logger.error("API timeout", exception=e)
    # Use cached data or fail gracefully
    data = load_cached_data()
except requests.HTTPError as e:
    if e.response.status_code == 429:
        # Rate limited - back off
        time.sleep(60)
        data = fetch_from_api()
    else:
        raise

# BAD: Catching all exceptions
try:
    data = fetch_from_api()
except Exception:
    pass  # Silent failure
```

---

## 8. Summary

Key takeaways:
- ✅ Observability is critical for production pipelines
- ✅ Use metrics, logs, and traces comprehensively
- ✅ Implement retries, circuit breakers, and DLQs
- ✅ Set up effective alerting and runbooks
- ✅ Debug systematically with checkpoints and profiling
- ✅ Test error handling logic thoroughly

**Next Module:** [Module 06 - MLOps & Experiment Tracking](../mod-06-mlops/README.md)
