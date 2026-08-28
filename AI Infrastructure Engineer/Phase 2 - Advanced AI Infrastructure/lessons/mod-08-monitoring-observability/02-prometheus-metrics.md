# Lesson 02: Prometheus for Metrics Collection

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand Prometheus architecture and data model
- Install and configure Prometheus for AI infrastructure
- Write PromQL queries to analyze metrics
- Instrument Python applications with Prometheus client libraries
- Configure service discovery and scraping
- Design effective metric collection strategies for ML systems

## Prerequisites
- Completion of Lesson 01 (Introduction to Observability)
- Basic understanding of HTTP and time-series data
- Familiarity with YAML configuration
- Python programming experience

## Introduction

Prometheus is an open-source monitoring and alerting system originally built at SoundCloud. It has become the de facto standard for metrics collection in cloud-native environments, particularly in Kubernetes ecosystems. For AI infrastructure engineers, Prometheus is essential for monitoring ML workloads, GPU utilization, model serving latency, and infrastructure health.

### Why Prometheus for AI Infrastructure?

1. **Pull-based architecture**: Prometheus scrapes metrics from targets, reducing complexity in dynamic ML environments
2. **Multi-dimensional data model**: Labels allow slicing metrics by model, version, GPU, node, etc.
3. **Powerful query language (PromQL)**: Analyze and aggregate metrics across dimensions
4. **Integration with Kubernetes**: Automatic service discovery for ML workloads
5. **Alerting capabilities**: Proactive detection of performance degradation
6. **Ecosystem integration**: Works seamlessly with Grafana, AlertManager, and exporters

---

## 1. Prometheus Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Prometheus Server                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Retrieval  │  │   Storage    │  │   HTTP Server   │  │
│  │   (Scraper)  │─>│   (TSDB)     │<─│   (PromQL API)  │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│         │                                      │            │
└─────────│──────────────────────────────────────│────────────┘
          │                                      │
          │ scrape                               │ query
          ↓                                      ↓
   ┌──────────────┐                      ┌──────────────┐
   │   Targets    │                      │   Grafana    │
   │              │                      │  AlertManager│
   │ • ML Models  │                      │   Clients    │
   │ • GPUs       │                      └──────────────┘
   │ • Services   │
   └──────────────┘
```

**Components:**

1. **Prometheus Server**: Scrapes and stores time-series data
2. **Time-Series Database (TSDB)**: Efficient storage for metrics
3. **Retrieval System**: Discovers and scrapes targets
4. **PromQL Engine**: Query language processor
5. **Alertmanager**: Handles alerts from Prometheus rules
6. **Exporters**: Expose metrics from third-party systems
7. **Pushgateway**: For ephemeral jobs (batch ML training)

### Data Model

Prometheus stores all data as **time series** - streams of timestamped values belonging to the same metric and set of labeled dimensions.

**Metric Structure:**
```
<metric_name>{<label_name>=<label_value>, ...} value timestamp
```

**Example:**
```promql
gpu_utilization{gpu_id="0", node="ml-node-1", model="bert-base"} 87.5 1634567890
```

**Metric Types:**

1. **Counter**: Monotonically increasing value (e.g., requests processed)
2. **Gauge**: Value that can go up or down (e.g., GPU memory usage)
3. **Histogram**: Observations in configurable buckets (e.g., request latency)
4. **Summary**: Similar to histogram, with client-side quantiles

---

## 2. Installing Prometheus

### Docker Installation

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus_data:

networks:
  monitoring:
    driver: bridge
```

### Kubernetes Installation with Helm

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack (includes Grafana, AlertManager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
```

### Basic Configuration

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s      # How often to scrape targets
  evaluation_interval: 15s  # How often to evaluate rules
  external_labels:
    cluster: 'ml-production'
    environment: 'prod'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'alertmanager:9093'

# Load rules once and periodically evaluate them
rule_files:
  - "alerts/*.yml"
  - "recording_rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # GPU metrics exporter
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-node-1:9835', 'gpu-node-2:9835']
        labels:
          node_type: 'gpu'

  # ML model serving
  - job_name: 'ml-models'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ml-serving
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

---

## 3. Prometheus Query Language (PromQL)

PromQL is a powerful functional query language for selecting and aggregating time-series data.

### Basic Queries

**Instant Vector Queries:**
```promql
# Current GPU utilization
gpu_utilization

# GPU utilization for specific GPU
gpu_utilization{gpu_id="0"}

# GPU utilization > 80%
gpu_utilization > 80

# Multiple label selectors
model_inference_latency{model="bert", version="v2", env="prod"}
```

**Range Vector Queries:**
```promql
# GPU utilization over last 5 minutes
gpu_utilization[5m]

# HTTP request rate over last hour
http_requests_total[1h]
```

### Operators and Functions

**Mathematical Operators:**
```promql
# GPU memory usage percentage
(gpu_memory_used / gpu_memory_total) * 100

# Request rate per second
rate(http_requests_total[5m])

# Average inference latency increase
delta(model_inference_latency_sum[1h]) / delta(model_inference_latency_count[1h])
```

**Aggregation Functions:**
```promql
# Average GPU utilization across all GPUs
avg(gpu_utilization)

# Maximum GPU temperature by node
max(gpu_temperature) by (node)

# Total requests per model
sum(rate(model_requests_total[5m])) by (model_name)

# 95th percentile inference latency
histogram_quantile(0.95, rate(model_inference_duration_bucket[5m]))
```

**Common Functions:**

| Function | Purpose | Example |
|----------|---------|---------|
| `rate()` | Per-second rate over time window | `rate(requests_total[5m])` |
| `irate()` | Instant rate (last 2 points) | `irate(requests_total[1m])` |
| `increase()` | Total increase over time window | `increase(errors_total[1h])` |
| `avg_over_time()` | Average over time window | `avg_over_time(cpu_usage[10m])` |
| `predict_linear()` | Linear prediction | `predict_linear(disk_usage[1h], 3600)` |

### AI Infrastructure Query Examples

**GPU Monitoring:**
```promql
# GPU utilization by model
avg(gpu_utilization) by (model_name, gpu_id)

# GPU memory saturation
(avg(gpu_memory_used) by (node) / avg(gpu_memory_total) by (node)) * 100

# GPUs running hot
gpu_temperature > 80
```

**Model Serving:**
```promql
# Request rate per model
sum(rate(model_requests_total[5m])) by (model_name, version)

# P99 latency per endpoint
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (endpoint, le)
)

# Error rate percentage
(sum(rate(http_requests_total{status=~"5.."}[5m])) /
 sum(rate(http_requests_total[5m]))) * 100

# Model prediction throughput
sum(rate(model_predictions_total[1m])) by (model_name)
```

**Training Jobs:**
```promql
# Training loss over time
avg(training_loss) by (job_name, epoch)

# Training GPU efficiency
rate(training_samples_processed[5m]) * on(node) group_left gpu_utilization

# Batch processing rate
rate(batch_processed_total[5m])
```

---

## 4. Instrumenting Python Applications

### Installing Prometheus Client

```bash
pip install prometheus-client==0.17.1
```

### Basic Instrumentation

**Example: ML Model Serving with FastAPI**

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client import multiprocess, CollectorRegistry
import time
import logging

# Create FastAPI app
app = FastAPI(title="ML Model Serving")

# Define metrics
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total number of inference requests',
    ['model_name', 'version', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_inference_duration_seconds',
    'Time spent processing inference request',
    ['model_name', 'version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'Current GPU utilization percentage',
    ['gpu_id', 'model_name']
)

GPU_MEMORY = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory currently in use',
    ['gpu_id']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is currently loaded (1) or not (0)',
    ['model_name', 'version']
)

BATCH_SIZE = Histogram(
    'inference_batch_size',
    'Batch size for inference requests',
    ['model_name'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# Inference endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    model_name = request.model_name
    version = request.version

    # Track request
    start_time = time.time()

    try:
        # Perform inference
        predictions = await model_inference(request.data, model_name, version)

        # Record metrics
        REQUEST_COUNT.labels(
            model_name=model_name,
            version=version,
            status="success"
        ).inc()

        BATCH_SIZE.labels(model_name=model_name).observe(len(request.data))

        # Update GPU metrics (example)
        update_gpu_metrics(model_name)

        return {"predictions": predictions}

    except Exception as e:
        REQUEST_COUNT.labels(
            model_name=model_name,
            version=version,
            status="error"
        ).inc()
        raise

    finally:
        # Record latency
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(
            model_name=model_name,
            version=version
        ).observe(duration)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

def update_gpu_metrics(model_name: str):
    """Update GPU utilization and memory metrics"""
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_UTILIZATION.labels(
                gpu_id=str(i),
                model_name=model_name
            ).set(util.gpu)

            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_MEMORY.labels(gpu_id=str(i)).set(mem_info.used)

    except Exception as e:
        logging.error(f"Failed to update GPU metrics: {e}")
```

### Advanced Instrumentation Patterns

**1. Decorators for Automatic Instrumentation:**

```python
from functools import wraps
from prometheus_client import Counter, Histogram
import time

# Metrics
FUNCTION_CALLS = Counter(
    'function_calls_total',
    'Total function calls',
    ['function_name', 'status']
)

FUNCTION_DURATION = Histogram(
    'function_duration_seconds',
    'Function execution time',
    ['function_name']
)

def monitor_performance(func):
    """Decorator to automatically monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__

        try:
            result = func(*args, **kwargs)
            FUNCTION_CALLS.labels(
                function_name=function_name,
                status='success'
            ).inc()
            return result

        except Exception as e:
            FUNCTION_CALLS.labels(
                function_name=function_name,
                status='error'
            ).inc()
            raise

        finally:
            duration = time.time() - start_time
            FUNCTION_DURATION.labels(
                function_name=function_name
            ).observe(duration)

    return wrapper

# Usage
@monitor_performance
def preprocess_data(data):
    # Data preprocessing logic
    pass

@monitor_performance
def run_inference(model, inputs):
    return model.predict(inputs)
```

**2. Context Managers for Resource Tracking:**

```python
from contextlib import contextmanager
from prometheus_client import Gauge
import psutil

MEMORY_USAGE = Gauge(
    'process_memory_bytes',
    'Process memory usage',
    ['phase']
)

CPU_USAGE = Gauge(
    'process_cpu_percent',
    'Process CPU usage',
    ['phase']
)

@contextmanager
def track_resources(phase_name: str):
    """Context manager to track resource usage during a phase"""
    process = psutil.Process()

    # Capture initial state
    initial_memory = process.memory_info().rss

    try:
        yield
    finally:
        # Capture final state
        final_memory = process.memory_info().rss
        cpu_percent = process.cpu_percent()

        MEMORY_USAGE.labels(phase=phase_name).set(final_memory)
        CPU_USAGE.labels(phase=phase_name).set(cpu_percent)

# Usage
with track_resources("data_loading"):
    data = load_large_dataset()

with track_resources("model_training"):
    model.fit(data)
```

**3. Custom Collectors for Complex Metrics:**

```python
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
import pynvml

class GPUCollector(Collector):
    """Custom collector for detailed GPU metrics"""

    def __init__(self):
        pynvml.nvmlInit()

    def collect(self):
        # GPU utilization metric family
        gpu_util = GaugeMetricFamily(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            labels=['gpu_id', 'gpu_name', 'uuid']
        )

        # GPU memory metric family
        gpu_mem = GaugeMetricFamily(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            labels=['gpu_id', 'gpu_name', 'uuid', 'memory_type']
        )

        # GPU temperature metric family
        gpu_temp = GaugeMetricFamily(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            labels=['gpu_id', 'gpu_name', 'uuid']
        )

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Device info
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util.add_metric(
                [str(i), name, uuid],
                util.gpu
            )

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem.add_metric(
                [str(i), name, uuid, 'used'],
                mem_info.used
            )
            gpu_mem.add_metric(
                [str(i), name, uuid, 'total'],
                mem_info.total
            )

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            gpu_temp.add_metric(
                [str(i), name, uuid],
                temp
            )

        yield gpu_util
        yield gpu_mem
        yield gpu_temp

# Register custom collector
from prometheus_client import REGISTRY
REGISTRY.register(GPUCollector())
```

---

## 5. Exporters and Service Discovery

### Common Exporters for AI Infrastructure

**1. Node Exporter (System Metrics):**
```yaml
# docker-compose.yml
services:
  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
```

**2. NVIDIA GPU Exporter:**
```yaml
services:
  nvidia-gpu-exporter:
    image: nvidia/dcgm-exporter:latest
    container_name: gpu-exporter
    ports:
      - "9400:9400"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

**3. Kubernetes State Metrics:**
```bash
helm install kube-state-metrics prometheus-community/kube-state-metrics \
  --namespace monitoring
```

### Kubernetes Service Discovery

Prometheus can automatically discover targets in Kubernetes:

```yaml
scrape_configs:
  # Discover pods with prometheus.io/scrape annotation
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod

    relabel_configs:
      # Only scrape pods with annotation prometheus.io/scrape=true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

      # Use custom metrics path if specified
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

      # Use custom port if specified
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

      # Add pod labels as metric labels
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)

      # Add namespace label
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace

      # Add pod name label
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
```

**Kubernetes Deployment with Annotations:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bert-model-serving
  namespace: ml-serving
spec:
  replicas: 3
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
      labels:
        app: bert-serving
        model: bert-base
        version: v1.2
    spec:
      containers:
        - name: model-server
          image: ml-registry/bert-serving:v1.2
          ports:
            - containerPort: 8000
              name: http
```

---

## 6. Recording Rules and Aggregation

Recording rules allow you to precompute expensive queries and store results as new time series.

**recording_rules.yml:**
```yaml
groups:
  - name: ml_infrastructure_rules
    interval: 30s
    rules:
      # GPU utilization aggregations
      - record: node:gpu_utilization:avg
        expr: avg(gpu_utilization) by (node)

      - record: model:gpu_utilization:avg
        expr: avg(gpu_utilization) by (model_name)

      # Request rate aggregations
      - record: model:requests:rate5m
        expr: sum(rate(model_requests_total[5m])) by (model_name, version)

      - record: model:errors:rate5m
        expr: sum(rate(model_requests_total{status="error"}[5m])) by (model_name)

      # Latency percentiles
      - record: model:latency:p50
        expr: |
          histogram_quantile(0.50,
            sum(rate(model_inference_duration_bucket[5m])) by (model_name, le)
          )

      - record: model:latency:p95
        expr: |
          histogram_quantile(0.95,
            sum(rate(model_inference_duration_bucket[5m])) by (model_name, le)
          )

      - record: model:latency:p99
        expr: |
          histogram_quantile(0.99,
            sum(rate(model_inference_duration_bucket[5m])) by (model_name, le)
          )

      # GPU memory saturation
      - record: node:gpu_memory_saturation:ratio
        expr: |
          sum(gpu_memory_used_bytes) by (node) /
          sum(gpu_memory_total_bytes) by (node)

      # Batch processing efficiency
      - record: job:batch_processing_rate:5m
        expr: rate(batch_processed_total[5m])

      # Model prediction throughput
      - record: model:predictions_per_second:5m
        expr: sum(rate(model_predictions_total[5m])) by (model_name, version)
```

---

## 7. Best Practices

### Metric Naming Conventions

Follow Prometheus conventions:
- Use `snake_case`
- Include unit suffix: `_bytes`, `_seconds`, `_total`, `_ratio`
- Format: `<namespace>_<name>_<unit>`

**Good Examples:**
```
model_inference_duration_seconds
gpu_memory_used_bytes
http_requests_total
cache_hit_ratio
```

**Bad Examples:**
```
ModelLatency  (not snake_case)
memory  (no unit)
requests  (should be requests_total for counter)
```

### Label Design

**Do:**
- Use labels for dimensions you want to aggregate/filter by
- Keep cardinality reasonable (< 1000 unique combinations per metric)
- Use consistent label names across metrics

**Don't:**
- Use high-cardinality labels (user IDs, timestamps, UUIDs)
- Use labels for values that change frequently
- Create too many label dimensions

**Example:**
```python
# Good: Reasonable cardinality
REQUEST_COUNT.labels(
    model_name="bert",      # ~10 models
    version="v1.2",         # ~5 versions per model
    status="success"        # 2-3 statuses
)  # Total: ~150 combinations

# Bad: High cardinality
REQUEST_COUNT.labels(
    user_id="abc123",       # Millions of users
    request_id="xyz789"     # Unique per request
)  # Total: Billions of combinations!
```

### Performance Optimization

1. **Scrape Interval:** Balance freshness vs. load
   ```yaml
   scrape_interval: 15s  # Default
   scrape_interval: 30s  # For stable metrics
   scrape_interval: 5s   # For critical, fast-changing metrics
   ```

2. **Retention:** Configure based on storage and needs
   ```yaml
   --storage.tsdb.retention.time=30d
   --storage.tsdb.retention.size=50GB
   ```

3. **Recording Rules:** Precompute expensive queries
   ```yaml
   - record: expensive:aggregation:5m
     expr: sum(rate(metric[5m])) by (dimension)
   ```

4. **Metric Relabeling:** Drop unnecessary metrics/labels
   ```yaml
   metric_relabel_configs:
     - source_labels: [__name__]
       regex: 'go_.*'  # Drop Go runtime metrics
       action: drop
   ```

### Security

1. **Authentication:** Use basic auth or OAuth proxy
2. **TLS:** Enable HTTPS for metrics endpoints
3. **Network Policies:** Restrict scrape access in Kubernetes
4. **Sensitive Data:** Never expose secrets in metrics/labels

---

## 8. Hands-On Example: Complete ML Serving Monitoring

**complete_monitoring_example.py:**

```python
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, REGISTRY, CollectorRegistry
)
from pydantic import BaseModel
import time
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="ML Model Serving with Monitoring")

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Request metrics
HTTP_REQUESTS = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Model inference metrics
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_name', 'model_version']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

BATCH_SIZE = Histogram(
    'model_batch_size',
    'Batch size for predictions',
    ['model_name'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256]
)

PREDICTION_ERRORS = Counter(
    'model_prediction_errors_total',
    'Total prediction errors',
    ['model_name', 'error_type']
)

# Model state metrics
MODEL_LOADED = Gauge(
    'model_loaded_info',
    'Model load status (1=loaded, 0=not loaded)',
    ['model_name', 'model_version', 'model_type']
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load model',
    ['model_name', 'model_version']
)

# Resource metrics
ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of requests currently being processed',
    ['model_name']
)

INPUT_SIZE = Summary(
    'model_input_size_bytes',
    'Size of model input data',
    ['model_name']
)

OUTPUT_SIZE = Summary(
    'model_output_size_bytes',
    'Size of model output data',
    ['model_name']
)

# ============================================================================
# REQUEST MODELS
# ============================================================================

class PredictRequest(BaseModel):
    model_name: str = "bert-base"
    model_version: str = "v1.0"
    inputs: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]
    latency_ms: float
    model_version: str

# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def monitor_requests(request, call_next):
    """Middleware to monitor all HTTP requests"""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path

    # Process request
    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    status = response.status_code

    HTTP_REQUESTS.labels(
        method=method,
        endpoint=endpoint,
        status=status
    ).inc()

    REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

    return response

# ============================================================================
# MOCK MODEL (Replace with real model)
# ============================================================================

class MockModel:
    """Mock model for demonstration"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.loaded = False

    def load(self):
        """Simulate model loading"""
        start_time = time.time()
        time.sleep(0.1)  # Simulate load time
        self.loaded = True

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.labels(
            model_name=self.name,
            model_version=self.version
        ).set(load_time)

        MODEL_LOADED.labels(
            model_name=self.name,
            model_version=self.version,
            model_type="classifier"
        ).set(1)

        logger.info(f"Model {self.name} v{self.version} loaded in {load_time:.3f}s")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Simulate prediction"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        # Simulate inference time based on batch size
        time.sleep(0.001 * len(inputs))

        # Return mock predictions
        return np.random.random(len(inputs))

# Global model registry
MODELS = {
    "bert-base": MockModel("bert-base", "v1.0"),
    "resnet-50": MockModel("resnet-50", "v2.1")
}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    for model in MODELS.values():
        model.load()

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Perform model inference

    Example:
    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{
        "model_name": "bert-base",
        "model_version": "v1.0",
        "inputs": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
      }'
    ```
    """
    model_name = request.model_name
    model_version = request.model_version

    # Get model
    if model_name not in MODELS:
        PREDICTION_ERRORS.labels(
            model_name=model_name,
            error_type="model_not_found"
        ).inc()
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model = MODELS[model_name]

    # Track active requests
    ACTIVE_REQUESTS.labels(model_name=model_name).inc()

    start_time = time.time()

    try:
        # Convert to numpy array
        inputs_array = np.array(request.inputs)

        # Track input size
        input_bytes = inputs_array.nbytes
        INPUT_SIZE.labels(model_name=model_name).observe(input_bytes)

        # Track batch size
        batch_size = len(inputs_array)
        BATCH_SIZE.labels(model_name=model_name).observe(batch_size)

        # Perform prediction
        predictions = model.predict(inputs_array)

        # Track output size
        output_bytes = predictions.nbytes
        OUTPUT_SIZE.labels(model_name=model_name).observe(output_bytes)

        # Record success metrics
        PREDICTIONS_TOTAL.labels(
            model_name=model_name,
            model_version=model_version
        ).inc(batch_size)

        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(latency)

        return PredictResponse(
            predictions=predictions.tolist(),
            latency_ms=latency * 1000,
            model_version=model_version
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_name=model_name,
            error_type=type(e).__name__
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.labels(model_name=model_name).dec()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": sum(1 for m in MODELS.values() if m.loaded)
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from starlette.responses import Response
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Prometheus configuration for this service:**

```yaml
scrape_configs:
  - job_name: 'ml-serving'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'ml-model-serving'
          env: 'dev'
```

**Example queries for this service:**

```promql
# Request rate
rate(http_requests_total[5m])

# P95 prediction latency by model
histogram_quantile(0.95,
  sum(rate(model_prediction_duration_seconds_bucket[5m])) by (model_name, le)
)

# Error rate percentage
(sum(rate(http_requests_total{status=~"5.."}[5m])) /
 sum(rate(http_requests_total[5m]))) * 100

# Average batch size
avg(model_batch_size) by (model_name)

# Predictions per second
sum(rate(model_predictions_total[1m])) by (model_name)
```

---

## Summary

In this lesson, you learned:

✅ Prometheus architecture and components for AI infrastructure
✅ Installing and configuring Prometheus in Docker and Kubernetes
✅ Writing PromQL queries for metrics analysis
✅ Instrumenting Python applications with prometheus-client
✅ Using exporters and service discovery
✅ Creating recording rules for performance
✅ Best practices for metric design and labeling
✅ Complete example of ML model serving with monitoring

## Next Steps

- **Lesson 03**: Learn to visualize Prometheus metrics with Grafana
- **Practice**: Instrument your own ML applications with Prometheus
- **Exercise**: Set up Prometheus for a multi-model serving system

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Python Client Library](https://github.com/prometheus/client_python)
- [GPU Metrics with DCGM](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html)

---

**Estimated Time:** 4-6 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 01, Python, Basic DevOps knowledge
