# Exercise 01: Production Observability Stack (Prometheus + Grafana + Loki)

**Estimated Time**: 28-36 hours

## Business Context

Your AI infrastructure team supports 50+ microservices processing 10M API requests/day for ML model inference. The current situation is chaotic:

- **No centralized metrics**: Each team uses different monitoring tools (CloudWatch, Datadog, custom scripts)
- **Alert fatigue**: 200+ alerts/day, 95% false positives
- **Slow incident resolution**: Average time to detect issues: 45 minutes, time to resolution: 4 hours
- **High costs**: $15,000/month for fragmented monitoring tools
- **Poor visibility**: Can't correlate metrics, logs, and traces across services

A recent production outage (3 hours downtime, $500K revenue loss) revealed critical gaps in observability. The CTO has mandated a **unified observability platform** with:

1. **Centralized metrics collection** for all services (Prometheus)
2. **Rich visualization** dashboards (Grafana)
3. **Centralized logging** (Loki/Grafana Loki)
4. **Intelligent alerting** (<10 alerts/day, >95% actionable)
5. **Cost efficiency** (<$5K/month for same coverage)

## Learning Objectives

After completing this exercise, you will be able to:

1. Deploy a production-grade Prometheus + Grafana + Loki stack on Kubernetes
2. Implement comprehensive metrics collection from infrastructure, applications, and ML models
3. Design effective Grafana dashboards following best practices
4. Configure intelligent alerting rules to minimize false positives
5. Set up centralized logging with structured logs and log aggregation
6. Implement SLIs (Service Level Indicators) and SLO (Service Level Objectives) tracking
7. Optimize monitoring costs and data retention policies

## Prerequisites

- Module 04 (Kubernetes fundamentals)
- Module 03 (Container best practices)
- Basic understanding of metrics, logs, and alerts
- Linux command-line proficiency
- Python programming

## Problem Statement

Build a **Production Observability Stack** that:

1. **Collects metrics** from Kubernetes, applications, and ML models
2. **Aggregates logs** from all services with structured logging
3. **Visualizes** system health through intuitive dashboards
4. **Alerts** on critical issues with minimal false positives
5. **Tracks SLOs** and error budgets
6. **Optimizes costs** through intelligent data retention

### Success Metrics

- All services (50+) instrumented with metrics and logs
- Alert volume reduced from 200/day to <10/day
- Alert accuracy >95% (actionable alerts)
- Mean time to detect (MTTD) issues: <5 minutes
- Mean time to resolution (MTTR): <30 minutes
- Total cost: <$5,000/month (70% cost reduction)
- Dashboard load time: <2 seconds for 30-day queries

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Production Observability Stack                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Prometheus  │    │ Grafana Loki │    │   Grafana    │  │
│  │  (Metrics)   │◀───│   (Logs)     │◀───│ (Visualize)  │  │
│  │  TSDB        │    │  LogQL       │    │  Dashboards  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                    ▲                    │          │
│         │                    │                    │          │
│         │                    │                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Exporters   │    │   Promtail   │    │ AlertManager │  │
│  │  - Node      │    │ (Log Agent)  │    │  (Alerts)    │  │
│  │  - Kube State│    │  - Scrape    │    │  - PagerDuty │  │
│  │  - cAdvisor  │    │  - Parse     │    │  - Slack     │  │
│  │  - Custom    │    │  - Forward   │    │  - Email     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                    ▲                              │
│         │                    │                              │
│         └────────────────────┴──────────────┐               │
│                                              │               │
│  ┌──────────────────────────────────────────┼─────────────┐ │
│  │         Kubernetes Cluster               │             │ │
│  │                                           │             │ │
│  │  ┌──────────────┐    ┌──────────────┐   │             │ │
│  │  │  ML Model    │    │  API Gateway │   │             │ │
│  │  │  Inference   │───▶│   (FastAPI)  │───┘             │ │
│  │  │  (PyTorch)   │    │              │                 │ │
│  │  └──────────────┘    └──────────────┘                 │ │
│  │         │                    │                         │ │
│  │         └────────────────────┴─────────────┐           │ │
│  │                                             ▼           │ │
│  │                                     ┌──────────────┐    │ │
│  │                                     │ Feature      │    │ │
│  │                                     │ Store        │    │ │
│  │                                     │ (Redis)      │    │ │
│  │                                     └──────────────┘    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Deploy Prometheus Stack on Kubernetes (7-9 hours)

Deploy Prometheus with high availability and persistent storage.

#### 1.1 Prometheus Deployment

Create `kubernetes/prometheus/deployment.yaml`:

```yaml
# TODO: Deploy Prometheus with persistent volume

apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s  # How often to scrape targets
      evaluation_interval: 15s  # How often to evaluate rules
      external_labels:
        cluster: 'production'
        region: 'us-west-2'

    # Alertmanager configuration
    alerting:
      alertmanagers:
        - static_configs:
            - targets: ['alertmanager:9093']

    # Load alerting rules
    rule_files:
      - '/etc/prometheus/rules/*.yml'

    scrape_configs:
      # Scrape Prometheus itself
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']

      # TODO: Scrape Kubernetes API server metrics
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
            action: keep
            regex: default;kubernetes;https

      # TODO: Scrape Kubernetes nodes (kubelet)
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)

      # TODO: Scrape Kubernetes pods with prometheus.io/scrape annotation
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          # Only scrape pods with prometheus.io/scrape: "true"
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          # Use custom port if specified
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            target_label: __address__
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
          # Add pod labels as metric labels
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - source_labels: [__meta_kubernetes_namespace]
            target_label: kubernetes_namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: kubernetes_pod_name

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
  namespace: monitoring
spec:
  serviceName: prometheus
  replicas: 2  # High availability
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
        - name: prometheus
          image: prom/prometheus:v2.45.0
          args:
            - '--config.file=/etc/prometheus/prometheus.yml'
            - '--storage.tsdb.path=/prometheus'
            - '--storage.tsdb.retention.time=30d'  # Keep 30 days of data
            - '--storage.tsdb.retention.size=50GB'  # Max 50GB per replica
            - '--web.enable-lifecycle'  # Allow reloading config via API
            - '--web.enable-admin-api'  # Enable admin APIs for debugging
          ports:
            - containerPort: 9090
              name: http
          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
            - name: storage
              mountPath: /prometheus
            - name: rules
              mountPath: /etc/prometheus/rules
          resources:
            requests:
              memory: "4Gi"
              cpu: "1000m"
            limits:
              memory: "8Gi"
              cpu: "2000m"
          # TODO: Implement readiness and liveness probes
          readinessProbe:
            httpGet:
              path: /-/ready
              port: 9090
            initialDelaySeconds: 30
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /-/healthy
              port: 9090
            initialDelaySeconds: 30
            periodSeconds: 15
      volumes:
        - name: config
          configMap:
            name: prometheus-config
        - name: rules
          configMap:
            name: prometheus-rules
  volumeClaimTemplates:
    - metadata:
        name: storage
      spec:
        accessModes: ['ReadWriteOnce']
        storageClassName: fast-ssd  # Use SSD for better query performance
        resources:
          requests:
            storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
    - port: 9090
      targetPort: 9090
  type: ClusterIP
```

#### 1.2 Exporters for Comprehensive Metrics

Create `kubernetes/prometheus/exporters.yaml`:

```yaml
# TODO: Deploy Node Exporter (hardware/OS metrics)

apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostNetwork: true  # Access host network stats
      hostPID: true      # Access host processes
      containers:
        - name: node-exporter
          image: prom/node-exporter:v1.6.0
          args:
            - '--path.procfs=/host/proc'
            - '--path.sysfs=/host/sys'
            - '--path.rootfs=/host/root'
            - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
          ports:
            - containerPort: 9100
              name: metrics
          volumeMounts:
            - name: proc
              mountPath: /host/proc
              readOnly: true
            - name: sys
              mountPath: /host/sys
              readOnly: true
            - name: root
              mountPath: /host/root
              readOnly: true
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
      volumes:
        - name: proc
          hostPath:
            path: /proc
        - name: sys
          hostPath:
            path: /sys
        - name: root
          hostPath:
            path: /

---
# TODO: Deploy Kube State Metrics (Kubernetes object state)

apiVersion: apps/v1
kind: Deployment
metadata:
  name: kube-state-metrics
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kube-state-metrics
  template:
    metadata:
      labels:
        app: kube-state-metrics
    spec:
      serviceAccountName: kube-state-metrics
      containers:
        - name: kube-state-metrics
          image: registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.9.2
          ports:
            - containerPort: 8080
              name: http-metrics
          resources:
            requests:
              memory: "256Mi"
              cpu: "200m"
            limits:
              memory: "512Mi"
              cpu: "500m"

# TODO: Add service and RBAC permissions for kube-state-metrics
```

#### 1.3 Custom Application Metrics

Create `src/instrumentation/prometheus_metrics.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, Summary
from prometheus_client import start_http_server, REGISTRY
from functools import wraps
import time
from typing import Callable

class PrometheusInstrumentation:
    """
    Instrumentation for ML inference service.

    Expose metrics on /metrics endpoint for Prometheus scraping.
    """

    def __init__(self, service_name: str, port: int = 8000):
        self.service_name = service_name

        # TODO: Define request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )

        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]  # SLO-aligned buckets
        )

        # TODO: Define ML-specific metrics
        self.prediction_count = Counter(
            'ml_predictions_total',
            'Total ML predictions made',
            ['model_name', 'model_version']
        )

        self.prediction_latency = Histogram(
            'ml_prediction_duration_seconds',
            'ML prediction latency',
            ['model_name', 'model_version'],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]  # Most predictions <100ms
        )

        self.model_load_time = Gauge(
            'ml_model_load_time_seconds',
            'Time to load model',
            ['model_name', 'model_version']
        )

        self.active_models = Gauge(
            'ml_active_models',
            'Number of models currently loaded',
            ['model_name']
        )

        # TODO: Feature store metrics
        self.feature_fetch_duration = Histogram(
            'feature_fetch_duration_seconds',
            'Time to fetch features',
            ['feature_store'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
        )

        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Feature cache hit ratio',
            ['cache_type']
        )

        # TODO: Resource metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )

        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used',
            ['gpu_id']
        )

    def track_request(self, method: str, endpoint: str):
        """
        TODO: Decorator to track HTTP request metrics

        Usage:
        @instrumentor.track_request('POST', '/predict')
        async def predict(request):
            ...

        Tracks:
        - Request count
        - Request duration
        - Error rate (5xx responses)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    status = getattr(result, 'status_code', 200)
                    return result
                except Exception as e:
                    status = 500
                    raise
                finally:
                    duration = time.time() - start_time

                    # TODO: Record metrics
                    self.request_count.labels(
                        method=method,
                        endpoint=endpoint,
                        status=status
                    ).inc()

                    self.request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)

            return wrapper
        return decorator

    def track_prediction(self, model_name: str, model_version: str):
        """
        TODO: Decorator to track ML prediction metrics

        Usage:
        @instrumentor.track_prediction('fraud-detector', 'v2.1')
        async def predict(features):
            ...

        Tracks:
        - Prediction count
        - Prediction latency
        - Model version usage
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # TODO: Record metrics
                self.prediction_count.labels(
                    model_name=model_name,
                    model_version=model_version
                ).inc()

                self.prediction_latency.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(duration)

                return result
            return wrapper
        return decorator

    def update_gpu_metrics(self):
        """
        TODO: Update GPU utilization metrics (called periodically)

        Use pynvml to query NVIDIA GPU stats:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            self.gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)
            self.gpu_memory_used.labels(gpu_id=str(i)).set(memory_info.used)

        pynvml.nvmlShutdown()
        """
        pass

    def start_metrics_server(self, port: int = 9090):
        """
        TODO: Start Prometheus metrics HTTP server

        start_http_server(port)

        Exposes /metrics endpoint for Prometheus to scrape
        """
        pass
```

### Part 2: Deploy Grafana + Loki for Logs (6-8 hours)

Set up centralized logging with Grafana Loki and log visualization.

#### 2.1 Loki Deployment

Create `kubernetes/loki/deployment.yaml`:

```yaml
# TODO: Deploy Loki for log aggregation

apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  loki.yaml: |
    auth_enabled: false

    server:
      http_listen_port: 3100

    ingester:
      lifecycler:
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1
      chunk_idle_period: 5m
      chunk_retain_period: 30s
      max_chunk_age: 1h
      max_transfer_retries: 0

    schema_config:
      configs:
        - from: 2023-01-01
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h

    storage_config:
      boltdb_shipper:
        active_index_directory: /loki/boltdb-shipper-active
        cache_location: /loki/boltdb-shipper-cache
        shared_store: filesystem
      filesystem:
        directory: /loki/chunks

    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h  # 7 days
      ingestion_rate_mb: 10
      ingestion_burst_size_mb: 20

    chunk_store_config:
      max_look_back_period: 744h  # 31 days

    table_manager:
      retention_deletes_enabled: true
      retention_period: 744h  # 31 days

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: loki
  namespace: monitoring
spec:
  serviceName: loki
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
        - name: loki
          image: grafana/loki:2.8.0
          args:
            - '-config.file=/etc/loki/loki.yaml'
          ports:
            - containerPort: 3100
              name: http
          volumeMounts:
            - name: config
              mountPath: /etc/loki
            - name: storage
              mountPath: /loki
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
      volumes:
        - name: config
          configMap:
            name: loki-config
  volumeClaimTemplates:
    - metadata:
        name: storage
      spec:
        accessModes: ['ReadWriteOnce']
        resources:
          requests:
            storage: 50Gi
```

#### 2.2 Promtail Log Collection

Create `kubernetes/loki/promtail.yaml`:

```yaml
# TODO: Deploy Promtail to scrape logs from all pods

apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: monitoring
data:
  promtail.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      # Scrape all pod logs
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          # Add namespace label
          - source_labels: [__meta_kubernetes_pod_namespace]
            target_label: namespace
          # Add pod name label
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          # Add container name label
          - source_labels: [__meta_kubernetes_pod_container_name]
            target_label: container
          # Add all pod labels as labels
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
        pipeline_stages:
          # Parse JSON logs
          - json:
              expressions:
                timestamp: timestamp
                level: level
                message: message
                trace_id: trace_id
          # Extract timestamp
          - timestamp:
              source: timestamp
              format: RFC3339
          # Set log level as label
          - labels:
              level: level

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      serviceAccountName: promtail
      containers:
        - name: promtail
          image: grafana/promtail:2.8.0
          args:
            - '-config.file=/etc/promtail/promtail.yaml'
          volumeMounts:
            - name: config
              mountPath: /etc/promtail
            - name: varlog
              mountPath: /var/log
              readOnly: true
            - name: varlibdockercontainers
              mountPath: /var/lib/docker/containers
              readOnly: true
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
      volumes:
        - name: config
          configMap:
            name: promtail-config
        - name: varlog
          hostPath:
            path: /var/log
        - name: varlibdockercontainers
          hostPath:
            path: /var/lib/docker/containers
```

#### 2.3 Structured Logging

Create `src/logging/structured_logger.py`:

```python
import logging
import json
from pythonjsonlogger import jsonlogger
from typing import Dict, Any, Optional
import traceback
from contextvars import ContextVar

# Context variable for trace ID (propagated across async calls)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)

class StructuredLogger:
    """
    Structured JSON logger for centralized log aggregation.

    All logs are JSON-formatted for easy parsing by Loki/Elasticsearch.
    """

    def __init__(self, service_name: str, level: str = 'INFO'):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # TODO: Configure JSON formatter
        json_handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s %(trace_id)s %(service)s'
        )
        json_handler.setFormatter(formatter)
        self.logger.addHandler(json_handler)

    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Add standard context to all log entries

        - timestamp (ISO 8601)
        - service name
        - trace_id (for distributed tracing correlation)
        - environment (dev/staging/prod)
        """
        from datetime import datetime

        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name,
            'trace_id': trace_id_var.get(),
            **extra
        }

        return context

    def info(self, message: str, **kwargs):
        """
        TODO: Log info message with structured context

        Usage:
        logger.info(
            "Model prediction completed",
            model_name="fraud-detector",
            model_version="v2.1",
            latency_ms=45.2,
            prediction="fraud"
        )

        Output:
        {
            "timestamp": "2023-10-25T10:30:45.123Z",
            "level": "INFO",
            "service": "ml-inference",
            "message": "Model prediction completed",
            "trace_id": "abc-123-def",
            "model_name": "fraud-detector",
            "model_version": "v2.1",
            "latency_ms": 45.2,
            "prediction": "fraud"
        }
        """
        extra = self._add_context(kwargs)
        self.logger.info(message, extra=extra)

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """
        TODO: Log error with exception traceback

        Include full exception details for debugging
        """
        extra = self._add_context(kwargs)

        if exc_info:
            extra['exception_type'] = type(exc_info).__name__
            extra['exception_message'] = str(exc_info)
            extra['traceback'] = traceback.format_exc()

        self.logger.error(message, extra=extra, exc_info=exc_info)

    def warn(self, message: str, **kwargs):
        """TODO: Log warning message"""
        extra = self._add_context(kwargs)
        self.logger.warning(message, extra=extra)

    def set_trace_id(self, trace_id: str):
        """Set trace ID for correlation with distributed tracing."""
        trace_id_var.set(trace_id)

# Usage example:
logger = StructuredLogger('ml-inference')
logger.set_trace_id('request-123-abc')
logger.info(
    "Processing prediction request",
    model_name="fraud-detector",
    input_features=["amount", "merchant", "location"]
)
```

### Part 3: Grafana Dashboards (7-9 hours)

Create comprehensive dashboards for system visibility.

#### 3.1 Grafana Deployment

Create `kubernetes/grafana/deployment.yaml`:

```yaml
# TODO: Deploy Grafana with Prometheus and Loki data sources

apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: monitoring
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
        jsonData:
          timeInterval: 15s

      - name: Loki
        type: loki
        access: proxy
        url: http://loki:3100

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
        - name: grafana
          image: grafana/grafana:10.0.0
          ports:
            - containerPort: 3000
              name: http
          env:
            - name: GF_SECURITY_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: grafana-credentials
                  key: admin-password
            - name: GF_INSTALL_PLUGINS
              value: "grafana-piechart-panel,grafana-clock-panel"
          volumeMounts:
            - name: datasources
              mountPath: /etc/grafana/provisioning/datasources
            - name: dashboards-config
              mountPath: /etc/grafana/provisioning/dashboards
            - name: dashboards
              mountPath: /var/lib/grafana/dashboards
            - name: storage
              mountPath: /var/lib/grafana
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
      volumes:
        - name: datasources
          configMap:
            name: grafana-datasources
        - name: dashboards-config
          configMap:
            name: grafana-dashboard-config
        - name: dashboards
          configMap:
            name: grafana-dashboards
        - name: storage
          emptyDir: {}
```

#### 3.2 Dashboard-as-Code

Create `src/dashboards/ml_inference_dashboard.py`:

```python
from grafanalib.core import (
    Dashboard, TimeSeries, Target, GridPos,
    RowPanel, single_y_axis, Alert, AlertCondition
)

def create_ml_inference_dashboard():
    """
    TODO: Create ML inference dashboard using grafanalib

    Dashboard sections:
    1. Overview: Request rate, error rate, latency
    2. Model Performance: Predictions/sec, latency by model
    3. Resource Usage: CPU, memory, GPU utilization
    4. Feature Store: Cache hit rate, fetch latency
    5. SLO Tracking: Error budget, uptime

    This is "dashboard as code" for version control and automation
    """

    # TODO: Panel 1 - Request Rate
    request_rate_panel = TimeSeries(
        title='Request Rate (req/s)',
        dataSource='Prometheus',
        targets=[
            Target(
                expr='rate(http_requests_total{job="ml-inference"}[5m])',
                legendFormat='{{endpoint}}',
            ),
        ],
        gridPos=GridPos(h=8, w=12, x=0, y=0),
        yAxes=single_y_axis(format='reqps'),
    )

    # TODO: Panel 2 - Error Rate
    error_rate_panel = TimeSeries(
        title='Error Rate (%)',
        dataSource='Prometheus',
        targets=[
            Target(
                expr='rate(http_requests_total{job="ml-inference",status=~"5.."}[5m]) / rate(http_requests_total{job="ml-inference"}[5m]) * 100',
                legendFormat='Error Rate',
            ),
        ],
        gridPos=GridPos(h=8, w=12, x=12, y=0),
        yAxes=single_y_axis(format='percent'),
        alert=Alert(
            name='High Error Rate',
            message='Error rate above 1%',
            conditions=[
                AlertCondition(
                    evaluator={'params': [1], 'type': 'gt'},
                    operator={'type': 'and'},
                    query={'params': ['A', '5m', 'now']},
                    reducer={'params': [], 'type': 'avg'},
                    type='query',
                ),
            ],
        ),
    )

    # TODO: Panel 3 - P95 Latency
    latency_panel = TimeSeries(
        title='P95 Latency (ms)',
        dataSource='Prometheus',
        targets=[
            Target(
                expr='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="ml-inference"}[5m])) * 1000',
                legendFormat='{{endpoint}}',
            ),
        ],
        gridPos=GridPos(h=8, w=12, x=0, y=8),
        yAxes=single_y_axis(format='ms'),
    )

    # TODO: Panel 4 - Predictions per Model
    predictions_panel = TimeSeries(
        title='Predictions/sec by Model',
        dataSource='Prometheus',
        targets=[
            Target(
                expr='rate(ml_predictions_total[5m])',
                legendFormat='{{model_name}} ({{model_version}})',
            ),
        ],
        gridPos=GridPos(h=8, w=12, x=12, y=8),
    )

    # TODO: Panel 5 - GPU Utilization
    gpu_util_panel = TimeSeries(
        title='GPU Utilization (%)',
        dataSource='Prometheus',
        targets=[
            Target(
                expr='gpu_utilization_percent',
                legendFormat='GPU {{gpu_id}}',
            ),
        ],
        gridPos=GridPos(h=8, w=12, x=0, y=16),
        yAxes=single_y_axis(format='percent', max=100),
    )

    # TODO: Create dashboard
    dashboard = Dashboard(
        title='ML Inference Service',
        description='Monitoring for ML inference service',
        tags=['ml', 'inference', 'production'],
        timezone='UTC',
        panels=[
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=0), title='Overview'),
            request_rate_panel,
            error_rate_panel,
            latency_panel,
            predictions_panel,
            gpu_util_panel,
        ],
        refresh='30s',
    ).auto_panel_ids()

    return dashboard

# TODO: Export dashboard to JSON
if __name__ == '__main__':
    import json
    from grafanalib._gen import DashboardEncoder

    dashboard = create_ml_inference_dashboard()
    print(json.dumps(dashboard.to_json_data(), cls=DashboardEncoder, indent=2))
```

### Part 4: Intelligent Alerting (4-6 hours)

Configure AlertManager with smart routing and deduplication.

#### 4.1 Alert Rules

Create `kubernetes/prometheus/alert-rules.yaml`:

```yaml
# TODO: Define comprehensive alerting rules

apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  alerts.yml: |
    groups:
      # TODO: Infrastructure alerts
      - name: infrastructure
        interval: 30s
        rules:
          - alert: HighCPUUsage
            expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
            for: 5m
            labels:
              severity: warning
              team: platform
            annotations:
              summary: "High CPU usage on {{ $labels.instance }}"
              description: "CPU usage is above 80% (current: {{ $value }}%)"

          - alert: HighMemoryUsage
            expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
            for: 5m
            labels:
              severity: warning
              team: platform
            annotations:
              summary: "High memory usage on {{ $labels.instance }}"
              description: "Memory usage is above 85% (current: {{ $value }}%)"

          - alert: DiskSpaceLow
            expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 15
            for: 5m
            labels:
              severity: critical
              team: platform
            annotations:
              summary: "Disk space low on {{ $labels.instance }}"
              description: "Disk space is below 15% (current: {{ $value }}%)"

      # TODO: ML service alerts
      - name: ml_service
        interval: 30s
        rules:
          - alert: HighErrorRate
            expr: |
              rate(http_requests_total{job="ml-inference",status=~"5.."}[5m])
              /
              rate(http_requests_total{job="ml-inference"}[5m])
              * 100 > 1
            for: 5m
            labels:
              severity: critical
              team: ml-platform
            annotations:
              summary: "High error rate in ML inference service"
              description: "Error rate is above 1% (current: {{ $value }}%)"

          - alert: HighLatency
            expr: |
              histogram_quantile(0.95,
                rate(http_request_duration_seconds_bucket{job="ml-inference"}[5m])
              ) * 1000 > 500
            for: 5m
            labels:
              severity: warning
              team: ml-platform
            annotations:
              summary: "High P95 latency in ML inference"
              description: "P95 latency is above 500ms (current: {{ $value }}ms)"

          - alert: ModelPredictionFailure
            expr: rate(ml_predictions_total{status="failure"}[5m]) > 10
            for: 2m
            labels:
              severity: critical
              team: ml-platform
            annotations:
              summary: "ML model prediction failures detected"
              description: "Prediction failure rate: {{ $value }}/sec for {{ $labels.model_name }}"

          - alert: GPUUtilizationLow
            expr: avg(gpu_utilization_percent) < 20
            for: 15m
            labels:
              severity: info
              team: ml-platform
            annotations:
              summary: "Low GPU utilization"
              description: "GPU utilization is below 20% (current: {{ $value }}%). Consider scaling down."

      # TODO: SLO alerts (error budget)
      - name: slo
        interval: 1m
        rules:
          - alert: ErrorBudgetBurnRateCritical
            expr: |
              (
                1 - (
                  sum(rate(http_requests_total{job="ml-inference",status!~"5.."}[1h]))
                  /
                  sum(rate(http_requests_total{job="ml-inference"}[1h]))
                )
              ) / (1 - 0.999) > 14.4  # Burn rate for 99.9% SLO
            for: 2m
            labels:
              severity: critical
              team: ml-platform
            annotations:
              summary: "Critical error budget burn rate"
              description: "At this rate, entire monthly error budget will be exhausted in 2 days"

          - alert: ErrorBudgetExhausted
            expr: |
              (
                1 - (
                  sum(rate(http_requests_total{job="ml-inference",status!~"5.."}[30d]))
                  /
                  sum(rate(http_requests_total{job="ml-inference"}[30d]))
                )
              ) >= (1 - 0.999)
            for: 5m
            labels:
              severity: critical
              team: ml-platform
            annotations:
              summary: "Monthly error budget exhausted"
              description: "99.9% SLO violated - error budget for the month is fully consumed"
```

#### 4.2 AlertManager Configuration

Create `kubernetes/alertmanager/config.yaml`:

```yaml
# TODO: Configure AlertManager for intelligent routing

apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
      pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

    # TODO: Route alerts based on severity and team
    route:
      receiver: 'default'
      group_by: ['alertname', 'severity']
      group_wait: 10s  # Wait before sending first notification
      group_interval: 5m  # Wait before sending updates for grouped alerts
      repeat_interval: 4h  # Resend alerts every 4 hours if still firing

      routes:
        # Critical alerts -> PagerDuty (24/7 on-call)
        - match:
            severity: critical
          receiver: 'pagerduty'
          continue: true  # Also send to Slack

        # Warning alerts -> Slack (business hours)
        - match:
            severity: warning
          receiver: 'slack-warnings'

        # Info alerts -> Slack (no page)
        - match:
            severity: info
          receiver: 'slack-info'

        # Team-specific routing
        - match:
            team: ml-platform
          receiver: 'slack-ml-team'

    # TODO: Configure notification receivers
    receivers:
      - name: 'default'
        slack_configs:
          - channel: '#alerts'
            title: 'Alert: {{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
            description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

      - name: 'slack-warnings'
        slack_configs:
          - channel: '#alerts-warnings'
            color: 'warning'
            title: 'Warning: {{ .GroupLabels.alertname }}'

      - name: 'slack-info'
        slack_configs:
          - channel: '#alerts-info'
            color: 'good'

      - name: 'slack-ml-team'
        slack_configs:
          - channel: '#ml-platform-alerts'

    # TODO: Inhibition rules (suppress related alerts)
    inhibit_rules:
      # If node is down, suppress all other alerts from that node
      - source_match:
          alertname: 'NodeDown'
        target_match_re:
          instance: '.*'
        equal: ['instance']

      # If service is down, suppress high latency alerts
      - source_match:
          alertname: 'ServiceDown'
        target_match:
          alertname: 'HighLatency'
        equal: ['job']
```

### Part 5: SLO Tracking and Cost Optimization (4-6 hours)

Implement SLO tracking and optimize monitoring costs.

#### 5.1 SLO Calculator

Create `src/slo/slo_tracker.py`:

```python
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta
import requests

@dataclass
class SLO:
    name: str
    target: float  # e.g., 0.999 for 99.9% availability
    window: timedelta  # e.g., 30 days

@dataclass
class ErrorBudget:
    slo: SLO
    total_requests: int
    failed_requests: int
    success_rate: float
    error_budget_remaining: float  # 0-1, where 1 = 100% budget remaining
    burn_rate: float  # How fast error budget is being consumed

class SLOTracker:
    """
    Track SLOs and error budgets using Prometheus metrics.

    SLI (Service Level Indicator): Actual measured performance
    SLO (Service Level Objective): Target performance
    Error Budget: Allowed failure (100% - SLO target)
    """

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def calculate_error_budget(self, slo: SLO) -> ErrorBudget:
        """
        TODO: Calculate current error budget for SLO

        Steps:
        1. Query Prometheus for total requests in SLO window
        2. Query for failed requests (5xx errors)
        3. Calculate success rate = (total - failed) / total
        4. Calculate error budget remaining:
           allowed_failures = total * (1 - slo.target)
           error_budget_remaining = (allowed_failures - failed) / allowed_failures

        5. Calculate burn rate (how fast budget is being consumed):
           recent_failure_rate = failed_last_hour / total_last_hour
           burn_rate = recent_failure_rate / (1 - slo.target)
           - burn_rate = 1: Consuming budget at expected rate
           - burn_rate > 1: Consuming budget too fast
           - burn_rate < 1: Below error rate, building buffer

        Return ErrorBudget with all calculated values
        """
        pass

    def _query_prometheus(self, query: str) -> float:
        """
        TODO: Query Prometheus and return result

        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={'query': query}
        )
        result = response.json()
        return float(result['data']['result'][0]['value'][1])
        """
        pass

    def get_slo_dashboard_metrics(self, slo_name: str) -> Dict:
        """
        TODO: Get metrics for SLO dashboard

        Return:
        {
            'slo_target': 99.9,
            'current_availability': 99.95,
            'error_budget_remaining_percent': 50.0,
            'estimated_budget_exhaustion_date': '2023-11-15',
            'burn_rate': 0.8,
            'requests_last_30d': 10_000_000,
            'errors_last_30d': 5000
        }
        """
        pass

    def check_slo_compliance(self, slo: SLO) -> bool:
        """
        TODO: Check if currently meeting SLO

        error_budget = self.calculate_error_budget(slo)
        return error_budget.success_rate >= slo.target
        """
        pass
```

#### 5.2 Cost Optimization

Create `src/optimization/metrics_optimizer.py`:

```python
class MetricsOptimizer:
    """
    Optimize Prometheus storage costs through:
    1. Metric cardinality reduction
    2. Intelligent downsampling
    3. Retention policy optimization
    """

    def analyze_cardinality(self, prometheus_url: str) -> Dict:
        """
        TODO: Analyze metric cardinality (number of unique time series)

        High cardinality = expensive storage and slow queries

        Query for cardinality per metric:
        count by (__name__) ({__name__=~".+"})

        Identify high-cardinality metrics (>10,000 series):
        - user_id labels: DON'T use user_id as label (millions of users)
        - request_id labels: DON'T use request_id as label
        - IP addresses: DON'T use IP as label

        Solution: Use aggregated metrics or sample subset
        """
        pass

    def recommend_retention_policy(
        self,
        metrics: List[str],
        query_patterns: Dict[str, int]  # metric -> days_queried_back
    ) -> Dict[str, int]:
        """
        TODO: Recommend retention based on actual query patterns

        Analyze:
        - Which metrics are queried frequently vs rarely
        - How far back queries typically go

        Recommendations:
        - Critical SLO metrics: 90 days
        - Debugging metrics: 30 days
        - Rarely queried metrics: 7 days
        - Very high cardinality: 1 day

        This can reduce storage by 50-70%
        """
        pass

    def calculate_storage_cost(
        self,
        num_metrics: int,
        cardinality: int,
        scrape_interval_seconds: int,
        retention_days: int,
        bytes_per_sample: int = 2
    ) -> Dict:
        """
        TODO: Estimate Prometheus storage requirements

        Formula:
        samples_per_day = cardinality * (86400 / scrape_interval_seconds)
        total_samples = samples_per_day * retention_days
        storage_bytes = total_samples * bytes_per_sample

        Also calculate query performance impact:
        - High cardinality + long retention = slow queries

        Return:
        {
            'storage_gb': 150.5,
            'estimated_monthly_cost': 45.15,  # AWS EBS: $0.10/GB-month
            'query_performance': 'good/moderate/poor'
        }
        """
        pass
```

## Acceptance Criteria

### Functional Requirements

- [ ] Prometheus deployed with HA (2 replicas) and 30-day retention
- [ ] All services (50+) instrumented with metrics (/metrics endpoint)
- [ ] Loki + Promtail deployed for centralized logging
- [ ] All services using structured JSON logging
- [ ] Grafana deployed with Prometheus and Loki data sources
- [ ] 5+ dashboards created (infrastructure, ML service, SLO, costs)
- [ ] AlertManager configured with PagerDuty and Slack integration
- [ ] 15+ alert rules covering critical scenarios
- [ ] SLO tracking implemented for key services

### Performance Requirements

- [ ] Dashboard load time <2 seconds for 30-day queries
- [ ] Prometheus query response time <1 second (P95)
- [ ] Log ingestion rate >10,000 logs/second
- [ ] Alert notification latency <30 seconds from incident

### Operational Requirements

- [ ] Alert volume <10/day with >95% actionable
- [ ] MTTD (Mean Time To Detect) <5 minutes
- [ ] Total monitoring cost <$5,000/month
- [ ] Prometheus storage <100GB per replica
- [ ] Log retention 30 days with <50GB storage

### Code Quality

- [ ] All dashboards defined as code (grafanalib)
- [ ] Comprehensive alert rules documentation
- [ ] Runbooks for all critical alerts
- [ ] Tests for custom metrics instrumentation

## Testing Strategy

### Unit Tests

```python
# tests/test_metrics.py
def test_prometheus_instrumentation():
    """Test that metrics are correctly incremented."""
    instrumentor = PrometheusInstrumentation('test-service')
    # TODO: Verify metrics work correctly
```

### Integration Tests

```python
# tests/test_observability_stack.py
def test_end_to_end_observability():
    """Test complete flow: app logs -> Loki -> Grafana query."""
    # 1. Generate test log
    # 2. Wait for Promtail to scrape
    # 3. Query Loki via Grafana API
    # 4. Verify log appears
```

## Deliverables

1. **Kubernetes Manifests** (`kubernetes/`):
   - `prometheus/` - Prometheus, exporters, alert rules
   - `loki/` - Loki, Promtail configuration
   - `grafana/` - Grafana deployment, data sources
   - `alertmanager/` - AlertManager configuration

2. **Source Code** (`src/`):
   - `instrumentation/prometheus_metrics.py` - Metrics instrumentation
   - `logging/structured_logger.py` - Structured logging
   - `dashboards/` - Dashboard-as-code definitions
   - `slo/slo_tracker.py` - SLO tracking and error budgets
   - `optimization/metrics_optimizer.py` - Cost optimization

3. **Documentation** (`docs/`):
   - `OBSERVABILITY_GUIDE.md` - Complete setup guide
   - `DASHBOARD_GUIDE.md` - Dashboard usage
   - `RUNBOOKS.md` - Alert runbooks
   - `SLO_DEFINITIONS.md` - SLO targets and calculations

4. **Dashboards** (`dashboards/`):
   - ML inference dashboard JSON
   - Infrastructure dashboard JSON
   - SLO dashboard JSON
   - Cost optimization dashboard JSON

## Bonus Challenges

1. **Distributed Tracing Integration** (+8 hours):
   - Deploy Jaeger/Tempo
   - Instrument services with OpenTelemetry
   - Correlate traces with logs and metrics
   - Create trace-based alerts

2. **Anomaly Detection** (+6 hours):
   - Implement ML-based anomaly detection on metrics
   - Use Prophet or ARIMA for forecasting
   - Alert on anomalies (not just thresholds)

3. **Multi-Cluster Observability** (+6 hours):
   - Federate Prometheus across multiple clusters
   - Central Grafana for all clusters
   - Cross-cluster alerting

## Resources

### Official Documentation

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Loki](https://grafana.com/docs/loki/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/)
- [AlertManager](https://prometheus.io/docs/alerting/latest/alertmanager/)

### Best Practices

- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [The RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)

## Submission

Submit your implementation via Git:

```bash
git add .
git commit -m "Complete Exercise 01: Production Observability Stack"
git push origin exercise-01-observability-stack
```

Ensure your submission includes:
- All Kubernetes manifests
- Instrumentation code
- Dashboards (both code and JSON)
- Alert rules and runbooks
- Documentation

---

**Estimated Time Breakdown**:
- Part 1 (Prometheus Stack): 7-9 hours
- Part 2 (Loki Logging): 6-8 hours
- Part 3 (Grafana Dashboards): 7-9 hours
- Part 4 (Alerting): 4-6 hours
- Part 5 (SLO & Optimization): 4-6 hours
- **Total**: 28-36 hours
