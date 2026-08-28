# Lesson 09: Monitoring and Troubleshooting Kubernetes

## Learning Objectives
By the end of this lesson, you will be able to:
- Use `kubectl` commands for debugging and troubleshooting
- Implement comprehensive logging strategies for ML workloads
- Set up monitoring with Prometheus and Grafana
- Create alerts for critical ML infrastructure issues
- Debug common Kubernetes problems in production
- Understand distributed tracing for ML services
- Apply best practices for production ML observability

## Prerequisites
- Completed lessons 01-08 (Kubernetes fundamentals)
- Basic understanding of metrics, logs, and traces
- Familiarity with ML training and inference workflows
- Experience with YAML and command-line tools

## Introduction

**Why monitoring and observability matter for ML infrastructure:**
- **Training failures:** GPU out-of-memory, data loading bottlenecks
- **Inference latency:** Model response time, queue depths
- **Resource utilization:** GPU/CPU/memory efficiency
- **Cost optimization:** Identify underutilized resources
- **SLA compliance:** Track uptime, error rates, latency percentiles

**The three pillars of observability:**
1. **Metrics:** Numerical time-series data (CPU%, latency, throughput)
2. **Logs:** Text records of events (errors, warnings, debug info)
3. **Traces:** Request flow through distributed systems

**Real-world examples:**
- **Uber:** Monitors 10,000+ ML jobs daily, alerts on GPU underutilization
- **Netflix:** Tracks model inference latency to maintain 99.99% SLA
- **Spotify:** Monitors feature store latency and data freshness
- **OpenAI:** Uses distributed tracing to debug multi-model orchestration

## 1. kubectl Debugging Commands

### 1.1 Essential kubectl Commands

**Check cluster health:**

```bash
# View cluster nodes
kubectl get nodes
kubectl describe node <node-name>

# Check system pods
kubectl get pods -n kube-system

# View cluster events
kubectl get events --all-namespaces --sort-by='.lastTimestamp'

# Check API server health
kubectl get --raw /healthz
kubectl get --raw /livez
kubectl get --raw /readyz
```

**Inspect pods:**

```bash
# List pods with detailed info
kubectl get pods -n ml-serving -o wide

# Describe pod (shows events, volumes, status)
kubectl describe pod <pod-name> -n ml-serving

# Get pod YAML
kubectl get pod <pod-name> -n ml-serving -o yaml

# Watch pod status
kubectl get pods -n ml-serving --watch
```

**View logs:**

```bash
# Get logs from pod
kubectl logs <pod-name> -n ml-serving

# Follow logs in real-time
kubectl logs -f <pod-name> -n ml-serving

# Get logs from previous container instance (if pod restarted)
kubectl logs <pod-name> -n ml-serving --previous

# Logs from specific container in multi-container pod
kubectl logs <pod-name> -c <container-name> -n ml-serving

# Last 100 lines
kubectl logs <pod-name> -n ml-serving --tail=100

# Logs since timestamp
kubectl logs <pod-name> -n ml-serving --since=1h
kubectl logs <pod-name> -n ml-serving --since-time=2023-10-15T10:00:00Z
```

**Execute commands in pods:**

```bash
# Interactive shell
kubectl exec -it <pod-name> -n ml-serving -- /bin/bash

# Run single command
kubectl exec <pod-name> -n ml-serving -- nvidia-smi
kubectl exec <pod-name> -n ml-serving -- ps aux
kubectl exec <pod-name> -n ml-serving -- df -h

# Test network connectivity
kubectl exec <pod-name> -n ml-serving -- curl http://another-service/health
```

**Port forwarding:**

```bash
# Forward local port to pod
kubectl port-forward <pod-name> 8080:8080 -n ml-serving

# Access at http://localhost:8080

# Forward to service
kubectl port-forward service/<service-name> 8080:80 -n ml-serving

# Forward to deployment (any pod)
kubectl port-forward deployment/<deployment-name> 8080:8080 -n ml-serving
```

**Resource usage:**

```bash
# Top nodes (CPU, memory usage)
kubectl top nodes

# Top pods
kubectl top pods -n ml-serving

# Sort by CPU
kubectl top pods -n ml-serving --sort-by=cpu

# Sort by memory
kubectl top pods -n ml-serving --sort-by=memory

# Show containers
kubectl top pods -n ml-serving --containers
```

### 1.2 Debugging ML Training Jobs

**Scenario: Training job stuck in Pending**

```bash
# 1. Check job status
kubectl get job training-job -n ml-training

# 2. Check pod status
kubectl get pods -l job-name=training-job -n ml-training

# 3. Describe pod to see why it's pending
kubectl describe pod <pod-name> -n ml-training

# Common reasons:
# - Insufficient GPU resources
# - Insufficient CPU/memory
# - Node selector doesn't match any nodes
# - Taints not tolerated
# - PVC not bound

# 4. Check events
kubectl get events -n ml-training --sort-by='.lastTimestamp'

# Example event:
# Warning  FailedScheduling  pod/training-job-xyz  0/3 nodes available: insufficient nvidia.com/gpu
```

**Solution: Check GPU availability**

```bash
# View GPU capacity across nodes
kubectl describe nodes | grep -A 10 "Allocated resources"

# Or install kubectl-gpu plugin
kubectl resource-capacity --pods --util --sort gpu.nvidia.com
```

**Scenario: Training job crashing (CrashLoopBackOff)**

```bash
# 1. Check pod status
kubectl get pods -n ml-training

# NAME               READY   STATUS             RESTARTS   AGE
# training-job-xyz   0/1     CrashLoopBackOff   5          10m

# 2. View logs from crashed container
kubectl logs training-job-xyz -n ml-training --previous

# Example error:
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB

# 3. Check resource limits
kubectl describe pod training-job-xyz -n ml-training | grep -A 10 Resources

# 4. Check if volume mounts are correct
kubectl describe pod training-job-xyz -n ml-training | grep -A 10 Mounts
```

**Solution examples:**

```bash
# Reduce batch size (if OOM error)
kubectl set env deployment/training-job BATCH_SIZE=16 -n ml-training

# Increase memory limit
kubectl patch deployment training-job -n ml-training -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"training","resources":{"limits":{"memory":"32Gi"}}}]}}}}'

# Or edit directly
kubectl edit deployment training-job -n ml-training
```

### 1.3 Debugging ML Inference Services

**Scenario: Inference API returning 502 Bad Gateway**

```bash
# 1. Check service and endpoints
kubectl get service bert-inference -n ml-serving
kubectl get endpoints bert-inference -n ml-serving

# If endpoints is <none>, no healthy pods are backing the service

# 2. Check pod health
kubectl get pods -l app=bert-inference -n ml-serving

# 3. Check readiness probes
kubectl describe pod <pod-name> -n ml-serving | grep -A 5 Readiness

# 4. Test endpoint directly
kubectl exec <pod-name> -n ml-serving -- curl http://localhost:8080/health
kubectl exec <pod-name> -n ml-serving -- curl http://localhost:8080/ready

# 5. Check logs for errors
kubectl logs <pod-name> -n ml-serving | grep -i error
```

**Scenario: High inference latency**

```bash
# 1. Check resource utilization
kubectl top pods -n ml-serving

# 2. Check GPU utilization (if GPU-based)
kubectl exec <pod-name> -n ml-serving -- nvidia-smi

# 3. Check HPA status (if autoscaling enabled)
kubectl get hpa -n ml-serving
kubectl describe hpa bert-inference-hpa -n ml-serving

# 4. Check for CPU throttling
kubectl exec <pod-name> -n ml-serving -- cat /sys/fs/cgroup/cpu/cpu.stat

# 5. Review application logs for slow requests
kubectl logs <pod-name> -n ml-serving | grep "request_duration"
```

## 2. Logging Strategies

### 2.1 Application Logging Best Practices

**Structured logging (JSON format):**

```python
# train.py
import logging
import json

logger = logging.getLogger(__name__)

# Configure JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "job_name": os.environ.get("JOB_NAME"),
            "epoch": getattr(record, 'epoch', None),
            "loss": getattr(record, 'loss', None),
        }
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Training started")
logger.info("Epoch complete", extra={"epoch": 1, "loss": 0.123})
```

**Output:**
```json
{"timestamp": "2023-10-15 10:30:00", "level": "INFO", "message": "Training started", "job_name": "resnet-training"}
{"timestamp": "2023-10-15 10:35:00", "level": "INFO", "message": "Epoch complete", "job_name": "resnet-training", "epoch": 1, "loss": 0.123}
```

**Benefits:**
- Easier to parse and query
- Can filter by fields (epoch, loss, etc.)
- Integrates well with log aggregators (Loki, Elasticsearch)

### 2.2 Log Aggregation with Loki

**Install Loki stack (Loki + Promtail + Grafana):**

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install loki-stack grafana/loki-stack \
  --namespace logging \
  --create-namespace \
  --set grafana.enabled=true \
  --set prometheus.enabled=true \
  --set promtail.enabled=true
```

**Promtail configuration (auto-installed):**

Promtail runs as a DaemonSet on each node, collecting logs from:
- `/var/log/pods/**/*.log` (all pod logs)

**Query logs in Grafana:**

```bash
# Port-forward to Grafana
kubectl port-forward service/loki-stack-grafana 3000:80 -n logging

# Get admin password
kubectl get secret loki-stack-grafana -n logging -o jsonpath="{.data.admin-password}" | base64 -d
```

**LogQL queries (Loki Query Language):**

```logql
# All logs from namespace
{namespace="ml-training"}

# Logs from specific job
{namespace="ml-training", job_name="resnet-training"}

# Error logs only
{namespace="ml-training"} |= "ERROR"

# Training metrics
{namespace="ml-training"} | json | loss > 1.0

# Count errors per minute
rate({namespace="ml-training"} |= "ERROR" [1m])
```

### 2.3 Fluentd for Advanced Log Aggregation

**Install Fluentd DaemonSet:**

```yaml
# fluentd-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
```

**Filter and enrich logs:**

```ruby
# fluentd.conf
<filter kubernetes.**>
  @type kubernetes_metadata
</filter>

<filter kubernetes.var.log.containers.training-**.log>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

<match kubernetes.var.log.containers.training-**.log>
  @type elasticsearch
  host elasticsearch.logging.svc.cluster.local
  port 9200
  logstash_format true
  logstash_prefix ml-training
</match>
```

## 3. Metrics and Monitoring with Prometheus

### 3.1 Install Prometheus Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi
```

**What gets installed:**
- Prometheus (metrics collection)
- Grafana (visualization)
- Alertmanager (alerting)
- Node Exporter (node metrics)
- Kube State Metrics (K8s object metrics)
- Pre-configured dashboards and alerts

### 3.2 Instrumenting ML Applications

**Python (Prometheus client library):**

```python
# inference_server.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions', ['model_name', 'status'])
prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model_name'])
gpu_memory_usage = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id'])
queue_depth = Gauge('prediction_queue_depth', 'Number of requests in queue')

# Expose metrics endpoint on port 8000
start_http_server(8000)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        # Run inference
        result = model.predict(request.data)

        # Record success
        prediction_counter.labels(model_name='bert', status='success').inc()

        return jsonify(result)

    except Exception as e:
        # Record failure
        prediction_counter.labels(model_name='bert', status='error').inc()
        raise

    finally:
        # Record latency
        duration = time.time() - start_time
        prediction_latency.labels(model_name='bert').observe(duration)
```

**Expose metrics endpoint in Kubernetes:**

```yaml
# service-with-metrics.yaml
apiVersion: v1
kind: Service
metadata:
  name: bert-inference
  namespace: ml-serving
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: bert
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 8000
    targetPort: 8000
```

**ServiceMonitor (for Prometheus Operator):**

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: bert-inference-metrics
  namespace: ml-serving
spec:
  selector:
    matchLabels:
      app: bert
  endpoints:
  - port: metrics
    interval: 30s
```

### 3.3 Key Metrics to Monitor

**ML Training Metrics:**

```promql
# Training throughput (samples/sec)
rate(training_samples_total[5m])

# Training loss (latest value)
training_loss

# GPU utilization
DCGM_FI_DEV_GPU_UTIL{namespace="ml-training"}

# GPU memory usage
DCGM_FI_DEV_FB_USED{namespace="ml-training"} / DCGM_FI_DEV_FB_TOTAL{namespace="ml-training"} * 100

# Data loading time
rate(data_loading_duration_seconds_sum[5m]) / rate(data_loading_duration_seconds_count[5m])
```

**ML Inference Metrics:**

```promql
# Request rate (req/sec)
rate(model_predictions_total[5m])

# Error rate
rate(model_predictions_total{status="error"}[5m]) / rate(model_predictions_total[5m]) * 100

# P50, P95, P99 latency
histogram_quantile(0.50, rate(model_prediction_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(model_prediction_duration_seconds_bucket[5m]))

# Queue depth
prediction_queue_depth

# Replica count
kube_deployment_status_replicas{namespace="ml-serving", deployment="bert-inference"}
```

**Infrastructure Metrics:**

```promql
# Node CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Node memory usage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100

# Pod CPU usage
sum(rate(container_cpu_usage_seconds_total{namespace="ml-serving"}[5m])) by (pod)

# Pod memory usage
sum(container_memory_working_set_bytes{namespace="ml-serving"}) by (pod)

# PVC usage
kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes * 100
```

### 3.4 Grafana Dashboards

**Access Grafana:**

```bash
kubectl port-forward service/prometheus-grafana 3000:80 -n monitoring

# Get admin password
kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 -d
```

**Import pre-built dashboards:**

1. Go to Dashboards → Import
2. Enter dashboard ID:
   - **6417**: Kubernetes Cluster Monitoring
   - **7249**: Kubernetes Cluster
   - **12239**: NVIDIA DCGM Exporter Dashboard
   - **13770**: Kubernetes / Views / Pods

**Create custom dashboard for ML inference:**

```json
{
  "dashboard": {
    "title": "ML Inference Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {"expr": "sum(rate(model_predictions_total[5m]))"}
        ]
      },
      {
        "title": "Error Rate %",
        "targets": [
          {"expr": "sum(rate(model_predictions_total{status='error'}[5m])) / sum(rate(model_predictions_total[5m])) * 100"}
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {"expr": "histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))"}
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {"expr": "DCGM_FI_DEV_GPU_UTIL{namespace='ml-serving'}"}
        ]
      }
    ]
  }
}
```

## 4. Alerting

### 4.1 Prometheus AlertManager

**Configure alerts:**

```yaml
# prometheus-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-inference-alerts
  namespace: monitoring
spec:
  groups:
  - name: ml-inference
    interval: 30s
    rules:
    # High error rate
    - alert: HighErrorRate
      expr: |
        sum(rate(model_predictions_total{status="error"}[5m])) /
        sum(rate(model_predictions_total[5m])) * 100 > 5
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate in ML inference"
        description: "Error rate is {{ $value }}% (threshold: 5%)"

    # High latency
    - alert: HighInferenceLatency
      expr: |
        histogram_quantile(0.95,
          rate(model_prediction_duration_seconds_bucket[5m])
        ) > 1.0
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High inference latency"
        description: "P95 latency is {{ $value }}s (threshold: 1s)"

    # GPU OOM risk
    - alert: GPUMemoryHigh
      expr: |
        DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100 > 95
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "GPU memory nearly full"
        description: "GPU {{ $labels.gpu }} memory usage is {{ $value }}%"

    # Training job stuck
    - alert: TrainingJobStuck
      expr: |
        time() - training_last_update_timestamp_seconds > 600
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Training job appears stuck"
        description: "No updates for {{ $value }}s"

    # Pod CrashLooping
    - alert: PodCrashLooping
      expr: |
        rate(kube_pod_container_status_restarts_total{namespace="ml-serving"}[15m]) > 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Pod {{ $labels.pod }} is crash looping"
        description: "Restart count increased in last 15 minutes"
```

**Configure AlertManager:**

```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-prometheus-kube-prometheus-alertmanager
  namespace: monitoring
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m

    route:
      group_by: ['alertname', 'severity']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'slack-notifications'
      routes:
      - match:
          severity: critical
        receiver: 'pagerduty'

    receivers:
    - name: 'slack-notifications'
      slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#ml-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'pagerduty'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

## 5. Distributed Tracing

### 5.1 OpenTelemetry for ML Pipelines

**Install OpenTelemetry:**

```bash
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm install opentelemetry-operator open-telemetry/opentelemetry-operator \
  --namespace tracing \
  --create-namespace
```

**Instrument Python application:**

```python
# inference_server.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(endpoint="http://opentelemetry-collector:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace inference request
@app.route('/predict', methods=['POST'])
def predict():
    with tracer.start_as_current_span("model_inference") as span:
        span.set_attribute("model.name", "bert")

        # Preprocessing
        with tracer.start_as_current_span("preprocess"):
            input_data = preprocess(request.data)

        # Inference
        with tracer.start_as_current_span("inference"):
            result = model.predict(input_data)

        # Postprocessing
        with tracer.start_as_current_span("postprocess"):
            output = postprocess(result)

        span.set_attribute("prediction.result", output)
        return jsonify(output)
```

**View traces in Jaeger:**

```bash
helm install jaeger jaegertracing/jaeger \
  --namespace tracing \
  --set collector.service.otlp.grpc.enabled=true

kubectl port-forward service/jaeger-query 16686:16686 -n tracing
# Access at http://localhost:16686
```

## 6. Production Best Practices

### 6.1 Observability Checklist

**For ML Training:**
- ✅ Log training progress (epoch, loss, metrics)
- ✅ Track GPU utilization and memory
- ✅ Monitor data loading time
- ✅ Alert on job failures or stuck jobs
- ✅ Export checkpoints with version tags

**For ML Inference:**
- ✅ Track request rate, error rate, latency (RED metrics)
- ✅ Monitor queue depth and concurrency
- ✅ Track model version served
- ✅ Implement health and readiness probes
- ✅ Alert on SLA violations (p99 latency > threshold)

### 6.2 Debugging Runbook

**Issue: Pod stuck in Pending**
1. `kubectl describe pod` → Check Events
2. Look for: Insufficient resources, unbound PVC, node selector mismatch
3. Solution: Scale cluster, fix PVC, adjust selectors

**Issue: Pod CrashLoopBackOff**
1. `kubectl logs --previous` → Check logs from crashed container
2. Common causes: OOM, missing dependencies, wrong command
3. Solution: Increase memory, fix Dockerfile, correct entrypoint

**Issue: Service not accessible**
1. `kubectl get endpoints` → Check if service has endpoints
2. `kubectl describe service` → Check selector matches pod labels
3. Test pod directly: `kubectl exec ... -- curl localhost:8080`

**Issue: High latency**
1. Check `kubectl top pods` → CPU/memory bottleneck?
2. Check GPU utilization → `nvidia-smi`
3. Check HPA → Is autoscaling working?
4. Check application logs → Slow queries? External API calls?

## 7. Summary

### Key Takeaways

✅ **kubectl debugging:**
- `kubectl describe` for events and status
- `kubectl logs` for application logs
- `kubectl exec` for interactive debugging
- `kubectl top` for resource usage

✅ **Logging:**
- Use structured logging (JSON) for easier parsing
- Aggregate logs with Loki or Fluentd
- Query logs with LogQL or Elasticsearch

✅ **Metrics:**
- Instrument applications with Prometheus client
- Monitor RED metrics (Rate, Errors, Duration)
- Track ML-specific metrics (GPU, loss, throughput)
- Create custom Grafana dashboards

✅ **Alerting:**
- Define alerts for critical issues (high error rate, OOM, crashes)
- Use AlertManager for routing (Slack, PagerDuty)
- Set appropriate thresholds and for-durations

✅ **Distributed tracing:**
- Use OpenTelemetry for tracing ML pipelines
- Visualize with Jaeger or Zipkin
- Identify bottlenecks in multi-step workflows

## Self-Check Questions

1. How would you check why a pod is stuck in Pending state?
2. What command retrieves logs from a previously crashed container?
3. What are the three pillars of observability?
4. How do you expose custom metrics from a Python application?
5. What query shows P95 latency in Prometheus?
6. How would you alert on high GPU memory usage?
7. What's the difference between liveness and readiness probes?
8. How do you debug a 502 error from a Kubernetes service?

## Additional Resources

- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Prometheus Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Loki LogQL](https://grafana.com/docs/loki/latest/logql/)

---

**Congratulations!** You've completed Module 04: Kubernetes Fundamentals. You now have the skills to deploy, manage, monitor, and troubleshoot ML workloads on Kubernetes.
