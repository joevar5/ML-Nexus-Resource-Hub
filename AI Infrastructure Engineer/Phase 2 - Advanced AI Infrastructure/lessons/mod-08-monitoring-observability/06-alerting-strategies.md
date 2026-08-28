# Lesson 06: Alerting Strategies and Alert Management

## Learning Objectives
By the end of this lesson, you will be able to:
- Design effective alerting strategies for AI infrastructure
- Configure Prometheus Alertmanager for alert routing
- Create meaningful alert rules for ML systems
- Implement alert routing, grouping, and deduplication
- Set up multiple notification channels (Slack, PagerDuty, email)
- Reduce alert fatigue with proper thresholds and severity levels
- Build on-call workflows and runbooks

## Prerequisites
- Completion of Lessons 01-05 (Observability fundamentals)
- Understanding of Prometheus and PromQL
- Familiarity with incident management concepts

## Introduction

Alerts are automated notifications triggered when systems deviate from expected behavior. For AI infrastructure, effective alerting is critical for:
- Detecting model degradation before users notice
- Identifying GPU failures and resource exhaustion
- Catching data pipeline issues early
- Preventing cascading failures in ML services

However, **bad alerting is worse than no alerting**. Alert fatigue from too many false positives leads teams to ignore critical alerts.

### Principles of Good Alerting

1. **Alert on symptoms, not causes**: Alert when users are impacted
2. **Make alerts actionable**: Every alert should have a clear response
3. **Avoid alert fatigue**: Only alert on what matters
4. **Classify by severity**: Critical vs warning vs informational
5. **Include context**: Provide debugging information in alerts
6. **Test your alerts**: Validate that alerts fire as expected

---

## 1. Alerting Strategy

### The Four Golden Signals (for ML Systems)

**1. Latency**: How long does inference take?
```promql
histogram_quantile(0.99,
  rate(model_inference_duration_seconds_bucket[5m])
) > 1.0
```

**2. Traffic**: How many requests are we serving?
```promql
sum(rate(model_requests_total[5m])) < 10
```

**3. Errors**: What percentage of requests are failing?
```promql
(sum(rate(model_requests_total{status="error"}[5m])) /
 sum(rate(model_requests_total[5m]))) > 0.05
```

**4. Saturation**: How full are our resources?
```promql
avg(gpu_utilization) > 95
avg(gpu_memory_utilization) > 90
```

### Alert Severity Levels

| Severity | When to Use | Response Time | Example |
|----------|-------------|---------------|---------|
| **Critical/Page** | User-facing impact | Immediate (minutes) | All models down, >50% error rate |
| **Error/High** | Potential user impact | Hours | Single model degraded, GPU failure |
| **Warning** | Needs investigation | Days | High latency, resource saturation |
| **Info** | FYI, no action needed | N/A | Deployment completed, scaling event |

### Alerting vs. Monitoring

**Monitor (don't alert)**:
- Successful requests
- Normal resource usage
- Routine maintenance
- Informational events

**Alert (with urgency)**:
- Service unavailable
- Error rate spike
- Critical resource exhaustion
- Data quality issues

---

## 2. Prometheus Alertmanager

Alertmanager handles alerts sent by Prometheus, providing routing, grouping, inhibition, and silencing.

### Architecture

```
┌───────────────┐
│  Prometheus   │
│  (Evaluates   │
│    Rules)     │
└───────┬───────┘
        │ Alerts
        ↓
┌───────────────┐
│ Alertmanager  │
│  - Route      │
│  - Group      │
│  - Dedupe     │
│  - Silence    │
└───────┬───────┘
        │
        ├─────> Slack
        ├─────> PagerDuty
        ├─────> Email
        └─────> Webhook
```

### Installing Alertmanager

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts:/etc/prometheus/alerts
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - monitoring
    depends_on:
      - alertmanager

volumes:
  alertmanager_data:
  prometheus_data:

networks:
  monitoring:
    driver: bridge
```

### Alertmanager Configuration

**alertmanager.yml:**

```yaml
global:
  # Default notification settings
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

# Templates for notifications
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# Alert routing tree
route:
  # Default receiver if no match
  receiver: 'default'

  # Group alerts by these labels
  group_by: ['alertname', 'cluster', 'service']

  # How long to wait before sending initial notification
  group_wait: 10s

  # How long to wait before sending notification about new alerts
  # added to a group
  group_interval: 10s

  # How often to re-send notifications for unresolved alerts
  repeat_interval: 12h

  # Child routes (evaluated in order)
  routes:
    # Critical alerts go to PagerDuty and Slack
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true  # Also match other routes

    - match:
        severity: critical
      receiver: 'slack-critical'

    # ML-specific alerts
    - match:
        team: ml-platform
      receiver: 'slack-ml-team'
      group_by: ['alertname', 'model_name']
      routes:
        # GPU alerts need immediate attention
        - match_re:
            alertname: '^GPU.*'
          receiver: 'slack-ml-team-urgent'
          repeat_interval: 4h

    # Data pipeline alerts
    - match:
        team: data-engineering
      receiver: 'slack-data-team'

    # Warning alerts (non-critical)
    - match:
        severity: warning
      receiver: 'slack-warnings'
      repeat_interval: 24h

# Inhibition rules (suppress alerts based on other alerts)
inhibit_rules:
  # Inhibit warnings if critical alert is firing
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'cluster', 'service']

  # If entire service is down, don't alert on individual instances
  - source_match:
      alertname: ServiceDown
    target_match_re:
      alertname: '^(HighLatency|HighErrorRate)$'
    equal: ['service']

# Notification receivers
receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#ml-critical-alerts'
        username: 'Alertmanager'
        icon_emoji: ':rotating_light:'
        color: 'danger'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
        send_resolved: true

  - name: 'slack-ml-team'
    slack_configs:
      - channel: '#ml-platform-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'slack-ml-team-urgent'
    slack_configs:
      - channel: '#ml-platform-urgent'
        color: 'warning'
        title: 'GPU ALERT: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Model:* {{ .Labels.model_name }}
          *GPU:* {{ .Labels.gpu_id }}
          *Node:* {{ .Labels.node }}
          *Issue:* {{ .Annotations.description }}
          {{ end }}

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#monitoring-warnings'
        color: 'warning'
        send_resolved: true

  - name: 'slack-data-team'
    slack_configs:
      - channel: '#data-platform-alerts'
```

---

## 3. Alert Rules

### Prometheus Alert Rules

**alerts/ml-platform-alerts.yml:**

```yaml
groups:
  - name: ml_infrastructure
    interval: 30s
    rules:
      # ========== SERVICE AVAILABILITY ==========

      - alert: ModelServingDown
        expr: up{job="ml-serving"} == 0
        for: 1m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "Model serving instance is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
          runbook_url: "https://runbooks.company.com/ModelServingDown"

      - alert: HighModelErrorRate
        expr: |
          (sum(rate(model_requests_total{status="error"}[5m])) by (model_name) /
           sum(rate(model_requests_total[5m])) by (model_name)) > 0.05
        for: 5m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "High error rate for model {{ $labels.model_name }}"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
          runbook_url: "https://runbooks.company.com/HighErrorRate"

      # ========== LATENCY ==========

      - alert: HighInferenceLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(model_inference_duration_seconds_bucket[5m])) by (model_name, le)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "High P99 latency for {{ $labels.model_name }}"
          description: "P99 latency is {{ $value }}s (threshold: 1.0s)"
          runbook_url: "https://runbooks.company.com/HighLatency"

      - alert: CriticalInferenceLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(model_inference_duration_seconds_bucket[5m])) by (model_name, le)
          ) > 5.0
        for: 5m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "CRITICAL: P99 latency for {{ $labels.model_name }}"
          description: "P99 latency is {{ $value }}s (threshold: 5.0s)"

      # ========== GPU RESOURCES ==========

      - alert: GPUHighTemperature
        expr: gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "GPU temperature high on {{ $labels.node }}"
          description: "GPU {{ $labels.gpu_id }} temperature is {{ $value }}°C"

      - alert: GPUCriticalTemperature
        expr: gpu_temperature_celsius > 90
        for: 2m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "CRITICAL: GPU overheating on {{ $labels.node }}"
          description: "GPU {{ $labels.gpu_id }} at {{ $value }}°C - risk of shutdown"
          runbook_url: "https://runbooks.company.com/GPUOverheating"

      - alert: GPUMemoryExhaustion
        expr: |
          (gpu_memory_used_bytes / gpu_memory_total_bytes) > 0.95
        for: 10m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "GPU memory near capacity on {{ $labels.node }}"
          description: "GPU {{ $labels.gpu_id }} memory at {{ $value | humanizePercentage }}"

      - alert: GPUUtilizationLow
        expr: avg(gpu_utilization) by (node, gpu_id) < 20
        for: 30m
        labels:
          severity: info
          team: ml-platform
        annotations:
          summary: "Low GPU utilization on {{ $labels.node }}"
          description: "GPU {{ $labels.gpu_id }} utilization at {{ $value }}% for 30min"

      # ========== TRAINING JOBS ==========

      - alert: TrainingJobStalled
        expr: |
          increase(training_samples_processed[10m]) == 0
        for: 10m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "Training job {{ $labels.job_name }} appears stalled"
          description: "No progress in 10 minutes"

      - alert: TrainingLossDivergence
        expr: training_loss > 1000
        for: 5m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "Training loss diverging for {{ $labels.job_name }}"
          description: "Loss value: {{ $value }}"

      # ========== DATA PIPELINES ==========

      - alert: DataPipelineFailed
        expr: airflow_dag_status{status="failed"} == 1
        for: 5m
        labels:
          severity: critical
          team: data-engineering
        annotations:
          summary: "Airflow DAG {{ $labels.dag_id }} failed"
          description: "Pipeline has been in failed state for 5 minutes"

      - alert: DataQualityCheckFailed
        expr: data_quality_check_passed == 0
        for: 1m
        labels:
          severity: critical
          team: data-engineering
        annotations:
          summary: "Data quality check failed: {{ $labels.check_name }}"
          description: "Dataset: {{ $labels.dataset }}"

      # ========== RESOURCE SATURATION ==========

      - alert: HighCPUUsage
        expr: |
          100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance) * 100) > 90
        for: 15m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage at {{ $value }}%"

      - alert: HighMemoryUsage
        expr: |
          (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
          node_memory_MemTotal_bytes > 0.90
        for: 10m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage at {{ $value | humanizePercentage }}"

      # ========== DISK SPACE ==========

      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.15
        for: 10m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "Disk space low on {{ $labels.instance }}"
          description: "Only {{ $value | humanizePercentage }} available on {{ $labels.mountpoint }}"

      - alert: DiskSpaceCritical
        expr: |
          (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.05
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "CRITICAL: Disk space critically low on {{ $labels.instance }}"
          description: "Only {{ $value | humanizePercentage }} available on {{ $labels.mountpoint }}"
```

---

## 4. Notification Templates

**alertmanager/templates/slack.tmpl:**

```
{{ define "slack.default.title" }}
[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .GroupLabels.SortedPairs.Values | join " " }}
{{ end }}

{{ define "slack.default.text" }}
{{ range .Alerts }}
*Alert:* {{ .Labels.alertname }}
*Severity:* {{ .Labels.severity }}
*Summary:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}
{{ if .Annotations.runbook_url }}*Runbook:* {{ .Annotations.runbook_url }}{{ end }}
*Labels:*
{{ range .Labels.SortedPairs }} • {{ .Name }}: `{{ .Value }}`
{{ end }}
{{ end }}
{{ end }}
```

---

## 5. Best Practices

### 1. Make Alerts Actionable

**Bad Alert:**
```yaml
- alert: SomethingWrong
  expr: metric > threshold
  annotations:
    summary: "Metric is high"
```

**Good Alert:**
```yaml
- alert: HighModelLatency
  expr: histogram_quantile(0.99, rate(latency_bucket[5m])) > 1.0
  annotations:
    summary: "Model {{ $labels.model_name }} P99 latency > 1s"
    description: |
      Current P99 latency: {{ $value }}s
      Check:
      1. GPU utilization and memory
      2. Batch size and concurrency
      3. Recent deployments
    runbook_url: "https://runbooks.company.com/HighLatency"
    dashboard_url: "https://grafana.company.com/d/model-perf"
```

### 2. Use Appropriate Thresholds

```yaml
# Too sensitive - alert fatigue
- alert: AnyError
  expr: errors > 0
  for: 1m

# Better - tolerate occasional errors
- alert: HighErrorRate
  expr: (errors / requests) > 0.05
  for: 5m
```

### 3. Group Related Alerts

```yaml
route:
  group_by: ['alertname', 'model_name', 'cluster']
  group_wait: 30s
  group_interval: 5m
```

### 4. Avoid Alert Storms

Use inhibition rules:

```yaml
inhibit_rules:
  - source_match:
      alertname: DatacenterDown
    target_match_re:
      alertname: '.*'
    equal: ['datacenter']
```

---

## 6. On-Call Workflows

### Runbook Example

**Runbook: HighModelLatency**

```markdown
# High Model Latency Runbook

## Severity
Warning / Critical (depending on latency)

## Impact
Users experiencing slow inference responses

## Diagnosis

1. Check GPU utilization:
   ```
   avg(gpu_utilization{model="$MODEL"}) by (node, gpu_id)
   ```

2. Check batch sizes:
   ```
   histogram_quantile(0.95, rate(batch_size_bucket[5m]))
   ```

3. Check recent deployments:
   ```bash
   kubectl get events --namespace ml-serving
   ```

4. Check model health:
   ```
   curl http://model-serving/health
   ```

## Common Causes

1. **GPU saturation**: Utilization > 95%
2. **Large batch sizes**: Batch > optimal size
3. **Memory pressure**: GPU memory > 90%
4. **Network issues**: Cross-AZ latency
5. **Model version issue**: Recent deployment

## Resolution

1. **Immediate**:
   - Scale up replicas if traffic spike
   - Reduce batch size if GPU saturated
   - Rollback if recent deployment

2. **Short-term**:
   - Add GPU nodes if sustained high load
   - Optimize model if consistently slow

3. **Long-term**:
   - Consider model quantization
   - Implement request batching
   - Add caching layer

## Escalation

- L1: ML Platform team (@ml-oncall)
- L2: ML Infrastructure lead (@ml-lead)
- L3: VP Engineering (@vp-eng)
```

---

## Summary

In this lesson, you learned:

✅ Designing effective alerting strategies (Four Golden Signals)
✅ Configuring Prometheus Alertmanager for routing and deduplication
✅ Creating meaningful alert rules for ML infrastructure
✅ Setting up notification channels (Slack, PagerDuty, email)
✅ Reducing alert fatigue with proper thresholds and grouping
✅ Building on-call workflows and runbooks

## Next Steps

- **Lesson 07**: ML-specific monitoring and model observability
- **Practice**: Create alert rules for your ML infrastructure
- **Exercise**: Design a complete alerting strategy

## Additional Resources

- [Prometheus Alerting Documentation](https://prometheus.io/docs/alerting/latest/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Alert Fatigue Guide](https://grafana.com/blog/2022/02/14/how-to-reduce-alert-fatigue/)

---

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate
**Prerequisites:** Lessons 01-05, Prometheus basics
