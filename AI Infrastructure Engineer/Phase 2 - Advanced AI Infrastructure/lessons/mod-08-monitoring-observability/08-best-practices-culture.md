# Lesson 08: Best Practices and Observability Culture

## Learning Objectives
By the end of this lesson, you will be able to:
- Implement observability best practices for AI infrastructure
- Build an observability culture within your organization
- Design cost-effective observability strategies
- Optimize monitoring performance and reduce overhead
- Create effective SLOs (Service Level Objectives) for ML systems
- Conduct blameless postmortems and learn from incidents
- Scale observability for large ML platforms

## Prerequisites
- Completion of Lessons 01-07 (Comprehensive observability knowledge)
- Understanding of organizational dynamics
- Experience with production ML systems

## Introduction

Observability is not just about tools—it's about **culture**, **practices**, and **mindset**. The most sophisticated monitoring stack is useless if teams don't know how to use it, if metrics are ignored, or if alerts cause fatigue rather than action.

This lesson covers the "soft" skills and organizational practices that make observability effective.

---

## 1. Observability Best Practices

### Principle 1: Design for Observability from Day One

**Don't:**
```python
# No instrumentation
def predict(data):
    result = model.predict(data)
    return result
```

**Do:**
```python
# Comprehensive instrumentation
def predict(data):
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("input.size", len(data))

        start_time = time.time()

        try:
            result = model.predict(data)

            latency = time.time() - start_time
            REQUEST_LATENCY.observe(latency)
            REQUEST_COUNT.labels(status="success").inc()

            span.set_attribute("predictions.count", len(result))
            logger.info(
                "Prediction completed",
                latency_ms=latency * 1000,
                batch_size=len(data)
            )

            return result

        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            span.record_exception(e)
            logger.error("Prediction failed", exc_info=True)
            raise
```

### Principle 2: Make Metrics, Logs, and Traces Correlatable

**Use consistent identifiers:**

```python
import uuid
from contextvars import ContextVar

# Request ID context
request_id_ctx = ContextVar("request_id", default=None)

@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_ctx.set(request_id)

    # Add to traces
    span = trace.get_current_span()
    span.set_attribute("request.id", request_id)

    # Add to logs
    structlog.contextvars.bind_contextvars(request_id=request_id)

    # Add to metrics (as label)
    with REQUEST_DURATION.labels(request_id=request_id[:8]).time():
        response = await call_next(request)

    return response
```

**Result:**
- Metrics tagged with request_id
- Logs include request_id
- Traces include request_id
- → Easy to jump from one to another in Grafana

### Principle 3: Instrument at Multiple Levels

```
Application Level:
├─ Business metrics (predictions/sec, model accuracy)
├─ Application metrics (request rate, latency, errors)
└─ Code metrics (function duration, cache hits)

Infrastructure Level:
├─ Service metrics (health checks, replicas)
├─ Container metrics (CPU, memory, network)
└─ Node metrics (disk I/O, network bandwidth)

Hardware Level:
├─ GPU metrics (utilization, memory, temperature)
├─ CPU metrics (load, temperature, throttling)
└─ Storage metrics (IOPS, throughput, latency)
```

### Principle 4: Monitor the Monitors

Your observability stack needs observability!

```yaml
# Prometheus self-monitoring
- alert: PrometheusDown
  expr: up{job="prometheus"} == 0
  for: 5m
  labels:
    severity: critical

- alert: PrometheusHighCardinality
  expr: prometheus_tsdb_symbol_table_size_bytes > 1e9
  for: 10m
  labels:
    severity: warning

- alert: PrometheusScrapeFailures
  expr: rate(prometheus_target_scrapes_failed_total[5m]) > 0.1
  for: 10m

# Grafana monitoring
- alert: GrafanaDown
  expr: up{job="grafana"} == 0
  for: 5m

# Loki monitoring
- alert: LokiIngestionRate
  expr: rate(loki_distributor_bytes_received_total[1m]) < 100
  for: 10m
```

---

## 2. Cost-Effective Observability

### Metric Cardinality Management

**Bad (High Cardinality):**
```python
# Millions of unique combinations!
REQUEST_COUNT.labels(
    user_id=user_id,           # 1M users
    request_id=request_id,     # Unique per request
    timestamp=str(time.time()) # Always unique
).inc()
```

**Good (Controlled Cardinality):**
```python
# Reasonable cardinality
REQUEST_COUNT.labels(
    service="ml-serving",     # 10 services
    model=model_name,         # 20 models
    status=status_code,       # 5-10 statuses
    region=region             # 3-5 regions
).inc()
# Total: 10 × 20 × 10 × 5 = 10,000 series
```

### Log Sampling

```python
import random

class SampledLogger:
    """Logger with sampling for high-volume logs"""

    def __init__(self, logger, sample_rate=0.1):
        self.logger = logger
        self.sample_rate = sample_rate

    def debug(self, msg, **kwargs):
        # Sample debug logs (keep 10%)
        if random.random() < self.sample_rate:
            self.logger.debug(msg, **kwargs)

    def info(self, msg, **kwargs):
        # Always log INFO and above
        self.logger.info(msg, **kwargs)

# Usage
logger = SampledLogger(structlog.get_logger(), sample_rate=0.1)
logger.debug("Processing item")  # Only 10% logged
logger.info("Batch completed")   # Always logged
```

### Retention Policies

```yaml
# Prometheus retention
storage:
  tsdb:
    retention:
      time: 30d      # 30 days recent data
      size: 50GB     # Or 50GB, whichever comes first

# Loki retention by stream
limits_config:
  retention_period: 168h  # 7 days

# Per-tenant overrides
overrides:
  "production":
    retention_period: 720h  # 30 days
  "development":
    retention_period: 72h   # 3 days
```

### Aggregation and Downsampling

```yaml
# Prometheus recording rules (pre-aggregate)
groups:
  - name: aggregations
    interval: 1m
    rules:
      # Instead of querying raw metrics, use pre-aggregated
      - record: job:model_requests:rate5m
        expr: sum(rate(model_requests_total[5m])) by (job, model_name)

      - record: job:model_latency:p99
        expr: histogram_quantile(0.99, sum(rate(duration_bucket[5m])) by (job, le))
```

### Cost Optimization Checklist

- [ ] Set appropriate retention periods (not infinite)
- [ ] Use recording rules for expensive queries
- [ ] Implement log sampling for verbose logs
- [ ] Control metric cardinality (avoid high-cardinality labels)
- [ ] Use object storage (S3) instead of local disks
- [ ] Archive old data to cheaper storage
- [ ] Monitor your monitoring costs
- [ ] Delete unused metrics and dashboards

---

## 3. Service Level Objectives (SLOs)

### SLI, SLO, SLA Definitions

| Term | Definition | Example |
|------|------------|---------|
| **SLI** (Service Level Indicator) | Quantitative measure of service | Request latency, error rate |
| **SLO** (Service Level Objective) | Target value for SLI | 99.9% of requests < 200ms |
| **SLA** (Service Level Agreement) | Contract with consequences | 99.95% uptime or refund |

### Defining SLOs for ML Systems

**Example 1: Model Serving SLO**

```yaml
service: bert-model-serving
slos:
  - name: availability
    description: "Service is reachable and responding"
    sli: |
      sum(rate(http_requests_total{status!~"5.."}[5m])) /
      sum(rate(http_requests_total[5m]))
    target: 0.999  # 99.9%
    window: 30d

  - name: latency
    description: "P99 latency under 500ms"
    sli: |
      histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
      )
    target: 0.5  # 500ms
    window: 30d

  - name: quality
    description: "Model accuracy above threshold"
    sli: model_accuracy{dataset="validation"}
    target: 0.92  # 92%
    window: 7d
```

**Example 2: Data Pipeline SLO**

```yaml
service: feature-engineering-pipeline
slos:
  - name: freshness
    description: "Data updated within 2 hours"
    sli: time() - data_last_updated_timestamp
    target: 7200  # 2 hours
    window: 7d

  - name: completeness
    description: "All expected records present"
    sli: actual_row_count / expected_row_count
    target: 0.999  # 99.9%
    window: 7d

  - name: data_quality
    description: "Data quality checks passing"
    sli: data_quality_checks_passed / data_quality_checks_total
    target: 0.95  # 95%
    window: 7d
```

### Error Budget

```
Error Budget = (1 - SLO) × Total Requests

Example:
- SLO: 99.9% availability
- Error budget: 0.1%
- Over 30 days: 0.1% × 30 days = 43 minutes of downtime allowed
```

**Error Budget Dashboard:**

```promql
# Burn rate (how fast we're consuming error budget)
(1 - sli_value) / (1 - slo_target)

# Remaining error budget
1 - (errors_in_window / (error_budget × requests_in_window))
```

---

## 4. Incident Response and Postmortems

### Blameless Postmortem Template

```markdown
# Incident Postmortem: [Title]

**Date:** 2025-10-15
**Duration:** 2 hours 15 minutes
**Severity:** P1 (Critical)
**Authors:** [Names]

## Summary
Brief description of what happened and impact.

## Impact
- **Users affected:** 50,000 (15% of user base)
- **Services affected:** ML model serving, API gateway
- **SLO impact:** Availability dropped to 95.2% (target: 99.9%)
- **Error budget consumed:** 2 weeks worth in 2 hours

## Root Cause
GPU memory leak in model serving container caused OOM crashes.

## Timeline (all times UTC)

| Time | Event |
|------|-------|
| 14:00 | Deployment of model v2.1 |
| 14:15 | First GPU OOM errors in logs |
| 14:30 | Alert: HighModelLatency fires |
| 14:45 | Alert: ModelServingDown fires |
| 14:50 | Engineer begins investigation |
| 15:00 | Root cause identified (memory leak) |
| 15:10 | Rollback initiated |
| 15:30 | Service restored |
| 16:15 | All metrics back to normal |

## What Went Well
- ✅ Alerts fired correctly
- ✅ Runbook was followed
- ✅ Rollback process worked smoothly
- ✅ Communication was clear and timely

## What Went Wrong
- ❌ Memory leak not caught in staging
- ❌ No memory limit on containers
- ❌ Alert delay (15 minutes before firing)
- ❌ Rollback took too long (40 minutes)

## Action Items

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| Add memory profiling to CI/CD | @eng1 | 2025-10-20 | P0 |
| Set container memory limits | @eng2 | 2025-10-18 | P0 |
| Reduce alert threshold | @eng3 | 2025-10-17 | P1 |
| Automate rollback process | @eng1 | 2025-10-25 | P1 |
| Add memory monitoring to dashboard | @eng2 | 2025-10-22 | P2 |

## Lessons Learned
1. Staging environment != production load
2. Resource limits are critical safety nets
3. Faster rollback process needed
4. Memory monitoring gaps identified
```

---

## 5. Building Observability Culture

### The Three Pillars of Observability Culture

**1. Psychological Safety**
- No blame for incidents
- Learning from failures
- "What" and "how", not "who"
- Encourage transparency

**2. Continuous Learning**
- Regular incident reviews
- Share learnings across teams
- Invest in training
- Document tribal knowledge

**3. Ownership and Accountability**
- Teams own their services
- On-call rotations
- "You build it, you run it"
- Empowerment to make changes

### Observability Champions

Designate "observability champions" in each team:

**Responsibilities:**
- Advocate for observability best practices
- Review instrumentation in PRs
- Maintain dashboards and alerts
- Lead postmortem discussions
- Share knowledge across teams

### Dashboard Reviews

**Monthly dashboard review process:**

1. **Audit existing dashboards**
   - Are they still used? (check view count)
   - Are they still relevant?
   - Are queries optimized?

2. **Identify gaps**
   - Missing metrics?
   - Blind spots?
   - User feedback?

3. **Cleanup**
   - Delete unused dashboards
   - Fix broken queries
   - Update documentation

### Knowledge Sharing

**Weekly observability office hours:**
- Open forum for questions
- Demo new dashboards
- Share incident learnings
- Discuss best practices

**Internal blog posts:**
- "How we monitor X"
- "Debugging Y with traces"
- "Lessons from incident Z"

---

## 6. Scaling Observability

### Federated Prometheus

```yaml
# Central Prometheus scrapes multiple regional Prometheus servers
global:
  scrape_interval: 60s  # Less frequent for federated metrics

scrape_configs:
  - job_name: 'federate-us-west'
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"ml-.*"}'  # Only scrape ML-related metrics
    static_configs:
      - targets:
          - 'prometheus-us-west:9090'

  - job_name: 'federate-us-east'
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"ml-.*"}'
    static_configs:
      - targets:
          - 'prometheus-us-east:9090'
```

### Multi-Tenancy in Loki

```yaml
# Different retention and limits per tenant
overrides:
  "team-ml":
    ingestion_rate_mb: 20
    ingestion_burst_size_mb: 40
    retention_period: 720h  # 30 days

  "team-data":
    ingestion_rate_mb: 10
    ingestion_burst_size_mb: 20
    retention_period: 168h  # 7 days
```

### Sharding and Replication

```yaml
# Prometheus sharding (scrape different targets)
# prometheus-shard-0.yml
global:
  external_labels:
    shard: '0'

scrape_configs:
  - job_name: 'ml-serving'
    relabel_configs:
      - source_labels: [__address__]
        modulus: 3
        target_label: __tmp_hash
        action: hashmod
      - source_labels: [__tmp_hash]
        regex: '0'
        action: keep

# prometheus-shard-1.yml
# ... regex: '1'

# prometheus-shard-2.yml
# ... regex: '2'
```

---

## Summary

In this lesson, you learned:

✅ Observability best practices (design from day one, correlation, multi-level instrumentation)
✅ Cost-effective observability strategies (cardinality management, sampling, retention)
✅ Defining Service Level Objectives for ML systems
✅ Conducting blameless postmortems and learning from incidents
✅ Building observability culture (psychological safety, continuous learning)
✅ Scaling observability for large platforms (federation, multi-tenancy, sharding)

## Final Thoughts

Observability is a **continuous journey**, not a destination:
- Start small, iterate
- Involve the whole team
- Measure and improve
- Learn from incidents
- Invest in culture, not just tools

**Remember:** The goal is not perfect observability—it's **actionable insights** that help you build better ML systems.

---

## Module 08 Complete!

Congratulations! You've completed the Monitoring & Observability module. You now have the knowledge to:
- Collect metrics with Prometheus
- Visualize with Grafana
- Aggregate logs with ELK/Loki
- Trace requests with Jaeger/Tempo
- Design effective alerts
- Monitor ML models and detect drift
- Build observability culture

## Next Module

**Module 09: Infrastructure as Code** - Learn to automate infrastructure deployment with Terraform and Pulumi.

---

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate
**Prerequisites:** Lessons 01-07, Production experience
