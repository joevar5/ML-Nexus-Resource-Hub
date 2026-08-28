# Lesson 01: Introduction to Observability

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain the difference between monitoring and observability
2. Understand the three pillars of observability: metrics, logs, and traces
3. Describe the OpenTelemetry standard and its components
4. Design observable systems from the ground up
5. Choose appropriate observability tools for different use cases

## Introduction

In modern distributed systems, especially ML infrastructure, understanding system behavior is crucial. Observability goes beyond traditional monitoring to help you understand **why** systems behave the way they do, not just **what** is happening.

## Monitoring vs Observability

### Traditional Monitoring

```
Traditional Monitoring:
┌──────────────────────────────────────────────┐
│ "Known Unknowns"                             │
│                                              │
│ • Pre-defined dashboards                    │
│ • Fixed metrics (CPU, memory, disk)         │
│ • Static alerts                             │
│ • Assumes you know what to look for        │
│                                              │
│ Example: "CPU > 80% → Alert"                │
└──────────────────────────────────────────────┘

Works well for:
✓ Stable, well-understood systems
✓ Known failure modes
✓ Infrastructure metrics

Fails for:
✗ Novel failures
✗ Complex distributed systems
✗ Debugging unknown issues
```

### Modern Observability

```
Observability:
┌──────────────────────────────────────────────┐
│ "Unknown Unknowns"                           │
│                                              │
│ • Ask arbitrary questions                   │
│ • Slice data any way you need              │
│ • Understand system behavior               │
│ • Debug issues you've never seen           │
│                                              │
│ Example: "Why did this specific user        │
│          request fail 5 minutes ago?"       │
└──────────────────────────────────────────────┘

Enables:
✓ Root cause analysis
✓ Debugging complex issues
✓ Understanding system behavior
✓ Proactive problem detection
```

### Key Differences

| Aspect | Monitoring | Observability |
|--------|------------|---------------|
| **Focus** | Known metrics | System internals |
| **Questions** | Pre-defined | Ad-hoc, arbitrary |
| **Scope** | "Is it working?" | "Why did it fail?" |
| **Data** | Aggregated metrics | High-cardinality events |
| **Time** | Real-time status | Historical analysis |
| **Approach** | Outside-in | Inside-out |

## The Three Pillars of Observability

### 1. Metrics

**Numerical measurements aggregated over time**

```python
# Examples of metrics
request_count = 1523           # Counter
response_time_ms = 45.3        # Gauge
error_rate_percent = 0.1       # Derived metric
cpu_utilization = 67.5         # Gauge

# Metrics are cheap to collect and store
# But lose individual event details
```

**Use Cases**:
- System health monitoring (CPU, memory, disk)
- Application performance (latency, throughput)
- Business metrics (requests/sec, revenue)
- Alerting and SLOs

**Characteristics**:
- ✓ Low storage cost
- ✓ Fast queries
- ✓ Good for trends and alerts
- ✗ Lose individual event context
- ✗ Limited dimensionality (cardinality)

---

### 2. Logs

**Discrete events with context**

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "ERROR",
  "message": "Failed to process payment",
  "user_id": "user_12345",
  "transaction_id": "txn_abc123",
  "amount": 99.99,
  "error_code": "INSUFFICIENT_FUNDS",
  "trace_id": "trace_xyz789"
}
```

**Use Cases**:
- Debugging specific failures
- Audit trails and compliance
- Understanding event sequences
- Detailed error analysis

**Characteristics**:
- ✓ Rich context per event
- ✓ Full detail available
- ✓ Searchable
- ✗ Expensive to store long-term
- ✗ Slow to query at scale
- ✗ Noise without proper filtering

---

### 3. Traces

**Request flow across distributed systems**

```
Trace: User Login Request
┌─────────────────────────────────────────────────┐
│ Frontend Service          [50ms]                │
│   ↓                                             │
│ API Gateway               [10ms]                │
│   ↓                                             │
│ Auth Service              [200ms]               │
│   ├─ DB Query            [180ms] ← Slow!       │
│   └─ Cache Check         [5ms]                 │
│   ↓                                             │
│ User Service              [30ms]                │
│                                                 │
│ Total: 290ms                                    │
└─────────────────────────────────────────────────┘

The slow DB query is the bottleneck!
```

**Use Cases**:
- Finding bottlenecks in distributed systems
- Understanding service dependencies
- Performance optimization
- Request-level debugging

**Characteristics**:
- ✓ Shows request flow
- ✓ Identifies bottlenecks
- ✓ Service dependency mapping
- ✗ Expensive to collect (sampling needed)
- ✗ Complex to implement
- ✗ High cardinality

## Combining the Three Pillars

Real-world debugging example:

```
1. METRICS alert: API latency > 500ms (anomaly detected)
   └─> "Something is slow, but what?"

2. TRACES show: Auth service taking 200ms (vs usual 50ms)
   └─> "Auth service is slow, but why?"

3. LOGS reveal: Database connection pool exhausted
   └─> "Root cause: connection pool too small"

Solution: Increase connection pool size
Verification: Metrics show latency back to normal
```

## OpenTelemetry

OpenTelemetry is the industry standard for observability:

```
┌──────────────────────────────────────────┐
│          OpenTelemetry                   │
│                                          │
│  ┌────────────┐  ┌────────────┐         │
│  │   API      │  │    SDK     │         │
│  │ (Standard) │  │ (Impl)     │         │
│  └────────────┘  └────────────┘         │
│         ↓              ↓                 │
│  ┌─────────────────────────────┐        │
│  │   Auto-instrumentation      │        │
│  │   (No code changes)         │        │
│  └─────────────────────────────┘        │
│         ↓                                │
│  ┌─────────────────────────────┐        │
│  │   Exporters                 │        │
│  │   (Send to backends)        │        │
│  └─────────────────────────────┘        │
└──────────────────────────────────────────┘

Backends: Prometheus, Jaeger, Datadog, etc.
```

### OpenTelemetry Components

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

# 1. Traces
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process_request"):
    # Your code here
    result = process_data()

# 2. Metrics
meter = metrics.get_meter(__name__)
request_counter = meter.create_counter(
    "requests_total",
    description="Total requests"
)
request_counter.add(1, {"endpoint": "/api/predict"})

# 3. Context Propagation (automatic)
# Trace context is automatically propagated across services
```

## Designing Observable Systems

### Principles of Observability

**1. High Cardinality**
```python
# BAD: Low cardinality (limited debugging)
log("Request failed")

# GOOD: High cardinality (rich context)
log({
    "event": "request_failed",
    "user_id": user_id,
    "endpoint": endpoint,
    "status_code": 500,
    "latency_ms": 245,
    "error": str(e),
    "trace_id": trace_id
})
```

**2. Structured Data**
```python
# BAD: Unstructured string
logger.info(f"User {user_id} made purchase of ${amount}")

# GOOD: Structured JSON
logger.info("purchase_completed", extra={
    "user_id": user_id,
    "amount": amount,
    "currency": "USD",
    "items_count": len(items)
})
```

**3. Context Propagation**
```python
# Trace ID propagates through entire request
import uuid

trace_id = str(uuid.uuid4())

# Include in all logs for this request
logger = logging.LoggerAdapter(logger, {"trace_id": trace_id})

# Forward to downstream services
headers = {"X-Trace-ID": trace_id}
response = requests.post(url, headers=headers)
```

**4. Sampling Strategy**
```python
# Sample 100% of errors, 1% of successes
def should_trace(response):
    if response.status_code >= 500:
        return True  # Always trace errors
    elif response.status_code >= 400:
        return random.random() < 0.1  # 10% of client errors
    else:
        return random.random() < 0.01  # 1% of success

if should_trace(response):
    send_trace_to_backend(trace)
```

### Instrumentation Strategy

```python
# Automatic instrumentation (preferred)
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

FlaskInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()

# Manual instrumentation (when needed)
@app.route('/api/predict')
def predict():
    with tracer.start_as_current_span("model_inference"):
        # Span automatically includes timing
        result = model.predict(data)
    return result
```

## Observable ML Systems

ML systems have unique observability needs:

```
ML System Observability:
┌──────────────────────────────────────────────┐
│                                              │
│  Infrastructure Metrics:                    │
│  • GPU utilization, memory                  │
│  • Training throughput                      │
│  • Data pipeline latency                    │
│                                              │
│  Model Metrics:                             │
│  • Inference latency                        │
│  • Prediction confidence                    │
│  • Model accuracy (online)                  │
│  • Feature distributions                    │
│                                              │
│  Data Quality:                              │
│  • Missing features                         │
│  • Out-of-range values                      │
│  • Data drift                               │
│  • Schema violations                        │
│                                              │
│  Business Metrics:                          │
│  • Predictions/second                       │
│  • Cost per prediction                      │
│  • Model version usage                      │
│  • A/B test metrics                         │
└──────────────────────────────────────────────┘
```

### Example: ML Inference Observability

```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Metrics
predictions_total = Counter(
    'ml_predictions_total',
    'Total predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency',
    ['model_version']
)

prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence score',
    ['model_version']
)

# Instrumented prediction function
def predict(features, model_version='v1'):
    start = time.time()

    try:
        # Trace the prediction
        with tracer.start_as_current_span("model_predict"):
            result = model.predict(features)

        # Record metrics
        latency = time.time() - start
        prediction_latency.labels(model_version=model_version).observe(latency)
        prediction_confidence.labels(model_version=model_version).observe(result.confidence)
        predictions_total.labels(model_version=model_version, status='success').inc()

        # Structured logging
        logger.info("prediction_completed", extra={
            "model_version": model_version,
            "latency_ms": latency * 1000,
            "confidence": result.confidence,
            "features_count": len(features)
        })

        return result

    except Exception as e:
        predictions_total.labels(model_version=model_version, status='error').inc()

        logger.error("prediction_failed", extra={
            "model_version": model_version,
            "error": str(e),
            "features_count": len(features)
        })
        raise
```

## Tool Selection Guide

### Metrics: When to Use What

```
Prometheus:
✓ Infrastructure metrics
✓ Application metrics
✓ Kubernetes environments
✓ On-premises deployment
✗ Long-term storage (use remote storage)
✗ High cardinality (label explosion)

InfluxDB:
✓ IoT and sensor data
✓ Time-series analytics
✓ High write throughput
✗ Complex queries
✗ Distributed setups

Cloud Providers (CloudWatch, Stackdriver):
✓ Cloud-native applications
✓ Managed services
✓ Easy integration
✗ Vendor lock-in
✗ Cost at scale
```

### Logs: When to Use What

```
ELK/EFK Stack:
✓ Full-text search
✓ Complex queries
✓ On-premises
✓ Full control
✗ Operational complexity
✗ Cost of running Elasticsearch

Loki (Grafana):
✓ Kubernetes environments
✓ Lower cost than ELK
✓ Grafana integration
✗ Limited query capabilities
✗ Newer, less mature

Cloud Providers (CloudWatch Logs):
✓ Cloud-native apps
✓ No infrastructure management
✗ Expensive at scale
✗ Limited query power
```

### Traces: When to Use What

```
Jaeger:
✓ Microservices
✓ Kubernetes
✓ OpenTelemetry compatible
✓ Open source
✗ Operational complexity

Zipkin:
✓ Simple setup
✓ Mature ecosystem
✗ Less active development

Cloud Providers (X-Ray, Cloud Trace):
✓ Cloud-native apps
✓ Automatic instrumentation
✗ Vendor lock-in
✗ Cost
```

## Observability Maturity Model

```
Level 1: Ad-hoc
• No standardization
• Manual log checking
• Reactive debugging

Level 2: Basic Monitoring
• Infrastructure metrics
• Simple dashboards
• Basic alerting

Level 3: Structured Observability
• Metrics + Logs + Traces
• Centralized logging
• Distributed tracing

Level 4: Advanced Observability
• High cardinality data
• Automated anomaly detection
• SLO-based alerting

Level 5: Proactive Intelligence
• AI-powered insights
• Predictive alerting
• Auto-remediation
```

## Practical Exercises

### Exercise 1: Instrumentation Planning

Design observability for this ML API:

```python
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Receives features, returns prediction

    TODO: Add observability
    - What metrics to track?
    - What to log?
    - What to trace?
    """
    data = request.json
    features = extract_features(data)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction})
```

**Your Tasks**:
1. List all metrics to collect
2. Design structured log format
3. Identify spans for tracing
4. Plan sampling strategy

---

### Exercise 2: Observability vs Monitoring

Classify these as Monitoring or Observability:

1. Dashboard showing average API latency
2. Searching logs for all requests from user_123
3. Alert when disk usage > 90%
4. Trace showing which service is slow for a specific request
5. Graph of CPU usage over last 24 hours
6. Filtering logs by custom attribute added 5 minutes ago

<details>
<summary>Answers</summary>

1. Monitoring (predefined metric)
2. Observability (arbitrary query)
3. Monitoring (known threshold)
4. Observability (request-level debugging)
5. Monitoring (standard metric)
6. Observability (ad-hoc dimension)
</details>

---

### Exercise 3: High Cardinality Design

Fix this low-cardinality logging:

```python
# BAD: Low cardinality
if error:
    logger.error("Request failed")
```

Rewrite with high cardinality.

## Summary

In this lesson, you learned:

1. **Monitoring vs Observability**: Monitoring tracks known issues, observability helps debug unknown issues
2. **Three Pillars**: Metrics (trends), Logs (events), Traces (request flow)
3. **OpenTelemetry**: Industry standard for observability instrumentation
4. **Observable Design**: High cardinality, structured data, context propagation
5. **ML-Specific**: Model performance, data quality, business metrics
6. **Tool Selection**: Choose based on use case, scale, and environment

## Key Takeaways

- **Observability is about understanding system behavior**, not just monitoring health
- **All three pillars work together** - metrics alert, traces narrow down, logs explain
- **High cardinality is essential** for debugging complex issues
- **Structured data enables** arbitrary queries and analysis
- **ML systems need specialized observability** beyond traditional infrastructure
- **Start simple, evolve** - don't try to implement everything at once

## Further Reading

- [Observability Engineering (Book)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Charity Majors on Observability](https://charity.wtf/tag/observability/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)

## Next Steps

In the next lesson, **Prometheus Metrics Collection**, we'll dive deep into implementing metrics collection with Prometheus, including metric types, exporters, and PromQL queries.

---

**Ready to build observable systems? Let's start collecting metrics!**
