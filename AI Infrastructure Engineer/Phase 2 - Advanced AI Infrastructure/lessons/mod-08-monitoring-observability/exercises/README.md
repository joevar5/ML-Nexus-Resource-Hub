# Module 08: Monitoring & Observability - Exercises

These hands-on exercises will help you master monitoring and observability for ML infrastructure.

## Exercise 1: Prometheus Metrics Setup

**Difficulty:** Beginner
**Duration:** 30 minutes

### Objective
Set up basic Prometheus metrics for a Python application.

### Tasks
1. Install prometheus-client library
2. Create Counter, Gauge, and Histogram metrics
3. Expose metrics endpoint on /metrics
4. Scrape metrics with Prometheus
5. View metrics in Prometheus UI

### Starter Code
```python
# exercises/01-prometheus-metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import random

# TODO: Define metrics
# requests_total = Counter(...)
# request_duration = Histogram(...)
# active_users = Gauge(...)

def process_request():
    """Simulate request processing"""
    # TODO: Increment counter
    # TODO: Record duration
    # TODO: Update gauge
    time.sleep(random.uniform(0.1, 0.5))

if __name__ == '__main__':
    # TODO: Start metrics server
    # TODO: Process some requests
    pass
```

### Success Criteria
- [ ] Metrics endpoint returns valid Prometheus format
- [ ] Counters increment correctly
- [ ] Histograms record latency
- [ ] Gauges update in real-time
- [ ] Prometheus successfully scrapes metrics

---

## Exercise 2: Grafana Dashboard Creation

**Difficulty:** Intermediate
**Duration:** 45 minutes

### Objective
Create a comprehensive Grafana dashboard for ML model monitoring.

### Tasks
1. Connect Grafana to Prometheus
2. Create dashboard with panels for:
   - Request rate (queries/sec)
   - Error rate (%)
   - P95/P99 latency
   - Model accuracy over time
   - Resource utilization (CPU, Memory)
3. Add alert rules
4. Export dashboard as JSON

### Dashboard Requirements
- At least 6 panels
- Use appropriate visualization types
- Include variables for filtering
- Set up at least 2 alerts

### Success Criteria
- [ ] Dashboard displays real-time metrics
- [ ] Panels update automatically
- [ ] Alerts trigger when thresholds exceeded
- [ ] Dashboard is shareable via JSON

---

## Exercise 3: Distributed Tracing with Jaeger

**Difficulty:** Intermediate
**Duration:** 60 minutes

### Objective
Implement distributed tracing for an ML inference pipeline.

### Tasks
1. Install OpenTelemetry SDK
2. Instrument code with traces and spans
3. Export traces to Jaeger
4. Visualize trace in Jaeger UI
5. Identify performance bottlenecks

### Starter Code
```python
# exercises/03-distributed-tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# TODO: Import Jaeger exporter

# TODO: Configure tracer
# provider = TracerProvider()
# jaeger_exporter = ...
# provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

tracer = trace.get_tracer(__name__)

def preprocess_data(data):
    # TODO: Create span for preprocessing
    with tracer.start_as_current_span("preprocess"):
        # TODO: Add span attributes
        # Process data
        pass

def run_inference(data):
    # TODO: Create span for inference
    # TODO: Link to preprocessing span
    pass

def ml_pipeline(input_data):
    # TODO: Create parent span
    # TODO: Call preprocessing and inference
    pass
```

### Success Criteria
- [ ] Traces visible in Jaeger UI
- [ ] Spans correctly nested
- [ ] Span attributes provide context
- [ ] Can identify slowest operations
- [ ] Trace links requests end-to-end

---

## Exercise 4: Log Aggregation and Analysis

**Difficulty:** Intermediate
**Duration:** 45 minutes

### Objective
Set up structured logging and aggregate logs for analysis.

### Tasks
1. Implement structured logging with Python's logging
2. Add context (user_id, request_id, etc.)
3. Send logs to centralized location
4. Parse and query logs
5. Create log-based alerts

### Starter Code
```python
# exercises/04-log-aggregation.py
import logging
import structlog

# TODO: Configure structured logging
# log = structlog.get_logger()

def process_ml_request(user_id, model_id, request_data):
    # TODO: Log with context
    # log.info("processing_request",
    #          user_id=user_id,
    #          model_id=model_id)

    try:
        # Process request
        pass
    except Exception as e:
        # TODO: Log error with context
        pass
```

### Success Criteria
- [ ] Logs are structured (JSON)
- [ ] All logs include trace context
- [ ] Logs can be queried by fields
- [ ] Errors include stack traces
- [ ] Log volume is reasonable

---

## Exercise 5: Model Performance Monitoring

**Difficulty:** Advanced
**Duration:** 90 minutes

### Objective
Implement comprehensive model performance monitoring.

### Tasks
1. Track model prediction metrics
2. Detect model drift
3. Monitor data quality
4. Set up alerting for degradation
5. Create drift detection dashboard

### Metrics to Track
- Prediction latency (P50, P95, P99)
- Prediction distribution
- Input feature distributions
- Model confidence scores
- Error rates by category

### Success Criteria
- [ ] All metrics tracked in Prometheus
- [ ] Drift detection implemented
- [ ] Dashboard shows drift metrics
- [ ] Alerts fire when drift detected
- [ ] Historical data preserved

---

## Exercise 6: SLO/SLA Implementation

**Difficulty:** Advanced
**Duration:** 60 minutes

### Objective
Define and monitor SLOs (Service Level Objectives) for ML service.

### Tasks
1. Define SLIs (Service Level Indicators)
2. Set SLO targets (e.g., 99.9% availability)
3. Calculate error budgets
4. Monitor SLO compliance
5. Create SLO dashboard

### Example SLOs
- **Availability**: 99.9% uptime
- **Latency**: 95% of requests < 200ms
- **Error Rate**: < 0.1% errors

### Starter Code
```python
# exercises/06-slo-monitoring.py

class SLOMonitor:
    def __init__(self, target_availability=0.999):
        self.target_availability = target_availability
        # TODO: Initialize metrics

    def record_request(self, success: bool, latency_ms: float):
        # TODO: Record request outcome
        # TODO: Update availability calculation
        # TODO: Update latency percentiles
        pass

    def check_slo_compliance(self) -> dict:
        # TODO: Calculate current SLO compliance
        # TODO: Calculate error budget remaining
        # TODO: Return compliance status
        pass
```

### Success Criteria
- [ ] SLOs clearly defined
- [ ] Real-time SLO compliance tracking
- [ ] Error budget calculation working
- [ ] Dashboard shows burn rate
- [ ] Alerts when SLO at risk

---

## Solutions

Solutions are provided in the `solutions/` directory. Try to complete exercises independently before referencing solutions.

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Tutorials](https://grafana.com/tutorials/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)

---

**Need help?** Ask in [GitHub Discussions](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/discussions)
