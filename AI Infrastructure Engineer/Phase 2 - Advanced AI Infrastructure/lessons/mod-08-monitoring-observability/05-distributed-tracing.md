# Lesson 05: Distributed Tracing with Jaeger and Tempo

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand distributed tracing concepts and terminology
- Install and configure Jaeger for trace collection
- Deploy Grafana Tempo for cost-effective tracing
- Instrument Python applications with OpenTelemetry
- Analyze traces to identify performance bottlenecks
- Correlate traces with logs and metrics
- Design tracing strategies for ML pipelines

## Prerequisites
- Completion of Lessons 01-04 (Observability fundamentals)
- Understanding of microservices architecture
- Python programming experience
- Familiarity with HTTP and async operations

## Introduction

Distributed tracing tracks requests as they flow through complex distributed systems. For AI infrastructure, tracing is essential for understanding:
- How long each stage of an ML pipeline takes
- Where bottlenecks occur in multi-service inference
- Dependencies between data processing, model loading, and prediction
- Latency distribution across GPU workers

While metrics tell you *what* is slow and logs tell you *why* it failed, traces tell you *where* the time is spent.

### Why Distributed Tracing for ML Systems?

1. **Multi-service inference**: Track requests across preprocessing → model serving → postprocessing
2. **Pipeline debugging**: Identify slow stages in data pipelines
3. **Latency analysis**: Understand where inference time is spent
4. **Dependency mapping**: Visualize service dependencies
5. **Performance optimization**: Find opportunities for parallelization or caching

---

## 1. Tracing Fundamentals

### Core Concepts

**Trace**: The complete journey of a request through a system
**Span**: A single operation within a trace (e.g., HTTP request, database query, model inference)
**Trace ID**: Unique identifier for the entire trace
**Span ID**: Unique identifier for a specific span
**Parent Span**: The span that initiated the current span

### Trace Structure

```
Trace ID: abc123
│
├─ Span: HTTP POST /predict (100ms)
│  ├─ Span: Load model (20ms)
│  ├─ Span: Preprocess data (15ms)
│  ├─ Span: Model inference (50ms)
│  │  ├─ Span: GPU transfer (5ms)
│  │  ├─ Span: Forward pass (40ms)
│  │  └─ Span: GPU sync (5ms)
│  └─ Span: Postprocess results (15ms)
```

### Span Attributes

Spans can include metadata:
- **Name**: Operation name (e.g., "model_inference")
- **Start time**: When the operation started
- **Duration**: How long it took
- **Tags**: Key-value metadata (model_name, batch_size, etc.)
- **Logs**: Events during the span
- **Status**: Success, error, or unknown

---

## 2. OpenTelemetry

OpenTelemetry (OTel) is the industry standard for instrumentation, providing vendor-neutral APIs and SDKs.

### Installing OpenTelemetry

```bash
# Core packages
pip install opentelemetry-api==1.20.0
pip install opentelemetry-sdk==1.20.0

# Instrumentation libraries
pip install opentelemetry-instrumentation==0.41b0
pip install opentelemetry-instrumentation-fastapi==0.41b0
pip install opentelemetry-instrumentation-requests==0.41b0

# Exporters
pip install opentelemetry-exporter-jaeger==1.20.0
pip install opentelemetry-exporter-otlp==1.20.0
```

### Basic Instrumentation

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource

# Configure resource (service identification)
resource = Resource.create({
    "service.name": "ml-model-serving",
    "service.version": "1.0.0",
    "deployment.environment": "production"
})

# Setup tracer provider
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

# Add span processor
tracer_provider.add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Get tracer
tracer = trace.get_tracer(__name__)

# Create spans
def process_request(data):
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("request.size", len(data))

        # Child span
        with tracer.start_as_current_span("preprocess"):
            preprocessed = preprocess(data)

        # Another child span
        with tracer.start_as_current_span("inference"):
            result = model.predict(preprocessed)

        span.set_attribute("result.size", len(result))
        return result
```

---

## 3. Jaeger

Jaeger is a distributed tracing platform originally built at Uber.

### Architecture

```
┌─────────────┐
│ Application │
│  (OpenTel)  │
└──────┬──────┘
       │ UDP/gRPC
       ↓
┌──────────────┐      ┌─────────────┐
│    Jaeger    │─────>│   Storage   │
│    Agent     │      │(Elasticsearch│
└──────┬───────┘      │  Cassandra) │
       │              └─────────────┘
       │                      │
       ↓                      ↓
┌──────────────┐      ┌─────────────┐
│   Jaeger     │<─────│   Jaeger    │
│  Collector   │      │    Query    │
└──────────────┘      └──────┬──────┘
                             │
                             ↓
                      ┌─────────────┐
                      │  Jaeger UI  │
                      └─────────────┘
```

### Installing Jaeger

**All-in-one Docker (Development):**

```yaml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:1.50
    container_name: jaeger
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
    ports:
      - "5775:5775/udp"   # Agent (compact thrift)
      - "6831:6831/udp"   # Agent (thrift)
      - "6832:6832/udp"   # Agent (binary thrift)
      - "5778:5778"       # Agent config
      - "16686:16686"     # UI
      - "14268:14268"     # Collector HTTP
      - "14250:14250"     # Collector gRPC
      - "9411:9411"       # Zipkin compatible
    networks:
      - tracing

networks:
  tracing:
    driver: bridge
```

**Production Deployment (Kubernetes with Elasticsearch):**

```yaml
# Elasticsearch backend
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: jaeger-es
  namespace: tracing
spec:
  version: 8.10.0
  nodeSets:
    - name: default
      count: 3
      config:
        node.store.allow_mmap: false
      volumeClaimTemplates:
        - metadata:
            name: elasticsearch-data
          spec:
            accessModes:
              - ReadWriteOnce
            resources:
              requests:
                storage: 100Gi

---
# Jaeger Operator
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger-production
  namespace: tracing
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: https://jaeger-es-es-http:9200
        index-prefix: jaeger
    secretName: jaeger-es-secret
  collector:
    replicas: 3
    resources:
      limits:
        cpu: 1000m
        memory: 2Gi
  query:
    replicas: 2
    resources:
      limits:
        cpu: 500m
        memory: 1Gi
```

### FastAPI Instrumentation

```python
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Setup tracing
resource = Resource.create({
    "service.name": "ml-serving",
    "service.version": "1.0.0"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="jaeger:4317", insecure=True)
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Create FastAPI app
app = FastAPI()

# Automatically instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Get tracer for manual instrumentation
tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(data: dict):
    """Instrumented prediction endpoint"""

    # Automatic span created by FastAPIInstrumentor
    # Add custom spans for specific operations

    with tracer.start_as_current_span("load_model") as span:
        span.set_attribute("model.name", data.get("model"))
        model = load_model(data.get("model"))

    with tracer.start_as_current_span("preprocess") as span:
        span.set_attribute("input.size", len(data.get("inputs")))
        processed = preprocess(data.get("inputs"))

    with tracer.start_as_current_span("inference") as span:
        span.set_attribute("batch.size", len(processed))
        predictions = model.predict(processed)
        span.set_attribute("predictions.count", len(predictions))

    return {"predictions": predictions}
```

---

## 4. Grafana Tempo

Tempo is a distributed tracing backend designed to be cost-effective and easy to operate.

### Tempo vs. Jaeger

| Feature | Tempo | Jaeger |
|---------|-------|--------|
| **Storage** | Object storage (S3, GCS) | Elasticsearch, Cassandra |
| **Indexing** | Trace ID only | Full trace indexing |
| **Search** | By trace ID or via metrics/logs | Full-text search on tags |
| **Cost** | Very low | Higher |
| **Query interface** | Grafana | Jaeger UI |
| **Integration** | Native Grafana integration | Separate UI |

### Installing Tempo

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  tempo:
    image: grafana/tempo:2.3.0
    container_name: tempo
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo-config.yaml:/etc/tempo.yaml
      - tempo_data:/tmp/tempo
    ports:
      - "3200:3200"   # Tempo HTTP
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_FEATURE_TOGGLES_ENABLE=traceqlEditor
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml
    networks:
      - monitoring
    depends_on:
      - tempo

volumes:
  tempo_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

**tempo-config.yaml:**

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        http:
        grpc:
    jaeger:
      protocols:
        thrift_http:
        grpc:

ingester:
  trace_idle_period: 10s
  max_block_bytes: 1_000_000
  max_block_duration: 5m

compactor:
  compaction:
    block_retention: 168h  # 7 days

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces
    pool:
      max_workers: 100
      queue_depth: 10000

overrides:
  per_tenant_override_config: /etc/tempo-overrides.yaml
  metrics_generator_processors:
    - service-graphs
    - span-metrics
```

**Grafana datasource configuration:**

```yaml
# grafana-datasources.yaml
apiVersion: 1

datasources:
  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    editable: false
    jsonData:
      tracesToLogs:
        datasourceUid: loki
        tags: ['job', 'instance', 'pod', 'namespace']
        mappedTags: [{ key: 'service.name', value: 'service' }]
        mapTagNamesEnabled: false
        spanStartTimeShift: '1h'
        spanEndTimeShift: '1h'
      tracesToMetrics:
        datasourceUid: prometheus
        tags: [{ key: 'service.name', value: 'service' }]
        queries:
          - name: 'Sample query'
            query: 'sum(rate(traces_spanmetrics_latency_bucket{$$__tags}[5m]))'
      serviceMap:
        datasourceUid: prometheus
      nodeGraph:
        enabled: true
```

---

## 5. Complete ML Pipeline Tracing Example

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import time
import logging

# Configure tracing
resource = Resource.create({
    "service.name": "ml-inference-pipeline",
    "service.version": "2.0.0",
    "deployment.environment": "production",
    "service.namespace": "ml-platform"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint="tempo:4317",
        insecure=True
    )
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

class MLInferencePipeline:
    """ML inference pipeline with comprehensive tracing"""

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.logger = logging.getLogger(__name__)

    def process_request(self, inputs: list, request_id: str):
        """
        Process inference request with distributed tracing

        Creates a trace showing:
        - Total request time
        - Time spent in each pipeline stage
        - GPU operations
        - Any errors or warnings
        """

        # Root span for entire request
        with tracer.start_as_current_span(
            "ml_inference_request",
            attributes={
                "request.id": request_id,
                "model.name": self.model_name,
                "model.version": self.model_version,
                "input.count": len(inputs)
            }
        ) as root_span:

            try:
                # Stage 1: Model loading
                if self.model is None:
                    self._load_model()

                # Stage 2: Input validation
                validated_inputs = self._validate_inputs(inputs)

                # Stage 3: Preprocessing
                preprocessed = self._preprocess(validated_inputs)

                # Stage 4: Inference
                predictions = self._infer(preprocessed)

                # Stage 5: Postprocessing
                results = self._postprocess(predictions)

                # Record success
                root_span.set_status(Status(StatusCode.OK))
                root_span.set_attribute("predictions.count", len(results))

                return results

            except Exception as e:
                # Record error in span
                root_span.set_status(
                    Status(StatusCode.ERROR, str(e))
                )
                root_span.record_exception(e)
                self.logger.error(f"Inference failed: {e}", exc_info=True)
                raise

    def _load_model(self):
        """Load model with tracing"""
        with tracer.start_as_current_span(
            "load_model",
            attributes={
                "model.name": self.model_name,
                "model.version": self.model_version
            }
        ) as span:
            start_time = time.time()

            try:
                # Simulate model loading
                time.sleep(0.5)
                self.model = f"MockModel-{self.model_name}"

                load_time = time.time() - start_time
                span.set_attribute("model.load_time_seconds", load_time)
                span.set_attribute("model.size_mb", 512)

                # Add event
                span.add_event(
                    "model_loaded",
                    attributes={"load_time": load_time}
                )

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def _validate_inputs(self, inputs: list):
        """Validate inputs with tracing"""
        with tracer.start_as_current_span(
            "validate_inputs",
            attributes={"input.count": len(inputs)}
        ) as span:

            if not inputs:
                span.set_status(Status(StatusCode.ERROR, "Empty inputs"))
                raise ValueError("Inputs cannot be empty")

            if len(inputs) > 128:
                span.add_event(
                    "large_batch_warning",
                    attributes={"batch_size": len(inputs)}
                )

            span.set_attribute("validation.passed", True)
            return inputs

    def _preprocess(self, inputs: list):
        """Preprocess with sub-operations traced"""
        with tracer.start_as_current_span(
            "preprocess",
            attributes={"input.count": len(inputs)}
        ) as span:

            # Sub-operation: Tokenization
            with tracer.start_as_current_span("tokenize") as token_span:
                time.sleep(0.02)
                token_span.set_attribute("tokens.count", len(inputs) * 512)

            # Sub-operation: Normalization
            with tracer.start_as_current_span("normalize") as norm_span:
                time.sleep(0.01)
                norm_span.set_attribute("normalization.method", "zscore")

            # Sub-operation: Batching
            with tracer.start_as_current_span("batch") as batch_span:
                batch_size = min(len(inputs), 32)
                batch_span.set_attribute("batch.size", batch_size)

            span.set_attribute("preprocessed.count", len(inputs))
            return inputs

    def _infer(self, preprocessed: list):
        """Inference with GPU operations traced"""
        with tracer.start_as_current_span(
            "inference",
            attributes={
                "batch.size": len(preprocessed),
                "model.name": self.model_name
            }
        ) as span:

            # GPU transfer
            with tracer.start_as_current_span(
                "gpu_transfer",
                attributes={"device": "cuda:0"}
            ) as gpu_span:
                time.sleep(0.005)
                gpu_span.set_attribute("transfer.direction", "host_to_device")
                gpu_span.set_attribute("data.size_mb", 10)

            # Forward pass
            with tracer.start_as_current_span(
                "forward_pass",
                attributes={"device": "cuda:0"}
            ) as forward_span:
                time.sleep(0.05)
                forward_span.set_attribute("gpu.utilization", 87.5)
                forward_span.set_attribute("gpu.memory_used_mb", 2048)

                # Add event for GPU metrics
                forward_span.add_event(
                    "gpu_stats",
                    attributes={
                        "gpu_id": 0,
                        "temperature": 72,
                        "power_watts": 180
                    }
                )

            # GPU sync
            with tracer.start_as_current_span("gpu_sync") as sync_span:
                time.sleep(0.003)
                sync_span.set_attribute("transfer.direction", "device_to_host")

            span.set_attribute("predictions.generated", len(preprocessed))
            return preprocessed  # Mock predictions

    def _postprocess(self, predictions: list):
        """Postprocess with tracing"""
        with tracer.start_as_current_span(
            "postprocess",
            attributes={"predictions.count": len(predictions)}
        ) as span:

            # Apply threshold
            with tracer.start_as_current_span("apply_threshold") as thresh_span:
                time.sleep(0.005)
                thresh_span.set_attribute("threshold", 0.5)

            # Format results
            with tracer.start_as_current_span("format_results") as format_span:
                time.sleep(0.003)
                format_span.set_attribute("format", "json")

            span.set_attribute("results.count", len(predictions))
            return predictions


# Usage example
if __name__ == "__main__":
    pipeline = MLInferencePipeline(
        model_name="bert-base",
        model_version="v1.2"
    )

    # Process request
    results = pipeline.process_request(
        inputs=["sample input 1", "sample input 2"],
        request_id="req-12345"
    )

    print(f"Results: {results}")
```

---

## 6. Correlating Traces, Logs, and Metrics

### Trace Context Propagation

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import logging
import structlog

# Get current span context
def log_with_trace_context(message: str, **kwargs):
    """Log with trace and span IDs"""

    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()

    logger = structlog.get_logger()
    logger.info(
        message,
        trace_id=format(span_context.trace_id, '032x'),
        span_id=format(span_context.span_id, '016x'),
        **kwargs
    )

# Usage
with tracer.start_as_current_span("operation") as span:
    log_with_trace_context(
        "Processing batch",
        batch_size=32,
        model="bert"
    )
```

### Linking Traces in Grafana

In Grafana, configure trace-to-logs correlation:

```yaml
# In Tempo datasource configuration
tracesToLogs:
  datasourceUid: 'loki'
  tags: ['service', 'namespace']
  mappedTags:
    - key: 'service.name'
      value: 'service'
  filterByTraceID: true
  filterBySpanID: false
```

---

## Summary

In this lesson, you learned:

✅ Distributed tracing concepts (traces, spans, context propagation)
✅ OpenTelemetry instrumentation for Python applications
✅ Setting up Jaeger for trace collection and analysis
✅ Deploying Grafana Tempo for cost-effective tracing
✅ Instrumenting ML pipelines with comprehensive tracing
✅ Analyzing traces to identify performance bottlenecks
✅ Correlating traces with logs and metrics

## Next Steps

- **Lesson 06**: Learn alerting strategies and alert management
- **Practice**: Instrument your ML services with OpenTelemetry
- **Exercise**: Trace a complete ML inference pipeline

## Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [Distributed Tracing Best Practices](https://opentelemetry.io/docs/concepts/signals/traces/)

---

**Estimated Time:** 4-5 hours
**Difficulty:** Intermediate-Advanced
**Prerequisites:** Lessons 01-04, Python, Microservices basics
