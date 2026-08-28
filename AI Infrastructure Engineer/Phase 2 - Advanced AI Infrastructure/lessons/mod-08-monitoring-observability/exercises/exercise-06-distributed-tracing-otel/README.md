# Exercise 06: Distributed Tracing with OpenTelemetry

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 05; Jaeger or Tempo running

## Objective

Instrument a 3-service Python stack (frontend → backend → feature-store) with OpenTelemetry, export traces to Tempo, correlate with logs and metrics, and use traces to diagnose a synthetic slow-call scenario.

## Why this matters

Traces are the only signal that answers "where did the time go in this request?" Once you have them, you stop guessing. Once your team is used to having them, going back to grep-the-logs feels archaic.

## Requirements

1. Three Python services instrumented with OTel.
2. Trace context propagated across HTTP boundaries automatically.
3. Custom spans for in-process work (model inference, feature lookup).
4. Traces exported to **Tempo** (or Jaeger).
5. **Logs include trace_id** (correlation from exercise 05).
6. **Metrics include trace exemplars** linking from a Grafana panel to a trace.
7. Use the trace to diagnose a synthetic slow scenario.

## Step-by-step

### Step 1 — Install OTel (15 min)
```bash
pip install opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-otlp \
  opentelemetry-instrumentation-fastapi \
  opentelemetry-instrumentation-requests \
  opentelemetry-instrumentation-logging
```

### Step 2 — Tempo (30 min)
```yaml
# tempo.yaml
server: { http_listen_port: 3200 }
distributor:
  receivers:
    otlp:
      protocols:
        grpc: {}
        http: {}
storage:
  trace:
    backend: local
    local: { path: /tmp/tempo/blocks }
```
```bash
docker run -d -p 3200:3200 -p 4317:4317 -p 4318:4318 \
  -v $PWD/tempo.yaml:/etc/tempo.yaml grafana/tempo:2.5.0 \
  -config.file=/etc/tempo.yaml
```

### Step 3 — Common OTel init (15 min)
```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

def init_tracing(service_name: str, otlp_endpoint: str = "http://localhost:4318/v1/traces"):
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=True)

def instrument_app(app, service_name: str):
    FastAPIInstrumentor().instrument_app(app, tracer_provider=trace.get_tracer_provider())
```

### Step 4 — Three services (60 min)

**Frontend** (`frontend.py`): receives client requests, calls backend.
**Backend** (`backend.py`): orchestrates; calls feature-store + runs inference.
**Feature-store** (`fs.py`): returns features (mock).

Each app:
```python
from fastapi import FastAPI
from tracing import init_tracing, instrument_app

init_tracing("backend")     # or frontend / feature-store
app = FastAPI()
instrument_app(app, "backend")

tracer = trace.get_tracer(__name__)

@app.post("/v1/predict")
async def predict(req):
    with tracer.start_as_current_span("fetch_features") as span:
        features = httpx.get("http://feature-store:8000/features/" + req.user_id).json()
        span.set_attribute("feature.count", len(features))
    with tracer.start_as_current_span("inference") as span:
        pred = model.predict(features)
        span.set_attribute("model.version", "v3")
    return {"prediction": pred}
```

Run all three; hit the frontend with a request. Open Tempo / Grafana Explore → find the trace → see all 3 services + custom spans.

### Step 5 — Logs ↔ Traces correlation (15 min)
With `LoggingInstrumentor`, Python logs auto-include trace_id when in a trace context. Verify your log lines now have `trace_id` populated. Combine with exercise 05's derived field setup → Loki → Tempo deep-link.

### Step 6 — Metrics exemplars (30 min)
Enable exemplar storage in Prometheus:
```yaml
global:
  scrape_interval: 5s
storage:
  exemplars:
    max_exemplars: 100000
```
Make sure your metrics include `trace_id` as exemplar:
```python
from prometheus_client import Histogram
LATENCY = Histogram("http_request_duration_seconds", "...")
LATENCY.observe(latency, exemplar={"trace_id": current_trace_id()})
```
In Grafana, latency panel now shows colored dots; click → opens the trace.

### Step 7 — Diagnose a slow scenario (15 min)
Add an artificial 500ms sleep in the feature-store. Observe in Tempo:
- Total request: ~520ms
- Time breakdown: 5ms frontend + 510ms backend (10ms inference + 500ms feature lookup)

You diagnosed the slow component in 10 seconds without reading any log.

## Deliverables

1. 3 instrumented Python services.
2. Tempo running and storing traces.
3. Grafana with Tempo data source + log correlation + exemplars.
4. A `DEBUGGING.md` describing how you'd use traces to diagnose 3 common slow scenarios.

## Validation

- [ ] A single client request produces a trace spanning all 3 services.
- [ ] Custom spans (fetch_features, inference) appear with correct attributes.
- [ ] Clicking a log line in Loki opens the corresponding trace in Tempo.
- [ ] Clicking a high-latency exemplar in a Grafana panel opens the corresponding trace.
- [ ] You can identify the slow component (feature-store) from a single trace.

## Stretch goals

- Add **gRPC** trace propagation; instrument a gRPC backend.
- Add **database span instrumentation** (`opentelemetry-instrumentation-asyncpg`).
- Implement **trace sampling** at the SDK level (1% sampling for healthy traffic, 100% for errors).

## Common pitfalls

- **Spans missing parent_id** — Propagation broken. Verify `traceparent` header forwarded across services.
- **BatchSpanProcessor swallows spans on crash** — Configure shorter `export_timeout_millis` for dev; use `SimpleSpanProcessor` for tests.
- **Tempo storage fills disk** — Tempo's `local` backend isn't for prod; use S3/GCS for any persistent setup.
- **OTel overhead noticeable** — Batched export adds ~5-10ms tail latency. Tune `max_export_batch_size`.
