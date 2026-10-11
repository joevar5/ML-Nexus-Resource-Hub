# Lab 04: Distributed Tracing with OpenTelemetry + Jaeger

**Duration:** 75 min  **Prerequisites:** Docker

## Objective
Instrument a 2-service Python pipeline (frontend → backend) with OpenTelemetry, export traces to Jaeger, and explore a single request across both services.

## Steps

### 1. Run Jaeger all-in-one
```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:1.57
```
UI at http://localhost:16686.

### 2. Install OTel
```bash
pip install opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi \
  opentelemetry-instrumentation-requests fastapi uvicorn requests
```

### 3. Backend service
```python
# backend.py
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

provider = TracerProvider(resource=Resource.create({"service.name": "backend"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")))
trace.set_tracer_provider(provider)

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)
tracer = trace.get_tracer(__name__)

@app.get("/work")
def work():
    with tracer.start_as_current_span("compute") as span:
        span.set_attribute("custom.label", "demo")
        return {"result": 42}
```

### 4. Frontend service
```python
# frontend.py
import requests
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

provider = TracerProvider(resource=Resource.create({"service.name": "frontend"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")))
trace.set_tracer_provider(provider)
RequestsInstrumentor().instrument()

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)

@app.get("/call")
def call():
    r = requests.get("http://localhost:8001/work")
    return {"backend": r.json()}
```

### 5. Run both
```bash
uvicorn backend:app --port 8001 &
uvicorn frontend:app --port 8000 &
```

### 6. Hit and observe
```bash
curl http://localhost:8000/call
```
Open Jaeger UI → service "frontend" → find the trace → see it span both services with the `compute` child span.

## Validation
- [ ] Trace appears in Jaeger within ~10s of the request.
- [ ] Trace spans both `frontend` and `backend` services.
- [ ] `compute` custom span shows the `custom.label=demo` attribute.

## Cleanup
```bash
pkill -f uvicorn
docker stop jaeger && docker rm jaeger
```

## Troubleshooting
- **No traces in Jaeger** — Verify OTLP endpoint reachable; default is 4318 for HTTP, 4317 for gRPC.
- **Spans appear but not linked** — Trace context not propagated; ensure `RequestsInstrumentor` is enabled.
- **Performance overhead** — BatchSpanProcessor batches; for production tune `max_queue_size` and `schedule_delay_millis`.
