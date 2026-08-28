# Lab 01: Instrument a Python Service with Prometheus

**Duration:** 60 min  **Prerequisites:** Python 3.11+, Docker

## Objective
Add Counter, Histogram, and Gauge metrics to a FastAPI service; expose them on `/metrics`; run Prometheus locally to scrape; query in the UI.

## Steps

### 1. Sample service
```python
# app.py
import time, random
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

REQUESTS  = Counter("predictions_total", "Total predictions", ["status"])
LATENCY   = Histogram("inference_latency_seconds", "Inference latency",
                       buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
INFLIGHT  = Gauge("inflight_requests", "In-flight requests")

app = FastAPI()
app.mount("/metrics", make_asgi_app())

@app.post("/predict")
async def predict(payload: dict):
    INFLIGHT.inc()
    t0 = time.perf_counter()
    try:
        # simulate inference
        time.sleep(random.uniform(0.005, 0.05))
        if random.random() < 0.02:
            REQUESTS.labels(status="error").inc()
            return {"error": "transient"}, 500
        REQUESTS.labels(status="ok").inc()
        return {"prediction": random.random()}
    finally:
        LATENCY.observe(time.perf_counter() - t0)
        INFLIGHT.dec()
```
```bash
pip install fastapi uvicorn prometheus-client
uvicorn app:app --port 8000 &
curl http://localhost:8000/metrics | head -20
```

### 2. prometheus.yml
```yaml
global: { scrape_interval: 5s }
scrape_configs:
  - job_name: app
    static_configs:
      - targets: ['host.docker.internal:8000']
```

### 3. Run Prometheus in Docker
```bash
docker run -d --name prom -p 9090:9090 -v $PWD/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

### 4. Drive traffic
```bash
for i in $(seq 1 200); do curl -s -X POST http://localhost:8000/predict -d '{}' -H 'content-type: application/json' > /dev/null; done
```

### 5. Query in Prometheus UI (http://localhost:9090)
- `predictions_total`
- `rate(predictions_total[1m])`
- `sum by (status) (rate(predictions_total[5m]))`
- `histogram_quantile(0.95, sum by (le) (rate(inference_latency_seconds_bucket[5m])))`
- `inflight_requests`

## Validation
- [ ] `/metrics` endpoint returns Prometheus exposition format.
- [ ] Prometheus targets page shows `app` UP.
- [ ] All 5 queries return non-empty data after the traffic drive.

## Cleanup
```bash
docker stop prom && docker rm prom
pkill -f uvicorn
rm prometheus.yml app.py
```

## Troubleshooting
- **Targets page shows DOWN** — `host.docker.internal` only resolves on macOS/Windows; on Linux use the host IP or `--add-host=host.docker.internal:host-gateway`.
- **No data for `inflight_requests`** — Gauge that hits zero between scrapes. Add a longer-running endpoint to see it.
- **`histogram_quantile returns NaN`** — Not enough samples in the window; widen `[5m]` to `[15m]` or drive more traffic.
