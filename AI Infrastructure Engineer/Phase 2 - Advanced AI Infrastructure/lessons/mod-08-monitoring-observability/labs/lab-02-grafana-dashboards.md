# Lab 02: Build a Grafana Dashboard from Scratch

**Duration:** 75 min  **Prerequisites:** Lab 01; Grafana installed

## Objective
Stand up Grafana, connect to Prometheus, and build a 5-panel dashboard for the FastAPI service: RPS, error rate, p95 latency, in-flight requests, error budget.

## Steps

### 1. Run Grafana
```bash
docker run -d --name grafana -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin grafana/grafana:11.1.0
```
http://localhost:3000  user `admin` pass `admin`.

### 2. Add Prometheus data source
- URL: `http://host.docker.internal:9090`
- Click "Save & Test" → green checkmark.

### 3. Build the panels
For each, "Add visualization" → choose Prometheus → paste the query → set unit → save.

**Panel 1: RPS (timeseries)**
- `sum(rate(predictions_total[1m]))`
- Unit: req/s. Legend: total RPS.

**Panel 2: Error Rate (stat)**
- `100 * sum(rate(predictions_total{status="error"}[5m])) / sum(rate(predictions_total[5m]))`
- Unit: percent (0-100). Thresholds: green <1, yellow 1-5, red >5.

**Panel 3: Latency p50/p95/p99 (timeseries)**
- `histogram_quantile(0.50, sum by (le) (rate(inference_latency_seconds_bucket[5m])))` legend `p50`
- p95 and p99 with corresponding quantiles.
- Unit: seconds.

**Panel 4: In-flight requests (timeseries)**
- `inflight_requests`

**Panel 5: 30-day error budget burn (stat)**
- `1 - (sum(increase(predictions_total{status="error"}[30d])) / sum(increase(predictions_total[30d]))) / 0.01`
- (Assumes SLO of 99% availability.) Unit: percent.

### 4. Templating variables
Settings → Variables → Add `service` variable with query `label_values(predictions_total, service)`. Use in queries as `{service=~"$service"}`.

### 5. Export and import
Settings → JSON Model → copy the JSON. Save as `dashboard.json`. To restore, "Import JSON" on a new Grafana.

## Validation
- [ ] Data source connection test succeeds.
- [ ] All 5 panels render with data after driving traffic.
- [ ] Time range selector affects all panels.
- [ ] Dashboard JSON exports cleanly.

## Cleanup
```bash
docker stop grafana && docker rm grafana
```

## Troubleshooting
- **All panels show "No data"** — Data source URL points to the wrong host; check from inside the Grafana container with `docker exec grafana wget -O- http://host.docker.internal:9090/api/v1/query?query=up`.
- **Latency panel returns 0** — Histogram buckets misspecified. Buckets must be cumulative (e.g., 0.005, 0.01, 0.025…).
- **Variables don't resolve** — Variable query needs to return labels, not values; use `label_values()`.
