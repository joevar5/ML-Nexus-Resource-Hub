# Exercise 08: Pipeline Monitoring and Data Lineage

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 02 + 07

## Objective

Instrument a multi-stage pipeline so you can answer: "is the pipeline healthy?", "which upstream produced this dataset?", and "what downstream consumers will be affected if I change this column?" Use OpenLineage for lineage and Prometheus for runtime metrics.

## Why this matters

Data engineers spend disproportionate time answering "where did this number come from?" and "what would break if I change this?" Lineage automation answers both. It's the single biggest leverage point for a data platform team.

## Requirements

1. **Per-stage metrics**: latency, row count in/out, freshness, success/failure counts.
2. **OpenLineage events** emitted from every Airflow task and dbt run.
3. **Marquez** as the lineage backend.
4. **Lineage queries**: given a column, list all upstream sources and downstream consumers.
5. **Freshness SLO** per dataset: alert when last-updated > N hours.
6. **Grafana dashboard** showing pipeline health.

## Step-by-step

### Step 1 — Marquez setup (15 min)
```bash
git clone https://github.com/MarquezProject/marquez && cd marquez
./docker/up.sh
# UI at http://localhost:3000
```

### Step 2 — OpenLineage Airflow integration (30 min)
```bash
pip install openlineage-airflow
```
In `airflow.cfg` or env:
```ini
AIRFLOW__OPENLINEAGE__TRANSPORT={"type":"http","url":"http://localhost:5000"}
AIRFLOW__OPENLINEAGE__NAMESPACE=ml-platform
```
After restarting Airflow, every DAG run emits start/complete events with dataset URIs.

### Step 3 — Run a DAG (15 min)
From exercise 02's `recs_training` DAG. Observe in Marquez UI: jobs, datasets, and edges between them.

### Step 4 — Per-stage Prometheus metrics (45 min)
```python
from airflow.decorators import task
from prometheus_client import Counter, Histogram, Gauge, push_to_gateway, CollectorRegistry

@task
def ingest(date: str) -> str:
    registry = CollectorRegistry()
    rows = Counter("pipeline_rows_processed", "Rows", ["stage","dataset"], registry=registry)
    latency = Histogram("pipeline_stage_seconds", "Latency", ["stage"], registry=registry)
    with latency.labels("ingest").time():
        df = ...
        rows.labels("ingest","events").inc(len(df))
    push_to_gateway("pushgateway:9091", job="recs_training", registry=registry)
    return path
```

### Step 5 — Freshness tracking (30 min)
For each dataset, record `last_updated_at` to a key-value store (Redis):
```python
import redis
r = redis.Redis()
r.set(f"freshness:{dataset_uri}", time.time())
```
Prometheus query: `time() - dataset_last_updated`.
Alert when freshness > 4h for "hot" datasets.

### Step 6 — Lineage queries (30 min)
Marquez exposes a REST API:
```bash
# Upstream of a dataset
curl "http://localhost:5000/api/v1/lineage?nodeId=dataset:my-namespace:training_data&depth=10" | jq

# All consumers of a column (column-level lineage)
curl "http://localhost:5000/api/v1/column-lineage?nodeId=column:..." | jq
```

Build a CLI:
```bash
lineage upstream events.click
lineage downstream training_data
lineage impact-of column-name
```

### Step 7 — Grafana dashboard (30 min)
Panels:
- Per-stage runtime trend (line chart)
- Rows in/out per stage (bar chart)
- Dataset freshness (gauges per dataset)
- Recent failures (table)
- Lineage depth (stat: "datasets you depend on" / "datasets depending on you")

## Deliverables

1. Marquez running with at least 5 datasets + 3 jobs from your real DAG runs.
2. `lineage` CLI for queries.
3. Grafana dashboard JSON.
4. `FRESHNESS_SLO.md` documenting per-dataset SLOs.
5. Demo: trace a downstream metric back to its source dataset and code.

## Validation

- [ ] Running a DAG produces lineage edges visible in Marquez.
- [ ] `lineage upstream <dataset>` returns the producing job.
- [ ] Per-stage Prometheus metrics scraped.
- [ ] Freshness alert fires when a dataset's last update is artificially aged.

## Stretch goals

- Add **column-level lineage** by parsing SQL via sqllineage or sqlfluff.
- Add a **breaking-change detector**: PR checker that fails if a column is renamed and downstream consumers haven't been updated.
- Integrate **dbt**'s OpenLineage adapter for SQL transformations.

## Common pitfalls

- **OpenLineage events lost on Airflow crash** — Transport buffers them; use HTTP + retries.
- **Lineage too noisy** — Synthetic / debug DAGs flood Marquez. Tag environments and filter.
- **Freshness based on event-time vs ingestion-time** — Pick one. Mixing is the most confusing class of incident.
- **Per-task push to Prometheus pushgateway** — Pushgateway is for batch; use it for terminal stages, not every task in a long DAG (overhead adds up).
