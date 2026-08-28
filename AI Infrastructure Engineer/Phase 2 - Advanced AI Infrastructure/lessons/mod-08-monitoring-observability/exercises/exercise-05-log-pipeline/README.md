# Exercise 05: Structured Log Pipeline (Loki + Vector)

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 01-02

## Objective

Build a complete log pipeline: structured JSON logs from a Python service, collected by Vector, indexed in Loki, queryable in Grafana. Add trace correlation (logs link to traces in Tempo) and cardinality control.

## Why this matters

Logs are the most-used and most-abused infrastructure signal. Done wrong, you pay $30K/month for logs nobody reads. Done right, they're the difference between a 5-minute and a 5-hour incident.

## Requirements

1. Python service emits **structured JSON** logs to stdout.
2. Each log line includes: timestamp, level, service, trace_id, span_id, message, additional context.
3. **Vector** (or Fluent Bit) collects logs from container stdout.
4. **Loki** indexes them with carefully-chosen labels (low cardinality).
5. **Grafana** has a logs explorer with deep-link to Tempo via trace_id.
6. **Promtail/Vector transforms** parse JSON and extract trace_id for correlation.
7. **Retention policy**: 7 days for INFO, 30 days for WARN/ERROR.
8. **Cardinality budget**: < 10K active streams.

## Step-by-step

### Step 1 — Structured logging from Python (30 min)
```python
# app.py
import logging, json, time, uuid
from pythonjsonlogger import jsonlogger

class TraceJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = "iris-api"
        log_record["timestamp"] = time.time()
        # trace_id injected by OpenTelemetry middleware (Exercise 06)
        log_record.setdefault("trace_id", getattr(record, "trace_id", ""))

handler = logging.StreamHandler()
handler.setFormatter(TraceJsonFormatter())
root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.INFO)

log = logging.getLogger("predict")
log.info("prediction made", extra={"model_version": "v3", "latency_ms": 42, "user": "alice"})
```

Sample output:
```json
{"timestamp":1717000000,"level":"INFO","service":"iris-api","trace_id":"","message":"prediction made","model_version":"v3","latency_ms":42,"user":"alice"}
```

### Step 2 — Vector configuration (30 min)
```yaml
# vector.yaml
sources:
  containers:
    type: kubernetes_logs
    extra_label_selector: app=iris-api

transforms:
  parse_json:
    type: remap
    inputs: [containers]
    source: |
      structured, err = parse_json(.message)
      if err == null {
        . = merge(., structured)
      }

  add_labels:
    type: remap
    inputs: [parse_json]
    source: |
      .labels.service = .service
      .labels.level   = .level
      .labels.namespace = .kubernetes.namespace_name

sinks:
  loki:
    type: loki
    inputs: [add_labels]
    endpoint: http://loki:3100
    labels:
      service: "{{ labels.service }}"
      level: "{{ labels.level }}"
      namespace: "{{ labels.namespace }}"
    encoding:
      codec: json
```

### Step 3 — Loki setup (15 min)
Run via docker-compose or Helm:
```yaml
services:
  loki:
    image: grafana/loki:3.0.0
    ports: ["3100:3100"]
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml
```

Retention config:
```yaml
limits_config:
  retention_period: 168h        # 7d default
table_manager:
  retention_deletes_enabled: true
  retention_period: 168h
```

### Step 4 — Grafana Loki data source (15 min)
Add Loki as data source. Test queries:
- `{service="iris-api"}`
- `{service="iris-api"} | json | latency_ms > 100`
- `count_over_time({service="iris-api", level="ERROR"}[5m])`

### Step 5 — Trace correlation (30 min)
In your dashboard data source, configure derived fields:
- Name: `TraceID`
- Regex: `"trace_id":"(\w+)"`
- URL: `https://grafana/explore?left={...tempo query for ${__value.raw}...}`

Now every log line with a trace_id has a clickable link to Tempo.

### Step 6 — Cardinality control (30 min)
Stream count = unique combinations of labels. With these labels: service × level × namespace. ~5 services × 3 levels × 3 namespaces = 45 streams. Safe.

But if you add `user` as a label: × 10,000 users = 450K streams → Loki dies.

**Rule:** Never label by anything with >100 unique values. Use log line fields instead.

### Step 7 — Per-level retention (15 min)
Run separate Loki instances per retention tier, or use Loki's multi-tenant feature with different limits per tenant (e.g., tenant "errors" with 30d, tenant "default" with 7d). Vector routes:
```yaml
sinks:
  loki_default:
    type: loki
    inputs: [add_labels]
    endpoint: http://loki:3100
    tenant_id: "default"
  loki_errors:
    type: loki
    inputs: [add_labels]
    endpoint: http://loki:3100
    tenant_id: "errors"
    condition: '.level == "ERROR" || .level == "WARN"'
```

## Deliverables

1. Python service emitting structured logs (`app.py`).
2. Vector config (`vector.yaml`).
3. Loki + Vector docker-compose.
4. Grafana derived field config (screenshot or export).
5. `OPERATIONS.md`: cardinality budget, retention policy, escalation procedure for log cost spikes.

## Validation

- [ ] Logs visible in Grafana within ~10s of being emitted.
- [ ] LogQL `| json | latency_ms > 100` extracts and filters correctly.
- [ ] Trace_id click-through opens Tempo query (or shows "no trace" gracefully).
- [ ] Active stream count is < 50 in your test stack.

## Stretch goals

- Add **structured log validation** at PR time: lint that every log call uses the JSON formatter, no f-strings in messages.
- Add **PII redaction** filter in Vector that masks emails, phone numbers.
- Set up **log-based metrics**: count `ERROR` lines per service per minute → emit as a metric to Prometheus.

## Common pitfalls

- **Multi-line tracebacks** — Default Vector treats each line as a log; configure `multiline` parser keyed on the leading whitespace.
- **High cardinality from per-request IDs** — Never put `request_id`, `user_id`, `trace_id` as Loki labels. Put them in the log line; query with `|=` or `|~`.
- **Logs written to disk inside the container** — Won't appear in stdout; either redirect to stdout or have Vector tail the file.
- **JSON parsing fails silently** — Add a fallback: if parse fails, keep the raw line as `message` and tag with `parse_error=true`.
