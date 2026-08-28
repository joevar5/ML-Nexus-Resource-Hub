# Lab 03: Send Logs to Loki

**Duration:** 60 min  **Prerequisites:** Docker

## Objective
Run Loki + Promtail locally, send structured JSON logs from a Python service, and query them in Grafana with LogQL.

## Steps

### 1. compose.yaml
```yaml
services:
  loki:
    image: grafana/loki:3.0.0
    ports: ["3100:3100"]
    command: ["-config.file=/etc/loki/local-config.yaml"]
  promtail:
    image: grafana/promtail:3.0.0
    volumes:
      - /var/log:/var/log:ro
      - ./promtail.yaml:/etc/promtail/config.yml
      - /var/run/docker.sock:/var/run/docker.sock
    command: ["-config.file=/etc/promtail/config.yml"]
  grafana:
    image: grafana/grafana:11.1.0
    ports: ["3000:3000"]
    environment: { GF_SECURITY_ADMIN_PASSWORD: admin }
```

### 2. promtail.yaml — scrape Docker container logs
```yaml
server: { http_listen_port: 9080 }
positions: { filename: /tmp/positions.yaml }
clients: [{ url: http://loki:3100/loki/api/v1/push }]
scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
```

### 3. Bring up
```bash
docker compose up -d
```

### 4. Sample app that emits structured JSON
```python
# app.py
import json, logging, sys, time, random

class JsonFmt(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **getattr(record, "extra", {}),
        })

h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFmt())
log = logging.getLogger("app")
log.addHandler(h); log.setLevel(logging.INFO)

while True:
    user = f"user-{random.randint(1,100)}"
    log.info("predict", extra={"extra": {"user": user, "latency_ms": random.randint(5,200)}})
    if random.random() < 0.05:
        log.error("inference failed", extra={"extra": {"user": user}})
    time.sleep(0.5)
```

Run it in a Docker container so Promtail picks it up:
```bash
docker build -t app . && docker run -d --name app app
```

### 5. Add Loki data source in Grafana
URL `http://loki:3100`. Save & Test.

### 6. LogQL queries
- `{container="app"}` — all logs
- `{container="app"} |= "error"` — errors
- `{container="app"} | json | latency_ms > 100` — JSON filter
- `count_over_time({container="app"} |= "error"[5m])` — error count

## Validation
- [ ] Loki ingests logs (visible in Grafana Explore).
- [ ] JSON parser extracts `latency_ms` and `user`.
- [ ] LogQL filter on `latency_ms > 100` returns a subset.

## Cleanup
```bash
docker compose down -v
docker rm -f app
```

## Troubleshooting
- **Promtail not scraping** — Check Docker socket mount; on macOS use `/Users/<you>/.docker/run/docker.sock` if standard path doesn't exist.
- **`Maximum chunk age exceeded`** — Loki retention defaults to 168h; logs older are dropped.
- **JSON parsing fails** — Logs must be valid JSON on a single line; multiline JSON requires multiline stage in promtail.
