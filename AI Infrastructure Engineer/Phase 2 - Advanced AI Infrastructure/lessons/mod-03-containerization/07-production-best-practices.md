# Lesson 07: Production Best Practices

A container that works in development can still be a liability in production — running as root, with no health check, no resource limits, and secrets baked into the image. This lesson is a checklist-driven pass through the practices that make a container safe and reliable to run: non-root users, minimal images, health checks, graceful shutdown, resource limits, and structured logging.

**Prerequisites:** Lessons 01-06 — this lesson assumes familiarity with Dockerfiles, Compose, registries, and (where relevant) GPU containers.

### Contents

1. [Security Best Practices](#security-best-practices)
2. [Health Checks](#health-checks)
3. [Restart Policies](#restart-policies)
4. [Graceful Shutdown](#graceful-shutdown)
5. [Resource Limits](#resource-limits)
6. [Logging](#logging)
7. [Production Dockerfile Template](#production-dockerfile-template)
8. [Practical Exercise](#practical-exercise)
9. [Key Takeaways](#key-takeaways)
10. [Additional Resources](#additional-resources)

---

## Security Best Practices

**Never run as root.** A compromised root container is a much shorter path to the host than a compromised unprivileged one.

```dockerfile
FROM python:3.11-slim

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser app.py /app/
USER appuser

CMD ["python", "/app/app.py"]
```

```bash
docker run --rm my-image whoami   # appuser, not root
```

**Use minimal base images** — fewer packages means fewer CVEs to track:

| Image | Size | CVEs (typical) |
|---|---|---|
| `ubuntu:22.04` | 77MB | 20-30 |
| `python:3.11` | 1GB | 50-100 |
| `python:3.11-slim` | 130MB | 10-20 |
| `distroless/python3` | 50MB | 0-5 |

`python:3.11-slim` is the practical default; distroless is worth it when you don't need a shell in the container at all.

**Scan every build**, and fail CI on high-severity findings (see Lesson 06 for scanning in more depth):

```bash
trivy image --severity HIGH,CRITICAL --exit-code 1 my-ml-api:v1.0
```

**Keep secrets out of the image.** Never `ENV API_KEY=...` or `COPY .env` into a layer — it's in the image history forever, readable by anyone who pulls it. Instead:

```bash
docker run -e API_KEY=sk-abc123 my-image           # env var at runtime
docker run -v /secrets:/secrets:ro my-image         # mounted secret
```

```python
# or fetch from a cloud secret manager at startup
import boto3
secret = boto3.client("secretsmanager").get_secret_value(SecretId="api-key")
```

**Read-only root filesystem and dropped capabilities** shrink what a compromised process can do:

```yaml
services:
  api:
    image: my-api:v1
    read_only: true
    tmpfs: [/tmp]
    cap_drop: [ALL]
    cap_add: [NET_BIND_SERVICE]   # only if binding to a port < 1024
```

---

## Health Checks

Without one, an orchestrator only knows the process is running, not that it's actually serving correctly — a stuck DB connection or a crashed model load can leave a container marked "up" while every request fails.

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

- `interval` — how often to check
- `timeout` — fail the check if it takes longer than this
- `start_period` — grace period after container start before failures count (covers model-loading time)
- `retries` — consecutive failures before the container is marked unhealthy

The endpoint should check what actually matters, not just "process is alive":

```python
@app.get("/health")
async def health():
    checks = {"model_loaded": model is not None, "gpu_available": torch.cuda.is_available()}
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "healthy", "checks": checks}
```

```bash
docker inspect --format='{{.State.Health.Status}}' container-name
```

---

## Restart Policies

| Policy | Behavior | Use Case |
|---|---|---|
| `no` | Never restart | Development |
| `on-failure` | Restart on non-zero exit | Transient failures, cap retries |
| `always` | Always restart, even after manual stop and daemon restart | Rarely what you want |
| `unless-stopped` | Like `always`, but respects an explicit manual stop | Production default |

```yaml
services:
  api:
    restart: unless-stopped
```

---

## Graceful Shutdown

Without it, Docker sends SIGTERM, gives the process 10 seconds, then SIGKILLs it — any in-flight request gets dropped mid-response.

```python
import signal, sys, time

active_requests = 0

@app.middleware("http")
async def count_requests(request, call_next):
    global active_requests
    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

def shutdown_handler(signum, frame):
    print(f"Received signal {signum}, draining {active_requests} active requests...")
    waited = 0
    while active_requests > 0 and waited < 30:
        time.sleep(1)
        waited += 1
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)
```

The Dockerfile detail that makes this work at all: use exec form for `CMD`, not shell form. Shell form (`CMD python app.py`) runs the process under a `/bin/sh` wrapper that swallows SIGTERM instead of forwarding it.

```dockerfile
CMD ["python", "app.py"]   # correct — exec form, signals reach the process
```

```bash
docker stop -t 30 my-container   # give the drain loop room to finish
```

---

## Resource Limits

Without limits, one container can starve the host or trigger the OOM killer, which picks a victim more or less at random.

```yaml
services:
  api:
    deploy:
      resources:
        limits: { cpus: '2.0', memory: 4G }
        reservations: { cpus: '1.0', memory: 2G }
```

Equivalent `docker run` flags: `--cpus=2`, `--memory=4g`, `--memory-swap=6g`, `--cpuset-cpus=0,1` for pinning to specific cores. GPU limits are set via `--gpus` device selection (Lesson 07) — Docker doesn't support fractional GPU memory limits directly, so cap usage inside the framework instead:

```bash
docker run --gpus '"device=0"' -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 my-image
```

---

## Logging

**Structure logs as JSON** so they're queryable downstream instead of grepped:

```python
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })
```

**Pick a log driver** appropriate to where logs need to end up — `json-file` (default), `syslog`, `journald`, `gelf` (Graylog), `fluentd`, `awslogs`, `gcplogs`:

```yaml
services:
  api:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

Always set `max-size`/`max-file` (or the equivalent for another driver) — unbounded `json-file` logs are a classic way to quietly fill a disk.

---

## Production Dockerfile Template

Everything above combined — multi-stage build, non-root user, health check, exec-form CMD:

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

LABEL maintainer="your-email@example.com" version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser && \
    mkdir -p /app /app/logs && chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py .

USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
```

CI should scan for leaked secrets and vulnerabilities before pushing, and run the test suite against the built image rather than the source tree:

```yaml
- name: Scan for secrets
  uses: trufflesecurity/trufflehog@main
  with: { path: ./ }

- name: Scan image for vulnerabilities
  run: |
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      aquasec/trivy:latest image --severity HIGH,CRITICAL --exit-code 1 ml-api:test

- name: Run tests
  run: docker run --rm ml-api:test pytest tests/
```

---

## Practical Exercise

Take a Dockerfile that runs as root with no health check and no resource limits, and bring it up to the production checklist below.

**Requirements:** convert to a non-root user, add a `/health` endpoint and `HEALTHCHECK`, set `restart: unless-stopped` and CPU/memory limits in Compose, implement SIGTERM draining, and confirm `docker inspect` reports `healthy`.

<details>
<summary><strong>Sample Solution</strong></summary>

Starting Dockerfile (don't ship this):

```dockerfile
FROM python:3.11-slim
COPY app.py /app/
CMD ["python", "/app/app.py"]
```

Fixed version — non-root, health check, exec form, minimal runtime deps:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser app.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "app.py"]
```

Compose side — restart policy and resource limits:

```yaml
services:
  api:
    build: .
    restart: unless-stopped
    deploy:
      resources:
        limits: { cpus: '2.0', memory: 4G }
        reservations: { cpus: '1.0', memory: 2G }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Verify:

```bash
docker compose up -d
docker inspect --format='{{.State.Health.Status}}' $(docker compose ps -q api)
docker compose stop -t 30 api   # confirm graceful drain, not a hard kill
```

</details>

---

## Key Takeaways

1. Run as a non-root user and drop unneeded capabilities — a compromised unprivileged container is far less dangerous than a compromised root one.
2. A health check should verify real readiness (model loaded, dependencies reachable), not just "process is alive."
3. Use exec-form `CMD` so SIGTERM reaches the process — shell form swallows it, breaking graceful shutdown.
4. Set explicit CPU/memory limits; without them one container can starve the host or trigger a random OOM kill.
5. Never bake secrets into an image layer — inject them at runtime or pull from a secrets manager.

---

## Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Production-Ready Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

**Next:** take the [Module 03 quiz](./quizzes/module-quiz.md) to check your understanding, then move on to **Module 04: Kubernetes** — where the health checks, resource limits, and rolling-update concerns from this lesson become first-class Kubernetes objects (liveness/readiness probes, resource requests/limits, and Deployments) at cluster scale.
