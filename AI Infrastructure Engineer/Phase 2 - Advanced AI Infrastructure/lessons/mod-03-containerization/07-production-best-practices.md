# Lesson 08: Production Best Practices

**Duration:** 4 hours
**Objectives:** Build secure, reliable, production-ready Docker containers for ML workloads

## Learning Objectives

By the end of this lesson, you will be able to:

1. Implement security best practices (non-root users, minimal images)
2. Configure health checks and restart policies
3. Set up proper logging for containerized ML services
4. Apply resource limits (CPU, memory, GPU)
5. Build production-ready Dockerfile templates
6. Implement graceful shutdown handling
7. Use secrets management for credentials

## Security Best Practices

### 1. Never Run as Root

**Why it matters:**
- If container is compromised, attacker has root access
- Can escape container and access host system
- Violates principle of least privilege

❌ **Bad (runs as root):**
```dockerfile
FROM python:3.11-slim
COPY app.py /app/
CMD ["python", "/app/app.py"]
```

✅ **Good (runs as non-root):**
```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 appuser

# Create app directory with correct ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy files as root
COPY --chown=appuser:appuser app.py /app/

# Switch to non-root user
USER appuser

CMD ["python", "/app/app.py"]
```

**Verify:**
```bash
docker run --rm my-image whoami
# Output: appuser (not root)
```

### 2. Use Minimal Base Images

**Attack surface comparison:**

| Image | Size | Packages | CVEs (avg) |
|-------|------|----------|------------|
| ubuntu:22.04 | 77MB | ~100 | 20-30 |
| python:3.11 | 1GB | ~500 | 50-100 |
| python:3.11-slim | 130MB | ~80 | 10-20 |
| distroless/python3 | 50MB | ~10 | 0-5 |

**Recommendation:**
```dockerfile
# Production: Use slim or distroless
FROM python:3.11-slim  # Good balance

# Or for maximum security
FROM gcr.io/distroless/python3-debian11  # Minimal
```

### 3. Scan for Vulnerabilities

```bash
# Install Trivy
sudo apt-get install trivy

# Scan image
trivy image my-ml-api:v1.0

# Fail build if HIGH/CRITICAL vulnerabilities
trivy image --severity HIGH,CRITICAL --exit-code 1 my-ml-api:v1.0

# In CI/CD
docker build -t my-ml-api:v1.0 .
trivy image --severity HIGH,CRITICAL --exit-code 1 my-ml-api:v1.0 || exit 1
docker push my-ml-api:v1.0
```

### 4. Don't Include Secrets in Images

❌ **NEVER do this:**
```dockerfile
# Secrets baked into image!
ENV API_KEY=sk-abc123
COPY .env /app/
COPY credentials.json /app/
```

✅ **Good approaches:**

**Option 1: Environment variables at runtime**
```bash
docker run -e API_KEY=sk-abc123 my-image
```

**Option 2: Volume mount secrets**
```bash
docker run -v /secrets:/secrets:ro my-image
```

**Option 3: Docker secrets (Swarm)**
```yaml
services:
  api:
    secrets:
      - api_key
secrets:
  api_key:
    external: true
```

**Option 4: Cloud secret managers**
```python
# In app: fetch from AWS Secrets Manager, GCP Secret Manager, etc.
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='api-key')
```

### 5. Read-Only Root Filesystem

```dockerfile
FROM python:3.11-slim

# App doesn't write to filesystem (except /tmp)
USER appuser

# In docker run:
# docker run --read-only --tmpfs /tmp my-image
```

**docker-compose.yml:**
```yaml
services:
  api:
    image: my-api:v1
    read_only: true
    tmpfs:
      - /tmp
```

### 6. Drop Unnecessary Capabilities

```bash
# Drop all capabilities, add only what's needed
docker run \
    --cap-drop=ALL \
    --cap-add=NET_BIND_SERVICE \
    my-image
```

**docker-compose.yml:**
```yaml
services:
  api:
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if binding to port < 1024
```

## Health Checks

### Why Health Checks Matter

**Without health checks:**
- Container runs but application crashed → traffic sent to broken container
- Database connection lost → container marked as "running" but failing

**With health checks:**
- Orchestrator knows real application status
- Automatic restart of unhealthy containers
- Load balancers remove unhealthy instances

### Implementing Health Checks

**In Dockerfile:**
```dockerfile
FROM python:3.11-slim

COPY app.py /app/
WORKDIR /app

RUN pip install fastapi uvicorn

HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=40s \
            --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

**Parameters explained:**
- `--interval=30s`: Check every 30 seconds
- `--timeout=10s`: Fail if check takes >10 seconds
- `--start-period=40s`: Don't count failures in first 40s (app startup)
- `--retries=3`: Mark unhealthy after 3 consecutive failures

**Health check endpoint (app.py):**
```python
from fastapi import FastAPI, HTTPException
import torch

app = FastAPI()

# Global model (loaded at startup)
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = torch.load("model.pth")

@app.get("/health")
async def health():
    """Health check endpoint"""
    checks = {
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available() if model else False
    }

    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)

    return {"status": "healthy", "checks": checks}
```

**In Docker Compose:**
```yaml
services:
  api:
    image: my-ml-api:v1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Check status:**
```bash
# View health status
docker ps

# Detailed health info
docker inspect --format='{{.State.Health.Status}}' container-name
```

## Restart Policies

**Policies:**

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `no` | Never restart | Development/testing |
| `on-failure` | Restart on non-zero exit | Temporary failures |
| `always` | Always restart (even after stop) | Critical services |
| `unless-stopped` | Like always, but respects manual stop | Production services |

**Docker run:**
```bash
# Restart on failure
docker run --restart=on-failure:3 my-image  # Max 3 retries

# Always restart
docker run --restart=unless-stopped my-image
```

**Docker Compose:**
```yaml
services:
  api:
    image: my-ml-api:v1
    restart: unless-stopped  # Production recommended
```

## Graceful Shutdown

### The Problem

**Without graceful shutdown:**
```
User request → Container
      ↓
Docker stops container (SIGTERM)
      ↓
Container killed after 10s (SIGKILL)
      ↓
Request fails, data potentially corrupted
```

**With graceful shutdown:**
```
User request → Container
      ↓
Docker stops container (SIGTERM)
      ↓
App finishes processing current requests
      ↓
App closes database connections
      ↓
Container exits cleanly
```

### Implementation

**app.py:**
```python
import signal
import sys
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Track active requests
active_requests = 0

@app.middleware("http")
async def count_requests(request, call_next):
    global active_requests
    active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests -= 1

def shutdown_handler(signum, frame):
    """Handle shutdown gracefully"""
    print(f"Received signal {signum}, shutting down gracefully...")

    # Stop accepting new requests
    print(f"Waiting for {active_requests} active requests to complete...")

    # Wait for active requests (with timeout)
    import time
    max_wait = 30
    waited = 0
    while active_requests > 0 and waited < max_wait:
        time.sleep(1)
        waited += 1

    print("Shutdown complete")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY app.py requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

# Important: Use exec form to receive signals
CMD ["python", "app.py"]

# NOT: CMD python app.py  # Shell form doesn't forward signals!
```

**Stop with grace period:**
```bash
# Default 10 seconds
docker stop my-container

# Custom grace period (30 seconds)
docker stop -t 30 my-container
```

## Resource Limits

### Why Limit Resources?

**Without limits:**
- One container can consume all host resources
- OOM killer randomly kills containers
- No fair resource sharing
- GPU memory conflicts

**With limits:**
- Predictable performance
- Fair resource allocation
- Better capacity planning

### CPU Limits

```bash
# Limit to 2 CPUs
docker run --cpus=2 my-image

# CPU shares (relative weight)
docker run --cpu-shares=1024 my-image

# Specific CPUs (pinning)
docker run --cpuset-cpus=0,1 my-image
```

**Docker Compose:**
```yaml
services:
  api:
    image: my-ml-api:v1
    deploy:
      resources:
        limits:
          cpus: '2.0'
        reservations:
          cpus: '1.0'
```

### Memory Limits

```bash
# Limit to 4GB
docker run --memory=4g my-image

# Memory + swap limit
docker run --memory=4g --memory-swap=6g my-image

# Memory reservation (soft limit)
docker run --memory-reservation=2g my-image
```

**Docker Compose:**
```yaml
services:
  model-server:
    image: my-model:v1
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### GPU Limits

```bash
# Specific GPU
docker run --gpus '"device=0"' my-image

# GPU memory fraction (if supported)
docker run --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 \
    my-image
```

## Logging Best Practices

### Structured Logging

**app.py:**
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Model loaded successfully", extra={"model_version": "v1.0"})
```

### Log Drivers

**Docker Compose:**
```yaml
services:
  api:
    image: my-ml-api:v1
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=api,environment=production"
```

**Available drivers:**
- `json-file`: Default, JSON logs
- `syslog`: Send to syslog
- `journald`: systemd journal
- `gelf`: Graylog
- `fluentd`: Fluentd
- `awslogs`: CloudWatch Logs
- `gcplogs`: Google Cloud Logging

### Centralized Logging

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    image: my-ml-api:v1
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: api

  fluentd:
    image: fluent/fluentd:latest
    ports:
      - "24224:24224"
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf

  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node

  kibana:
    image: kibana:8.10.0
    ports:
      - "5601:5601"
```

## Production-Ready Dockerfile Template

**Complete template for ML services:**

```dockerfile
# ========================================
# Build stage
# ========================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Runtime stage
# ========================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="Production ML API"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 appuser && \
    mkdir -p /app /app/logs && \
    chown -R appuser:appuser /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Set PATH
ENV PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py .

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=40s \
            --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application (exec form for signal handling)
CMD ["python", "main.py"]
```

## CI/CD Integration

**GitHub Actions workflow:**

**.github/workflows/docker-build.yml:**
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to registry
        uses: docker/login-action@v2
        with:
          registry: ${{ secrets.REGISTRY_URL }}
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - name: Scan for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./

      - name: Build image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: ml-api:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image for vulnerabilities
        run: |
          docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image \
            --severity HIGH,CRITICAL \
            --exit-code 1 \
            ml-api:test

      - name: Run tests
        run: |
          docker run --rm ml-api:test pytest tests/

      - name: Push image
        if: success()
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.REGISTRY_URL }}/ml-api:${{ github.sha }}
            ${{ secrets.REGISTRY_URL }}/ml-api:latest
```

## Summary

In this lesson, you learned:

1. **Security** - Non-root users, minimal images, secret management
2. **Health Checks** - Application-level health monitoring
3. **Restart Policies** - Automatic recovery from failures
4. **Graceful Shutdown** - Handle SIGTERM properly
5. **Resource Limits** - CPU, memory, GPU constraints
6. **Logging** - Structured logs and centralized collection
7. **Production Templates** - Complete Dockerfile examples

**Production Checklist:**
- [ ] Runs as non-root user
- [ ] Uses minimal base image
- [ ] Scanned for vulnerabilities
- [ ] Health check implemented
- [ ] Restart policy configured
- [ ] Graceful shutdown handling
- [ ] Resource limits set
- [ ] Structured logging
- [ ] Secrets not in image
- [ ] Multi-stage build (if applicable)

## Module 03 Complete!

**You've learned:**
- Docker fundamentals for ML
- Writing optimized Dockerfiles
- Multi-container applications with Compose
- Container registries and CI/CD
- GPU-accelerated containers
- Production best practices

**Next steps:**
- Complete Module 03 practical assessment
- Build production-ready ML container
- Move to Module 04: Kubernetes Fundamentals

---

## Self-Check Questions

1. Why should containers never run as root?
2. What's the purpose of a health check?
3. How do you implement graceful shutdown in Python?
4. What's the difference between memory limit and reservation?
5. How do you scan images for vulnerabilities in CI/CD?

## Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Production-Ready Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)
- [Container Security Guide](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

**Congratulations on completing Module 03!**
