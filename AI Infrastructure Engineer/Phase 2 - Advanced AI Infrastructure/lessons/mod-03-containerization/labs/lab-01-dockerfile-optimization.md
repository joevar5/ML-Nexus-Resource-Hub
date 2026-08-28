# Lab 01: Dockerfile Optimization for ML Workloads

**Duration:** 60 min  **Prerequisites:** Docker installed; lab 02 of mod-101 complete

## Objective
Take a naive ML Dockerfile and apply optimizations: layer ordering, multi-stage builds, BuildKit cache mounts, and `.dockerignore`. Measure image size and build time before/after.

## Steps

### 1. Start with a naive Dockerfile
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```
Build it and note size: `docker build -t naive .` → typically 1.2-1.5 GB.

### 2. Switch to slim base + .dockerignore
```dockerfile
FROM python:3.11-slim
...
```
Add `.dockerignore`:
```
__pycache__/
*.pyc
.git/
venv/
tests/
*.md
```

### 3. Order layers by change frequency
Copy `requirements.txt` and install before copying source — cached if requirements don't change:
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

### 4. Use BuildKit cache mount for pip
```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
```
Build with `DOCKER_BUILDKIT=1 docker build`.

### 5. Multi-stage to drop build tools
```dockerfile
FROM python:3.11 AS builder
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
```

### 6. Measure
```bash
docker images | grep -E 'naive|optimized'
hyperfine 'docker build --no-cache -t opt .'
```

## Validation
- [ ] Optimized image < 400 MB (vs naive > 1 GB).
- [ ] Second build (no source changes) < 5 seconds (cache hit on pip layer).
- [ ] `docker history opt` shows fewer than 10 layers.

## Cleanup
```bash
docker rmi naive opt
```

## Troubleshooting
- **Cache mount has no effect** — Ensure BuildKit is enabled (`docker buildx version` reports >=0.10).
- **Slim image missing system libs** — Install `gcc`/`g++` in builder stage, not runtime.
