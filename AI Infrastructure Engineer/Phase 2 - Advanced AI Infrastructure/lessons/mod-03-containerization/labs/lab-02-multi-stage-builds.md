# Lab 02: Multi-Stage Builds for Slim Production Images

**Duration:** 60 min  **Prerequisites:** Lab 01 complete

## Objective
Use multi-stage Dockerfiles to separate build-time concerns (compilers, large dev SDKs) from runtime, producing an image that contains only what's needed to serve traffic.

## Steps

### 1. Identify the problem
Many ML libraries (`xgboost`, `lightgbm`, `pyarrow`, native CUDA bindings) need a C/C++ toolchain at install time but not at runtime. A single-stage image ships hundreds of MB of unnecessary build tools.

### 2. Three-stage build for a PyTorch service
```dockerfile
# syntax=docker/dockerfile:1.6

# Stage 1: deps with compilers
FROM python:3.11 AS deps
WORKDIR /build
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2: app build (e.g., compile Cython extensions)
FROM python:3.11 AS appbuild
WORKDIR /src
COPY pyproject.toml setup.py ./
COPY app/ ./app/
RUN pip install build && python -m build --wheel --outdir /dist

# Stage 3: runtime — only Python + installed wheels
FROM python:3.11-slim AS runtime
RUN groupadd -r app && useradd -r -g app -u 1000 app
WORKDIR /app
COPY --from=deps /wheels /tmp/wheels
COPY --from=appbuild /dist /tmp/dist
RUN pip install --no-index --find-links=/tmp/wheels --find-links=/tmp/dist \
        myapp && rm -rf /tmp/wheels /tmp/dist
USER 1000
CMD ["myapp"]
```

### 3. Verify size delta
```bash
docker build --target deps -t app:deps .
docker build --target runtime -t app:runtime .
docker images app
```
`runtime` should be < 50% of `deps`.

### 4. Reproduce a real-world example with PyTorch
Use `pytorch/pytorch:2.3-runtime` in the runtime stage; copy your model + entrypoint only.

## Validation
- [ ] `docker image inspect app:runtime` shows `pip` and `python` present but no `gcc`/`g++`.
- [ ] Runtime image size < 1.5 GB (or whatever your target is).
- [ ] App starts and responds successfully.

## Cleanup
```bash
docker rmi app:deps app:appbuild app:runtime 2>/dev/null
```

## Troubleshooting
- **Missing shared libs at runtime** — Some PyTorch ops need `libgomp1`; add `apt-get install -y libgomp1` in runtime stage.
- **Wheels from stage 1 incompatible** — Stage 1 and stage 3 must use the same Python minor version (3.11.x ↔ 3.11.x).
