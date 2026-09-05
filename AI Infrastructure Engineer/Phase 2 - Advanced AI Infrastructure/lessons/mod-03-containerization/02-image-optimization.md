# Lesson 02: Image Optimization

Unoptimized ML images routinely hit 3-5GB, and 10GB+ when build tools and multiple frameworks pile up. That costs real time and money: slower pulls, slower deploys, more storage, and a bigger security surface. This lesson covers the techniques that typically cut image size 50-80% — multi-stage builds, layer-caching discipline, and a handful of targeted size reductions.

**Prerequisites:** Lesson 01 (Dockerfile instructions, base images, ML dependency installation).


### Contents

1. [Why It Matters](#why-it-matters)
2. [Multi-Stage Builds](#multi-stage-builds)
3. [Layer Caching Strategy](#layer-caching-strategy)
4. [Reducing Image Size](#reducing-image-size)
5. [Build Arguments and Secrets](#build-arguments-and-secrets)
6. [Complete Before/After Example](#complete-beforeafter-example)
7. [Practical Exercise](#practical-exercise)
8. [Key Takeaways](#key-takeaways)
9. [Additional Resources](#additional-resources)

---

## Why It Matters

| Metric | Unoptimized (5GB) | Optimized (500MB) |
|---|---|---|
| Pull time (100 Mbps) | ~7 min | ~40 sec |
| Build time | 15-20 min | 2-3 min |
| Storage cost / image / month | $0.10-0.20 | $0.01-0.02 |
| Security surface | Large (many packages) | Small |

At 50 deploys/day, that gap is roughly 5+ hours of aggregate waiting time saved per day — not a rounding error.

## Multi-Stage Builds

A multi-stage Dockerfile uses more than one `FROM`: a **build stage** installs compilers and builds artifacts, and a **runtime stage** copies over only what's needed to run, leaving the build tools behind entirely.

```dockerfile
# Stage 1: build
FROM python:3.11 AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY src/ ./src/
COPY main.py .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "main.py"]
```

This alone typically takes a ~2.8GB single-stage image to ~1.1GB — about a 60% reduction, because `gcc`/`g++` and apt's package lists never make it into the final stage.

**GPU variant** — build against the `devel` CUDA image (has compilers), ship against `runtime`:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --user --no-cache-dir -r requirements.txt

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
WORKDIR /app
COPY . /app
ENV PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
CMD ["python3.11", "main.py"]
```

`devel` base: ~8.2GB. `runtime`-based final image: ~4.1GB — roughly 50% smaller, and GPU images stay large regardless because CUDA/cuDNN itself is multi-gigabyte.

For dependencies that need compiling from source (a custom C++ extension, say), add an intermediate stage that builds the library, and copy just the compiled `.so`/binary into the runtime stage — never the compiler toolchain.

## Layer Caching Strategy

A layer is invalidated when its instruction changes, when files it copies change, or when any earlier layer is invalidated — invalidation cascades forward. The single most common ML Dockerfile mistake is copying source code before installing dependencies, which forces a full reinstall on every code change:

```dockerfile
# Bad — reinstalls on every code change
COPY . /app
RUN pip install -r requirements.txt

# Good — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
```

Generalize this into an order-by-change-frequency layout: system packages (rarely change) → core Python deps → dev/optional deps → application code (changes constantly). You can also scope `COPY` with wildcards (`COPY src/*.py ./src/`) to avoid invalidating a layer over unrelated file changes.

BuildKit cache mounts persist the pip download cache across builds, independent of layer caching:

```dockerfile
# syntax=docker/dockerfile:1
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision
```

Enable BuildKit with `export DOCKER_BUILDKIT=1` (or use `docker buildx build`, which enables it by default).

> [!NOTE]
> Put stable/rarely changing things FIRST, and frequently changing things LAST.

## Reducing Image Size

**Base image choice** — for ML specifically, `slim` is the sweet spot; `alpine`'s musl libc breaks or slows down the compilation of many ML wheels (numpy, torch, etc.), so it's usually not worth the extra savings:

| Image | Size | Notes |
|---|---|---|
| `python:3.11` | ~1GB | Everything included |
| `python:3.11-slim` | ~130MB | Best balance for ML |
| `python:3.11-alpine` | ~50MB | Compilation issues with many ML libs — avoid |

**Remove build dependencies in the same layer that installs them** — layers are additive, so removing a package in a *later* layer doesn't shrink the image; the earlier layer's bytes are still there:

```dockerfile
# Bad: three layers, ~250MB total — gcc/g++ still present underneath
RUN apt-get install -y gcc g++
RUN pip install some-package-requiring-compilation
RUN apt-get remove -y gcc g++

# Good: one layer, ~50MB total
RUN apt-get install -y gcc g++ \
    && pip install --no-cache-dir some-package-requiring-compilation \
    && apt-get purge -y gcc g++ && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
```

**Clean package caches**: `rm -rf /var/lib/apt/lists/*` after `apt-get install`, `pip install --no-cache-dir`, `conda clean -afy` after conda installs.

**Use `.dockerignore`** to keep `__pycache__`, `.venv`, datasets, notebooks, `.git`, and secrets out of the build context — faster builds, smaller images, and no accidental secret leaks.

**Install only what you need** — `pip install transformers` alone pulls in PyTorch, TensorFlow, Flax, and JAX as optional backends unless you're explicit; `pip install transformers torch` narrows that.

**Inspect what you shipped**: `docker history <image>` shows per-layer size; the [dive](https://github.com/wagoodman/dive) tool (`dive my-image:latest`) breaks down wasted space and gives an efficiency score.

**Distroless** images (`gcr.io/distroless/python3-debian11`) contain your app and its runtime deps only — no shell, no package manager — for minimal attack surface, at the cost of much harder debugging:

```dockerfile
FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . /app

FROM gcr.io/distroless/python3-debian11
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
CMD ["main.py"]
```

`docker build --squash` merges all layers into one — it trades away layer-cache reuse for a smaller final artifact, so use it only for final release images, not dev iterations.

## Build Arguments and Secrets

`ARG` parameterizes a build (`ARG PYTHON_VERSION=3.11`, then `FROM python:${PYTHON_VERSION}-slim`, set via `docker build --build-arg PYTHON_VERSION=3.10`).

Never bake secrets into an image with `ENV` or by `COPY`-ing a `.env` file — they persist in the image history even if a later layer deletes them. Use BuildKit secret mounts instead, which are never written to a layer:

```dockerfile
# syntax=docker/dockerfile:1
RUN --mount=type=secret,id=pip_token \
    pip install --extra-index-url=https://$(cat /run/secrets/pip_token)@my-pypi.com/simple my-package
```

```bash
docker build --secret id=pip_token,src=./pip_token.txt -t my-image .
```

## Complete Before/After Example

**Before (5.2GB):**

```dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y build-essential cmake git curl vim wget
COPY . /app
WORKDIR /app
RUN pip install torch torchvision transformers fastapi uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

**After (680MB):**

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY src/ ./src/
COPY main.py .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Result: 5.2GB → 680MB (87% reduction), build time 12min → 2min with cache warm.

## Practical Exercise

Optimize this Dockerfile for a PyTorch NLP service (~4.8GB as written):

```dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y build-essential git curl wget vim nano htop tree
COPY . /app
WORKDIR /app
RUN pip install torch==2.1.0 transformers==4.35.0 fastapi==0.104.1 uvicorn==0.24.0 pandas numpy scikit-learn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

Target: under 1.5GB, multi-stage, cached layer order, non-root user, health check, `.dockerignore`.

<details>
<summary><strong>Sample Solution</strong></summary>

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY main.py .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt` should list pinned versions of `torch`, `transformers`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn` — dropping the dev tools (`vim`, `nano`, `htop`, `tree`) entirely, since they add weight with no runtime purpose. This gets the image under 1.5GB and typically closer to 1GB.

</details>

Verify after any optimization pass — deleted packages sometimes turn out to be load-bearing: `docker run my-image python -c "import torch; print(torch.__version__)"` and hit the health endpoint before calling it done. Prioritize by impact: compilers and dev packages first (hundreds of MB each), then unused system packages, then caches — chasing a 5MB doc file while a 500MB build toolchain sits unremoved is optimizing the wrong thing.

---

## Key Takeaways

1. Multi-stage builds are the highest-leverage optimization — they remove compilers and build-only packages from the shipped image entirely.
2. Layer order should follow change frequency: rarely-changing layers first, frequently-changing layers (application code) last.
3. Install and clean up in the *same* `RUN` layer — layers are additive, so removing something in a later layer doesn't shrink the image.
4. `slim` is the ML sweet spot; `alpine` usually costs more in compilation pain than it saves in size.
5. Never trade away security for size — a non-root user and a health check are worth the few extra MB.

## Additional Resources

- [Multi-Stage Build Documentation](https://docs.docker.com/build/building/multi-stage/)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [dive - Image Analysis Tool](https://github.com/wagoodman/dive)
- [Distroless Images](https://github.com/GoogleContainerTools/distroless)

---

**Next Lesson:** [03-docker-networking-volumes.md](./03-docker-networking-volumes.md) — Networking and persistent storage
