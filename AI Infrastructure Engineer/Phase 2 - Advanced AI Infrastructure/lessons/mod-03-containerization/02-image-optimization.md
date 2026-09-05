# Lesson 03: Image Optimization

**Duration:** 5 hours
**Objectives:** Optimize Docker images for size, build speed, and runtime performance

## Learning Objectives

By the end of this lesson, you will be able to:

1. Use multi-stage builds to separate build and runtime dependencies
2. Implement layer caching strategies for fast rebuilds
3. Reduce ML image sizes from 3GB+ to <1GB
4. Write effective .dockerignore files
5. Use build arguments and secrets securely
6. Choose between alpine, slim, and full base images
7. Apply security scanning and vulnerability management

## Why Image Optimization Matters

### The Problem: Bloated ML Images

Without optimization, ML Docker images easily reach:
- **3-5GB** for basic PyTorch/TensorFlow applications
- **10-15GB** for images with build tools and multiple frameworks
- **20GB+** for images with unnecessary system packages

**Real-World Impact:**

| Metric | Unoptimized (5GB) | Optimized (500MB) |
|--------|-------------------|-------------------|
| Pull time (100 Mbps) | ~7 minutes | ~40 seconds |
| Build time | 15-20 minutes | 2-3 minutes |
| Storage cost (per image) | $0.10-0.20/month | $0.01-0.02/month |
| Deployment speed | Slow | Fast |
| Security surface | Large (many packages) | Small (minimal packages) |

**For a team deploying 50 times/day:**
- Unoptimized: ~6 hours of waiting time
- Optimized: ~30 minutes of waiting time
- **Saved: 5.5 hours per day!**

## Multi-Stage Builds

### Concept

Multi-stage builds use multiple `FROM` statements in one Dockerfile:
1. **Build stage**: Install compilers, build tools, compile code
2. **Runtime stage**: Copy only compiled artifacts, minimal runtime dependencies

**Before (Single-Stage):**
```dockerfile
FROM python:3.11
RUN apt-get update && apt-get install -y \
    gcc g++ make cmake \
    git curl wget
RUN pip install torch torchvision
COPY . /app
# Result: 3.5GB image with build tools still present
```

**After (Multi-Stage):**
```dockerfile
# Stage 1: Build
FROM python:3.11 AS builder
RUN apt-get update && apt-get install -y gcc g++
RUN pip install --user torch torchvision

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
# Result: 1.2GB image without build tools
```

### Basic Multi-Stage Example

```dockerfile
# ========================================
# Stage 1: Build dependencies
# ========================================
FROM python:3.11 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages to user directory
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Stage 2: Runtime image
# ========================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY main.py .

# Set environment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
```

**Size Comparison:**
- Single-stage: ~2.8GB
- Multi-stage: ~1.1GB
- **Savings: 60%**

### Advanced Multi-Stage: Compiling from Source

For packages that need compilation:

```dockerfile
# ========================================
# Stage 1: Compile dependencies
# ========================================
FROM python:3.11 AS compiler

WORKDIR /build

# Install compilers and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build custom library
RUN git clone https://github.com/example/custom-ml-lib.git && \
    cd custom-ml-lib && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

# ========================================
# Stage 2: Python dependencies
# ========================================
FROM python:3.11 AS python-builder

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Stage 3: Final runtime
# ========================================
FROM python:3.11-slim

# Copy compiled library
COPY --from=compiler /usr/local/lib /usr/local/lib

# Copy Python packages
COPY --from=python-builder /root/.local /root/.local

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ENV PATH=/root/.local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

CMD ["python", "main.py"]
```

### Multi-Stage for GPU Images

```dockerfile
# ========================================
# Build stage with CUDA development tools
# ========================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch with CUDA
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --user --no-cache-dir -r requirements.txt

# ========================================
# Runtime stage with CUDA runtime only
# ========================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python (no dev packages)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

WORKDIR /app
COPY . /app

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

CMD ["python3.11", "main.py"]
```

**Size Comparison:**
- devel base: 8.2GB
- runtime base: 4.1GB
- **Savings: 50%**

## Layer Caching Strategies

### Understanding Docker Layer Caching

Docker caches each layer. A layer is invalidated when:
1. The instruction changes
2. Files copied by the instruction change
3. A previous layer is invalidated

**Example:**
```dockerfile
FROM python:3.11-slim                  # Layer 1 (cached)
COPY requirements.txt .                # Layer 2 (cached if file unchanged)
RUN pip install -r requirements.txt    # Layer 3 (cached if Layer 2 cached)
COPY . /app                            # Layer 4 (invalidated if ANY file changes)
```

### Optimization Rule: Order by Change Frequency

❌ **Bad (frequent cache invalidation):**
```dockerfile
FROM python:3.11-slim
COPY . /app                           # Changes often
RUN pip install -r requirements.txt   # Reinstalls every time!
```

✅ **Good (efficient caching):**
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .               # Changes rarely
RUN pip install -r requirements.txt   # Cached!
COPY . /app                           # Changes often (but doesn't affect pip)
```

### Separate Dependencies by Change Frequency

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1. System dependencies (change rarely)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Python core dependencies (change occasionally)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# 3. Python dev dependencies (change more frequently)
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# 4. Application code (changes frequently)
COPY src/ ./src/
COPY main.py .

CMD ["python", "main.py"]
```

### Wildcard Copying for Specific Files

```dockerfile
# Copy only Python files (ignore other changes)
COPY src/*.py ./src/

# Copy configuration files
COPY *.json *.yaml ./

# Copy model files separately
COPY models/*.pth ./models/
```

### Cache Mounting (BuildKit)

Use BuildKit cache mounts for package caches:

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Mount pip cache during build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision

# Subsequent builds reuse downloaded packages!
```

**Enable BuildKit:**
```bash
export DOCKER_BUILDKIT=1
docker build -t my-image .
```

## Reducing Image Size

### Technique 1: Use Slim/Alpine Base Images

**Base image comparison:**

| Image | Size | Pros | Cons |
|-------|------|------|------|
| python:3.11 | 1GB | Everything included | Large |
| python:3.11-slim | 130MB | Good balance | Missing some tools |
| python:3.11-alpine | 50MB | Tiny | Compilation issues, slower builds |

**For ML workloads:**
- ✅ **Use slim**: Best balance for ML packages
- ⚠️ **Avoid alpine**: Many ML libraries fail to compile or run slowly

```dockerfile
# Good for ML
FROM python:3.11-slim

# Problematic for ML (compilation issues)
FROM python:3.11-alpine
```

### Technique 2: Remove Build Dependencies After Use

```dockerfile
FROM python:3.11-slim

# Install build deps, use them, remove them in ONE layer
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && pip install --no-cache-dir some-package-requiring-compilation \
    && apt-get purge -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
```

**Why in one RUN command?**
- Each RUN creates a layer
- Layers are additive (deleting in later layer doesn't reduce size)

❌ **Bad (doesn't reduce size):**
```dockerfile
RUN apt-get install gcc g++     # Layer 1: +200MB
RUN pip install package          # Layer 2: +50MB
RUN apt-get remove gcc g++       # Layer 3: +0MB (but Layer 1 still exists!)
# Total: 250MB
```

✅ **Good (actually reduces size):**
```dockerfile
RUN apt-get install gcc g++ && \
    pip install package && \
    apt-get remove gcc g++       # Layer 1: +50MB
# Total: 50MB
```

### Technique 3: Clean Package Caches

```dockerfile
FROM ubuntu:22.04

# Clean apt cache
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*  # Remove apt cache

# Clean pip cache
RUN pip install --no-cache-dir torch  # --no-cache-dir prevents caching

# Clean conda cache (if using conda)
RUN conda install numpy && \
    conda clean -afy
```

### Technique 4: Use .dockerignore

Create `.dockerignore` to exclude files from build context:

```
# .dockerignore

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.venv/
venv/
ENV/

# Data
data/
datasets/
*.csv
*.parquet

# Models (if not needed in image)
models/*.pth
models/*.h5
*.onnx

# Development
.git/
.gitignore
.vscode/
.idea/
*.md
LICENSE
Dockerfile
docker-compose.yml

# Tests
tests/
test_*.py
*_test.py

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Notebooks
*.ipynb
.ipynb_checkpoints/

# Large files
*.zip
*.tar.gz
*.pkl
```

**Impact:**
- Faster builds (less data to send to Docker daemon)
- Smaller images (won't accidentally copy large files)
- Better security (won't copy .env, credentials)

### Technique 5: Install Only What You Need

❌ **Bad (installs everything):**
```dockerfile
RUN pip install transformers
# Includes: PyTorch, TensorFlow, Flax, JAX, and all dependencies
```

✅ **Good (minimal dependencies):**
```dockerfile
RUN pip install transformers torch
# Explicit about what you need
```

**Check what's included:**
```bash
# See all installed packages
docker run my-image pip list

# Check image layers
docker history my-image

# Dive tool for detailed analysis
dive my-image
```

## Advanced Optimization Techniques

### Technique 6: Distroless Images

Google's distroless images contain only your application and runtime dependencies:

```dockerfile
# Build stage
FROM python:3.11 AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . /app

# Runtime with distroless
FROM gcr.io/distroless/python3-debian11

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

ENV PATH=/root/.local/bin:$PATH
WORKDIR /app

CMD ["main.py"]
```

**Benefits:**
- Minimal attack surface (no shell, no package manager)
- Smaller size
- Better security

**Drawbacks:**
- Harder to debug (no shell)
- Limited to specific languages

### Technique 7: Compression and Squashing

```bash
# Build with compression
docker build --compress -t my-image .

# Squash layers (experimental)
docker build --squash -t my-image .

# Export and import to squash
docker save my-image | docker load
```

⚠️ **Note:** Squashing loses layer caching benefits

### Technique 8: Static Analysis with dive

```bash
# Install dive
wget https://github.com/wagoodman/dive/releases/download/v0.11.0/dive_0.11.0_linux_amd64.deb
sudo apt install ./dive_0.11.0_linux_amd64.deb

# Analyze image
dive my-ml-image:latest
```

**What dive shows:**
- Size of each layer
- Wasted space
- Efficiency score
- File changes between layers

## Build Arguments and Secrets

### Build Arguments

Pass variables at build time:

```dockerfile
# Define build argument
ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1

# Use in FROM
FROM python:${PYTHON_VERSION}-slim

# Use in RUN
RUN echo "Building with CUDA ${CUDA_VERSION}"

ARG MODEL_VERSION=v1.0
ENV MODEL_VERSION=${MODEL_VERSION}
```

**Building with arguments:**
```bash
docker build \
    --build-arg PYTHON_VERSION=3.10 \
    --build-arg MODEL_VERSION=v2.0 \
    -t my-image .
```

### Secrets Management

❌ **NEVER do this:**
```dockerfile
# Secrets baked into image!
ENV AWS_SECRET_KEY=abc123
COPY .env /app/
```

✅ **Use BuildKit secrets:**
```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Mount secret during build (not stored in image)
RUN --mount=type=secret,id=pip_token \
    pip install --extra-index-url=https://$(cat /run/secrets/pip_token)@my-pypi.com/simple my-package
```

**Building with secrets:**
```bash
docker build --secret id=pip_token,src=./pip_token.txt -t my-image .
```

## Complete Optimized Example

### Before: Unoptimized Dockerfile (5.2GB)

```dockerfile
FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget

COPY . /app
WORKDIR /app

RUN pip install torch torchvision transformers fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### After: Optimized Dockerfile (680MB)

```dockerfile
# syntax=docker/dockerfile:1

# ========================================
# Build stage
# ========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements (with caching)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir -r requirements.txt

# ========================================
# Runtime stage
# ========================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application (use .dockerignore to exclude unnecessary files)
COPY src/ ./src/
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set environment
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Optimizations applied:**
1. ✅ Multi-stage build (builder + runtime)
2. ✅ Slim base image
3. ✅ Minimal dependencies
4. ✅ BuildKit cache mounts
5. ✅ Removed package caches
6. ✅ Non-root user
7. ✅ Health check
8. ✅ .dockerignore (external file)
9. ✅ Proper layer ordering

**Results:**
- Size: 5.2GB → 680MB (87% reduction)
- Build time: 12 min → 2 min (with cache)
- Security: Better (non-root, minimal packages)

## Measuring and Comparing

### Check Image Size

```bash
# List image sizes
docker images

# Detailed layer information
docker history my-image

# Total size with all tags
docker images my-image
```

### Compare Optimization Impact

```bash
# Before optimization
docker build -t my-image:unoptimized -f Dockerfile.old .
docker images my-image:unoptimized

# After optimization
docker build -t my-image:optimized .
docker images my-image:optimized

# Compare
docker history my-image:unoptimized
docker history my-image:optimized
```

## Hands-On Exercise: Optimize a Bloated Image

### Scenario

You have this unoptimized Dockerfile for a PyTorch NLP application:

```dockerfile
FROM python:3.11

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree

COPY . /app
WORKDIR /app

RUN pip install \
    torch==2.1.0 \
    transformers==4.35.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pandas \
    numpy \
    scikit-learn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

**Current size:** ~4.8GB

### Your Task

Optimize this Dockerfile to:
- [ ] Reduce size to <1.5GB
- [ ] Use multi-stage build
- [ ] Apply layer caching best practices
- [ ] Add non-root user
- [ ] Add health check
- [ ] Create .dockerignore

**Target: 70%+ size reduction**

### Solution Template

```dockerfile
# TODO: Implement multi-stage build

# TODO: Build stage


# TODO: Runtime stage


# TODO: Add security and health checks

```

## Common Optimization Mistakes

### Mistake 1: Optimizing the Wrong Thing

❌ **Focus:** Removing 5MB of documentation
✅ **Focus:** Removing 500MB of build tools

**Prioritize:**
1. Large dependencies (compilers, dev packages)
2. Unused system packages
3. Package caches
4. Then worry about small optimizations

### Mistake 2: Breaking Functionality for Size

Don't remove packages your application needs!

```bash
# Test thoroughly after optimization
docker run my-image python -c "import torch; print(torch.__version__)"
docker run my-image curl http://localhost:8000/health
```

### Mistake 3: Sacrificing Security for Size

❌ **Bad:** Running as root to save a layer
✅ **Good:** Add non-root user even if it adds 50MB

## Summary

In this lesson, you learned:

1. **Multi-Stage Builds** - Separate build and runtime dependencies
2. **Layer Caching** - Order instructions by change frequency
3. **Size Reduction** - Slim images, clean caches, remove build deps
4. **.dockerignore** - Exclude unnecessary files
5. **Build Arguments** - Parameterize builds
6. **Secrets** - Never bake secrets into images
7. **Measurement** - Use docker history and dive

**Key Principles:**
- Start with the smallest viable base image
- Use multi-stage builds to separate concerns
- Order layers from least to most frequently changed
- Clean up in the same layer where you install
- Use .dockerignore aggressively
- Measure and iterate

**Typical Results:**
- **Size:** 3-5GB → 500MB-1GB (70-85% reduction)
- **Build time:** 10-15 min → 2-3 min (with caching)
- **Pull time:** 5-7 min → 30-40 sec

## What's Next?

In the next lesson, **04-docker-networking-volumes.md**, you'll learn:
- Docker networking modes
- Container-to-container communication
- Port mapping for ML services
- Volumes for persistent data and models
- Bind mounts for development

---

## Self-Check Questions

1. What is a multi-stage build and when should you use it?
2. Why should requirements.txt be copied before application code?
3. What's the difference between `python:3.11` and `python:3.11-slim`?
4. How do you clean apt caches to reduce image size?
5. What's wrong with: `RUN apt install gcc && ... && RUN apt remove gcc`?

## Additional Resources

- [Multi-Stage Build Documentation](https://docs.docker.com/build/building/multi-stage/)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [dive - Image Analysis Tool](https://github.com/wagoodman/dive)
- [Distroless Images](https://github.com/GoogleContainerTools/distroless)

---

**Next:** [04-docker-networking-volumes.md](./04-docker-networking-volumes.md)
