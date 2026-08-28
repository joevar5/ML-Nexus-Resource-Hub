# Lesson 02: Dockerfiles for ML Applications

**Duration:** 5 hours
**Objectives:** Write production-ready Dockerfiles for machine learning applications

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand Dockerfile syntax and instructions
2. Choose appropriate base images for ML workloads
3. Install ML frameworks (PyTorch, TensorFlow) in containers
4. Copy application code and model files into images
5. Configure entry points and commands for ML services
6. Handle Python dependencies effectively
7. Write your first Dockerfile for a PyTorch application

## Introduction to Dockerfiles

A **Dockerfile** is a text file containing instructions to build a Docker image. Think of it as a recipe:

```dockerfile
# Recipe for a Docker image
FROM python:3.11-slim          # Start with this base
RUN apt-get update             # Run these commands
COPY app.py /app/              # Copy these files
CMD ["python", "app.py"]       # Run this when container starts
```

When you run `docker build`, Docker executes each instruction in sequence, creating a new **layer** for each step. These layers are cached, making subsequent builds faster.

## Dockerfile Instructions

### FROM: Choose Your Base Image

Every Dockerfile starts with `FROM`, which specifies the base image:

```dockerfile
# Official Python image
FROM python:3.11-slim

# NVIDIA CUDA image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Pre-built PyTorch image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

**Choosing the Right Base:**
- **python:3.11-slim**: Small Python image (~50MB), good for CPU workloads
- **python:3.11**: Full Python image (~300MB), includes more tools
- **nvidia/cuda**: CUDA support for GPU workloads
- **pytorch/pytorch**: PyTorch pre-installed with CUDA

### RUN: Execute Commands

`RUN` executes commands during image build:

```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.35.0 \
    fastapi==0.104.1
```

**Best Practices:**
- Chain commands with `&&` to reduce layers
- Clean up package caches (`rm -rf /var/lib/apt/lists/*`)
- Use `--no-cache-dir` with pip to save space

### WORKDIR: Set Working Directory

`WORKDIR` sets the directory for subsequent instructions:

```dockerfile
WORKDIR /app

# Now all commands run in /app
COPY model.py .           # Copies to /app/model.py
RUN python setup.py       # Runs in /app
```

### COPY and ADD: Copy Files

`COPY` copies files from your machine into the image:

```dockerfile
# Copy a single file
COPY requirements.txt /app/

# Copy entire directory
COPY ./src /app/src

# Copy model weights (be careful with large files!)
COPY models/resnet50.pth /app/models/
```

**COPY vs ADD:**
- Use `COPY` for simple file copying
- `ADD` can extract tar files and download URLs (usually avoid it)

### ENV: Set Environment Variables

`ENV` sets environment variables:

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    MODEL_PATH=/app/models/model.pth
```

**Common ML Environment Variables:**
- `PYTHONUNBUFFERED=1`: See logs in real-time
- `CUDA_VISIBLE_DEVICES`: Control GPU access
- `OMP_NUM_THREADS`: Control CPU threading
- `TOKENIZERS_PARALLELISM=false`: Avoid tokenizer warnings

### EXPOSE: Document Ports

`EXPOSE` documents which ports the container listens on:

```dockerfile
# Document that API runs on port 8000
EXPOSE 8000
```

**Note:** `EXPOSE` is documentation only. You still need `-p` when running:
```bash
docker run -p 8000:8000 my-ml-app
```

### CMD and ENTRYPOINT: Container Startup

**CMD** specifies the default command:

```dockerfile
# Exec form (preferred)
CMD ["python", "app.py"]

# Shell form (avoid - doesn't handle signals properly)
CMD python app.py
```

**ENTRYPOINT** defines the executable:

```dockerfile
# Container always runs python
ENTRYPOINT ["python"]

# Default argument (can be overridden)
CMD ["app.py"]

# Running: docker run my-image train.py
# Executes: python train.py
```

**Best Practice for ML:**
```dockerfile
# Flexible entry point
ENTRYPOINT ["python", "-m"]
CMD ["serve"]

# docker run my-image serve     → python -m serve
# docker run my-image train     → python -m train
```

## Base Images for ML Applications

### Python Official Images

**python:3.11-slim** (Recommended for CPU)
```dockerfile
FROM python:3.11-slim
# Size: ~50MB base, ~500MB with ML packages
# Pros: Small, fast builds, official
# Cons: No GPU support, minimal tools
```

**python:3.11**
```dockerfile
FROM python:3.11
# Size: ~300MB base
# Pros: More tools included
# Cons: Larger, unnecessary packages
```

### NVIDIA CUDA Images

**For GPU workloads:**

```dockerfile
# Runtime image (for inference)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Devel image (for training/compilation)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

**Understanding CUDA image tags:**
- `12.1.0`: CUDA version
- `cudnn8`: cuDNN version (deep learning primitives)
- `runtime` vs `devel`: Runtime = smaller, Devel = includes compilers
- `ubuntu22.04`: Base OS

**When to use which:**
- **runtime**: Inference, model serving (smaller)
- **devel**: Training, building from source (larger but complete)

### Framework-Specific Images

**PyTorch Official Images:**

```dockerfile
# CPU
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# GPU runtime
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# GPU development
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

**TensorFlow Official Images:**

```dockerfile
# CPU
FROM tensorflow/tensorflow:2.15.0

# GPU
FROM tensorflow/tensorflow:2.15.0-gpu
```

**Hugging Face Images:**

```dockerfile
FROM huggingface/transformers-pytorch-gpu:latest
```

### Choosing Your Base Image: Decision Tree

```
Do you need GPU?
├─ No  → python:3.11-slim
└─ Yes → Do you need to compile code?
    ├─ No  → nvidia/cuda:12.1-runtime or pytorch/pytorch:*-runtime
    └─ Yes → nvidia/cuda:12.1-devel or pytorch/pytorch:*-devel
```

## Installing ML Dependencies

### Using requirements.txt

**Best Practice: Separate requirements file**

```dockerfile
# Copy only requirements first (for caching)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Then copy code
COPY . /app/
```

**Why this order matters:**
- Docker caches layers
- If code changes but requirements don't, Docker reuses cached layer
- Fast rebuilds!

**requirements.txt example:**

```txt
# requirements.txt
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
pillow==10.1.0
```

### Using conda (Alternative)

```dockerfile
FROM continuumio/miniconda3

# Copy environment file
COPY environment.yml /tmp/

# Create environment
RUN conda env create -f /tmp/environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Run app in conda environment
CMD ["conda", "run", "-n", "myenv", "python", "app.py"]
```

**When to use conda:**
- Need non-Python dependencies (compiled C++ libraries)
- Data science packages with complex dependencies
- Team already uses conda

**When to use pip:**
- Simpler dependencies
- Smaller images (conda adds 500MB+)
- Faster builds

### Installing PyTorch

**CPU-only:**

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.1):**

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Troubleshooting: Version Matching**

⚠️ **Common Pitfall:** CUDA version mismatch

```dockerfile
# WRONG: CUDA 12.1 base image, PyTorch built for CUDA 11.8
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install torch  # Defaults to CUDA 11.8!

# CORRECT: Match versions
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Installing TensorFlow

**CPU:**

```dockerfile
RUN pip install --no-cache-dir tensorflow==2.15.0
```

**GPU:**

```dockerfile
# Use TensorFlow's official GPU image
FROM tensorflow/tensorflow:2.15.0-gpu

# Or install manually
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.15.0
```

## Copying Code and Models

### Copying Application Code

```dockerfile
# Create app directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Copy entry point
COPY main.py /app/
```

### Handling Model Files

**Small Models (<100MB):**

```dockerfile
# Can include in image
COPY models/resnet50.pth /app/models/
```

**Large Models (>100MB):**

❌ **Don't include in image** - makes images huge and slow to pull

✅ **Better approaches:**

1. **Download on startup:**

```dockerfile
# startup.sh
#!/bin/bash
if [ ! -f /app/models/model.pth ]; then
    echo "Downloading model..."
    wget https://storage.googleapis.com/my-models/model.pth -O /app/models/model.pth
fi
python main.py
```

2. **Mount as volume:**

```bash
docker run -v /host/models:/app/models my-image
```

3. **Use model registry:**

```python
# app.py - download from MLflow or S3
import mlflow
model = mlflow.pytorch.load_model("models:/my-model/production")
```

### Using .dockerignore

Create `.dockerignore` to exclude unnecessary files:

```
# .dockerignore
**/__pycache__
**/.git
**/.venv
**/.pytest_cache
*.pyc
*.pyo
*.pyd
.DS_Store
.env
*.log
notebooks/
tests/
docs/
data/large_dataset/
```

**Benefits:**
- Faster builds (less data to copy)
- Smaller images
- Avoid copying secrets accidentally

## Complete Dockerfile Examples

### Example 1: Simple PyTorch Inference API

```dockerfile
# Start with Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY main.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pth

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Example 2: GPU-Accelerated Training Container

```dockerfile
# CUDA base image with cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .
COPY utils/ ./utils/
COPY configs/ ./configs/

# Set environment variables for GPU
ENV CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Entry point for flexible commands
ENTRYPOINT ["python"]
CMD ["train.py"]
```

### Example 3: Hugging Face Transformers Model

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install git (needed for transformers)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY config.json .

# Create cache directory for models
RUN mkdir -p /app/model_cache

# Environment variables
ENV TRANSFORMERS_CACHE=/app/model_cache \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Download model at build time (optional)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Building and Running Images

### Building an Image

```bash
# Build with tag
docker build -t my-ml-app:v1.0 .

# Build with build arguments
docker build --build-arg CUDA_VERSION=12.1 -t my-ml-app:v1.0 .

# Build and view progress
docker build --progress=plain -t my-ml-app:v1.0 .
```

### Running a Container

```bash
# Basic run
docker run my-ml-app:v1.0

# Run with port mapping
docker run -p 8000:8000 my-ml-app:v1.0

# Run with GPU
docker run --gpus all my-ml-app:v1.0

# Run in detached mode
docker run -d -p 8000:8000 --name ml-api my-ml-app:v1.0

# Run with environment variables
docker run -e MODEL_NAME=resnet50 -p 8000:8000 my-ml-app:v1.0

# Run with volume mount
docker run -v $(pwd)/models:/app/models -p 8000:8000 my-ml-app:v1.0
```

## Common Pitfalls and Solutions

### Pitfall 1: Large Image Sizes

❌ **Problem:**
```dockerfile
FROM python:3.11
RUN pip install torch torchvision
# Result: 5GB+ image
```

✅ **Solution:**
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Result: ~2GB image (we'll optimize further in next lesson)
```

### Pitfall 2: Cache Invalidation

❌ **Problem:**
```dockerfile
COPY . /app/                    # Copies everything
RUN pip install -r requirements.txt  # Reinstalls every time code changes
```

✅ **Solution:**
```dockerfile
COPY requirements.txt /app/
RUN pip install -r requirements.txt  # Cached unless requirements change
COPY . /app/                          # Code changes don't affect pip layer
```

### Pitfall 3: Running as Root

❌ **Problem:**
```dockerfile
# Container runs as root (security risk)
CMD ["python", "app.py"]
```

✅ **Solution:**
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "app.py"]
```

### Pitfall 4: No Health Checks

❌ **Problem:**
```dockerfile
# No way to know if service is healthy
CMD ["python", "app.py"]
```

✅ **Solution:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

## Hands-On Exercise: Build Your First ML Dockerfile

### Objective
Create a Dockerfile for a FastAPI application that serves a PyTorch image classification model.

### Application Structure
```
ml-api/
├── Dockerfile
├── requirements.txt
├── app.py
├── model.py
└── models/
    └── resnet18.pth
```

### Step 1: Create Application Files

**app.py:**
```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from model import ImageClassifier
import io

app = FastAPI()
classifier = ImageClassifier()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    result = classifier.predict(image)
    return {"prediction": result}
```

**requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
python-multipart==0.0.6
```

### Step 2: Write Dockerfile

**TODO: Complete this Dockerfile**

```dockerfile
# TODO: Choose appropriate base image


# TODO: Set working directory


# TODO: Install system dependencies (if needed)


# TODO: Copy and install Python requirements


# TODO: Copy application code


# TODO: Set environment variables


# TODO: Expose port


# TODO: Add health check


# TODO: Set command to run the application

```

### Step 3: Build and Test

```bash
# Build the image
docker build -t ml-api:v1 .

# Run the container
docker run -d -p 8000:8000 --name ml-api ml-api:v1

# Test health endpoint
curl http://localhost:8000/health

# Test prediction (with an image)
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

### Success Criteria
- [ ] Image builds successfully
- [ ] Container starts without errors
- [ ] Health check returns 200 OK
- [ ] Can make predictions via API
- [ ] Image size < 2GB

## Summary

In this lesson, you learned:

1. **Dockerfile Syntax** - FROM, RUN, COPY, CMD, ENTRYPOINT, etc.
2. **Base Images** - Choosing between Python, CUDA, and framework-specific images
3. **ML Dependencies** - Installing PyTorch, TensorFlow with proper versions
4. **Code Management** - Copying code and handling model files
5. **Best Practices** - Layer caching, .dockerignore, non-root users
6. **Building and Running** - docker build and docker run commands

**Key Takeaways:**
- Order matters: dependencies before code for caching
- Choose minimal base images when possible
- Use .dockerignore to exclude unnecessary files
- Match CUDA versions between base image and packages
- Don't include large models in images

## What's Next?

In the next lesson, **03-image-optimization.md**, you'll learn:
- Multi-stage builds to reduce image size by 50-70%
- Advanced layer caching strategies
- Optimizing for both build time and runtime
- Security scanning and vulnerability management

---

## Self-Check Questions

1. What's the difference between `CMD` and `ENTRYPOINT`?
2. Why should you copy `requirements.txt` before copying application code?
3. When would you use `nvidia/cuda` base image vs `pytorch/pytorch`?
4. How do you install PyTorch with CUDA 12.1 support?
5. Why is running containers as root a security risk?

## Additional Resources

- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)

---

**Next:** [03-image-optimization.md](./03-image-optimization.md)
