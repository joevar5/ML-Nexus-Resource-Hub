# Multi-stage Dockerfile for ML Model Serving
# This Dockerfile uses multi-stage builds to create an optimized production image

# ==============================================================================
# Stage 1: Builder Stage
# ==============================================================================
# TODO: Choose appropriate base image
# Options:
# - python:3.11-slim (lightweight, ~120MB)
# - python:3.11 (full, ~900MB)
# - pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime (with CUDA)
#
# For CPU-only serving, use slim image
# For GPU serving, use CUDA image

FROM python:3.11-slim as builder

# TODO: Set working directory
# Standard practice: /app or /opt/app
WORKDIR /app

# TODO: Install system dependencies
# Common needs:
# - build-essential (for compiling Python packages)
# - curl (for health checks)
# - git (if installing from git repos)
#
# Example:
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# TODO: Install system dependencies here
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# TODO: Copy requirements file
# Copy only requirements first to leverage Docker layer caching
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .

# TODO: Install Python dependencies
# Best practices:
# - Use --no-cache-dir to reduce image size
# - Use --user to install in user directory
# - Consider using pip-tools for reproducible builds
#
# For PyTorch CPU-only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Example:
# RUN pip install --no-cache-dir --user -r requirements.txt

# TODO: Install dependencies here
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Stage 2: Runtime Stage
# ==============================================================================
# TODO: Use same base image as builder for compatibility
FROM python:3.11-slim

# TODO: Set metadata labels
# Good practice: Add labels for maintainability
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="ML Model Serving API"

# TODO: Set working directory
WORKDIR /app

# TODO: Create non-root user for security
# Running as non-root is a security best practice
# Steps:
# 1. Create user and group
# 2. Create app directory with correct permissions
# 3. Switch to non-root user
#
# Example:
# RUN groupadd -r appuser && useradd -r -g appuser appuser
# RUN chown -R appuser:appuser /app
# USER appuser

# TODO: Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# TODO: Copy Python packages from builder stage
# Copy installed packages from builder to keep image small
# The --from=builder flag copies from the previous stage
COPY --from=builder /root/.local /home/appuser/.local

# TODO: Update PATH to include user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# TODO: Copy application code
# Copy only necessary files to keep image small
# Consider using .dockerignore to exclude unnecessary files
COPY --chown=appuser:appuser src/ ./src/
# TODO: Copy other necessary files (models, configs, etc.)
# COPY --chown=appuser:appuser models/ ./models/

# TODO: Set environment variables
# Common environment variables:
# - PYTHONUNBUFFERED=1 (show Python logs in real-time)
# - PYTHONDONTWRITEBYTECODE=1 (don't create .pyc files)
# - MODEL_NAME=resnet18
# - DEVICE=cpu
# - PORT=8000
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME=resnet18 \
    DEVICE=cpu \
    PORT=8000

# TODO: Expose port
# Document which port the application listens on
# This doesn't actually publish the port, just documents it
EXPOSE 8000

# TODO: Add health check
# Docker will periodically run this command to check container health
# Kubernetes can also use this for liveness/readiness probes
#
# Example:
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# TODO: Implement health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# TODO: Switch to non-root user
USER appuser

# TODO: Set entrypoint and command
# ENTRYPOINT: The main command that always runs
# CMD: Default arguments (can be overridden)
#
# Option 1: Direct Python execution
# CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
#
# Option 2: Shell script for more complex startup
# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["--workers", "4"]
#
# For production, use:
# CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
#
# For development (with auto-reload):
# CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# TODO: Set command to start the application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Build Instructions
# ==============================================================================
# To build this image:
# docker build -t ml-model-serving:v1.0 .
#
# To build with specific model:
# docker build --build-arg MODEL_NAME=resnet50 -t ml-model-serving:v1.0 .
#
# To run the container:
# docker run -p 8000:8000 ml-model-serving:v1.0
#
# To run with GPU:
# docker run --gpus all -p 8000:8000 ml-model-serving-gpu:v1.0
#
# ==============================================================================

# ==============================================================================
# Optimization Tips
# ==============================================================================
# 1. Use multi-stage builds (already implemented above)
# 2. Order commands from least to most frequently changing
# 3. Combine RUN commands to reduce layers
# 4. Use .dockerignore to exclude unnecessary files
# 5. Use specific base image tags (not 'latest')
# 6. Clean up package manager caches
# 7. Consider using distroless images for production
# 8. Use COPY instead of ADD unless you need ADD's features
#
# Target image size: <1GB for CPU, <3GB for GPU
# ==============================================================================

# ==============================================================================
# Security Best Practices
# ==============================================================================
# ✅ Run as non-root user
# ✅ Use specific image versions
# ✅ Don't include secrets in the image
# ✅ Use HEALTHCHECK for container health
# ✅ Minimize installed packages
# ✅ Scan images for vulnerabilities
#
# Scan with:
# docker scan ml-model-serving:v1.0
# trivy image ml-model-serving:v1.0
# ==============================================================================

# ==============================================================================
# Alternative: GPU Support
# ==============================================================================
# For GPU support, replace the base image:
#
# FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime as builder
# ...
# FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
#
# And set DEVICE=cuda in environment variables
# ==============================================================================

# ==============================================================================
# Alternative: Production Optimization with Distroless
# ==============================================================================
# For even smaller images, consider distroless:
#
# FROM gcr.io/distroless/python3-debian11
# COPY --from=builder /root/.local /root/.local
# COPY --from=builder /app /app
# ENV PATH=/root/.local/bin:$PATH
# WORKDIR /app
# CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0"]
#
# Note: Distroless images don't have shell, so debugging is harder
# ==============================================================================
