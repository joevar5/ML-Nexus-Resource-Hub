# Multi-stage Dockerfile for LLM Deployment Platform
# Optimized for GPU inference with vLLM

# ============================================================================
# Stage 1: Base Image with CUDA Support
# ============================================================================
ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base as dependencies

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support first (large dependency)
# TODO: Update PyTorch version based on CUDA compatibility
RUN pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Split into layers for better caching
RUN pip install --no-cache-dir \
    vllm==0.4.2 \
    transformers==4.38.0 \
    accelerate==0.27.0 \
    bitsandbytes==0.42.0

RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.6.0 \
    pydantic-settings==2.1.0 \
    sse-starlette==1.8.2

RUN pip install --no-cache-dir \
    langchain==0.1.9 \
    langchain-community==0.0.24 \
    sentence-transformers==2.4.0 \
    tiktoken==0.6.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Application Build
# ============================================================================
FROM dependencies as builder

# Copy application source code
COPY src/ /app/src/
COPY prompts/ /app/prompts/

# Create necessary directories
RUN mkdir -p /models /data /var/log/llm-platform

# TODO: Optionally download and cache models at build time
# Uncomment the following lines to bake models into the image (increases image size significantly)
# ENV HF_HOME=/models/.cache
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"

# ============================================================================
# Stage 4: Production Image
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 as production

# Install runtime dependencies only (smaller image)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user for security
RUN useradd -m -u 1000 llmuser && \
    mkdir -p /models /data /var/log/llm-platform && \
    chown -R llmuser:llmuser /models /data /var/log/llm-platform

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder --chown=llmuser:llmuser /app/src ./src
COPY --from=builder --chown=llmuser:llmuser /app/prompts ./prompts

# Copy startup scripts
COPY --chown=llmuser:llmuser scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment variables
ENV PYTHONPATH=/app \
    HF_HOME=/models/.cache \
    TRANSFORMERS_CACHE=/models/.cache \
    TORCH_HOME=/models/.cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Switch to non-root user
USER llmuser

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 5: Development Image (Optional)
# ============================================================================
FROM builder as development

# Install development tools
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    black \
    ruff \
    mypy \
    pytest \
    pytest-asyncio \
    pytest-cov

# Expose additional ports for Jupyter
EXPOSE 8000 8001 8888

# Development user (root for easier development)
USER root

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]

# ============================================================================
# Build Instructions:
# ============================================================================
#
# Production build:
#   docker build --target production -t llm-deployment-platform:latest .
#
# Development build:
#   docker build --target development -t llm-deployment-platform:dev .
#
# Build with specific CUDA version:
#   docker build --build-arg CUDA_VERSION=12.1.0 -t llm-deployment-platform:latest .
#
# Build with model baked in (uncomment model download in builder stage):
#   docker build --target production -t llm-deployment-platform:mistral-7b .
#
# ============================================================================
# Run Instructions:
# ============================================================================
#
# Run with Docker:
#   docker run --gpus all -p 8000:8000 -p 8001:8001 \
#     -v $(pwd)/models:/models \
#     -v $(pwd)/data:/data \
#     -e LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
#     llm-deployment-platform:latest
#
# Run with Docker Compose:
#   docker-compose up
#
# ============================================================================
