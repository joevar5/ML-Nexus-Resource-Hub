# Use Python 3.11 slim base image for a small, secure container footprint
FROM python:3.11-slim-bookworm

# Set metadata labels
LABEL maintainer="AI Infrastructure Curriculum Team" \
      project="project-02-kubernetes-serving" \
      version="1.0"

# Set workspace directory
WORKDIR /app

# Install curl for container healthcheck and diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to optimize Docker layer caching
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Set production environment variables
ENV MODEL_NAME=resnet50 \
    MODEL_VERSION=1.0 \
    LOG_LEVEL=INFO \
    PORT=5000 \
    MODEL_LOAD_SECONDS=1.5 \
    SHUTDOWN_GRACE_SECONDS=2.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose HTTP port for the application and metrics endpoint
EXPOSE 5000

# Set default env variables for writeable paths
ENV TORCH_HOME=/tmp/torch \
    PORT=5000 \
    HOME=/tmp

# Run FastAPI application using Gunicorn with 1 worker process to fit memory limits
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-tmp-dir", "/tmp", "-k", "uvicorn.workers.UvicornWorker", "src.app:app"]
