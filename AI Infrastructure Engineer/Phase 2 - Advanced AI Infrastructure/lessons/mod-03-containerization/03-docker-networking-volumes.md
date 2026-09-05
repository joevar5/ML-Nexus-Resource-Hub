# Lesson 04: Docker Networking and Volumes

**Duration:** 5 hours
**Objectives:** Master Docker networking and persistent storage for ML applications

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand Docker networking modes and when to use each
2. Enable container-to-container communication
3. Map ports for ML inference services
4. Use volumes for persistent model storage
5. Understand bind mounts vs named volumes
6. Share data between containers
7. Build multi-container ML applications with proper networking

## Docker Networking Fundamentals

### Why Networking Matters for ML

ML applications often require multiple services:
- **Model server** serving predictions
- **API gateway** handling requests
- **Database** storing prediction logs
- **Cache** (Redis) for feature caching
- **Message queue** for async processing

These services need to communicate reliably and securely.

### Docker Network Drivers

Docker provides several network drivers:

| Driver | Use Case | Isolation | Performance |
|--------|----------|-----------|-------------|
| **bridge** | Single host, container-to-container | Isolated network | Good |
| **host** | Container uses host network | No isolation | Best |
| **none** | No networking | Complete isolation | N/A |
| **overlay** | Multi-host (Swarm/Kubernetes) | Across hosts | Good |

## Bridge Network (Default)

### How Bridge Networks Work

```
┌──────────────────────────────────────────────┐
│            Host Machine                       │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │     Docker Bridge Network           │    │
│  │     (172.17.0.0/16)                 │    │
│  │                                      │    │
│  │  ┌──────────┐      ┌──────────┐    │    │
│  │  │Container1│      │Container2│    │    │
│  │  │172.17.0.2│─────▶│172.17.0.3│    │    │
│  │  └──────────┘      └──────────┘    │    │
│  └─────────────────────────────────────┘    │
│             │                                 │
│             ▼                                 │
│      Host Network                            │
└──────────────────────────────────────────────┘
```

### Default Bridge Network

Every container on the default bridge can communicate:

```bash
# Start container 1
docker run -d --name model-server nginx

# Start container 2 (can ping container 1 by IP)
docker run --name client alpine ping 172.17.0.2
```

**Limitations:**
- Must use IP addresses (no DNS)
- All containers share same network
- Less secure

### Custom Bridge Networks (Recommended)

```bash
# Create custom network
docker network create ml-network

# Run containers on custom network
docker run -d --name model-server --network ml-network my-model:v1
docker run -d --name api-gateway --network ml-network my-api:v1

# Containers can reach each other by name!
# Inside api-gateway container:
curl http://model-server:8000/predict
```

**Benefits:**
- **DNS resolution** by container name
- **Better isolation** (only containers on network can communicate)
- **Easy service discovery**

### Creating and Managing Networks

```bash
# Create network
docker network create ml-network

# List networks
docker network ls

# Inspect network
docker network inspect ml-network

# Connect running container to network
docker network connect ml-network my-container

# Disconnect from network
docker network disconnect ml-network my-container

# Remove network
docker network rm ml-network
```

## Port Mapping

### Publishing Ports

Map container ports to host ports:

```bash
# Format: -p HOST_PORT:CONTAINER_PORT

# Map container port 8000 to host port 8000
docker run -p 8000:8000 my-ml-api

# Map to different host port
docker run -p 8080:8000 my-ml-api

# Map to localhost only (more secure)
docker run -p 127.0.0.1:8000:8000 my-ml-api

# Map multiple ports
docker run -p 8000:8000 -p 8001:8001 my-ml-api

# Map random host port (Docker chooses)
docker run -p 8000 my-ml-api
```

### Finding Mapped Ports

```bash
# See port mappings
docker port my-container

# Output:
# 8000/tcp -> 0.0.0.0:8000
```

### Example: ML Inference API

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn torch

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Running:**
```bash
# Build image
docker build -t ml-api .

# Run with port mapping
docker run -d -p 8000:8000 --name ml-api ml-api

# Test API
curl http://localhost:8000/health
```

## Host Network Mode

### When to Use Host Network

Host mode removes network isolation:

```bash
# Container uses host's network directly
docker run --network host my-ml-api
```

**Use cases:**
- Maximum network performance (no NAT overhead)
- Need to bind to specific host interfaces
- Testing locally

**Drawbacks:**
- No port isolation (conflicts possible)
- Less secure
- Less portable (host-dependent)

**Example: High-Performance Inference**

```bash
# For latency-critical ML serving
docker run --network host \
    --gpus all \
    nvidia/triton-inference-server:latest
```

## Multi-Container Communication

### Scenario: ML Application Stack

```
┌─────────────┐
│   Client    │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ API Gateway │ (port 8000)
└─────┬───────┘
      │
      ├─────────────────┐
      ▼                 ▼
┌─────────────┐   ┌─────────────┐
│Model Server │   │  Redis      │
│  (PyTorch)  │   │  (Cache)    │
└─────┬───────┘   └─────────────┘
      │
      ▼
┌─────────────┐
│ PostgreSQL  │
│   (Logs)    │
└─────────────┘
```

### Implementation

**1. Create network:**
```bash
docker network create ml-stack
```

**2. Start PostgreSQL:**
```bash
docker run -d \
    --name postgres \
    --network ml-stack \
    -e POSTGRES_PASSWORD=secret \
    -e POSTGRES_DB=predictions \
    postgres:15
```

**3. Start Redis:**
```bash
docker run -d \
    --name redis \
    --network ml-stack \
    redis:7-alpine
```

**4. Start Model Server:**
```bash
docker run -d \
    --name model-server \
    --network ml-stack \
    --gpus all \
    my-model-server:v1
```

**5. Start API Gateway:**
```bash
docker run -d \
    --name api-gateway \
    --network ml-stack \
    -p 8000:8000 \
    -e MODEL_URL=http://model-server:8000 \
    -e REDIS_URL=redis://redis:6379 \
    -e DB_URL=postgresql://postgres:secret@postgres:5432/predictions \
    my-api-gateway:v1
```

**API Gateway Code (Python):**
```python
import os
import httpx
import redis
from fastapi import FastAPI

app = FastAPI()

# Service discovery via environment variables
MODEL_URL = os.getenv("MODEL_URL")
redis_client = redis.from_url(os.getenv("REDIS_URL"))

@app.post("/predict")
async def predict(data: dict):
    # Check cache
    cache_key = f"prediction:{hash(str(data))}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"prediction": cached, "cached": True}

    # Call model server
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{MODEL_URL}/predict", json=data)
        result = response.json()

    # Cache result
    redis_client.setex(cache_key, 3600, result["prediction"])

    return {"prediction": result["prediction"], "cached": False}
```

## Docker Volumes

### Why Volumes?

Container filesystems are **ephemeral**:
- Data is lost when container is removed
- Cannot share data between containers
- Not suitable for models, datasets, logs

**Volumes solve this:**
- Persistent storage (survives container deletion)
- Shareable between containers
- Better performance than bind mounts
- Managed by Docker

### Volume Types

```
┌──────────────────────────────────────────────────────┐
│                    Volume Types                       │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. Named Volumes (Managed by Docker)                │
│     docker volume create my-volume                   │
│     Location: /var/lib/docker/volumes/               │
│                                                       │
│  2. Anonymous Volumes                                │
│     Automatically created, cleaned up                │
│                                                       │
│  3. Bind Mounts (Host directory)                     │
│     -v /host/path:/container/path                    │
│     Full control, host-dependent                     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Named Volumes

**Create and use:**

```bash
# Create volume
docker volume create model-weights

# List volumes
docker volume ls

# Inspect volume
docker volume inspect model-weights

# Use volume
docker run -v model-weights:/app/models my-ml-app

# Remove volume
docker volume rm model-weights
```

**Example: Persistent Model Storage**

```bash
# Create volume for models
docker volume create ml-models

# Download model into volume
docker run --rm \
    -v ml-models:/models \
    alpine sh -c "wget https://example.com/model.pth -O /models/model.pth"

# Use model in inference container
docker run -d \
    -v ml-models:/app/models \
    -p 8000:8000 \
    my-inference-server

# Model persists even if container is deleted!
```

### Bind Mounts

Map host directory to container:

```bash
# Format: -v /absolute/host/path:/container/path

# Mount current directory
docker run -v $(pwd):/app my-image

# Mount specific directory
docker run -v /data/models:/app/models my-image

# Read-only mount
docker run -v /data/models:/app/models:ro my-image
```

**When to use bind mounts:**
- Development (live code reloading)
- Access to specific host directories
- Configuration files
- Datasets on host

**Example: Development Workflow**

```bash
# Mount code for live reloading
docker run -it --rm \
    -v $(pwd)/src:/app/src \
    -v $(pwd)/models:/app/models \
    -p 8000:8000 \
    -e RELOAD=true \
    my-ml-api
```

### tmpfs Mounts (Temporary Storage)

Store data in host memory (not persisted):

```bash
# Mount tmpfs for temporary data
docker run --tmpfs /tmp:size=1g my-image
```

**Use cases:**
- Sensitive data (passwords, tokens)
- Temporary processing files
- Fast I/O needs

## Volumes for ML Workloads

### Scenario 1: Model Storage

```bash
# Create volume for model weights
docker volume create model-artifacts

# Training container saves model
docker run --rm \
    -v model-artifacts:/app/outputs \
    --gpus all \
    my-training-image python train.py

# Inference container loads model
docker run -d \
    -v model-artifacts:/app/models \
    -p 8000:8000 \
    my-inference-image
```

### Scenario 2: Dataset Management

```bash
# Mount large dataset from host
docker run --rm \
    -v /data/imagenet:/dataset:ro \
    -v training-logs:/logs \
    --gpus all \
    my-training-image python train.py --data /dataset
```

### Scenario 3: Sharing Between Containers

```bash
# Create shared volume
docker volume create shared-data

# Container 1: Preprocess data
docker run --rm \
    -v shared-data:/data \
    my-preprocessor python preprocess.py --output /data/processed

# Container 2: Train on processed data
docker run --rm \
    -v shared-data:/data \
    --gpus all \
    my-trainer python train.py --input /data/processed
```

### Scenario 4: Logs and Metrics

```bash
# Persistent logs
docker run -d \
    -v app-logs:/app/logs \
    -v prometheus-data:/prometheus \
    my-ml-service
```

## Volume Best Practices for ML

### 1. Don't Include Large Files in Images

❌ **Bad:**
```dockerfile
COPY models/large-model.pth /app/models/  # 5GB added to image!
```

✅ **Good:**
```bash
# Download model at runtime
docker run -v models:/app/models my-image
# Or download in startup script
```

### 2. Use Read-Only Mounts for Safety

```bash
# Prevent accidental modification of dataset
docker run -v /data/dataset:/dataset:ro my-training-image
```

### 3. Separate Code, Data, and Outputs

```bash
docker run \
    -v $(pwd)/src:/app/src:ro \      # Code (read-only)
    -v /data/training:/data:ro \     # Data (read-only)
    -v model-outputs:/outputs \       # Outputs (read-write)
    --gpus all \
    my-training-image
```

### 4. Volume Backup

```bash
# Backup volume to tar file
docker run --rm \
    -v model-weights:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore volume from backup
docker run --rm \
    -v model-weights:/data \
    -v $(pwd):/backup \
    alpine sh -c "cd /data && tar xzf /backup/models-backup.tar.gz"
```

## Hands-On Exercise: Multi-Container ML Application

### Objective

Build a multi-container ML application with:
- **Model server** (PyTorch inference)
- **API gateway** (FastAPI)
- **Redis** (feature caching)
- **PostgreSQL** (prediction logging)

### Architecture

```
User → API Gateway (8000) → Model Server (internal)
                ↓              ↓
             Redis          PostgreSQL
           (cache)           (logs)
```

### Implementation Steps

**1. Create network:**
```bash
docker network create ml-app
```

**2. Start services:**

```bash
# PostgreSQL
docker run -d \
    --name postgres \
    --network ml-app \
    -e POSTGRES_PASSWORD=mlpassword \
    -e POSTGRES_DB=predictions \
    -v postgres-data:/var/lib/postgresql/data \
    postgres:15

# Redis
docker run -d \
    --name redis \
    --network ml-app \
    -v redis-data:/data \
    redis:7-alpine

# Model Server (you'll build this)
docker build -t model-server ./model-server
docker run -d \
    --name model-server \
    --network ml-app \
    -v model-weights:/app/models \
    model-server

# API Gateway (you'll build this)
docker build -t api-gateway ./api-gateway
docker run -d \
    --name api-gateway \
    --network ml-app \
    -p 8000:8000 \
    -e MODEL_URL=http://model-server:8000 \
    -e REDIS_URL=redis://redis:6379 \
    -e DATABASE_URL=postgresql://postgres:mlpassword@postgres:5432/predictions \
    api-gateway
```

**3. Test:**

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'

# Check logs in database
docker exec -it postgres psql -U postgres -d predictions \
    -c "SELECT * FROM predictions LIMIT 10;"
```

### TODO: Complete the Exercise

**model-server/Dockerfile:**
```dockerfile
# TODO: Create Dockerfile for model server
# - Use PyTorch base image
# - Copy model code
# - Mount volume for model weights
# - Expose port 8000
```

**api-gateway/Dockerfile:**
```dockerfile
# TODO: Create Dockerfile for API gateway
# - Use Python 3.11-slim
# - Install FastAPI, Redis, PostgreSQL clients
# - Copy API code
# - Expose port 8000
```

## Troubleshooting Networking and Volumes

### Network Issues

**Problem: Containers can't communicate**

```bash
# Check if containers are on same network
docker network inspect ml-network

# Verify DNS resolution
docker exec container1 ping container2

# Check firewall rules
docker exec container1 telnet container2 8000
```

**Problem: Port already in use**

```bash
# Find what's using the port
sudo lsof -i :8000

# Use different host port
docker run -p 8001:8000 my-image
```

### Volume Issues

**Problem: Permission denied**

```bash
# Check volume permissions
docker run --rm -v my-volume:/data alpine ls -la /data

# Fix ownership
docker run --rm -v my-volume:/data alpine chown -R 1000:1000 /data
```

**Problem: Volume not mounting**

```bash
# Inspect volume
docker volume inspect my-volume

# Verify mount
docker inspect my-container | grep -A 10 Mounts
```

## Summary

In this lesson, you learned:

1. **Docker Networks** - Bridge, host, custom networks for service discovery
2. **Port Mapping** - Exposing ML services to external traffic
3. **Multi-Container Apps** - Connecting model servers, APIs, databases
4. **Volumes** - Named volumes vs bind mounts for persistent storage
5. **ML-Specific Patterns** - Model storage, dataset management, log persistence

**Key Takeaways:**
- Use custom bridge networks for service discovery
- Named volumes for persistent data (models, logs)
- Bind mounts for development
- Read-only mounts for datasets
- Separate networks for security

## What's Next?

In the next lesson, **05-docker-compose.md**, you'll learn:
- Define multi-container applications in YAML
- Manage entire ML stacks with single commands
- Environment variables and configuration
- Health checks and dependencies
- Scaling services

---

## Self-Check Questions

1. What's the difference between bridge and host networking?
2. How do containers on a custom network discover each other?
3. When would you use a named volume vs a bind mount?
4. How do you share data between two containers?
5. What's the format for port mapping?

## Additional Resources

- [Docker Networking Overview](https://docs.docker.com/network/)
- [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
- [Container Networking Best Practices](https://docs.docker.com/network/network-tutorial-standalone/)

---

**Next:** [05-docker-compose.md](./05-docker-compose.md)
