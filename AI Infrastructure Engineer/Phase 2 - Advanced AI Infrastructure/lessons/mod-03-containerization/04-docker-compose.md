# Lesson 05: Docker Compose for ML Applications

**Duration:** 5 hours
**Objectives:** Orchestrate multi-container ML applications with Docker Compose

## Learning Objectives

By the end of this lesson, you will be able to:

1. Write docker-compose.yml files for ML stacks
2. Define multi-service applications (API, model, database, cache)
3. Manage environment variables and secrets
4. Configure service dependencies and health checks
5. Use Compose for development and testing workflows
6. Scale services with Docker Compose
7. Debug and troubleshoot Compose applications

## Introduction to Docker Compose

### What is Docker Compose?

**Docker Compose** is a tool for defining and running multi-container Docker applications using a YAML configuration file.

**Without Compose:**
```bash
# Start each service manually
docker network create ml-network
docker run -d --name postgres --network ml-network ...
docker run -d --name redis --network ml-network ...
docker run -d --name model-server --network ml-network ...
docker run -d --name api-gateway --network ml-network -p 8000:8000 ...
```

**With Compose:**
```bash
# Single command starts entire stack
docker compose up
```

### Why Use Docker Compose for ML?

**Benefits:**
1. **Declarative configuration** - Define entire stack in one file
2. **Reproducibility** - Same stack every time
3. **Development workflow** - Quick setup for local development
4. **Testing** - Spin up stack for integration tests
5. **Documentation** - compose.yml documents your architecture
6. **Easy sharing** - Team members can run stack instantly

**Perfect for:**
- Local development of ML applications
- Integration testing with databases and services
- Demo environments
- Small production deployments (1-3 servers)

**Not ideal for:**
- Large-scale production (use Kubernetes)
- Multi-host deployments (use Swarm or K8s)

## Docker Compose Basics

### Installation

```bash
# Docker Compose v2 (included with Docker Desktop)
docker compose version

# If not installed, install plugin
sudo apt-get install docker-compose-plugin

# Verify
docker compose version
# Output: Docker Compose version v2.20.0
```

### Basic compose.yml Structure

```yaml
version: '3.8'  # Optional in newer versions

services:
  # Service definitions
  service1:
    image: nginx
    ports:
      - "80:80"

  service2:
    build: ./app
    depends_on:
      - service1

volumes:
  # Named volumes

networks:
  # Custom networks
```

### Simple Example: Web App + Database

```yaml
version: '3.8'

services:
  web:
    image: python:3.11-slim
    command: python -m http.server 8000
    ports:
      - "8000:8000"
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```

**Running:**
```bash
# Start all services
docker compose up

# Start in detached mode
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
```

## ML Application Stack with Docker Compose

### Architecture

```
┌─────────────────────────────────────────────┐
│           ML Application Stack               │
├─────────────────────────────────────────────┤
│                                              │
│  API Gateway (FastAPI) ──────┐             │
│       ↓                       ↓              │
│  Model Server (PyTorch)   Redis (Cache)    │
│       ↓                       ↓              │
│  PostgreSQL (Logs)    Prometheus (Metrics)  │
│                                              │
└─────────────────────────────────────────────┘
```

### Complete ML Stack Example

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # ===================================
  # API Gateway
  # ===================================
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_SERVER_URL=http://model-server:8000
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:mlpassword@postgres:5432/predictions
      - LOG_LEVEL=info
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      model-server:
        condition: service_healthy
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # ===================================
  # Model Server (PyTorch)
  # ===================================
  model-server:
    build:
      context: ./model-server
      dockerfile: Dockerfile
    environment:
      - MODEL_PATH=/models/model.pth
      - BATCH_SIZE=32
      - GPU_ENABLED=false
    volumes:
      - model-weights:/models
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  # ===================================
  # PostgreSQL Database
  # ===================================
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=mlpassword
      - POSTGRES_DB=predictions
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===================================
  # Redis Cache
  # ===================================
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ===================================
  # Prometheus Monitoring
  # ===================================
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    networks:
      - ml-network
    restart: unless-stopped

  # ===================================
  # Grafana Dashboards
  # ===================================
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - ml-network
    restart: unless-stopped

# ===================================
# Volumes
# ===================================
volumes:
  model-weights:
    driver: local
  postgres-data:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

# ===================================
# Networks
# ===================================
networks:
  ml-network:
    driver: bridge
```

## Service Configuration

### Building Images

```yaml
services:
  api:
    # Use pre-built image
    image: my-api:v1.0

    # Or build from Dockerfile
    build:
      context: ./api
      dockerfile: Dockerfile

    # Build with arguments
    build:
      context: ./api
      args:
        PYTHON_VERSION: 3.11
        BUILD_DATE: "2024-01-15"

    # Build target (multi-stage)
    build:
      context: ./api
      target: production
```

### Environment Variables

**Method 1: Inline in compose.yml**
```yaml
services:
  api:
    environment:
      - DEBUG=false
      - MODEL_NAME=resnet50
      - API_KEY=secret123  # Don't do this for real secrets!
```

**Method 2: Environment file**

**.env:**
```bash
DEBUG=false
MODEL_NAME=resnet50
DB_PASSWORD=secret123
```

**compose.yml:**
```yaml
services:
  api:
    env_file:
      - .env
      - .env.production  # Can have multiple
```

**Method 3: Variable substitution**

**.env:**
```bash
POSTGRES_PASSWORD=mysecret
```

**compose.yml:**
```yaml
services:
  db:
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
```

### Ports and Networking

```yaml
services:
  api:
    # Publish to host
    ports:
      - "8000:8000"          # host:container
      - "127.0.0.1:8001:8001"  # localhost only
      - "8002"               # random host port

    # Expose to other containers (not host)
    expose:
      - "8000"

    # Join networks
    networks:
      - frontend
      - backend

networks:
  frontend:
  backend:
```

### Volumes and Bind Mounts

```yaml
services:
  model-server:
    volumes:
      # Named volume
      - model-data:/app/models

      # Bind mount (development)
      - ./src:/app/src:ro  # read-only

      # Anonymous volume
      - /app/logs

      # tmpfs (in-memory)
    tmpfs:
      - /tmp

volumes:
  model-data:
```

### Dependencies and Startup Order

```yaml
services:
  api:
    depends_on:
      # Simple dependency (starts after db)
      - db

      # With condition (waits for health check)
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  db:
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Resource Limits

```yaml
services:
  model-server:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

      # GPU support
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Docker Compose Commands

### Basic Commands

```bash
# Start services
docker compose up

# Start in detached mode
docker compose up -d

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# View logs
docker compose logs

# Follow logs
docker compose logs -f

# Logs for specific service
docker compose logs -f api

# List running services
docker compose ps

# Execute command in service
docker compose exec api bash

# Rebuild images
docker compose build

# Rebuild and restart
docker compose up --build

# Scale services
docker compose up --scale model-server=3
```

### Advanced Commands

```bash
# Validate compose file
docker compose config

# View resolved config
docker compose config --resolve-image-digests

# Pull images
docker compose pull

# Push images
docker compose push

# Restart specific service
docker compose restart api

# Pause services
docker compose pause

# Unpause services
docker compose unpause

# View resource usage
docker compose top
```

## Development Workflow with Compose

### Hot Reloading for Development

**compose.dev.yml:**
```yaml
version: '3.8'

services:
  api:
    build:
      context: ./api
      target: development  # Multi-stage build target
    volumes:
      - ./api/src:/app/src  # Mount source for live reload
    environment:
      - RELOAD=true
      - DEBUG=true
    command: uvicorn main:app --reload --host 0.0.0.0
    ports:
      - "8000:8000"
      - "5678:5678"  # Debugger port

  model-server:
    build:
      context: ./model-server
      target: development
    volumes:
      - ./model-server/src:/app/src
    environment:
      - DEBUG=true
```

**Usage:**
```bash
# Development mode
docker compose -f compose.dev.yml up

# Production mode
docker compose -f compose.yml up
```

### Testing with Compose

**compose.test.yml:**
```yaml
version: '3.8'

services:
  api:
    build:
      context: ./api
      target: test
    environment:
      - TESTING=true
      - DATABASE_URL=postgresql://postgres:test@test-db:5432/test
    command: pytest tests/ -v
    depends_on:
      - test-db

  test-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=test
      - POSTGRES_DB=test
```

**Run tests:**
```bash
# Run tests
docker compose -f compose.test.yml up --abort-on-container-exit

# Clean up
docker compose -f compose.test.yml down -v
```

### Multiple Compose Files

```bash
# Override base config with dev config
docker compose -f compose.yml -f compose.dev.yml up

# Production with monitoring
docker compose -f compose.yml -f compose.monitoring.yml up
```

## Scaling ML Services

### Scaling Stateless Services

```yaml
services:
  model-server:
    build: ./model-server
    deploy:
      replicas: 3  # Start 3 instances
```

**Runtime scaling:**
```bash
# Scale up
docker compose up --scale model-server=5

# Scale down
docker compose up --scale model-server=2
```

### Load Balancing

**Add nginx as load balancer:**

**compose.yml:**
```yaml
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
    depends_on:
      - model-server

  model-server:
    build: ./model-server
    deploy:
      replicas: 3
```

**nginx.conf:**
```nginx
upstream model_servers {
    server model-server:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://model_servers;
    }
}
```

## Hands-On Exercise: Build Complete ML Stack

### Objective

Create a production-ready ML application stack with:
- FastAPI serving predictions
- PyTorch model server
- Redis for caching
- PostgreSQL for logging
- Prometheus + Grafana for monitoring

### Project Structure

```
ml-stack/
├── docker-compose.yml
├── .env
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── model-server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── serve.py
├── postgres/
│   └── init.sql
└── prometheus/
    └── prometheus.yml
```

### TODO: Implement the Stack

**1. Create docker-compose.yml** (see example above)

**2. API Service (api/main.py):**
```python
from fastapi import FastAPI
import httpx
import redis
import psycopg2
import os

app = FastAPI()

# Configuration from environment
MODEL_URL = os.getenv("MODEL_SERVER_URL")
redis_client = redis.from_url(os.getenv("REDIS_URL"))

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: dict):
    # TODO: Implement prediction with caching
    # 1. Check Redis cache
    # 2. If not cached, call model server
    # 3. Cache result
    # 4. Log to PostgreSQL
    # 5. Return prediction
    pass
```

**3. Model Server (model-server/serve.py):**
```python
from fastapi import FastAPI
import torch

app = FastAPI()

# TODO: Load model
# model = torch.load("/models/model.pth")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    # TODO: Run inference
    pass
```

**4. Test the stack:**
```bash
# Start stack
docker compose up -d

# Check services
docker compose ps

# Test API
curl http://localhost:8000/health

# View logs
docker compose logs -f api

# Monitor metrics
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana

# Stop stack
docker compose down
```

## Troubleshooting Docker Compose

### Common Issues

**Issue 1: Services not starting**
```bash
# View detailed logs
docker compose logs service-name

# Check service status
docker compose ps

# Restart service
docker compose restart service-name
```

**Issue 2: Network errors**
```bash
# Inspect network
docker network inspect ml-stack_default

# Test connectivity
docker compose exec api ping model-server
```

**Issue 3: Volume permissions**
```bash
# Check volume
docker volume inspect ml-stack_model-data

# Fix permissions
docker compose exec api chown -R appuser:appuser /data
```

**Issue 4: Port conflicts**
```bash
# Change port in compose.yml
ports:
  - "8001:8000"  # Use different host port
```

## Production Considerations

### Security

```yaml
services:
  api:
    # Don't run as root
    user: "1000:1000"

    # Read-only root filesystem
    read_only: true

    # No new privileges
    security_opt:
      - no-new-privileges:true

    # Limit capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

### Secrets Management

**Use Docker secrets (Swarm) or external secret managers:**

```yaml
services:
  api:
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### Logging

```yaml
services:
  api:
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## Summary

In this lesson, you learned:

1. **Docker Compose Basics** - Define multi-container apps in YAML
2. **ML Stack Configuration** - API, model server, database, cache, monitoring
3. **Service Dependencies** - Health checks and startup order
4. **Development Workflow** - Hot reloading, testing, multiple configs
5. **Scaling** - Replicate services and load balancing
6. **Troubleshooting** - Debug Compose applications

**Key Takeaways:**
- Compose simplifies multi-container development
- Use health checks and depends_on for reliability
- Separate configs for dev/test/prod
- Named volumes for persistence
- Environment files for configuration

## What's Next?

In the next lesson, **06-container-registries.md**, you'll learn:
- Push images to Docker Hub, ECR, GCR, ACR
- Image tagging strategies
- CI/CD integration with registries
- Private registries for enterprises
- Image scanning for vulnerabilities

---

## Self-Check Questions

1. What's the difference between `depends_on` and `depends_on` with condition?
2. How do you override configurations for different environments?
3. What's the purpose of health checks in Compose?
4. How do you scale a service to 5 replicas?
5. What's the difference between named volumes and bind mounts?

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Compose File Reference](https://docs.docker.com/compose/compose-file/)
- [Awesome Compose Examples](https://github.com/docker/awesome-compose)

---

**Next:** [06-container-registries.md](./06-container-registries.md)
