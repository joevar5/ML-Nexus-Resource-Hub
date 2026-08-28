# Exercise 02: Multi-Container ML Application

**Estimated Time:** 3-4 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01 completed, Docker Compose installed

## Objective

Build a complete ML application stack with Docker Compose containing:
- FastAPI ML inference service
- PostgreSQL database for logging predictions
- Redis cache for results
- Prometheus for metrics

## Architecture

```
User → API (FastAPI) → Model Server
           ↓              ↓
       PostgreSQL      Redis
           ↓
       Prometheus
```

## Project Structure

```
ml-stack/
├── docker-compose.yml
├── .env
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── postgres/
│   └── init.sql
├── prometheus/
│   └── prometheus.yml
└── README.md
```

## Tasks

### Task 1: Create Docker Compose File

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # TODO: Define api service
  # - Build from ./api
  # - Expose port 8000
  # - Environment variables for DB and Redis
  # - Depends on postgres and redis
  # - Add health check

  # TODO: Define postgres service
  # - Use postgres:15-alpine
  # - Set POSTGRES_PASSWORD, POSTGRES_DB
  # - Mount volume for data persistence
  # - Mount init.sql for schema
  # - Add health check

  # TODO: Define redis service
  # - Use redis:7-alpine
  # - Mount volume for persistence
  # - Add health check

  # TODO: Define prometheus service (stretch goal)
  # - Use prom/prometheus:latest
  # - Mount prometheus.yml config
  # - Expose port 9090

# TODO: Define named volumes

# TODO: Define custom network
```

### Task 2: API Service

**api/Dockerfile:**
```dockerfile
# TODO: Create optimized Dockerfile
# - Use multi-stage build
# - Run as non-root user
# - Health check
```

**api/requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
python-multipart==0.0.6
redis==5.0.1
psycopg2-binary==2.9.9
prometheus-client==0.19.0
```

**api/main.py:**
```python
from fastapi import FastAPI, HTTPException
import redis
import psycopg2
import os
from datetime import datetime

app = FastAPI()

# TODO: Initialize Redis client
# redis_client = redis.from_url(os.getenv("REDIS_URL"))

# TODO: Initialize PostgreSQL connection
# db_conn = psycopg2.connect(os.getenv("DATABASE_URL"))

@app.get("/health")
def health():
    # TODO: Check Redis and Postgres connectivity
    pass

@app.post("/predict")
async def predict(data: dict):
    # TODO: Implement prediction logic
    # 1. Check Redis cache
    # 2. If not cached, run model inference
    # 3. Cache result in Redis
    # 4. Log prediction to PostgreSQL
    pass

# TODO: Add metrics endpoint for Prometheus
```

### Task 3: Database Schema

**postgres/init.sql:**
```sql
-- TODO: Create predictions table
-- Columns: id, input_data, prediction, confidence, timestamp

CREATE TABLE IF NOT EXISTS predictions (
    -- TODO: Define schema
);

-- TODO: Create index on timestamp for faster queries
```

### Task 4: Prometheus Configuration

**prometheus/prometheus.yml:**
```yaml
# TODO: Configure Prometheus to scrape API metrics
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    # TODO: Configure scrape target
```

## Testing

```bash
# Start the stack
docker compose up -d

# Check all services are running
docker compose ps

# View logs
docker compose logs -f api

# Test health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'

# Make same prediction again (should be cached)
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'

# Check database
docker compose exec postgres psql -U postgres -d predictions \
    -c "SELECT * FROM predictions LIMIT 10;"

# Check Redis cache
docker compose exec redis redis-cli KEYS "*"

# View metrics (stretch goal)
open http://localhost:9090

# Stop stack
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Success Criteria

- [ ] All services start successfully
- [ ] API health check passes
- [ ] Predictions are logged to PostgreSQL
- [ ] Duplicate predictions return cached results
- [ ] Redis persists data across restarts
- [ ] Services can communicate by name
- [ ] Graceful shutdown works
- [ ] Prometheus collects metrics (stretch)

## Stretch Goals

1. **Add Grafana**: Visualize metrics from Prometheus
2. **Load balancing**: Run multiple API replicas with nginx
3. **Async predictions**: Use message queue (RabbitMQ/Kafka)
4. **Separate dev/prod configs**: docker-compose.override.yml
5. **Model versioning**: Store multiple model versions

## Common Issues

**Services can't connect:**
- Verify all services are on same network
- Check service names match (postgres, redis, api)
- Wait for health checks to pass before connecting

**Database connection fails:**
- Ensure POSTGRES_PASSWORD matches in both services
- Check DATABASE_URL format: postgresql://user:password@host:port/database
- Verify init.sql executed (check docker compose logs postgres)

**Cache not working:**
- Check Redis connection with docker compose exec redis redis-cli ping
- Verify REDIS_URL environment variable
- Check for connection errors in logs

---

**Next Exercise:** [exercise-03-gpu-container.md](./exercise-03-gpu-container.md) (GPU-enabled containers for ML workloads)
**Related Exercises:** [exercise-04-container-security/](./exercise-04-container-security/) and [exercise-08-production-model-serving](../../mod-101-foundations/exercises/exercise-08-production-model-serving/) for production-readiness depth
