# Lab 03: Compose a Multi-Service Stack

**Duration:** 75 min  **Prerequisites:** Lab 02 complete

## Objective
Use Docker Compose to stand up a representative ML serving stack: an API container, a Redis feature-store cache, and a Postgres metadata DB, with healthchecks and a dependency-ordered startup.

## Steps

### 1. compose.yaml
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      REDIS_URL: redis://cache:6379/0
      POSTGRES_URL: postgresql://app:app@db:5432/models
    depends_on:
      cache: { condition: service_healthy }
      db:    { condition: service_healthy }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5

  cache:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: models
    volumes: [pg_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d models"]
      interval: 5s
      retries: 10

volumes: { pg_data: {} }
```

### 2. Bring up the stack
```bash
docker compose up -d
docker compose ps
docker compose logs -f api
```

### 3. Exercise the API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -d '...'
```

### 4. Inspect the network
```bash
docker network ls
docker network inspect <project>_default
```
Note: services reach each other by service name (`cache`, `db`) on the user-defined bridge.

### 5. Restart-with-rebuild loop
```bash
docker compose up -d --build api
```

### 6. Volumes and data persistence
```bash
docker compose down            # data persists
docker compose down -v         # data deleted
```

## Validation
- [ ] `docker compose ps` shows all 3 services as `running (healthy)`.
- [ ] API can read from Redis and Postgres (test via an endpoint that exercises both).
- [ ] After `docker compose down && up`, Postgres still has your data.

## Cleanup
```bash
docker compose down -v
docker image prune -f
```

## Troubleshooting
- **API starts before DB is ready** — `depends_on.condition: service_healthy` requires a healthcheck on the dependency, not just `service_started`.
- **`Cannot connect to Postgres`** — Inside Compose network, use service name `db`, not `localhost`.
- **Volume not persisting** — Named volume (`pg_data`) persists; anonymous (`/var/lib/postgresql/data`) does too on the host but gets pruned by `down -v`.
