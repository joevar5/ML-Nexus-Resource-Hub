# Exercise 01: Dockerfile & Multi-Container Basics

**Duration:** 5-7 hours
**Difficulty:** Beginner → Intermediate
**Prerequisites:** Docker installed, basic Python knowledge (Part B also needs Docker Compose)

## Objective

**Part A** — write a production-quality Dockerfile from scratch for a FastAPI service that serves image classification (ResNet18) predictions. No boilerplate provided — you write every instruction and justify each one.

**Part B** — turn that single container into a four-service stack (API + PostgreSQL + Redis + Prometheus) where services discover each other by name, start in the correct order, and survive a restart with data intact.

## Why this matters

Nearly every ML model reaches production inside a container, and a single-container demo never reflects how these services actually run in practice: predictions get logged for audit, repeated inputs get cached to save compute, and everything gets scraped for metrics. A Dockerfile that "just builds" but ignores caching, image size, and non-root execution passes a demo and fails a security review or a slow CI pipeline — and Compose is the smallest environment where you can practice real service dependencies and health-gated startup before moving to Kubernetes.

---

## Part A — Single-container Dockerfile

### Requirements

1. App files: `app.py` (FastAPI with `/`, `/health`, `/predict`), `model.py` (loads `resnet18` via `torchvision`, runs inference), `requirements.txt`.
2. Final image under 2GB (ideally under 1GB) — CPU-only PyTorch, not the CUDA build.
3. Container runs as a non-root user, with a `HEALTHCHECK` wired to `/health`, and a `.dockerignore`.

### Step-by-step

**1. Application (45 min)** — `model.py`: an `ImageClassifier` class loading `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, exposing `predict(image_bytes) -> list[dict]` (top-5 class + probability). `app.py`: a FastAPI app that loads the model once at startup and exposes `/health` and `POST /predict` (accepts `UploadFile`, rejects non-image content types).

**2. Dockerfile (45 min)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["python", "app.py"]
```
Copy `requirements.txt` and install dependencies *before* the rest of the code — this is what makes rebuilds after a code change reuse the cached dependency layer instead of reinstalling torch every time.

**3. `.dockerignore` (10 min)** — exclude `__pycache__/`, `*.pyc`, `.git`, `.venv`, `*.md`, test data.

**4. Build and test (30 min)**
```bash
docker build -t ml-api:v1.0 .
docker images ml-api:v1.0
docker run -d -p 8000:8000 --name ml-api ml-api:v1.0
docker ps
curl http://localhost:8000/health
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
docker stop ml-api && docker rm ml-api
```

### Validation

- [ ] `docker build` succeeds; image size under 2GB (under 1GB is the stretch bar).
- [ ] `docker ps` shows the container as `healthy`; `/predict` returns real top-5 predictions.
- [ ] Container runs as a non-root user (`docker exec ml-api whoami` is not `root`).
- [ ] Rebuilding after a code-only change reuses the dependency layer (no `pip install` re-run in the build log).

### Common pitfalls

- **`torch not found` at runtime** — code was copied before `requirements.txt` was installed.
- **Permission denied on startup** — files owned by `root` after `COPY`; use `COPY --chown=appuser:appuser` or `chown` before `USER`.
- **Health check never passes** — no HTTP client in the slim image; use Python's `urllib` instead of installing `curl`.
- **Image balloons past 3GB** — the default `torch` wheel pulls in CUDA libraries even on a CPU-only box; pin a `+cpu` build or use the CPU wheel index.

---

## Part B — Multi-container ML stack

### Requirements

1. `docker-compose.yml` defining `api`, `postgres`, `redis`, and (stretch) `prometheus`, on a shared custom network.
2. `api` depends on `postgres` and `redis` being *healthy*, not just started.
3. Predictions are cached in Redis (identical input returns a cached result) and logged to PostgreSQL (input, prediction, confidence, timestamp).
4. Named volumes so Postgres and Redis data survive `docker compose restart`, with health checks on every service.

### Step-by-step

**1. Compose skeleton (30 min)**
```yaml
services:
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/predictions
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
  postgres:
    image: postgres:15-alpine
    environment: { POSTGRES_PASSWORD: postgres, POSTGRES_DB: predictions }
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck: { test: ["CMD-SHELL", "pg_isready -U postgres"], interval: 5s, retries: 5 }
  redis:
    image: redis:7-alpine
    volumes: ["redisdata:/data"]
    healthcheck: { test: ["CMD", "redis-cli", "ping"], interval: 5s, retries: 5 }
volumes:
  pgdata:
  redisdata:
```
`depends_on` with `condition: service_healthy` is what actually gates startup order — a plain `depends_on` list only waits for the container to start, not for the database to accept connections.

**2. API service (60 min)** — build `api/Dockerfile` as a non-root image reusing Part A's pattern. In `api/main.py`, wire up a Redis client and a `psycopg2` connection: `POST /predict` checks Redis first by a hash of the input, runs inference on a miss, writes the result to both Redis (with a TTL) and Postgres. `/health` checks both connections, not just process liveness.

**3. Database schema (20 min)** — `postgres/init.sql` creates a `predictions` table (`id serial primary key, input_data jsonb, prediction jsonb, confidence float, created_at timestamptz default now()`) with an index on `created_at`.

**4. Prometheus (stretch, 30 min)** — add `prometheus-client` to the API, expose `/metrics`, add a `prometheus` service scraping it every 15s.

**5. Test the stack (30 min)**
```bash
docker compose up -d
docker compose ps
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":[1.0,2.0,3.0]}'
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":[1.0,2.0,3.0]}'  # should hit cache
docker compose exec postgres psql -U postgres -d predictions -c "SELECT count(*) FROM predictions;"
docker compose restart postgres redis
docker compose exec postgres psql -U postgres -d predictions -c "SELECT count(*) FROM predictions;"  # data survived
docker compose down -v
```

### Validation

- [ ] `docker compose up -d` brings all services to `healthy`.
- [ ] A repeated prediction request returns a cached result.
- [ ] Every prediction appears as a row in `predictions`; `docker compose restart` does not lose data.
- [ ] Services reach each other by service name (`postgres`, `redis`), not `localhost`.

### Common pitfalls

- **Services can't connect by name** — not attached to the same custom network; declare one shared network explicitly.
- **API starts before Postgres is ready** — `depends_on` without `condition: service_healthy` only waits for the container process to start.
- **Cache silently never hits** — cache key built from unserialized/unordered JSON hashes identically-meaning input differently; normalize (sort keys) before hashing.
- **Data lost after cleanup** — `docker compose down -v` deletes named volumes; use it only when you intend to wipe state.

## Deliverables

1. Part A: `Dockerfile`, `.dockerignore`, `app.py`, `model.py`, `requirements.txt`.
2. Part B: full `ml-stack/` project (compose file, `api/`, `postgres/init.sql`, `prometheus/prometheus.yml`).
3. A `README.md` covering both parts — build/run instructions, startup order, and how to verify caching.

## Stretch goals

- Convert Part A to a multi-stage build; add a Redis-backed cache to Part A directly.
- Add Grafana on top of Prometheus for dashboards.
- Scale `api` to multiple replicas behind an nginx load balancer.
- Replace synchronous prediction logging with a queue (RabbitMQ or Kafka).
