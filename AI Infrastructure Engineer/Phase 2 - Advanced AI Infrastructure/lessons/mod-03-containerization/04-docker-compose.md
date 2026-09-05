# Lesson 04: Docker Compose for ML Applications

Docker Compose defines multi-container applications in a single YAML file and starts them with one command. Instead of `docker network create` and four separate `docker run` invocations, a ML stack of an API, model server, database, and cache comes up with `docker compose up`. This lesson covers writing compose files for ML stacks, managing configuration across environments, and scaling services.

**Prerequisites:** comfort with Docker images, containers, volumes, and networking (Lessons 01-03).

### Contents

1. [Why Compose for ML](#why-compose-for-ml)
2. [Compose File Basics](#compose-file-basics)
3. [A Complete ML Stack](#a-complete-ml-stack)
4. [Configuration Patterns](#configuration-patterns)
5. [Commands Reference](#commands-reference)
6. [Dev, Test, and Multi-File Setups](#dev-test-and-multi-file-setups)
7. [Scaling Services](#scaling-services)
8. [Practical Exercise](#practical-exercise)
9. [Troubleshooting](#troubleshooting)
10. [Key Takeaways](#key-takeaways)
11. [Additional Resources](#additional-resources)

---

## Why Compose for ML

| | Without Compose | With Compose |
|---|---|---|
| Startup | Manual network + 4 `docker run` commands | `docker compose up` |
| Reproducibility | Depends on who typed the commands | Same stack every time, checked into git |
| Team onboarding | Tribal knowledge | `git clone && docker compose up` |

Good fit: local development, integration testing against real databases/caches, demo environments, small single-host production deployments (1-3 servers). Not a fit for large-scale or multi-host production — that's Kubernetes territory (Module 04).

Verify the plugin is present (Compose v2 ships with Docker Desktop and as the `docker-compose-plugin` package on Linux):

```bash
docker compose version
# Docker Compose version v2.20.0
```

---

## Compose File Basics

```yaml
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

```bash
docker compose up          # start, attached
docker compose up -d       # start, detached
docker compose down        # stop and remove containers/networks
docker compose down -v     # also remove volumes
```

The top-level `version:` key is obsolete in Compose v2 and safe to omit — the file above doesn't need it.

---

## A Complete ML Stack

```mermaid
flowchart LR
    Client --> API["api (FastAPI)"]
    API --> Redis["redis (cache)"]
    API --> MS["model-server (PyTorch)"]
    API --> PG["postgres (predictions log)"]

    subgraph Observability
        direction LR
        Prom["prometheus"] --> Graf["grafana"]
    end

    API -.->|"/metrics"| Prom
    MS -.->|"/metrics"| Prom
```

```yaml
services:
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
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      model-server:
        condition: service_healthy
    networks: [ml-network]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  model-server:
    build: ./model-server
    environment:
      - MODEL_PATH=/models/model.pth
      - BATCH_SIZE=32
    volumes:
      - model-weights:/models
    networks: [ml-network]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    # GPU: uncomment when running on a GPU host (see Lesson 07)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=mlpassword
      - POSTGRES_DB=predictions
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks: [ml-network]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks: [ml-network]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports: ["9090:9090"]
    networks: [ml-network]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on: [prometheus]
    networks: [ml-network]

volumes:
  model-weights:
  postgres-data:
  redis-data:
  prometheus-data:

networks:
  ml-network:
    driver: bridge
```

`depends_on` with `condition: service_healthy` waits for the healthcheck to pass, not just for the container to start — the API won't try to hit Postgres before it's actually accepting connections.

> [!NOTE]
> Compose = define the whole application stack in YAML, then let Docker create and connect the containers for you.

---

## Configuration Patterns

**Build args and multi-stage targets**

```yaml
services:
  api:
    build:
      context: ./api
      args:
        PYTHON_VERSION: "3.11"
      target: production   # picks a stage from a multi-stage Dockerfile
```

**Environment variables** — three layers, most explicit wins: inline `environment:` values, an `env_file:` (can list several), and `${VAR}` substitution pulled from a `.env` file sitting next to the compose file.

```yaml
services:
  api:
    env_file: [.env, .env.production]
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
```

Don't put real secrets in inline `environment:` values committed to git — use `env_file` with a gitignored file, or a secrets manager in production.

**Volumes**

```yaml
services:
  model-server:
    volumes:
      - model-data:/app/models      # named volume, persists
      - ./src:/app/src:ro            # bind mount, read-only, for local dev
    tmpfs:
      - /tmp                         # in-memory, gone on restart
```

**Resource limits** (see Lesson 08 for the full rationale):

```yaml
services:
  model-server:
    deploy:
      resources:
        limits: { cpus: '2.0', memory: 4G }
        reservations: { cpus: '1.0', memory: 2G }
```

---

## Commands Reference

```bash
docker compose logs -f api          # follow logs for one service
docker compose ps                   # running services
docker compose exec api bash        # shell into a running service
docker compose build                # rebuild images
docker compose up --build           # rebuild then start
docker compose up --scale model-server=3   # runtime scaling
docker compose config               # validate + print resolved config
docker compose restart api
docker compose pull                 # pull images without starting
```

---

## Dev, Test, and Multi-File Setups

Keep environment-specific overrides in separate files and layer them at run time — later files override earlier ones:

```bash
docker compose -f compose.yml -f compose.dev.yml up
```

**compose.dev.yml** — mounts source for hot reload and enables a debugger:

```yaml
services:
  api:
    build:
      context: ./api
      target: development
    volumes:
      - ./api/src:/app/src
    environment:
      - RELOAD=true
    command: uvicorn main:app --reload --host 0.0.0.0
    ports:
      - "8000:8000"
      - "5678:5678"
```

**compose.test.yml** — runs the test suite as the container's command, against a throwaway database:

```yaml
services:
  api:
    build:
      context: ./api
      target: test
    environment:
      - DATABASE_URL=postgresql://postgres:test@test-db:5432/test
    command: pytest tests/ -v
    depends_on: [test-db]

  test-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=test
      - POSTGRES_DB=test
```

```bash
docker compose -f compose.test.yml up --abort-on-container-exit
docker compose -f compose.test.yml down -v
```

---

## Scaling Services

```yaml
services:
  model-server:
    build: ./model-server
    deploy:
      replicas: 3
```

```bash
docker compose up --scale model-server=5
```

Scaling like this only works for stateless services — anything with `ports:` mapped to a fixed host port will collide across replicas, so front them with a load balancer instead:

```nginx
upstream model_servers {
    server model-server:8000;
}
server {
    listen 80;
    location / { proxy_pass http://model_servers; }
}
```

```yaml
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    ports: ["80:80"]
    depends_on: [model-server]

  model-server:
    build: ./model-server
    deploy:
      replicas: 3
```

---

## Practical Exercise

Build the ML stack above with real (minimal) services instead of placeholders:

**Requirements:** a FastAPI `api` service that checks Redis for a cached prediction before calling `model-server`, then logs the result to Postgres; a `model-server` that loads a PyTorch model and serves `/predict`; Prometheus + Grafana wired up to scrape both services' `/metrics`.

Sketch your `docker-compose.yml` and the two Dockerfiles before expanding the solution.

<details>
<summary><strong>Sample Solution</strong></summary>

Project layout:

```
ml-stack/
├── docker-compose.yml
├── .env
├── api/{Dockerfile, requirements.txt, main.py}
├── model-server/{Dockerfile, requirements.txt, serve.py}
├── postgres/init.sql
└── prometheus/prometheus.yml
```

`api/main.py`:

```python
from fastapi import FastAPI
import httpx, redis, os, json

app = FastAPI()
MODEL_URL = os.getenv("MODEL_SERVER_URL")
r = redis.from_url(os.getenv("REDIS_URL"))

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(data: dict):
    cache_key = json.dumps(data, sort_keys=True)
    if cached := r.get(cache_key):
        return json.loads(cached)
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{MODEL_URL}/predict", json=data)
    result = resp.json()
    r.setex(cache_key, 300, json.dumps(result))
    return result
```

`model-server/serve.py`:

```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("/models/model.pth", map_location="cpu")
model.eval()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    x = torch.tensor(data["features"], dtype=torch.float32)
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    return {"prediction": out.argmax(dim=1).item()}
```

Bring it up and check the pieces:

```bash
docker compose up -d
docker compose ps
curl http://localhost:8000/health
docker compose logs -f api
open http://localhost:9090   # Prometheus
open http://localhost:3000   # Grafana
docker compose down
```

</details>

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Service won't start | `docker compose logs service-name`, `docker compose ps` |
| Can't reach another service | `docker compose exec api ping model-server`; inspect the network with `docker network inspect <project>_default` |
| Permission errors on a volume | `docker volume inspect <project>_model-data`; fix ownership with `docker compose exec api chown -R appuser:appuser /data` |
| Port already in use | Change the host side of the mapping: `"8001:8000"` |

---

## Key Takeaways

1. Compose replaces manual `docker network create` + multiple `docker run` commands with one declarative file.
2. `depends_on` with `condition: service_healthy` waits for real readiness, not just container start.
3. Split dev/test/prod config into overlay files and layer them with `-f`.
4. Named volumes persist data; bind mounts are for live-editing source in development.
5. Compose is for single-host setups — multi-host or large-scale production belongs on Kubernetes.

---

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Compose File Reference](https://docs.docker.com/compose/compose-file/)
- [Awesome Compose Examples](https://github.com/docker/awesome-compose)

---

**Next Lesson:** [05-container-registries.md](./05-container-registries.md) — pushing and managing images across Docker Hub, ECR, GCR, and ACR
