# Lesson 03: Docker Networking and Volumes

A real ML deployment is rarely one container — a model server, an API gateway, a cache, and a database typically all run separately and need to find each other and persist data across restarts. This lesson covers Docker's networking modes, port mapping, and the volume types that keep models, datasets, and logs alive when a container is removed.

**Prerequisites:** Lessons 01-02 (writing and optimizing Dockerfiles).

### Contents

1. [Networking Fundamentals](#networking-fundamentals)
2. [Bridge Networks](#bridge-networks)
3. [Port Mapping](#port-mapping)
4. [Host Network Mode](#host-network-mode)
5. [Multi-Container Communication](#multi-container-communication)
6. [Volumes](#volumes)
7. [Volume Patterns for ML](#volume-patterns-for-ml)
8. [Troubleshooting](#troubleshooting)
9. [Practical Exercise](#practical-exercise)
10. [Key Takeaways](#key-takeaways)
11. [Additional Resources](#additional-resources)

---

## Networking Fundamentals

An ML stack typically separates concerns into multiple containers — a model server, an API gateway, a cache (Redis) for features or predictions, a database for logging, and sometimes a message queue — all of which need reliable communication.

| Driver | Use case | Isolation | Performance |
|---|---|---|---|
| `bridge` | Single-host container-to-container (default) | Isolated virtual network | Good |
| `host` | Container shares the host's network stack | None | Best (no NAT) |
| `none` | No networking | Complete | N/A |
| `overlay` | Multi-host (Swarm/Kubernetes) | Across hosts | Good |

## Bridge Networks

```mermaid
flowchart TB
    subgraph Host["Host Machine"]
        subgraph Bridge["Docker Bridge Network (172.17.0.0/16)"]
            C1["Container 1<br/>172.17.0.2"] --> C2["Container 2<br/>172.17.0.3"]
        end
    end
    Bridge --> HN["Host Network"]
```

The **default bridge** lets containers reach each other, but only by IP — there's no DNS, and everything on the host shares one flat network, which is both inconvenient and insecure. A **custom bridge network** fixes this:

```bash
docker network create ml-network

docker run -d --name model-server --network ml-network my-model:v1
docker run -d --name api-gateway --network ml-network my-api:v1

# from inside api-gateway:
curl http://model-server:8000/predict   # resolves by container name
```

Custom networks give you name-based DNS resolution and isolate communication to only the containers attached to that network — always prefer a custom network over the default bridge.

```bash
docker network create|ls|inspect|rm ml-network
docker network connect|disconnect ml-network my-container
```

## Port Mapping

```bash
docker run -p 8000:8000 my-ml-api               # host:container
docker run -p 8080:8000 my-ml-api               # different host port
docker run -p 127.0.0.1:8000:8000 my-ml-api     # bind to localhost only
docker run -p 8000:8000 -p 8001:8001 my-ml-api  # multiple ports
docker port my-container                        # show active mappings
```

`EXPOSE` in a Dockerfile is documentation only — `docker run` still needs an explicit `-p` to actually publish the port to the host.

## Host Network Mode

`docker run --network host my-ml-api` removes network isolation entirely — the container binds directly to host interfaces with no NAT overhead. It's useful for latency-critical serving (e.g., `nvidia/triton-inference-server` under tight SLAs) or when a container must bind a specific host interface, but it costs port isolation and portability, so it's the exception, not the default.

## Multi-Container Communication

```mermaid
flowchart TB
    U["Client"] --> GW["API Gateway :8000"]
    GW --> MS["Model Server (PyTorch)"]
    GW --> R["Redis (cache)"]
    MS --> PG["PostgreSQL (logs)"]
```

```bash
docker network create ml-stack

docker run -d --name postgres --network ml-stack \
    -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=predictions postgres:15

docker run -d --name redis --network ml-stack redis:7-alpine

docker run -d --name model-server --network ml-stack --gpus all my-model-server:v1

docker run -d --name api-gateway --network ml-stack -p 8000:8000 \
    -e MODEL_URL=http://model-server:8000 \
    -e REDIS_URL=redis://redis:6379 \
    -e DB_URL=postgresql://postgres:secret@postgres:5432/predictions \
    my-api-gateway:v1
```

Service discovery here is just environment variables plus DNS by container name — the gateway never needs to know an IP:

```python
import os, httpx, redis
from fastapi import FastAPI

app = FastAPI()
MODEL_URL = os.getenv("MODEL_URL")
redis_client = redis.from_url(os.getenv("REDIS_URL"))

@app.post("/predict")
async def predict(data: dict):
    cache_key = f"prediction:{hash(str(data))}"
    if cached := redis_client.get(cache_key):
        return {"prediction": cached, "cached": True}
    async with httpx.AsyncClient() as client:
        result = (await client.post(f"{MODEL_URL}/predict", json=data)).json()
    redis_client.setex(cache_key, 3600, result["prediction"])
    return {"prediction": result["prediction"], "cached": False}
```

## Volumes

Container filesystems are ephemeral by default — data disappears when the container is removed, and it can't be shared between containers. Volumes fix both problems and are generally faster than bind mounts because Docker manages the storage driver directly.

| Type | Managed by | Use case |
|---|---|---|
| Named volume | Docker (`/var/lib/docker/volumes/...`) | Models, databases, logs — anything that should survive container removal |
| Anonymous volume | Docker (auto-created, auto-cleaned) | Ephemeral scratch space |
| Bind mount | You (`-v /host/path:/container/path`) | Live code reload, host datasets, config files |
| tmpfs mount | Host RAM (never persisted) | Secrets in memory, fast scratch I/O |

```bash
docker volume create|ls|inspect|rm model-weights
docker run -v model-weights:/app/models my-ml-app

docker run -v $(pwd):/app my-image                       # bind mount, dev use
docker run -v /data/models:/app/models:ro my-image        # read-only bind mount
docker run --tmpfs /tmp:size=1g my-image                  # tmpfs
```

Example — a model downloaded once into a named volume outlives any single container:

```bash
docker volume create ml-models
docker run --rm -v ml-models:/models alpine sh -c \
    "wget https://example.com/model.pth -O /models/model.pth"
docker run -d -v ml-models:/app/models -p 8000:8000 my-inference-server
```

## Volume Patterns for ML

Rule of thumb: the image is what the app *is* (code, runtime, dependencies); a volume is what data it *uses* (models, datasets, logs, state) — and that data should outlive any single container.

**Pattern 1 — Model handoff** — a training container writes a new checkpoint to a named volume, and the inference container picks it up on its next read, with no image rebuild and no redeploy:

```bash
docker volume create model-weights

docker run --rm -v model-weights:/output my-training-job \
    python train.py --output /output/model-v2.pth

# inference server already has -v model-weights:/app/models mounted —
# it just needs to notice the new file (poll, inotify, or a restart)
docker restart inference-server
```

> [!IMPORTANT]
> A new file landing in the volume does **not** mean the inference server is using it. Most model-serving processes load the model into memory once at startup and never look at the file again — writing `model-v2.pth` into the volume just leaves it sitting there unused until something forces a reload. Pair this pattern with a `/reload` endpoint or a file-watcher if you want the swap to actually take effect without a full restart (see Lesson 04's hot-swap pattern); otherwise `docker restart inference-server` is the only way the new weights get picked up.

**Hot swap vs. restart** — `restart` is simplest but causes an availability gap on every model update; a hot-swap endpoint avoids that gap but requires app-level support. Default to `restart`; only build the reload path once that gap actually costs you uptime.

**Pattern 2 — Dataset access** — mount a large dataset read-only rather than copying it into the image or a volume:

```bash
docker run --rm -v /data/imagenet:/dataset:ro -v model-weights:/output \
    my-training-job python train.py --data /dataset --output /output
```

`:ro` isn't just a convenience — it's a guardrail. A training script with a bug that writes into its own dataset directory (a bad "delete corrupted samples" step, say) fails loudly and immediately instead of silently corrupting a multi-terabyte dataset that took days to assemble.

**Pattern 3 — Pipeline sharing** — a preprocessing stage and a training stage as separate containers, connected only by a shared volume, so either can be rerun or swapped independently:

```yaml
services:
  preprocess:
    build: ./preprocess
    volumes:
      - raw-data:/input:ro
      - processed-data:/output

  train:
    build: ./train
    depends_on:
      preprocess:
        condition: service_completed_successfully
    volumes:
      - processed-data:/data:ro
      - model-weights:/output

volumes:
  raw-data:
  processed-data:
  model-weights:
```

`condition: service_completed_successfully` (Compose v2.20+) makes `train` wait for `preprocess` to exit `0`, not just start — the right condition for a one-shot pipeline stage rather than a long-running service. Plain `depends_on` (no condition) only orders startup, not readiness — `train` could start while `preprocess` is still running and find an empty `processed-data`.

**Pattern 4 — Persistent logs/metrics** — mount `app-logs` and `prometheus-data` as named volumes so a container restart or redeploy doesn't wipe observability history:

```bash
docker run -d -v app-logs:/var/log/app -v prometheus-data:/prometheus \
    --name model-server my-inference-server
```

Without this, `docker compose up --force-recreate` or a crash-and-restart quietly loses every metric and log line the container ever wrote — the kind of gap you only notice during an incident post-mortem, when you need it most.

**Pattern 5 — Backup and restore** — a throwaway Alpine container is enough to tar a volume's contents to the host and back:

```bash
# Backup
docker run --rm -v model-weights:/data -v $(pwd):/backup alpine \
    tar czf /backup/models-backup.tar.gz -C /data .

# Restore into a fresh (or existing) volume
docker volume create model-weights-restored
docker run --rm -v model-weights-restored:/data -v $(pwd):/backup alpine \
    tar xzf /backup/models-backup.tar.gz -C /data
```

Restoring into a *new* volume name first, rather than overwriting the original, means a bad backup doesn't destroy your only copy — verify it, then swap references over.

**Pattern 6 — Permissions across containers** — the most common volume failure in ML stacks isn't a missing mount, it's a UID mismatch: a training container running as root writes files owned by `root:root`, and an inference container running as `appuser` (UID 1000) can't read them.

```bash
# Symptom
docker run --rm -v model-weights:/models --user 1000:1000 my-inference-server \
    cat /models/model.pth
# cat: /models/model.pth: Permission denied

# Fix: write as the same UID everywhere, or fix ownership after
docker run --rm -v model-weights:/output --user 1000:1000 my-training-job ...
# or, one-off:
docker run --rm -v model-weights:/data alpine chown -R 1000:1000 /data
```

Standardizing on one non-root UID (commonly 1000) across every image that touches a given volume avoids this entirely — worth putting in a shared base image rather than re-solving per-Dockerfile.` |

## Practical Exercise

Wire up a small ML stack on a custom network: PostgreSQL (prediction logging), Redis (feature cache), a model server, and an API gateway that talks to all three by name, with Postgres and Redis backed by named volumes.

<details>
<summary><strong>Sample Solution</strong></summary>

```bash
docker network create ml-app

docker run -d --name postgres --network ml-app \
    -e POSTGRES_PASSWORD=mlpassword -e POSTGRES_DB=predictions \
    -v postgres-data:/var/lib/postgresql/data postgres:15

docker run -d --name redis --network ml-app \
    -v redis-data:/data redis:7-alpine

docker build -t model-server ./model-server
docker run -d --name model-server --network ml-app \
    -v model-weights:/app/models model-server

docker build -t api-gateway ./api-gateway
docker run -d --name api-gateway --network ml-app -p 8000:8000 \
    -e MODEL_URL=http://model-server:8000 \
    -e REDIS_URL=redis://redis:6379 \
    -e DATABASE_URL=postgresql://postgres:mlpassword@postgres:5432/predictions \
    api-gateway

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
    -d '{"features": [1.0, 2.0, 3.0]}'
docker exec -it postgres psql -U postgres -d predictions -c "SELECT * FROM predictions LIMIT 10;"
```

`model-server/Dockerfile` builds from a PyTorch base, copies the serving code, and expects `/app/models` to be populated by the mounted volume. `api-gateway/Dockerfile` builds from `python:3.11-slim`, installs FastAPI plus Redis/Postgres client libraries, and exposes 8000 — following the layering and caching practices from lessons 02-03.

</details>

---

## Key Takeaways

1. Always create a custom bridge network for multi-container apps — you get name-based DNS resolution and traffic isolation that the default bridge doesn't provide.
2. `EXPOSE` documents a port; only `-p` at `docker run` actually publishes it to the host.
3. Named volumes are the right tool for anything that must outlive a container — models, databases, logs; bind mounts are for development and host-resident data.
4. Layers are additive but volumes are not — data written to a volume persists independently of the container's lifecycle, which is exactly what makes model handoff between containers possible.
5. Default to read-only mounts for datasets and code you don't want the container to mutate.

## Additional Resources

- [Docker Networking Overview](https://docs.docker.com/network/)
- [Docker Volumes Documentation](https://docs.docker.com/storage/volumes/)
- [Container Networking Tutorial](https://docs.docker.com/network/network-tutorial-standalone/)

---

**Next Lesson:** [04-docker-compose.md](./04-docker-compose.md) — Defining multi-container stacks in a single file
