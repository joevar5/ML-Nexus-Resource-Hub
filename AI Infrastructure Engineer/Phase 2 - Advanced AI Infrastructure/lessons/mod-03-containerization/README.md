# Module 03: Containerization with Docker

**Duration:** 35 hours · **Difficulty:** Intermediate · **Prerequisites:** Module 02

## Overview

"Works on my machine" is a special kind of nightmare in ML — mismatched CUDA versions, fragile Python environments, a model that behaves differently on the training box than the serving box. Docker is how you make that problem go away: package the code, the weights, and every dependency into one image that runs the same everywhere, from your laptop to a GPU cluster.

This module takes you from `docker run hello-world` to shipping production-grade, GPU-enabled containers for real ML workloads.

## What you'll walk away with

- Dockerfiles you'd actually trust in production — multi-stage, cached, and lean
- The instinct for why an image is 4GB when it should be 400MB, and how to fix it
- GPU access working correctly inside a container (this trips up almost everyone the first time)
- A multi-container ML stack (model + API + cache + DB) running via Docker Compose
- Comfort pushing images to Docker Hub, ECR, GCR, or ACR as part of a CI/CD flow

## Lessons

| # | Lesson | Hours |
|---|--------|-------|
| 01 | [Docker Introduction](./01-docker-introduction.md) — containers vs. VMs, the Docker daemon, your first container | 4 |
| 02 | [Dockerfiles for ML Apps](./02-dockerfile-ml-apps.md) — base images, installing PyTorch/TensorFlow, structuring an image | 5 |
| 03 | [Image Optimization](./03-image-optimization.md) — multi-stage builds, layer caching, cutting image size 50–80% | 5 |
| 04 | [Networking & Volumes](./04-docker-networking-volumes.md) — port mapping, container-to-container traffic, persistent storage | 5 |
| 05 | [Docker Compose](./05-docker-compose.md) — orchestrating multi-service ML stacks | 5 |
| 06 | [Container Registries](./06-container-registries.md) — Docker Hub, ECR, GCR, ACR, tagging strategy | 4 |
| 07 | [GPU Support in Docker](./07-gpu-docker.md) — NVIDIA Container Toolkit, CUDA base images, multi-GPU containers | 5 |
| 08 | [Production Best Practices](./08-production-best-practices.md) — security, health checks, graceful shutdown | 2 |

Each lesson has a hands-on exercise attached — don't skip them, this is a module you learn by doing, not reading.

## Hands-on activities

1. **Containerize an image classifier** — ship a ResNet model behind FastAPI in an image under 500MB
2. **Multi-stage build optimization** — take a 2GB image down to under 500MB
3. **Multi-container ML app** — model server + API gateway + cache + database, wired up with Compose
4. **GPU deployment** — get a Stable Diffusion container actually using the GPU
5. **CI/CD to a registry** — a GitHub Actions workflow that builds and pushes on every commit

Deeper, guided versions of these live in [`exercises/`](./exercises/) and [`labs/`](./labs/).

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 20 questions, 70% to pass
- **Practical** — build a production-ready container: multi-stage Dockerfile, image under 500MB (CPU) or 2GB (GPU), non-root user, health check endpoint, and a README explaining your choices

## Before you start

```bash
docker --version
docker run hello-world
docker compose version

# optional, if you have an NVIDIA GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

If any of those fail, sort out Docker (and the NVIDIA Container Toolkit, if relevant) before diving into Lesson 01.

## The mistakes everyone makes here

- **Images that balloon past 5GB** — fix with multi-stage builds and slim base images; you're aiming for <500MB (CPU) or <2GB (GPU)
- **Every code change rebuilds the whole image** — order your Dockerfile so dependency installs are cached separately from app code; this alone turns 20-minute builds into 30-second ones
- **`nvidia-smi` fails inside the container** — almost always a missing or misconfigured NVIDIA Container Toolkit
- **A 10GB model won't fit in the image** — don't bake it in; mount it as a volume or pull it at startup from a model registry

## What's next

Pass the quiz, finish the practical, then move on to **Module 04: Kubernetes** — where these containers learn to scale.

More reading in [`resources.md`](./resources.md).
