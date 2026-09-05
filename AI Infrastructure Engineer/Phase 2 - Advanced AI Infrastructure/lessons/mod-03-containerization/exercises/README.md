# Module 03 Exercises: Containerization with Docker

## Overview

Six hands-on exercises, each covering multiple related sub-topics as lettered parts (Part A, B, C) rather than one narrow skill each — this keeps closely related work together instead of splitting it across many thin files. They roughly track the lessons: core Dockerfile/Compose/GPU skills first, then production concerns — build optimization, registry management, security/supply-chain, and operational debugging.

## Exercise List

### Exercise 01: Dockerfile & Multi-Container Basics
**Duration:** 5-7 hours · **Difficulty:** Beginner → Intermediate
**File:** [`exercise-01-dockerfile.md`](./exercise-01-dockerfile.md)

Part A: write a production-quality Dockerfile from scratch for a FastAPI/ResNet18 service. Part B: turn it into a four-service Compose stack (API + PostgreSQL + Redis + Prometheus) with health-gated startup and persistent data.

### Exercise 02: GPU-Accelerated ML Container
**Duration:** 2-3 hours · **Difficulty:** Intermediate
**Prerequisites:** NVIDIA GPU, NVIDIA Container Toolkit
**File:** [`exercise-02-gpu-container.md`](./exercise-02-gpu-container.md)

Build a CUDA-enabled image that trains a small CNN on MNIST, expose the host GPU correctly, and measure the real GPU-vs-CPU speedup.

### Exercise 03: Build & Cache Optimization
**Duration:** 13-15 hours (splits into 3 sessions) · **Difficulty:** Intermediate → Advanced
**Prerequisites:** Exercise 01
**File:** [`exercise-03-build-optimization.md`](./exercise-03-build-optimization.md)

Part A: build `image-optimizer`, a CLI that rewrites a Dockerfile into a multi-stage build, cutting size 50%+. Part B: apply BuildKit cache mounts/secrets/remote cache to cut CI build time 70%+. Part C: produce a multi-arch (amd64+arm64) manifest-list image and benchmark emulated vs. native builds.

### Exercise 04: Container Registry Manager
**Duration:** 9-11 hours · **Difficulty:** Advanced
**File:** [`exercise-04-registry-manager.md`](./exercise-04-registry-manager.md)

Build `registry-manager`, a CLI that syncs images across ECR/GCR/ACR, runs a dev→staging→prod promotion pipeline with approval gates, enforces retention policies, and logs an audit trail.

### Exercise 05: Container Security & Supply Chain
**Duration:** 14-16 hours (splits into 3 sessions) · **Difficulty:** Intermediate
**Prerequisites:** `cosign` installed for Part B
**File:** [`exercise-05-container-security.md`](./exercise-05-container-security.md)

Part A: build `containersec`, a CLI scanning images with Trivy/Grype against a YAML policy. Part B: extend it into supply-chain integrity — sign with cosign, attach SBOM + SLSA provenance, gate Kubernetes deploys with Kyverno. Part C: extend it again into a full CVE remediation workflow — discover, triage, auto-patch, document accepted risk, report.

### Exercise 06: Runtime Debugging & ML Container Patterns
**Duration:** 5-5.5 hours · **Difficulty:** Intermediate
**Prerequisites:** Exercise 01
**File:** [`exercise-06-runtime-patterns.md`](./exercise-06-runtime-patterns.md)

Part A: diagnose three deliberately-broken running containers using only container-native tools. Part B: implement four ML-specific patterns — startup warmup, init-container preloading, sidecar dynamic batching, hot-swappable models — each with a measurable metric.

## Prerequisites

Before starting these exercises:

- [ ] Completed the relevant Module 03 lessons
- [ ] Docker Engine or Docker Desktop installed, with Compose and BuildKit
- [ ] Python 3.9+ for the CLI-building exercises (03, 04, 05)
- [ ] An NVIDIA GPU + Container Toolkit for Exercise 02
- [ ] Cloud CLI access (AWS/GCP/Azure) for Exercise 04
- [ ] `cosign` installed for Exercise 05, Part B

## Exercise Guidelines

### Best Practices

1. **Document everything** — keep notes on commands, errors, and solutions
2. **Version control** — commit progress frequently
3. **Clean up** — remove containers, images, and cloud resources after each exercise
4. **Security** — never commit credentials, API keys, or `.env` files
5. **Verify, don't assume** — every exercise has a Validation checklist; run it

### Getting Help

If you get stuck:

1. Review the relevant lesson material
2. Check the exercise's "Common pitfalls" section
3. Check Docker's official documentation
4. Ask questions in GitHub Discussions

## Deeper, guided walkthroughs

Six shorter, more structured labs covering the same core skills (Dockerfile optimization, multi-stage builds, Compose stacks, GPU containers, private registries, security scanning) live in [`../labs/`](../labs/) — use them if you want a more guided path before tackling these exercises solo.

## Next Steps

After completing these exercises:
1. Compare your solutions against each exercise's Validation checklist
2. Complete the Module 03 quiz ([`../quizzes/module-quiz.md`](../quizzes/module-quiz.md))
3. Proceed to Module 04: Kubernetes

---

**Questions?** Open an issue in the GitHub repository or post in Discussions.
