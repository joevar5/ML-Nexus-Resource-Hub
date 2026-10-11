# Module 03 Exercises: Containerization with Docker

## Overview

One hands-on exercise, tracking the GPU Docker lesson.

## Exercise List

### Exercise 01: GPU-Accelerated ML Container
**Duration:** 2-3 hours · **Difficulty:** Intermediate
**Prerequisites:** NVIDIA GPU, NVIDIA Container Toolkit
**File:** [`exercise-01-gpu-container.md`](./exercise-01-gpu-container.md)

Build a CUDA-enabled image that trains a small CNN on MNIST, expose the host GPU correctly, and measure the real GPU-vs-CPU speedup.

## Prerequisites

Before starting these exercises:

- [ ] Completed the relevant Module 03 lessons
- [ ] Docker Engine or Docker Desktop installed, with Compose and BuildKit
- [ ] An NVIDIA GPU + Container Toolkit for Exercise 01

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
3. Proceed to Module 04: Kubernetes

---

**Questions?** Open an issue in the GitHub repository or post in Discussions.
