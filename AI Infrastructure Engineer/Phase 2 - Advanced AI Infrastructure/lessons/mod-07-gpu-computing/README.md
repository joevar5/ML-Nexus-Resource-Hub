# Module 07: GPU Computing & Distributed Training

**Duration:** ~30–35 hours · **Difficulty:** Intermediate to Advanced · **Prerequisites:** Module 03 (Docker), PyTorch/TensorFlow basics

## Overview

At some point a single GPU isn't enough — the model doesn't fit, or training takes too long to iterate on. This module covers why GPUs are fast for ML in the first place, how to actually use that speed in PyTorch, and how to scale from one GPU to many, on one node or across a cluster, without your job silently running slower than it should.

## What you'll walk away with

- A real understanding of GPU architecture — enough to know why a workload is slow, not just that it is
- Comfort writing basic CUDA and reasoning about GPU memory
- Multi-GPU training working correctly with DistributedDataParallel
- The ability to fix an OOM error instead of just reducing batch size and hoping
- A working vocabulary for data/model/pipeline parallelism, and when each one applies

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Introduction to GPU Computing](./01-introduction-gpu-computing.md) — GPU vs. CPU, CUDA/tensor cores, memory hierarchy |
| 02 | [CUDA Programming Fundamentals](./02-cuda-programming-fundamentals.md) — kernels, thread organization, memory management |
| 03 | [PyTorch GPU Acceleration](./03-pytorch-gpu-acceleration.md) — moving tensors to GPU, mixed precision, profiling |
| 04 | [Distributed Training Fundamentals](./04-distributed-training-fundamentals.md) — data vs. model parallelism, AllReduce |
| 05 | [Multi-GPU Training Strategies](./05-multi-gpu-training-strategies.md) — DataParallel vs. DDP, data loading at scale |
| 06 | [Model & Pipeline Parallelism](./06-model-pipeline-parallelism.md) — when a model doesn't fit on one GPU |
| 07 | [GPU Memory Management](./07-gpu-memory-management.md) — profiling, gradient checkpointing, fixing OOM errors |
| 08 | [Advanced GPU Optimization](./08-advanced-gpu-optimization.md) — nsys/nvprof, kernel fusion, batch size tuning |

## Hands-on

Five labs in [`labs/`](./labs/): write a CUDA kernel from scratch, move a model from CPU to GPU, implement DDP training, profile and fix an OOM error, then run training across a multi-node cluster.

## Before you start

An NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better) covers most of this module; the distributed-training labs need 2+ GPUs or a cloud GPU instance.

```bash
nvidia-smi   # confirm the GPU is visible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 questions
- **Practical** — take a deliberately slow GPU training pipeline and optimize it
- **Capstone** — implement distributed training for a large model

## What's next

Once training scales cleanly across GPUs, move to **Module 08: Monitoring & Observability** to keep an eye on all that compute in production.

More reading in [`resources.md`](./resources.md).
