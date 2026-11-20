# Module 02: CUDA Programming

**Duration**: ~30 hours
**Prerequisites**: Module 01 (GPU fundamentals); C++ basics; Linux + nvcc setup

## Overview

Mod-001 explained GPU architecture conceptually. Module 02 takes you from
"understands warps + occupancy" to "writes a custom CUDA kernel + integrates
it into PyTorch." We cover the C++ side (kernels, memory, launch config)
+ the PyTorch C++ extension build chain.

## Objectives

1. Write a memory-bound CUDA kernel that reaches ≥ 80% of device-to-device `cudaMemcpy` bandwidth (the practical ceiling).
2. Walk the matmul ladder (naive → coalesced → shared-memory tiles → register tiles) and reach ≥ 50% of cuBLAS FP32 SGEMM.
3. Pack a custom kernel as a PyTorch C++ extension; benchmark vs `torch.add`.
4. Read PTX + SASS; identify register spills + bank conflicts.
5. Tune launch config (block size, grid size) for compute-bound + memory-bound kernels.

**Hardware:** from this module on you need an NVIDIA GPU (any of T4,
A10G, L4, A100 is fine; a free Colab/Kaggle T4 works for every exercise).
Phase 2 [Mod 07 · GPU computing](../../../Phase%202%20-%20Advanced%20AI%20Infrastructure/lessons/mod-07-gpu-computing/)
already covered CUDA basics and PyTorch on GPU; this module assumes it and goes
deeper into kernel writing.

```
mod-002-cuda-programming/
├── README.md
├── lecture-notes/  (3 files)
├── exercises/      (5)
├── quiz.md, quiz-answers.md
└── resources.md
```

Companion: [engineer-solutions/mod-107 ex-02 (cuda-kernel)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel).

---

## Advanced layer: source digest

[source-digest.md](source-digest.md) summarizes what the linked sources (papers, docs, worklogs) actually say, as steps and numbers, with anything not read marked. Read it after the lecture notes.
