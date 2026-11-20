# Phase 3: LLM Inference & GPU Performance

Phase 1 got a model running. Phase 2 got LLM infrastructure running. Phase 3 is about making it **fast, cheap and explainable**: why a request is slow, where the GPU time goes, and what a design costs.

It is built around **25 questions**. If you can answer all of them with numbers (not hand-waving), you know LLM inference.

---

## How this phase is built (three layers)

| Layer | What it is | Where |
| :--- | :--- | :--- |
| **1. Course** | An 8-module performance-engineering course: lecture notes, exercises, quiz, resources. Module 1 is complete; Modules 2–8 are shorter outlines | `modules/` |
| **2. Advanced layer** | Each lecture note ends with a **Deep dive** section, and each module has a **`source-digest.md`**: what the papers, docs and worklogs actually say, as steps and numbers, with anything unread marked | `modules/mod-00X-*/source-digest.md` |
| **3. Reading list** | The full curated link list from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT), also split per module and project | `resources/reading-lists/` |

**How to use it:** read a module's lecture notes → read its source digest → do the exercises → answer the matching questions below out loud → do the project.

---

## The modules

| # | Module | Digest | What it covers |
| :-: | :--- | :--- | :--- |
| 001 | [GPU fundamentals](modules/mod-001-gpu-fundamentals/) | [digest](modules/mod-001-gpu-fundamentals/source-digest.md) | SMs, memory hierarchy, warps, occupancy, roofline, generations |
| 002 | [CUDA programming](modules/mod-002-cuda-programming/) | [digest](modules/mod-002-cuda-programming/source-digest.md) | kernels, coalescing, shared memory, matmul ladder, PyTorch extensions |
| 003 | [Performance profiling](modules/mod-003-performance-profiling/) | [digest](modules/mod-003-performance-profiling/source-digest.md) | Nsight Systems/Compute, bottleneck classification, memory profiling |
| 004 | [Transformer optimization](modules/mod-004-transformer-optimization/) | [digest](modules/mod-004-transformer-optimization/source-digest.md) | FlashAttention, GQA/MLA, kernel fusion, speculative decoding |
| 005 | [Model compression](modules/mod-005-model-compression/) | [digest](modules/mod-005-model-compression/source-digest.md) | GPTQ, AWQ, SmoothQuant, QuaRot, KV quantization, NVFP4, sparse attention |
| 006 | [Distributed inference](modules/mod-006-distributed-inference/) | [digest](modules/mod-006-distributed-inference/source-digest.md) | TP/PP/EP, MoE, prefill/decode disaggregation, autoscaling, routing |
| 007 | [Production deployment](modules/mod-007-production-deployment/) | [digest](modules/mod-007-production-deployment/source-digest.md) | Orca, vLLM, Sarathi-Serve, SGLang, goodput, rollouts, benchmarking |
| 008 | [Advanced topics](modules/mod-008-advanced-topics/) | [digest](modules/mod-008-advanced-topics/source-digest.md) | NCCL device API, interconnect, CUDA graphs, FP8 |

Suggested order: 001 → 008. Modules 001 and 003 are the foundation.

---

## The 25 questions

The questions this phase is built around, with where each is answered and whether it is covered, partial or still a gap: **[25-questions.md](25-questions.md)**.

---

## Projects

Three project briefs from the upstream course (each with README, requirements, architecture, step-by-step and rubric). See [projects/](projects/).

| Project | Questions it supports | Reading |
| :--- | :--- | :--- |
| [01 · Model optimization](projects/project-201-llm-optimization-pipeline/) | 10, 11, 24 | Quantization, speculative decoding, KV cache links are at the end of its README |
| [02 · GPU optimization](projects/project-202-custom-gpu-kernels/) | 13, 14, 15 | Triton, profiling, matmul, attention links at the end of its README |
| [03 · Distributed inference](projects/project-203-llm-serving-engine/) | 6, 8, 18, 22, 23 | Engines, routing, MoE and disaggregation links at the end of its README |

## Reference material

| Resource | Where |
| :--- | :--- |
| Full Wafer GPU performance reading list (~130 links) | [resources/reading-lists/wafer-gpu-perf-reading-list.md](resources/reading-lists/wafer-gpu-perf-reading-list.md) |
| Upstream course the modules come from | [ai-infra-performance-learning](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) (MIT) |
| Phase 2 prerequisite: LLM infrastructure | [Phase 2 · Mod 10](../Phase%202%20-%20Advanced%20AI%20Infrastructure/lessons/mod-10-llm-infrastructure/) |

## Before you start

Comfortable with Phase 1, and with Phase 2 Mod 07 (GPU computing) and Mod 10 (LLM infrastructure). A GPU is not needed to read the notes and digests; the exercises in Modules 1–3 and the projects benefit from one (an A10G, L4 or A100 is enough).

## Honest limits

- **Source digests** report what was read. Items marked "not read" were not opened (some were password-protected, timed out, or were not fetched). Manuals such as Nsight were read only in part.
- The **Deep dive** sections in the lecture notes were written from general knowledge and are only partly checked against the digests. Where they disagree, trust the digest.
- Hardware figures and prices change by vendor and date. Verify before spending money.
- Modules 2–8 of the upstream course are shorter than Module 1, and the exercises there are mostly outlines.
