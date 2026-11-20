# Lecture 01: Serving Frameworks

## The landscape

| Framework | Best for |
|---|---|
| FastAPI + custom | small models, control over every request |
| Triton Inference Server | multi-model, multi-framework, NVIDIA-shop (also fronts TensorRT-LLM) |
| TorchServe | PyTorch models, modest scale (in limited maintenance since 2025 — check status before adopting) |
| TensorFlow Serving | TF models |
| **vLLM** | LLMs: broadest model/hardware support, the default choice |
| **SGLang** | LLMs: prefix-heavy, agentic and structured-output workloads (RadixAttention) |
| **TensorRT-LLM** | LLMs: peak NVIDIA performance, at the cost of an engine build step |
| TGI (Hugging Face) | LLMs; check project status (HF now points users to vLLM/SGLang backends) |
| BentoML | model-ops mature; pythonic |
| Ray Serve | streaming + composition heavy (can host vLLM) |

## Decision tree

- LLM inference (autoregressive) → vLLM, SGLang or TensorRT-LLM; pick by benchmarking your traffic (question 18, Project 03)
- Classification / vision / embeddings → Triton or TorchServe
- Custom logic per request → FastAPI
- Need composition (chain of model calls) → Ray Serve / BentoML
- Already in TF ecosystem → TF Serving

## Anti-patterns

- One serving framework per model → fragmented ops
- vLLM for classification (the wrong tool)
- FastAPI alone for high-QPS LLM (no continuous batching)
- TorchServe for the highest-traffic LLM (vLLM is much faster)

---

## Deep dive: how it works (core logic)

### The evolution, one idea at a time

**1. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling.** Old systems batched whole requests: everyone waits for the longest. Orca schedules **per decode iteration**: after each step, finished requests leave and waiting ones join. It also introduces *selective batching*: linear layers batch all tokens together (flattened), but attention can't batch requests of different lengths, so it runs per request. This is **continuous batching**.

**2. [PagedAttention and vLLM](https://arxiv.org/html/2309.06180): paged KV.** The KV cache of each request is split into fixed **blocks** (say 16 tokens). A **block table** maps logical to physical blocks:
```
token position p  →  physical_block = table[p // 16],  offset = p % 16
```
Blocks are allocated only as the sequence grows, so waste is at most one partial block per request instead of a whole max-length buffer. Sharing falls out for free: two requests with the same prefix point to the **same physical blocks**, with **copy-on-write** when one diverges (parallel sampling, beam search). More requests fit → bigger batches → higher throughput. Cost: the attention kernel reads through the block table.

**3. [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf): stall-free chunked prefill.** A huge prefill monopolizes an iteration and freezes everyone's decoding. Fix: give each iteration a **token budget**. Put all running decodes in first (1 token each), then fill the rest of the budget with a **chunk** of a pending prefill. Every iteration then has similar cost, so inter-token latency stays flat, and prefill finishes over several iterations.

**4. [SGLang](https://arxiv.org/html/2312.07104): reuse and programs.** RadixAttention keeps a radix tree of cached prefixes (LRU eviction), so shared system prompts, few-shot examples and multi-turn history are not recomputed. Plus a frontend language for multi-call LLM programs and fast structured decoding.

### Structured output: [XGrammar](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf)
Constrained decoding masks logits so only grammar-valid tokens can be sampled. Naively checking the whole vocabulary (100K+ tokens) at every step is slow. XGrammar compiles the grammar into a pushdown automaton and splits tokens into **context-independent** ones (validity depends only on the current state, so the mask is *precomputed and cached*) and a small **context-dependent** set (checked at runtime against the stack). Mask computation becomes nearly free.

### The engines
[vLLM](https://github.com/vllm-project/vllm), [SGLang](https://arxiv.org/html/2312.07104) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) implement these ideas with different trade-offs (breadth and ease vs. prefix/program features vs. compiled NVIDIA peak). Choose by benchmarking your traffic (see Lesson 03 here and Project 03).

### Read the originals

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180)
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf)
- [SGLang](https://arxiv.org/html/2312.07104)
- [XGrammar](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
