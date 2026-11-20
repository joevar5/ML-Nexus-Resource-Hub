# Lecture 03: Memory Profiling

## PyTorch memory snapshot

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# train one iteration
torch.cuda.memory._dump_snapshot("snapshot.bin")
torch.cuda.memory._record_memory_history(enabled=None)
```

Upload `snapshot.bin` to <https://pytorch.org/memory_viz>. You'll see every
allocation + deallocation + the call stack that made it.

## When OOM strikes

For **inference** (this phase), memory is almost entirely weights + KV
cache; the training items below matter only when you fine-tune (Phase 2
Mod 07 covers training memory in depth).

The most common patterns:
- **Activation memory** grows linearly with batch size + sequence length
- **Gradient checkpointing** (training only) cuts activation memory by half or more at the cost of ~30% recompute
- **Optimizer state** (training only) for Adam = 2 FP32 values per param (8 bytes); AdamW8bit ≈ 1/4 of that
- **KV cache** in LLM inference: linear in seq len × batch × layers

`torch.cuda.memory_summary()` is a coarse but quick view.

## Reducing memory pressure

| Technique | Memory savings | Speed cost |
|---|---|---|
| Gradient checkpointing | ~50% activation | +30% time |
| Mixed precision (bf16 autocast) | ~50% activations (FP32 master weights stay) | none (often faster) |
| Pure bf16 / FP16 inference weights | 50% of FP32 weights | none |
| AdamW8bit | ~75% optimizer state | small |
| Tensor parallel | linear in TP degree | comm overhead |
| Pipeline parallel | linear in PP degree | bubbles |
| FlashAttention | attention scores: quadratic → linear in seq | none (much faster) |

## Companion

[engineer-solutions/mod-107 ex-07 (gpu-memory-profiling)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-07-gpu-memory-profiling) ships baseline + optimized + snapshot scripts.

---

## Deep dive: how it works (core logic)

### Step 0: do the bytes arithmetic
Before profiling, predict memory:
```
weights   = params × bytes_per_param            (70B × 2 = 140 GB in FP16)
KV cache  = 2 × layers × kv_heads × head_dim × bytes × tokens
activations (inference) = small, scales with tokens in a step
```
If the profiler disagrees with your prediction, you have found a leak, a duplicate copy, or fragmentation. [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) shows this accounting end to end.

### PyTorch's allocator
PyTorch caches freed blocks instead of returning them to the driver. So:
- `memory_allocated()` = bytes in live tensors; `memory_reserved()` = bytes held by the cache. A big gap = **fragmentation** (free space exists, but not in a big-enough contiguous piece, so a large allocation OOMs).
- `torch.cuda.memory._record_memory_history()` + `memory._snapshot()` gives a timeline of every allocation with its stack trace. Look for the tall plateau (peak), and for long-lived small blocks pinning large free regions.
- Fixes: free earlier, reuse buffers, `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF`, smaller micro-batches.

### The serving version of the same problem
Reserving one contiguous KV buffer per request at maximum length wastes most of it (internal fragmentation) and leaves unusable gaps (external). [PagedAttention and vLLM](https://arxiv.org/html/2309.06180) solves it by allocating KV in fixed-size **blocks** through a block table, as an OS pages memory (details in Mod 7).

### Memory *errors* vs. memory *usage*
Usage → the snapshot tools above. Errors (out-of-bounds, uninitialized reads, races) → [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/).

### Read the originals

- [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/)
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
