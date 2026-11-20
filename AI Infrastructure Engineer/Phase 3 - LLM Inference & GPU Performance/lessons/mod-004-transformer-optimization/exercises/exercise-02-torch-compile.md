# Ex 02: torch.compile on a Decode Loop

**Goal:** separate the two things `torch.compile` gives you — fusion and CUDA
graphs — and measure each on prefill and decode.
**Time:** ~2 h. **Needs:** NVIDIA GPU, PyTorch ≥ 2.3. (If you did Mod 3 Ex 03,
reuse that setup.)

## Steps

Use a ≤1B causal LM in BF16 with a **static** KV cache. Time three variants:
eager, `torch.compile(mode="default")` (fusion only), and
`torch.compile(mode="reduce-overhead")` (fusion + CUDA graphs). Warm up
each until compilation has finished (watch for recompiles with
`TORCH_LOGS=recompiles`).

| shape | what it stresses |
|---|---|
| prefill, 1 × 2,048 tokens | big GEMMs, compute |
| decode, batch 1 | launch overhead |
| decode, batch 32 | bandwidth |

## Report

Median ms for each of the 9 cells, and the number of kernels per step from a
quick `torch.profiler` run for eager vs compiled.

## Pass criteria

- A table of 9 numbers, compile time noted separately.
- You can say which mode wins for each shape and **why** (expected: CUDA
  graphs help batch-1 decode most; prefill gains little).
- Note any graph breaks or recompiles and how you removed them.
