# Ex 03: 2:4 Sparsity — Kernel Speedup vs End-to-End Speedup

**Goal:** measure what NVIDIA's "2× with sparsity" means in practice.
**Time:** ~2 h. **Needs:** Ampere or newer GPU (A100, A10G, L4, H100), FP16.

## Steps

1. **Kernel level:** for a `[M, K] @ [K, N]` linear with K = N = 8192 and
   M ∈ {16, 256, 4096} tokens, prune W to 2:4 by magnitude (keep the 2 largest
   of every 4), convert with `to_sparse_semi_structured`, and time sparse vs
   dense `F.linear` (median, CUDA events). Reference:
   [PyTorch semi-structured sparsity tutorial](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html).
2. **Model level:** apply the same magnitude 2:4 pruning to every MLP linear of
   a small LM (e.g. `Qwen2.5-0.5B`), with no fine-tuning, and measure WikiText-2
   perplexity before and after.
3. Optional: recover quality with a short fine-tune while keeping the mask fixed.

## Report

| M (tokens) | dense ms | 2:4 ms | speedup |
|---|---|---|---|

plus perplexity dense vs pruned (vs pruned + fine-tuned if done).

## Pass criteria

- The table shows where the speedup is largest (big M, compute-bound) and
  where it nearly vanishes (small M, memory-bound decode) — explain with the roofline.
- Perplexity numbers show why 2:4 pruning without retraining is rarely used for
  LLMs, unlike quantization.
