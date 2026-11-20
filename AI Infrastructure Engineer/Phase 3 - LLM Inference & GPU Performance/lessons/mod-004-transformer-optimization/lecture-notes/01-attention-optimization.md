# Lecture 01: Attention Optimization

## Why attention is the bottleneck

Standard attention: `softmax(QK^T / sqrt(d_k)) V` — compute and memory are
both O(N²) in sequence length. At short context the MLP matmuls dominate; as
N grows past a few thousand tokens, attention's share of FLOPs and (unfused)
memory traffic climbs until it dominates.

## FlashAttention

Re-formulates attention to:
- Tile-based: compute attention block-by-block in SRAM
- Memory: O(N) instead of O(N²); intermediate matrices never materialized
- Speed: 2-4× faster on A100/H100 for long sequences

Available in PyTorch as `torch.nn.functional.scaled_dot_product_attention`.

```python
# Drop-in replacement
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Use this everywhere; don't hand-roll attention.

## KV cache

During autoregressive generation, K and V from prior tokens are reused. Cache
them; only compute new tokens' Q.

Memory cost: `batch × seq_len × layers × kv_heads × head_dim × 2 (K+V) × precision_bytes`.

Llama-2-7B style (no GQA: kv_heads = 32), seq_len=4096, layers=32, head_dim=128, bf16:
4096 × 32 × 32 × 128 × 2 × 2 = 2.1 GB per request.
Llama-3-8B uses GQA with 8 KV heads, so the same context costs 0.54 GB, 4× less.

## Paged attention (vLLM)

KV cache is large + variable per request. Naive allocation fragments memory.
PagedAttention uses fixed-size pages (like virtual memory) — each request's
KV cache is a list of page indices.

Result: 2-4× more concurrent requests on the same GPU.

---

## Deep dive: how it works (core logic)

### The problem
`O = softmax(QKᵀ/√d)·V`. Naively: compute S = QKᵀ (N×N), write to HBM; read S, softmax, write P (N×N); read P, multiply by V. For N=8K, S is 64M entries *per head*. The FLOPs are fine; the **HBM round trips of the N×N matrix** are the cost, so attention is memory-bound.

### Online softmax: the enabling trick ([Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867))
Normal softmax needs the max of the whole row first (for stability), then the sum. Online softmax keeps a running max `m` and running sum `l` and fixes them up as new values arrive:
```
m_new = max(m, x)
l     = l * exp(m - m_new) + exp(x - m_new)
m     = m_new
```
Old terms are rescaled by exp(m − m_new) whenever the max grows. One pass, no full row needed.

### FlashAttention: tile it ([FlashAttention](https://arxiv.org/abs/2205.14135))
(The loop order below, Q blocks outside, is FlashAttention-2's; v1 looped over K/V blocks outside.)
```
for each block Q_i (kept in SRAM):
    m = -inf, l = 0, O = 0
    for each block K_j, V_j (streamed from HBM):
        S = Q_i K_jᵀ                         # small tile, in SRAM
        m_new = max(m, rowmax(S))
        P = exp(S - m_new)
        l = l*exp(m - m_new) + rowsum(P)
        O = O*exp(m - m_new) + P V_j         # rescale old output, add new
        m = m_new
    O = O / l ;  store O (and m,l) to HBM
```
S and P never touch HBM. HBM traffic goes from O(N² ) to O(N²d²/M) (M = SRAM size), a large cut in practice, and extra memory drops from O(N²) to O(N). The result is **exact**. (Backward pass recomputes S from stored m, l instead of saving it: trading cheap FLOPs for scarce bandwidth.)

### What the later versions change
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691):** swaps the loops (outer loop over Q blocks so blocks parallelize across the sequence), cuts non-matmul FLOPs (which run on slow units), and splits work across warps better to avoid shared-memory traffic. Result: much better GPU utilization.
- **[FlashAttention-3](https://arxiv.org/abs/2407.08608):** uses Hopper features: TMA and warp specialization (some warps move data, others compute), and *overlaps softmax with the matmuls* (pingpong between two warpgroups). Adds FP8 support.
- **[FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf):** re-schedules the same computation for Blackwell's tensor memory and asynchronous MMA.
- **[FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html):** the serving angle. KV lives in paged/block-sparse layouts, requests have wildly different lengths, so it uses a *plan* step to balance work across SMs and keeps kernels CUDA-Graph compatible.

### Shrinking the KV cache by changing the model
- **[Grouped-Query Attention](https://arxiv.org/abs/2305.13245):** H query heads share G key/value heads (G<H). KV cache shrinks by H/G (e.g. 64 → 8 = 8×).
- **[DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA):** project keys/values down into one small **latent** vector per token, cache only that, and up-project at attention time (the up-projection is folded into the query/output matmuls). Much smaller cache than even GQA, with quality close to full multi-head.

### Read the originals

- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf)
- [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
