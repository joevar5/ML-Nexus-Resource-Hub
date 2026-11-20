# Ex 01: Measure What FlashAttention Buys You

**Goal:** see the O(N²) → O(N) memory change and the speedup with your own
numbers, for the prefill shape that matters in serving. Question 5.
**Time:** ~2 h. **Needs:** NVIDIA GPU (Ampere+ for the flash backend; T4 falls
back to the memory-efficient backend — note which one ran).

## Steps

1. Write a naive causal attention:
   `softmax((q @ k.transpose(-1,-2)) / sqrt(d) + causal_mask) @ v`.
2. Compare it with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
   for B = 1, H = 32, d = 128, BF16, N ∈ {512, 2048, 8192, 16384}.
   Force the backend to be sure what you measure:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):   # or EFFICIENT_ATTENTION / MATH
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

3. For each N record time (CUDA events, median) and peak memory
   (`torch.cuda.reset_peak_memory_stats()` → `max_memory_allocated()`).
4. Check `torch.testing.assert_close(naive, flash, atol=2e-2, rtol=2e-2)` (BF16).

## Report

| N | naive ms | flash ms | speedup | naive peak MB | flash peak MB |
|---|---|---|---|---|---|

Then answer: at which N does naive attention OOM on your GPU, and what does
the N×N score matrix alone cost there (`B·H·N²·2` bytes)?

## Pass criteria

- Outputs match within BF16 tolerance.
- Flash peak memory grows linearly with N; naive grows quadratically (show it).
- One paragraph explaining the speedup with **HBM bytes**, not FLOPs (both do the same FLOPs).
