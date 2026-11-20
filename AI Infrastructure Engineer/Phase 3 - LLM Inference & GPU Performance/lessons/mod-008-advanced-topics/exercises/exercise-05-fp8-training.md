# Ex 05: FP8 GEMMs — Speed and Error, Measured

**Goal:** see what FP8 buys at the kernel level and how scaling granularity
controls the error (questions 11 and 9). **Time:** ~2 h. **Needs:** an
FP8-capable GPU (L4, L40S, H100/H200, Blackwell), PyTorch ≥ 2.4.

## Steps

1. For shapes `M ∈ {16, 256, 4096}`, `K = N = 8192`, time BF16 `A @ B` against
   an FP8 GEMM with `torch._scaled_mm` (E4M3 inputs, BF16 output):

```python
def to_fp8(x, dim=None):                      # dim=None -> per-tensor, dim=-1 -> per-row
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = (amax.float() / 448.0).clamp(min=1e-12)
    return (x / scale).to(torch.float8_e4m3fn), scale

a8, sa = to_fp8(A)                            # A: [M, K] bf16
b8, sb = to_fp8(B.t().contiguous())           # B: [K, N]; quantize each column -> pass as column-major
out = torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
```

   (`_scaled_mm` is a private API; its argument rules change between versions —
   per-row scales need `scale_a` of shape `[M, 1]` and `scale_b` of shape `[1, N]`.)
2. Error: relative Frobenius error vs an FP32 reference, for per-tensor and
   per-row scaling. Then inject outliers into 4 columns of A (×100) and repeat.
3. TFLOPS = `2MNK / time` for each case.

## Report

| M | BF16 TFLOPS | FP8 TFLOPS | speedup | rel. error per-tensor | per-row | per-row with outliers |
|---|---|---|---|---|---|---|

## Pass criteria

- FP8 reaches ≥ 1.5× BF16 TFLOPS at M = 4096 on Hopper (less on Ada — state your GPU).
- You show where the speedup vanishes (small M: memory-bound) and why FP8
  *weights* still help decode there (half the bytes, Mod 1 Lecture 1.5).
- The outlier experiment shows why per-row (or per-block, Mod 5) scaling exists.
- Optional: train a small model with Transformer Engine `fp8_autocast` vs BF16
  and compare loss curves — the training side of the same idea.
