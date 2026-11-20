# Ex 05: A Fused RMSNorm in Triton

**Goal:** write the kernel from question 15, benchmark it honestly, and explain
the result in bytes. **Time:** ~3 h. **Needs:** NVIDIA GPU, `triton`.

## Steps

1. Start from the `rmsnorm` kernel in Lecture 02's deep dive. Launch one
   program per row; `BLOCK = triton.next_power_of_2(N)`.
2. Reference (eager, unfused):
   `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w`, plus
   `torch.nn.functional.rms_norm` (PyTorch ≥ 2.4) and `torch.compile` of the
   eager version as two stronger baselines.
3. Correctness: BF16 input, FP32 accumulation; `assert_close` vs an FP32
   reference with `atol=1e-2`. Test N not a power of 2 (e.g. 5120) and rows = 1.
4. Benchmark `rows = 4096`, `N ∈ {2048, 4096, 8192}` with `triton.testing.do_bench`.
   Bandwidth = `2 · rows · N · 2 bytes / time` (read x, write y).
5. **Stretch:** fuse the residual add (`x = x + residual; y = rmsnorm(x)`,
   returning both) — the version real models use.

## Report

| N | eager ms | F.rms_norm ms | compiled ms | Triton ms | Triton GB/s | % of copy BW |
|---|---|---|---|---|---|---|

## Pass criteria

- Correct for every shape, including N = 5120.
- Triton is **≥ 2× faster than eager** and within 20% of `torch.compile` at N = 4096.
- Triton reaches **≥ 70% of copy bandwidth** (Mod 2 Ex 01) at N = 8192.
- Two sentences: how many HBM passes does eager make vs your kernel?
