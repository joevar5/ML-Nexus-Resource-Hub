# Lecture 02: Bottleneck Identification

## The roofline model

Plot kernel performance against arithmetic intensity (FLOPS/byte). A kernel
hits one of two ceilings:
- **Memory-bound**: below the slope (bandwidth limit)
- **Compute-bound**: at the horizontal ceiling (peak FLOPS)

If you're memory-bound, more FLOPS won't help — optimize memory access.
If you're compute-bound, more bandwidth won't help — optimize the math
(tensor cores, fp16/bf16, better algorithm).

ncu reports arithmetic intensity directly.

## Common bottlenecks

| Symptom (nsys) | Diagnosis | Fix |
|---|---|---|
| Long CUDA API calls | sync between host + device | reduce `cudaMemcpy`; use streams |
| Small kernel, called many times | launch overhead dominates | batch larger; use CUDA graphs |
| One kernel takes 80%+ | true hotspot; ncu it | per-kernel optimization |
| Many tiny kernels in sequence | kernel fusion opportunity | torch.compile / custom kernel |
| Long memcpy time | host↔device transfer is the bottleneck | overlap with compute via streams |

## Speed-of-Light interpretation

ncu's "GPU Speed of Light" section reports:
- % of peak FLOPS
- % of peak memory bandwidth

If both are low → the kernel is **latency-bound**: it isn't using the GPU well.
Usual causes: too few blocks to fill the SMs (small batch, decode GEMVs),
warps stalled on dependencies or barriers (check *warp stall reasons*),
or a kernel so short that launch overhead dominates. Divergence is a
rarer cause; the stall-reason breakdown tells you which.

## NVTX ranges

Decorate your code:

```python
with torch.cuda.nvtx.range("forward"):
    out = model(x)
with torch.cuda.nvtx.range("backward"):
    out.sum().backward()
```

Now nsys shows your high-level structure on the timeline. Critical for
finding which Python-level phase is the hotspot.

---

## Deep dive: how it works (core logic)

### A repeatable diagnosis loop
1. **Define the metric** (latency of one step? tokens/s? p99?). Without it you can't know you improved.
2. **Locate** with a timeline (`nsys`): which kernel or gap dominates?
3. **Classify** that kernel with the roofline: compute AI = FLOPs ÷ bytes and compare with the ridge point ([Roofline: An Insightful Visual Performance Model](https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf)); Nsight Compute draws this for you.
4. **Act on the class:**
   - *memory-bound* → fuse ops, quantize, use FlashAttention-style tiling, improve coalescing;
   - *compute-bound* → use tensor cores / lower precision, fix tile shapes;
   - *latency/launch-bound* → CUDA graphs, fuse tiny kernels, larger batches;
   - *CPU-bound* → remove Python in the hot loop, async work.
5. **Change one thing, re-measure, keep a log.**

### Measuring correctly ([CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html))
- **Warm up** (first runs include JIT, allocation, clock ramp).
- **Synchronize** before reading timers (`torch.cuda.synchronize()` or CUDA events), since kernel launches are asynchronous.
- **Fix clocks** or at least report them; boost clocks drift with temperature/power.
- **Flush L2** between runs when benchmarking memory-bound kernels, or you measure cache, not HBM.
- Report median plus spread over many iterations, with the shape, dtype and baseline stated.

### System-level metrics ([Etalon](https://arxiv.org/html/2407.07000))
Kernel-level wins do not always show up for users. Etalon defines serving metrics beyond averages: TTFT, TPOT, and a *fluidity* view where every token has a deadline, so a stall mid-stream counts against you even if the average TPOT is fine.

### Read the originals

- [Roofline: An Insightful Visual Performance Model](https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf)
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
- [Etalon](https://arxiv.org/html/2407.07000)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
