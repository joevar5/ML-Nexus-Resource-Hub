# Source Digest — CUDA Programming

An **advanced layer** on top of the lecture notes. Each entry is what the source actually says, in steps and numbers. All entries were read from the source unless marked *(not read)*.

---

## 1. The SGEMM worklog: 1.3% to 93.7% of cuBLAS
**Source:** [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) (fp32 SGEMM)

| # | Kernel | What changed | GFLOPs | % of cuBLAS |
| :-: | :--- | :--- | :-: | :-: |
| 1 | Naive | One thread = one C element, all reads from global | 309 | 1.3% |
| 2 | **Coalescing** | Remap threads so a warp reads consecutive addresses (~15 → ~110 GB/s) | 1,986 | 8.5% |
| 3 | **Shared-memory blocking** | Block loads A and B tiles into SMEM, reuses them | 2,980 | 12.8% |
| 4 | **1D blocktiling** | Each thread computes several C entries held in registers | 8,475 | 36.5% |
| 5 | **2D blocktiling** | Each thread computes an 8×8 tile; threads cooperatively fill SMEM | 15,972 | 68.7% |
| 6 | **Vectorized access** | Transpose A tile in SMEM; `float4` global loads/stores | 18,237 | 78.4% |
| 9 | **Autotuning** | Benchmarked ~400 tile-size combinations | 19,721 | 84.8% |
| 10 | **Warptiling** | Add a warp-level tile between block and thread tiles | **21,779** | **93.7%** |

Final: 21,779 vs cuBLAS 23,250 GFLOPs. (Kernels 7–8 were skipped because slower.)

**Lessons the author draws:**
1. **Coalescing alone gave the biggest early jump** (6.4× from kernel 1 to 2).
2. **Reuse data:** each thread computing more outputs raises arithmetic intensity (FLOPs per byte moved).
3. **Let the profiler pick the next step:** warps stalled on shared-memory instruction queues, which led to register tiling.
4. **Tuning is GPU-specific**, so autotune. cuBLAS picks among hundreds of kernels by shape, so a hand kernel falls further behind on small matrices.
5. **Not every idea works:** thread swizzling for L2 locality gave nothing (L2 hit rate was already ~80%).

---

## 2. Hopper matmul: from 32 to 764 TFLOPs
**Sources:** [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul)

cuBLAS reaches **716 TFLOPs** (N=4096, bf16). The worklog's kernels (TFLOPs / % of cuBLAS):

| Step | Technique | TFLOPs | % |
| :--- | :--- | :-: | :-: |
| 1 | Warp-tiled CUDA-core kernel, bf16 | 32 | 4% |
| 2 | **Tensor cores (`wgmma`) + TMA loads with swizzled layout** | 317 | 44% |
| 3 | Larger tile (BM=128, BN=128, BK=64) | 423 | 59% |
| 4 | **Warp specialization:** producer warpgroup loads into a circular SMEM buffer, consumers do MMA | 498 | 70% |
| 5 | 128×256 tiles, two consumer warpgroups (avoids register spills); per-warpgroup register reallocation | 631 | 88% |
| 6 | **Persistent kernel** (one block per SM = 132) + tile grouping for L2 reuse (L2 hit 83% vs cuBLAS 70%) | 660 | 92% |
| 7 | Lower-level PTX `mbarrier` instead of CUDA barrier API | 704 | 98% |
| 8 | **Clusters + TMA multicast** (two tiles share B loads) | 734 | 102% |
| 9 | Micro-optimizations (store order, skip accumulator zeroing) | 747 | 104% |
| 10 | Async stores via shared memory + TMA | 758 | 106% |
| 11 | **Hilbert-curve tile ordering** for L2 locality | 764 | 107% |

**Ideas to understand, per the Anatomy article:**
- The memory hierarchy on an H100 SXM5 (132 SMs, four warp schedulers per SM, up to 64 resident warps): bandwidth rises and latency falls as data moves closer (HBM → L2 → distributed SMEM → L1/SMEM → registers).
- **Naive kernel, same code:** ~3,171 GFLOP/s coalesced vs ~243 GFLOP/s uncoalesced on H100 PCIe (swapping two index operators): ~13× slower.
- **Matmul as a sum of outer products** → use square output tiles and several outputs per thread to raise FLOPs per byte.
- **128B swizzle:** XOR-remap shared-memory addresses so TMA-written tiles avoid bank conflicts.
- **Producer/consumer pipeline:** a ring of "full"/"empty" barriers lets one warpgroup drive TMA while others feed tensor cores.

---

## 3. Why LLM outputs differ run to run (and how to fix it)
**Source:** [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

1. **Common myth:** "concurrency + floating point" causes it. **Actual cause:** atomics are rare in LLM forward passes, so a forward pass is run-to-run deterministic *for a given batch*. What varies is **batch size**, which changes with server load, and kernels aren't **batch-invariant**.
2. **Batch-invariant** = each element's reduction order stays fixed regardless of batch size. Three ops need it: **RMSNorm, matmul, attention**.
3. **RMSNorm:** assign each row to one core so reductions stay inside a core; avoid split reductions at small batch.
4. **Matmul:** use **one kernel configuration for all shapes**. Don't switch to Split-K or smaller tensor-core instructions for small batches. Cost: ~**20% slower than cuBLAS** (unoptimized).
5. **Attention:** update the KV cache/page table *before* the attention kernel so layouts are identical for prefill and decode; use a **fixed split size** (not a fixed number of splits) for split-KV decoding.
6. **Serving cost (Qwen3-8B, 1 GPU, 1,000 requests):** vLLM default 26 s; deterministic 55 s; with improved attention 42 s.
7. **Experiment:** Qwen3-235B, 1,000 completions at temperature 0 of 1,000 tokens → **80 unique outputs**; identical for 102 tokens, diverging at token 103 ("Queens, New York" ×992 vs "New York City" ×8). With batch-invariant kernels: **all 1,000 identical**.
8. **Why it matters for RL:** sampler/trainer numeric mismatch makes "on-policy" RL secretly off-policy; reward collapsed without importance weighting, and bitwise-identical numerics gave KL = 0.

---

## 4. Compilation: what nvcc does with your code
**Sources:** [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)

1. **Pipeline:** nvcc splits device and host code. Device code → **PTX** → `ptxas` → **cubin** (real GPU binary). Both can be packed into a **fatbinary** embedded in the host object.
2. **Naming:** `compute_XY` = virtual architecture (PTX); `sm_XY` = real GPU binary. A cubin runs on the same major compute capability and equal-or-higher minor; not across majors.
3. **Flags:** `-arch=X -code=Y` ≡ `-gencode=arch=X,code=Y`; repeat `-gencode` to cover several GPUs. Default arch is `sm_75`.
4. **Forward compatibility:** embed PTX and the driver **JIT-compiles** it for newer GPUs (a `compute_70` PTX runs on CC ≥ 7.0).
5. **Useful flags:** `-ptx` / `-cubin` (device-only output), `-lineinfo` (line numbers for profilers without `-G` debug cost), `-keep`/`-save-temps` (keep intermediates).
6. **Hand-written PTX** is "not usually necessary or advisable", but sometimes used in hot inner loops (`cuda::ptx` namespace in libcu++, or inline PTX).

---

## 5. Compute Sanitizer: four tools, four bug classes
**Source:** [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

| Tool | Catches |
| :--- | :--- |
| `memcheck` (default) | out-of-bounds / misaligned access (global, local, shared), device malloc/free errors, leaks (`--leak-check full`) |
| `racecheck` | shared-memory hazards: WAW, WAR, RAW |
| `initcheck` | reads of uninitialized global memory (`--initcheck-address-space shared` for shared) |
| `synccheck` | invalid `__syncthreads()` / `__syncwarp()`, including divergent barriers |

Usage: `compute-sanitizer --tool <name> ./app`. Compile with `-lineinfo` (or `-G`) for source lines; `-fdevice-sanitize=memcheck` gives faster, better instrumentation.

---

## 6. How to benchmark a GEMM honestly
**Source:** [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)

1. **Warm up** in a separate loop. Back-to-back GEMMs show clock oscillation for roughly the first ~3 s.
2. **Fixed-frequency test:** lock a sustainable clock, zero-fill data. **Fixed-power test:** run as shipped (DVFS on), fill with uniform [-1, 1] data.
3. **Allocate everything before the loops**; no allocations, copies or extra kernels between launches.
4. **Iterations:** >1,000 for ~4096³; for large Blackwell GEMMs, 10,000 warm-ups and 4,000 profiled runs.
5. **Timing:** Nsight Systems or CUDA events, then compute achieved FLOPS.
6. **Defeat the L2 cache:** make total input ≥ **2× L2**, use ≥2 buffers per tensor and rotate pointers each iteration; confirm with L2 hit rates (`--cache-control none` in ncu).
7. **Report:** clock, power, temperature, and compare to speed-of-light.

---

## 7. Still to read *(not fetched)*
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf), [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/).
- The reduction, transpose and Best Practices sources for this module are summarized in the [Module 1 digest](../mod-001-gpu-fundamentals/source-digest.md).
