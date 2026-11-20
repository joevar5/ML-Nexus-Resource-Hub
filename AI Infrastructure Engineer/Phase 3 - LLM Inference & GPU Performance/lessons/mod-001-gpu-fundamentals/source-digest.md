# Source Digest — GPU Fundamentals

An **advanced layer** on top of the lecture notes. Each entry is what the source actually says, boiled down to steps and numbers. Everything here was read from the linked page or paper, except where marked *(not read)*.

How to use it: read the lecture note first, then use each entry below as a 5-minute "what does the original say" check.

---

## 1. Transpose: the cleanest lesson in coalescing and bank conflicts
**Source:** [Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) (NVIDIA)

1. **Copy kernel** is the ceiling: 105.2 GB/s (Tesla M2050), 136.0 GB/s (K20c).
2. **Naive transpose:** reads are coalesced, but writes stride 1,024 elements (4,096 B) apart between neighbouring threads, so they're uncoalesced. **18.8 GB/s** (M2050), 55.3 GB/s (K20c).
3. **Via shared memory:** read a 32×32 tile with coalesced rows into `__shared__ float tile[32][32]`, `__syncthreads()`, write it back out in coalesced rows (blocks of 32×8 threads, each handling four elements). 51.3 GB/s (M2050).
4. **Control experiment:** copy *through* shared memory without transposing: 104.6 GB/s. So shared memory and the barrier are not the problem. The remaining gap is bank conflicts.
5. **Pad the tile:** `tile[32][33]`. Column reads from a 32-wide tile all hit the same bank (32-way conflict); one extra column spreads them. **99.5 GB/s, ~95% of copy speed.**

**Remember:** coalesce global accesses first (18.8 → 51.3 GB/s, ~2.7× on M2050), then fix banks (~2× more). The control experiment is a good habit: isolate each suspect.

---

## 2. Parallel reduction: seven kernels, 30× faster
**Source:** [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (Mark Harris, NVIDIA)

Reducing 4M ints (time / bandwidth / step speedup / cumulative):

| Kernel | Change | Time | GB/s | Step | Cum. |
| :-: | :--- | :-: | :-: | :-: | :-: |
| 1 | Interleaved addressing, divergent branch (`tid % (2s)`) | 8.054 ms | 2.08 | — | — |
| 2 | Interleaved addressing, strided index (no divergence) | 3.456 ms | 4.85 | 2.33× | 2.33× |
| 3 | **Sequential addressing** (fixes bank conflicts) | 1.722 ms | 9.74 | 2.01× | 4.68× |
| 4 | **First add during global load** (halves idle threads) | 0.965 ms | 17.4 | 1.78× | 8.34× |
| 5 | **Unroll last warp** (no `__syncthreads` needed inside a warp) | 0.536 ms | 31.3 | 1.80× | 15.0× |
| 6 | Completely unrolled (block size as a template parameter) | 0.381 ms | 44.0 | 1.41× | 21.2× |
| 7 | **Multiple elements per thread** (algorithm cascading) | 0.268 ms | 62.7 | 1.42× | 30.0× |

Kernel 7 reaches **73 GB/s on 32M elements**, close to the card's peak (the deck computes ~86.4 GB/s from clock × bus width).

**Method to steal:** the deck first computes the *target* (1 FLOP per element loaded → bandwidth-bound → aim for peak GB/s), then each step is explained by the bottleneck it removes. Always know your ceiling before you optimize.

---

## 3. Roofline: where the ceilings come from
**Source:** [Roofline: An Insightful Visual Performance Model](https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf) (Williams, Waterman, Patterson)

1. **Operational intensity** = FLOPs per byte of **DRAM** traffic (after caches). The authors prefer this over "arithmetic intensity" because it counts only traffic that reaches memory.
2. **The model:** `Attainable GFLOPs/s = min(Peak GFLOPs/s, Peak memory bandwidth × operational intensity)`.
3. **Plot** (log-log): x = operational intensity, y = performance. A sloped line (memory-bound) meets a flat line (compute-bound).
4. **Ridge point** = the lowest intensity that reaches peak compute. If it sits far right, only kernels with very high reuse can hit peak, so the machine is harder to program.
5. **Worked example:** AMD Opteron X2: 17.6 GFLOPs/s peak, 15 GB/s memory → ridge ≈ 1.0. Its successor X4 (more cores, same memory bandwidth) shifts the ridge to **4.4**: kernels must now do more than 4× the work per byte to benefit.
6. **Ceilings:** extra lower lines for "you haven't done optimization X" (e.g., no SIMD, no ILP, no prefetch). To pass through a ceiling you must do the matching optimization. Gives an ordered to-do list.
7. **How to use it:** place your kernel's dot on the plot. Below the sloped line → fix memory behaviour. Below the flat line → fix compute behaviour.

**Modern echo:** GPU ridge points keep moving right each generation (see Section 8).

---

## 4. Volkov: fastest GEMM came from breaking the "rules"
**Source:** [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf) (Volkov & Demmel, SC08)

1. **Claim:** their GEMM runs **up to 60% faster than the vendor library** and approaches hardware peak; LU/QR/Cholesky reach 80–90% of peak GEMM speed.
2. **Mental model change:** treat the GPU as a **multithreaded multicore vector unit**, not a sea of scalar threads. Use blocking like vector machines do.
3. **Method:** microbenchmark the hardware first: cache and TLB sizes, memory latencies (in cycles), kernel start-up costs and arithmetic throughput. Then choose optimizations that match what you measured.
4. **Key guideline reversal:** use the **register file as the primary on-chip storage** and run **fewer threads with more work each** (shorter "vectors" at program level), instead of maximizing occupancy.
5. **Also shown:** use the CPU in parallel for panel factorizations (BLAS1/2-dominated), and an LU variant using explicit inverses of diagonal blocks, faster and as numerically stable.

**Remember:** high occupancy is a way to hide latency, not the goal. Enough *independent work in flight* is the goal, and you can get it from registers and ILP.

---

## 5. How to scale a model's inference
**Source:** [How to Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) (JAX team)

1. **Two phases:** prefill (whole prompt, builds KV cache) and generation (one token per step).
2. **KV size:** `2 × bytes × head_dim × kv_heads × layers × tokens`. LLaMA-2 13B in bf16: ~**6.7 GB for one 8,192-token sequence**.
3. **Compute-bound threshold:** B_crit = (FLOPs/s ÷ HBM bandwidth) × (bits/param ÷ bits/activation). About **240 tokens** for bf16 on TPU v5e; halving weight precision roughly halves it.
4. **Attention intensity:** prefill ≈ T/2 (compute-bound beyond ~480 tokens); **generation ≈ 1 regardless of batch** (always memory-bound).
5. **Latency floor:** step time ≥ (batch × KV size + parameter bytes) ÷ total memory bandwidth. Example: 30B int8 on 16 v5e, batch 4 → ~2.5 ms; batch 256 → ~21 ms.
6. **Batching trade-off:** LLaMA-2 13B on 8 v5e: ~200 tok/s at batch 1 → ~964 at batch 240. Cutting KV 5× (GQA) → ~4,500 tok/s at batch 240.
7. **Parallelism:** prefill ≈ training (model sharding, then sequence sharding). Generation: avoid FSDP and plain data parallelism; shard the model, and shard KV by heads then batch.
8. **Serving design:** disaggregate prefill and generation; continuous batching; prefix caching.
9. **KV reduction levers:** GQA, local attention, quantization, paged attention.

---

## 6. Transformer inference arithmetic
**Source:** [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) (kipply)

Worked on a 52B model (d_model 8192, 64 layers, 2 bytes/param):

1. **Weights:** ~104 GB (so three 40 GB A100s, leaving ~16 GB for KV).
2. **FLOPs per token:** ≈ **2 × P** (≈103 B FLOPs for this model).
3. **KV per token:** `4 × n_layers × d_model` bytes = 2,097,152 B (~2 MB), so ~**8,000 tokens** fit in 16 GB.
4. **A100 crossover:** 312 TFLOPs ÷ 1.5 TB/s ≈ **208 FLOPs/byte**. Below that, memory-bound; above, compute-bound.
5. **Time per token (batch ≤ 208):** `12·2·n_layers·d_model² / 1.5e12` ≈ **69 ms** on one GPU (≈17 ms across four).
6. **Minimum batch to become compute-bound:** ≈ **208**.
7. **Communication:** 4 per layer; total `4·(N−1)·d_model/N` bytes; ~8 µs per message latency. Example: a 260B model on 16 GPUs ≈ 22 ms/token at small batch.

*(The page's formulas were partly lost in conversion; the KV and parameter steps are derived from the surrounding text.)*

---

## 7. Etalon: honest serving metrics
**Source:** [Etalon](https://arxiv.org/html/2407.07000)

1. **Fluidity-index:** each token *i* has a deadline `D_i = D_p + i × D_d` (prefill target + per-token decode target). Score = fraction of deadlines met (0–1). After a miss, later deadlines reset from the miss time; early tokens earn slack.
2. **Fluid token generation rate:** the highest playback rate that still meets a fluidity target (e.g. 99% of requests with index ≥ 0.9).
3. **Why old metrics fail:** TTFT depends on prompt length; normalized latency hides scheduling delay; TPOT averages away stalls; tail TBT shows stalls exist but not how long or how often, and misreads speculative-decoding bursts.
4. **Findings:** Groq's TPOT implied ~600 tok/s but tail TBT implied ~4× lower; Fireworks delivered ~90% of decode tokens in bursts (speculative decoding) with the lowest, steadiest TTFT; Llama3-8B on H100: vLLM and Sarathi-Serve capacity matched at 0.6 QPS, but Sarathi-Serve had fewer deadline misses thanks to chunked prefill.

---

## 8. What the CUDA docs state (Hopper and Blackwell)
**Sources:** [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/), [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

| Fact | Hopper (CC 9.0) | Blackwell (CC 10.0) |
| :--- | :--- | :--- |
| Max warps per SM | 64 | 64 |
| Registers per SM | 64K × 32-bit (max 255 per thread) | same |
| Max blocks per SM | 32 | 32 |
| Shared memory per SM | 228 KB (A100: 164 KB) | 228 KB |
| Max shared per block | 227 KB | 227 KB |
| L1+shared combined | 256 KB | 256 KB |
| L2 cache | 50 MB | up to 126 MB on GB200 |
| Memory | HBM3/HBM2e up to 80 GB, up to ~3 TB/s | HBM3/3e up to 180 GB |
| NVLink | 4th gen: 18 links × 50 GB/s = **900 GB/s** | 5th gen (faster; figure not on the page) |
| Cluster size | portable 8; 16 with opt-in | same |

- **TMA (Hopper):** one thread issues a copy of a 1D–5D tensor between global and shared memory (and between SMs in a cluster), with optional reductions on write. Frees registers and instructions; enables warp-specialized kernels via `cuda::memcpy_async` + `cuda::barrier`/`pipeline`.
- **Clusters + distributed shared memory:** blocks in a cluster can read each other's shared memory. Keep DSM accesses coalesced and 32-byte aligned; use `cudaOccupancyMaxActiveClusters`.
- **Static shared memory is capped at 48 KB;** more needs an explicit opt-in (carveouts: 0…228 KB).
- **Advice from both guides:** first apply standard practice (coalescing, fewer host-device transfers, less divergence). Occupancy behaviour is similar to Ampere/Hopper without code changes.
- *Correction to be aware of:* the Blackwell **tuning guide** page does not itself discuss tensor memory or the new data types. Treat those as architecture-brief material *(brief was password-protected; not read)*.

---

## 9. Programming model and first kernel
**Sources:** [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [CUDA C++ basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)

1. Host launches kernels; both can run concurrently. Launch is asynchronous.
2. Threads → blocks (1D/2D/3D, all on **one SM**) → grid. Blocks may run in any order, so they **must not depend on each other**.
3. Optional **clusters** (CC 9.0+) schedule adjacent blocks together in one GPC.
4. A block's threads split into **warps of 32** (SIMT); diverging threads are masked. Use block sizes that are multiples of 32.
5. Memory: registers (thread) → shared (block/cluster) → L1 → L2 (all SMs) → global → host memory (slow, over PCIe/NVLink). Unified memory gives one address space.
6. **Write a kernel in 5 steps:** `__global__` function with bounds check; index = `threadIdx.x + blockIdx.x*blockDim.x`; pick 256 threads/block and `ceil_div(n, 256)` blocks; launch `<<<blocks, threads>>>`; either `cudaMallocManaged` + `cudaDeviceSynchronize`, or `cudaMalloc` + `cudaMemcpy` + sync.
7. **Check errors:** compare every API result to `cudaSuccess`; after a launch call `cudaGetLastError()` then `cudaDeviceSynchronize()`.

---

## 10. Best Practices Guide: the priorities
**Source:** [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) *(first ~100K of 239K characters read)*

1. **APOD cycle:** Assess (profile; use Amdahl) → Parallelize (libraries first: cuBLAS, cuFFT, Thrust) → Optimize (measure each change) → Deploy early, repeat.
2. **Biggest memory win:** minimize host↔device transfers. V100 device memory ≈ 898 GB/s vs PCIe x16 Gen3 ≈ 16 GB/s (pinned ≈ 12 GB/s).
3. **Coalescing:** on CC 6.0+, 32 threads reading adjacent 4-byte words need four 32-byte transactions; misaligned adds a fifth; stride-2 gives 50% efficiency.
4. **Shared memory:** 32 banks × 32 bits; conflicting addresses serialize; same-address reads broadcast. Use it to reorder uncoalesced access.
5. **Occupancy:** modern GPUs run up to 2,048 threads per SM (160,000+ threads on 80 SMs); block sizes should be multiples of 32; `cudaMalloc` aligns to ≥256 B.

---

## 11. Low-precision formats (spec level)
**Sources:** [OCP FP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf), [OCP Microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

**FP8:**
- **E4M3:** max normal **448**.
- **E5M2:** max normal **57,344**.

**MX (read from the spec):**
1. A **block of k = 32 elements** shares one **scale X** of type **E8M0** (8 bits, power-of-two scale with a Float32-style biased exponent, no sign bit).
2. A block costs `w + k·d` bits (w = scale bits, d = element bits).

| Format | Element type | d | k | Scale |
| :--- | :--- | :-: | :-: | :--- |
| MXFP8 | E5M2 or E4M3 | 8 | 32 | E8M0 |
| MXFP6 | E3M2 or E2M3 | 6 | 32 | E8M0 |
| MXFP4 | E2M1 | 4 | 32 | E8M0 |
| MXINT8 | INT8 | 8 | 32 | E8M0 |

3. **Effective cost:** MXFP4 = (8 + 32×4) ÷ 32 = **4.25 bits per element**; MXFP8 = 8.25.

**Compare with NVFP4** (Mod 5): blocks of **16** with an FP8 scale, giving finer scaling for the same 4-bit elements.

---

## 12. Other vendors: facts read from the docs
- **AWS Trainium3** ([architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html)): 8 NeuronCore-v4, 4 HBM stacks = **144 GiB at 4.7 TB/s**; SBUF 32 MiB, PSUM 2 MiB; Tensor Engine peak **315 MXFP8/MXFP4 TFLOPS**, 79 BF16/FP16/TF32 TFLOPS, 20 FP32 TFLOPS; MX matmul with **32-element scaling groups** at 4× BF16 throughput; hardware background transpose.
- *Not read (fetch failed or locked):* [Blackwell architecture brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) (password-protected), [CDNA 4 whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) (timed out), [TPUv4i lessons](https://doi.org/10.1109/ISCA52012.2021.00010).

---

## One-page summary: what Module 1 teaches, from the sources

1. Know your **ceiling** (peak GB/s or FLOPs) before optimizing: the reduction deck and the roofline paper both start there.
2. **Memory first:** coalescing gave the biggest jump in the transpose (~2.7× on M2050 vs. naive), then bank conflicts (~2×).
3. **Occupancy is a means, not a goal** (Volkov): registers + ILP + fewer threads can win.
4. LLM decode is **memory-bound at any batch below ~200–300** (roofline ridge): this one fact explains batching, quantization, GQA and KV tricks.
5. **Measure with honest metrics** (Etalon): deadlines per token, not averages.
6. Every new architecture adds async data movement (TMA) and lower-precision formats. Without them, code gets *relatively* slower on new chips.
