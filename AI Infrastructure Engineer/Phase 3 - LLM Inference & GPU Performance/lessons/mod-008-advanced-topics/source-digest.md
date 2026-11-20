# Source Digest — Advanced Topics

An **advanced layer** on top of the lecture notes. What the sources actually say, as steps and numbers. Read from the source text unless marked *(not read)*.

---

## 1. NCCL device-side communication
**Source:** [NCCL device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)

1. Since **NCCL 2.28**, your own CUDA kernels can call communication primitives directly instead of relying only on host-launched collectives.
2. **Prerequisite:** symmetric memory windows (needs `NCCL_CUMEM_ENABLE` / cuMem management).
3. **Three paths; choose by where the peer sits:**

| Path | Reaches | How |
| :--- | :--- | :--- |
| **LSA** (Load/Store Accessible) | peers on CUDA P2P (NVLink, some PCIe) | ordinary loads/stores on remote buffers via `ncclGetLsaPointer` |
| **Multimem** | peers with NVLink SHARP multicast (Hopper+ datacenter GPUs) | one multicast pointer reaches all ranks; **reductions in hardware** (`multimem.ld_reduce`) |
| **GIN** (GPU-Initiated Networking, from 2.28.7) | peers across the network | kernels issue **one-sided puts**, attach signals, wait on signals/barriers (CPU-proxy and GDAKI backends) |

4. **Host setup:** `ncclMemAlloc` → register with `ncclCommWindowRegister` (`NCCL_WIN_COLL_SYMMETRIC`) → set `ncclDevCommRequirements` (e.g. `lsaBarrierCount` = CTA count) → `ncclDevCommCreate` → launch kernel with the device communicator.
5. **Device code:** open an `ncclLsaBarrierSession`, address peers with `ncclGetLsaPointer(win, offset, peer)`, barrier before exit. The doc's in-place AllReduce example is intentionally unoptimized.

**Why it matters:** this is what lets MoE dispatch/combine (DeepEP, which uses NCCL's GIN backend) overlap communication with compute inside one kernel. See [Module 6 digest §2](../mod-006-distributed-inference/source-digest.md).

---

## 2. Interconnect facts from the sources
- **Hopper NVLink (4th gen):** 18 links × 50 GB/s = **900 GB/s** per GPU (A100: 12 links, 600 GB/s) — [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- **Splitwise's A100 vs H100 table** lists **InfiniBand 200 GB/s vs 400 GB/s (2×) per 8-GPU node** (8 × 200 vs 8 × 400 Gb/s ports), noting memory bandwidth grew slower than compute, so token generation (memory-bound) benefits less from newer GPUs than prompt processing.
- **DeepSeek-V3 prefill all-to-all:** tokens cross nodes over **InfiniBand**, then are forwarded among GPUs inside the node over **NVLink**; **decode** uses direct IB point-to-point with **IBGDA** to minimize latency — [DeepSeek-V3](https://arxiv.org/abs/2412.19437).
- **Multi-node NVLink (GB200 NVL):** the tuning guide's landing page lists sections on system, networking, UCX, NCCL, power/thermals and benchmarking; the technical pages were not read.
- **[Ultra Ethernet spec](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf):** *not read.*

---

## 3. CUDA graphs and streams
**Sources:** the Nsight Systems rules and DeepEP/DeepSeek notes; *no dedicated CUDA-graph paper is in the Wafer list.*
1. **Finding the problem:** Nsight Systems `gpu_gaps` and `gpu_time_util` rules flag idle GPU between work ([Module 3 digest §1](../mod-003-performance-profiling/source-digest.md)).
2. **A serving example:** DeepEP's decode mode sets `do_cpu_sync=False` and pre-allocates output capacity with GPU-side counts, which is what makes the kernels **capturable in CUDA graphs** (no CPU round trip for sizes).
3. **Warp-specialized pipelines** (async producer/consumer) as in the H100 matmul worklog: [Module 2 digest §2](../mod-002-cuda-programming/source-digest.md).
4. **FlashInfer** (CUDA-graph-compatible load balancing) and CUTLASS pipeline docs: *not read.*

---

## 4. FP8: format facts and where it is used
**Sources:** [OCP FP8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), [FlashAttention-3](https://arxiv.org/abs/2407.08608)
1. **Formats:** E4M3 max **448**; E5M2 max **57,344** (the spec text gives these as `max_E4M3` and `max_E5M2`).
2. **Transformer Engine:** FP8 on Hopper, Ada, Blackwell (and Rubin); **MXFP8 and NVFP4 on Blackwell**; tracks **scaling factors and amax histories**; PyTorch and JAX.
3. **FlashAttention-3 in FP8:** reaches close to **1.2 PFLOPs/s** with **2.6× lower numerical error** than baseline FP8 attention, using **block quantization** and **incoherent processing** (details: [Module 4 digest §4](../mod-004-transformer-optimization/source-digest.md)).
4. **MX/NVFP4 block formats:** [Module 1 digest §11](../mod-001-gpu-fundamentals/source-digest.md) and [Module 5 digest §7](../mod-005-model-compression/source-digest.md).
5. **DeepGEMM** (FP8 GEMM with fine-grained scaling for Hopper): *not read.* Earlier claims in the lecture note about its accumulation strategy are from general knowledge and unverified.

---

## 5. MIG and the frontier
- **MIG:** not covered by the Wafer list; use NVIDIA's MIG documentation.
- **Frontier items** (KernelBench, KernelBench-Verified, FlashInfer-Bench, SOL-ExecBench, KernelGuard, hardware watchlist): *not read.* They sit in the "Frontier" section of the [Wafer reading list](../../resources/reading-lists/wafer-gpu-perf-reading-list.md).
- **Ring Attention** (distributed exact attention): the paper distributes long sequences across devices by **circulating KV blocks around a ring while overlapping communication with blockwise attention compute**, enabling sequences up to **device-count times longer** than memory-efficient baselines, without approximation — [Ring Attention](https://arxiv.org/abs/2310.01889).
