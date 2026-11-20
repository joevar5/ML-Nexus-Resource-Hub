# Source Digest — Performance Profiling

An **advanced layer** on top of the lecture notes: what the sources say, as steps. The tool manuals are huge; I read the first ~100K characters of each (noted below) and extracted the working method.

---

## 1. Nsight Systems: find where time goes
**Source:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) *(first 100K of 553K characters read; Timeline View section not reached)*

1. **Don't profile everything.** By default it profiles the whole run. Limit with `cudaProfilerStart()/Stop()` or **NVTX ranges**, then capture with `--capture-range=cudaProfilerApi` (or `nvtx` with `-p <range>`).
2. **Annotate with NVTX** so CPU thread activity is labelled and projected onto the GPU timeline. Name threads, streams and contexts.
3. **Run:** `nsys profile [options] ./app`.
4. **Key flags:** `--trace/-t` (default `cuda,opengl,nvtx,osrt`; add `cudnn,cublas,nccl,mpi`), `--capture-range`, `--duration/-d`, `--delay/-y`, `--sample/-s`, `--backtrace/-b`, `--stats=true`, `--output/-o`. Memory-usage tracing (`--cuda-memory-usage`) adds heavy overhead, so enable it only when needed.
5. **CUDA trace:** `cuda` uses hardware trace on Blackwell and newer; `cuda-sw` forces legacy software trace.
6. **Post-process:** `nsys stats` (summary tables), `nsys analyze` (expert rules), `nsys export` (other formats), `nsys recipe` (plots).
7. **Built-in rules tell you what to look for:** `gpu_gaps` (idle GPU between work), `gpu_time_util` (low utilization), `cuda_api_sync`, `cuda_memcpy_sync`, `cuda_memset_sync` (blocking synchronous calls), `cuda_memcpy_async`. List them with `nsys analyze --help-rules`.

---

## 2. Nsight Compute: why one kernel is slow
**Source:** [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) *(first 100K of 236K characters read)*

1. **Start light:** default `basic` set; `--set full` collects everything; `--list-sets`/`--list-sections` to browse. More metrics = more **replay passes** = more overhead.
2. **Replay types:** *kernel replay* re-runs the kernel and restores memory between passes; *application replay* reruns the program; *range replay* captures a span of API calls.
3. **Speed of Light first:** each unit's achieved % of its theoretical peak, with sub-metrics showing the limiting component.
4. **Roofline charts** classify the kernel (overview, plus per-precision: single, double, half, tensor).
5. **Memory Workload Analysis:** *Mem Busy*, *Max Bandwidth* (are paths saturated?), *Mem Pipes Busy* (is instruction issue the limit?), cache hit/miss rates, transfer volumes.
6. **Scheduler Statistics:** active vs eligible vs issued warps per scheduler. Many skipped issue slots = poor latency hiding.
7. **Warp State Statistics (stall reasons):** average cycles per instruction spent in each state. The guide's rule: **"Only focus on stalls if the schedulers fail to issue every cycle."** Map to source with Source Counters.
8. **Occupancy:** active ÷ max warps; limiters are registers, shared memory, block size and barriers. A big gap between theoretical and achieved usually means **load imbalance**.
9. **Measurement quality:** multi-pass collection can produce odd values if the workload varies between passes. Make the kernel long enough (**≈20 µs or more**) or collect fewer metrics.
10. **Loop:** change one thing, re-profile the same sections, confirm the bottleneck moved.

---

## 3. Methodology, from other sources
- **Honest benchmarking rules** (warm-up, locked clocks, L2 flush by buffer rotation, >1,000 iterations): see [Module 2 digest §6](../mod-002-cuda-programming/source-digest.md).
- **Roofline classification:** see [Module 1 digest §3](../mod-001-gpu-fundamentals/source-digest.md).
- **Bytes-accounting and the 208 FLOPs/byte A100 crossover:** [Module 1 digest §6](../mod-001-gpu-fundamentals/source-digest.md).
- **Serving-level metrics** (fluidity index, deadlines per token): [Module 1 digest §7](../mod-001-gpu-fundamentals/source-digest.md).

---

## 4. A profiling checklist built only from the above
1. Pick one metric (e.g. decode step time). Warm up, lock clocks, rotate buffers (Mod 2 §6).
2. `nsys` with NVTX ranges: run `nsys analyze` and read `gpu_gaps`, `gpu_time_util`, sync rules.
3. Choose the biggest kernel; `ncu --set full` on **one** launch.
4. Read Speed of Light → roofline → memory workload → scheduler stats → stalls (only if issue slots are skipped) → occupancy.
5. One change, re-measure, record.

---

## 5. Not read *(not fetched or not on this page)*
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) (AMD equivalent).
- **PyTorch profiler / memory-snapshot** content is not in the Wafer list; the lecture notes and exercises cover it.
