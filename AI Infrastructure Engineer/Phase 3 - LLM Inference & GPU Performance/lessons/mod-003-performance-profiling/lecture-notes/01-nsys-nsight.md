# Lecture 01: nsys + Nsight Compute

## Two profilers

| Tool | Granularity | Use for |
|---|---|---|
| `nsys` (Nsight Systems) | timeline view; what happens when | "where is the time going across the full pipeline?" |
| `ncu` (Nsight Compute) | per-kernel deep dive | "why is this specific kernel slow?" |

Start with nsys to find hotspots; drop into ncu to optimize them.

## nsys workflow

```bash
nsys profile -o trace.nsys-rep python train.py
nsys-ui trace.nsys-rep    # GUI
nsys stats trace.nsys-rep # text summary
```

Key views:
- **CUDA API**: time spent in malloc/free/memcpy/launch
- **CUDA kernels**: per-kernel total time + invocation count
- **NVTX ranges**: your code-defined regions ("forward", "backward", "optimizer step")

## ncu workflow

```bash
ncu --set full --kernel-id ::matmul_kernel:1 python train.py
ncu -i report.ncu-rep --section MemoryWorkloadAnalysis
```

Key sections:
- **GPU Speed of Light**: roofline placement
- **Memory Workload Analysis**: bandwidth + cache hit rates
- **Compute Workload Analysis**: throughput per pipeline
- **Source/SASS view**: per-instruction hotspots

## PyTorch profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
) as prof:
    for _ in range(10):
        train_step()
        prof.step()
```

Then `tensorboard --logdir=log`. Easier than nsys for PyTorch-only workloads;
nsys when you need to see beyond PyTorch.

## Companion

[engineer-solutions/mod-107 ex-07 (gpu-memory-profiling)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-07-gpu-memory-profiling).

---

## Deep dive: how it works (core logic)

### Workflow: timeline first, then microscope
**Step 1: Nsight Systems ([Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)).**
```
nsys profile -o report --trace=cuda,nvtx,osrt --cuda-graph-trace=node python run.py
```
Add NVTX ranges (`torch.cuda.nvtx.range_push("prefill")`) so the timeline has your own labels. Read three things:
- **GPU rows:** are kernels back to back, or are there gaps? Gaps = launch/CPU/sync problems.
- **Memcpy rows:** unexpected host↔device copies.
- **CPU rows:** the Python/launch thread; if it is 100% busy while the GPU idles you are CPU-bound.
Pick the *biggest* block of time; ignore the rest.

**Step 2: Nsight Compute ([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).**
```
ncu --set full -k regex:attention -s 5 -c 1 -o k python run.py
```
(`-s` skips warm-up launches, `-c` limits captured launches. ncu replays the kernel many times, so it is slow; target one kernel.)
Read the **Speed of Light** section first: SM throughput % vs memory throughput %. Then the roofline plot, then **warp stall reasons** (waiting on memory, on a barrier, on a dependency...) and occupancy/launch statistics.

### Interpretation table
| Speed of Light shows | Meaning |
| :--- | :--- |
| Memory % high, SM % low | memory-bound → move fewer bytes |
| SM % high | compute-bound → better algorithm/precision |
| Both low | latency-bound: too little parallelism, stalls, or launch overhead |

On AMD, [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) plays the ncu role (counters + roofline).

### Read the originals

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
