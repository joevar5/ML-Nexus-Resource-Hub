# Lecture 01: CUDA Graphs + Streams

## Streams

CUDA streams = ordered queue of GPU work. Operations in the same stream run
sequentially; operations in different streams may run in parallel.

```python
import torch
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.cuda.stream(s1):
    out1 = expensive_op(x)
with torch.cuda.stream(s2):
    out2 = another_op(y)
torch.cuda.synchronize()
```

Use for: overlapping compute with data transfer (e.g., copying the next batch in while the current one runs), or communication with compute.

## CUDA Graphs

Capture a sequence of CUDA ops once; replay many times. Eliminates per-launch overhead.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(x)
# Replay
g.replay()
```

Gains range from a few percent (large, compute-bound steps) to 2× or more for
small-batch decode of small models, where launch overhead dominates (Mod 3 Ex 03).
`torch.compile(mode="reduce-overhead")` enables this automatically.

## Caveats

- Captured graph is fixed-shape; new input shape = recapture
- Allocations during capture are pinned for replay (no malloc/free)
- Debugging is harder (errors surface at replay, not capture)

---

## Deep dive: how it works (core logic)

### Problem: launch overhead
Each kernel launch costs the CPU roughly 5–10 µs. A decode step for a deep model is hundreds of small kernels, many of which run for only microseconds. The GPU finishes each one and **waits for the CPU to issue the next**. On the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) timeline this shows up as gaps between kernels with the GPU mostly idle.

### CUDA graphs: record once, replay many
Capture a sequence of kernel launches into a graph; replaying it submits the *whole* sequence with one CPU call.
```python
g = torch.cuda.CUDAGraph()
static_in = torch.zeros(B, ...)                 # fixed buffers
with torch.cuda.graph(g):
    static_out = model(static_in)               # captured, not executed per call
# serving loop:
static_in.copy_(batch)                          # write into the same memory
g.replay()                                      # one launch
result = static_out.clone()
```
Constraints: **shapes and memory addresses are frozen at capture**. Real engines therefore capture one graph per **batch-size bucket** (1, 2, 4, 8, 16...), pad the batch up to the nearest bucket, and use static input buffers. Anything that changes per step (sequence lengths, block tables) must live *in* those buffers, not in kernel arguments.

### Making attention graph-friendly ([FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html))
Variable-length requests make work per block uneven. FlashInfer separates a CPU-side **plan** step (computes a balanced split of work across SMs from the current lengths) from the kernel, whose launch shape stays fixed, so it can be captured in a CUDA graph and still balance load.

### Streams and overlap
GPU work in different **streams** may run concurrently. Use a compute stream and a communication/copy stream, with events for dependencies, to hide all-reduce or host-copy time behind compute. [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) goes further: kernels issue communication themselves, so compute and comms can interleave inside one kernel. Within a kernel, [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) describes the producer/consumer pipeline (some warps load, others compute) behind async copy.

### Read the originals

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)
- [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
- [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
