# Ex 01: Vector Add Kernel + Bandwidth Ceiling

**Goal:** write your first kernel, call it from PyTorch, and measure it against the
real ceiling for a memory-bound op: device-to-device copy bandwidth.
**Time:** ~2 h. **Needs:** any NVIDIA GPU (Colab/Kaggle T4 is fine).

## Steps

1. Build the kernel with `load_inline` (no setup.py needed yet):

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <ATen/cuda/CUDAContext.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // TODO 1: global index + bounds guard
    // TODO 2 (after it works): rewrite as a grid-stride loop
}
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dtype() == torch::kFloat32);
    TORCH_CHECK(a.sizes() == b.sizes());
    auto c = torch::empty_like(a);
    int n = a.numel(), block = 256, grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
"""
ext = load_inline("my_add", cpp_sources="torch::Tensor add(torch::Tensor a, torch::Tensor b);",
                  cuda_sources=cuda_src, functions=["add"],
                  extra_include_paths=[], extra_cuda_cflags=["-lineinfo"])
```

2. Correctness: `torch.testing.assert_close(ext.add(a, b), a + b)` for n in
   `[1, 31, 32, 33, 1_000_003, 50_000_000]` (odd sizes catch a missing guard).
3. Timing helper (use it for every exercise in this phase):

```python
def bench(fn, iters=200, warmup=20):
    for _ in range(warmup): fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))          # ms
    times.sort(); return times[len(times)//2]     # median
```

4. Measure, for n = 10K, 1M, 50M: your kernel, `torch.add`, and
   `dst.copy_(src)` on a tensor of `n` floats (the copy ceiling).
5. Convert to bandwidth: add moves `12·n` bytes, copy moves `8·n` bytes.

## Report (`REPORT.md`)

| n | your add (GB/s) | torch.add (GB/s) | copy (GB/s) | your add / copy |
|---|---|---|---|---|

## Pass criteria

- All correctness cases pass, including n = 1 and 33.
- At n = 50M your kernel reaches **≥ 80% of copy bandwidth** and is within 10% of `torch.add`.
- One paragraph: why is n = 10K far below the ceiling? (Hint: launch overhead,
  Mod 1 Lecture 1.1, and how many waves the grid has.)

## Common bugs
`idx <= n` instead of `<` · forgetting `.contiguous()` · timing without warmup ·
launching on the default stream instead of PyTorch's current stream.
