# Lecture 03: PyTorch C++ Extensions

## Why

You wrote a CUDA kernel. Calling it from Python without ceremony is the goal.

PyTorch's `torch.utils.cpp_extension` builds C++ + CUDA sources into a Python
module that:
- Accepts `torch.Tensor` arguments
- Returns `torch.Tensor` outputs
- Plays well with autograd if you wire backward

## The wrapper

```cpp
#include <torch/extension.h>

extern "C" void launch_add(int n, const float* a, const float* b, float* c);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto c = torch::empty_like(a);
    launch_add(a.numel(), a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Custom add (CUDA)");
}
```

## setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[CUDAExtension("my_ops", ["my_ops.cpp", "add_kernel.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
```

## Build + use

```bash
pip install -e .
python -c "import my_ops, torch; print(my_ops.add(torch.ones(10).cuda(), torch.ones(10).cuda()))"
```

## Benchmarking vs torch.add

Use `torch.cuda.synchronize()` between launches. Run inside a warm loop;
report median + p95. Don't trust single-run numbers — kernel JIT, cuBLAS
init, and CUDA graph capture all skew the first call.

## Companion

[engineer-solutions/mod-107 ex-02](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel) — working setup.py + bench.py.

---

## Deep dive: how it works (core logic)

### Anatomy of a PyTorch CUDA extension
1. **Kernel** (`.cu`): the `__global__` function.
2. **C++ wrapper:** takes `torch::Tensor`s, validates, launches:
```cpp
torch::Tensor my_op(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "need contiguous CUDA tensor");
  auto y = torch::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();   // launch on PyTorch's stream
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "my_op", [&] {
      kernel<scalar_t><<<G, B, 0, stream>>>(x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), x.numel());
  });
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("my_op", &my_op); }
```
3. **Build:** `torch.utils.cpp_extension.load_inline` / `load` (JIT) or a `setup.py` with `CUDAExtension`.
Three things people forget: launch on the **current stream** (or you break ordering), check dtype/contiguity, and make the output the right shape/dtype.

### Why correctness testing needs care ([Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/))
Floating-point addition is **not associative**: (a+b)+c ≠ a+(b+c) in the last bits. The result of a reduction depends on the *order* of additions. Kernels often change that order with batch size (e.g. split-K, different tile sizes, atomics). So the same prompt can give different logits depending on what else is in the batch. The fix is **batch-invariant kernels**: fix the reduction strategy independent of batch size (for RMSNorm, matmul and attention), at a small speed cost. Lesson for your extensions: compare against a reference with `allclose` and tolerances, and know your reduction order.

### Often you don't need CUDA
The [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) build fused softmax, matmul, fused attention in ~50 lines of Python each and get close to hand-written speed. Reach for raw CUDA only when Triton can't express it. Benchmark properly with [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html) (warm-up, fixed clocks, many iterations, median), and run [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/).

### Read the originals

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
