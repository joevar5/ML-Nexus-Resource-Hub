# Lecture 01: Kernels + Launch Configuration

## The shape of a CUDA kernel

```cpp
__global__ void add(int n, const float* a, const float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Host launch:
int block = 256;
int grid = (n + block - 1) / block;
add<<<grid, block>>>(n, a_d, b_d, c_d);
```

## Launch geometry concepts

- **Thread**: smallest unit; runs one instance of the kernel
- **Block**: group of threads (up to 1024); can use shared memory together
- **Grid**: collection of blocks; the full kernel launch

Choose `block` to be a multiple of warp size (32) to avoid wasted threads.

## Picking block size

| Workload | Typical block |
|---|---|
| Memory-bound, simple | 128, 256 |
| Compute-bound, register-heavy | 64, 128 |
| Reduction / cooperative | 256 (full warps) |

Tune empirically. The default starting point is 256.

## Occupancy

Threads / SM is bounded by:
- 32-bit registers per SM (registers per thread × threads)
- Shared memory per SM (per-block × blocks)
- Max threads per block (typically 1024)
- Max blocks per SM

Use `cudaOccupancyMaxPotentialBlockSize` to get a starting point. Then
benchmark variations.

## Common mistakes

- **Block size not a multiple of warp size** — a 48-thread block still occupies 2 warps, with 16 lanes of the second idle (25% waste)
- **Branch divergence within a warp** — `if (threadIdx.x < 16)` splits work
- **Bank conflicts in shared memory** — threads in the same warp accessing
  the same memory bank serialize
- **Insufficient grid size** — grid < SM count leaves SMs idle

## Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)

---

## Deep dive: how it works (core logic)

### From source to silicon ([NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/))
`nvcc` splits your file: host code goes to the normal C++ compiler; device code is compiled to **PTX** (a virtual, stable instruction set) and then to **SASS** (the real machine code for one architecture, e.g. `sm_90`). The binary can embed PTX so newer GPUs can JIT-compile it. Why care: inspecting the output tells you if the compiler did what you expect.
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/): the layer between CUDA and machine code.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/): `cuobjdump -sass a.out` shows the real instructions; `nvdisasm` disassembles.
- Compile with `-lineinfo` so profilers map SASS back to your source lines.

### Choosing a launch configuration
- Block size: a **multiple of 32**, usually 128–256. Too small wastes warp slots; very large limits how many blocks fit per SM.
- Grid size: enough blocks to cover the data *and* to fill all SMs several times over (hundreds+ for a big GPU).
- For arbitrary N use a **grid-stride loop**, which decouples launch size from data size:
```cuda
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
    c[i] = a[i] + b[i];
```
- Always check errors: `cudaGetLastError()` after the launch and `cudaDeviceSynchronize()` while debugging. Launches are asynchronous, so errors show up later than you expect.

### Correctness tools ([Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/))
`compute-sanitizer --tool memcheck|racecheck|initcheck|synccheck ./app` finds out-of-bounds accesses, shared-memory races, uninitialized reads and bad `__syncthreads()`. Run it before you optimize; fast wrong kernels are worse than slow right ones. Rules of thumb for tuning live in [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

### Read the originals

- [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
