# Lecture 02: Memory Coalescing + Shared Memory

## Coalescing

Threads in a warp access memory simultaneously. If consecutive threads access
consecutive memory addresses (aligned), the GPU coalesces them into one
128-byte transaction. Otherwise, each thread takes a separate transaction.

```cpp
// Coalesced (good): each thread reads its own element from a row
c[blockIdx.x * blockDim.x + threadIdx.x] = ...

// Strided (bad): thread i reads element i * 32 — separate transactions
c[threadIdx.x * 32] = ...
```

## Shared memory

Shared memory is a per-block on-chip scratchpad: ~20–30 cycles latency vs
~400–800 for global memory, and much higher aggregate bandwidth.
Use it for data reused within a block.

### Tiled matmul (classic example)

Naive matmul reads each A and B element multiple times from global memory.
Tile-based matmul loads tiles into shared memory once + each block reuses
them many times.

```cpp
#define TS 32   // tile size; TS*TS threads per block, so TS <= 32
// Assumes N is a multiple of TS; add bounds checks for general N.
__global__ void matmul_tiled(int N, const float* A, const float* B, float* C) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N / TS; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TS + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TS + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TS; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

On its own this reaches only ~10–15% of cuBLAS FP32 (12.8% in Boehm's
worklog, see the deep dive below): each thread still does one FMA per two
shared-memory loads. The big jumps come next, from **register tiling**
(each thread computes an 8×8 patch of C, ~69% of cuBLAS), then vectorized
loads, autotuning and warp tiling (~94%). All of that is hand-writable;
tensor cores (TF32/BF16) are a separate, larger step.

## Bank conflicts

Shared memory is divided into 32 banks. If multiple threads in a warp access
the same bank simultaneously, accesses serialize.

Fix by padding:
```cpp
__shared__ float As[TS][TS + 1];   // pad to avoid bank conflicts
```

## Reading

- [GPU Memory Bandwidth (Mark Harris)](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)

---

## Deep dive: how it works (core logic)

### The matmul optimization ladder ([How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM))
The task: C = A·B. Each step removes one bottleneck, and each is ~2–10× faster than the last:

1. **Naive:** one thread per C element, loops over K reading A row and B column straight from global memory. Neighbouring threads read non-adjacent B elements → uncoalesced.
2. **Coalesced:** reassign indices so a warp reads consecutive addresses.
3. **Shared-memory tiling:** each block loads a tile of A and a tile of B into shared memory, then computes from there:
```
for t in range(K/T):
    sA[ty][tx] = A[row][t*T + tx]      # coalesced load
    sB[ty][tx] = B[t*T + ty][col]
    __syncthreads()
    for k in range(T): acc += sA[ty][k] * sB[k][tx]
    __syncthreads()
C[row][col] = acc
```
Each loaded element is reused T times → global traffic ÷ T.
4. **Register tiling (1D/2D blocktiling):** each thread computes a small TM×TN block of C. Per step it loads TM values of A and TN values of B into registers and does TM×TN multiply-adds (an outer product). FLOPs per shared-memory load rise from 1 to ~TM·TN/(TM+TN).
5. **Vectorized loads, warp-level tiling, double buffering:** `float4` loads, a warp works on its own sub-tile, and the next tile loads while the current computes. This lands close to cuBLAS.

### Drills that build the instincts
- **Reduction ([Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)):** contiguous active threads, warp shuffles. (See Mod 1 Lesson 04.)
- **Prefix scan ([Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)):** a naive scan reads data multiple times. *Decoupled look-back* has each block publish its local sum plus a status flag; later blocks look back at predecessors' flags to compute their offset, all in **one pass** over memory. Idea to remember: use flags in global memory to chain blocks without a second kernel.
- **Transpose ([Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)):** coalesce both sides through a padded shared tile.

### Going further
[Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) connects layouts, tiling, PTX and the roofline; [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) shows the Hopper-era steps (tensor cores, async movement).

### Read the originals

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)
- [Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul)
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
