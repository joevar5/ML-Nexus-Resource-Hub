# Lesson 02: CUDA Programming Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the CUDA programming model and execution hierarchy
2. Explain thread organization (grids, blocks, warps, threads)
3. Manage memory transfers between host (CPU) and device (GPU)
4. Write basic CUDA kernels for simple operations
5. Understand how PyTorch leverages CUDA under the hood
6. Debug common CUDA programming issues

## Introduction

While frameworks like PyTorch abstract away most CUDA programming, understanding CUDA fundamentals helps you:
- Write custom GPU operations when needed
- Optimize PyTorch code
- Debug GPU-related issues
- Understand performance characteristics

## CUDA Programming Model

### Host vs Device

```
┌─────────────────────┐         ┌─────────────────────┐
│      Host (CPU)     │  PCIe   │    Device (GPU)     │
│                     │◄───────►│                     │
│  - Program control  │         │  - Parallel compute │
│  - Serial code      │         │  - Kernel execution │
│  - CPU memory (RAM) │         │  - GPU memory (VRAM)│
└─────────────────────┘         └─────────────────────┘

Host:   Runs main program, launches kernels
Device: Executes kernels in parallel
```

### CUDA C++ Basics

```cpp
// CUDA program structure
#include <cuda_runtime.h>

// Kernel definition (runs on GPU)
__global__ void myKernel(float* data, int n) {
    // GPU code here
}

int main() {
    // Host code (CPU)

    // 1. Allocate GPU memory
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // 2. Copy data to GPU
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch kernel
    myKernel<<<blocks, threads>>>(d_data, n);

    // 4. Copy results back
    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free GPU memory
    cudaFree(d_data);

    return 0;
}
```

### CUDA Function Qualifiers

```cpp
// __global__: Kernel function (called from host, runs on device)
__global__ void kernelFunc() {
    // Executed on GPU
    // Called from CPU
}

// __device__: Device function (called from device, runs on device)
__device__ float deviceFunc(float x) {
    // Helper function executed on GPU
    // Called from GPU code only
    return x * x;
}

// __host__: Host function (default for regular C++ functions)
__host__ void hostFunc() {
    // Executed on CPU
}

// Can combine __host__ and __device__
__host__ __device__ float bothFunc(float x) {
    // Compiled for both CPU and GPU
    return x + 1.0f;
}
```

## Thread Hierarchy

CUDA organizes threads in a 3-level hierarchy:

```
┌─────────────────────────────────────────────────────┐
│                       Grid                          │
│  ┌────────────────┐  ┌────────────────┐            │
│  │  Block (0,0)   │  │  Block (1,0)   │  ...       │
│  │ ┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐  │            │
│  │ │T0││T1││T2│  │  │ │T0││T1││T2│  │            │
│  │ └──┘└──┘└──┘  │  │ └──┘└──┘└──┘  │            │
│  │ ┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐  │            │
│  │ │T3││T4││T5│  │  │ │T3││T4││T5│  │            │
│  │ └──┘└──┘└──┘  │  │ └──┘└──┘└──┘  │            │
│  └────────────────┘  └────────────────┘            │
│  ┌────────────────┐  ┌────────────────┐            │
│  │  Block (0,1)   │  │  Block (1,1)   │  ...       │
│  │ ┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐  │            │
│  │ │T0││T1││T2│  │  │ │T0││T1││T2│  │            │
│  │ └──┘└──┘└──┘  │  │ └──┘└──┘└──┘  │            │
│  │ ┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐  │            │
│  │ │T3││T4││T5│  │  │ │T3││T4││T5│  │            │
│  │ └──┘└──┘└──┘  │  │ └──┘└──┘└──┘  │            │
│  └────────────────┘  └────────────────┘            │
└─────────────────────────────────────────────────────┘

Grid:   Collection of blocks
Block:  Collection of threads (up to 1024 threads)
Thread: Individual execution unit
```

### Built-in Variables

CUDA provides built-in variables to identify threads:

```cpp
__global__ void kernel() {
    // Thread indices within block
    int tx = threadIdx.x;  // Thread X index (0 to blockDim.x-1)
    int ty = threadIdx.y;  // Thread Y index
    int tz = threadIdx.z;  // Thread Z index

    // Block indices within grid
    int bx = blockIdx.x;   // Block X index
    int by = blockIdx.y;   // Block Y index
    int bz = blockIdx.z;   // Block Z index

    // Block dimensions
    int bdx = blockDim.x;  // Block size in X dimension
    int bdy = blockDim.y;  // Block size in Y dimension
    int bdz = blockDim.z;  // Block size in Z dimension

    // Grid dimensions
    int gdx = gridDim.x;   // Grid size in X dimension
    int gdy = gridDim.y;   // Grid size in Y dimension
    int gdz = gridDim.z;   // Grid size in Z dimension

    // Calculate global thread ID
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
}
```

### Example: 1D Thread Indexing

```cpp
// Add two arrays: c[i] = a[i] + b[i]
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Launch configuration
int n = 1000000;  // 1 million elements
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

// Launch kernel
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

/*
Execution breakdown:
- blocksPerGrid = (1,000,000 + 255) / 256 = 3907 blocks
- Each block has 256 threads
- Total threads = 3907 * 256 = 1,000,192 threads
- Thread 0 processes element 0
- Thread 1 processes element 1
- ...
- Thread 999,999 processes element 999,999
- Threads 1,000,000 to 1,000,191 do nothing (i >= n check)
*/
```

### Example: 2D Thread Indexing

```cpp
// Matrix addition: C = A + B
__global__ void matrixAdd(float* A, float* B, float* C, int width, int height) {
    // Calculate 2D indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate 1D index (row-major order)
    int idx = row * width + col;

    // Check bounds
    if (row < height && col < width) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch configuration (2D)
dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
dim3 blocksPerGrid(
    (width + 15) / 16,   // X dimension
    (height + 15) / 16   // Y dimension
);

// Launch kernel
matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
```

## Warps and SIMT Execution

### What is a Warp?

A **warp** is a group of 32 threads that execute together:

```
Block (256 threads) = 8 Warps
┌────────────────────────────────┐
│ Warp 0:  Threads 0-31          │ Execute together
├────────────────────────────────┤
│ Warp 1:  Threads 32-63         │ Execute together
├────────────────────────────────┤
│ Warp 2:  Threads 64-95         │ Execute together
├────────────────────────────────┤
│ ...                            │
├────────────────────────────────┤
│ Warp 7:  Threads 224-255       │ Execute together
└────────────────────────────────┘

All threads in a warp execute the same instruction
at the same time (SIMT - Single Instruction, Multiple Threads)
```

### Warp Divergence

```cpp
__global__ void divergenceExample(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // WARP DIVERGENCE: Some threads in warp take if, some take else
        if (data[i] > 0) {
            data[i] = data[i] * 2;      // Some threads execute this
        } else {
            data[i] = data[i] * 3;      // Other threads execute this
        }
    }
}

/*
When warp divergence occurs:
1. Threads that take 'if' branch execute, others wait
2. Then threads that take 'else' branch execute, others wait
3. Execution time = sum of both branches
4. Performance penalty!

Best practice: Minimize divergence within warps
*/
```

### Avoiding Warp Divergence

```cpp
// GOOD: All threads in warp take same path
__global__ void noDivergence(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // Process data uniformly
        data[i] = data[i] * 2;
    }
}

// GOOD: Divergence happens across warps, not within
__global__ void goodDivergence(int* data, int n, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Entire warps take same branch (if threshold aligned with warp size)
    if (i < threshold) {
        data[i] *= 2;
    } else if (i < n) {
        data[i] *= 3;
    }
}
```

## Memory Management

### Memory Types

```cpp
// Global Memory (VRAM) - largest, slowest
__global__ void kernel(float* globalMem) {
    // Accessible by all threads
    // Persists across kernel launches
    // ~400 cycle latency
    float value = globalMem[0];
}

// Shared Memory - fast, limited size
__global__ void sharedMemKernel() {
    // Declare shared memory (per-block)
    __shared__ float sharedMem[256];

    // Accessible by all threads in block
    // ~10 cycle latency
    // Limited to ~48KB per block
    sharedMem[threadIdx.x] = threadIdx.x;
    __syncthreads();  // Synchronize block threads
}

// Registers - fastest, very limited
__global__ void registerKernel() {
    // Local variables stored in registers
    // Per-thread, private
    // ~1 cycle latency
    float reg1 = 1.0f;
    float reg2 = 2.0f;
}

// Constant Memory - read-only, cached
__constant__ float constMem[256];

__global__ void constKernel() {
    // Fast reads, cached
    // Must be read-only
    float value = constMem[0];
}
```

### Host-Device Memory Transfers

```cpp
#include <cuda_runtime.h>

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    // 1. Allocate host (CPU) memory
    float* h_data = (float*)malloc(bytes);

    // 2. Allocate device (GPU) memory
    float* d_data;
    cudaMalloc(&d_data, bytes);

    // 3. Initialize host data
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    // 4. Copy host → device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // 5. Launch kernel
    // kernel<<<blocks, threads>>>(d_data);

    // 6. Copy device → host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // 7. Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

### Pinned Memory (Page-Locked)

```cpp
// Regular pageable memory (slow transfers)
float* h_data = (float*)malloc(bytes);

// Pinned (page-locked) memory (2-3x faster transfers)
float* h_pinned;
cudaMallocHost(&h_pinned, bytes);  // or cudaHostAlloc()

// Transfer is faster because:
// - Memory cannot be paged out
// - DMA (Direct Memory Access) can be used
// - Asynchronous transfers possible

// Copy with pinned memory
cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);

// Free pinned memory
cudaFreeHost(h_pinned);

/*
Trade-off:
+ Faster transfers
- Limited resource (pins physical RAM)
- Can impact system performance if overused
*/
```

### Unified Memory

```cpp
// Unified Memory: Single pointer for CPU and GPU
float* unified_data;
cudaMallocManaged(&unified_data, bytes);

// Can access from CPU
for (int i = 0; i < n; i++) {
    unified_data[i] = i;
}

// Can access from GPU
kernel<<<blocks, threads>>>(unified_data);

// Can access from CPU again (automatic migration!)
cudaDeviceSynchronize();  // Wait for GPU
float result = unified_data[0];

// Free
cudaFree(unified_data);

/*
Benefits:
+ Simplified code (no explicit transfers)
+ Automatic migration
+ Page faulting handles oversubscription

Drawbacks:
- Performance overhead (page faults)
- Less control over transfers
- Best for prototyping, not production
*/
```

## Writing CUDA Kernels

### Example 1: Vector Addition

```cpp
// Kernel: Add two vectors
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code
void runVectorAdd(int n) {
    size_t bytes = n * sizeof(float);

    // Allocate and initialize host memory
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.2f\n", i, h_c[i]);
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}
```

### Example 2: Matrix Multiplication (Naive)

```cpp
// Naive matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matMulNaive(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Dot product of row and column
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Launch
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid(
    (N + 15) / 16,
    (M + 15) / 16
);
matMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
```

### Example 3: Optimized with Shared Memory

```cpp
// Tiled matrix multiplication using shared memory
__global__ void matMulShared(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    // Tile size = blockDim.x = blockDim.y
    const int TILE_SIZE = 16;

    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // Wait for all threads to load

        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // Wait before loading next tile
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/*
Performance improvement:
- Naive: Each thread reads from global memory K times
- Shared: Each thread reads from global memory ~K/TILE_SIZE times,
          rest from fast shared memory
- Speedup: ~10-15x for large matrices!
*/
```

## Error Checking

Always check for CUDA errors:

```cpp
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

// Check kernel launch
kernel<<<blocks, threads>>>(d_data);
CUDA_CHECK(cudaGetLastError());  // Check for launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Wait and check for execution errors
```

## CUDA in PyTorch

### How PyTorch Uses CUDA

```python
import torch

# PyTorch provides high-level CUDA interface
x = torch.randn(1000, 1000)

# Move to GPU
x_gpu = x.cuda()  # or x.to('cuda')

# PyTorch automatically:
# 1. Allocates GPU memory
# 2. Copies data to GPU
# 3. Uses optimized CUDA kernels for operations
# 4. Manages memory lifecycle

# Operations run on GPU
y = x_gpu + 1.0
z = torch.mm(x_gpu, x_gpu)  # Matrix multiplication

# Check where tensor lives
print(x_gpu.device)  # cuda:0
print(x_gpu.is_cuda)  # True
```

### Custom CUDA Kernels in PyTorch

You can write custom CUDA operations:

```python
from torch.utils.cpp_extension import load_inline

# CUDA kernel code
cuda_source = '''
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}
'''

cpp_source = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

# Compile and load
module = load_inline(
    name='custom_add',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['add_cuda'],
    verbose=True
)

# Use custom kernel
a = torch.randn(1000).cuda()
b = torch.randn(1000).cuda()
c = module.add_cuda(a, b)
```

## Common CUDA Patterns

### Reduction (Sum)

```cpp
__global__ void reduceSum(float* input, float* output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Scan (Prefix Sum)

```cpp
__global__ void scanInclusive(float* input, float* output, int n) {
    __shared__ float temp[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input
    temp[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Up-sweep (reduce)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    // Write output
    if (i < n) {
        output[i] = temp[tid];
    }
}
```

## Debugging CUDA Code

### Using printf in Kernels

```cpp
__global__ void debugKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 10) {  // Limit output
        printf("Thread %d: data[%d] = %.2f\n", i, i, data[i]);
    }

    if (i < n) {
        data[i] *= 2.0f;
    }
}
```

### CUDA-MEMCHECK

```bash
# Check for memory errors
cuda-memcheck ./my_program

# Common errors detected:
# - Out of bounds access
# - Uninitialized memory
# - Race conditions
# - Memory leaks
```

### Compute-Sanitizer (newer)

```bash
# Memory checker
compute-sanitizer --tool memcheck ./program

# Race condition detector
compute-sanitizer --tool racecheck ./program

# Initialization checker
compute-sanitizer --tool initcheck ./program
```

## Performance Tips

1. **Minimize Host-Device Transfers**
   - Transfers are slow (PCIe bottleneck)
   - Keep data on GPU when possible

2. **Coalesce Memory Access**
   - Adjacent threads should access adjacent memory
   - Allows efficient memory transactions

3. **Avoid Warp Divergence**
   - Keep threads in same warp on same code path
   - Arrange data to minimize branching

4. **Use Shared Memory**
   - 10-100x faster than global memory
   - Reduce global memory accesses

5. **Occupancy**
   - Keep GPU busy with enough blocks/threads
   - Balance registers, shared memory, and occupancy

6. **Asynchronous Execution**
   - Overlap transfers with computation
   - Use CUDA streams

## Practical Exercises

### Exercise 1: Vector Scaling

Implement a kernel that scales a vector by a constant:

```cpp
// TODO: Implement kernel
__global__ void vectorScale(float* data, float scale, int n) {
    // Your code here
    // Multiply each element by scale
}

// Test
int n = 1000000;
float scale = 2.5f;

// TODO: Allocate memory
// TODO: Initialize data
// TODO: Copy to device
// TODO: Launch kernel
vectorScale<<<blocks, threads>>>(d_data, scale, n);
// TODO: Copy back and verify
```

### Exercise 2: Matrix Transpose

```cpp
// TODO: Implement matrix transpose
__global__ void matrixTranspose(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // Your code here
    // output[col * rows + row] = input[row * cols + col]
}
```

### Exercise 3: Dot Product

```cpp
// TODO: Implement dot product with reduction
__global__ void dotProduct(
    const float* a,
    const float* b,
    float* partial_sums,
    int n
) {
    // Your code here
    // Use shared memory for reduction
}
```

## Summary

In this lesson, you learned:

1. **CUDA Programming Model**: Host-device execution, kernel launches
2. **Thread Hierarchy**: Grids, blocks, warps, threads
3. **Memory Management**: Host-device transfers, memory types
4. **Writing Kernels**: Basic kernel patterns and optimizations
5. **CUDA in PyTorch**: How frameworks leverage CUDA
6. **Debugging**: Tools and techniques for CUDA development

## Key Takeaways

- **Warps execute together** - avoid divergence within warps
- **Memory hierarchy matters** - use shared memory for performance
- **Coalesced access is critical** - ensure adjacent threads access adjacent memory
- **PyTorch abstracts CUDA** - but understanding helps optimization
- **Always check errors** - CUDA errors can be silent

## Further Reading

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Professional CUDA C Programming](https://www.oreilly.com/library/view/professional-cuda-c/9781118739310/)

## Next Steps

In the next lesson, **PyTorch GPU Acceleration**, we'll explore how to effectively use GPUs in PyTorch for training and inference, including mixed precision training and memory optimization.

---

**Ready to accelerate PyTorch models? Let's dive into GPU optimization!**
