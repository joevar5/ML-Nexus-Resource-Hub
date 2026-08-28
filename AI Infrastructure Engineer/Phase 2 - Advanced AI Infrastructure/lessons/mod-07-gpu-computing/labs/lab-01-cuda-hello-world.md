# Lab 01: CUDA Hello-World and Device Introspection

**Duration:** 45 min  **Prerequisites:** NVIDIA GPU + CUDA Toolkit installed

## Objective
Verify your CUDA toolchain is healthy, write and compile a minimal kernel, and query device properties programmatically.

## Steps

### 1. Verify the toolchain
```bash
nvidia-smi                 # GPU detected
nvcc --version             # CUDA compiler installed
```

### 2. Minimal kernel: vector add
```cuda
// vector_add.cu
#include <cstdio>

__global__ void add(int n, const float *x, const float *y, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    float *x_h = (float*)malloc(bytes), *y_h = (float*)malloc(bytes), *o_h = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { x_h[i] = 1.0f; y_h[i] = 2.0f; }

    float *x, *y, *o;
    cudaMalloc(&x, bytes); cudaMalloc(&y, bytes); cudaMalloc(&o, bytes);
    cudaMemcpy(x, x_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_h, bytes, cudaMemcpyHostToDevice);

    int block = 256, grid = (N + block - 1) / block;
    add<<<grid, block>>>(N, x, y, o);
    cudaDeviceSynchronize();

    cudaMemcpy(o_h, o, bytes, cudaMemcpyDeviceToHost);
    printf("o[0]=%f o[N-1]=%f\n", o_h[0], o_h[N-1]);

    cudaFree(x); cudaFree(y); cudaFree(o);
    free(x_h); free(y_h); free(o_h);
    return 0;
}
```
```bash
nvcc -O2 -o vector_add vector_add.cu
./vector_add
# Expected: o[0]=3.000000 o[N-1]=3.000000
```

### 3. Device introspection
```cuda
// device_info.cu
#include <cstdio>
int main() {
    int n; cudaGetDeviceCount(&n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        printf("device %d: %s\n", i, p.name);
        printf("  compute cap: %d.%d\n", p.major, p.minor);
        printf("  global mem:  %.2f GB\n", p.totalGlobalMem / 1e9);
        printf("  SMs:         %d\n", p.multiProcessorCount);
        printf("  max threads/block: %d\n", p.maxThreadsPerBlock);
    }
}
```
```bash
nvcc -o device_info device_info.cu && ./device_info
```

### 4. Profile with nsys
```bash
nsys profile --stats=true ./vector_add
```
Look for kernel time vs memcpy time.

## Validation
- [ ] `nvidia-smi` shows your GPU.
- [ ] `vector_add` prints `3.0` for both first and last element.
- [ ] `device_info` lists at least one device.
- [ ] `nsys` stats show non-trivial CUDA API calls.

## Cleanup
```bash
rm vector_add device_info *.cu
```

## Troubleshooting
- **`nvcc: command not found`** — Install CUDA Toolkit; add `/usr/local/cuda/bin` to PATH.
- **`no CUDA-capable device is detected`** — Driver not loaded; `sudo modprobe nvidia` on Linux.
- **Kernel returns wrong values** — Check grid/block dim arithmetic; for N=1M and block=256, grid must be 4096.
