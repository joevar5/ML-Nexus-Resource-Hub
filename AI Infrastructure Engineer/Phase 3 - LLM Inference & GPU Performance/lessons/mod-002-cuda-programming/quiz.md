# Module 02: CUDA Programming — Quiz

15 questions. Pass mark: 12 / 15. Answers in [quiz-answers.md](quiz-answers.md).

### 1. A sensible default block size for a simple elementwise kernel is:
- a) 1,024, always
- b) A prime number, to spread work evenly
- c) A multiple of 32, usually 128 or 256
- d) Exactly the number of SMs

### 2. Coalesced global memory access means:
- a) All threads read the same address
- b) Consecutive threads in a warp access consecutive, aligned addresses
- c) Every access goes through atomics
- d) Reads go through texture memory

### 3. Compared with global memory (HBM), shared memory has roughly:
- a) The same latency, but larger capacity
- b) 2× lower latency
- c) 20× lower latency (~20–30 vs ~400–800 cycles) and much higher aggregate bandwidth
- d) 1,000× lower latency

### 4. A shared-memory bank conflict happens when:
- a) Several threads of one warp access different addresses in the same bank at the same time
- b) The block uses more than 48 KB of shared memory
- c) Two blocks on different SMs read the same global address
- d) All 32 threads read the same shared-memory word

### 5. Warp divergence means:
- a) Two warps of one block run on different SMs
- b) A warp's 32 threads finish at different times because of cache misses
- c) Blocks are scheduled out of order
- d) Threads of one warp take different branches, so the warp runs each path in turn with the other lanes masked off

### 6. Which of these limit how many warps can be resident on one SM? (pick all)
- a) Registers per thread
- b) Shared memory per block
- c) The maximum warps (threads) per SM
- d) The maximum blocks per SM

### 7. A shared-memory tiled matmul (one output per thread, no register tiling) typically reaches what share of cuBLAS FP32 throughput?
- a) ~10–15%
- b) ~95%
- c) More than 100%
- d) ~0.1%

### 8. What raises a hand-written SGEMM from ~13% to ~70% of cuBLAS?
- a) Increasing the block size to 1,024
- b) Register tiling: each thread computes a small TM×TN patch of C from values held in registers
- c) Removing `__syncthreads()`
- d) Switching from row-major to column-major output

### 9. PyTorch C++/CUDA extensions are usually built with:
- a) `gcc` directly
- b) `make` and a hand-written Makefile only
- c) `torch.utils.cpp_extension` (`CUDAExtension` + setuptools, or `load`/`load_inline` for JIT)
- d) `pip install cuda`

### 10. Why must a GPU benchmark call `torch.cuda.synchronize()` (or use CUDA events)?
- a) Kernel launches are asynchronous; without it you time the launch (~µs), not the kernel
- b) It speeds the kernel up
- c) It triggers JIT compilation
- d) setuptools requires it

### 11. In `kernel<<<A, B>>>(...)`, `A` is:
- a) Threads per block
- b) Shared memory bytes
- c) The grid: number of blocks
- d) The stream

### 12. `__shfl_down_sync(0xffffffff, v, 1)` lets a thread:
- a) Read `v` from the lane one above it in the same warp, without shared memory
- b) Atomically add `v` to global memory
- c) Synchronize all blocks in the grid
- d) Coalesce its next load

### 13. `cudaOccupancyMaxPotentialBlockSize` gives you:
- a) The guaranteed fastest block size
- b) A block size that maximizes theoretical occupancy — a starting point to benchmark from
- c) The number of SMs
- d) The warp size

### 14. Declaring `__shared__ float tile[32][33]` instead of `[32][32]`:
- a) Prevents register spills
- b) Doubles shared-memory bandwidth for row reads
- c) Shifts each row by one bank, so column reads by a warp hit 32 different banks
- d) Is required for `__syncthreads()` to work

### 15. Why is the first call to a CUDA op slower than later calls?
- a) GPUs always run slower when cold, permanently
- b) One-time costs: CUDA context creation, lazy module loading/JIT of PTX, library handle init (cuBLAS/cuDNN), caching-allocator growth
- c) Network latency to the driver
- d) The first call runs on the CPU
