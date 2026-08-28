# Lesson 01: Introduction to GPU Computing for AI

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain the fundamental differences between GPU and CPU architectures
2. Understand why GPUs are well-suited for machine learning workloads
3. Identify when to use GPUs vs CPUs for different ML tasks
4. Describe the GPU memory hierarchy and its impact on performance
5. Understand CUDA cores, tensor cores, and their roles in ML acceleration

## Introduction

Graphics Processing Units (GPUs) have become the backbone of modern AI infrastructure. Originally designed for rendering graphics, GPUs have proven exceptionally well-suited for the parallel computations required in machine learning. Understanding GPU architecture and when to leverage GPUs is critical for any AI infrastructure engineer.

## CPU vs GPU Architecture

### CPU Architecture

CPUs are designed for **sequential processing** and **low-latency operations**:

```
┌─────────────────────────────────────┐
│         CPU Architecture            │
│                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │Core1│ │Core2│ │Core3│ │Core4│  │
│  │     │ │     │ │     │ │     │  │
│  └─────┘ └─────┘ └─────┘ └─────┘  │
│                                     │
│  Large L1/L2/L3 Cache              │
│  Complex Control Units              │
│  Out-of-order Execution            │
│  Branch Prediction                  │
└─────────────────────────────────────┘

Characteristics:
- 4-64 powerful cores (typically)
- High clock speeds (3-5 GHz)
- Large cache hierarchy
- Complex instruction sets
- Optimized for single-thread performance
```

### GPU Architecture

GPUs are designed for **massive parallelism** and **high throughput**:

```
┌──────────────────────────────────────────────────┐
│            GPU Architecture (Simplified)         │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │   SM 1   │ │   SM 2   │ │  SM ...  │        │
│  │┌────────┐│ │┌────────┐│ │┌────────┐│        │
│  ││ 64-128 ││ ││ 64-128 ││ ││ 64-128 ││        │
│  ││  CUDA  ││ ││  CUDA  ││ ││  CUDA  ││        │
│  ││  Cores ││ ││  Cores ││ ││  Cores ││        │
│  │└────────┘│ │└────────┘│ │└────────┘│        │
│  │┌────────┐│ │┌────────┐│ │┌────────┐│        │
│  ││ Tensor ││ ││ Tensor ││ ││ Tensor ││        │
│  ││  Cores ││ ││  Cores ││ ││  Cores ││        │
│  │└────────┘│ │└────────┘│ │└────────┘│        │
│  │ Shared   │ │ Shared   │ │ Shared   │        │
│  │ Memory   │ │ Memory   │ │ Memory   │        │
│  └──────────┘ └──────────┘ └──────────┘        │
│                                                  │
│         Global Memory (VRAM) - 8-80GB           │
└──────────────────────────────────────────────────┘

Characteristics:
- Thousands of simpler cores (3,000-16,000+)
- Lower clock speeds (1-2 GHz)
- Smaller cache, larger VRAM
- SIMT (Single Instruction, Multiple Threads)
- Optimized for parallel throughput
```

### Key Architectural Differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Cores** | 4-64 powerful cores | 1,000s-10,000s simpler cores |
| **Design Goal** | Minimize latency | Maximize throughput |
| **Cache** | Large (MB per core) | Small (KB per SM) |
| **Memory** | System RAM (32-512GB) | VRAM (8-80GB) |
| **Clock Speed** | 3-5 GHz | 1-2 GHz |
| **Best For** | Serial tasks, complex logic | Parallel tasks, simple operations |
| **Parallelism** | Task-level | Data-level |

## Why GPUs Excel at Machine Learning

### 1. Matrix Operations

Machine learning is fundamentally about matrix operations:

```python
# Neural network forward pass (simplified)
def forward_pass(input, weights, bias):
    # Matrix multiplication - highly parallelizable
    output = input @ weights + bias
    return activation(output)

# Example: 1000x1000 matrix multiplication
# CPU: Processes row-by-row or column-by-column
# GPU: Computes thousands of elements simultaneously
```

**GPU Advantage**: Can compute thousands of matrix elements in parallel.

### 2. SIMT Execution Model

GPUs use **Single Instruction, Multiple Threads** (SIMT):

```python
# Adding two arrays element-wise
a = [1, 2, 3, 4, 5, 6, 7, 8, ...]  # Million elements
b = [2, 3, 4, 5, 6, 7, 8, 9, ...]  # Million elements

# CPU approach (simplified)
for i in range(len(a)):
    c[i] = a[i] + b[i]  # Sequential

# GPU approach
# Launch thousands of threads, each handling one (or few) elements
# All threads execute: c[thread_id] = a[thread_id] + b[thread_id]
# Parallel execution!
```

### 3. High Memory Bandwidth

```
Memory Bandwidth Comparison:

CPU (DDR4/DDR5):     50-100 GB/s
GPU (GDDR6/HBM2):    600-2,000 GB/s

For ML workloads that are memory-bound, this 10-20x
bandwidth advantage significantly accelerates training.
```

### 4. Specialized Hardware

Modern GPUs include specialized units for ML:

- **Tensor Cores**: Accelerate matrix multiply-accumulate operations
- **RT Cores**: Ray tracing (less relevant for ML)
- **NVLink**: High-speed GPU-to-GPU communication

## NVIDIA GPU Architecture

### Streaming Multiprocessors (SMs)

The fundamental building block of NVIDIA GPUs:

```
Streaming Multiprocessor (SM)
┌────────────────────────────────────┐
│                                    │
│  CUDA Cores (64-128 per SM)       │
│  ┌───┐┌───┐┌───┐┌───┐...          │
│  │FP ││FP ││INT││INT│...          │
│  │32 ││32 ││32 ││32 │             │
│  └───┘└───┘└───┘└───┘             │
│                                    │
│  Tensor Cores (4-8 per SM)        │
│  ┌──────────┐┌──────────┐         │
│  │  TC 1    ││  TC 2    │         │
│  └──────────┘└──────────┘         │
│                                    │
│  Special Function Units (SFUs)     │
│  Load/Store Units (LD/ST)         │
│                                    │
│  Shared Memory / L1 Cache          │
│  (64-128 KB)                       │
│                                    │
│  Registers (64K 32-bit registers)  │
└────────────────────────────────────┘
```

### CUDA Cores

**CUDA Cores** are the basic processing units:

- Execute floating-point and integer operations
- Operate on single values (scalars)
- Thousands per GPU
- Work in parallel within warps (groups of 32 threads)

```python
# Example: Each CUDA core processes one element
import torch

# Create tensors
a = torch.randn(1000000).cuda()  # Million elements
b = torch.randn(1000000).cuda()

# Element-wise operations utilize CUDA cores
c = a + b        # Addition
d = a * b        # Multiplication
e = torch.relu(a)  # Activation
```

### Tensor Cores

**Tensor Cores** accelerate matrix operations (introduced in Volta architecture):

- Perform mixed-precision matrix multiply-accumulate (MMA) operations
- Much faster than CUDA cores for certain operations
- Critical for modern deep learning

```python
# Matrix multiplication benefits from Tensor Cores
# D = A × B + C

A = torch.randn(1024, 1024, dtype=torch.float16).cuda()
B = torch.randn(1024, 1024, dtype=torch.float16).cuda()
C = torch.randn(1024, 1024, dtype=torch.float16).cuda()

# This operation can utilize Tensor Cores
D = torch.mm(A, B) + C

# Tensor Cores provide:
# - Up to 8x speedup for FP16 operations
# - Support for mixed precision (FP16/BF16 compute, FP32 accumulate)
```

**Performance Comparison**:

```
Operation: 1024x1024 Matrix Multiplication

CUDA Cores (FP32):  ~100 TFLOPS  (RTX 4090)
Tensor Cores (FP16): ~660 TFLOPS (RTX 4090)

6-7x speedup with Tensor Cores!
```

### GPU Memory Hierarchy

Understanding memory hierarchy is critical for performance:

```
┌──────────────────────────────────────┐
│         Registers                    │  Fastest
│         (per thread)                 │  ~1 cycle
│         ~256 KB per SM               │
├──────────────────────────────────────┤
│      Shared Memory / L1 Cache        │  Very Fast
│         (per SM)                     │  ~10 cycles
│         64-128 KB                    │
├──────────────────────────────────────┤
│         L2 Cache                     │  Fast
│         (GPU-wide)                   │  ~100 cycles
│         40-96 MB                     │
├──────────────────────────────────────┤
│      Global Memory (VRAM)            │  Slow
│         (GPU-wide)                   │  ~400 cycles
│         8-80 GB                      │  Slowest
└──────────────────────────────────────┘
```

**Memory Access Patterns**:

```python
# BAD: Uncoalesced memory access
for i in range(N):
    value = data[i * stride]  # Scattered access

# GOOD: Coalesced memory access
for i in range(N):
    value = data[i]  # Sequential access

# PyTorch handles this for you, but good to understand!
```

## GPU Generations and Compute Capability

NVIDIA GPUs are organized by architecture and compute capability:

### Recent Architectures

| Architecture | Year | Compute Capability | Notable GPUs | Key Features |
|--------------|------|-------------------|--------------|--------------|
| **Hopper** | 2022 | 9.0 | H100 | 4th gen Tensor Cores, Transformer Engine |
| **Ada Lovelace** | 2022 | 8.9 | RTX 4090 | 4th gen Tensor Cores, DLSS 3 |
| **Ampere** | 2020 | 8.0, 8.6 | A100, RTX 3090 | 3rd gen Tensor Cores, TF32 |
| **Turing** | 2018 | 7.5 | RTX 2080 Ti | 2nd gen Tensor Cores, RT Cores |
| **Volta** | 2017 | 7.0 | V100 | 1st gen Tensor Cores |
| **Pascal** | 2016 | 6.1 | GTX 1080 Ti | No Tensor Cores |

### Compute Capability Impact

```python
# Check GPU compute capability
import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    capability = torch.cuda.get_device_capability(device)
    print(f"Compute Capability: {capability[0]}.{capability[1]}")

    # Tensor Core support requires compute capability >= 7.0
    has_tensor_cores = capability[0] >= 7
    print(f"Tensor Cores: {'Yes' if has_tensor_cores else 'No'}")
```

**Why It Matters**:
- Older GPUs may not support certain features (e.g., Tensor Cores)
- Software compatibility (CUDA version requirements)
- Performance characteristics vary significantly

## When to Use GPUs vs CPUs

### Use GPUs When:

✅ **Training Deep Neural Networks**
- Lots of matrix operations
- Large batch sizes
- Convolutional or transformer architectures

```python
# Example: Training a neural network
model = LargeTransformer().cuda()  # Move to GPU
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    inputs, targets = batch
    inputs, targets = inputs.cuda(), targets.cuda()  # GPU

    outputs = model(inputs)  # GPU acceleration!
    loss = criterion(outputs, targets)
    loss.backward()  # GPU-accelerated backprop
    optimizer.step()
```

✅ **Inference at Scale**
- High throughput requirements
- Batch processing
- Real-time inference with batching

✅ **Large-Scale Data Processing**
- Image processing pipelines
- Video processing
- Large matrix operations

✅ **Parallel Hyperparameter Search**
- Multiple experiments in parallel
- Grid search / random search

### Use CPUs When:

❌ **Small Models**
- Overhead of GPU transfer exceeds compute benefits
- Simple models (e.g., linear regression on small datasets)

```python
# For small data, CPU might be faster!
small_data = torch.randn(100, 10)  # 100 samples, 10 features
model = torch.nn.Linear(10, 1)

# GPU overhead not worth it:
# - Data transfer to GPU
# - Kernel launch overhead
# - GPU underutilization
```

❌ **Single/Small Batch Inference**
- Low latency requirements
- Cannot batch requests
- GPU transfer overhead too high

❌ **Memory-Limited Scenarios**
- Model doesn't fit in GPU memory
- Data doesn't fit in GPU memory
- CPU RAM >> GPU VRAM (e.g., 256GB RAM vs 24GB VRAM)

❌ **Sequential Operations**
- Recurrent operations with dependencies
- Can't be parallelized effectively

### Decision Framework

```python
def should_use_gpu(model_size, batch_size, data_size, latency_requirement):
    """
    Simplified decision framework for GPU usage
    """
    # Rule 1: Model size
    if model_size > 100_000_000:  # 100M+ parameters
        return True  # Large models almost always benefit

    # Rule 2: Batch size
    if batch_size >= 32:
        return True  # Good GPU utilization

    # Rule 3: Training vs Inference
    if is_training:
        return True  # Training benefits from GPU

    # Rule 4: Latency requirements
    if latency_requirement < 10:  # ms
        # GPU might add overhead
        return batch_size > 8

    # Rule 5: Data size
    if data_size > 10_000:
        return True  # Amortize GPU overhead

    return False
```

## GPU Specifications to Consider

When selecting GPUs for ML workloads, consider:

### 1. Memory Capacity (VRAM)

```
Model Size Requirements (rough estimates):

Small Models (BERT-base):       4-8 GB
Medium Models (GPT-2):          8-16 GB
Large Models (LLaMA-7B):        16-24 GB
Very Large (LLaMA-70B):         80-160 GB (multi-GPU)

Rule of thumb: Need ~4x model size for training
- Model parameters
- Gradients
- Optimizer states (AdamW: 2x parameters)
- Activations
```

### 2. Memory Bandwidth

```
GPU Memory Bandwidth:

RTX 3090:      936 GB/s   (GDDR6X)
A100 (40GB):   1,555 GB/s (HBM2)
A100 (80GB):   2,039 GB/s (HBM2e)
H100:          3,350 GB/s (HBM3)

Higher bandwidth = faster data movement
Critical for large models and batch sizes
```

### 3. Compute Performance

```
Tensor TFLOPS (FP16 with Tensor Cores):

RTX 4090:     ~660 TFLOPS
A100:         ~312 TFLOPS
H100:         ~990 TFLOPS

Note: Raw TFLOPS isn't everything!
- Memory bandwidth often more important
- Software optimization matters
- Batch size affects utilization
```

### 4. GPU-to-GPU Communication

```
Multi-GPU Communication:

PCIe 4.0 x16:   ~32 GB/s (typical workstations)
NVLink 3.0:     ~600 GB/s (A100)
NVLink 4.0:     ~900 GB/s (H100)

NVLink provides 20-30x faster inter-GPU communication
Critical for multi-GPU training efficiency
```

### 5. Cost Considerations

```
Price/Performance (approximate, 2024):

Consumer GPUs:
- RTX 4090:     $1,600  |  Best performance/$ for single GPU
- RTX 4080:     $1,200  |  Good balance
- RTX 3090:     $1,000  |  Previous gen, still capable

Data Center GPUs:
- A100 (40GB):  $10,000 |  Production workloads
- A100 (80GB):  $15,000 |  Large models
- H100:         $30,000 |  Cutting edge

Cloud Options (per hour):
- AWS p3.2xlarge (V100):   $3.06/hr
- AWS p4d.24xlarge (A100): $32.77/hr
- GCP a2-highgpu-1g (A100): $3.67/hr
```

## Checking GPU Availability

### Using nvidia-smi

```bash
# Check GPU information
nvidia-smi

# Output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   32C    P0    56W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Monitor specific metrics
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

### Using PyTorch

```python
import torch

# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

# Get number of GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Get GPU names
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Get current GPU properties
if torch.cuda.is_available():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    print(f"\nGPU Properties:")
    print(f"  Name: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi Processors: {props.multi_processor_count}")
    print(f"  CUDA Cores: ~{props.multi_processor_count * 128}")  # Approximate
```

### Using nvidia-ml-py

```python
import nvidia_smi

nvidia_smi.nvmlInit()

# Get GPU count
device_count = nvidia_smi.nvmlDeviceGetCount()
print(f"Number of GPUs: {device_count}")

# Get detailed info for each GPU
for i in range(device_count):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

    # GPU name
    name = nvidia_smi.nvmlDeviceGetName(handle)

    # Memory info
    mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # Utilization
    util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)

    # Temperature
    temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)

    print(f"\nGPU {i}: {name}")
    print(f"  Memory: {mem_info.used / 1024**3:.2f} GB / {mem_info.total / 1024**3:.2f} GB")
    print(f"  GPU Utilization: {util.gpu}%")
    print(f"  Memory Utilization: {util.memory}%")
    print(f"  Temperature: {temp}°C")

nvidia_smi.nvmlShutdown()
```

## Common GPU Issues and Solutions

### Issue 1: CUDA Out of Memory (OOM)

```python
# Problem:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...

# Solutions:
# 1. Reduce batch size
batch_size = 16  # Instead of 32

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# 4. Clear cache
torch.cuda.empty_cache()

# 5. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue 2: GPU Underutilization

```python
# Check GPU utilization
nvidia-smi

# If GPU utilization is low (<50%):

# Solution 1: Increase batch size
batch_size = 64  # Instead of 16

# Solution 2: Optimize data loading
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2  # Prefetch batches
)

# Solution 3: Profile to find bottlenecks
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch in train_loader:
        outputs = model(batch)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Issue 3: Slow Data Transfer

```python
# Problem: CPU-GPU transfer is slow

# Solution 1: Use pinned memory
train_loader = DataLoader(
    dataset,
    pin_memory=True  # Faster transfers
)

# Solution 2: Move data asynchronously
for batch in train_loader:
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

# Solution 3: Keep data on GPU when possible
# (for small datasets that fit in memory)
full_data = dataset.tensors[0].cuda()
full_labels = dataset.tensors[1].cuda()
```

## Practical Exercise

### Exercise 1: GPU Benchmarking

Compare CPU vs GPU performance:

```python
import torch
import time

def benchmark_matmul(device, size=1024, iterations=100):
    """Benchmark matrix multiplication"""
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        C = torch.mm(A, B)
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU
    end = time.time()

    elapsed = end - start
    ops = 2 * size**3 * iterations  # FLOPs for matmul
    tflops = ops / elapsed / 1e12

    return elapsed, tflops

# TODO: Run benchmark on CPU
cpu_time, cpu_tflops = benchmark_matmul(torch.device('cpu'))
print(f"CPU: {cpu_time:.3f}s, {cpu_tflops:.2f} TFLOPS")

# TODO: Run benchmark on GPU
if torch.cuda.is_available():
    gpu_time, gpu_tflops = benchmark_matmul(torch.device('cuda'))
    print(f"GPU: {gpu_time:.3f}s, {gpu_tflops:.2f} TFLOPS")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("CUDA not available")
```

### Exercise 2: Memory Profiling

Monitor GPU memory usage:

```python
import torch

def profile_memory():
    """Profile GPU memory usage"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # TODO: Record initial memory
    initial = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial memory: {initial:.3f} GB")

    # TODO: Create large tensor
    size = 1000
    tensor = torch.randn(size, size, size, device='cuda')

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"After allocation:")
    print(f"  Allocated: {allocated:.3f} GB")
    print(f"  Reserved: {reserved:.3f} GB")

    # TODO: Delete tensor and clear cache
    del tensor
    torch.cuda.empty_cache()

    final = torch.cuda.memory_allocated() / 1024**3
    print(f"After cleanup: {final:.3f} GB")

profile_memory()
```

## Summary

In this lesson, you learned:

1. **GPU Architecture**: GPUs have thousands of simple cores optimized for parallel operations, unlike CPUs with fewer powerful cores
2. **Why GPUs for ML**: Matrix operations, SIMT execution, high memory bandwidth, and specialized hardware (Tensor Cores)
3. **CUDA Cores vs Tensor Cores**: CUDA cores for general compute, Tensor Cores for accelerated matrix operations
4. **Memory Hierarchy**: Registers → Shared Memory → L2 Cache → Global Memory (VRAM)
5. **When to Use GPUs**: Large models, high batch sizes, training workloads, parallel processing
6. **When to Use CPUs**: Small models, low latency inference, memory-limited scenarios
7. **GPU Selection**: Consider VRAM, memory bandwidth, compute performance, and cost

## Key Takeaways

- **GPUs are essential for modern ML**, providing 10-100x speedups for appropriate workloads
- **Not all tasks benefit from GPUs** - understand the workload characteristics
- **Memory is often the bottleneck** - VRAM capacity and bandwidth are critical
- **Tensor Cores provide massive speedups** for FP16/BF16 matrix operations
- **GPU utilization matters** - inefficient code wastes expensive resources

## Further Reading

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA GPU Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/resources/ai-inferencing-technical-overview/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Understanding GPU Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

## Next Steps

In the next lesson, we'll dive into **CUDA Programming Fundamentals**, where you'll learn to write GPU code and understand how PyTorch utilizes CUDA under the hood.

---

**Ready to write GPU code? Let's explore CUDA programming!**
