# Lesson 04: Distributed Training Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the need for distributed training and when to use it
2. Explain data parallelism vs model parallelism strategies
3. Understand collective communication operations (AllReduce, Broadcast, etc.)
4. Set up PyTorch distributed training basics
5. Implement synchronous and asynchronous training
6. Calculate distributed training efficiency and identify bottlenecks

## Introduction

As models and datasets grow larger, single-GPU training becomes impractical. Distributed training allows us to:
- Train larger models that don't fit on a single GPU
- Speed up training by leveraging multiple GPUs/machines
- Process larger batch sizes
- Enable faster experimentation

## Why Distributed Training?

### The Scale Challenge

```
Model Size Growth:
┌─────────────────────────────────────────┐
│ GPT-2 (2019):     1.5B params  │  6 GB  │
│ GPT-3 (2020):     175B params  │ 700 GB │
│ PaLM (2022):      540B params  │ 2.2 TB │
│ GPT-4 (2023):     ~1.7T params │  ~7 TB │
└─────────────────────────────────────────┘

Single A100 GPU: 80 GB VRAM
→ Need distributed training!
```

### Training Time Reduction

```
Single GPU Training Time:
Model: BERT-Large on 1M samples
- 1x RTX 4090:  ~10 days
- 4x RTX 4090:  ~3 days   (3.3x speedup)
- 8x A100:      ~1 day    (10x speedup)
- 64x A100:     ~2 hours  (120x speedup)

Note: Speedup < N due to communication overhead
```

## Parallelism Strategies

### Data Parallelism

Split data across devices, replicate model:

```
┌──────────────────────────────────────────────────────┐
│              Data Parallelism                        │
│                                                      │
│  GPU 0:  Model Copy    GPU 1:  Model Copy           │
│          ┌────────┐             ┌────────┐          │
│          │ Model  │             │ Model  │          │
│          └────────┘             └────────┘          │
│             ↑ ↓                    ↑ ↓              │
│          Batch 1                Batch 2             │
│          [0-15]                 [16-31]             │
│                                                      │
│  After forward/backward:                            │
│  ┌────────────────────────────────────┐            │
│  │  Synchronize Gradients (AllReduce) │            │
│  └────────────────────────────────────┘            │
│                     ↓                               │
│  Both GPUs update with averaged gradients          │
└──────────────────────────────────────────────────────┘

Characteristics:
✓ Easy to implement
✓ Works for most models
✓ Linear scaling (ideally)
✗ Model must fit on single GPU
✗ Communication overhead
```

### Model Parallelism

Split model across devices, same data:

```
┌──────────────────────────────────────────────────────┐
│             Model Parallelism                        │
│                                                      │
│  GPU 0:  Layers 1-5    GPU 1:  Layers 6-10          │
│          ┌────────┐             ┌────────┐          │
│          │Layer 1 │             │Layer 6 │          │
│          │Layer 2 │             │Layer 7 │          │
│          │Layer 3 │──Forward───▶│Layer 8 │          │
│          │Layer 4 │             │Layer 9 │          │
│          │Layer 5 │◀──Backward──│Layer 10│          │
│          └────────┘             └────────┘          │
│                                                      │
│  Same batch flows through both GPUs sequentially    │
└──────────────────────────────────────────────────────┘

Characteristics:
✓ Can train very large models
✓ Lower memory per GPU
✗ More complex to implement
✗ Sequential execution (GPU idle time)
✗ Requires careful pipeline design
```

### Pipeline Parallelism

Split model into stages, pipeline mini-batches:

```
┌──────────────────────────────────────────────────────┐
│           Pipeline Parallelism                       │
│                                                      │
│  Time →                                             │
│                                                      │
│  GPU 0 (Stage 1):  [B1] [B2] [B3] [B4]             │
│  GPU 1 (Stage 2):      [B1] [B2] [B3] [B4]         │
│  GPU 2 (Stage 3):          [B1] [B2] [B3] [B4]     │
│  GPU 3 (Stage 4):              [B1] [B2] [B3] [B4] │
│                                                      │
│  Better GPU utilization than naive model parallel   │
└──────────────────────────────────────────────────────┘

Characteristics:
✓ Better GPU utilization than model parallel
✓ Can train very large models
✗ Complex to implement
✗ "Bubble" overhead at start/end
✗ Requires careful batch splitting
```

### Hybrid Parallelism

Combine strategies for optimal performance:

```
┌──────────────────────────────────────────────────────┐
│        Hybrid: Data + Model Parallelism             │
│                                                      │
│  Data Parallel Replicas (across nodes)              │
│      │                         │                    │
│      ├─ Node 1 ────────────────├─ Node 2           │
│      │                         │                    │
│  Model Parallel (within node)                       │
│  GPU 0  GPU 1  GPU 2  GPU 3   GPU 4  GPU 5  ...   │
│  [L1-2] [L3-4] [L5-6] [L7-8]  [L1-2] [L3-4] ...   │
│                                                      │
│  Example: GPT-3 training uses this approach         │
└──────────────────────────────────────────────────────┘
```

## Collective Communication

Distributed training relies on collective communication operations:

### AllReduce

Combine values from all processes and distribute result:

```
┌──────────────────────────────────────────────────────┐
│                   AllReduce                          │
│                                                      │
│  Before:                                            │
│  GPU 0: grad = [1, 2, 3]                           │
│  GPU 1: grad = [4, 5, 6]                           │
│  GPU 2: grad = [7, 8, 9]                           │
│                                                      │
│  After AllReduce (sum):                             │
│  GPU 0: grad = [12, 15, 18]                        │
│  GPU 1: grad = [12, 15, 18]                        │
│  GPU 2: grad = [12, 15, 18]                        │
│                                                      │
│  After averaging:                                   │
│  All GPUs: grad = [4, 5, 6]                        │
└──────────────────────────────────────────────────────┘

Most important operation for data parallelism!
```

### Ring AllReduce

Efficient AllReduce implementation:

```
┌──────────────────────────────────────────────────────┐
│              Ring AllReduce                          │
│                                                      │
│  GPUs arranged in ring topology:                    │
│  GPU 0 ↔ GPU 1 ↔ GPU 2 ↔ GPU 3 ↔ GPU 0            │
│                                                      │
│  Phase 1: Reduce-Scatter                            │
│    Each GPU receives partial sum from neighbor      │
│    2(N-1) messages total                            │
│                                                      │
│  Phase 2: AllGather                                 │
│    Each GPU shares complete result with neighbor    │
│    2(N-1) messages total                            │
│                                                      │
│  Total: 4(N-1) messages                             │
│  Bandwidth: O(1) per GPU (optimal!)                 │
└──────────────────────────────────────────────────────┘

Benefits:
- Bandwidth doesn't increase with GPU count
- Scales to many GPUs
- Used by PyTorch DDP
```

### Other Collective Operations

```python
# Broadcast: Send from one process to all
# Before: GPU 0: [1,2,3]   GPU 1: [0,0,0]
# After:  GPU 0: [1,2,3]   GPU 1: [1,2,3]
dist.broadcast(tensor, src=0)

# Reduce: Combine from all to one
# Before: GPU 0: [1,2]  GPU 1: [3,4]
# After:  GPU 0: [4,6]  GPU 1: [3,4]  (unchanged)
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

# AllGather: Gather from all, send to all
# Before: GPU 0: [1,2]  GPU 1: [3,4]
# After:  GPU 0: [1,2,3,4]  GPU 1: [1,2,3,4]
dist.all_gather(tensor_list, tensor)

# Scatter: Split from one to all
# Before: GPU 0: [1,2,3,4]
# After:  GPU 0: [1,2]  GPU 1: [3,4]
dist.scatter(tensor, scatter_list, src=0)

# Gather: Collect from all to one
# Before: GPU 0: [1,2]  GPU 1: [3,4]
# After:  GPU 0: [1,2,3,4]  GPU 1: [3,4]
dist.gather(tensor, gather_list, dst=0)
```

## PyTorch Distributed Basics

### Process Groups

```python
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """Initialize distributed process group"""
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # NCCL for GPU, Gloo for CPU
        rank=rank,       # Process ID (0 to world_size-1)
        world_size=world_size  # Total number of processes
    )

def cleanup():
    """Cleanup distributed"""
    dist.destroy_process_group()

# Launch processes
def main(rank, world_size):
    setup(rank, world_size)

    # Your distributed code here
    print(f"Process {rank}/{world_size}")

    cleanup()

if __name__ == '__main__':
    world_size = 4  # 4 GPUs
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### Distributed Data Loading

```python
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(rank, world_size, batch_size):
    """Create distributed dataloader"""
    # Create dataset
    dataset = YourDataset()

    # Create distributed sampler
    # Each process gets different subset of data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=4,
        pin_memory=True
    )

    return dataloader

# Important: Set epoch for sampler (shuffling)
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # Ensure different shuffle each epoch

    for batch in train_loader:
        # Training code
        pass
```

## Synchronous vs Asynchronous Training

### Synchronous Training (More Common)

```python
# All workers synchronize gradients after each batch
def synchronous_training_step(model, data, target):
    """Standard synchronous training"""
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Implicit synchronization via AllReduce
    # All workers wait for all gradients
    optimizer.step()
    optimizer.zero_grad()

# Characteristics:
# ✓ Deterministic (same as single GPU with larger batch)
# ✓ Better convergence
# ✗ Speed limited by slowest worker
# ✗ Stragglers cause delays
```

### Asynchronous Training (Less Common)

```python
# Workers update independently without waiting
# Implemented with parameter servers

# Characteristics:
# ✓ No waiting for stragglers
# ✓ Higher throughput
# ✗ Stale gradients
# ✗ Worse convergence
# ✗ Complex implementation

# Rarely used in modern deep learning
# Synchronous SGD preferred due to better convergence
```

## Gradient Synchronization Strategies

### Standard DDP (Default)

```python
# Synchronize gradients after backward pass
loss.backward()  # Compute gradients
# ← AllReduce happens here (automatically)
optimizer.step()  # Update with synchronized gradients
```

### Gradient Accumulation

```python
# Accumulate gradients over multiple batches
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        # Synchronize accumulated gradients
        optimizer.step()
        optimizer.zero_grad()

# Effective batch size = batch_size * accumulation_steps * world_size
# Useful when memory limited
```

### No Sync Context

```python
# Skip gradient synchronization for some steps
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model)

for i, (data, target) in enumerate(train_loader):
    # Skip sync for intermediate steps
    if i % accumulation_steps != 0:
        with model.no_sync():
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # No AllReduce
    else:
        # Sync on last step
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # AllReduce happens
        optimizer.step()
        optimizer.zero_grad()
```

## Distributed Training Efficiency

### Linear Scaling

Ideal speedup: Training with N GPUs is N times faster

```python
# Linear scaling formula:
# Speedup = T_1GPU / T_NGPU

# Example:
# 1 GPU:  100 seconds/epoch
# 4 GPUs: 25 seconds/epoch
# Speedup: 100/25 = 4x (perfect linear scaling!)

# In practice:
# 4 GPUs: ~28 seconds/epoch
# Speedup: 100/28 = 3.57x (88% efficiency)

# Efficiency = Speedup / N
# Efficiency = 3.57 / 4 = 89%
```

### Scaling Efficiency Factors

```
┌──────────────────────────────────────────────────────┐
│         What Affects Scaling Efficiency?            │
│                                                      │
│  Communication Overhead:                            │
│  - AllReduce time vs computation time              │
│  - Network bandwidth (NVLink > PCIe > Ethernet)    │
│  - Model size (larger = more gradients to sync)    │
│                                                      │
│  Load Imbalance:                                    │
│  - Uneven data distribution                        │
│  - Stragglers (slow workers)                       │
│  - Different GPU speeds                            │
│                                                      │
│  Batch Size Effects:                                │
│  - Too small: Overhead dominates                   │
│  - Too large: Convergence issues                   │
│  - Sweet spot: Balance efficiency and convergence  │
└──────────────────────────────────────────────────────┘
```

### Communication to Computation Ratio

```python
def calculate_comm_comp_ratio(model, batch_size, world_size):
    """
    Estimate communication/computation ratio
    """
    # Gradient size (bytes)
    param_size = sum(p.numel() for p in model.parameters()) * 4  # FP32

    # Computation (FLOPs)
    # Rough estimate for transformer
    seq_len = 512
    hidden = 768
    flops = 6 * param_size * batch_size * seq_len

    # Communication (bytes)
    # AllReduce: 2(N-1)/N * param_size for Ring AllReduce
    comm_bytes = 2 * (world_size - 1) / world_size * param_size

    # Communication time (assuming 100 GB/s)
    bandwidth = 100e9  # bytes/sec
    comm_time = comm_bytes / bandwidth

    # Computation time (assuming 100 TFLOPS)
    compute_power = 100e12  # FLOPS
    comp_time = flops / compute_power

    ratio = comm_time / comp_time

    print(f"Communication time: {comm_time*1000:.2f} ms")
    print(f"Computation time: {comp_time*1000:.2f} ms")
    print(f"Comm/Comp ratio: {ratio:.2f}")

    if ratio > 0.1:
        print("⚠ Communication overhead is significant!")
    else:
        print("✓ Good compute/communication balance")

    return ratio
```

### Weak Scaling vs Strong Scaling

```
Strong Scaling (fixed total workload):
┌────────────────────────────────────┐
│ 1 GPU:  1000 samples, 100s         │
│ 2 GPUs: 1000 samples,  50s (ideal) │
│ 4 GPUs: 1000 samples,  25s (ideal) │
└────────────────────────────────────┘
Each GPU processes fewer samples
Total batch size constant

Weak Scaling (fixed per-GPU workload):
┌────────────────────────────────────┐
│ 1 GPU:  1000 samples, 100s         │
│ 2 GPUs: 2000 samples, 100s (ideal) │
│ 4 GPUs: 4000 samples, 100s (ideal) │
└────────────────────────────────────┘
Each GPU processes same number of samples
Total batch size increases

ML typically uses weak scaling:
- Larger batch size with more GPUs
- Maintains GPU efficiency
- May need to adjust learning rate
```

## Practical Example: Simple Distributed Training

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Running on rank {rank}")
    setup(rank, world_size)

    # Create model and move to GPU
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Create distributed dataset
    dataset = torch.randn(1000, 10)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle

        for batch in loader:
            batch = batch.to(rank)

            optimizer.zero_grad()
            output = ddp_model(batch)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        # Only print from rank 0
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

if __name__ == '__main__':
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## Debugging Distributed Training

### Common Issues

```python
# Issue 1: Hanging at init_process_group
# Cause: Firewall, wrong MASTER_ADDR/PORT
# Solution: Check network, verify environment variables

# Issue 2: NCCL timeout
# Cause: Slow network, unbalanced load
# Solution: Increase timeout
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes

# Issue 3: Different random seeds across processes
# Solution: Set seeds properly
def set_seed(seed, rank):
    torch.manual_seed(seed + rank)  # Different seed per rank
    np.random.seed(seed + rank)

# Issue 4: Incorrect batch size scaling
# Total batch size = per_gpu_batch_size * world_size
# Remember to adjust learning rate accordingly

# Issue 5: Non-deterministic behavior
# Cause: Non-deterministic CUDA operations
# Solution:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Debugging Tools

```python
# Enable verbose logging
os.environ['NCCL_DEBUG'] = 'INFO'

# Check process group initialization
if dist.is_initialized():
    print(f"Rank: {dist.get_rank()}")
    print(f"World size: {dist.get_world_size()}")

# Barrier for synchronization debugging
dist.barrier()  # All processes wait here
print(f"Rank {rank} passed barrier")

# Test collective operations
tensor = torch.tensor([rank], device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {rank}: AllReduce result = {tensor.item()}")
```

## Practical Exercises

### Exercise 1: Setup Distributed Environment

```python
# TODO: Implement distributed setup
def setup_distributed(rank, world_size, backend='nccl'):
    """
    Initialize distributed environment
    - Set environment variables
    - Initialize process group
    - Set device
    """
    pass

# Test with 2 processes
mp.spawn(setup_distributed, args=(2,), nprocs=2)
```

### Exercise 2: Distributed Training Loop

```python
# TODO: Complete distributed training function
def distributed_train(rank, world_size, epochs=10):
    """
    Implement complete distributed training:
    - Setup process group
    - Create DDP model
    - Create distributed dataloader
    - Training loop
    - Synchronization
    """
    pass
```

### Exercise 3: Measure Scaling Efficiency

```python
# TODO: Measure and compare training time
def benchmark_scaling(model, dataset, num_gpus_list):
    """
    Measure training time with different GPU counts
    Calculate and plot scaling efficiency
    """
    results = {}

    for num_gpus in num_gpus_list:
        start = time.time()
        # Train for 100 iterations
        elapsed = time.time() - start
        results[num_gpus] = elapsed

    # Calculate efficiency
    baseline = results[1]
    for num_gpus, elapsed in results.items():
        speedup = baseline / elapsed
        efficiency = speedup / num_gpus
        print(f"{num_gpus} GPUs: {speedup:.2f}x speedup, {efficiency*100:.1f}% efficiency")

# Test with 1, 2, 4, 8 GPUs
benchmark_scaling(model, dataset, [1, 2, 4, 8])
```

## Summary

In this lesson, you learned:

1. **Parallelism Strategies**: Data parallelism, model parallelism, pipeline parallelism
2. **Collective Operations**: AllReduce, Broadcast, Gather, Scatter
3. **PyTorch Distributed**: Process groups, distributed sampling, DDP
4. **Synchronization**: Synchronous vs asynchronous, gradient accumulation
5. **Efficiency Metrics**: Linear scaling, communication overhead, weak vs strong scaling
6. **Debugging**: Common issues and tools for distributed training

## Key Takeaways

- **Data parallelism is most common** - works for models that fit on single GPU
- **AllReduce is critical** - efficient gradient synchronization
- **Communication overhead matters** - use fast interconnects (NVLink)
- **Scaling isn't linear** - expect 80-90% efficiency
- **Batch size scaling** - adjust learning rate with total batch size
- **DistributedSampler is essential** - ensures each process sees different data

## Further Reading

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Model parallelism at scale

## Next Steps

In the next lesson, **Multi-GPU Training Strategies**, we'll implement DistributedDataParallel, optimize multi-GPU training on a single node, and explore advanced optimization techniques.

---

**Ready to implement distributed training? Let's build scalable training pipelines!**
