# Lesson 05: Multi-GPU Training Strategies

## Learning Objectives

By the end of this lesson, you will be able to:

1. Implement DataParallel and DistributedDataParallel in PyTorch
2. Understand the differences and trade-offs between DP and DDP
3. Set up efficient multi-GPU training on a single node
4. Optimize data loading for multi-GPU setups
5. Handle gradient synchronization and accumulation
6. Monitor and optimize GPU utilization across multiple GPUs

## Introduction

Multi-GPU training on a single machine (single-node, multi-GPU) is the most common distributed training setup. This lesson focuses on practical implementation using PyTorch's DataParallel (DP) and DistributedDataParallel (DDP).

## DataParallel vs DistributedDataParallel

### DataParallel (DP) - Legacy Approach

```python
import torch
import torch.nn as nn

# Simple to use but has limitations
model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

# Training works as usual
for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**How DataParallel Works:**

```
┌──────────────────────────────────────────────────────┐
│               DataParallel Flow                      │
│                                                      │
│  1. Scatter input across GPUs                       │
│     GPU 0: batch[0:8]   GPU 1: batch[8:16]         │
│                                                      │
│  2. Replicate model to all GPUs                     │
│     GPU 0: model_copy    GPU 1: model_copy         │
│                                                      │
│  3. Forward pass (parallel)                         │
│     GPU 0: out[0:8]     GPU 1: out[8:16]           │
│                                                      │
│  4. Gather outputs to GPU 0                         │
│     GPU 0: loss = criterion(concat(outs), target)  │
│                                                      │
│  5. Backward on GPU 0, scatter gradients            │
│     GPU 0: grads → GPU 1: grads                    │
│                                                      │
│  6. Update parameters on GPU 0                      │
│     GPU 0: optimizer.step()                         │
│                                                      │
│  7. Broadcast parameters to other GPUs              │
│     GPU 0: params → GPU 1: params                  │
└──────────────────────────────────────────────────────┘

Limitations:
✗ GPU 0 bottleneck (gathering, loss, gradients)
✗ Single-process (Python GIL contention)
✗ Slower than DDP
✗ Unbalanced GPU utilization
✓ Easy to use (one line change)
```

### DistributedDataParallel (DDP) - Recommended

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and wrap with DDP
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create distributed dataloader
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.Adam(ddp_model.parameters())

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for data, target in loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

# Launch
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

**How DistributedDataParallel Works:**

```
┌──────────────────────────────────────────────────────┐
│         DistributedDataParallel Flow                │
│                                                      │
│  Each process (GPU) runs independently:             │
│                                                      │
│  Process 0 (GPU 0)     Process 1 (GPU 1)           │
│  ┌────────────┐        ┌────────────┐              │
│  │ Model copy │        │ Model copy │              │
│  │ Batch 0-7  │        │ Batch 8-15 │              │
│  │ Forward    │        │ Forward    │              │
│  │ Backward   │        │ Backward   │              │
│  └────────────┘        └────────────┘              │
│        ↓ AllReduce ↓                                │
│  ┌────────────────────────────┐                    │
│  │  Synchronized gradients    │                    │
│  └────────────────────────────┘                    │
│        ↓                    ↓                       │
│  GPU 0: update         GPU 1: update               │
│                                                      │
│  No bottleneck! All GPUs equally utilized          │
└──────────────────────────────────────────────────────┘

Benefits:
✓ No GPU bottleneck
✓ Multi-process (no GIL)
✓ Faster than DataParallel
✓ Balanced GPU utilization
✓ Works across multiple nodes
✗ More code to set up
```

### Performance Comparison

```python
import time

def benchmark_dp_vs_ddp():
    """Compare DataParallel vs DistributedDataParallel"""

    model = LargeModel()
    batch_size = 64

    # Benchmark DataParallel
    print("Testing DataParallel...")
    dp_model = nn.DataParallel(model).cuda()
    dp_optimizer = torch.optim.Adam(dp_model.parameters())

    start = time.time()
    for _ in range(100):
        data = torch.randn(batch_size, 3, 224, 224).cuda()
        output = dp_model(data)
        loss = output.sum()
        loss.backward()
        dp_optimizer.step()
        dp_optimizer.zero_grad()
    dp_time = time.time() - start

    print(f"DataParallel: {dp_time:.2f}s")

    # Benchmark DDP (would need multi-process setup)
    # Typical results: DDP is 1.5-2x faster than DP

benchmark_dp_vs_ddp()
```

## Implementing DDP: Complete Example

### Single-File DDP Setup

```python
#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.targets = torch.randint(0, 10, (length,))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.len

def setup(rank, world_size):
    """Initialize distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """Cleanup distributed environment"""
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, criterion, rank, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(rank)
        target = target.to(rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0 and rank == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def main_worker(rank, world_size):
    """Main training function for each process"""
    print(f"Running on rank {rank}")
    setup(rank, world_size)

    # Create model and move to GPU
    model = SimpleModel().to(rank)

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Create dataset and sampler
    dataset = RandomDataset(20, 10000)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    # Create optimizer and loss
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch

        loss = train_epoch(
            ddp_model,
            train_loader,
            optimizer,
            criterion,
            rank,
            epoch
        )

        # Synchronize and print from rank 0
        dist.barrier()
        if rank == 0:
            print(f"Epoch {epoch} completed, Average Loss: {loss:.4f}\n")

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    mp.spawn(
        main_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### Launch Scripts

**Using torchrun (recommended):**

```bash
#!/bin/bash
# launch_ddp.sh

# Single node, 4 GPUs
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --epochs 10 \
    --batch-size 32
```

**Using torch.distributed.launch (legacy):**

```bash
#!/bin/bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train.py
```

## Optimizing Data Loading

### Efficient DataLoader Configuration

```python
def create_dataloader(dataset, rank, world_size, batch_size):
    """Create optimized dataloader for DDP"""

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Ensure all ranks have same number of batches
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True,  # Keep workers alive
        drop_last=True  # Consistent batch sizes
    )

    return loader

# Usage
train_loader = create_dataloader(train_dataset, rank, world_size, 32)
```

### Data Loading Bottleneck Detection

```python
import time

def profile_dataloader(loader, num_batches=100):
    """Profile data loading performance"""
    load_times = []
    transfer_times = []

    for i, (data, target) in enumerate(loader):
        if i >= num_batches:
            break

        # Measure transfer time
        start = time.time()
        data_gpu = data.cuda(non_blocking=True)
        target_gpu = target.cuda(non_blocking=True)
        torch.cuda.synchronize()
        transfer_time = time.time() - start
        transfer_times.append(transfer_time)

    print(f"Avg transfer time: {sum(transfer_times)/len(transfer_times)*1000:.2f}ms")

    if sum(transfer_times)/len(transfer_times) > 0.010:  # 10ms
        print("⚠ Data transfer is slow! Consider:")
        print("  - Increase num_workers")
        print("  - Enable pin_memory")
        print("  - Reduce data preprocessing")
```

## Gradient Synchronization Optimization

### Default Behavior

```python
# DDP automatically synchronizes gradients during backward()
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # ← AllReduce happens here
    optimizer.step()
```

### Gradient Accumulation with no_sync()

```python
# Accumulate gradients without synchronization
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    data, target = data.to(rank), target.to(rank)

    # Skip synchronization for intermediate steps
    if i % accumulation_steps != 0:
        with model.no_sync():
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()  # No AllReduce!
    else:
        # Synchronize on last step
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()  # AllReduce
        optimizer.step()
        optimizer.zero_grad()

# Effective batch size: batch_size * accumulation_steps * world_size
```

### Gradient Clipping in DDP

```python
from torch.nn.utils import clip_grad_norm_

# Gradient clipping with DDP
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Clip gradients (works correctly with DDP)
    clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

# DDP handles gradient synchronization before clipping
```

## Mixed Precision with DDP

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(rank, world_size):
    """Training with Automatic Mixed Precision and DDP"""
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters())
    scaler = GradScaler()

    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        for data, target in loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()

            # Forward with autocasting
            with autocast():
                output = ddp_model(data)
                loss = criterion(output, target)

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    cleanup()

# Mixed precision + DDP = maximum performance!
```

## Monitoring Multi-GPU Training

### GPU Utilization Monitoring

```python
import nvidia_smi

def monitor_gpus():
    """Monitor all GPU utilization"""
    nvidia_smi.nvmlInit()

    device_count = nvidia_smi.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {i}:")
        print(f"  Utilization: {util.gpu}%")
        print(f"  Memory: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB")

    nvidia_smi.nvmlShutdown()

# Run in background during training
import threading

def monitor_loop():
    while training:
        monitor_gpus()
        time.sleep(5)

monitor_thread = threading.Thread(target=monitor_loop)
monitor_thread.start()
```

### Per-GPU Memory Tracking

```python
def track_memory_all_gpus():
    """Track memory on all GPUs"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Call periodically during training
if batch_idx % 100 == 0:
    track_memory_all_gpus()
```

## Synchronization and Communication

### Synchronization Primitives

```python
# Barrier: Wait for all processes
dist.barrier()
print(f"Rank {rank} passed barrier")

# Broadcast: Send tensor from one process to all
if rank == 0:
    tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
else:
    tensor = torch.zeros(3).cuda()

dist.broadcast(tensor, src=0)
print(f"Rank {rank}: {tensor}")  # All ranks have [1, 2, 3]

# AllReduce: Combine and distribute
tensor = torch.tensor([rank]).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(f"Rank {rank}: {tensor.item()}")  # All ranks have sum of ranks

# Gather: Collect tensors from all processes to one
if rank == 0:
    gather_list = [torch.zeros(1).cuda() for _ in range(world_size)]
else:
    gather_list = None

tensor = torch.tensor([rank]).cuda()
dist.gather(tensor, gather_list, dst=0)

if rank == 0:
    print(f"Gathered: {[t.item() for t in gather_list]}")
```

### Reducing Metrics Across Processes

```python
def reduce_metric(metric_tensor, world_size):
    """Average metric across all processes"""
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    metric_tensor /= world_size
    return metric_tensor

# Usage
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Reduce loss across all processes
    loss_tensor = torch.tensor([loss.item()]).cuda()
    avg_loss = reduce_metric(loss_tensor, world_size)

    if rank == 0:
        print(f"Average loss across all GPUs: {avg_loss.item():.4f}")
```

## Saving and Loading Checkpoints

### DDP Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, rank, world_size):
    """Save checkpoint (only from rank 0)"""
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # .module for DDP
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

def load_checkpoint(model, optimizer, checkpoint_path, rank):
    """Load checkpoint on all ranks"""
    # Map to correct device
    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']

# Usage in training loop
for epoch in range(start_epoch, num_epochs):
    # Training...

    # Save checkpoint
    save_checkpoint(ddp_model, optimizer, epoch, rank, world_size)

    # All ranks wait
    dist.barrier()
```

## Common Issues and Solutions

### Issue 1: Uneven Batch Sizes

```python
# Problem: Different ranks have different number of batches
# Causes hanging (waiting for AllReduce that never comes)

# Solution: Use drop_last=True
sampler = DistributedSampler(dataset, drop_last=True)
loader = DataLoader(dataset, sampler=sampler, drop_last=True)
```

### Issue 2: Different Random Seeds

```python
# Problem: All processes have same random initialization

# Solution: Set different seeds per rank
def set_seed(seed, rank):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

set_seed(42, rank)
```

### Issue 3: Deadlocks

```python
# Problem: Ranks have different control flow

# Bad: Conditional that differs across ranks
if rank == 0:
    dist.barrier()  # Only rank 0 reaches here → deadlock!

# Good: All ranks execute same code
dist.barrier()  # All ranks reach here

if rank == 0:
    print("Checkpoint saved")
```

## Practical Exercises

### Exercise 1: Convert Single-GPU to Multi-GPU

```python
# TODO: Convert this single-GPU code to use DDP

def train_single_gpu():
    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Implement multi-GPU version with DDP
```

### Exercise 2: Implement Gradient Accumulation

```python
# TODO: Add gradient accumulation to reduce memory usage
def train_with_accumulation(model, loader, accumulation_steps=4):
    """
    Implement gradient accumulation with DDP
    - Use no_sync() for intermediate steps
    - Synchronize on final step
    - Track effective batch size
    """
    pass
```

### Exercise 3: Benchmark Multi-GPU Scaling

```python
# TODO: Measure training time with 1, 2, 4 GPUs
def benchmark_scaling():
    """
    Measure and plot:
    - Training time vs number of GPUs
    - Speedup and efficiency
    - GPU utilization
    """
    pass
```

## Summary

In this lesson, you learned:

1. **DP vs DDP**: DataParallel (easy, slow) vs DistributedDataParallel (complex, fast)
2. **DDP Implementation**: Complete setup with process groups, samplers, and training loops
3. **Data Loading**: Optimized DataLoader configuration for multi-GPU
4. **Gradient Synchronization**: no_sync() for accumulation, gradient clipping
5. **Mixed Precision**: Combining AMP with DDP for maximum performance
6. **Monitoring**: Tracking GPU utilization and memory across devices
7. **Checkpointing**: Saving and loading models in distributed settings

## Key Takeaways

- **Always use DDP over DP** for production training
- **DistributedSampler is essential** for correct data distribution
- **drop_last=True** prevents hanging from uneven batches
- **Save from rank 0 only** to avoid file conflicts
- **Monitor all GPUs** to ensure balanced utilization
- **Mixed precision + DDP** provides best performance

## Further Reading

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Getting Started with DDP](https://pytorch.org/tutorials/beginner/ddp_series_intro.html)
- [DDP Best Practices](https://pytorch.org/tutorials/intermediate/ddp_series_multigpu.html)

## Next Steps

In the next lesson, **Model and Pipeline Parallelism**, we'll explore strategies for training models that don't fit on a single GPU, including layer-wise splitting and pipeline parallelism.

---

**Ready to scale even larger? Let's explore model parallelism!**
