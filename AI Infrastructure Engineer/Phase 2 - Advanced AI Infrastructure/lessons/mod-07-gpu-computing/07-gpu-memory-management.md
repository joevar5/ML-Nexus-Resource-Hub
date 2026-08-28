# Lesson 07: GPU Memory Management & Optimization

## Learning Objectives

By the end of this lesson, you will be able to:

1. Profile and analyze GPU memory usage in PyTorch
2. Implement gradient checkpointing to reduce memory consumption
3. Optimize activation memory with various techniques
4. Diagnose and fix Out of Memory (OOM) errors
5. Use memory-efficient attention mechanisms
6. Apply advanced memory optimization strategies

## Introduction

GPU memory is often the limiting factor in training large models. Understanding memory usage patterns and optimization techniques is critical for AI infrastructure engineers. This lesson provides practical strategies for managing and optimizing GPU memory.

## Understanding GPU Memory Usage

### Memory Components During Training

```
┌──────────────────────────────────────────────────────┐
│         GPU Memory Breakdown (Training)              │
│                                                      │
│ 1. Model Parameters             │  ~M bytes         │
│    - Weights, biases                                │
│                                                      │
│ 2. Gradients                    │  ~M bytes         │
│    - Same size as parameters                        │
│                                                      │
│ 3. Optimizer States             │  ~2M bytes (Adam) │
│    - Momentum: M bytes                              │
│    - Variance: M bytes                              │
│                                                      │
│ 4. Activations/Intermediates    │  Variable (large!)│
│    - Forward pass activations                       │
│    - Depends on: batch size, sequence length        │
│                                                      │
│ 5. Temporary Buffers            │  Variable         │
│    - Workspace for operations                       │
│    - cuDNN, cuBLAS buffers                         │
│                                                      │
│ Total ≈ 4M + Activations                           │
│                                                      │
│ Example: 7B parameter model                         │
│   Parameters: 7B × 4 bytes = 28 GB                 │
│   Total (FP32): ~120 GB                            │
│   Total (FP16): ~60 GB                             │
└──────────────────────────────────────────────────────┘
```

### Profiling Memory Usage

```python
import torch

def profile_memory():
    """Profile current GPU memory usage"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = 0

    # Currently allocated memory
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    print(f"Allocated: {allocated:.2f} GB")

    # Reserved by caching allocator
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"Reserved: {reserved:.2f} GB")

    # Peak memory usage
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Peak: {peak:.2f} GB")

    # Total GPU memory
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"Total: {total:.2f} GB")

    # Utilization
    print(f"Utilization: {allocated/total*100:.1f}%")

    return {
        'allocated': allocated,
        'reserved': reserved,
        'peak': peak,
        'total': total
    }

# Usage
profile_memory()

# Reset peak statistics
torch.cuda.reset_peak_memory_stats()
```

### Detailed Memory Snapshot

```python
def detailed_memory_profile():
    """Get detailed memory breakdown"""
    if not torch.cuda.is_available():
        return

    # Get memory summary
    print(torch.cuda.memory_summary())

    # Get memory statistics
    stats = torch.cuda.memory_stats()

    print(f"\nAllocations:")
    print(f"  Active: {stats['active.all.current'] / 1024**2:.1f} MB")
    print(f"  Peak: {stats['active.all.peak'] / 1024**2:.1f} MB")

    print(f"\nAllocated:")
    print(f"  Current: {stats['allocated_bytes.all.current'] / 1024**3:.2f} GB")
    print(f"  Peak: {stats['allocated_bytes.all.peak'] / 1024**3:.2f} GB")

    print(f"\nReserved:")
    print(f"  Current: {stats['reserved_bytes.all.current'] / 1024**3:.2f} GB")
    print(f"  Peak: {stats['reserved_bytes.all.peak'] / 1024**3:.2f} GB")

detailed_memory_profile()
```

## Gradient Checkpointing

### The Problem: Activation Memory

```
Forward pass without checkpointing:
┌────────────────────────────────────────┐
│ Layer 1 → Act1 (store for backward)   │
│ Layer 2 → Act2 (store)                │
│ Layer 3 → Act3 (store)                │
│ ...                                    │
│ Layer N → ActN (store)                │
│                                        │
│ Memory: O(N × batch_size × hidden)    │
└────────────────────────────────────────┘

For large models:
- BERT-Large: 24 layers
- GPT-3: 96 layers
- Activation memory >> parameter memory!
```

### Gradient Checkpointing Solution

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    """Model with gradient checkpointing"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(24)
        ])

    def forward(self, x):
        # Without checkpointing (high memory)
        # for layer in self.layers:
        #     x = layer(x)

        # With checkpointing (low memory, slower)
        for layer in self.layers:
            x = checkpoint(layer, x)

        return x

# How it works:
# 1. Forward: Only store inputs to checkpointed functions
# 2. Backward: Recompute activations on-the-fly
#
# Trade-off:
#   Memory: ↓ (store ~√N instead of N activations)
#   Speed: ↓ (~33% slower due to recomputation)
```

### Selective Checkpointing

```python
class SmartCheckpointedModel(nn.Module):
    """Checkpoint only expensive layers"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(24)
        ])

    def forward(self, x):
        # Checkpoint every Nth layer
        checkpoint_frequency = 4

        for i, layer in enumerate(self.layers):
            if i % checkpoint_frequency == 0:
                # Checkpoint this layer
                x = checkpoint(layer, x)
            else:
                # Normal forward
                x = layer(x)

        return x

# Balance memory savings vs speed
# Checkpoint frequency: 1 (all) → N (none)
```

### Measuring Checkpoint Impact

```python
import time
import torch

def benchmark_checkpointing(model, input_data, use_checkpoint=False):
    """Compare with/without checkpointing"""

    torch.cuda.reset_peak_memory_stats()

    # Forward + Backward
    start = time.time()

    output = model(input_data)
    loss = output.sum()
    loss.backward()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Checkpointing: {use_checkpoint}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Peak Memory: {peak_memory:.2f} GB")

    return elapsed, peak_memory

# Compare
model_normal = MyModel()
model_checkpointed = CheckpointedModel()

data = torch.randn(32, 512, 768).cuda()

print("Without checkpointing:")
t1, m1 = benchmark_checkpointing(model_normal, data, False)

print("\nWith checkpointing:")
t2, m2 = benchmark_checkpointing(model_checkpointed, data, True)

print(f"\nMemory savings: {m1/m2:.2f}x")
print(f"Speed penalty: {t2/t1:.2f}x slower")
```

## Memory-Efficient Attention

### Standard Attention Memory Problem

```python
def standard_attention(Q, K, V):
    """
    Standard attention: O(N²) memory

    Q, K, V: [batch, seq_len, hidden]
    Attention matrix: [batch, seq_len, seq_len]
    """
    # Compute attention scores: Q @ K^T
    # Shape: [batch, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1))  # O(N²) memory!

    # Softmax
    attn_weights = torch.softmax(scores / (Q.size(-1) ** 0.5), dim=-1)

    # Apply attention: attn_weights @ V
    output = torch.matmul(attn_weights, V)

    return output

# Problem: Attention matrix is O(N²)
# For seq_len=4096: 4096² = 16M elements
# With batch=8, heads=16: 2GB just for attention!
```

### Flash Attention

```python
# Flash Attention: Memory-efficient attention
# Paper: https://arxiv.org/abs/2205.14135

try:
    from flash_attn import flash_attn_func

    def flash_attention(Q, K, V):
        """
        Flash Attention: O(N) memory

        - Fused kernel (faster)
        - Recomputes attention on-the-fly (less memory)
        - No O(N²) matrix materialization
        """
        # Q, K, V: [batch, seq_len, num_heads, head_dim]
        output = flash_attn_func(Q, K, V, causal=False)
        return output

    # Benefits:
    # - 3-4x faster than standard attention
    # - 10-20x less memory
    # - Enables longer sequences

except ImportError:
    print("Flash Attention not installed")
    print("pip install flash-attn")
```

### Memory-Efficient Attention (xFormers)

```python
# Alternative: xFormers memory-efficient attention
try:
    from xformers.ops import memory_efficient_attention

    def efficient_attention(Q, K, V):
        """
        xFormers memory-efficient attention

        - Automatic algorithm selection
        - Supports various attention variants
        - Easy drop-in replacement
        """
        output = memory_efficient_attention(
            Q, K, V,
            attn_bias=None,  # Optional attention mask
            scale=1.0 / (Q.size(-1) ** 0.5)
        )
        return output

except ImportError:
    print("xFormers not installed")
    print("pip install xformers")
```

## Optimization Techniques

### 1. Reduce Batch Size

```python
# Simple but effective
# Problem: Batch too large for GPU memory

# Solution 1: Reduce batch size
batch_size = 16  # Instead of 32

# Solution 2: Gradient accumulation (simulate larger batch)
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps  # 64

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Reduce memory by 2x with minimal accuracy loss
scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Forward in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Memory savings:
# - FP32: 4 bytes per parameter
# - FP16: 2 bytes per parameter
# - 2x memory reduction!
```

### 3. Empty Cache Strategically

```python
# Clear unused cached memory
# Use sparingly (overhead)

for epoch in range(num_epochs):
    # Training
    for batch in train_loader:
        train_step(batch)

    # Validation (different memory pattern)
    torch.cuda.empty_cache()  # Free cached memory

    for batch in val_loader:
        val_step(batch)

    torch.cuda.empty_cache()  # Before next epoch

# Don't use empty_cache() in tight loops!
# Only between major phases
```

### 4. Delete Unnecessary Tensors

```python
def train_step_memory_efficient(model, data, target):
    """Explicitly delete intermediate tensors"""

    output = model(data)
    loss = criterion(output, target)

    # Delete output tensor (not needed after loss)
    del output

    loss.backward()

    # Delete loss (not needed after backward)
    del loss

    optimizer.step()
    optimizer.zero_grad()

# Python garbage collector + CUDA allocator
# will free memory faster with explicit del
```

### 5. Model Sharding (ZeRO)

```python
# Use DeepSpeed ZeRO to partition model
import deepspeed

ds_config = {
    "zero_optimization": {
        "stage": 2,  # Partition gradients + optimizer
        # Memory reduction: ~8x
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# ZeRO stages:
# Stage 1: Partition optimizer states (4x reduction)
# Stage 2: + partition gradients (8x reduction)
# Stage 3: + partition parameters (linear reduction)
```

## Diagnosing OOM Errors

### Common OOM Scenarios

```python
# 1. Forward pass OOM (activations too large)
try:
    output = model(large_batch)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM during forward pass")
        print("Solutions:")
        print("- Reduce batch size")
        print("- Enable gradient checkpointing")
        print("- Reduce sequence length")

# 2. Backward pass OOM (gradient computation)
try:
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM during backward pass")
        print("Solutions:")
        print("- Gradient checkpointing")
        print("- Reduce batch size")
        print("- Use gradient accumulation")

# 3. Optimizer step OOM (optimizer states)
try:
    optimizer.step()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM during optimizer step")
        print("Solutions:")
        print("- Use different optimizer (SGD instead of Adam)")
        print("- Use ZeRO optimizer")
        print("- Offload optimizer to CPU")
```

### Memory Debugging Tool

```python
def debug_memory_usage():
    """Debug memory allocation"""
    print("\n" + "="*50)
    print("Memory Debug Info")
    print("="*50)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Current state
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\nCurrent State:")
    print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved:  {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"  Peak:      {peak:.2f} GB ({peak/total*100:.1f}%)")
    print(f"  Total:     {total:.2f} GB")

    # Fragmentation
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    print(f"\nFragmentation: {fragmentation:.1f}%")

    if fragmentation > 20:
        print("⚠ High fragmentation detected!")
        print("Consider calling torch.cuda.empty_cache()")

    # Allocations
    stats = torch.cuda.memory_stats()
    num_allocs = stats.get('num_alloc_retries', 0)
    print(f"\nAllocation retries: {num_allocs}")

    if num_allocs > 0:
        print("⚠ Memory pressure detected!")

    print("="*50 + "\n")

# Use during training
debug_memory_usage()
```

## Advanced Optimization Strategies

### CPU Offloading

```python
# Offload to CPU when not actively used
class CPUOffloadModel(nn.Module):
    """Offload layers to CPU between forward passes"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            HugeLayer().cpu() for _ in range(10)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Move layer to GPU
            layer = layer.cuda()

            # Forward
            x = layer(x)

            # Move back to CPU
            layer = layer.cpu()

            torch.cuda.empty_cache()

        return x

# Trade-off:
# + Enables very large models
# - Very slow (CPU-GPU transfer overhead)
# Use only when necessary!
```

### Automatic Memory Management

```python
# PyTorch 2.0+ automatic memory management
import torch._dynamo as dynamo

@dynamo.optimize("inductor")
def optimized_forward(model, x):
    """Automatically optimized by PyTorch compiler"""
    return model(x)

# Compiler can:
# - Fuse operations (reduce intermediates)
# - Optimize memory layout
# - Remove unnecessary copies
```

## Practical Exercises

### Exercise 1: Profile Memory Usage

```python
# TODO: Profile memory usage of this training loop
def profile_training_memory(model, train_loader):
    """
    Profile memory at each stage:
    - Before forward
    - After forward
    - After backward
    - After optimizer step

    Identify memory bottlenecks
    """
    pass
```

### Exercise 2: Implement Gradient Checkpointing

```python
# TODO: Add gradient checkpointing to reduce memory
class MemoryHungryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(4096, 4096) for _ in range(48)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Modify to use checkpointing
# Measure memory savings
```

### Exercise 3: Fix OOM Error

```python
# TODO: This code causes OOM - fix it!
def oom_prone_training():
    model = VeryLargeModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in large_dataloader:  # batch_size=128
            data, target = batch
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Apply multiple techniques to fix:
# - Reduce batch size
# - Gradient accumulation
# - Mixed precision
# - Gradient checkpointing
```

## Summary

In this lesson, you learned:

1. **Memory Profiling**: Tools to monitor and analyze GPU memory usage
2. **Gradient Checkpointing**: Trade compute for memory (√N activations)
3. **Memory-Efficient Attention**: Flash Attention, xFormers for O(N) memory
4. **Optimization Techniques**: Mixed precision, batching, caching, sharding
5. **OOM Debugging**: Diagnose and fix out-of-memory errors
6. **Advanced Strategies**: CPU offloading, compiler optimizations

## Key Takeaways

- **Activations are often the memory bottleneck** - use gradient checkpointing
- **Mixed precision training** provides easy 2x memory savings
- **Gradient accumulation** simulates larger batches without OOM
- **Profile before optimizing** - measure to find actual bottlenecks
- **Flash Attention** is essential for long sequence models
- **ZeRO** enables training models N times larger
- **empty_cache() sparingly** - use between phases, not in loops

## Further Reading

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)

## Next Steps

In the final lesson, **Advanced GPU Optimization**, we'll cover performance profiling, kernel optimization, and production deployment best practices for GPU-accelerated ML systems.

---

**Ready to maximize GPU performance? Let's dive into advanced optimization!**
