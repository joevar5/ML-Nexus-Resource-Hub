# Lesson 06: Model and Pipeline Parallelism

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand when and why to use model parallelism
2. Implement simple model parallelism by splitting layers across GPUs
3. Understand pipeline parallelism and its benefits
4. Implement tensor parallelism for large layers
5. Use libraries like DeepSpeed and Megatron for advanced parallelism
6. Calculate memory savings and performance trade-offs

## Introduction

When models become too large to fit on a single GPU, data parallelism alone isn't sufficient. Model parallelism techniques split the model itself across multiple devices, enabling training of models with hundreds of billions or even trillions of parameters.

## When to Use Model Parallelism

### The Memory Problem

```
Model Size Examples:
┌────────────────────────────────────────────┐
│ BERT-Base:         340M params   │   1.3 GB │
│ GPT-2:             1.5B params   │   6 GB   │
│ BERT-Large:        340M params   │   1.3 GB │
│ GPT-3 Small:       6.7B params   │  27 GB   │
│ GPT-3:             175B params   │ 700 GB   │
│ LLaMA-70B:         70B params    │ 280 GB   │
│ GPT-4:            ~1.7T params   │  ~7 TB   │
└────────────────────────────────────────────┘

A100 GPU: 80 GB VRAM
→ GPT-3 doesn't fit on single GPU!
→ Need model parallelism
```

### Decision Framework

```python
def should_use_model_parallelism(model_size_gb, gpu_memory_gb, num_gpus):
    """
    Decide if model parallelism is needed

    During training, memory requirements:
    - Model parameters
    - Gradients (same size as parameters)
    - Optimizer states (2x parameters for Adam)
    - Activations (depends on batch size)

    Total ≈ 4x model size (without activations)
    """
    training_memory_needed = model_size_gb * 4

    # Can it fit with data parallelism?
    if training_memory_needed <= gpu_memory_gb:
        print("✓ Use data parallelism (model fits on single GPU)")
        return False

    # Need model parallelism
    memory_per_gpu = training_memory_needed / num_gpus

    if memory_per_gpu <= gpu_memory_gb:
        print("⚠ Need model parallelism (model too large for single GPU)")
        return True
    else:
        print("❌ Need more GPUs or reduce model/batch size")
        return True

# Example
should_use_model_parallelism(
    model_size_gb=70,  # 70GB model
    gpu_memory_gb=80,   # A100
    num_gpus=4
)
```

## Types of Model Parallelism

### 1. Naive Model Parallelism (Layer-wise)

Split model layers across GPUs:

```python
import torch
import torch.nn as nn

class NaiveModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        # First half on GPU 0
        self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
        self.layer2 = nn.Linear(1000, 1000).to('cuda:0')

        # Second half on GPU 1
        self.layer3 = nn.Linear(1000, 1000).to('cuda:1')
        self.layer4 = nn.Linear(1000, 10).to('cuda:1')

    def forward(self, x):
        # Start on GPU 0
        x = x.to('cuda:0')
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        # Move to GPU 1
        x = x.to('cuda:1')
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)

        return x

# Usage
model = NaiveModelParallel()
data = torch.randn(32, 1000)
output = model(data)  # Output on cuda:1
```

**Execution Flow:**

```
Time →
┌────────────────────────────────────────┐
│ GPU 0: [Layer1][Layer2][IDLE][IDLE]   │
│                    ↓                   │
│        Transfer data to GPU 1          │
│                    ↓                   │
│ GPU 1: [IDLE][IDLE][Layer3][Layer4]   │
└────────────────────────────────────────┘

Problem: GPUs are underutilized!
- GPU 0 idle while GPU 1 works
- GPU 1 idle while GPU 0 works
- Sequential execution (bubble)
```

### 2. Pipeline Parallelism

Split model into stages and pipeline mini-batches:

```
Micro-batching Pipeline:
Time →
┌──────────────────────────────────────────────┐
│ GPU 0: [B1][B2][B3][B4][B5]...               │
│                                               │
│ GPU 1:     [B1][B2][B3][B4][B5]...           │
│                                               │
│ GPU 2:         [B1][B2][B3][B4][B5]...       │
│                                               │
│ GPU 3:             [B1][B2][B3][B4][B5]...   │
└──────────────────────────────────────────────┘

Benefits:
✓ Better GPU utilization
✓ Pipeline different micro-batches
✓ Reduced bubble overhead

Challenges:
✗ Complex implementation
✗ Still has "bubble" at start/end
✗ Requires micro-batch management
```

### 3. Tensor Parallelism

Split individual layers/tensors across GPUs:

```
Matrix Multiplication: Y = XW

Column-wise split:
┌─────────────────────────────────────┐
│ GPU 0: Y₁ = X × W₁  (first half)   │
│ GPU 1: Y₂ = X × W₂  (second half)  │
│                                     │
│ Result: Y = [Y₁, Y₂]  (concatenate)│
└─────────────────────────────────────┘

Row-wise split:
┌─────────────────────────────────────┐
│ GPU 0: Y₁ = X₁ × W  (first half)   │
│ GPU 1: Y₂ = X₂ × W  (second half)  │
│                                     │
│ Result: Y = Y₁ + Y₂  (all-reduce)  │
└─────────────────────────────────────┘

Benefits:
✓ Fine-grained parallelism
✓ All GPUs work simultaneously
✓ Good for transformer layers

Challenges:
✗ Communication overhead
✗ Requires collective ops
✗ Complex to implement
```

## Implementing Pipeline Parallelism

### Simple Pipeline with Micro-batching

```python
import torch
import torch.nn as nn
from torch.distributed import rpc

class PipelineParallelModel(nn.Module):
    def __init__(self, num_stages):
        super().__init__()
        self.num_stages = num_stages

        # Stage 0: Input layers (GPU 0)
        self.stage0 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        ).cuda(0)

        # Stage 1: Middle layers (GPU 1)
        self.stage1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        ).cuda(1)

        # Stage 2: Output layers (GPU 2)
        self.stage2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).cuda(2)

    def forward(self, x, micro_batch_size):
        """Forward with micro-batching"""
        batch_size = x.size(0)
        num_micro_batches = batch_size // micro_batch_size

        outputs = []

        for i in range(num_micro_batches):
            start = i * micro_batch_size
            end = start + micro_batch_size

            # Micro-batch
            micro_batch = x[start:end]

            # Stage 0
            x0 = self.stage0(micro_batch.cuda(0))

            # Stage 1
            x1 = self.stage1(x0.cuda(1))

            # Stage 2
            x2 = self.stage2(x1.cuda(2))

            outputs.append(x2)

        return torch.cat(outputs, dim=0)

# Better: Use torch.distributed.pipeline
from torch.distributed.pipeline.sync import Pipe

# Define model
layers = nn.Sequential(
    nn.Linear(1000, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10)
)

# Split across GPUs
model = Pipe(layers, chunks=8, checkpoint='never')
# Automatically pipelines with 8 micro-batches
```

### GPipe-style Pipeline

```python
class GPipeModel(nn.Module):
    """
    GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
    https://arxiv.org/abs/1811.06965
    """
    def __init__(self, layers_per_stage, num_stages):
        super().__init__()
        self.stages = nn.ModuleList()

        for i in range(num_stages):
            stage = nn.Sequential(
                *[nn.Linear(512, 512), nn.ReLU()] * layers_per_stage
            )
            stage = stage.cuda(i)
            self.stages.append(stage)

    def forward(self, x, num_micro_batches=4):
        # Split into micro-batches
        micro_batches = torch.chunk(x, num_micro_batches, dim=0)

        # Pipeline execution
        outputs = []
        for micro_batch in micro_batches:
            # Forward through all stages
            activations = micro_batch
            for i, stage in enumerate(self.stages):
                activations = activations.cuda(i)
                activations = stage(activations)

            outputs.append(activations)

        return torch.cat(outputs, dim=0)
```

## Implementing Tensor Parallelism

### Column-wise Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise parallelism

    Y = XW → Y = X[W₁|W₂] = [XW₁|XW₂]
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Each rank gets a portion of columns
        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size

        # Partition of weight matrix
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition)
        )

    def forward(self, x):
        # x is replicated across ranks
        # Each rank computes a portion of output
        output_parallel = torch.matmul(x, self.weight)

        # No communication needed - outputs are independent
        return output_parallel

class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise parallelism

    Y = XW → Y = [X₁|X₂]W = X₁W + X₂W (AllReduce)
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        # Each rank gets a portion of rows
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size

        # Partition of weight matrix
        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features)
        )

    def forward(self, x):
        # x is partitioned across ranks (each rank has portion)
        # Each rank computes partial output
        output_parallel = torch.matmul(x, self.weight)

        # AllReduce to get final output
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        return output_parallel
```

### Megatron-style Transformer Parallelism

```python
class ParallelAttention(nn.Module):
    """
    Tensor parallel attention from Megatron-LM
    """
    def __init__(self, hidden_size, num_heads, world_size, rank):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.world_size = world_size
        self.rank = rank

        # QKV projection - column parallel
        # Each GPU computes subset of attention heads
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            world_size,
            rank
        )

        # Output projection - row parallel
        # Reduces results across GPUs
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            world_size,
            rank
        )

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # QKV projection (column parallel)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V (each GPU has subset of heads)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Attention computation (independent per head)
        attention_output = self.compute_attention(q, k, v)

        # Output projection (row parallel with AllReduce)
        output = self.out_proj(attention_output)

        return output

    def compute_attention(self, q, k, v):
        # Standard attention math
        # Each GPU computes attention for its subset of heads
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output
```

## Using DeepSpeed for Model Parallelism

### DeepSpeed ZeRO

ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters:

```python
import deepspeed

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 32,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "zero_optimization": {
        "stage": 3,  # Partition optimizer states, gradients, and parameters
        "offload_optimizer": {
            "device": "cpu"  # Offload to CPU
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": True
    }
}

# Initialize DeepSpeed
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    training_data=train_dataset
)

# Training loop
for batch in train_loader:
    data, target = batch
    output = model_engine(data)
    loss = criterion(output, target)

    model_engine.backward(loss)
    model_engine.step()
```

**ZeRO Stages:**

```
┌──────────────────────────────────────────────────────┐
│              DeepSpeed ZeRO Stages                   │
│                                                      │
│ Stage 0 (Disabled):                                 │
│   No partitioning, standard DDP                     │
│                                                      │
│ Stage 1 (Optimizer States):                         │
│   Partition optimizer states across GPUs            │
│   Memory reduction: 4x                              │
│                                                      │
│ Stage 2 (+ Gradients):                              │
│   Partition gradients across GPUs                   │
│   Memory reduction: 8x                              │
│                                                      │
│ Stage 3 (+ Parameters):                             │
│   Partition model parameters across GPUs            │
│   Memory reduction: N (linear with GPUs)            │
│   Can train models N times larger!                  │
└──────────────────────────────────────────────────────┘
```

### DeepSpeed Pipeline Parallelism

```python
from deepspeed.pipe import PipelineModule, LayerSpec

# Define model as pipeline stages
layers = [
    LayerSpec(nn.Linear, 1000, 512),
    LayerSpec(nn.ReLU),
    LayerSpec(nn.Linear, 512, 512),
    LayerSpec(nn.ReLU),
    LayerSpec(nn.Linear, 512, 10)
]

# Create pipeline
model = PipelineModule(
    layers=layers,
    num_stages=4,  # Number of GPUs
    loss_fn=nn.CrossEntropyLoss()
)

# DeepSpeed config for pipeline
ds_config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 8,
    "pipeline": {
        "pipe_partitioned": True,
        "grad_partitioned": True
    }
}

# Initialize
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

## Memory Savings Calculation

```python
def calculate_memory_savings(
    num_params,
    num_gpus,
    use_zero_stage=0,
    use_amp=False,
    use_gradient_checkpointing=False
):
    """
    Calculate memory savings from various techniques

    Args:
        num_params: Number of model parameters
        num_gpus: Number of GPUs
        use_zero_stage: 0, 1, 2, or 3
        use_amp: Mixed precision training
        use_gradient_checkpointing: Activation checkpointing
    """
    bytes_per_param = 2 if use_amp else 4  # FP16 vs FP32

    # Base memory (single GPU, no optimizations)
    model_memory = num_params * bytes_per_param
    gradient_memory = num_params * bytes_per_param
    optimizer_memory = num_params * bytes_per_param * 2  # Adam states

    base_memory = model_memory + gradient_memory + optimizer_memory

    # With optimizations
    if use_zero_stage == 0:
        # Standard DDP - full replication
        final_memory = base_memory
    elif use_zero_stage == 1:
        # Partition optimizer states
        final_memory = model_memory + gradient_memory + (optimizer_memory / num_gpus)
    elif use_zero_stage == 2:
        # Partition gradients + optimizer
        final_memory = model_memory + (gradient_memory / num_gpus) + (optimizer_memory / num_gpus)
    elif use_zero_stage == 3:
        # Partition everything
        final_memory = (model_memory / num_gpus) + (gradient_memory / num_gpus) + (optimizer_memory / num_gpus)

    # Gradient checkpointing saves activation memory (not counted here)
    # but reduces memory by ~sqrt(num_layers) factor

    savings_ratio = base_memory / final_memory

    print(f"Base memory: {base_memory / 1024**3:.2f} GB")
    print(f"Optimized memory: {final_memory / 1024**3:.2f} GB")
    print(f"Savings: {savings_ratio:.2f}x")
    print(f"Can train model {savings_ratio:.1f}x larger!")

    return final_memory

# Example: 7B parameter model
calculate_memory_savings(
    num_params=7_000_000_000,
    num_gpus=8,
    use_zero_stage=3,
    use_amp=True,
    use_gradient_checkpointing=True
)
```

## Performance Trade-offs

```
┌──────────────────────────────────────────────────────┐
│          Model Parallelism Trade-offs                │
│                                                      │
│ Naive Model Parallel:                                │
│   Memory: ✓✓✓ (N-way reduction)                    │
│   Speed:  ✗✗✗ (sequential, low utilization)        │
│                                                      │
│ Pipeline Parallel:                                   │
│   Memory: ✓✓ (N-way reduction)                     │
│   Speed:  ✓ (better utilization, bubble overhead)  │
│                                                      │
│ Tensor Parallel:                                     │
│   Memory: ✓ (per-layer reduction)                  │
│   Speed:  ✓✓ (good utilization, communication)    │
│                                                      │
│ ZeRO Stage 3:                                        │
│   Memory: ✓✓✓ (N-way reduction)                    │
│   Speed:  ✓ (communication overhead)               │
│                                                      │
│ Hybrid (Pipeline + Tensor + Data):                  │
│   Memory: ✓✓✓ (best)                               │
│   Speed:  ✓✓ (complex but effective)              │
└──────────────────────────────────────────────────────┘
```

## Practical Exercises

### Exercise 1: Implement Naive Model Parallelism

```python
# TODO: Split this model across 2 GPUs
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 10)
        )

# Implement model parallel version
# - Split layers across GPUs
# - Handle data movement
# - Measure GPU utilization
```

### Exercise 2: Compare Memory Usage

```python
# TODO: Measure memory usage with different techniques
def compare_memory_techniques():
    """
    Measure peak memory with:
    - Standard training
    - Gradient checkpointing
    - Mixed precision
    - ZeRO Stage 2
    - ZeRO Stage 3
    """
    pass
```

### Exercise 3: Implement Simple Pipeline

```python
# TODO: Implement micro-batch pipeline
def pipeline_forward(model_stages, data, micro_batch_size):
    """
    Implement pipelined forward pass:
    - Split data into micro-batches
    - Pipeline through stages
    - Collect outputs
    """
    pass
```

## Summary

In this lesson, you learned:

1. **When to Use**: Model parallelism for models too large for single GPU
2. **Types**: Naive (layer-wise), pipeline, tensor parallelism
3. **Pipeline Parallelism**: Micro-batching to improve GPU utilization
4. **Tensor Parallelism**: Column/row splitting for fine-grained parallelism
5. **DeepSpeed ZeRO**: Partition optimizer, gradients, parameters
6. **Memory Savings**: Calculate and optimize memory usage
7. **Trade-offs**: Memory vs speed vs complexity

## Key Takeaways

- **Model parallelism enables training very large models** that don't fit on single GPU
- **Pipeline parallelism** improves on naive approach but has bubble overhead
- **Tensor parallelism** provides fine-grained parallelism with communication cost
- **DeepSpeed ZeRO** is easiest path to model parallelism in practice
- **Hybrid approaches** (data + model + pipeline) scale to largest models
- **Memory is expensive** - optimize aggressively for production

## Further Reading

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Tensor parallelism for transformers
- [GPipe Paper](https://arxiv.org/abs/1811.06965) - Pipeline parallelism
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) - Zero redundancy optimizer
- [PyTorch Pipeline Parallelism](https://pytorch.org/docs/stable/pipeline.html)

## Next Steps

In the next lesson, **GPU Memory Management & Optimization**, we'll dive deep into memory optimization techniques, profiling tools, and troubleshooting OOM errors.

---

**Ready to optimize memory usage? Let's master GPU memory management!**
