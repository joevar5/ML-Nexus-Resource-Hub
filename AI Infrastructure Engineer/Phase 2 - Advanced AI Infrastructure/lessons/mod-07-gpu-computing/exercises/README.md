# Module 07: GPU Computing & Distributed Training - Hands-on Exercises

## Overview

This module contains 5 hands-on labs that progressively build your GPU computing and distributed training skills. Each lab includes starter code, detailed instructions, and solution hints.

**Total Time**: 15-20 hours
**Prerequisites**: Completed Module 07 lessons, GPU access (cloud or local)

---

## Lab 1: GPU Basics & Profiling (3-4 hours)

### Objectives
- Set up GPU development environment
- Profile GPU vs CPU performance
- Understand GPU memory management
- Use PyTorch Profiler

### Tasks

#### Task 1.1: Environment Setup
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install profiling tools
pip install nvidia-ml-py3 torch-tb-profiler
```

#### Task 1.2: GPU vs CPU Benchmark
Create a script that:
1. Compares matrix multiplication performance (CPU vs GPU)
2. Tests different matrix sizes: 128, 512, 1024, 2048, 4096
3. Plots speedup vs matrix size
4. Calculates achieved TFLOPS

**Starter Code**:
```python
import torch
import time
import matplotlib.pyplot as plt

def benchmark_matmul(device, size, iterations=100):
    """Benchmark matrix multiplication"""
    # TODO: Implement benchmark
    # - Create random matrices
    # - Warmup runs
    # - Timed runs with synchronization
    # - Calculate TFLOPS
    pass

# Test different sizes
sizes = [128, 512, 1024, 2048, 4096]
# TODO: Run benchmarks and plot results
```

**Deliverables**:
- Benchmark script
- Performance plot (speedup vs matrix size)
- Report analyzing results

---

#### Task 1.3: Memory Profiling
Profile memory usage during model training:
1. Track memory at each training stage
2. Identify memory bottlenecks
3. Compare FP32 vs FP16 memory usage

**Starter Code**:
```python
def profile_training_memory():
    """Profile GPU memory during training"""
    model = create_model().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # TODO: Implement memory profiling
    # - Track memory before/after each stage
    # - Create visualization
    # - Test with different precisions
    pass
```

---

## Lab 2: Mixed Precision Training (3-4 hours)

### Objectives
- Implement Automatic Mixed Precision (AMP)
- Compare training speed and memory usage
- Handle numerical stability issues
- Validate accuracy with FP16

### Tasks

#### Task 2.1: AMP Implementation
Convert a training loop to use mixed precision:

**Starter Code**:
```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(model, train_loader, epochs=10):
    """Implement AMP training"""
    scaler = GradScaler()

    for epoch in range(epochs):
        for data, target in train_loader:
            # TODO: Implement AMP training loop
            # - Forward pass with autocast
            # - Loss computation
            # - Backward with scaling
            # - Optimizer step
            pass

# TODO: Compare FP32 vs FP16:
# - Training speed
# - Memory usage
# - Final accuracy
```

#### Task 2.2: Numerical Stability
Test and fix numerical stability issues:
1. Train a model that's prone to instability
2. Detect NaN/Inf values
3. Implement loss scaling adjustments
4. Compare FP16 vs BF16

**Expected Findings**:
- ~2x speedup with FP16 on Tensor Core GPUs
- ~50% memory reduction
- <1% accuracy difference

---

## Lab 3: Distributed Data Parallel (4-5 hours)

### Objectives
- Implement DistributedDataParallel (DDP)
- Set up multi-GPU training
- Measure scaling efficiency
- Debug common DDP issues

### Tasks

#### Task 3.1: Single-Node Multi-GPU Setup
Implement DDP training on a single machine:

**Starter Code**:
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize distributed environment"""
    # TODO: Implement setup
    pass

def cleanup():
    """Cleanup distributed environment"""
    # TODO: Implement cleanup
    pass

def train(rank, world_size):
    """Main training function"""
    print(f"Running on rank {rank}")
    setup(rank, world_size)

    # TODO: Implement DDP training
    # - Create model and wrap with DDP
    # - Create DistributedSampler
    # - Training loop
    # - Save checkpoints (rank 0 only)

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

#### Task 3.2: Scaling Efficiency Analysis
Measure and analyze scaling efficiency:
1. Train with 1, 2, 4 GPUs (if available)
2. Measure training time and throughput
3. Calculate speedup and efficiency
4. Identify communication bottlenecks

**Deliverables**:
- DDP training script
- Scaling efficiency plot
- Analysis of bottlenecks

---

#### Task 3.3: Gradient Accumulation
Implement gradient accumulation with DDP:

**Starter Code**:
```python
def train_with_accumulation(model, train_loader, accumulation_steps=4):
    """DDP training with gradient accumulation"""
    # TODO: Implement
    # - Use model.no_sync() for intermediate steps
    # - Synchronize on final step
    # - Track effective batch size
    pass
```

**Test**: Verify that accumulation_steps=4 with batch_size=8 gives same results as batch_size=32.

---

## Lab 4: Memory Optimization (3-4 hours)

### Objectives
- Implement gradient checkpointing
- Optimize large model training
- Use Flash Attention
- Debug OOM errors

### Tasks

#### Task 4.1: Gradient Checkpointing
Add checkpointing to reduce memory:

**Starter Code**:
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    """Transformer with gradient checkpointing"""
    def __init__(self, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(num_layers)
        ])

    def forward(self, x):
        # TODO: Implement forward with checkpointing
        # - Checkpoint every N layers
        # - Measure memory savings
        pass
```

**Compare**:
- Peak memory: with vs without checkpointing
- Training speed: overhead from recomputation
- Find optimal checkpoint frequency

---

#### Task 4.2: Flash Attention Integration
Replace standard attention with Flash Attention:

```python
# Before: Standard attention (O(N²) memory)
def standard_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    attn_weights = torch.softmax(scores / np.sqrt(d_k), dim=-1)
    return torch.matmul(attn_weights, V)

# TODO: Implement Flash Attention
# - Install: pip install flash-attn
# - Replace standard attention
# - Benchmark memory and speed
```

**Expected Results**:
- 10-20x memory reduction for long sequences
- 2-4x speedup
- Exact same outputs

---

#### Task 4.3: OOM Debugging
Debug and fix an OOM-prone training script:

**Intentionally Broken Code**:
```python
# This code will OOM - fix it!
def train_large_model():
    model = VeryLargeModel(num_layers=48, hidden_size=4096).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in large_dataloader:  # batch_size=64
            data, target = batch
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)

            # Accumulate losses (MEMORY LEAK!)
            all_losses.append(loss)

            loss.backward()
            optimizer.step()

# TODO: Fix all OOM issues:
# - Reduce batch size
# - Add gradient accumulation
# - Enable mixed precision
# - Add gradient checkpointing
# - Fix memory leaks
# - Use torch.no_grad() for validation
```

---

## Lab 5: Production GPU Pipeline (4-5 hours)

### Objectives
- Build production-ready training pipeline
- Implement monitoring and logging
- Add checkpointing and resuming
- Optimize end-to-end performance

### Tasks

#### Task 5.1: Complete Training Pipeline
Build a production pipeline with:
- DDP multi-GPU training
- Mixed precision (AMP)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
- GPU metrics monitoring

**Template Structure**:
```python
class ProductionTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        # TODO: Initialize all components
        # - DDP setup
        # - AMP scaler
        # - Optimizers
        # - Schedulers
        # - Loggers
        pass

    def train_epoch(self, epoch):
        # TODO: Training loop with all optimizations
        pass

    def validate(self):
        # TODO: Validation with torch.no_grad()
        pass

    def save_checkpoint(self, epoch):
        # TODO: Save model, optimizer, scheduler states
        pass

    def load_checkpoint(self, path):
        # TODO: Resume from checkpoint
        pass
```

---

#### Task 5.2: Performance Profiling
Profile the complete pipeline:
1. Use PyTorch Profiler to identify bottlenecks
2. Optimize data loading
3. Tune batch size
4. Achieve >80% GPU utilization

**Profiling Script**:
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    # TODO: Run training for 100 steps
    # - Add record_function markers
    # - Export to TensorBoard
    # - Analyze results
    pass

# TODO: Create optimization report
```

---

#### Task 5.3: Monitoring Dashboard
Create a real-time monitoring dashboard:

```python
class GPUMonitor:
    """Real-time GPU monitoring"""
    def __init__(self):
        # TODO: Initialize monitoring
        # - Setup Prometheus metrics
        # - TensorBoard writers
        # - nvidia-ml-py
        pass

    def log_metrics(self, step, metrics):
        # TODO: Log all metrics
        # - Training loss/accuracy
        # - GPU utilization
        # - GPU memory
        # - Temperature
        # - Throughput
        pass

# TODO: Integrate with training loop
# - Log every N steps
# - Create visualizations
# - Set up alerts for issues
```

---

## Evaluation Criteria

### Lab 1: GPU Basics (20 points)
- [5] Correct benchmark implementation
- [5] Accurate performance measurements
- [5] Memory profiling completeness
- [5] Quality of analysis/report

### Lab 2: Mixed Precision (20 points)
- [5] Correct AMP implementation
- [5] Proper gradient scaling
- [5] Numerical stability handling
- [5] Performance comparison

### Lab 3: Distributed Training (25 points)
- [8] Correct DDP setup
- [7] Scaling efficiency analysis
- [5] Gradient accumulation
- [5] Checkpointing implementation

### Lab 4: Memory Optimization (20 points)
- [5] Gradient checkpointing
- [5] Flash Attention integration
- [5] OOM fixes
- [5] Memory analysis

### Lab 5: Production Pipeline (15 points)
- [5] Complete pipeline implementation
- [5] Performance profiling
- [5] Monitoring dashboard

**Total: 100 points**
**Passing: 70 points**

---

## Submission

Create a Git repository with:
```
gpu-computing-labs/
├── lab1-gpu-basics/
│   ├── benchmark.py
│   ├── memory_profiling.py
│   ├── results/
│   └── REPORT.md
├── lab2-mixed-precision/
│   ├── amp_training.py
│   ├── stability_tests.py
│   └── REPORT.md
├── lab3-distributed/
│   ├── ddp_training.py
│   ├── scaling_analysis.py
│   └── REPORT.md
├── lab4-memory/
│   ├── gradient_checkpointing.py
│   ├── flash_attention.py
│   ├── oom_fixes.py
│   └── REPORT.md
├── lab5-production/
│   ├── trainer.py
│   ├── monitor.py
│   ├── profiling.py
│   └── REPORT.md
└── README.md
```

## Resources

- Sample datasets: Use CIFAR-10 or MNIST for quick testing
- Cloud GPUs: Google Colab, Kaggle, AWS, GCP
- Reference implementations: See `solutions/` directory (available after submission deadline)

---

## Getting Help

- **Office Hours**: Tuesdays 2-4 PM
- **Discussion Forum**: [Link]
- **Slack Channel**: #gpu-computing-labs
- **TA Email**: gpu-labs@example.com

---

## Tips for Success

1. **Start Early**: These labs take time
2. **Test Incrementally**: Don't write everything at once
3. **Use Profilers**: Measure before optimizing
4. **Read Error Messages**: They're helpful!
5. **Ask Questions**: Use office hours and forums
6. **Document**: Write clear reports
7. **Save Checkpoints**: Don't lose hours of training

---

**Ready to become a GPU expert? Let's get started!**
