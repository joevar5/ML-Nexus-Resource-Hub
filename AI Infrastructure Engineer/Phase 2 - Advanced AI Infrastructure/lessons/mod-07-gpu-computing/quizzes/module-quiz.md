# Module 07: GPU Computing & Distributed Training - Quiz

## Instructions

- 25 multiple choice and short answer questions
- Passing score: 80% (20/25 correct)
- Time limit: 45 minutes
- Open book (can reference lesson materials)

---

## Multiple Choice Questions (1-20)

### Question 1: GPU Architecture
**What is the primary difference between GPU and CPU architecture that makes GPUs suitable for machine learning?**

A) GPUs have higher clock speeds
B) GPUs have thousands of simpler cores optimized for parallel operations
C) GPUs have larger cache memory
D) GPUs have better branch prediction

<details>
<summary>Answer</summary>
B) GPUs have thousands of simpler cores optimized for parallel operations
</details>

---

### Question 2: CUDA Cores vs Tensor Cores
**What is the main advantage of Tensor Cores over CUDA Cores for deep learning?**

A) Higher clock speeds
B) Larger memory bandwidth
C) Accelerated matrix multiply-accumulate operations with mixed precision
D) Better support for integer operations

<details>
<summary>Answer</summary>
C) Accelerated matrix multiply-accumulate operations with mixed precision (up to 8x speedup for FP16)
</details>

---

### Question 3: Memory Hierarchy
**Which GPU memory type has the FASTEST access speed?**

A) Global Memory (VRAM)
B) Shared Memory
C) Registers
D) L2 Cache

<details>
<summary>Answer</summary>
C) Registers (~1 cycle latency)
</details>

---

### Question 4: Warp Divergence
**A warp consists of how many threads that execute together in SIMT fashion?**

A) 8 threads
B) 16 threads
C) 32 threads
D) 64 threads

<details>
<summary>Answer</summary>
C) 32 threads
</details>

---

### Question 5: Mixed Precision Training
**What is the typical memory reduction when using FP16 mixed precision training compared to FP32?**

A) 1.5x
B) 2x
C) 4x
D) 8x

<details>
<summary>Answer</summary>
B) 2x (FP16 uses 2 bytes vs FP32's 4 bytes per parameter)
</details>

---

### Question 6: DataParallel vs DistributedDataParallel
**Which statement about DataParallel (DP) vs DistributedDataParallel (DDP) is TRUE?**

A) DP is faster than DDP
B) DP uses multiple processes, DDP uses a single process
C) DDP has a GPU 0 bottleneck, DP does not
D) DDP is faster and uses multiple processes

<details>
<summary>Answer</summary>
D) DDP is faster and uses multiple processes (no GIL contention, no GPU 0 bottleneck)
</details>

---

### Question 7: AllReduce Operation
**In distributed training, what does the AllReduce operation do?**

A) Sends data from rank 0 to all other ranks
B) Combines values from all ranks and distributes the result to all ranks
C) Gathers data from all ranks to rank 0
D) Scatters data from rank 0 to all ranks

<details>
<summary>Answer</summary>
B) Combines values from all ranks (e.g., sum, average) and distributes the result to all ranks
</details>

---

### Question 8: Data Parallelism
**In data parallelism, what is replicated across GPUs?**

A) The data
B) The model
C) The optimizer
D) The loss function

<details>
<summary>Answer</summary>
B) The model is replicated; data is split across GPUs
</details>

---

### Question 9: Model Parallelism
**When is model parallelism necessary instead of data parallelism?**

A) When you want faster training
B) When the model is too large to fit on a single GPU
C) When you have a large dataset
D) When you want better accuracy

<details>
<summary>Answer</summary>
B) When the model is too large to fit on a single GPU's memory
</details>

---

### Question 10: Gradient Checkpointing
**What is the memory vs speed trade-off of gradient checkpointing?**

A) More memory, faster training
B) Less memory, faster training
C) More memory, slower training
D) Less memory, slower training

<details>
<summary>Answer</summary>
D) Less memory (store ~√N activations instead of N), slower training (~33% slower due to recomputation)
</details>

---

### Question 11: Ring AllReduce
**What is the communication complexity (bandwidth per GPU) of Ring AllReduce?**

A) O(N) where N is number of GPUs
B) O(N²)
C) O(log N)
D) O(1) - constant

<details>
<summary>Answer</summary>
D) O(1) - bandwidth requirement per GPU is constant regardless of GPU count (this is why it scales well)
</details>

---

### Question 12: Pipeline Parallelism
**What is the main overhead/inefficiency in pipeline parallelism?**

A) Communication overhead
B) Memory fragmentation
C) Pipeline "bubble" at start and end
D) Gradient synchronization

<details>
<summary>Answer</summary>
C) Pipeline "bubble" where some GPUs are idle at the start and end of each batch
</details>

---

### Question 13: ZeRO Stage 3
**What does DeepSpeed ZeRO Stage 3 partition across GPUs?**

A) Only optimizer states
B) Optimizer states and gradients
C) Optimizer states, gradients, and model parameters
D) Only model parameters

<details>
<summary>Answer</summary>
C) Optimizer states, gradients, and model parameters (maximum memory reduction)
</details>

---

### Question 14: Flash Attention
**What is the memory complexity of Flash Attention compared to standard attention?**

A) O(N²) same as standard
B) O(N log N)
C) O(N)
D) O(√N)

<details>
<summary>Answer</summary>
C) O(N) - Flash Attention avoids materializing the O(N²) attention matrix
</details>

---

### Question 15: GPU Utilization
**If GPU utilization is consistently <50%, what is the MOST LIKELY bottleneck?**

A) GPU compute is too slow
B) Data loading or CPU preprocessing
C) Not enough GPU memory
D) Network bandwidth

<details>
<summary>Answer</summary>
B) Data loading or CPU preprocessing (GPU is waiting for data)
</details>

---

### Question 16: Memory During Training
**For a model with M parameters trained with Adam optimizer in FP32, approximately how much memory is needed (excluding activations)?**

A) M bytes
B) 2M bytes
C) 4M bytes
D) 16M bytes

<details>
<summary>Answer</summary>
D) ~16M bytes = 4M (parameters) + 4M (gradients) + 8M (Adam states: momentum + variance)
</details>

---

### Question 17: Pinned Memory
**What is the main benefit of using pinned memory in PyTorch DataLoader?**

A) Faster CPU processing
B) Larger batch sizes
C) Faster CPU-to-GPU memory transfers
D) Lower memory usage

<details>
<summary>Answer</summary>
C) Faster CPU-to-GPU memory transfers (2-3x faster via DMA)
</details>

---

### Question 18: DistributedSampler
**Why is DistributedSampler necessary in DDP training?**

A) To ensure each GPU processes different data samples
B) To increase batch size
C) To improve data loading speed
D) To reduce memory usage

<details>
<summary>Answer</summary>
A) To ensure each GPU processes different data samples (no overlap)
</details>

---

### Question 19: Gradient Accumulation
**If batch_size=16, accumulation_steps=4, world_size=8, what is the effective batch size?**

A) 16
B) 64
C) 128
D) 512

<details>
<summary>Answer</summary>
D) 512 (16 × 4 × 8 = 512)
</details>

---

### Question 20: Tensor Parallelism
**In tensor parallelism with column-wise weight splitting, what operation is needed after computation?**

A) AllReduce (sum)
B) Concatenation
C) Broadcast
D) Scatter

<details>
<summary>Answer</summary>
B) Concatenation - each GPU computes a portion of columns, results are concatenated
</details>

---

## Short Answer Questions (21-25)

### Question 21: GPU Memory Breakdown
**List the 4 main components that consume GPU memory during training and give approximate size for each (assuming M parameters).**

<details>
<summary>Answer</summary>

1. Model Parameters: M parameters × 4 bytes (FP32) = 4M bytes
2. Gradients: M × 4 bytes = 4M bytes
3. Optimizer States (Adam): M × 8 bytes = 8M bytes (momentum + variance)
4. Activations: Varies with batch size and sequence length (can be largest component)

Total: ~16M + activations
</details>

---

### Question 22: Profiling Workflow
**Describe the steps you would take to diagnose why a training job has low GPU utilization (<50%).**

<details>
<summary>Answer</summary>

1. Monitor GPU utilization with `nvidia-smi` or similar
2. Profile data loading time vs compute time
3. Check batch size (too small = underutilization)
4. Profile with PyTorch Profiler or Nsight Systems
5. Look for CPU-GPU synchronization points (.item(), .cpu())
6. Check DataLoader configuration (num_workers, pin_memory)
7. Verify no excessive disk I/O or preprocessing
8. Consider increasing batch size if memory allows

Key insight: GPU util <50% usually indicates data loading bottleneck, not compute.
</details>

---

### Question 23: DDP Setup
**Write the minimal code to set up a DistributedDataParallel model in PyTorch. Include process group initialization and model wrapping.**

<details>
<summary>Answer</summary>

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Setup
setup(rank, world_size)

# Create model and move to GPU
model = MyModel().to(rank)

# Wrap with DDP
ddp_model = DDP(model, device_ids=[rank])

# Use ddp_model for training
# ...

# Cleanup
dist.destroy_process_group()
```
</details>

---

### Question 24: Memory Optimization Techniques
**List 5 techniques to reduce GPU memory usage during training and briefly explain each.**

<details>
<summary>Answer</summary>

1. **Mixed Precision (FP16/BF16)**: Use 16-bit floats instead of 32-bit (2x reduction)

2. **Gradient Checkpointing**: Recompute activations during backward pass instead of storing (√N memory)

3. **Gradient Accumulation**: Simulate larger batch with smaller physical batches

4. **Reduce Batch Size**: Smaller batches use less activation memory

5. **ZeRO Optimizer**: Partition optimizer states, gradients, parameters across GPUs

Bonus:
- Flash Attention: O(N) vs O(N²) for attention
- CPU Offloading: Move unused layers to CPU
- Model Parallelism: Split model across GPUs
</details>

---

### Question 25: Scaling Efficiency
**Explain the difference between strong scaling and weak scaling in distributed training. Which is more commonly used in deep learning and why?**

<details>
<summary>Answer</summary>

**Strong Scaling:**
- Fixed total workload
- More GPUs = same dataset processed faster
- Example: 1000 samples on 1 GPU vs 1000 samples on 4 GPUs
- Ideal: 4x speedup with 4 GPUs
- Issue: Communication overhead limits scaling

**Weak Scaling:**
- Fixed per-GPU workload
- More GPUs = larger total batch size
- Example: 1000 samples per GPU × N GPUs
- Ideal: Same time regardless of GPU count
- Maintains GPU efficiency

**Most common in DL: Weak scaling**

Why:
- Maintains good GPU utilization
- Larger batches can improve convergence (with proper LR scaling)
- Avoids communication becoming bottleneck
- Better matches how we scale training in practice

Trade-off: May need to adjust learning rate for larger effective batch sizes.
</details>

---

## Scoring Guide

- **20-25 correct**: Excellent! Strong understanding of GPU computing and distributed training
- **16-19 correct**: Good! Review topics you struggled with
- **12-15 correct**: Pass. Review lessons and try again
- **<12 correct**: Please review all lessons before retaking

---

## Next Steps

After passing the quiz:
1. Complete hands-on exercises in Module 07
2. Implement a small distributed training project
3. Proceed to Module 08: Monitoring & Observability

---

**Good luck!**
