# Module 07: GPU Computing — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-07-gpu-computing/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the primary difference between GPU and CPU architecture that makes GPUs suitable for machine learning?

**Answer:** B) GPUs have thousands of simpler cores optimized for parallel operations

**Explanation:**
GPUs are throughput-oriented processors built around massive Single-Instruction-Multiple-Thread (SIMT) parallelism, with thousands of small ALUs that can perform the same operation across many data elements simultaneously. CPUs, by contrast, are latency-oriented with a few powerful cores. Deep learning workloads (matrix multiplies, convolutions, elementwise ops) decompose naturally into parallel work, which maps onto a GPU's wide compute fabric.

**Common Mistakes:**
- Choosing A) Higher clock speeds: CPUs typically have higher clock speeds than GPUs; raw frequency is not the GPU advantage.
- Choosing C) Larger cache memory: CPUs actually have larger per-core caches; GPUs trade cache size for more compute.
- Choosing D) Better branch prediction: CPUs have far more sophisticated branch prediction; warp divergence is a known GPU weakness.

**Related Material:** `lessons/mod-07-gpu-computing/01-introduction-gpu-computing.md`

---

## Question 2
**Q:** What is the main advantage of Tensor Cores over CUDA Cores for deep learning?

**Answer:** C) Accelerated matrix multiply-accumulate operations with mixed precision

**Explanation:**
Tensor Cores are specialized hardware units that perform a fused matrix-multiply-accumulate (D = A*B + C) on small tile sizes in a single clock, typically with FP16/BF16/TF32 inputs and FP32 accumulation. This delivers up to ~8x throughput over general-purpose CUDA cores doing the same MMA. They are the reason mixed-precision training is so much faster on modern NVIDIA GPUs.

**Common Mistakes:**
- Choosing A) Higher clock speeds: Tensor Cores share the SM clock; the speedup comes from throughput per cycle, not frequency.
- Choosing B) Larger memory bandwidth: Tensor Cores do not change HBM bandwidth; they consume on-chip operands from registers/shared memory.
- Choosing D) Better support for integer operations: Tensor Cores do support INT8, but their headline benefit is FP16/BF16 GEMM acceleration for training.

**Related Material:** `lessons/mod-07-gpu-computing/01-introduction-gpu-computing.md`

---

## Question 3
**Q:** Which GPU memory type has the FASTEST access speed?

**Answer:** C) Registers (~1 cycle latency)

**Explanation:**
Registers sit directly inside each SM and are the fastest storage in the GPU memory hierarchy, with effectively single-cycle latency and per-thread private scope. Shared memory is the next tier (low tens of cycles), followed by L2 cache, and finally global HBM (hundreds of cycles). Kernel performance often hinges on keeping hot data in registers and shared memory.

**Common Mistakes:**
- Choosing A) Global Memory (VRAM): This is the slowest tier with hundreds of cycles of latency.
- Choosing B) Shared Memory: Fast, but still slower than registers and shared across a thread block.
- Choosing D) L2 Cache: Sits between SMs and global memory; faster than HBM but much slower than registers.

**Related Material:** `lessons/mod-07-gpu-computing/01-introduction-gpu-computing.md`

---

## Question 4
**Q:** A warp consists of how many threads that execute together in SIMT fashion?

**Answer:** C) 32 threads

**Explanation:**
On all current NVIDIA architectures, a warp is exactly 32 threads that execute the same instruction in lockstep under SIMT. Block sizes are typically chosen as multiples of 32 so that warps are fully populated, avoiding wasted lanes. Branch divergence within a warp serializes the divergent paths, which is why warp size matters for performance.

**Common Mistakes:**
- Choosing A) 8 threads: Too small; this is closer to an AVX SIMD lane width on CPUs.
- Choosing B) 16 threads: Half a warp; sometimes referenced as a "half-warp" in older architectures but not the warp size itself.
- Choosing D) 64 threads: AMD GCN wavefronts are 64, but CUDA warps are 32.

**Related Material:** `lessons/mod-07-gpu-computing/02-cuda-programming-fundamentals.md`

---

## Question 5
**Q:** What is the typical memory reduction when using FP16 mixed precision training compared to FP32?

**Answer:** B) 2x (FP16 uses 2 bytes vs FP32's 4 bytes per parameter)

**Explanation:**
FP16 and BF16 are 16-bit formats (2 bytes), while FP32 is 32-bit (4 bytes), so storing tensors in half precision halves their footprint. Real-world mixed-precision training keeps a master FP32 copy of weights for stability, so the overall memory savings depend on what you measure, but per-tensor activation/gradient memory is cut roughly 2x. Combined with Tensor Cores, this also yields major speedups.

**Common Mistakes:**
- Choosing A) 1.5x: Underestimates the byte ratio; FP16 is half of FP32, not three-quarters.
- Choosing C) 4x: This would correspond to going from FP32 to INT8/FP8, not FP16.
- Choosing D) 8x: Would require 4-bit quantization, used only for inference.

**Related Material:** `lessons/mod-07-gpu-computing/03-pytorch-gpu-acceleration.md`

---

## Question 6
**Q:** Which statement about DataParallel (DP) vs DistributedDataParallel (DDP) is TRUE?

**Answer:** D) DDP is faster and uses multiple processes (no GIL contention, no GPU 0 bottleneck)

**Explanation:**
`DataParallel` runs in a single Python process and scatters inputs/gathers outputs through GPU 0, which becomes both a memory and compute bottleneck and is throttled by Python's GIL. `DistributedDataParallel` spawns one process per GPU and uses NCCL AllReduce to average gradients in a ring, avoiding both bottlenecks. PyTorch recommends DDP for all multi-GPU training, even on a single node.

**Common Mistakes:**
- Choosing A) DP is faster than DDP: Reverses the truth; DP is consistently slower due to scatter/gather and the GIL.
- Choosing B) DP uses multiple processes, DDP uses a single process: Exactly backwards.
- Choosing C) DDP has a GPU 0 bottleneck, DP does not: The bottleneck belongs to DP; DDP uses a symmetric ring.

**Related Material:** `lessons/mod-07-gpu-computing/05-multi-gpu-training-strategies.md`

---

## Question 7
**Q:** In distributed training, what does the AllReduce operation do?

**Answer:** B) Combines values from all ranks and distributes the result to all ranks

**Explanation:**
AllReduce performs an associative reduction (typically sum or average) over a tensor that exists on every rank, then broadcasts the reduced result back to every rank so they all end up with identical values. In DDP it is used to average gradients across workers after each backward pass. It is logically equivalent to Reduce followed by Broadcast, but implementations like Ring or Tree AllReduce are far more bandwidth-efficient.

**Common Mistakes:**
- Choosing A) Sends data from rank 0 to all other ranks: That is Broadcast, not AllReduce.
- Choosing C) Gathers data from all ranks to rank 0: That is Reduce (or Gather), which leaves only rank 0 with the result.
- Choosing D) Scatters data from rank 0 to all ranks: That is Scatter, which splits data, not combines it.

**Related Material:** `lessons/mod-07-gpu-computing/04-distributed-training-fundamentals.md`

---

## Question 8
**Q:** In data parallelism, what is replicated across GPUs?

**Answer:** B) The model is replicated; data is split across GPUs

**Explanation:**
Data parallelism puts a full copy of the model (and its optimizer state) on each GPU and splits the global batch into shards, one per worker. Each GPU computes the forward and backward pass on its shard, then gradients are AllReduced so every replica sees the same averaged gradient before stepping the optimizer. This keeps the replicas in sync and is the workhorse scaling pattern for models that still fit on one GPU.

**Common Mistakes:**
- Choosing A) The data: The data is split (sharded), not replicated; that is the whole point of "data" parallelism.
- Choosing C) The optimizer: Optimizer state is replicated as a consequence of the model being replicated, but the defining property is model replication.
- Choosing D) The loss function: The loss function is just code; it isn't the unit of replication.

**Related Material:** `lessons/mod-07-gpu-computing/05-multi-gpu-training-strategies.md`

---

## Question 9
**Q:** When is model parallelism necessary instead of data parallelism?

**Answer:** B) When the model is too large to fit on a single GPU's memory

**Explanation:**
Data parallelism requires every GPU to hold a full copy of the model, gradients, and optimizer state. Once the model exceeds the memory of a single device, you must shard the model itself — via tensor parallelism, pipeline parallelism, or ZeRO-3 — so that each GPU only stores a slice. Model parallelism is fundamentally about fitting the model, not about going faster.

**Common Mistakes:**
- Choosing A) When you want faster training: Model parallelism usually adds communication overhead and is often slower per step than pure data parallelism.
- Choosing C) When you have a large dataset: A large dataset is handled by data parallelism (more shards), not model parallelism.
- Choosing D) When you want better accuracy: Parallelism strategy does not change model accuracy.

**Related Material:** `lessons/mod-07-gpu-computing/06-model-pipeline-parallelism.md`

---

## Question 10
**Q:** What is the memory vs speed trade-off of gradient checkpointing?

**Answer:** D) Less memory (store ~√N activations instead of N), slower training (~33% slower due to recomputation)

**Explanation:**
Gradient checkpointing saves only a subset of activations during the forward pass and recomputes the rest on demand during backprop. This drops activation memory from O(N) to roughly O(√N) layers but adds an extra partial forward pass per backward, typically ~30% step-time overhead. It is a key technique when activation memory dominates, especially for long sequences or deep transformers.

**Common Mistakes:**
- Choosing A) More memory, faster training: Wrong on both axes; checkpointing reduces memory and slows training.
- Choosing B) Less memory, faster training: Correct on memory, wrong on speed — recomputation costs time.
- Choosing C) More memory, slower training: Inverted memory trade-off; checkpointing exists specifically to save memory.

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 11
**Q:** What is the communication complexity (bandwidth per GPU) of Ring AllReduce?

**Answer:** D) O(1) - constant

**Explanation:**
Ring AllReduce splits the gradient tensor into N chunks and uses a reduce-scatter followed by an all-gather around a ring of N GPUs. Each GPU sends and receives 2*(N-1)/N * tensor_size bytes total — which approaches a constant (2 * tensor_size) as N grows, independent of GPU count. That bandwidth-optimality is precisely why Horovod and NCCL popularized the ring pattern for large-scale training.

**Common Mistakes:**
- Choosing A) O(N): Describes a naive parameter-server / Reduce + Broadcast pattern where the central node's bandwidth grows linearly.
- Choosing B) O(N²): That would be an all-to-all without aggregation; far worse than ring.
- Choosing C) O(log N): Closer to a tree AllReduce's latency term, not the per-GPU bandwidth of a ring.

**Related Material:** `lessons/mod-07-gpu-computing/04-distributed-training-fundamentals.md`

---

## Question 12
**Q:** What is the main overhead/inefficiency in pipeline parallelism?

**Answer:** C) Pipeline "bubble" at start and end

**Explanation:**
In pipeline parallelism, layers are partitioned across GPUs and micro-batches flow through the stages. At the start of a batch the later stages are idle waiting for the first micro-batch to reach them, and at the end the early stages are idle waiting for the last micro-batch to finish — these idle periods are the pipeline "bubble." Increasing the number of micro-batches (e.g., GPipe, 1F1B schedules) reduces but never fully eliminates the bubble.

**Common Mistakes:**
- Choosing A) Communication overhead: Inter-stage sends exist, but they are small (just activations between adjacent stages) compared to the bubble cost.
- Choosing B) Memory fragmentation: Not a pipeline-specific issue.
- Choosing D) Gradient synchronization: Pipeline parallelism does not require an AllReduce across stages on the partitioned layers.

**Related Material:** `lessons/mod-07-gpu-computing/06-model-pipeline-parallelism.md`

---

## Question 13
**Q:** What does DeepSpeed ZeRO Stage 3 partition across GPUs?

**Answer:** C) Optimizer states, gradients, and model parameters

**Explanation:**
ZeRO (Zero Redundancy Optimizer) progressively shards training state across data-parallel workers: Stage 1 partitions optimizer states, Stage 2 adds gradient partitioning, and Stage 3 additionally partitions model parameters themselves. With Stage 3, each GPU only holds 1/N of every tensor and parameters are gathered on demand for forward/backward, giving the maximum memory reduction at the cost of extra communication.

**Common Mistakes:**
- Choosing A) Only optimizer states: That is ZeRO Stage 1.
- Choosing B) Optimizer states and gradients: That is ZeRO Stage 2.
- Choosing D) Only model parameters: ZeRO always partitions optimizer state first; partitioning only parameters is not a ZeRO stage.

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 14
**Q:** What is the memory complexity of Flash Attention compared to standard attention?

**Answer:** C) O(N)

**Explanation:**
Standard attention materializes the full N×N attention score matrix in HBM, which is O(N²) memory and bandwidth-bound. Flash Attention tiles Q, K, V into blocks that fit in SRAM and computes softmax-weighted sums incrementally with online softmax, never writing the full attention matrix to global memory. This drops memory to O(N) and also speeds up the kernel by being IO-aware.

**Common Mistakes:**
- Choosing A) O(N²) same as standard: Misses the entire point of Flash Attention; it explicitly avoids the O(N²) matrix.
- Choosing B) O(N log N): No log factor appears in the analysis.
- Choosing D) O(√N): Confuses Flash Attention with gradient checkpointing's √N activation savings.

**Related Material:** `lessons/mod-07-gpu-computing/08-advanced-gpu-optimization.md`

---

## Question 15
**Q:** If GPU utilization is consistently <50%, what is the MOST LIKELY bottleneck?

**Answer:** B) Data loading or CPU preprocessing

**Explanation:**
Persistently low GPU utilization almost always means the GPU is starved — it finishes its batch and then waits for the next one. The usual culprits are slow disk reads, single-worker DataLoaders, heavy CPU augmentation, missing `pin_memory`, or `.cpu()`/`.item()` syncs in the training loop. Fix it by overlapping IO with compute: more `num_workers`, `pin_memory=True`, prefetching, and moving augmentations to GPU when possible.

**Common Mistakes:**
- Choosing A) GPU compute is too slow: If compute were the bottleneck you would see high (≈100%) utilization, not low.
- Choosing C) Not enough GPU memory: OOM manifests as crashes, not as low utilization.
- Choosing D) Network bandwidth: Only relevant in multi-node training, and even there it typically shows up as high "comm" time, not idle GPUs.

**Related Material:** `lessons/mod-07-gpu-computing/08-advanced-gpu-optimization.md`

---

## Question 16
**Q:** For a model with M parameters trained with Adam optimizer in FP32, approximately how much memory is needed (excluding activations)?

**Answer:** D) 16M bytes

**Explanation:**
With FP32 (4 bytes/element), parameters take 4M bytes, gradients another 4M bytes, and Adam keeps two moment buffers (m and v) at 4M bytes each, adding 8M bytes — totaling 16M bytes before activations. That is why a 7B-parameter model needs roughly 112 GB just for weights+grads+optimizer state in FP32, and is the motivation for mixed precision and ZeRO sharding.

**Common Mistakes:**
- Choosing A) M bytes: Counts only one byte per parameter, which assumes INT8 and ignores grads/optimizer entirely.
- Choosing B) 2M bytes: Forgets that FP32 is 4 bytes and ignores gradients and optimizer state.
- Choosing C) 4M bytes: Counts only the parameters; misses gradients (4M) and Adam moments (8M).

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 17
**Q:** What is the main benefit of using pinned memory in PyTorch DataLoader?

**Answer:** C) Faster CPU-to-GPU memory transfers (2-3x faster via DMA)

**Explanation:**
Pinned (page-locked) memory cannot be paged out by the OS, which allows the GPU's DMA engine to copy directly from host RAM without an intermediate staging buffer. Combined with `non_blocking=True`, this lets the H2D transfer overlap with GPU compute. The result is faster and more deterministic data transfer, typically 2–3x over pageable memory.

**Common Mistakes:**
- Choosing A) Faster CPU processing: Pinning does not change CPU throughput; it only affects the host→device path.
- Choosing B) Larger batch sizes: Batch size is bounded by GPU memory, not host pinning.
- Choosing D) Lower memory usage: Pinned memory actually constrains the OS more and can increase pressure on host RAM.

**Related Material:** `lessons/mod-07-gpu-computing/03-pytorch-gpu-acceleration.md`

---

## Question 18
**Q:** Why is DistributedSampler necessary in DDP training?

**Answer:** A) To ensure each GPU processes different data samples (no overlap)

**Explanation:**
By default every DDP rank would create the same DataLoader and iterate the dataset identically, so every GPU would train on the same samples — defeating the purpose of data parallelism. `DistributedSampler` deterministically partitions indices by rank so each worker sees a disjoint shard of the dataset per epoch. You also need to call `sampler.set_epoch(epoch)` so shuffling is consistent across ranks but varies between epochs.

**Common Mistakes:**
- Choosing B) To increase batch size: Effective batch size grows because of having multiple GPUs, not because of the sampler.
- Choosing C) To improve data loading speed: That is what `num_workers`/`pin_memory` do; the sampler is about correctness.
- Choosing D) To reduce memory usage: The sampler does not change per-GPU memory footprint.

**Related Material:** `lessons/mod-07-gpu-computing/05-multi-gpu-training-strategies.md`

---

## Question 19
**Q:** If batch_size=16, accumulation_steps=4, world_size=8, what is the effective batch size?

**Answer:** D) 512

**Explanation:**
Effective batch size = per-GPU micro-batch × accumulation steps × number of GPUs = 16 × 4 × 8 = 512. Gradient accumulation simulates a larger batch by deferring the optimizer step, and data parallelism multiplies it again across workers. This is the number to use when scaling the learning rate (e.g., linear or square-root scaling rules).

**Common Mistakes:**
- Choosing A) 16: Only the per-GPU micro-batch; ignores accumulation and world size.
- Choosing B) 64: Multiplies micro-batch by accumulation but forgets the 8 GPUs.
- Choosing C) 128: Multiplies micro-batch by world size but forgets the accumulation factor.

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 20
**Q:** In tensor parallelism with column-wise weight splitting, what operation is needed after computation?

**Answer:** B) Concatenation

**Explanation:**
In Megatron-style column-parallel linear layers, the weight matrix is split along its output (column) dimension and each GPU computes a slice of the output features from the full input. To assemble the complete output activation, the per-GPU partial outputs are concatenated along the feature dimension (typically via an AllGather collective). Row-parallel layers are the dual case: they require an AllReduce sum because each GPU computes a partial sum over a subset of input features.

**Common Mistakes:**
- Choosing A) AllReduce (sum): That is required after a *row-wise* split, where each GPU produces a partial sum, not after a column-wise split where outputs are disjoint.
- Choosing C) Broadcast: Broadcast sends one rank's data to all; it does not assemble distributed outputs.
- Choosing D) Scatter: Scatter splits data outward; the column-wise pattern needs to combine outputs, not split them further.

**Related Material:** `lessons/mod-07-gpu-computing/06-model-pipeline-parallelism.md`

---

## Question 21
**Q:** List the 4 main components that consume GPU memory during training and give approximate size for each (assuming M parameters).

**Answer:**
1. **Model parameters** — M × 4 bytes = **4M bytes** in FP32 (2M in FP16/BF16).
2. **Gradients** — one gradient per parameter, M × 4 bytes = **4M bytes** in FP32.
3. **Optimizer states (Adam)** — momentum + variance buffers, 2 × M × 4 bytes = **8M bytes** in FP32 (Adam alone; SGD-momentum is only 4M).
4. **Activations** — saved intermediate tensors for backprop; **scales with batch size, sequence length, and depth** and is often the *largest* term for transformers/long sequences.

Total fixed cost ≈ **16M bytes** (params + grads + Adam states), plus a variable activations term.

**Explanation:**
The 16M-byte "training tax" on top of the model itself is why naive FP32 training of a 7B model needs ~112 GB before activations, and motivates mixed precision, ZeRO sharding, and gradient checkpointing. Knowing the breakdown lets you predict which optimization (FP16, ZeRO-1/2/3, checkpointing, smaller batch) attacks which term.

**Common Mistakes:**
- Reporting only parameters (4M) and forgetting that gradients and optimizer state each add comparable amounts.
- Reporting Adam optimizer state as 4M instead of 8M (forgetting it stores both first and second moments).
- Omitting activations entirely — they are variable but frequently dominate for long-context transformers.

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 22
**Q:** Describe the steps you would take to diagnose why a training job has low GPU utilization (<50%).

**Answer:**
1. **Observe baseline** with `nvidia-smi dmon` / `nvtop` to confirm sustained low SM utilization and check memory throughput.
2. **Time the data pipeline vs the step**: wrap the DataLoader iterator and the forward/backward to compare seconds-per-batch on each side. If the loader dominates, the GPU is starved.
3. **Inspect DataLoader config**: increase `num_workers`, enable `pin_memory=True`, set `persistent_workers=True`, and use `non_blocking=True` on `.to(device)` so H2D transfers overlap compute.
4. **Profile** with the PyTorch Profiler (with `record_shapes` and `with_stack`) or Nsight Systems to see the actual kernel timeline and any gaps.
5. **Hunt for host-device syncs**: `.item()`, `.cpu()`, `torch.cuda.synchronize()`, Python prints/logging in the hot loop — each one stalls the GPU.
6. **Check batch size and kernel size**: tiny batches or shapes that fall off Tensor Core tile boundaries (not multiples of 8/16) leave compute on the floor.
7. **Verify storage and preprocessing**: shard sizes, decoding cost (JPEG/audio), and disk vs network IO; consider moving augmentations to GPU (Kornia, DALI).
8. **Re-measure** after each change so you can attribute improvements.

**Explanation:**
The diagnostic mindset is "if util is low, the GPU is waiting" — find what it is waiting for (data, sync, Python). A profiler timeline beats guessing because it makes the gap between kernels visible and tells you whether it is host code, IO, or comm. Most <50% utilization cases trace back to the input pipeline, not the model.

**Common Mistakes:**
- Jumping to "buy a faster GPU" or "use mixed precision" before confirming compute is actually the bottleneck.
- Forgetting that `.item()` / Python-side logging inside the loop forces a full device sync each iteration.
- Increasing `num_workers` without `pin_memory` or `non_blocking=True`, so transfers still serialize against compute.

**Related Material:** `lessons/mod-07-gpu-computing/08-advanced-gpu-optimization.md`

---

## Question 23
**Q:** Write the minimal code to set up a DistributedDataParallel model in PyTorch. Include process group initialization and model wrapping.

**Answer:**
```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup() -> None:
    dist.destroy_process_group()

def train(rank: int, world_size: int) -> None:
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    # Use a DistributedSampler so each rank sees a unique shard.
    # optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    # for epoch in range(num_epochs):
    #     sampler.set_epoch(epoch)
    #     for batch in loader:
    #         ...

    cleanup()
```
Launch with `torchrun --nproc_per_node=<N> train.py` (which sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK` for you).

**Explanation:**
The four essentials are: (1) initialize a process group with the NCCL backend on GPUs, (2) pin each process to its own device with `torch.cuda.set_device(rank)`, (3) wrap the model with `DDP(..., device_ids=[rank])` so gradient AllReduce is registered, and (4) destroy the process group at the end. In production you should pair this with `DistributedSampler` and use `torchrun` rather than hand-managing env vars.

**Common Mistakes:**
- Omitting `torch.cuda.set_device(rank)`, which can cause all ranks to land on GPU 0 and silently serialize.
- Using the `gloo` backend on GPUs (it works but is much slower than NCCL for GPU collectives).
- Forgetting `dist.destroy_process_group()`, leaving NCCL communicators dangling at shutdown.

**Related Material:** `lessons/mod-07-gpu-computing/05-multi-gpu-training-strategies.md`

---

## Question 24
**Q:** List 5 techniques to reduce GPU memory usage during training and briefly explain each.

**Answer:**
1. **Mixed precision (FP16/BF16)** — store activations, gradients, and (with `autocast`) compute in 16-bit. Roughly halves activation/grad memory and unlocks Tensor Core throughput; pair with a loss scaler for FP16.
2. **Gradient checkpointing** — keep only a subset of activations from the forward pass and recompute the rest in backward. Drops activation memory from O(N) to ≈O(√N) layers in exchange for ~30% extra compute.
3. **Gradient accumulation** — run K micro-batches before stepping the optimizer. Lets you train at an effective batch size of K × micro-batch without holding K batches of activations at once.
4. **ZeRO / FSDP sharding** — partition optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across data-parallel ranks so each GPU holds only 1/N of the training state.
5. **Flash Attention** — IO-aware, tiled attention kernel that avoids materializing the N×N attention matrix, turning attention memory from O(N²) into O(N) and speeding it up at the same time.

Honorable mentions: CPU/NVMe offloading (DeepSpeed offload, FSDP CPU offload), reducing batch size or sequence length, parameter-efficient fine-tuning (LoRA), activation/weight quantization for inference.

**Explanation:**
The right tool depends on which term in the memory budget (parameters, gradients, optimizer state, activations) is biggest. Mixed precision and checkpointing attack activations; ZeRO/FSDP attacks the fixed 16M-byte training tax; gradient accumulation lets you trade time for memory without changing the optimization. Combine them — they are mostly orthogonal.

**Common Mistakes:**
- Listing both "smaller batch" and "gradient accumulation" as if they save different memory (they attack the same axis differently).
- Confusing gradient checkpointing with gradient accumulation — different mechanisms, different trade-offs.
- Calling FP16 a "free" 2x savings without mentioning the FP32 master-weights and loss-scaling overhead.

**Related Material:** `lessons/mod-07-gpu-computing/07-gpu-memory-management.md`

---

## Question 25
**Q:** Explain the difference between strong scaling and weak scaling in distributed training. Which is more commonly used in deep learning and why?

**Answer:**
- **Strong scaling**: hold the *total* problem size fixed (same dataset, same global batch) and add GPUs to finish faster. Ideal speedup is N× on N GPUs; in practice, communication and synchronization overhead eat into this and per-GPU work shrinks, hurting kernel efficiency.
- **Weak scaling**: hold the *per-GPU* problem size fixed (same local batch) and let the global batch grow with N. Ideal time-per-step stays constant as N grows, and per-GPU kernels remain large enough to keep the device busy.

**More common in deep learning: weak scaling.**

Why:
- Keeps Tensor Core utilization high because each GPU still sees a healthy local batch.
- Amortizes communication: the larger the local compute per step, the smaller the relative AllReduce cost.
- Matches how teams actually scale — they buy more GPUs to train bigger models or push more tokens, not to shrink a fixed job.
- Caveat: a growing global batch needs a learning-rate adjustment (linear scaling rule, LARS/LAMB, warmup) and may eventually hurt convergence at extreme batch sizes.

**Explanation:**
Strong scaling is the "shrink the wall-clock" framing; weak scaling is the "fill more iron" framing. Deep learning workloads are throughput-bound and respond well to bigger effective batches (with proper LR tuning), so the field defaults to weak scaling. Strong scaling matters mostly for latency-critical retraining or when batch size cannot grow further.

**Common Mistakes:**
- Swapping the definitions (saying weak scaling holds total work fixed).
- Claiming strong scaling is more common because "we want training to be faster" — ignoring that shrinking per-GPU work breaks kernel efficiency.
- Forgetting the LR-scaling caveat: weak scaling without adjusting the learning rate often diverges or generalizes worse.

**Related Material:** `lessons/mod-07-gpu-computing/04-distributed-training-fundamentals.md`

---
