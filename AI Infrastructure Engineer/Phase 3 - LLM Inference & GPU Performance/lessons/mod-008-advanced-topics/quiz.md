# Module 08 — Quiz

10 questions. 70% pass.

### 1. CUDA Graphs primarily eliminate:
- a) GPU memory pressure
- b) GPU thermal throttling
- c) Per-kernel-launch overhead by capturing an op sequence + replaying it
- d) Inter-kernel synchronization

### 2. CUDA streams enable:
- a) Overlapping independent operations (e.g., compute on one stream while copying on another)
- b) Reduced GPU memory consumption
- c) A requirement for any GPU code (they aren't)
- d) Higher numeric precision

### 3. NVLink vs PCIe (for GPU-to-GPU communication):
- a) Roughly the same speed
- b) PCIe is the faster option
- c) NVLink is a software-only abstraction
- d) NVLink is ~10× faster than PCIe; effectively required for in-node tensor parallelism

### 4. InfiniBand's role in distributed ML training:
- a) A storage-only protocol
- b) Inter-node fabric with RDMA + low latency for multi-node collectives
- c) A replacement for NVLink within a node
- d) Required even for single-GPU training

### 5. GPUDirect RDMA is:
- a) A replacement for NCCL
- b) Requires local SSDs
- c) Remote GPU memory access without involving the CPU on either side
- d) Identical to PCIe peer access

### 6. MIG (Multi-Instance GPU) partitioning:
- a) Splits one A100/H100 into hardware-isolated independent GPU instances
- b) A multi-GPU NCCL configuration
- c) A software-only sharing scheme
- d) Always degrades performance vs full GPU

### 7. FP8 training on H100 with Transformer Engine typically yields (vs BF16):
- a) Same speed as bf16
- b) Always significant accuracy loss
- c) 30-50% training speedup with minimal accuracy loss
- d) Is required for any inference workload

### 8. The dynamic range issue with FP8:
- a) Same range as fp16
- b) Not a real concern
- c) Wider range than bf16
- d) Narrow range and few mantissa bits — scaling factors (per tensor, per row or per block) are required

### 9. CUDA Graph capture is bounded by:
- a) GPU hardware model
- b) Input shape — new shapes require recapture
- c) NVIDIA driver version
- d) Container runtime version

### 10. The right first diagnostic for "slow distributed training":
- a) Replace the model architecture
- b) Add more GPUs
- c) Switch frameworks
- d) Run nccl-tests (all-reduce benchmark) — rules out interconnect-fabric issues

Answers: [quiz-answers.md](quiz-answers.md)
