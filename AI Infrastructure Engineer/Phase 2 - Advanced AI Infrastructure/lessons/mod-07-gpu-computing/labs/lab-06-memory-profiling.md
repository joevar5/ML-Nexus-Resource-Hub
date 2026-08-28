# Lab 06: GPU Memory Profiling and OOM Diagnosis

**Duration:** 60 min  **Prerequisites:** PyTorch + GPU

## Objective
Use PyTorch's memory profiling tools to find where memory grows during training, distinguish "activation" vs "parameter" vs "optimizer state" memory, and apply gradient checkpointing.

## Steps

### 1. Snapshot current GPU usage
```python
import torch
print(torch.cuda.memory_summary(device=0, abbreviated=False))
```

### 2. Track step-by-step allocations
```python
torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... training loop ...

torch.cuda.memory._dump_snapshot("snapshot.bin")   # binary snapshot
torch.cuda.memory._record_memory_history(enabled=None)
```
Upload `snapshot.bin` to https://pytorch.org/memory_viz to see a graphical timeline.

### 3. Force an OOM and trace
```python
import torch
device = torch.device("cuda")
tensors = []
try:
    while True:
        tensors.append(torch.randn(2_000_000_000, device=device))   # 2B floats ≈ 8GB each
except torch.cuda.OutOfMemoryError as e:
    print(f"OOM after {len(tensors)} allocations")
    print(torch.cuda.memory_summary())
```

### 4. Gradient checkpointing to fit a larger model
```python
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn as nn

big = nn.Sequential(*[nn.Sequential(nn.Linear(4096,4096), nn.ReLU()) for _ in range(20)]).cuda()

x = torch.randn(8, 4096, device="cuda", requires_grad=True)
# Without checkpointing: stores activations for all 20 layers
# With checkpointing: re-computes activations during backward, saves memory
y = checkpoint_sequential(big, 4, x)         # group into 4 segments
y.sum().backward()
```

### 5. Optimizer state breakdown
For Adam: each parameter has 1 fp32 copy + 2 momentum tensors → 4× param size. SGD with momentum: 2×.

### 6. Calculate from first principles
For a model with P parameters, training memory ≈
- Params: 4P bytes (fp32)
- Adam states: 8P bytes
- Activations: depends on batch + depth, often dominates
- Gradients: 4P bytes

For a 1B param model with Adam in fp32: ~16 GB just for params + states, plus activations.

### 7. Tooling
```bash
nvidia-smi dmon -s u -c 60                  # 60s of GPU util sampling
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -lms 500
```

## Validation
- [ ] `memory_summary` shows current allocator state.
- [ ] Recorded snapshot visualizes in the PyTorch memory viz tool.
- [ ] Gradient checkpointing reduces peak memory measurably (20-40% in this example).

## Cleanup
```bash
rm -f snapshot.bin
```

## Troubleshooting
- **Memory keeps growing across epochs** — Tensors held in Python (`losses.append(loss)`); call `.item()` to free.
- **OOM at validation but not training** — `model.train(False)` alone doesn't free anything; wrap validation in `torch.no_grad()` to skip activation storage.
- **`memory_summary` shows fragmentation** — Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
