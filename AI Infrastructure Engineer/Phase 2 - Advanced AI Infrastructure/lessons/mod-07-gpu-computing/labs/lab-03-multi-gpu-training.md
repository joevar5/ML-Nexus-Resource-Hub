# Lab 03: Multi-GPU Training with DistributedDataParallel

**Duration:** 90 min  **Prerequisites:** Machine with ≥2 GPUs OR a cloud instance with multiple

## Objective
Train a small model across multiple GPUs on a single node with PyTorch's DistributedDataParallel, observing per-GPU gradient sync and near-linear scaling.

## Steps

### 1. Training script
```python
# train_ddp.py
import os, time, torch, torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # synthetic dataset
    N = 1_000_000
    X = torch.randn(N, 128)
    y = torch.randint(0, 10, (N,))
    ds = TensorDataset(X, y)
    sampler = DistributedSampler(ds)
    loader = DataLoader(ds, batch_size=4096, sampler=sampler, num_workers=4, pin_memory=True)

    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if local_rank == 0:
            print(f"epoch {epoch} took {time.perf_counter()-t0:.1f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### 2. Launch with torchrun
```bash
torchrun --standalone --nproc-per-node=2 train_ddp.py
# Or for 4 GPUs: --nproc-per-node=4
```

### 3. Single-GPU baseline for comparison
Disable DDP and run on `cuda:0` only. Compare epoch time.

### 4. Observe NCCL traffic
```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc-per-node=2 train_ddp.py 2>&1 | grep -i 'nccl' | head
```

### 5. Inspect with `nvidia-smi dmon`
In another terminal:
```bash
nvidia-smi dmon -i 0,1 -s pucvmet -c 30
```
See per-GPU utilization and memory during training.

## Validation
- [ ] Both/all GPUs show non-zero utilization during training.
- [ ] Epoch time on N GPUs is roughly 1/N of single-GPU time (minus small NCCL overhead).
- [ ] `loss` decreases over epochs.

## Cleanup
None.

## Troubleshooting
- **`Address already in use`** — Previous run left a master port. Wait or change `--master-port`.
- **NCCL hang** — Check inter-GPU connectivity; `nvidia-smi topo -m` shows NVLink/PCIe topology.
- **No scaling** — DataLoader bottlenecked; increase `num_workers` or `pin_memory=True`.
- **`each process gets 100% of data`** — Forgot `DistributedSampler`; each rank reads the full dataset.
