# Lab 05: Mixed-Precision Training with `torch.amp`

**Duration:** 45 min  **Prerequisites:** PyTorch with CUDA, Volta+ GPU

## Objective
Convert an fp32 training loop to use mixed precision (fp16/bf16). Measure speedup and memory savings.

## Steps

### 1. Baseline fp32 loop
```python
import time, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda")
X = torch.randn(50_000, 1024); y = torch.randint(0, 10, (50_000,))
loader = DataLoader(TensorDataset(X, y), batch_size=512, num_workers=2, pin_memory=True)

model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 10)).to(device)
opt = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

def train_one_epoch(amp=False, dtype=None):
    scaler = torch.amp.GradScaler("cuda") if amp and dtype is torch.float16 else None
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        opt.zero_grad()
        if amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                loss = loss_fn(model(xb), yb)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            loss_fn(model(xb), yb).backward(); opt.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

print(f"fp32: {train_one_epoch():.2f}s peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
print(f"fp16: {train_one_epoch(amp=True, dtype=torch.float16):.2f}s peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
print(f"bf16: {train_one_epoch(amp=True, dtype=torch.bfloat16):.2f}s peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

### 2. Run and compare
Expected on Ampere+ (RTX 30xx, A100, H100):
- fp16 ~1.5-2× faster than fp32
- bf16 similar speed to fp16, but more numerically stable
- Peak memory: ~50% of fp32 for fp16/bf16

### 3. Diagnose unstable fp16
If loss goes NaN with fp16, increase `init_scale` in `GradScaler` or switch to bf16.

### 4. fp16 weight storage
For inference-only:
```python
model_fp16 = model.half().to(device)
```
Halves model memory footprint.

## Validation
- [ ] fp16 epoch time < fp32 epoch time.
- [ ] Peak memory in fp16 < peak memory in fp32.
- [ ] Loss converges in all three precisions (no NaN).

## Cleanup
None.

## Troubleshooting
- **`RuntimeError: Found dtype Half but expected Float`** — Some op isn't autocast-safe. Wrap that op in `with torch.amp.autocast(..., enabled=False)`.
- **fp16 actually slower** — Old GPU without Tensor Cores. Pascal (GTX 10xx) doesn't benefit.
- **bf16 not supported** — bf16 requires Ampere or newer.
