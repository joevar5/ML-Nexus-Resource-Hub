# Exercise 03: End-to-End PyTorch GPU Pipeline

**Duration:** 3 hours
**Difficulty:** Beginner+
**Prerequisites:** PyTorch with CUDA, exercise 02

## Objective

Build a complete training pipeline for a small ResNet on CIFAR-10 that runs entirely on GPU: DataLoader with proper pinned-memory + workers, AMP mixed precision, gradient accumulation, learning-rate scheduling, checkpointing. Then build the matching inference pipeline that loads the trained model and benchmarks throughput.

## Why this matters

This is the smallest realistic ML training loop. Every "real" project is an elaboration. Doing it correctly once internalizes the patterns; doing it from memory the second time costs nothing.

## Requirements

### Training (`train.py`)
1. `torchvision` ResNet-18 on CIFAR-10.
2. DataLoader with `pin_memory=True`, `num_workers ≥ 4`, `persistent_workers=True`.
3. AMP with `torch.amp.autocast("cuda", dtype=torch.float16)` and `GradScaler`.
4. Gradient accumulation factor configurable via CLI (test with 1, 4, 16).
5. CosineAnnealingLR scheduler.
6. Checkpoint every epoch including model + optimizer + scheduler + scaler state. Prefer **safetensors** over raw torch save for safety.
7. Resume from `--resume <ckpt>` produces bit-identical loss within the next 10 steps.
8. Per-epoch metrics logged as JSON to stdout.

### Inference (`infer.py`)
9. Load model checkpoint.
10. Run on a held-out batch.
11. Benchmark throughput (images/sec) at batch sizes 1, 16, 128.
12. Use `torch.inference_mode()` and disable autograd everywhere.

## Step-by-step

### Step 1 — Skeleton (15 min)
Project structure:
```
cifar-train/
├── train.py
├── infer.py
├── models/      # checkpoints
└── data/        # CIFAR-10 download cache
```

### Step 2 — DataLoader (30 min)
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])
train_ds = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True,
                          prefetch_factor=4)
```

### Step 3 — Model + optimizer + scaler (15 min)
```python
import torch
from torchvision.models import resnet18

device = torch.device("cuda")
model = resnet18(num_classes=10).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
scaler = torch.amp.GradScaler("cuda")
loss_fn = torch.nn.CrossEntropyLoss()
```

### Step 4 — Training loop with AMP + accumulation (45 min)
```python
ACCUM = args.accum
for epoch in range(args.epochs):
    model.train()
    t0 = time.perf_counter()
    for step, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(xb), yb) / ACCUM
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
    sched.step()
    save_ckpt(epoch, model, opt, sched, scaler)
    log({"epoch": epoch, "loss": loss.item()*ACCUM, "time_s": time.perf_counter()-t0})
```

### Step 5 — Checkpoint + resume with safetensors (30 min)
```python
from safetensors.torch import save_file, load_file
import json

def save_ckpt(epoch, model, opt, sched, scaler, path):
    save_file(model.state_dict(), f"{path}.safetensors")
    # Optimizer/scheduler/scaler state via torch's native serializer
    torch.save({
        "epoch": epoch,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
    }, f"{path}.aux.pt")

def load_ckpt(path, model, opt, sched, scaler):
    model.load_state_dict(load_file(f"{path}.safetensors"))
    aux = torch.load(f"{path}.aux.pt", weights_only=True)
    opt.load_state_dict(aux["opt"])
    sched.load_state_dict(aux["sched"])
    scaler.load_state_dict(aux["scaler"])
    return aux["epoch"]
```
Note: `weights_only=True` opts out of arbitrary code execution paths in the loader — recommended for any untrusted checkpoint source.

### Step 6 — Inference benchmark (30 min)
```python
# infer.py
import torch, time
from torchvision.models import resnet18
from safetensors.torch import load_file

model = resnet18(num_classes=10).cuda()
model.load_state_dict(load_file("models/epoch-final.safetensors"))
model.train(False)   # inference mode

for bs in (1, 16, 128):
    x = torch.randn(bs, 3, 32, 32, device="cuda")
    with torch.inference_mode():
        for _ in range(5): model(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50): model(x); torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    print(f"bs={bs:3d} throughput={bs*50/elapsed:7.0f} img/s lat/img={elapsed/(50*bs)*1000:.2f}ms")
```

### Step 7 — Resume test (15 min)
Train 2 epochs, save ckpt at epoch 1, kill the job. Resume from ckpt; verify loss trajectory matches a non-interrupted run within numerical noise.

## Deliverables

1. `train.py` + `infer.py` satisfying all 12 requirements.
2. `BENCHMARK.md` with throughput numbers per batch size.
3. Demonstration that `--resume` produces identical trajectory.

## Validation

- [ ] Training converges; final test accuracy ≥ 88% after 200 epochs (CIFAR-10 ResNet-18 baseline).
- [ ] AMP enabled — verify mixed precision via `nvidia-smi`-observed fp16 utilization.
- [ ] Gradient accumulation with factor 4 produces same effective batch as 4× larger batch (validate by short comparison run).
- [ ] Inference benchmark shows superlinear throughput scaling with batch size.
- [ ] Resume from checkpoint reproduces next-step loss within 1e-3.

## Stretch goals

- Add **`torch.compile`** and report speedup.
- Add **DDP** for multi-GPU (combine with mod-07 lab 03 patterns).
- Implement **per-step profiling** with `torch.profiler` and produce a trace viewable in chrome://tracing.

## Common pitfalls

- **`num_workers=0`** — Falls back to single-threaded DataLoader. CPUs becomes the bottleneck even on a 4090.
- **AMP gradients NaN early** — Initial `GradScaler` scale too high. Default behavior is to halve and retry; just lets it stabilize over the first ~100 steps.
- **`scheduler.step()` before `optimizer.step()`** — PyTorch warns now, but it used to silently produce wrong LR.
- **CIFAR-10 download fails behind proxy** — Pre-download manually and place in `data/cifar-10-batches-py/`.
- **Untrusted checkpoint loading** — Always use `weights_only=True` for `torch.load` from untrusted sources, or prefer safetensors which doesn't execute code.
