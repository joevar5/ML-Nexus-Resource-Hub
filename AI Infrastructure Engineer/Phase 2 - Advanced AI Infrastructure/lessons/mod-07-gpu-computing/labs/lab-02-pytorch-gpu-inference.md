# Lab 02: PyTorch GPU Inference and Benchmarking

**Duration:** 60 min  **Prerequisites:** PyTorch with CUDA installed

## Objective
Load a real model (ResNet-50 from torchvision), move it to GPU, and benchmark CPU vs GPU throughput on a synthetic batch.

## Steps

### 1. Setup
```python
import time, torch
from torchvision.models import resnet50, ResNet50_Weights

assert torch.cuda.is_available(), "no GPU"
device = torch.device("cuda")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.train(False)   # inference mode (equivalent to .eval())
```

### 2. CPU benchmark
```python
x = torch.randn(8, 3, 224, 224)
with torch.inference_mode():
    # warm up
    for _ in range(3): model(x)
    t0 = time.perf_counter()
    for _ in range(50): model(x)
    cpu_s = (time.perf_counter() - t0) / 50
print(f"cpu per-batch: {cpu_s*1000:.1f} ms")
```

### 3. GPU benchmark
```python
model_gpu = model.to(device)
x_gpu = x.to(device)
with torch.inference_mode():
    for _ in range(3): model_gpu(x_gpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): model_gpu(x_gpu); torch.cuda.synchronize()
    gpu_s = (time.perf_counter() - t0) / 50
print(f"gpu per-batch: {gpu_s*1000:.1f} ms   speedup: {cpu_s/gpu_s:.1f}x")
```

### 4. Throughput vs batch size
```python
for bs in (1, 4, 8, 16, 32, 64):
    x = torch.randn(bs, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(3): model_gpu(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): model_gpu(x); torch.cuda.synchronize()
        s = (time.perf_counter()-t0)/20
    print(f"bs={bs:3d} per-batch={s*1000:6.1f}ms throughput={bs/s:7.1f} img/s")
```

### 5. fp16 with autocast
```python
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
    for _ in range(3): model_gpu(x_gpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): model_gpu(x_gpu); torch.cuda.synchronize()
    fp16_s = (time.perf_counter()-t0)/50
print(f"fp16 per-batch: {fp16_s*1000:.1f}ms")
```

### 6. TorchScript / `torch.compile` (PyTorch 2.x)
```python
compiled = torch.compile(model_gpu, mode="reduce-overhead")
# Compare against eager.
```

## Validation
- [ ] GPU speedup ≥ 5× over CPU for batch 8 (consumer GPU).
- [ ] Per-image latency drops as batch size grows (until you hit memory limits).
- [ ] fp16 is 1.5-2× faster than fp32 on Ampere or newer.

## Cleanup
None.

## Troubleshooting
- **`cuda out of memory`** — Reduce batch size. Check `nvidia-smi` for other processes hogging memory.
- **fp16 slower than fp32** — Pascal-era GPU without Tensor Cores. fp16 only wins on Volta+.
- **`torch.compile` errors** — Some ops not yet supported; fall back to eager with `model_compiled = torch.compile(model_gpu, dynamic=True)`.
