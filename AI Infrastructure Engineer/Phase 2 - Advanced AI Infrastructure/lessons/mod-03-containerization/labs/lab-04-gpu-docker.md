# Lab 04: Run a GPU Container with PyTorch CUDA

**Duration:** 60 min  **Prerequisites:** A machine with an NVIDIA GPU + NVIDIA Container Toolkit installed

## Objective
Run a PyTorch inference workload inside a Docker container that uses the host's NVIDIA GPU. Verify CUDA visibility and run a simple benchmark.

## Steps

### 1. Verify host GPU access
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
Both should list your GPU. If the second fails, install NVIDIA Container Toolkit.

### 2. PyTorch CUDA Dockerfile
```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
```

### 3. Sample app.py
```python
import time, torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device name:", torch.cuda.get_device_name(0))

# Tiny benchmark
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    x = x @ x
torch.cuda.synchronize()
print(f"50x 4096x4096 matmul: {time.perf_counter()-t0:.3f}s")
```

### 4. Build and run with GPU
```bash
docker build -t torch-gpu .
docker run --rm --gpus all torch-gpu
```

### 5. Limit GPU access
```bash
docker run --rm --gpus '"device=0"' torch-gpu              # specific GPU
docker run --rm --gpus '"device=0,1"' torch-gpu            # multiple
docker run --rm --gpus 'count=1,capabilities=compute,utility' torch-gpu
```

## Validation
- [ ] `torch.cuda.is_available()` returns True inside the container.
- [ ] Benchmark completes (typically <2s for matmul on RTX 30xx+).
- [ ] Running without `--gpus all` falls back to CPU and is much slower (verify by removing the flag).

## Cleanup
```bash
docker rmi torch-gpu
```

## Troubleshooting
- **`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA Container Toolkit not installed or `docker` daemon not restarted after install.
- **CUDA OOM** — Match container image CUDA major version to host driver (`nvidia-smi` shows max supported CUDA).
- **WSL2 on Windows** — Use WSL2 kernel 5.10.43+ and Docker Desktop's GPU integration.
