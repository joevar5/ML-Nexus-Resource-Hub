# Lesson 07: GPU Support in Docker

**Duration:** 5 hours
**Objectives:** Run GPU-accelerated ML workloads in Docker containers

## Learning Objectives

By the end of this lesson, you will be able to:

1. Install and configure NVIDIA Container Toolkit
2. Run GPU-accelerated containers
3. Use CUDA-enabled base images
4. Map specific GPUs to containers
5. Monitor GPU usage in containerized workloads
6. Deploy multi-GPU training containers
7. Troubleshoot common GPU container issues

## Why GPU Containers?

### The Challenge

Machine learning training and inference often require GPUs for acceptable performance:
- **Training**: 100x faster with GPU vs CPU
- **Inference**: 10-50x faster for large models
- **Cost**: More cost-effective (faster = cheaper cloud bills)

**But GPUs add complexity:**
- CUDA drivers must match toolkit versions
- Library dependencies are fragile
- GPU resources need careful allocation
- Environment setup is error-prone

### The Solution: GPU Containers

Docker containers with GPU support solve these problems:

```
┌──────────────────────────────────────────────┐
│           GPU Container Benefits              │
├──────────────────────────────────────────────┤
│                                               │
│  ✅ Reproducible GPU environments            │
│  ✅ No driver/CUDA version conflicts         │
│  ✅ Easy multi-GPU resource allocation       │
│  ✅ Portable across different machines       │
│  ✅ Isolation between workloads              │
│                                               │
└──────────────────────────────────────────────┘
```

**Real-World Impact:**
- Setup time: 2-3 hours → 5 minutes
- Consistency: "works on my machine" eliminated
- Resource utilization: 40% → 80%+ (better GPU sharing)

## NVIDIA Container Toolkit

### Architecture

```
┌────────────────────────────────────────────────┐
│                  Docker Container               │
│  ┌──────────────────────────────────────────┐  │
│  │     ML Application (PyTorch/TF)          │  │
│  │              ↓                            │  │
│  │        CUDA Libraries                     │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────┬──────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  NVIDIA Container Toolkit   │
        │    (nvidia-docker2)         │
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼──────────────┐
        │    Docker Engine            │
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   NVIDIA Driver (Host)      │
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      GPU Hardware           │
        └─────────────────────────────┘
```

**Key Components:**
1. **NVIDIA Driver** - On host OS (e.g., 525.125.06)
2. **NVIDIA Container Toolkit** - Exposes GPUs to Docker
3. **CUDA Libraries** - Inside container (can be different versions!)

### Installation

**Prerequisites:**
```bash
# Check if you have NVIDIA GPU
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi
```

**Install NVIDIA Container Toolkit:**

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify Installation:**

```bash
# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Should show your GPU(s)
```

## Running GPU Containers

### Basic GPU Container

```bash
# Run with all GPUs
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Run with specific number of GPUs
docker run --rm --gpus 2 nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Run with specific GPU device
docker run --rm --gpus '"device=0"' nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Run with multiple specific GPUs
docker run --rm --gpus '"device=0,2"' nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### GPU Flags Explained

```bash
# All GPUs
--gpus all

# Specific count
--gpus 2

# Specific devices
--gpus '"device=0"'
--gpus '"device=0,1,3"'

# GPU capabilities (for advanced use)
--gpus '"capabilities=compute,utility"'
```

## CUDA Base Images

### NVIDIA Official Images

**Image Naming Convention:**
```
nvidia/cuda:[CUDA_VERSION]-[FLAVOR]-[OS]

Examples:
nvidia/cuda:12.1.0-base-ubuntu22.04
nvidia/cuda:12.1.0-runtime-ubuntu22.04
nvidia/cuda:12.1.0-devel-ubuntu22.04
nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
```

**Flavors:**

| Flavor | Size | Contents | Use Case |
|--------|------|----------|----------|
| **base** | ~200MB | CUDA runtime only | Minimal GPU access |
| **runtime** | ~1.5GB | CUDA runtime + libraries | Inference |
| **devel** | ~3.5GB | Runtime + compilers, headers | Training, building from source |
| **cudnn** | Varies | + cuDNN (deep learning) | Deep learning workloads |

### Choosing the Right Image

```
Need to compile CUDA code?
├─ Yes → devel
└─ No  → Do you need cuDNN?
    ├─ Yes → cudnn8-runtime
    └─ No  → runtime
```

**Examples:**

```dockerfile
# Inference (lightweight)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Training (need compilers)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Minimal GPU access
FROM nvidia/cuda:12.1.0-base-ubuntu22.04
```

## Building GPU-Enabled Images

### PyTorch GPU Image

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

# Verify GPU access
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]
```

**Build and Run:**
```bash
# Build
docker build -t pytorch-gpu:v1 .

# Run with GPU
docker run --rm --gpus all pytorch-gpu:v1
```

### TensorFlow GPU Image

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install TensorFlow with GPU support
RUN pip3 install --no-cache-dir tensorflow[and-cuda]==2.15.0

COPY . /app

# Verify GPU
RUN python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

CMD ["python3", "train.py"]
```

### Hugging Face Transformers GPU

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install transformers with PyTorch GPU support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    transformers==4.35.0 \
    accelerate \
    bitsandbytes

COPY . /app

CMD ["python3", "inference.py"]
```

## GPU Resource Management

### Limiting GPU Memory

**Set memory limit:**
```bash
# Limit to 4GB
docker run --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 \
    pytorch-gpu:v1
```

**In Python code:**
```python
import torch

# Limit GPU memory fraction
torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # Use 50% of GPU 0

# Or set memory limit
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### Multi-GPU Allocation

```bash
# Container 1: Use GPU 0
docker run -d --name worker1 --gpus '"device=0"' pytorch-gpu:v1

# Container 2: Use GPU 1
docker run -d --name worker2 --gpus '"device=1"' pytorch-gpu:v1

# Container 3: Use GPUs 2 and 3
docker run -d --name worker3 --gpus '"device=2,3"' pytorch-gpu:v1
```

### GPU Sharing (MIG - Multi-Instance GPU)

For NVIDIA A100/H100 GPUs:

```bash
# Enable MIG mode (requires reboot)
sudo nvidia-smi -mig 1

# Create GPU instances
sudo nvidia-smi mig -cgi 9,9,9,9  # 4 instances

# Run container with MIG instance
docker run --rm --gpus '"device=0:0"' pytorch-gpu:v1
```

## Monitoring GPU Usage

### nvidia-smi in Containers

```bash
# Install nvidia-smi in container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Monitor GPU while container runs
docker run -d --name training --gpus all pytorch-gpu:v1
watch -n 1 nvidia-smi
```

### DCGM (Data Center GPU Manager)

```bash
# Run DCGM exporter for Prometheus
docker run -d \
    --gpus all \
    --name dcgm-exporter \
    -p 9400:9400 \
    nvidia/dcgm-exporter:latest

# Scrape metrics
curl http://localhost:9400/metrics | grep gpu
```

### Python Monitoring

**monitor_gpu.py:**
```python
import torch
import time

def monitor_gpu():
    while True:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        time.sleep(5)

if __name__ == "__main__":
    monitor_gpu()
```

## Multi-GPU Training

### Data Parallel Training

**train.py:**
```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Define model
model = MyModel()

# Use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

model = model.cuda()

# Training loop
for batch in dataloader:
    # Data automatically distributed across GPUs
    outputs = model(batch)
```

**Run:**
```bash
# Use all GPUs
docker run --rm --gpus all \
    -v $(pwd)/data:/data \
    pytorch-gpu:v1 python train.py
```

### Distributed Data Parallel (DDP)

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

COPY . /app

# Use all GPUs with DDP
CMD ["python3", "-m", "torch.distributed.launch", \
     "--nproc_per_node=auto", "train_ddp.py"]
```

## Docker Compose with GPUs

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  # Training job on GPU 0
  trainer-1:
    build: ./trainer
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    volumes:
      - training-data:/data
      - model-outputs:/outputs

  # Training job on GPU 1
  trainer-2:
    build: ./trainer
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    volumes:
      - training-data:/data
      - model-outputs:/outputs

  # Inference service on GPU 2
  inference:
    build: ./inference
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [gpu]
    ports:
      - "8000:8000"

volumes:
  training-data:
  model-outputs:
```

## Common GPU Container Issues

### Issue 1: CUDA Version Mismatch

**Error:**
```
RuntimeError: CUDA version mismatch: PyTorch compiled with 12.1 but running with 11.8
```

**Solution:**
```dockerfile
# Match CUDA versions
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install PyTorch built for CUDA 12.1
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: GPU Not Visible

**Error:**
```
RuntimeError: CUDA not available
```

**Checklist:**
```bash
# 1. Verify NVIDIA driver on host
nvidia-smi

# 2. Check Container Toolkit installation
nvidia-ctk --version

# 3. Run with --gpus flag
docker run --gpus all ...  # Don't forget this!

# 4. Check container sees GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

### Issue 3: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 16  # Try smaller value

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Issue 4: Slow Performance

**Diagnosis:**
```bash
# Monitor GPU utilization
nvidia-smi dmon -s pucvmet

# Should see high GPU utilization (>80%)
# If low, you have a bottleneck (likely data loading)
```

**Solutions:**
```python
# Use multiple data loading workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)

# Prefetch to GPU
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
```

## Hands-On Exercise: GPU Training Pipeline

### Objective

Build a complete GPU-accelerated training pipeline using Docker.

### Requirements

1. Multi-stage Dockerfile for training
2. Data preprocessing container
3. Training container with GPU
4. Model export container
5. Docker Compose to orchestrate

### Project Structure

```
gpu-training/
├── docker-compose.yml
├── preprocess/
│   ├── Dockerfile
│   └── preprocess.py
├── train/
│   ├── Dockerfile
│   └── train.py
└── export/
    ├── Dockerfile
    └── export.py
```

### Implementation

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  preprocess:
    build: ./preprocess
    volumes:
      - raw-data:/data/raw
      - processed-data:/data/processed
    command: python preprocess.py

  train:
    build: ./train
    depends_on:
      - preprocess
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - processed-data:/data
      - models:/models
    environment:
      - EPOCHS=10
      - BATCH_SIZE=32
    command: python train.py

  export:
    build: ./export
    depends_on:
      - train
    volumes:
      - models:/models
      - artifacts:/artifacts
    command: python export.py

volumes:
  raw-data:
  processed-data:
  models:
  artifacts:
```

**preprocess/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir pandas numpy scikit-learn

COPY preprocess.py .
CMD ["python", "preprocess.py"]
```

**preprocess/preprocess.py:**
```python
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW = Path("/data/raw")
OUT = Path("/data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.concat(pd.read_csv(p) for p in RAW.glob("*.csv"))
df = df.dropna().drop_duplicates()
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_parquet(OUT / "train.parquet")
test.to_parquet(OUT / "test.parquet")
print(f"preprocess: train={len(train)} test={len(test)}")
```

**train/Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir torch torchvision pandas pyarrow

COPY train.py .
CMD ["python3", "train.py"]
```

**train/train.py:**
```python
import os
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

EPOCHS = int(os.environ.get("EPOCHS", 10))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"train: device={device}")

train = pd.read_parquet("/data/train.parquet")
X = torch.tensor(train.drop(columns=["label"]).values, dtype=torch.float32, device=device)
y = torch.tensor(train["label"].values, dtype=torch.long, device=device)

model = nn.Sequential(
    nn.Linear(X.shape[1], 128), nn.ReLU(),
    nn.Linear(128, int(y.max().item()) + 1),
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for i in range(0, len(X), BATCH_SIZE):
        opt.zero_grad()
        out = model(X[i:i + BATCH_SIZE])
        loss = loss_fn(out, y[i:i + BATCH_SIZE])
        loss.backward()
        opt.step()
    print(f"epoch={epoch} loss={loss.item():.4f}")

Path("/models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "/models/model.pt")
```

**export/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch onnx

COPY export.py .
CMD ["python", "export.py"]
```

**export/export.py:**
```python
import torch
import torch.nn as nn
from pathlib import Path

# Reconstruct the architecture identically to train.py.
# In production, persist the architecture or use torch.jit.script.
input_dim = 16  # match feature count
num_classes = 10
model = nn.Sequential(
    nn.Linear(input_dim, 128), nn.ReLU(),
    nn.Linear(128, num_classes),
)
model.load_state_dict(torch.load("/models/model.pt", map_location="cpu"))
model.train(False)  # switch to inference mode (equivalent to .eval())

dummy = torch.randn(1, input_dim)
out = Path("/artifacts")
out.mkdir(exist_ok=True)
torch.onnx.export(model, dummy, out / "model.onnx", opset_version=17)
print("export: wrote /artifacts/model.onnx")
```

Run the full pipeline with:

```bash
docker compose build
docker compose up --abort-on-container-exit
```

The train service requests one GPU via the `nvidia` driver. Verify GPU
visibility inside the container with `docker compose run train nvidia-smi`.

## Summary

In this lesson, you learned:

1. **NVIDIA Container Toolkit** - Installation and configuration
2. **GPU Containers** - Running CUDA workloads in Docker
3. **CUDA Base Images** - Choosing base, runtime, devel, cudnn variants
4. **GPU Allocation** - Mapping specific GPUs to containers
5. **Monitoring** - nvidia-smi, DCGM for GPU metrics
6. **Multi-GPU** - Data parallel and distributed training
7. **Troubleshooting** - Common issues and solutions

**Key Takeaways:**
- GPU containers eliminate driver/CUDA version conflicts
- Use runtime images for inference, devel for training
- Monitor GPU utilization to identify bottlenecks
- Allocate GPUs explicitly in multi-GPU systems
- Match CUDA versions between base image and frameworks

## What's Next?

In the next lesson, **08-production-best-practices.md**, you'll learn:
- Security best practices for containers
- Health checks and restart policies
- Logging and monitoring in production
- Resource limits (CPU, memory, GPU)
- Production-ready Dockerfile templates

---

## Self-Check Questions

1. What's the difference between CUDA base, runtime, and devel images?
2. How do you run a container with access to GPU 2 only?
3. What causes "CUDA version mismatch" errors?
4. How do you monitor GPU memory usage in a running container?
5. What's the difference between DataParallel and DistributedDataParallel?

## Additional Resources

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda)

---

**Next:** [08-production-best-practices.md](./08-production-best-practices.md)
