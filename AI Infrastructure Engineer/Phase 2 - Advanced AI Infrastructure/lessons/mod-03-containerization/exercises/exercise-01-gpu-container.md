# Exercise 02: GPU-Accelerated ML Container

**Duration:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** NVIDIA GPU, NVIDIA Container Toolkit installed

## Objective

Build a CUDA-enabled Docker image that trains a small CNN on MNIST, expose the host GPU to the container correctly, and measure the actual GPU speedup versus CPU-only training on the same image.

## Why this matters

GPU containers fail in ways CPU containers don't: wrong CUDA base image, driver/toolkit mismatch, or a PyTorch wheel built for the wrong CUDA version all produce `torch.cuda.is_available() == False` with no obvious error. Debugging that reliably — and knowing what the GPU speedup should actually look like — is a core skill before you ever touch a real training job.

## Requirements

1. `train.py`: a small CNN trained on MNIST, logging device, GPU memory usage, and per-epoch time.
2. A CUDA-based Dockerfile that installs the GPU build of PyTorch matching the CUDA runtime.
3. Container correctly detects and uses the GPU via `--gpus all`.
4. Recorded GPU utilization during training and a CPU-vs-GPU timing comparison.
5. Final image under 4GB.

## Step-by-step

### Step 1 — Write the training script (30 min)
A `SimpleCNN` (two conv layers, two FC layers) trained for 5 epochs on MNIST. Log `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`, and `torch.cuda.memory_allocated(0)` at startup and periodically during training. Save the trained model to `/models/mnist_cnn.pth`.

### Step 2 — Write the Dockerfile (45 min)
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
COPY train.py .
RUN mkdir -p /models
ENV PYTHONUNBUFFERED=1
CMD ["python3", "train.py"]
```
The `--index-url` must match the CUDA version in the base image tag — a mismatch (e.g. cu121 wheels on a cu118 base) is the single most common cause of `CUDA not available` inside an otherwise-correct container.

### Step 3 — Build and run (30 min)
```bash
docker build -t gpu-training:v1.0 .
docker images gpu-training:v1.0                     # check size

docker run --rm --gpus all \
  -v $(pwd)/models:/models -v $(pwd)/data:/data \
  gpu-training:v1.0

# specific GPU
docker run --rm --gpus '"device=0"' -v $(pwd)/models:/models gpu-training:v1.0
```
In a second terminal while training runs: `watch -n 1 nvidia-smi` to confirm GPU utilization.

### Step 4 — CPU vs GPU comparison (30 min)
```bash
# CPU-only run (same image)
docker run --rm -v $(pwd)/models:/models -e CUDA_VISIBLE_DEVICES="" gpu-training:v1.0

# GPU run
docker run --rm --gpus all -v $(pwd)/models:/models gpu-training:v1.0
```
Record wall-clock time for both. Expect roughly 5-10x speedup on GPU for this workload.

## Deliverables

1. `Dockerfile`, `train.py`, `requirements.txt`.
2. `README.md` with build/run instructions and the CPU-vs-GPU timing table.

## Validation

- [ ] `torch.cuda.is_available()` returns `True` inside the container.
- [ ] Training runs on GPU, confirmed by `nvidia-smi` showing utilization above 70% during training.
- [ ] GPU training completes noticeably faster than the CPU run on the same image (aim for 5x or more).
- [ ] Model file is saved to the mounted `/models` volume.
- [ ] No CUDA version-mismatch errors or warnings in the logs.
- [ ] Final image size is under 4GB.

## Stretch goals

- Multi-GPU training with `DataParallel` or `DistributedDataParallel`.
- Automatic mixed precision (`torch.cuda.amp`) and measure the memory/speed tradeoff.
- Add a TensorBoard service via Docker Compose for live training curves.
- Add gradient checkpointing and compare peak GPU memory.

## Common pitfalls

- **"CUDA not available" inside the container** — check `nvidia-smi` on the host first, then `nvidia-ctk --version`, then confirm the PyTorch wheel's CUDA build matches the base image's CUDA runtime version.
- **Driver/toolkit version mismatch** — the container's CUDA runtime must be less than or equal to the host driver's supported CUDA version; check with `nvidia-smi | grep "CUDA Version"` on the host.
- **Out of memory on smaller GPUs** — reduce batch size or call `torch.cuda.empty_cache()` between runs; MNIST at batch size 64 should fit on virtually any CUDA GPU, so an OOM here usually means another process is holding memory.
- **Forgetting `--gpus all`** — without it, the container silently falls back to CPU with no error; always verify device selection in the training script's own logging, not just by assuming the flag worked.
