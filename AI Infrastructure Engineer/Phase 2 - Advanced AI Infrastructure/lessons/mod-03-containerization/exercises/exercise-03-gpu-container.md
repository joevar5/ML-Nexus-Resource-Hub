# Exercise 03: GPU-Accelerated ML Container

**Estimated Time:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** NVIDIA GPU, NVIDIA Container Toolkit installed

## Objective

Build and run a GPU-accelerated PyTorch training container that trains a simple neural network on MNIST dataset.

## Learning Goals

- Use CUDA-enabled base images
- Configure GPU access in containers
- Monitor GPU usage during training
- Handle GPU memory efficiently

## Project Structure

```
gpu-training/
├── Dockerfile
├── requirements.txt
├── train.py
└── README.md
```

## Step 1: Create Training Script

**train.py:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
                    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                    print(f'GPU Memory - Allocated: {mem_allocated:.2f} MB, Reserved: {mem_reserved:.2f} MB')

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}')

    # Save model
    torch.save(model.state_dict(), '/models/mnist_cnn.pth')
    print("Model saved to /models/mnist_cnn.pth")

if __name__ == "__main__":
    train()
```

**requirements.txt:**
```txt
torch==2.1.0
torchvision==0.16.0
```

## Step 2: Create Dockerfile

**TODO: Complete this GPU-enabled Dockerfile**

```dockerfile
# TODO: Choose CUDA base image
# Recommendation: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04


# TODO: Install Python 3.11


# TODO: Set working directory


# TODO: Copy requirements and install PyTorch with CUDA support
# Use: --index-url https://download.pytorch.org/whl/cu121


# TODO: Copy training script


# TODO: Create directory for saving models


# TODO: Verify GPU access at build time (optional)
# RUN python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"


# TODO: Set environment variables
# CUDA_VISIBLE_DEVICES, PYTHONUNBUFFERED


# TODO: Set command to run training
# CMD ["python", "train.py"]

```

## Step 3: Build and Run

```bash
# Build the image
docker build -t gpu-training:v1.0 .

# Check image size
docker images gpu-training:v1.0

# Run with all GPUs
docker run --rm --gpus all \
    -v $(pwd)/models:/models \
    -v $(pwd)/data:/data \
    gpu-training:v1.0

# Run with specific GPU
docker run --rm --gpus '"device=0"' \
    -v $(pwd)/models:/models \
    -v $(pwd)/data:/data \
    gpu-training:v1.0

# Run and monitor GPU
# Terminal 1: Run training
docker run --rm --gpus all \
    -v $(pwd)/models:/models \
    --name gpu-train \
    gpu-training:v1.0

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Or use DCGM
docker run --rm --gpus all \
    nvidia/dcgm-exporter:latest
```

## Success Criteria

- [ ] Container sees GPU (CUDA available: True)
- [ ] Training runs on GPU (not CPU)
- [ ] GPU utilization >70% during training
- [ ] Training completes in <2 minutes (with GPU)
- [ ] Model saved successfully to volume
- [ ] No CUDA errors or warnings
- [ ] Image size <4GB

## Stretch Goals

1. **Multi-GPU training**: Use DataParallel or DDP
2. **Mixed precision**: Implement automatic mixed precision (AMP)
3. **GPU monitoring**: Add custom GPU metrics
4. **Memory optimization**: Implement gradient checkpointing
5. **Docker Compose**: Add TensorBoard service

## Performance Comparison

Run same training on CPU and GPU:

```bash
# CPU training
docker run --rm \
    -v $(pwd)/models:/models \
    -e CUDA_VISIBLE_DEVICES="" \
    gpu-training:v1.0

# Time: ~10-15 minutes

# GPU training
docker run --rm --gpus all \
    -v $(pwd)/models:/models \
    gpu-training:v1.0

# Time: ~1-2 minutes
# Speedup: 5-10x
```

## Troubleshooting

**"CUDA not available":**
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Container Toolkit
nvidia-ctk --version

# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

**"CUDA version mismatch":**
```bash
# Check host CUDA version
nvidia-smi | grep "CUDA Version"

# Use matching PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"Out of memory":**
```python
# Reduce batch size in train.py
batch_size = 32  # Try 16 or 8

# Clear cache
torch.cuda.empty_cache()
```

## Additional Challenges

1. **Distributed training**: Train on 2 GPUs simultaneously
2. **Experiment tracking**: Add MLflow for tracking
3. **Checkpointing**: Save model after each epoch
4. **Early stopping**: Stop training when validation loss plateaus
5. **Hyperparameter tuning**: Use Ray Tune

---

**Congratulations!** You've completed the Docker containerization exercises.

**Next Module:** [Module 04: Kubernetes](../../mod-04-kubernetes/)
