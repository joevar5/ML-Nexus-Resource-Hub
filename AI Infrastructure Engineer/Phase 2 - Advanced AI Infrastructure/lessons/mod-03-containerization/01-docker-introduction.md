# Lesson 01: Docker Introduction

**Duration:** 4 hours
**Objectives:** Understand containers, Docker architecture, and run your first ML container

## What are Containers?

### The Shipping Container Analogy

Docker's name comes from shipping docks where standardized containers revolutionized cargo transport. Similarly, software containers revolutionized application deployment.

**Traditional Shipping (Pre-Containers):**
- Each cargo type required different handling
- Loading/unloading was slow and error-prone
- Different ships needed different loading equipment
- High costs and inefficiencies

**Container Shipping:**
- Standardized containers (20ft, 40ft)
- Same handling equipment everywhere
- Fast loading/unloading
- Works with ships, trains, trucks identically

**The Same Logic for Software:**
- **Before Containers:** Each application needs custom setup, different dependencies, conflicts
- **With Containers:** Package app + dependencies into standard unit, runs anywhere identically

### Containers vs Virtual Machines

#### Virtual Machines (VMs)
```
┌─────────────────────────────────────┐
│         Application A               │
│    ┌──────────────────────┐         │
│    │   Guest OS (Ubuntu)  │         │
│    │      4GB RAM         │         │
│    └──────────────────────┘         │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│         Application B               │
│    ┌──────────────────────┐         │
│    │   Guest OS (CentOS)  │         │
│    │      4GB RAM         │         │
│    └──────────────────────┘         │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│          Hypervisor (VMware)        │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│        Host OS (Linux)              │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│       Hardware (8GB RAM)            │
└─────────────────────────────────────┘
```

#### Containers
```
┌───────────────┬───────────────┬───────────────┐
│  Container A  │  Container B  │  Container C  │
│  App + Libs   │  App + Libs   │  App + Libs   │
│   (50MB)      │   (30MB)      │   (60MB)      │
└───────────────┴───────────────┴───────────────┘
┌─────────────────────────────────────────────────┐
│           Docker Engine                         │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│           Host OS (Linux)                       │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│           Hardware (8GB RAM)                    │
└─────────────────────────────────────────────────┘
```

#### Key Differences

| Feature | Virtual Machines | Containers |
|---------|-----------------|------------|
| **Size** | GBs (full OS) | MBs (app + libs) |
| **Startup** | Minutes | Seconds |
| **Performance** | Slower (virtualization overhead) | Near-native |
| **Isolation** | Strong (separate kernel) | Process-level |
| **Resource Usage** | High (multiple OS copies) | Low (shared kernel) |
| **Portability** | Limited (hypervisor-specific) | High (run anywhere Docker runs) |
| **Use Case** | Different OS needed | Same OS, different apps |

### Why Containers for Machine Learning?

1. **Dependency Hell → Dependency Heaven**
   - ML requires specific versions: PyTorch 2.0, CUDA 11.8, cuDNN 8.6
   - Container packages everything together
   - No conflicts with other projects

2. **"Works on My Machine" → "Works Everywhere"**
   - Developer laptop: Mac M1
   - Training server: Ubuntu with NVIDIA GPUs
   - Production: Kubernetes cluster
   - Same container runs identically on all

3. **Reproducibility**
   - ML research: publish container with paper
   - Anyone can reproduce exact results
   - No ambiguity about environment

4. **Scalability**
   - Deploy 100 instances of model server in seconds
   - Kubernetes orchestrates containers automatically
   - Easy horizontal scaling

5. **Resource Efficiency**
   - Run 10 different models on same machine
   - Each in isolated container
   - Minimal overhead compared to VMs

## Docker Architecture

### Core Components

```
┌──────────────────────────────────────────────────┐
│              Docker Client (CLI)                 │
│        $ docker run pytorch/pytorch              │
└─────────────────┬────────────────────────────────┘
                  │ Docker API (REST)
┌─────────────────▼────────────────────────────────┐
│              Docker Daemon                       │
│  ┌────────────────────────────────────────┐     │
│  │  Container Runtime (containerd)        │     │
│  └────────────────────────────────────────┘     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │Container │  │Container │  │Container │      │
│  │    A     │  │    B     │  │    C     │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│                                                  │
│  ┌────────────────────────────────────────┐     │
│  │      Images (Local Storage)            │     │
│  │  pytorch:2.0  tensorflow:2.10          │     │
│  └────────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

#### 1. Docker Client
- Command-line tool you interact with
- Commands: `docker run`, `docker build`, `docker ps`
- Communicates with daemon via REST API

#### 2. Docker Daemon (dockerd)
- Background service managing containers
- Builds images, runs containers, manages networks
- Listens for API requests from client

#### 3. Container Runtime
- Low-level component that actually runs containers
- Default: containerd (donated to CNCF by Docker)
- Interfaces with OS kernel

#### 4. Images
- Read-only templates for creating containers
- Stored locally after pulling/building
- Composed of layers (more on this later)

#### 5. Containers
- Running instances of images
- Isolated processes on host machine
- Have own filesystem, networking, process space

#### 6. Docker Registry
- Storage for Docker images
- Docker Hub (public registry)
- Private registries (ECR, GCR, ACR, self-hosted)

### Docker Workflow

```
1. Write Dockerfile  →  2. Build Image  →  3. Push to Registry
     (recipe)              (template)          (share)
        │                      │                    │
        └──────────────────────┴────────────────────┘
                               │
                               ▼
4. Pull Image  →  5. Run Container  →  6. Container Running
  (download)         (instantiate)         (your app)
```

**Example:**
```bash
# 1. Write Dockerfile (create file)
# 2. Build image
docker build -t my-ml-model:v1 .

# 3. Push to registry
docker push myrepo/my-ml-model:v1

# 4. Pull image (on production server)
docker pull myrepo/my-ml-model:v1

# 5. Run container
docker run -p 8000:8000 myrepo/my-ml-model:v1

# 6. Container is running and serving predictions
```

## Installing Docker

### Prerequisites

**System Requirements:**
- **Linux:** Kernel 3.10+ (most distributions supported)
- **macOS:** macOS 10.15+ (Catalina or newer)
- **Windows:** Windows 10/11 Pro, Enterprise, or Education (with WSL2)
- **RAM:** 4GB minimum, 8GB+ recommended
- **Disk:** 20GB+ free space

### Installation Options

#### Option 1: Docker Desktop (Recommended for Mac/Windows)

**Pros:**
- GUI for managing containers
- Includes Docker Engine, CLI, Compose, and Kubernetes
- Easy installation

**Cons:**
- Requires license for enterprise use (companies with >250 employees or >$10M revenue)
- Resource overhead (runs Linux VM on Mac/Windows)

**Download:** https://www.docker.com/products/docker-desktop

#### Option 2: Docker Engine (Linux, Free and Open Source)

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (avoid using sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

**CentOS/RHEL:**
```bash
# Install dependencies
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Post-Installation Verification

```bash
# Check Docker version
docker --version
# Output: Docker version 24.0.7, build afdd53b

# Check Docker info
docker info

# Run test container
docker run hello-world
# Should download and run successfully

# Check Docker Compose
docker compose version
# Output: Docker Compose version v2.23.0
```

### Troubleshooting Installation

**Issue:** "permission denied" when running docker commands
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
```

**Issue:** Docker daemon not running
```bash
# Linux: Start Docker service
sudo systemctl start docker
sudo systemctl status docker

# Mac/Windows: Start Docker Desktop application
```

**Issue:** "Cannot connect to the Docker daemon"
```bash
# Check if Docker daemon is running
sudo systemctl status docker

# Check Docker socket permissions
ls -l /var/run/docker.sock
```

## Your First Docker Commands

### Basic Commands

#### 1. Check Docker Installation
```bash
docker --version
docker info
```

#### 2. Run Your First Container
```bash
# Run official Python container
docker run python:3.11 python --version

# What happened:
# 1. Docker looked for 'python:3.11' image locally
# 2. Didn't find it, so pulled from Docker Hub
# 3. Created container from image
# 4. Ran 'python --version' inside container
# 5. Printed output and exited
```

#### 3. Run Interactive Container
```bash
# Run Python container interactively
docker run -it python:3.11 python

# Now you're inside Python REPL in container
>>> import sys
>>> print(sys.version)
>>> exit()
```

**Flags:**
- `-i`: Interactive (keep STDIN open)
- `-t`: Allocate pseudo-TTY (terminal)
- Combined: `-it` for interactive terminal

#### 4. Run Container in Background (Detached)
```bash
# Run Nginx web server in background
docker run -d -p 8080:80 nginx

# Flags:
# -d: Detached mode (background)
# -p 8080:80: Map host port 8080 to container port 80

# Visit http://localhost:8080 in browser
```

#### 5. List Running Containers
```bash
docker ps

# Output:
# CONTAINER ID   IMAGE   COMMAND                  CREATED         STATUS         PORTS                  NAMES
# a1b2c3d4e5f6   nginx   "/docker-entrypoint.…"   2 minutes ago   Up 2 minutes   0.0.0.0:8080->80/tcp   brave_tesla
```

#### 6. List All Containers (Including Stopped)
```bash
docker ps -a
```

#### 7. Stop a Container
```bash
docker stop <container_id or name>
# Example:
docker stop a1b2c3d4e5f6
# or
docker stop brave_tesla
```

#### 8. Start a Stopped Container
```bash
docker start <container_id or name>
```

#### 9. Remove a Container
```bash
docker rm <container_id or name>

# Force remove running container:
docker rm -f <container_id>
```

#### 10. List Downloaded Images
```bash
docker images

# Output:
# REPOSITORY   TAG     IMAGE ID       CREATED       SIZE
# python       3.11    1234567890ab   2 weeks ago   1.01GB
# nginx        latest  abcdef123456   3 weeks ago   142MB
```

#### 11. Remove an Image
```bash
docker rmi <image_id or name:tag>

# Example:
docker rmi python:3.11
```

#### 12. View Container Logs
```bash
docker logs <container_id or name>

# Follow logs in real-time:
docker logs -f <container_id>
```

#### 13. Execute Command in Running Container
```bash
docker exec -it <container_id> bash

# This opens a shell inside the running container
# Now you can explore filesystem, check processes, etc.
```

#### 14. Inspect Container Details
```bash
docker inspect <container_id>
# Returns JSON with all container configuration and state
```

#### 15. Pull Image Without Running
```bash
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

### Docker Command Cheat Sheet

```bash
# Images
docker pull <image>           # Download image
docker images                 # List images
docker rmi <image>            # Remove image
docker build -t <name> .      # Build image from Dockerfile

# Containers
docker run <image>            # Create and start container
docker ps                     # List running containers
docker ps -a                  # List all containers
docker stop <container>       # Stop container
docker start <container>      # Start stopped container
docker restart <container>    # Restart container
docker rm <container>         # Remove container
docker logs <container>       # View logs
docker exec -it <container> bash  # Shell into container

# Cleanup
docker system prune           # Remove unused data
docker container prune        # Remove stopped containers
docker image prune            # Remove dangling images
docker volume prune           # Remove unused volumes

# Information
docker info                   # Docker system info
docker version                # Docker version
docker inspect <container>    # Container details
docker stats                  # Container resource usage
```

## Running Your First ML Container

### Example 1: PyTorch Container

```bash
# Pull official PyTorch image
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Run interactive PyTorch session
docker run -it pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime python

# Inside Python:
>>> import torch
>>> print(torch.__version__)
2.0.0
>>> print(torch.cuda.is_available())
False  # (False if no GPU mapped to container)
>>> exit()
```

### Example 2: TensorFlow Container

```bash
# Pull TensorFlow image
docker pull tensorflow/tensorflow:latest

# Run Jupyter notebook
docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter

# Open browser to http://localhost:8888
# Token will be printed in terminal
```

### Example 3: Run Jupyter with PyTorch

```bash
# Run Jupyter with PyTorch
docker run -it \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    pytorch/pytorch:latest \
    sh -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --allow-root"

# Breakdown:
# -it: Interactive terminal
# -p 8888:8888: Map Jupyter port
# -v $(pwd):/workspace: Mount current directory
# pytorch/pytorch:latest: Image to use
# sh -c "...": Command to run
```

## Understanding Docker Images and Layers

### Image Layers

Docker images are built from layers. Each layer represents a set of filesystem changes.

```
┌───────────────────────────────────┐
│  Layer 5: pip install pytorch     │  ← 2GB (PyTorch binaries)
├───────────────────────────────────┤
│  Layer 4: pip install numpy       │  ← 50MB (NumPy)
├───────────────────────────────────┤
│  Layer 3: apt install python3-pip │  ← 100MB (pip + dependencies)
├───────────────────────────────────┤
│  Layer 2: apt update               │  ← 50MB (package lists)
├───────────────────────────────────┤
│  Layer 1: Base Ubuntu 22.04       │  ← 77MB (OS)
└───────────────────────────────────┘
Total Image Size: ~2.3GB
```

**Key Benefits:**
1. **Reusability:** Base layers shared across images
2. **Efficiency:** Only changed layers downloaded/stored
3. **Fast Builds:** Unchanged layers cached

### Inspecting Image Layers

```bash
# View image history (layers)
docker history pytorch/pytorch:latest

# Output shows each layer:
# IMAGE          CREATED BY                                      SIZE
# <missing>      pip install pytorch                             2.0GB
# <missing>      pip install numpy                               50MB
# <missing>      apt install python3-pip                         100MB
# ...
```

## Docker Naming Conventions

### Image Names
```
registry.com/username/repository:tag
```

**Examples:**
- `pytorch/pytorch:2.0.0` - Docker Hub, pytorch org, pytorch repo, tag 2.0.0
- `gcr.io/myproject/model-server:v1.2` - GCR, myproject, model-server, v1.2
- `nginx:latest` - Docker Hub (implicit), nginx repo, latest tag

**Tags:**
- `latest` - Most recent version (default if tag omitted)
- `2.0.0` - Semantic version
- `2.0.0-cuda11.7` - Version with additional info
- `dev`, `prod`, `staging` - Environment tags

### Container Names

```bash
# Docker auto-generates names like "brave_tesla" if not specified
docker run nginx
# Creates container with random name

# Specify custom name:
docker run --name my-web-server nginx
```

## Hands-On Exercise

### Exercise 1: Run Pre-Built ML Model Container

Deploy a pre-trained model using a public Docker image:

```bash
# TODO: Pull Hugging Face model serving container
docker pull huggingface/transformers-pytorch-cpu:latest

# TODO: Run container serving BERT model
docker run -p 8080:8080 huggingface/transformers-pytorch-cpu:latest

# TODO: Test inference endpoint
# curl -X POST http://localhost:8080/predict -d '{"text": "Hello world"}'

# TODO: View container logs
docker logs <container_id>

# TODO: Stop and remove container
docker stop <container_id>
docker rm <container_id>
```

### Exercise 2: Explore Container Filesystem

```bash
# TODO: Run Ubuntu container interactively
docker run -it ubuntu:22.04 bash

# Inside container:
# pwd                    # Where am I?
# ls -la                 # What files exist?
# apt update             # Update package lists
# apt install python3    # Install Python
# python3 --version      # Verify Python installed
# exit                   # Leave container
```

### Exercise 3: Mount Local Directory

```bash
# TODO: Create local directory with Python script
mkdir ~/ml-docker-test
cd ~/ml-docker-test
echo 'print("Hello from Docker!")' > hello.py

# TODO: Run container with volume mount
docker run -it -v $(pwd):/workspace python:3.11 bash

# Inside container:
# cd /workspace
# ls                     # See hello.py from host
# python hello.py        # Run script
# exit
```

## Common Issues and Solutions

### Issue 1: Port Already in Use
```bash
# Error: "bind: address already in use"
# Solution: Use different host port
docker run -p 8081:80 nginx  # Instead of 8080:80
```

### Issue 2: Container Exits Immediately
```bash
# Container runs and exits right away
# Reason: No foreground process running
# Solution: Use -it for interactive, or run long-running process
docker run -it ubuntu bash
```

### Issue 3: Cannot Access Container from Host
```bash
# Container running but can't connect
# Solution: Check port mapping
docker ps  # Verify PORTS column shows mapping
# Ensure application binds to 0.0.0.0, not 127.0.0.1
```

### Issue 4: Out of Disk Space
```bash
# Error: "no space left on device"
# Solution: Clean up Docker resources
docker system prune -a  # Remove all unused images, containers, networks
```

## Next Steps

In the next lesson, you'll learn to:
1. Write Dockerfiles from scratch
2. Build custom images for ML applications
3. Understand Dockerfile instructions in depth
4. Create production-ready images

## Self-Check Questions

Before proceeding to Lesson 02, ensure you can answer:

1. What is the difference between an image and a container?
2. What are the main benefits of containers vs VMs for ML?
3. How do you run a container in detached mode?
4. What does the `-p` flag do in `docker run`?
5. How do you access a shell inside a running container?
6. What are Docker image layers and why do they matter?

---

**Congratulations!** You've completed your introduction to Docker. You now understand container fundamentals and can run pre-built containers. In the next lesson, you'll learn to build your own images for ML applications.

**Next:** [02-dockerfile-ml-apps.md](./02-dockerfile-ml-apps.md) - Writing Dockerfiles for ML Applications
