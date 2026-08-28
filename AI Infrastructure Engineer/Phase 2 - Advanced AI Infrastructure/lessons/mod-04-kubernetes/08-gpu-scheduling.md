# Lesson 08: GPU Scheduling in Kubernetes

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand how Kubernetes manages GPU resources
- Install and configure NVIDIA GPU Operator
- Schedule ML workloads on GPU-enabled nodes
- Implement GPU resource requests and limits
- Use node affinity and taints for GPU workloads
- Understand GPU sharing strategies (time-slicing, MIG)
- Monitor GPU utilization in Kubernetes clusters
- Troubleshoot common GPU scheduling issues

## Prerequisites
- Completed lessons 01-04 (Kubernetes fundamentals)
- Understanding of GPU computing basics (Module 03, Lesson 06)
- Familiarity with Docker GPU support
- Access to a cluster with GPU nodes (or minikube with GPU support)

## Introduction

GPUs are essential for modern ML infrastructure, enabling:
- **Training acceleration**: 10-100x faster than CPUs for deep learning
- **Inference optimization**: Lower latency for production models
- **Cost efficiency**: Better throughput per dollar for ML workloads
- **Scale**: Handle larger models and datasets

However, GPUs in Kubernetes require special configuration:
- **Device plugins**: Expose GPUs as schedulable resources
- **Drivers**: NVIDIA/AMD drivers must be installed on nodes
- **Resource management**: Prevent over-allocation
- **Monitoring**: Track utilization and performance

**Real-world examples:**
- **OpenAI**: Runs large-scale GPU clusters on Kubernetes
- **Spotify**: Uses K8s GPU scheduling for ML training pipelines
- **Uber**: Schedules thousands of GPU jobs daily via K8s
- **Netflix**: Runs recommendation model training on K8s with GPUs

## 1. GPU Support in Kubernetes

### 1.1 How Kubernetes Discovers GPUs

Kubernetes uses the **Device Plugin Framework** to expose specialized hardware:

```yaml
# Device plugins advertise resources to kubelet
# GPUs become schedulable resources like CPU/memory

Node Resources:
  cpu: 32 cores
  memory: 256Gi
  nvidia.com/gpu: 8        # GPUs advertised by device plugin
  nvidia.com/mig-1g.5gb: 56 # MIG instances (if enabled)
```

**Architecture:**

```
┌─────────────────────────────────────────┐
│           Kubernetes Node               │
│                                         │
│  ┌────────────┐      ┌──────────────┐  │
│  │  kubelet   │◄────►│ Device Plugin│  │
│  │            │      │  (NVIDIA)    │  │
│  └────────────┘      └──────┬───────┘  │
│                             │          │
│                             ▼          │
│                      ┌──────────────┐  │
│                      │ GPU Devices  │  │
│                      │ /dev/nvidia0 │  │
│                      │ /dev/nvidia1 │  │
│                      └──────────────┘  │
└─────────────────────────────────────────┘
```

### 1.2 NVIDIA GPU Operator

The **NVIDIA GPU Operator** automates GPU management in K8s:

**What it does:**
- Installs NVIDIA drivers on nodes
- Deploys the NVIDIA device plugin
- Configures the container runtime (containerd/Docker)
- Sets up GPU monitoring with DCGM (Data Center GPU Manager)
- Manages GPU Feature Discovery (GFD)

**Components installed:**
```yaml
Components:
  - NVIDIA Driver Container: Installs drivers on each node
  - NVIDIA Container Toolkit: Enables GPU access in containers
  - NVIDIA Device Plugin: Advertises GPUs to Kubernetes
  - NVIDIA DCGM Exporter: Exports GPU metrics to Prometheus
  - GPU Feature Discovery: Labels nodes with GPU capabilities
  - Node Feature Discovery: Detects node hardware features
```

## 2. Installing NVIDIA GPU Operator

### 2.1 Prerequisites

**On GPU Nodes:**
```bash
# 1. Verify GPU is present
lspci | grep -i nvidia
# Output: 00:1e.0 3D controller: NVIDIA Corporation Tesla V100 (rev a1)

# 2. Check kernel version (must be compatible)
uname -r
# Output: 5.10.0-18-amd64

# 3. Ensure nouveau driver is disabled (conflicts with NVIDIA)
lsmod | grep nouveau
# Should return nothing

# 4. Verify containerd is configured for GPUs
cat /etc/containerd/config.toml | grep nvidia
```

### 2.2 Install GPU Operator with Helm

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Create namespace
kubectl create namespace gpu-operator

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true

# Watch installation
kubectl get pods -n gpu-operator -w
```

**Expected output:**
```
NAME                                       READY   STATUS    RESTARTS   AGE
gpu-feature-discovery-abcde                1/1     Running   0          2m
gpu-operator-1234567890-xyz                1/1     Running   0          3m
nvidia-container-toolkit-daemonset-fghij   1/1     Running   0          2m
nvidia-dcgm-exporter-klmno                 1/1     Running   0          2m
nvidia-device-plugin-daemonset-pqrst       1/1     Running   0          2m
nvidia-driver-daemonset-uvwxy              1/1     Running   0          5m
```

### 2.3 Verify GPU Nodes

```bash
# Check node labels (added by GPU Feature Discovery)
kubectl get nodes --show-labels | grep nvidia

# Check node capacity
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Output:
#  nvidia.com/gpu:     8
#  nvidia.com/gpu:     8  (allocatable)

# View GPU details
kubectl describe node <gpu-node-name>
```

**Node labels added automatically:**
```yaml
Labels:
  nvidia.com/cuda.driver.major=525
  nvidia.com/cuda.driver.minor=116
  nvidia.com/cuda.driver.rev=04
  nvidia.com/cuda.runtime.major=12
  nvidia.com/cuda.runtime.minor=0
  nvidia.com/gpu.count=8
  nvidia.com/gpu.product=Tesla-V100-SXM2-32GB
  nvidia.com/gpu.memory=32768
  nvidia.com/mig.strategy=single
```

## 3. Scheduling GPU Workloads

### 3.1 Basic GPU Pod

**Request a single GPU:**

```yaml
# pytorch-training-gpu.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training-gpu
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["python"]
    args: ["-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"]
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
      requests:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "4"
  restartPolicy: OnFailure
```

**Deploy and verify:**

```bash
kubectl apply -f pytorch-training-gpu.yaml
kubectl logs pytorch-training-gpu

# Output:
# CUDA available: True
# GPU count: 1
```

### 3.2 Multi-GPU Training Job

**Request multiple GPUs for distributed training:**

```yaml
# multi-gpu-training.yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-gpu-training
spec:
  containers:
  - name: distributed-training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["torchrun"]
    args:
      - "--nproc_per_node=4"  # 4 processes (one per GPU)
      - "train_distributed.py"
    resources:
      limits:
        nvidia.com/gpu: 4  # Request 4 GPUs
      requests:
        nvidia.com/gpu: 4
        memory: "64Gi"
        cpu: "16"
    volumeMounts:
    - name: training-data
      mountPath: /data
    - name: model-output
      mountPath: /output
  volumes:
  - name: training-data
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name: model-output
    persistentVolumeClaim:
      claimName: model-output-pvc
  restartPolicy: OnFailure
```

### 3.3 GPU Inference Deployment

**Production ML inference with GPU:**

```yaml
# gpu-inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-inference-gpu
spec:
  replicas: 3  # 3 replicas, each gets 1 GPU
  selector:
    matchLabels:
      app: model-inference-gpu
  template:
    metadata:
      labels:
        app: model-inference-gpu
    spec:
      containers:
      - name: inference-server
        image: myregistry.io/bert-inference:v1.0
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: /models/bert-base
        - name: BATCH_SIZE
          value: "32"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"  # Use first GPU
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: model-inference-gpu-service
spec:
  selector:
    app: model-inference-gpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## 4. Node Affinity and GPU Scheduling

### 4.1 Node Selectors for GPU Nodes

**Schedule only on GPU-enabled nodes:**

```yaml
# gpu-job-node-selector.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-job-node-selector
spec:
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.product: Tesla-V100-SXM2-32GB  # Specific GPU model
      containers:
      - name: training
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 1
      restartPolicy: Never
```

### 4.2 Node Affinity for Advanced Scheduling

**Prefer specific GPU types, fallback to others:**

```yaml
# gpu-affinity.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-affinity-pod
spec:
  affinity:
    nodeAffinity:
      # REQUIRED: Must have GPU
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.present
            operator: In
            values: ["true"]

      # PREFERRED: Prefer V100, but accept A100 or T4
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: ["Tesla-V100-SXM2-32GB"]
      - weight: 80
        preference:
          matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: ["NVIDIA-A100-SXM4-40GB"]
      - weight: 50
        preference:
          matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: ["Tesla-T4"]

  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: OnFailure
```

### 4.3 Taints and Tolerations for GPU Nodes

**Reserve GPU nodes only for GPU workloads:**

```bash
# Taint GPU nodes to prevent non-GPU pods from scheduling
kubectl taint nodes <gpu-node-name> nvidia.com/gpu=present:NoSchedule

# Verify taint
kubectl describe node <gpu-node-name> | grep Taints
# Output: Taints: nvidia.com/gpu=present:NoSchedule
```

**Tolerate GPU taints in pod spec:**

```yaml
# gpu-toleration.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-toleration-pod
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"

  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: OnFailure
```

**Result:** Only pods with GPU requests AND toleration can schedule on GPU nodes.

## 5. GPU Sharing Strategies

### 5.1 Time-Slicing GPUs

**Problem:** GPUs are expensive; underutilized GPUs waste money.

**Solution:** Time-slicing allows multiple pods to share a single GPU.

**Configure time-slicing:**

```yaml
# gpu-time-slicing-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # Each GPU appears as 4 schedulable resources
```

**Apply configuration:**

```bash
kubectl apply -f gpu-time-slicing-config.yaml

# Restart device plugin to apply changes
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-operator

# Verify: each GPU now shows as 4 resources
kubectl describe node <gpu-node> | grep nvidia.com/gpu
# Output:
#  nvidia.com/gpu: 32  (8 GPUs × 4 replicas = 32 schedulable units)
```

**Deploy 4 pods sharing 1 GPU:**

```yaml
# time-sliced-inference.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shared-gpu-inference
spec:
  replicas: 4  # 4 pods share 1 physical GPU
  selector:
    matchLabels:
      app: shared-inference
  template:
    metadata:
      labels:
        app: shared-inference
    spec:
      containers:
      - name: inference
        image: myregistry.io/small-model-inference:v1
        resources:
          limits:
            nvidia.com/gpu: 1  # Each pod requests 1 "slice"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
  restartPolicy: Always
```

**Caution:**
- Time-slicing does NOT provide GPU memory isolation
- One pod can starve others if it uses too much memory
- Best for inference workloads with predictable memory usage
- Monitor GPU memory closely

### 5.2 Multi-Instance GPU (MIG)

**What is MIG?**
- Available on NVIDIA A100, A30, H100 GPUs
- **Hardware partitioning** of a single GPU into isolated instances
- Each MIG instance has dedicated memory and compute

**MIG profiles (A100 40GB example):**
```
Profile      | Instances | Memory/Instance | Use Case
-------------|-----------|-----------------|---------------------------
1g.5gb       | 7         | 5 GB            | Small inference, dev/test
2g.10gb      | 3         | 10 GB           | Medium models
3g.20gb      | 2         | 20 GB           | Large inference
4g.20gb      | 1         | 20 GB           | Training small models
7g.40gb      | 1         | 40 GB           | Full GPU (no partitioning)
```

**Enable MIG on GPU nodes:**

```bash
# SSH to GPU node
nvidia-smi -mig 1  # Enable MIG mode

# Create MIG instances (example: 7 × 1g.5gb)
nvidia-smi mig -cgi 19,19,19,19,19,19,19  # Create 7 GPU instances
nvidia-smi mig -cci                        # Create compute instances

# Verify
nvidia-smi -L
# Output:
# GPU 0: NVIDIA A100 (UUID: GPU-xxx)
#   MIG 1g.5gb Device 0: (UUID: MIG-yyy-0)
#   MIG 1g.5gb Device 1: (UUID: MIG-yyy-1)
#   ...
```

**Configure GPU Operator for MIG:**

```yaml
# mig-strategy-config.yaml
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set mig.strategy=mixed \  # Support both MIG and non-MIG
  --set devicePlugin.enabled=true
```

**Schedule workloads on MIG instances:**

```yaml
# mig-inference-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference
spec:
  containers:
  - name: inference
    image: nvcr.io/nvidia/pytorch:23.01-py3
    resources:
      limits:
        nvidia.com/mig-1g.5gb: 1  # Request 1 MIG instance (1g.5gb profile)
      requests:
        nvidia.com/mig-1g.5gb: 1
        memory: "4Gi"
        cpu: "2"
  restartPolicy: OnFailure
```

**Comparison:**

| Feature              | Time-Slicing            | MIG                           |
|----------------------|-------------------------|-------------------------------|
| Isolation            | None (shared memory)    | Hardware isolation            |
| Memory Safety        | No                      | Yes (dedicated memory)        |
| GPU Support          | All NVIDIA GPUs         | A100, A30, H100 only          |
| Overhead             | Context switching       | None                          |
| Use Case             | Dev/test, small models  | Production multi-tenancy      |
| Cost                 | Better utilization      | Better isolation              |

## 6. Monitoring GPU Utilization

### 6.1 NVIDIA DCGM Exporter (Prometheus Metrics)

**Install DCGM Exporter (included in GPU Operator):**

```bash
# Verify DCGM Exporter is running
kubectl get pods -n gpu-operator | grep dcgm

# Check metrics endpoint
kubectl port-forward -n gpu-operator <dcgm-exporter-pod> 9400:9400
curl localhost:9400/metrics | grep DCGM
```

**Key metrics exposed:**

```prometheus
# GPU utilization (0-100%)
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-xxx"} 87

# GPU memory used (MB)
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-xxx"} 28672

# GPU temperature (Celsius)
DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-xxx"} 68

# GPU power usage (Watts)
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-xxx"} 245

# GPU memory bandwidth utilization (%)
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-xxx"} 42

# SM (Streaming Multiprocessor) occupancy
DCGM_FI_DEV_SM_OCCUPANCY{gpu="0",UUID="GPU-xxx"} 0.76
```

### 6.2 Grafana Dashboard for GPU Monitoring

**Import NVIDIA DCGM dashboard:**

```bash
# Dashboard ID: 12239 (NVIDIA DCGM Exporter Dashboard)
# URL: https://grafana.com/grafana/dashboards/12239

# Or use this JSON config
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-gpu-dashboard
  namespace: monitoring
data:
  gpu-dashboard.json: |
    {
      "dashboard": {
        "title": "GPU Cluster Monitoring",
        "panels": [
          {
            "title": "GPU Utilization",
            "targets": [{"expr": "DCGM_FI_DEV_GPU_UTIL"}]
          },
          {
            "title": "GPU Memory Usage",
            "targets": [{"expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100"}]
          },
          {
            "title": "GPU Temperature",
            "targets": [{"expr": "DCGM_FI_DEV_GPU_TEMP"}]
          }
        ]
      }
    }
EOF
```

**Prometheus alerts for GPU issues:**

```yaml
# gpu-alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-gpu-alerts
  namespace: monitoring
data:
  gpu-alerts.rules: |
    groups:
    - name: gpu-alerts
      interval: 30s
      rules:
      # GPU temperature too high
      - alert: GPUHighTemperature
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} temperature is high"
          description: "GPU temperature is {{ $value }}°C"

      # GPU memory exhaustion
      - alert: GPUMemoryHigh
        expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) > 0.95
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} memory is nearly full"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"

      # GPU not being utilized (waste)
      - alert: GPUUnderutilized
        expr: DCGM_FI_DEV_GPU_UTIL < 10
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "GPU {{ $labels.gpu }} is underutilized"
          description: "GPU utilization is only {{ $value }}%"
```

### 6.3 kubectl Plugin for GPU Monitoring

**Install kubectl-gpu plugin:**

```bash
# Install krew (kubectl plugin manager)
curl -fsSL https://krew.sh/install | bash

# Install gpu plugin
kubectl krew install resource-capacity

# View GPU allocation across cluster
kubectl resource-capacity --pods --util --sort gpu.nvidia.com
```

**Output example:**

```
NODE            GPU TOTAL   GPU ALLOCATED   GPU UTILIZATION
gpu-node-1      8           6 (75%)         87%
gpu-node-2      8           8 (100%)        92%
gpu-node-3      8           4 (50%)         45%
```

## 7. Best Practices for GPU Scheduling

### 7.1 Resource Requests and Limits

**Always set GPU requests equal to limits:**

```yaml
# ✅ GOOD: Request = Limit
resources:
  limits:
    nvidia.com/gpu: 2
  requests:
    nvidia.com/gpu: 2  # Must be equal
```

```yaml
# ❌ BAD: Request ≠ Limit (will cause errors)
resources:
  limits:
    nvidia.com/gpu: 2
  requests:
    nvidia.com/gpu: 1  # ERROR: GPUs are not overcommitable
```

**Why?** Kubernetes treats GPUs as **non-overcommitable extended resources**. Requests and limits MUST be equal.

### 7.2 Set Appropriate CPU and Memory

GPUs need sufficient CPU and memory to avoid bottlenecks:

```yaml
# ✅ GOOD: Balanced resources
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "16Gi"  # Enough for model + batch
    cpu: "8"        # Enough for data preprocessing
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
    cpu: "8"
```

```yaml
# ❌ BAD: GPU bottlenecked by CPU/memory
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "2Gi"   # Too little memory
    cpu: "1"        # CPU will starve GPU
```

**Rule of thumb:**
- **Training:** 8-16 CPU cores per GPU, 32-64 GB RAM per GPU
- **Inference:** 4-8 CPU cores per GPU, 8-16 GB RAM per GPU

### 7.3 Use PodDisruptionBudgets for Critical Workloads

Protect long-running training jobs from disruption:

```yaml
# pdb-gpu-training.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gpu-training-pdb
spec:
  minAvailable: 2  # At least 2 pods must remain during disruptions
  selector:
    matchLabels:
      app: distributed-training
```

### 7.4 Use Node Pools for GPU Isolation

**Separate GPU node pools by use case:**

```bash
# Production inference: V100 GPUs
kubectl label nodes gpu-prod-1 gpu-prod-2 gpu-prod-3 \
  workload=inference \
  gpu-type=v100

# Training/experimentation: T4 GPUs
kubectl label nodes gpu-train-1 gpu-train-2 \
  workload=training \
  gpu-type=t4

# Development: Shared A100 with MIG
kubectl label nodes gpu-dev-1 \
  workload=development \
  gpu-type=a100-mig
```

**Schedule to appropriate pools:**

```yaml
# inference-deployment.yaml
spec:
  template:
    spec:
      nodeSelector:
        workload: inference
        gpu-type: v100
      containers:
      - name: inference
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 7.5 Implement Resource Quotas

Prevent GPU resource exhaustion by teams:

```yaml
# gpu-resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota-team-ml
  namespace: team-ml
spec:
  hard:
    requests.nvidia.com/gpu: "16"  # Max 16 GPUs for team-ml
    limits.nvidia.com/gpu: "16"
    pods: "50"
```

```yaml
# gpu-limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limit-range
  namespace: team-ml
spec:
  limits:
  - max:
      nvidia.com/gpu: "4"  # Max 4 GPUs per pod
    min:
      nvidia.com/gpu: "1"  # Min 1 GPU per pod
    type: Container
```

## 8. Troubleshooting GPU Scheduling Issues

### 8.1 Pod Stuck in Pending State

**Symptom:**

```bash
kubectl get pods
# NAME                READY   STATUS    RESTARTS   AGE
# gpu-training-pod    0/1     Pending   0          10m
```

**Diagnose:**

```bash
kubectl describe pod gpu-training-pod | grep -A 10 Events

# Common reasons:
# 1. Insufficient GPU resources
#    Warning  FailedScheduling  pod unschedulable: Insufficient nvidia.com/gpu
#
# 2. No nodes with GPUs
#    Warning  FailedScheduling  no nodes available with label nvidia.com/gpu.present=true
#
# 3. Taints not tolerated
#    Warning  FailedScheduling  node(s) had taint {nvidia.com/gpu: present}, that the pod didn't tolerate
```

**Solutions:**

```bash
# 1. Check GPU availability across cluster
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# 2. Verify GPU operator is running
kubectl get pods -n gpu-operator

# 3. Check if GPU nodes are cordoned
kubectl get nodes | grep SchedulingDisabled

# 4. Review pod tolerations and node selectors
kubectl get pod gpu-training-pod -o yaml | grep -A 10 tolerations
```

### 8.2 GPU Not Detected in Container

**Symptom:**

```python
import torch
print(torch.cuda.is_available())  # False
```

**Diagnose:**

```bash
# 1. Verify GPU is allocated to pod
kubectl describe pod <pod-name> | grep nvidia.com/gpu
# Should show: nvidia.com/gpu: 1

# 2. Check container runtime configuration
kubectl exec <pod-name> -- nvidia-smi
# Should show GPU details

# 3. Verify NVIDIA container toolkit
kubectl exec <pod-name> -- ls -la /dev/nvidia*
# Should list: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm
```

**Solutions:**

```bash
# 1. Ensure nvidia-container-toolkit is installed on nodes
ssh <node> "which nvidia-container-toolkit"

# 2. Verify containerd runtime configuration
ssh <node> "cat /etc/containerd/config.toml | grep nvidia"

# 3. Restart device plugin daemonset
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-operator

# 4. Check pod spec includes GPU resource request
kubectl get pod <pod-name> -o yaml | grep -A 5 resources
```

### 8.3 GPU Memory Errors (OOM)

**Symptom:**

```bash
kubectl logs <pod-name>
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnose:**

```bash
# Check GPU memory usage
kubectl exec <pod-name> -- nvidia-smi

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1234      C   python                          31000MiB |
# +-----------------------------------------------------------------------------+
```

**Solutions:**

```yaml
# 1. Reduce batch size or model size in code

# 2. Request more GPU memory (use A100 instead of V100)
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB  # 80GB model
  resources:
    limits:
      nvidia.com/gpu: 1

# 3. Use gradient accumulation (training)
# Accumulate gradients over multiple batches to simulate larger batch sizes

# 4. Use MIG for smaller workloads
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1  # Use 20GB MIG instance instead of full GPU
```

### 8.4 GPU Utilization is Low

**Symptom:**

GPU utilization showing 15-30% despite running training job.

**Diagnose:**

```bash
# Check if CPU/memory is bottlenecking
kubectl top pod <pod-name>
# NAME           CPU(cores)   MEMORY(bytes)
# training-pod   7950m        15Gi  # CPU maxed out at 8 cores

# Check I/O wait
kubectl exec <pod-name> -- iostat -x 1 5
```

**Solutions:**

```yaml
# 1. Increase CPU allocation
resources:
  limits:
    nvidia.com/gpu: 1
    cpu: "16"  # Increase from 8 to 16
    memory: "32Gi"

# 2. Use faster storage for data loading
volumeMounts:
- name: training-data
  mountPath: /data
volumes:
- name: training-data
  persistentVolumeClaim:
    claimName: fast-ssd-pvc  # Use SSD instead of HDD

# 3. Optimize data loading in code
# - Use DataLoader with num_workers=4-8
# - Prefetch data to GPU asynchronously
# - Use mixed precision training (FP16)
```

## 9. Hands-On Exercise: Deploy GPU-Accelerated ML Pipeline

**Objective:** Deploy a complete GPU-accelerated training pipeline with:
- GPU scheduling with node affinity
- Persistent storage for datasets and models
- GPU monitoring with Prometheus
- Resource quotas

**Step 1: Create namespace with GPU quota**

```bash
kubectl create namespace gpu-training

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: gpu-training
spec:
  hard:
    requests.nvidia.com/gpu: "4"
    limits.nvidia.com/gpu: "4"
EOF
```

**Step 2: Create PVCs for data and models**

```yaml
# training-pvcs.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: gpu-training
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-output-pvc
  namespace: gpu-training
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

**Step 3: Deploy training job with GPU**

```yaml
# gpu-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: resnet50-training
  namespace: gpu-training
spec:
  template:
    metadata:
      labels:
        app: resnet50-training
    spec:
      restartPolicy: OnFailure

      # Schedule on V100 GPUs with affinity
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: ["Tesla-V100-SXM2-32GB"]

      # Tolerate GPU node taints
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: present
        effect: NoSchedule

      containers:
      - name: pytorch-training
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command: ["python", "train_resnet50.py"]
        args:
          - "--data-dir=/data/imagenet"
          - "--output-dir=/output"
          - "--epochs=90"
          - "--batch-size=256"
          - "--learning-rate=0.1"

        resources:
          limits:
            nvidia.com/gpu: 2  # Use 2 GPUs
            memory: "32Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: 2
            memory: "32Gi"
            cpu: "16"

        volumeMounts:
        - name: training-data
          mountPath: /data
          readOnly: true
        - name: model-output
          mountPath: /output

        env:
        - name: NCCL_DEBUG
          value: "INFO"  # Enable NCCL debugging for multi-GPU
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"

      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: model-output
        persistentVolumeClaim:
          claimName: model-output-pvc
```

**Step 4: Monitor GPU usage**

```bash
# Apply all resources
kubectl apply -f training-pvcs.yaml
kubectl apply -f gpu-training-job.yaml

# Watch job progress
kubectl get jobs -n gpu-training -w

# Check GPU utilization
kubectl exec -n gpu-training <pod-name> -- nvidia-smi

# View logs
kubectl logs -n gpu-training <pod-name> -f

# Check resource usage
kubectl top pod -n gpu-training

# View Prometheus metrics (if port-forwarded)
curl localhost:9400/metrics | grep "DCGM_FI_DEV_GPU_UTIL.*pod.*resnet50"
```

**Expected output:**

```
GPU utilization: 95-100%
Training speed: ~500 images/second
Memory usage: 28GB / 32GB
Time to complete: ~24 hours
```

## 10. Summary

### Key Takeaways

✅ **GPU scheduling requires specialized components:**
- NVIDIA GPU Operator automates driver and device plugin installation
- Device plugins advertise GPUs as schedulable resources
- GPU nodes should be labeled and tainted for proper isolation

✅ **Resource management:**
- GPU requests must equal limits (non-overcommitable)
- Always provide sufficient CPU and memory to avoid bottlenecks
- Use resource quotas to prevent exhaustion

✅ **GPU sharing strategies:**
- **Time-slicing:** Share GPUs without isolation (dev/test)
- **MIG:** Hardware-partitioned GPU instances (production multi-tenancy)

✅ **Monitoring and troubleshooting:**
- DCGM Exporter provides detailed GPU metrics to Prometheus
- Common issues: insufficient resources, missing tolerations, OOM errors
- Use `nvidia-smi` in pods to debug GPU detection

✅ **Production best practices:**
- Use node affinity to schedule on appropriate GPU types
- Implement PodDisruptionBudgets for critical workloads
- Create separate node pools for inference vs training
- Monitor GPU utilization to optimize costs

### Real-World Impact

Companies running ML at scale rely heavily on Kubernetes GPU scheduling:

- **OpenAI:** Trains GPT models on thousands of GPUs orchestrated by K8s
- **Uber:** Schedules 10,000+ GPU jobs/day for ML experimentation
- **Spotify:** Uses K8s GPU scheduling for personalization models
- **NVIDIA:** Recommends GPU Operator as standard for production K8s

### What's Next?

In the next lesson, we'll cover **Monitoring and Troubleshooting** Kubernetes clusters, including:
- Advanced `kubectl` debugging techniques
- Log aggregation with Fluentd/Loki
- Metrics collection with Prometheus
- Distributed tracing for ML services
- Common production issues and solutions

## Self-Check Questions

1. What components does the NVIDIA GPU Operator install?
2. Why must GPU resource requests equal limits in Kubernetes?
3. What's the difference between GPU time-slicing and MIG?
4. How do you prevent non-GPU workloads from scheduling on GPU nodes?
5. What Prometheus metrics are most important for GPU monitoring?
6. How would you troubleshoot a pod stuck in Pending due to insufficient GPUs?
7. What's the recommended CPU and memory allocation per GPU for training workloads?
8. How do you schedule a pod on a specific GPU model (e.g., V100)?

## Additional Resources

### Official Documentation
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [NVIDIA Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)

### Tutorials and Guides
- [Running GPU Jobs on Kubernetes](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [GPU Time-Slicing Configuration](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-sharing.html)
- [DCGM Exporter Metrics Guide](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)

### Tools
- [kubectl-gpu plugin](https://github.com/kubernetes-sigs/kubectl-gpu)
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm)
- [Grafana DCGM Dashboard](https://grafana.com/grafana/dashboards/12239)

### Community Resources
- [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/)
- [Cloud Native Computing Foundation (CNCF) Blog](https://www.cncf.io/blog/)
- [Kubernetes Slack #sig-scheduling](https://kubernetes.slack.com/)

---

**Congratulations!** You now understand how to schedule and manage GPU workloads in Kubernetes, a critical skill for ML infrastructure engineers. Practice deploying GPU workloads and monitoring their performance to solidify these concepts.
