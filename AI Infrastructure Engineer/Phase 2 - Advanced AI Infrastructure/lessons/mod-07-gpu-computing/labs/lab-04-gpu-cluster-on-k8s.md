# Lab 04: Run a GPU Pod on Kubernetes

**Duration:** 60 min  **Prerequisites:** Kubernetes cluster with GPU nodes; NVIDIA device plugin installed

## Objective
Verify GPU availability in the cluster, schedule a Pod with a GPU resource request, and run a quick inference job.

## Steps

### 1. Verify GPU nodes
```bash
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
# expect at least one node with a non-zero GPU count
```

### 2. Install NVIDIA device plugin (if not already)
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/nvidia-device-plugin.yml
```
Wait for the daemonset to be Ready.

### 3. Single-GPU pod
```yaml
# gpu-pod.yaml
apiVersion: v1
kind: Pod
metadata: { name: cuda-test }
spec:
  restartPolicy: Never
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits: { nvidia.com/gpu: 1 }
```
```bash
kubectl apply -f gpu-pod.yaml
kubectl logs cuda-test
```

### 4. PyTorch inference Job
```yaml
apiVersion: batch/v1
kind: Job
metadata: { name: pt-infer }
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: pt
          image: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
          command: ["python", "-c", "import torch; \
              print('cuda available:', torch.cuda.is_available()); \
              print('device:', torch.cuda.get_device_name(0))"]
          resources:
            limits: { nvidia.com/gpu: 1 }
```

### 5. Multi-GPU Job
```yaml
resources:
  limits: { nvidia.com/gpu: 2 }
```
Then `torchrun --standalone --nproc-per-node=2 train_ddp.py` inside the container.

### 6. Node affinity for specific GPU types
```yaml
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
```

### 7. MIG / time-slicing (if supported)
Configure the device plugin with a ConfigMap for MIG or time-slicing strategies — usually done by the cluster admin.

## Validation
- [ ] `cuda-test` pod completes Successful and logs show GPU info.
- [ ] PyTorch Job prints CUDA available True.
- [ ] Multi-GPU pod sees both GPUs (`torch.cuda.device_count() == 2`).

## Cleanup
```bash
kubectl delete -f gpu-pod.yaml
kubectl delete job pt-infer
```

## Troubleshooting
- **Pod pending: `Insufficient nvidia.com/gpu`** — Device plugin not running on the node, or all GPUs already allocated.
- **`failed to call CDI driver`** — Container runtime not configured for NVIDIA. Verify with `kubectl describe node | grep nvidia`.
- **`Forbidden: pod requires more nvidia.com/gpu than allowed`** — A ResourceQuota or LimitRange in the namespace is capping GPU requests.
