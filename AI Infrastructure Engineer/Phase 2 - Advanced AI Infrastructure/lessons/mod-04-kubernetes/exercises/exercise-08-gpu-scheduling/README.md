# Exercise 08: GPU Scheduling Patterns

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Cluster with GPU nodes; exercise 10 of mod-07 (sharing strategies)

## Objective

Implement three GPU scheduling patterns: dedicated GPU pods, MIG-partitioned pods, and time-sliced GPU sharing. Add node affinity, taints, and PriorityClass for production-quality scheduling.

## Requirements

1. NVIDIA Device Plugin deployed.
2. Tainted GPU node group (only GPU workloads admitted).
3. Three deployment patterns demonstrated.
4. PriorityClass for inference (high) vs training (low) — training preempted when inference needs capacity.
5. PodAntiAffinity spreading inference replicas across nodes.

## Step-by-step

### Step 1 — Verify GPU cluster (15 min)
```bash
kubectl get nodes -L nvidia.com/gpu.product
# expect at least one node with a GPU product label
kubectl describe node <gpu-node> | grep -i nvidia.com/gpu
```

### Step 2 — Taint GPU nodes (15 min)
```bash
kubectl taint node <gpu-node> workload=gpu:NoSchedule
```
Now only pods with matching toleration land here.

### Step 3 — Pattern 1: dedicated GPU pod (30 min)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: llm-serve }
spec:
  replicas: 2
  selector: { matchLabels: { app: llm-serve } }
  template:
    metadata: { labels: { app: llm-serve } }
    spec:
      tolerations:
        - { key: workload, value: gpu, effect: NoSchedule }
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-L40S
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.5.0
          resources:
            limits: { nvidia.com/gpu: 1, cpu: 8, memory: 32Gi }
          args: ["--model", "mistralai/Mistral-7B-Instruct-v0.3"]
```

### Step 4 — Pattern 2: MIG partition (30 min)
With MIG mode + device plugin config (see mod-07 ex-10):
```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1
```
Each pod gets a hardware-isolated slice. Verify 4-7 pods fit on one A100 depending on slice mix.

### Step 5 — Pattern 3: time-sliced sharing (15 min)
With device plugin time-slicing enabled (4 replicas per GPU):
```yaml
resources:
  limits: { nvidia.com/gpu: 1 }   # gets 1/4 of actual GPU
```
4 pods schedule on 1 GPU. For dev only.

### Step 6 — PriorityClass + preemption (30 min)
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: { name: inference-critical }
value: 1000
globalDefault: false
description: "Inference workloads; preempts training"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: { name: training-batch }
value: 100
description: "Training; preemptible by inference"
```
Set `spec.priorityClassName` on each Deployment. Fill GPUs with training; deploy inference; observe preemption.

### Step 7 — Spread across nodes (15 min)
```yaml
spec:
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: ScheduleAnyway
      labelSelector: { matchLabels: { app: llm-serve } }
```
Now 2 inference replicas don't both land on the same node.

## Deliverables

1. All 3 pattern manifests.
2. PriorityClass definitions + a demo of preemption.
3. `GPU_SCHEDULING.md` documenting when to use each pattern.

## Validation

- [ ] Dedicated GPU pod runs on the GPU node and not elsewhere.
- [ ] MIG / time-slicing successfully schedule multiple pods on one GPU.
- [ ] Deploying inference preempts a training pod (verify with `kubectl describe pod` showing `Preempting`).
- [ ] Two inference replicas land on different nodes.

## Common pitfalls

- **Tolerations without nodeSelector** — Pod tolerates the taint but doesn't prefer the GPU node; lands elsewhere.
- **MIG instances mismatch** — Asking for `mig-1g.5gb` when cluster has `mig-2g.10gb` only → unschedulable forever.
- **Preemption causes OOM during model load** — Add `preemptionPolicy: Never` to inference if model load is expensive.
