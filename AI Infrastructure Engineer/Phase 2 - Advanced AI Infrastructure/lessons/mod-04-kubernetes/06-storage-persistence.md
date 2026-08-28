# Lesson 06: Storage and Persistence in Kubernetes

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand Kubernetes storage concepts (Volumes, PersistentVolumes, PersistentVolumeClaims)
- Configure different volume types for ML workloads
- Implement StatefulSets for stateful ML applications
- Design storage strategies for ML datasets, models, and checkpoints
- Use StorageClasses for dynamic provisioning
- Optimize storage performance for training and inference
- Handle backup and disaster recovery for ML data

## Prerequisites
- Completed lessons 01-05 (Kubernetes fundamentals and networking)
- Understanding of Linux file systems
- Familiarity with cloud storage (S3, GCS, Azure Blob)
- Basic understanding of ML training workflows

## Introduction

Storage is critical for ML infrastructure because:
- **Training data:** Datasets can be terabytes in size
- **Model checkpoints:** Must persist if pods restart during training
- **Model artifacts:** Trained models need durable storage
- **Logs and metrics:** Experiment tracking requires persistent storage
- **Inference caching:** Speed up predictions with cached results

**Challenges:**
- Pods are ephemeral (data lost when pod dies)
- Multi-node clusters need shared storage
- Performance: GPUs can be bottlenecked by slow I/O
- Cost: Fast storage (NVMe SSD) is expensive

**Real-world examples:**
- **OpenAI:** Stores training checkpoints on distributed file systems
- **Uber:** Uses persistent volumes for ML model registry and feature store
- **Spotify:** Persists ML experiment data in K8s PersistentVolumes
- **Netflix:** Stores recommendation models in S3-backed persistent storage

## 1. Kubernetes Storage Concepts

### 1.1 Volumes vs PersistentVolumes

**Volumes:**
- Defined in Pod spec
- Lifecycle tied to Pod (deleted with Pod)
- Simple, but not durable

**PersistentVolumes (PV):**
- Cluster-wide storage resources
- Independent lifecycle (survives Pod deletion)
- Can be reclaimed or retained

**PersistentVolumeClaims (PVC):**
- Request for storage by users
- Abstracts storage details from users
- Binds to a matching PV

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│                    Cluster                      │
│                                                 │
│  ┌───────────────┐         ┌────────────────┐  │
│  │  User/Pod     │         │ StorageClass   │  │
│  │               │         │ (AWS EBS, etc) │  │
│  └───────┬───────┘         └────────┬───────┘  │
│          │                          │          │
│          │ 1. Request               │ 4. Auto  │
│          │    Storage               │    Provision│
│          ▼                          ▼          │
│  ┌──────────────┐          ┌────────────────┐  │
│  │    PVC       │ 2. Bind  │       PV       │  │
│  │ (100Gi SSD)  │◄─────────┤ (100Gi EBS)    │  │
│  └──────┬───────┘          └────────┬───────┘  │
│         │                           │          │
│         │ 3. Mount                  │ 5. Attach│
│         ▼                           ▼          │
│  ┌──────────────┐          ┌────────────────┐  │
│  │    Pod       │          │  Cloud Storage │  │
│  │ /data → PVC  │          │  (AWS EBS vol) │  │
│  └──────────────┘          └────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 1.2 Volume Types

**Common volume types for ML:**

| Volume Type    | Use Case                          | Performance | Durability | Shared Access |
|----------------|-----------------------------------|-------------|------------|---------------|
| emptyDir       | Temp scratch space                | High        | No         | Same pod only |
| hostPath       | Access node filesystem (dev only) | Very High   | Node-level | Same node only|
| PVC (EBS/PD)   | Model checkpoints, single-pod     | Medium      | Yes        | No (RWO)      |
| PVC (EFS/GCE)  | Shared datasets, multi-pod        | Medium      | Yes        | Yes (RWX)     |
| NFS            | Legacy shared storage             | Medium-Low  | Yes        | Yes (RWX)     |
| CSI (S3/GCS)   | Massive datasets, object storage  | Variable    | Yes        | Yes (RWX)     |

## 2. Basic Volumes

### 2.1 emptyDir (Temporary Storage)

**Use case:** Scratch space for data preprocessing, temporary model files.

```yaml
# emptydir-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-with-scratch
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["python", "train.py"]
    volumeMounts:
    - name: scratch
      mountPath: /tmp/scratch  # Fast temporary storage

  volumes:
  - name: scratch
    emptyDir:
      sizeLimit: 50Gi  # Limit size to 50GB
      medium: Memory   # Use RAM instead of disk (faster, but expensive)
```

**Characteristics:**
- Created when Pod starts
- **Deleted when Pod is removed** (not durable!)
- Can use node disk or RAM (`medium: Memory`)
- Good for: preprocessing pipelines, temporary caches

### 2.2 hostPath (Node Filesystem)

**Use case:** Access local SSD on GPU nodes for fast I/O.

```yaml
# hostpath-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-with-local-ssd
spec:
  nodeSelector:
    storage: local-ssd  # Schedule on nodes with local SSD

  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    volumeMounts:
    - name: local-ssd
      mountPath: /data

  volumes:
  - name: local-ssd
    hostPath:
      path: /mnt/disks/ssd0  # Path on node
      type: Directory
```

**⚠️ Caution:**
- **Not portable** (tied to specific node)
- **Not durable** (lost if node fails)
- **Security risk** (can access host filesystem)
- Use ONLY for development or when you control pod scheduling

## 3. PersistentVolumes and PersistentVolumeClaims

### 3.1 Static Provisioning (Manual)

**Step 1: Administrator creates PersistentVolume**

```yaml
# pv-manual.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ml-datasets-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
  - ReadWriteOnce  # Single pod can mount read-write
  persistentVolumeReclaimPolicy: Retain  # Keep data after PVC deletion
  storageClassName: manual
  hostPath:  # Using hostPath for example (use cloud storage in production)
    path: /mnt/data/ml-datasets
```

**Step 2: User creates PersistentVolumeClaim**

```yaml
# pvc-manual.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-datasets-pvc
  namespace: ml-training
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: manual
```

**Step 3: Pod uses PVC**

```yaml
# pod-with-pvc.yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
  namespace: ml-training
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["python", "train.py"]
    args: ["--data-dir=/data/imagenet"]
    volumeMounts:
    - name: datasets
      mountPath: /data

  volumes:
  - name: datasets
    persistentVolumeClaim:
      claimName: ml-datasets-pvc
```

### 3.2 Dynamic Provisioning (Automatic)

**Recommended approach:** Use StorageClass for automatic provisioning.

**Step 1: Create StorageClass (or use pre-existing)**

```yaml
# storageclass-aws-ebs.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com  # AWS EBS CSI driver
parameters:
  type: gp3  # General Purpose SSD v3
  iops: "3000"
  throughput: "125"  # MB/s
  encrypted: "true"
allowVolumeExpansion: true  # Allow resizing
reclaimPolicy: Delete  # Delete EBS volume when PVC is deleted
```

**Step 2: Create PVC (PV auto-created)**

```yaml
# pvc-dynamic.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints-pvc
  namespace: ml-training
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast-ssd  # Use StorageClass
  resources:
    requests:
      storage: 100Gi
```

**Result:**
- Kubernetes automatically provisions a 100Gi gp3 EBS volume
- PV is created and bound to PVC
- When PVC is deleted, EBS volume is deleted (reclaimPolicy: Delete)

### 3.3 Access Modes

| Access Mode      | Abbreviation | Description                               | Use Case                  |
|------------------|--------------|-------------------------------------------|---------------------------|
| ReadWriteOnce    | RWO          | Single pod can mount read-write           | Training checkpoints      |
| ReadOnlyMany     | ROX          | Multiple pods can mount read-only         | Shared datasets (read)    |
| ReadWriteMany    | RWX          | Multiple pods can mount read-write        | Distributed training logs |
| ReadWriteOncePod | RWOP         | Single pod exclusively (K8s 1.22+)        | High-security workloads   |

**Example: Shared dataset (read-only for all training pods)**

```yaml
# pvc-shared-dataset.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: imagenet-dataset-pvc
spec:
  accessModes:
  - ReadOnlyMany  # Multiple pods can read
  resources:
    requests:
      storage: 1Ti
  storageClassName: aws-efs  # EFS supports ReadOnlyMany
```

## 4. StorageClasses for ML Workloads

### 4.1 AWS EBS StorageClasses

**General Purpose (balanced):**

```yaml
# sc-aws-gp3.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-balanced
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"      # 3000 IOPS baseline
  throughput: "125" # 125 MB/s baseline
  encrypted: "true"
allowVolumeExpansion: true
```

**High-performance (for GPU training):**

```yaml
# sc-aws-io2.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: io2-high-performance
provisioner: ebs.csi.aws.com
parameters:
  type: io2
  iops: "64000"     # Up to 64,000 IOPS
  encrypted: "true"
allowVolumeExpansion: true
```

**Cost-optimized (for archival):**

```yaml
# sc-aws-st1.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: st1-throughput-optimized
provisioner: ebs.csi.aws.com
parameters:
  type: st1  # Throughput Optimized HDD (cheap, sequential I/O)
  encrypted: "true"
```

### 4.2 Google Cloud PD StorageClasses

```yaml
# sc-gcp-ssd.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd-gcp
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd  # SSD Persistent Disk
  replication-type: regional-pd  # Replicated across zones
allowVolumeExpansion: true
```

### 4.3 Shared Storage (EFS/GCS/Azure Files)

**AWS EFS (ReadWriteMany):**

```yaml
# sc-aws-efs.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-efs
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap  # Access Point mode
  fileSystemId: fs-0123456789abcdef  # Your EFS filesystem ID
  directoryPerms: "700"
```

**Use case:** Shared datasets accessed by multiple training pods simultaneously.

```yaml
# pvc-efs.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-datasets-pvc
spec:
  accessModes:
  - ReadWriteMany  # Multiple pods can read/write
  storageClassName: aws-efs
  resources:
    requests:
      storage: 2Ti
```

## 5. StatefulSets for ML Applications

### 5.1 When to Use StatefulSets

**Use StatefulSets when:**
- Each replica needs unique persistent storage (distributed training)
- Pods need stable network identities (parameter servers)
- Ordered startup/shutdown is required

**Examples:**
- Distributed training with Horovod/DeepSpeed
- ML feature stores (Feast)
- Model serving with caching layers
- Experiment tracking servers (MLflow)

### 5.2 StatefulSet Example: Distributed Training

```yaml
# statefulset-distributed-training.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: distributed-training
  namespace: ml-training
spec:
  serviceName: training-headless  # Headless service for DNS
  replicas: 4  # 4 training workers

  selector:
    matchLabels:
      app: distributed-training

  template:
    metadata:
      labels:
        app: distributed-training
    spec:
      containers:
      - name: training
        image: horovod/horovod:0.28.1-tf2.12.0-torch2.0.0-mxnet1.9.1-py3.10-gpu
        command: ["horovodrun"]
        args:
          - "-np"
          - "4"
          - "python"
          - "train_distributed.py"
          - "--checkpoint-dir=/checkpoints"

        resources:
          limits:
            nvidia.com/gpu: 1

        volumeMounts:
        - name: checkpoints
          mountPath: /checkpoints  # Each pod gets its own storage

        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

  volumeClaimTemplates:  # Auto-create PVC for each replica
  - metadata:
      name: checkpoints
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi
```

**Result:**
- Creates 4 pods: `distributed-training-0`, `distributed-training-1`, `distributed-training-2`, `distributed-training-3`
- Creates 4 PVCs: `checkpoints-distributed-training-0`, `checkpoints-distributed-training-1`, etc.
- Each pod has stable hostname: `distributed-training-0.training-headless.ml-training.svc.cluster.local`

**Headless Service (for DNS):**

```yaml
# headless-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: training-headless
  namespace: ml-training
spec:
  clusterIP: None  # Headless service (no cluster IP)
  selector:
    app: distributed-training
  ports:
  - port: 12345
    name: communication
```

## 6. Storage Strategies for ML Workflows

### 6.1 Training Data Storage

**Strategy 1: Pre-load datasets to fast PVC**

```yaml
# dataset-loader-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: load-imagenet-dataset
spec:
  template:
    spec:
      containers:
      - name: downloader
        image: amazon/aws-cli
        command: ["sh", "-c"]
        args:
          - |
            aws s3 sync s3://my-bucket/imagenet/ /data/imagenet/
        volumeMounts:
        - name: dataset
          mountPath: /data
      volumes:
      - name: dataset
        persistentVolumeClaim:
          claimName: imagenet-pvc
      restartPolicy: OnFailure
```

**Strategy 2: Stream from object storage (S3/GCS)**

```yaml
# training-with-s3-streaming.yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-s3-streaming
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["python", "train_with_s3.py"]
    env:
    - name: S3_BUCKET
      value: "s3://my-datasets/imagenet"
    - name: AWS_REGION
      value: "us-west-2"
    # No volume needed - stream directly from S3 in code
```

**Trade-offs:**

| Approach       | Pros                          | Cons                        | Best For               |
|----------------|-------------------------------|-----------------------------|------------------------|
| Pre-load to PVC| Fast training, consistent I/O | Expensive storage, slow setup| High IOPS needed       |
| Stream from S3 | Cheap, no pre-loading needed  | Slower, network dependent   | Large datasets, batch  |

### 6.2 Model Checkpoint Storage

**Best practice:** Use separate PVC for checkpoints.

```yaml
# training-with-checkpoints.yaml
apiVersion: v1
kind: Pod
metadata:
  name: resnet-training
spec:
  containers:
  - name: training
    image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
    command: ["python", "train.py"]
    args:
      - "--checkpoint-dir=/checkpoints"
      - "--checkpoint-interval=1000"  # Save every 1000 steps

    volumeMounts:
    - name: datasets
      mountPath: /data
      readOnly: true
    - name: checkpoints
      mountPath: /checkpoints

  volumes:
  - name: datasets
    persistentVolumeClaim:
      claimName: imagenet-pvc  # Read-only dataset
  - name: checkpoints
    persistentVolumeClaim:
      claimName: resnet-checkpoints-pvc  # Writable checkpoint storage
```

**Why separate PVCs?**
- Dataset can be shared (ReadOnlyMany)
- Checkpoints are per-job (ReadWriteOnce)
- Easier to manage lifecycle (delete checkpoints but keep datasets)

### 6.3 Model Artifact Storage

**Store final trained models in persistent registry:**

```yaml
# model-registry-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-registry-pvc
  namespace: ml-platform
spec:
  accessModes:
  - ReadWriteMany  # Multiple pods can read models
  storageClassName: aws-efs  # Shared storage
  resources:
    requests:
      storage: 500Gi
```

**Training job uploads to registry:**

```yaml
# training-with-upload.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-and-upload
spec:
  template:
    spec:
      containers:
      - name: training
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command: ["sh", "-c"]
        args:
          - |
            python train.py
            cp /checkpoints/best_model.pth /registry/models/resnet50-v1.2.pth
        volumeMounts:
        - name: checkpoints
          mountPath: /checkpoints
        - name: registry
          mountPath: /registry

      volumes:
      - name: checkpoints
        emptyDir: {}  # Temporary checkpoints
      - name: registry
        persistentVolumeClaim:
          claimName: model-registry-pvc  # Persistent model storage

      restartPolicy: OnFailure
```

## 7. Performance Optimization

### 7.1 I/O Bottleneck Detection

**Symptom:** GPU utilization < 50% during training.

**Diagnose:**

```bash
# Check pod I/O stats
kubectl exec <training-pod> -- iostat -x 1 5

# Output:
# Device  r/s   w/s  rkB/s  wkB/s  %util
# sda     450   120  15000  5000   99%   <-- Disk is saturated!

# Check if data loading is slow
kubectl logs <training-pod> | grep "data loading time"
```

**Solutions:**

1. **Use faster storage class:**
   ```yaml
   storageClassName: io2-high-performance  # Switch from gp3 to io2
   ```

2. **Increase DataLoader workers:**
   ```python
   DataLoader(dataset, batch_size=64, num_workers=8)  # More parallel loading
   ```

3. **Prefetch to RAM:**
   ```python
   DataLoader(dataset, batch_size=64, pin_memory=True)  # Pin to GPU memory
   ```

4. **Use local SSD for datasets:**
   ```yaml
   volumeMounts:
   - name: local-ssd
     mountPath: /data
   volumes:
   - name: local-ssd
     hostPath:
       path: /mnt/disks/ssd0  # Local NVMe SSD
   ```

### 7.2 Storage Performance Tiers

**Choose based on workload:**

| Tier            | IOPS      | Throughput  | Cost/GB/mo | Use Case                    |
|-----------------|-----------|-------------|------------|-----------------------------|
| AWS io2         | 64,000    | 1,000 MB/s  | $0.125     | GPU training (fast datasets)|
| AWS gp3         | 16,000    | 1,000 MB/s  | $0.08      | General ML workloads        |
| AWS EFS         | 7,000+    | 3+ GB/s     | $0.30      | Shared datasets (RWX)       |
| AWS st1 (HDD)   | 500       | 500 MB/s    | $0.045     | Archival, batch inference   |
| Local NVMe      | 500,000+  | 7,000 MB/s  | Varies     | Extreme performance         |

## 8. Backup and Disaster Recovery

### 8.1 VolumeSnapshots

**Create snapshot of PVC:**

```yaml
# volumesnapshot.yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: resnet-checkpoint-snapshot
  namespace: ml-training
spec:
  volumeSnapshotClassName: csi-aws-vsc  # Snapshot class
  source:
    persistentVolumeClaimName: resnet-checkpoints-pvc
```

**Restore from snapshot:**

```yaml
# pvc-from-snapshot.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: resnet-checkpoints-restored
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast-ssd
  dataSource:
    name: resnet-checkpoint-snapshot  # Restore from snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
  resources:
    requests:
      storage: 50Gi
```

### 8.2 Backup to Object Storage

**Scheduled backup job:**

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-model-registry
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: amazon/aws-cli
            command: ["sh", "-c"]
            args:
              - |
                aws s3 sync /registry s3://my-backups/model-registry-$(date +%Y-%m-%d)/
            volumeMounts:
            - name: registry
              mountPath: /registry
              readOnly: true
          volumes:
          - name: registry
            persistentVolumeClaim:
              claimName: model-registry-pvc
          restartPolicy: OnFailure
```

## 9. Hands-On Exercise: ML Training with Persistent Storage

**Objective:** Deploy a training job with:
- Dataset loaded from S3 to PVC
- Checkpoints saved to separate PVC
- Final model uploaded to model registry

**Step 1: Create PVCs**

```yaml
# ml-training-pvcs.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-dataset-pvc
  namespace: ml-training
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: gp3-balanced
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-checkpoints-pvc
  namespace: ml-training
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-registry-pvc
  namespace: ml-training
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: aws-efs
  resources:
    requests:
      storage: 500Gi
```

**Step 2: Load dataset (one-time job)**

```yaml
# dataset-loader.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: load-cifar10-dataset
  namespace: ml-training
spec:
  template:
    spec:
      containers:
      - name: downloader
        image: python:3.10
        command: ["sh", "-c"]
        args:
          - |
            pip install torchvision
            python -c "
            from torchvision import datasets
            datasets.CIFAR10(root='/data', train=True, download=True)
            datasets.CIFAR10(root='/data', train=False, download=True)
            "
        volumeMounts:
        - name: dataset
          mountPath: /data
      volumes:
      - name: dataset
        persistentVolumeClaim:
          claimName: training-dataset-pvc
      restartPolicy: Never
```

**Step 3: Training job**

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cifar10-training
  namespace: ml-training
spec:
  template:
    spec:
      containers:
      - name: training
        image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
        command: ["python", "train_cifar10.py"]
        args:
          - "--data-dir=/data"
          - "--checkpoint-dir=/checkpoints"
          - "--output-dir=/registry/models/cifar10-resnet18-v1.0"
          - "--epochs=100"

        resources:
          limits:
            nvidia.com/gpu: 1

        volumeMounts:
        - name: dataset
          mountPath: /data
          readOnly: true
        - name: checkpoints
          mountPath: /checkpoints
        - name: registry
          mountPath: /registry

      volumes:
      - name: dataset
        persistentVolumeClaim:
          claimName: training-dataset-pvc
      - name: checkpoints
        persistentVolumeClaim:
          claimName: training-checkpoints-pvc
      - name: registry
        persistentVolumeClaim:
          claimName: model-registry-pvc

      restartPolicy: OnFailure
```

**Step 4: Deploy and monitor**

```bash
kubectl create namespace ml-training

kubectl apply -f ml-training-pvcs.yaml
kubectl apply -f dataset-loader.yaml

# Wait for dataset to load
kubectl wait --for=condition=complete job/load-cifar10-dataset -n ml-training

# Start training
kubectl apply -f training-job.yaml

# Monitor training
kubectl logs -f job/cifar10-training -n ml-training

# Check checkpoints are being saved
kubectl exec deployment/<training-pod> -n ml-training -- ls -lh /checkpoints

# After training completes, verify model is in registry
kubectl exec deployment/<any-pod> -n ml-training -- ls -lh /registry/models/cifar10-resnet18-v1.0
```

## 10. Summary

### Key Takeaways

✅ **Volume types:**
- **emptyDir:** Temporary scratch space (deleted with Pod)
- **PVC:** Persistent, durable storage (survives Pod deletion)
- **hostPath:** Node-local storage (fast but not portable)

✅ **PersistentVolumes and PersistentVolumeClaims:**
- PV: cluster-wide storage resource
- PVC: user request for storage
- StorageClass: automates PV provisioning

✅ **Access modes:**
- **RWO:** Single pod read-write (training checkpoints)
- **ROX:** Multi-pod read-only (shared datasets)
- **RWX:** Multi-pod read-write (distributed training logs)

✅ **StatefulSets:**
- Use for distributed training, parameter servers
- Each replica gets unique PVC
- Stable network identities

✅ **Performance:**
- Choose fast storage (io2, local NVMe) for GPU workloads
- Monitor I/O utilization with `iostat`
- Pre-load datasets to fast PVCs vs streaming from S3

✅ **Best practices:**
- Separate PVCs for datasets, checkpoints, and models
- Use VolumeSnapshots for backups
- Implement scheduled backups to S3/GCS

## Self-Check Questions

1. What's the difference between a Volume and a PersistentVolume?
2. When would you use RWO vs RWX access mode?
3. How does dynamic provisioning work with StorageClasses?
4. Why use StatefulSets instead of Deployments for distributed training?
5. What storage type would you choose for a GPU training job? Why?
6. How would you backup ML model checkpoints?
7. What's the difference between emptyDir and hostPath?
8. How would you troubleshoot slow data loading during training?

## Additional Resources

- [Kubernetes Storage Documentation](https://kubernetes.io/docs/concepts/storage/)
- [AWS EBS CSI Driver](https://github.com/kubernetes-sigs/aws-ebs-csi-driver)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Volume Snapshots](https://kubernetes.io/docs/concepts/storage/volume-snapshots/)

---

**Next lesson:** Helm Package Manager
