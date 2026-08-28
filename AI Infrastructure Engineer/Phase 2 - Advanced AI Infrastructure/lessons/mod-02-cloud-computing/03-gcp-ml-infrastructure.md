# Lesson 03: Google Cloud Platform for ML Infrastructure

**Duration:** 7 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 01 (Cloud Architecture), Lesson 02 (AWS ML Infrastructure)

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Set up and secure** a Google Cloud Platform account with proper IAM
2. **Deploy and manage** Compute Engine instances for ML workloads
3. **Use Cloud Storage** for datasets and models with lifecycle management
4. **Work with TPUs** for accelerated deep learning training
5. **Deploy ML models** on Google Kubernetes Engine (GKE)
6. **Use Vertex AI** for managed ML training and deployment
7. **Configure VPC networking** for secure ML infrastructure
8. **Optimize costs** using GCP pricing models and best practices

---

## Table of Contents

1. [Introduction to GCP for ML](#introduction-to-gcp-for-ml)
2. [GCP Account Setup and IAM](#gcp-account-setup-and-iam)
3. [Compute Engine for ML](#compute-engine-for-ml)
4. [Cloud Storage for Data and Models](#cloud-storage-for-data-and-models)
5. [Tensor Processing Units (TPUs)](#tensor-processing-units-tpus)
6. [Google Kubernetes Engine (GKE)](#google-kubernetes-engine-gke)
7. [Vertex AI Platform](#vertex-ai-platform)
8. [GCP Networking for ML](#gcp-networking-for-ml)
9. [Cost Optimization Strategies](#cost-optimization-strategies)
10. [Hands-on Exercise](#hands-on-exercise)

---

## Introduction to GCP for ML

Google Cloud Platform (GCP) is particularly strong for ML workloads because:

- **Native ML heritage**: Built by the creators of TensorFlow
- **TPU access**: Exclusive access to Tensor Processing Units
- **BigQuery ML**: Run ML models directly on data warehouse
- **Vertex AI**: Comprehensive managed ML platform
- **Strong open-source integration**: TensorFlow, Kubeflow, Ray
- **Competitive pricing**: Sustained use discounts, preemptible VMs

### GCP ML Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                       GCP ML Ecosystem                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data Ingestion       Processing           Training/Serving    │
│  ──────────────       ──────────           ────────────────    │
│  Cloud Storage   →    Dataflow        →    Vertex AI          │
│  BigQuery        →    Dataproc        →    GKE                │
│  Pub/Sub         →    AI Platform     →    Cloud Run          │
│                                                                 │
│  Infrastructure       Networking           Monitoring          │
│  ──────────────       ──────────           ──────────          │
│  Compute Engine       VPC                  Cloud Monitoring   │
│  GKE                  Cloud Load          Cloud Logging       │
│  Cloud Run            Balancing           Cloud Trace         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### When to Choose GCP for ML

**Choose GCP when:**
- You're heavily invested in TensorFlow
- You need TPUs for large-scale training
- You want strong BigQuery integration
- You prefer Google's managed services
- You need strong open-source Kubernetes support

**Consider alternatives when:**
- Your team is already on AWS/Azure
- You need a wider range of GPU types
- You require more regional availability
- You need specific enterprise integrations

---

## GCP Account Setup and IAM

### Creating a GCP Account

1. **Sign up for GCP**:
   - Visit: https://cloud.google.com/free
   - Get **$300 free credit** valid for 90 days
   - No automatic charges after trial ends
   - Credit card required for verification

2. **Free tier (Always Free)**:
   ```
   Compute Engine:
   - 1 f1-micro instance (US regions only)
   - 30 GB standard persistent disk
   - 1 GB snapshot storage

   Cloud Storage:
   - 5 GB standard storage
   - 1 GB network egress (North America)

   BigQuery:
   - 1 TB queries/month
   - 10 GB storage

   Cloud Functions:
   - 2 million invocations/month
   ```

3. **Create a project**:
   ```bash
   # Install gcloud CLI
   # macOS
   brew install google-cloud-sdk

   # Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL

   # Initialize gcloud
   gcloud init

   # Create a new project
   gcloud projects create ml-infrastructure-project --name="ML Infrastructure"

   # Set as default project
   gcloud config set project ml-infrastructure-project

   # Enable required APIs
   gcloud services enable compute.googleapis.com
   gcloud services enable container.googleapis.com
   gcloud services enable storage.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```

### IAM Best Practices

GCP uses **Identity and Access Management (IAM)** with a fine-grained permission model.

#### Key Concepts

1. **Principal**: Who (user, service account, group)
2. **Role**: What permissions (predefined or custom)
3. **Resource**: Where (project, folder, organization)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Principal   │────▶│     Role     │────▶│  Resource    │
├──────────────┤     ├──────────────┤     ├──────────────┤
│ - User       │     │ - Primitive  │     │ - Project    │
│ - SA         │     │ - Predefined │     │ - Bucket     │
│ - Group      │     │ - Custom     │     │ - VM         │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### Creating Service Accounts

Service accounts are used for application-to-application authentication.

```bash
# Create a service account for ML training
gcloud iam service-accounts create ml-training-sa \
  --display-name="ML Training Service Account" \
  --description="Service account for ML training jobs"

# Grant necessary permissions
gcloud projects add-iam-policy-binding ml-infrastructure-project \
  --member="serviceAccount:ml-training-sa@ml-infrastructure-project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ml-infrastructure-project \
  --member="serviceAccount:ml-training-sa@ml-infrastructure-project.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Create and download key
gcloud iam service-accounts keys create ~/ml-training-key.json \
  --iam-account=ml-training-sa@ml-infrastructure-project.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/ml-training-key.json
```

#### Common ML Roles

```
┌─────────────────────────────────┬──────────────────────────────────────┐
│ Role                            │ Use Case                             │
├─────────────────────────────────┼──────────────────────────────────────┤
│ roles/aiplatform.user           │ Run Vertex AI training jobs          │
│ roles/storage.objectAdmin       │ Read/write Cloud Storage             │
│ roles/compute.instanceAdmin     │ Manage Compute Engine VMs            │
│ roles/container.admin           │ Manage GKE clusters                  │
│ roles/monitoring.metricWriter   │ Write custom metrics                 │
│ roles/logging.logWriter         │ Write application logs               │
└─────────────────────────────────┴──────────────────────────────────────┘
```

#### Security Best Practices

1. **Principle of least privilege**: Grant minimum required permissions
2. **Use service accounts**: Never use personal credentials in code
3. **Rotate keys regularly**: Set up key rotation policy (90 days)
4. **Enable audit logs**: Track all IAM changes
5. **Use organization policies**: Enforce security constraints

```bash
# Enable audit logs
gcloud logging sinks create ml-audit-sink \
  storage.googleapis.com/ml-audit-logs-bucket \
  --log-filter='protoPayload.methodName:"iam.googleapis.com"'

# Set key expiration reminder (via monitoring)
gcloud alpha iam service-accounts keys list \
  --iam-account=ml-training-sa@ml-infrastructure-project.iam.gserviceaccount.com \
  --format="table(name,validAfterTime,validBeforeTime)"
```

---

## Compute Engine for ML

Compute Engine provides virtual machines with flexible configurations for ML workloads.

### Instance Types for ML

#### General Purpose (N-series)

Best for: Development, small models, CPU inference

```
┌──────────────┬─────────┬────────────┬──────────┬───────────────┐
│ Machine Type │ vCPUs   │ Memory     │ Network  │ Cost/hour     │
├──────────────┼─────────┼────────────┼──────────┼───────────────┤
│ n1-standard-4│ 4       │ 15 GB      │ 10 Gbps  │ $0.190        │
│ n2-standard-8│ 8       │ 32 GB      │ 32 Gbps  │ $0.389        │
│ n2d-highmem-16│16      │ 128 GB     │ 32 Gbps  │ $0.777        │
└──────────────┴─────────┴────────────┴──────────┴───────────────┘
```

#### GPU Instances (Accelerator-optimized)

Best for: Training, GPU inference

```
┌──────────────┬──────────────┬─────────────┬─────────┬──────────────┐
│ Machine Type │ GPU          │ GPU Memory  │ vCPUs   │ Cost/hour    │
├──────────────┼──────────────┼─────────────┼─────────┼──────────────┤
│ n1 + T4      │ 1x NVIDIA T4 │ 16 GB       │ 4       │ $0.35        │
│ n1 + V100    │ 1x NVIDIA V100│16 GB       │ 8       │ $2.48        │
│ n1 + A100    │ 1x NVIDIA A100│40 GB       │ 12      │ $3.67        │
│ a2 + A100    │ 8x NVIDIA A100│320 GB      │ 96      │ $29.39       │
└──────────────┴──────────────┴─────────────┴─────────┴──────────────┘
```

#### TPU Instances

Best for: TensorFlow training at scale

```
┌──────────────┬──────────────┬─────────────┬──────────────┐
│ TPU Type     │ Cores        │ Memory      │ Cost/hour    │
├──────────────┼──────────────┼─────────────┼──────────────┤
│ v2-8         │ 8            │ 64 GB HBM   │ $4.50        │
│ v3-8         │ 8            │ 128 GB HBM  │ $8.00        │
│ v4-8         │ 8            │ 32 GB HBM2e │ $3.67        │
│ v4-32        │ 32           │ 128 GB HBM2e│ $14.69       │
└──────────────┴──────────────┴─────────────┴──────────────┘
```

### Creating a GPU Instance

```bash
# List available GPU types in region
gcloud compute accelerator-types list --filter="zone:us-central1-a"

# Create instance with T4 GPU
gcloud compute instances create ml-training-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# SSH into instance
gcloud compute ssh ml-training-gpu --zone=us-central1-a

# Verify GPU
nvidia-smi
```

### Deep Learning VM Images

GCP provides pre-configured images with ML frameworks:

```bash
# List available images
gcloud compute images list \
  --project deeplearning-platform-release \
  --no-standard-images

# Create instance with PyTorch
gcloud compute instances create ml-pytorch \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True,proxy-mode=project_editors"

# Create instance with TensorFlow
gcloud compute instances create ml-tensorflow \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=tf-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True"
```

### Preemptible VMs (Spot Instances)

Save up to **80%** on compute costs with preemptible VMs.

```bash
# Create preemptible instance
gcloud compute instances create ml-training-preemptible \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --preemptible \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True"

# Cost comparison
# Regular: $0.35/hour
# Preemptible: $0.07/hour (80% savings)
```

**Important considerations:**
- Maximum runtime: 24 hours
- Can be terminated at any time with 30-second warning
- Not always available (capacity-based)
- Perfect for fault-tolerant workloads

### Checkpointing for Preemptible VMs

```python
import os
import signal
import sys
import torch

def signal_handler(signum, frame):
    """Handle preemption signal"""
    print("Received preemption signal, saving checkpoint...")
    save_checkpoint({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
    }, 'checkpoint_preempted.pth')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, signal_handler)

# Training loop with checkpointing
def train_with_checkpointing(model, train_loader, epochs=10):
    checkpoint_dir = "/mnt/disks/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from latest checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Save checkpoint every 100 batches
            if batch_idx % 100 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)

        # Save checkpoint after each epoch
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

        print(f"Epoch {epoch} completed, checkpoint saved")

def save_checkpoint(state, filename):
    """Save checkpoint to file"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")
```

### Startup Scripts

Automate setup with startup scripts:

```bash
# Create startup script
cat > startup.sh << 'EOF'
#!/bin/bash

# Update system
apt-get update

# Install Python packages
pip install torch torchvision wandb

# Download dataset
mkdir -p /data
gsutil -m rsync -r gs://my-bucket/datasets/imagenet /data/imagenet

# Clone training code
cd /home
git clone https://github.com/myorg/ml-training.git

# Start training
cd ml-training
python train.py --data-path /data/imagenet --epochs 100 --checkpoint-dir /data/checkpoints
EOF

# Create instance with startup script
gcloud compute instances create ml-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --preemptible \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata-from-file=startup-script=startup.sh \
  --scopes=storage-rw,logging-write
```

---

## Cloud Storage for Data and Models

Cloud Storage is GCP's object storage service, similar to AWS S3.

### Storage Classes

```
┌────────────────────┬────────────────┬─────────────────┬────────────────┐
│ Storage Class      │ Cost/GB/month  │ Retrieval Cost  │ Use Case       │
├────────────────────┼────────────────┼─────────────────┼────────────────┤
│ Standard           │ $0.020         │ Free            │ Active data    │
│ Nearline           │ $0.010         │ $0.01/GB        │ <1/month access│
│ Coldline           │ $0.004         │ $0.02/GB        │ <1/quarter     │
│ Archive            │ $0.0012        │ $0.05/GB        │ <1/year        │
└────────────────────┴────────────────┴─────────────────┴────────────────┘
```

### Creating and Managing Buckets

```bash
# Create a bucket
gsutil mb -l us-central1 -c STANDARD gs://my-ml-data-bucket

# Upload files
gsutil cp model.pth gs://my-ml-data-bucket/models/

# Upload directory (parallel)
gsutil -m cp -r ./datasets gs://my-ml-data-bucket/

# Download files
gsutil cp gs://my-ml-data-bucket/models/model.pth ./

# Sync directories (like rsync)
gsutil -m rsync -r ./local-dir gs://my-ml-data-bucket/remote-dir

# List files
gsutil ls gs://my-ml-data-bucket/models/

# Delete files
gsutil rm gs://my-ml-data-bucket/models/old-model.pth

# Get bucket info
gsutil du -sh gs://my-ml-data-bucket
```

### Versioning and Lifecycle Management

```bash
# Enable versioning
gsutil versioning set on gs://my-ml-data-bucket

# Create lifecycle policy
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
          "age": 30,
          "matchesStorageClass": ["STANDARD"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 90,
          "matchesStorageClass": ["NEARLINE"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesStorageClass": ["COLDLINE"]
        }
      }
    ]
  }
}
EOF

# Apply lifecycle policy
gsutil lifecycle set lifecycle.json gs://my-ml-data-bucket

# View lifecycle policy
gsutil lifecycle get gs://my-ml-data-bucket
```

### Python SDK (google-cloud-storage)

```python
from google.cloud import storage
import os

# Initialize client
client = storage.Client()

# Create bucket
bucket = client.create_bucket("my-ml-data-bucket", location="us-central1")

# Upload file
def upload_file(bucket_name, source_file, destination_blob):
    """Upload file to Cloud Storage"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)
    print(f"File {source_file} uploaded to {destination_blob}")

# Upload model
upload_file("my-ml-data-bucket", "model.pth", "models/resnet50-v1.pth")

# Download file
def download_file(bucket_name, source_blob, destination_file):
    """Download file from Cloud Storage"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)

    blob.download_to_filename(destination_file)
    print(f"File {source_blob} downloaded to {destination_file}")

# Download model
download_file("my-ml-data-bucket", "models/resnet50-v1.pth", "./model.pth")

# List files with prefix
def list_files(bucket_name, prefix):
    """List files in bucket with prefix"""
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        print(blob.name)

# List all models
list_files("my-ml-data-bucket", "models/")

# Generate signed URL (temporary access)
def generate_signed_url(bucket_name, blob_name, expiration_minutes=60):
    """Generate signed URL for temporary access"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        expiration=timedelta(minutes=expiration_minutes),
        method='GET'
    )

    return url

# Get temporary URL for model
url = generate_signed_url("my-ml-data-bucket", "models/resnet50-v1.pth", 120)
print(f"Temporary URL (valid for 2 hours): {url}")

# Stream large files
def download_large_file_in_chunks(bucket_name, source_blob, destination_file):
    """Download large file in chunks"""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)

    with open(destination_file, 'wb') as f:
        blob.download_to_file(f)

    print(f"Large file downloaded: {destination_file}")

# Copy file within Cloud Storage
def copy_blob(bucket_name, source_blob, destination_bucket, destination_blob):
    """Copy file within Cloud Storage"""
    source_bucket = client.bucket(bucket_name)
    source = source_bucket.blob(source_blob)
    destination_bucket = client.bucket(destination_bucket)

    source_bucket.copy_blob(source, destination_bucket, destination_blob)
    print(f"Copied {source_blob} to {destination_blob}")

# Promote model from staging to production
copy_blob(
    "ml-staging-bucket", "models/resnet50-v2.pth",
    "ml-production-bucket", "models/resnet50-latest.pth"
)
```

### Best Practices for ML Data Storage

1. **Organize by lifecycle**:
   ```
   gs://my-ml-bucket/
   ├── raw-data/           # Standard storage
   ├── processed-data/     # Standard → Nearline after 30 days
   ├── models/
   │   ├── staging/        # Standard
   │   ├── production/     # Standard
   │   └── archived/       # Coldline
   └── experiments/        # Delete after 90 days
   ```

2. **Use regional buckets**: Keep data close to compute for lower latency

3. **Enable versioning**: Protect against accidental deletion

4. **Set up notifications**: Trigger Cloud Functions on new data

5. **Use requester pays**: For public datasets, make users pay egress costs

---

## Tensor Processing Units (TPUs)

TPUs are Google's custom-designed ASICs for accelerating machine learning workloads.

### TPU vs GPU

```
┌─────────────────┬───────────────────┬────────────────────┐
│ Aspect          │ GPU               │ TPU                │
├─────────────────┼───────────────────┼────────────────────┤
│ Architecture    │ General purpose   │ ML-specific        │
│ Precision       │ FP32, FP16, INT8  │ BFloat16 optimized │
│ Framework       │ Any framework     │ TensorFlow, JAX    │
│ Memory          │ 16-80 GB HBM      │ 8-32 GB HBM        │
│ Performance     │ High              │ Very High (TF)     │
│ Cost            │ Medium            │ Medium-Low         │
│ Flexibility     │ Very flexible     │ Less flexible      │
│ Best for        │ Any ML/DL task    │ Large TF models    │
└─────────────────┴───────────────────┴────────────────────┘
```

### When to Use TPUs

**Use TPUs when:**
- Training large TensorFlow models (transformers, large CNNs)
- Using mixed-precision training (bfloat16)
- Training for extended periods (cost-effective)
- Need high throughput for inference

**Use GPUs when:**
- Using PyTorch as primary framework
- Prototyping and experimentation
- Need maximum flexibility
- Training models with complex custom ops

### Creating a TPU VM

```bash
# Create TPU v2-8 (8 cores)
gcloud compute tpus tpu-vm create ml-tpu-v2 \
  --zone=us-central1-a \
  --accelerator-type=v2-8 \
  --version=tpu-vm-tf-2.13.0

# Create TPU v3-8 (more memory)
gcloud compute tpus tpu-vm create ml-tpu-v3 \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.13.0

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh ml-tpu-v2 --zone=us-central1-a

# List TPUs
gcloud compute tpus tpu-vm list

# Delete TPU
gcloud compute tpus tpu-vm delete ml-tpu-v2 --zone=us-central1-a
```

### Training on TPUs with TensorFlow

```python
import tensorflow as tf
import os

# Initialize TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

# Create TPU strategy
strategy = tf.distribute.TPUStrategy(resolver)

print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")

# Define model inside strategy scope
with strategy.scope():
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Load dataset
def load_dataset():
    # Load from Cloud Storage
    dataset = tf.data.TFRecordDataset(
        'gs://my-ml-bucket/datasets/imagenet/train-*.tfrecord'
    )

    # Parse and preprocess
    def parse_example(example):
        features = tf.io.parse_single_example(example, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
        image = tf.io.decode_jpeg(features['image'], channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image, features['label']

    dataset = dataset.map(parse_example)
    dataset = dataset.batch(128)  # Global batch size
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_dataset = load_dataset()

# Train on TPU
model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=1000
)

# Save model
model.save('gs://my-ml-bucket/models/resnet-tpu-trained')
```

### TPU Pods (Multi-TPU Training)

For very large models, use TPU Pods (multiple TPUs connected):

```bash
# Create TPU Pod (v3-32 = 4 TPU v3-8 chips)
gcloud compute tpus tpu-vm create ml-tpu-pod \
  --zone=us-central1-a \
  --accelerator-type=v3-32 \
  --version=tpu-vm-tf-2.13.0

# Cost: v3-32 = $32/hour (4x the cost of v3-8)
```

### TPU Best Practices

1. **Use bfloat16 for training**: TPUs are optimized for bfloat16
   ```python
   policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

2. **Batch size**: Use large batch sizes (128, 256, 512)

3. **Data pipeline**: Preprocess data on CPU, use `tf.data.Dataset`

4. **Checkpointing**: Save to Cloud Storage, not local disk

5. **Preemptible TPUs**: Save 70% with preemptible TPUs (similar to preemptible VMs)

---

## Google Kubernetes Engine (GKE)

GKE is Google's managed Kubernetes service for container orchestration.

### Creating a GKE Cluster

```bash
# Create standard cluster
gcloud container clusters create ml-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --disk-size=100GB \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10

# Create cluster with GPU nodes
gcloud container clusters create ml-gpu-cluster \
  --zone=us-central1-a \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --addons=GcePersistentDiskCsiDriver \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5

# Install NVIDIA device plugin (for GPU support)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Get cluster credentials
gcloud container clusters get-credentials ml-gpu-cluster --zone=us-central1-a

# Verify nodes
kubectl get nodes
```

### Deploying ML Model on GKE

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: gcr.io/my-project/ml-model:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "gs://my-ml-bucket/models/resnet50-latest.pth"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  type: LoadBalancer
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

Deploy:
```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get services

# Get external IP
kubectl get service ml-model-service
```

### GPU Deployment on GKE

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-gpu-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-gpu-model
  template:
    metadata:
      labels:
        app: ml-gpu-model
    spec:
      containers:
      - name: gpu-model-server
        image: gcr.io/my-project/ml-model-gpu:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

### Autoscaling on GKE

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### GKE Autopilot

GKE Autopilot is a fully managed mode that handles all cluster management:

```bash
# Create Autopilot cluster
gcloud container clusters create-auto ml-autopilot-cluster \
  --region=us-central1

# Benefits:
# - No node management
# - Pay per pod (not per node)
# - Automatic scaling
# - Security hardening
# - Lower operational overhead
```

---

## Vertex AI Platform

Vertex AI is GCP's unified ML platform for building, training, and deploying models.

### Vertex AI Components

```
┌──────────────────────────────────────────────────────────────┐
│                      Vertex AI Platform                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Workbench         Training          Prediction             │
│  ──────────        ────────          ──────────             │
│  - Notebooks       - Custom          - Online               │
│  - Managed         - AutoML          - Batch                │
│  - Git integration - Hyperparameter  - Edge                 │
│                      tuning                                  │
│                                                              │
│  Feature Store     Model Registry    Pipelines              │
│  ─────────────     ──────────────    ─────────              │
│  - Feature mgmt    - Versioning      - Kubeflow             │
│  - Serving         - Lineage         - TFX                  │
│  - Monitoring      - Evaluation      - Orchestration        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Training a Model on Vertex AI

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='my-project-id',
    location='us-central1',
    staging_bucket='gs://my-ml-bucket/staging'
)

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='resnet-training',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['torchvision', 'wandb'],
)

# Run training job
model = job.run(
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    args=[
        '--epochs', '100',
        '--batch-size', '64',
        '--learning-rate', '0.001',
    ],
    environment_variables={
        'MODEL_NAME': 'resnet50',
        'DATA_PATH': 'gs://my-ml-bucket/datasets/imagenet',
    },
)

print(f"Training job completed: {model.resource_name}")
```

### Deploying Model to Vertex AI Endpoint

```python
# Upload model
model = aiplatform.Model.upload(
    display_name='resnet50-v1',
    artifact_uri='gs://my-ml-bucket/models/resnet50/',
    serving_container_image_uri='gcr.io/my-project/model-server:latest',
)

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name='resnet50-endpoint',
)

# Deploy model to endpoint
endpoint.deploy(
    model=model,
    deployed_model_display_name='resnet50-v1',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
)

print(f"Model deployed to endpoint: {endpoint.resource_name}")

# Make prediction
instances = [
    {'image_bytes': base64.b64encode(open('cat.jpg', 'rb').read()).decode()}
]

prediction = endpoint.predict(instances=instances)
print(f"Prediction: {prediction.predictions}")
```

### AutoML on Vertex AI

For quick prototyping without custom training code:

```python
# Create AutoML image classification model
dataset = aiplatform.ImageDataset.create(
    display_name='my-image-dataset',
    gcs_source='gs://my-ml-bucket/datasets/images.csv',
)

job = aiplatform.AutoMLImageTrainingJob(
    display_name='automl-image-classification',
    prediction_type='classification',
    multi_label=False,
)

model = job.run(
    dataset=dataset,
    model_display_name='automl-resnet-v1',
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=8000,  # 8 hours
)
```

### Vertex AI Pricing

```
Training:
- n1-standard-4: $0.190/hour
- n1-standard-4 + T4: $0.526/hour
- n1-standard-8 + V100: $2.67/hour

Prediction (Hosting):
- n1-standard-2: $0.095/hour
- n1-standard-4 + T4: $0.526/hour

AutoML:
- Training: $3.15/hour
- Prediction: $1.25/hour + $0.10/1000 predictions
```

---

## GCP Networking for ML

### VPC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VPC Network (Global)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  us-central1                      us-east1                 │
│  ────────────                     ────────                 │
│  ┌─────────────────┐             ┌──────────────────┐     │
│  │ Subnet: 10.0.1.0/24│         │ Subnet: 10.0.2.0/24│   │
│  │                 │             │                  │     │
│  │ ML Training     │             │ ML Inference     │     │
│  │ Instances       │             │ Instances        │     │
│  │ (GPU VMs)       │             │ (GKE Cluster)    │     │
│  └─────────────────┘             └──────────────────┘     │
│                                                             │
│  Cloud Storage                   Cloud SQL                 │
│  (Private endpoint)              (Private IP)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ▲                               ▲
         │                               │
    Cloud NAT                    Cloud Load Balancer
    (Outbound)                   (Inbound - Public)
```

### Creating VPC Network

```bash
# Create VPC network
gcloud compute networks create ml-vpc \
  --subnet-mode=custom

# Create subnet for training (us-central1)
gcloud compute networks subnets create ml-training-subnet \
  --network=ml-vpc \
  --region=us-central1 \
  --range=10.0.1.0/24

# Create subnet for inference (us-east1)
gcloud compute networks subnets create ml-inference-subnet \
  --network=ml-vpc \
  --region=us-east1 \
  --range=10.0.2.0/24

# Create firewall rule (allow SSH)
gcloud compute firewall-rules create ml-allow-ssh \
  --network=ml-vpc \
  --allow=tcp:22 \
  --source-ranges=0.0.0.0/0

# Create firewall rule (allow internal)
gcloud compute firewall-rules create ml-allow-internal \
  --network=ml-vpc \
  --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/16
```

### Cloud Load Balancing

```bash
# Create instance group
gcloud compute instance-groups managed create ml-inference-group \
  --base-instance-name=ml-inference \
  --template=ml-inference-template \
  --size=3 \
  --zone=us-central1-a

# Create health check
gcloud compute health-checks create http ml-health-check \
  --port=8000 \
  --request-path=/health

# Create backend service
gcloud compute backend-services create ml-backend-service \
  --protocol=HTTP \
  --health-checks=ml-health-check \
  --global

# Add instance group to backend
gcloud compute backend-services add-backend ml-backend-service \
  --instance-group=ml-inference-group \
  --instance-group-zone=us-central1-a \
  --global

# Create URL map
gcloud compute url-maps create ml-load-balancer \
  --default-service=ml-backend-service

# Create target HTTP proxy
gcloud compute target-http-proxies create ml-http-proxy \
  --url-map=ml-load-balancer

# Create forwarding rule (public IP)
gcloud compute forwarding-rules create ml-forwarding-rule \
  --global \
  --target-http-proxy=ml-http-proxy \
  --ports=80

# Get public IP
gcloud compute forwarding-rules describe ml-forwarding-rule --global
```

### Cloud CDN

Enable Cloud CDN for model serving with caching:

```bash
# Enable Cloud CDN on backend service
gcloud compute backend-services update ml-backend-service \
  --enable-cdn \
  --cache-mode=CACHE_ALL_STATIC \
  --default-ttl=3600 \
  --global
```

---

## Cost Optimization Strategies

### 1. Committed Use Discounts (CUDs)

Save **57%** with 3-year commitment:

```
┌────────────────────┬──────────────┬──────────────┬──────────────┐
│ Resource           │ On-Demand    │ 1-Year CUD   │ 3-Year CUD   │
├────────────────────┼──────────────┼──────────────┼──────────────┤
│ n1-standard-4      │ $0.190/hr    │ $0.128/hr    │ $0.082/hr    │
│ n1-standard-8      │ $0.380/hr    │ $0.256/hr    │ $0.164/hr    │
│ T4 GPU             │ $0.35/hr     │ $0.244/hr    │ $0.158/hr    │
└────────────────────┴──────────────┴──────────────┴──────────────┘

Savings:           -              33%           57%
```

Purchase:
```bash
# Purchase CUD (via console or API)
# Console: Billing → Commitments → Purchase commitment
```

### 2. Sustained Use Discounts

Automatic discounts for sustained use (no commitment required):

```
Running time per month: Discount
- 25% of month: 0%
- 50% of month: 10%
- 75% of month: 20%
- 100% of month: 30%
```

### 3. Preemptible VMs and Spot VMs

Save **60-91%** on compute:

```bash
# Preemptible VM (can be terminated anytime)
gcloud compute instances create ml-training \
  --preemptible \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1

# Spot VM (successor to preemptible, with more features)
gcloud compute instances create ml-training-spot \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --machine-type=n1-standard-8
```

### 4. Autoscaling

Scale down when not in use:

```bash
# GKE cluster autoscaling
gcloud container clusters update ml-cluster \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=10 \
  --zone=us-central1-a

# Instance group autoscaling
gcloud compute instance-groups managed set-autoscaling ml-inference-group \
  --max-num-replicas=10 \
  --min-num-replicas=1 \
  --target-cpu-utilization=0.7 \
  --zone=us-central1-a
```

### 5. Storage Lifecycle Management

Automatically move data to cheaper storage:

```bash
# Lifecycle policy (from earlier)
# Standard → Nearline (30 days) → Coldline (90 days) → Delete (365 days)

# Savings example:
# 1 TB for 1 year:
# - All Standard: $240
# - With lifecycle: $80 (67% savings)
```

### 6. Budget Alerts

Set up billing alerts:

```bash
# Create budget (via console)
# Billing → Budgets & alerts → Create budget

# Set thresholds:
# - 50% of budget: Email alert
# - 90% of budget: Email alert + Pub/Sub notification
# - 100% of budget: Emergency notification
```

### 7. Cost Monitoring

```bash
# Install cost management tool
pip install google-cloud-billing

# View current costs
gcloud billing accounts list
gcloud billing projects describe my-project-id

# Export billing data to BigQuery for analysis
gcloud alpha billing accounts describe ACCOUNT_ID \
  --format="value(billingAccountName)"
```

### Cost Optimization Checklist

- [ ] Use preemptible/spot VMs for training (60-91% savings)
- [ ] Commit to 1-year or 3-year CUDs (33-57% savings)
- [ ] Enable autoscaling (scale to zero when idle)
- [ ] Use storage lifecycle policies (67% savings on old data)
- [ ] Right-size instances (don't over-provision)
- [ ] Delete unused resources (snapshots, disks, IPs)
- [ ] Use regional resources (cheaper than multi-regional)
- [ ] Leverage free tier (always free resources)
- [ ] Set up budget alerts (prevent surprise bills)
- [ ] Review costs weekly (identify waste early)

---

## Hands-on Exercise

### Exercise: Deploy Complete ML System on GCP

**Objective**: Deploy an image classification model with the following architecture:

```
User Request
    ↓
Cloud Load Balancer (with Cloud CDN)
    ↓
GKE Cluster (autoscaling 1-10 pods)
    ↓
Model Server (T4 GPU, loads model from Cloud Storage)
    ↓
Cloud Storage (model weights)
```

**Requirements**:
1. Use preemptible GKE nodes
2. Enable autoscaling (CPU > 70%)
3. Implement health checks
4. Enable Cloud CDN for caching
5. Set up monitoring with Cloud Monitoring
6. Estimate monthly cost (assume 1M requests/month)

**Steps**:

1. **Create Cloud Storage bucket and upload model**:
   ```bash
   gsutil mb -l us-central1 gs://my-ml-exercise-bucket
   gsutil cp model.pth gs://my-ml-exercise-bucket/models/
   ```

2. **Build and push Docker image**:
   ```bash
   # Build image
   docker build -t gcr.io/my-project/ml-model:v1 .

   # Push to Google Container Registry
   docker push gcr.io/my-project/ml-model:v1
   ```

3. **Create GKE cluster with GPU nodes**:
   ```bash
   gcloud container clusters create ml-exercise-cluster \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --num-nodes=2 \
     --enable-autoscaling \
     --min-nodes=1 \
     --max-nodes=10 \
     --preemptible
   ```

4. **Deploy model to GKE**:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yaml
   ```

5. **Set up load balancer and Cloud CDN**:
   ```bash
   # Follow load balancer setup from earlier
   # Enable Cloud CDN
   ```

6. **Configure monitoring**:
   ```bash
   # Set up Cloud Monitoring dashboard
   # Create alerts for high CPU, errors, latency
   ```

7. **Test and calculate costs**:
   ```bash
   # Load test
   ab -n 10000 -c 50 http://LOAD_BALANCER_IP/predict

   # Calculate costs:
   # - GKE: 2 preemptible n1-standard-4 + T4
   # - Cloud Storage: 1 GB
   # - Load Balancer: 1M requests
   # - Cloud CDN: 500 GB egress
   # - Cloud Monitoring: Basic tier
   ```

**Expected Cost** (1M requests/month):
- GKE nodes (preemptible): 2 × $0.07/hr × 730hr = $102
- Cloud Storage: 1 GB × $0.020 = $0.02
- Load Balancer: 1M requests × $0.008/1000 = $8
- Cloud CDN: 500 GB × $0.08/GB = $40
- **Total: ~$150/month**

**Bonus challenges**:
- Add Cloud Armor (DDoS protection)
- Implement A/B testing (split traffic between model versions)
- Set up CI/CD with Cloud Build
- Add distributed tracing with Cloud Trace

---

## Self-Check Questions

1. **What is the difference between Compute Engine, GKE, and Vertex AI for training models?**
   <details>
   <summary>Answer</summary>

   - **Compute Engine**: Full control, manual management, good for custom setups
   - **GKE**: Container orchestration, auto-scaling, good for production serving
   - **Vertex AI**: Fully managed, automated scaling, good for quick deployment
   </details>

2. **When would you choose a TPU over a GPU?**
   <details>
   <summary>Answer</summary>

   Choose TPU when:
   - Training large TensorFlow models
   - Using bfloat16 precision
   - Training for extended periods (cost-effective)
   - Need high matrix multiplication throughput

   Choose GPU when:
   - Using PyTorch
   - Need flexibility for custom operations
   - Prototyping and experimentation
   </details>

3. **How can you save 80% on compute costs?**
   <details>
   <summary>Answer</summary>

   - Use preemptible/spot VMs (60-91% savings)
   - Implement checkpointing for fault tolerance
   - Use autoscaling (scale to zero when idle)
   - Purchase committed use discounts (33-57% savings)
   </details>

4. **What's the best storage class for archived model checkpoints?**
   <details>
   <summary>Answer</summary>

   **Coldline** or **Archive** storage:
   - Coldline: $0.004/GB/month (access <1/quarter)
   - Archive: $0.0012/GB/month (access <1/year)
   - Use lifecycle policies to automatically transition
   </details>

5. **How do you enable autoscaling on GKE?**
   <details>
   <summary>Answer</summary>

   ```bash
   # Cluster autoscaler (nodes)
   gcloud container clusters update ml-cluster \
     --enable-autoscaling --min-nodes=1 --max-nodes=10

   # Horizontal Pod Autoscaler (pods)
   kubectl autoscale deployment ml-model \
     --cpu-percent=70 --min=2 --max=20
   ```
   </details>

---

## Additional Resources

### Official Documentation
- [GCP ML Documentation](https://cloud.google.com/products/ai)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [TPU Documentation](https://cloud.google.com/tpu/docs)

### Tutorials
- [GCP ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Vertex AI Tutorials](https://cloud.google.com/vertex-ai/docs/tutorials)
- [GKE ML Serving Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Tools
- [gcloud CLI](https://cloud.google.com/sdk/gcloud)
- [gsutil](https://cloud.google.com/storage/docs/gsutil)
- [kubectl](https://kubernetes.io/docs/reference/kubectl/)

### Cost Management
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
- [Cost Management Tools](https://cloud.google.com/cost-management)
- [Free Tier Details](https://cloud.google.com/free)

### Community
- [GCP Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform)
- [Reddit: r/googlecloud](https://www.reddit.com/r/googlecloud/)

---

## Summary

In this lesson, you learned:

✅ Set up GCP account with proper IAM and security
✅ Deploy Compute Engine instances with GPUs for ML training
✅ Use Cloud Storage for datasets and models with lifecycle management
✅ Work with TPUs for accelerated TensorFlow training
✅ Deploy models on GKE with autoscaling
✅ Use Vertex AI for managed ML training and serving
✅ Configure VPC networking and load balancing
✅ Optimize costs with preemptible VMs, CUDs, and lifecycle policies

**Key Takeaways**:
- GCP is strong for TensorFlow workloads and TPU access
- Preemptible VMs save 60-91% on compute costs
- Use GKE for production model serving with autoscaling
- Vertex AI simplifies ML workflow management
- Cloud Storage lifecycle policies save 67% on old data

**Next Steps**:
- Complete hands-on exercise
- Explore Vertex AI Pipelines for MLOps
- Learn about Kubeflow on GKE
- Proceed to Lesson 04: Azure for ML Infrastructure

---

**Estimated Time to Complete**: 7 hours (including hands-on exercise)
**Difficulty**: Intermediate
**Next Lesson**: [04-azure-ml-infrastructure.md](./04-azure-ml-infrastructure.md)
