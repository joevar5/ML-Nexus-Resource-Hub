# Lesson 05: Cloud Storage for ML

**Duration:** 6 hours
**Difficulty:** Intermediate
**Prerequisites:** Lessons 01-04 (Cloud providers overview)

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Compare and choose** appropriate storage types for ML workloads
2. **Implement data lakes** for large-scale ML datasets
3. **Optimize storage costs** with lifecycle policies and tiering
4. **Design caching strategies** for training and inference
5. **Implement data versioning** for reproducibility
6. **Configure data pipelines** for efficient data movement
7. **Apply security best practices** for sensitive data
8. **Benchmark storage performance** for ML workloads

---

## Table of Contents

1. [Storage Types Overview](#storage-types-overview)
2. [Object Storage Deep Dive](#object-storage-deep-dive)
3. [Block Storage for ML](#block-storage-for-ml)
4. [File Storage Systems](#file-storage-systems)
5. [Data Lakes for ML](#data-lakes-for-ml)
6. [Caching Strategies](#caching-strategies)
7. [Data Versioning](#data-versioning)
8. [Performance Optimization](#performance-optimization)
9. [Cost Optimization](#cost-optimization)
10. [Hands-on Exercise](#hands-on-exercise)

---

## Storage Types Overview

ML workloads require different storage types for different stages of the ML lifecycle.

### Storage Type Comparison

```
┌────────────────┬──────────────────┬─────────────────┬─────────────────┐
│ Storage Type   │ Use Case         │ Performance     │ Cost            │
├────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Object Storage │ Datasets, models │ Medium          │ Low             │
│                │ Archives         │ High throughput │ $0.01-0.02/GB   │
│                │                  │                 │                 │
│ Block Storage  │ Training data    │ Very High       │ Medium-High     │
│                │ Databases        │ Low latency     │ $0.08-0.15/GB   │
│                │                  │                 │                 │
│ File Storage   │ Shared datasets  │ High            │ Medium          │
│                │ Notebooks        │ POSIX compliant │ $0.05-0.20/GB   │
│                │                  │                 │                 │
│ In-Memory      │ Feature cache    │ Extremely High  │ Very High       │
│ (Redis/Memcached)│ Real-time      │ <1ms latency    │ $0.02-0.10/GB/hr│
└────────────────┴──────────────────┴─────────────────┴─────────────────┘
```

### ML Lifecycle Storage Mapping

```
┌──────────────────────────────────────────────────────────────────┐
│                    ML Lifecycle Storage                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Collection          Data Processing         Training       │
│  ───────────────          ───────────────         ────────       │
│  Object Storage      →    Block Storage      →    Block Storage │
│  (S3, Blob, GCS)          (EBS, Persistent       (SSD, NVMe)    │
│                           Disk)                                  │
│                                                                  │
│  Model Storage           Inference Cache         Serving         │
│  ──────────────          ────────────────         ───────        │
│  Object Storage     →    Redis/Memcached    →    File Storage   │
│  (Versioned)             (Feature vectors)        (Shared)      │
│                                                                  │
│  Archival                Logs & Metrics           Backups        │
│  ────────                ───────────────          ───────        │
│  Glacier/Archive         Time-series DB           Object Storage │
│  (Cold storage)          (Prometheus)             (Versioned)    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

**Use Object Storage when:**
- Storing large datasets (>1TB)
- Cost is primary concern
- Data access is infrequent
- Need versioning and lifecycle management
- Sharing data across teams/regions

**Use Block Storage when:**
- Need low latency (<10ms)
- Running databases
- Training with small files
- Require IOPS > 10,000
- Need snapshot capabilities

**Use File Storage when:**
- Multiple compute instances need shared access
- Using traditional file-based workflows
- Need POSIX compliance
- Collaborative notebooks (Jupyter)
- Shared model repository

---

## Object Storage Deep Dive

Object storage is the foundation for ML data management across all cloud providers.

### Provider Comparison

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature         │ AWS S3       │ GCS          │ Azure Blob   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Standard Price  │ $0.023/GB    │ $0.020/GB    │ $0.0184/GB   │
│ Nearline/Cool   │ $0.0125/GB   │ $0.010/GB    │ $0.01/GB     │
│ Archive         │ $0.004/GB    │ $0.0012/GB   │ $0.00099/GB  │
│                 │              │              │              │
│ Transfer Out    │ $0.09/GB     │ $0.12/GB     │ $0.087/GB    │
│ Max Object Size │ 5TB          │ 5TB          │ 4.75TB       │
│ Consistency     │ Strong       │ Strong       │ Strong       │
│ Versioning      │ Yes          │ Yes          │ Yes          │
│ Lifecycle Mgmt  │ Yes          │ Yes          │ Yes          │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

### Storage Classes Strategy

#### AWS S3 Storage Classes

```python
"""
S3 Storage Classes for ML Data Lifecycle
"""

# Active training data (accessed daily)
STANDARD = {
    'storage_class': 'STANDARD',
    'cost_per_gb': 0.023,
    'retrieval_cost': 0,
    'use_case': 'Active datasets, frequently accessed models'
}

# Validation/test data (accessed weekly)
STANDARD_IA = {
    'storage_class': 'STANDARD_IA',
    'cost_per_gb': 0.0125,
    'retrieval_cost': 0.01,
    'minimum_storage_days': 30,
    'use_case': 'Validation sets, model archives'
}

# Old experiments (accessed monthly)
GLACIER_INSTANT = {
    'storage_class': 'GLACIER_INSTANT_RETRIEVAL',
    'cost_per_gb': 0.004,
    'retrieval_cost': 0.03,
    'minimum_storage_days': 90,
    'use_case': 'Experiment archives, old model versions'
}

# Compliance/audit (rarely accessed)
DEEP_ARCHIVE = {
    'storage_class': 'DEEP_ARCHIVE',
    'cost_per_gb': 0.00099,
    'retrieval_cost': 0.02,
    'retrieval_time': '12 hours',
    'minimum_storage_days': 180,
    'use_case': 'Regulatory compliance, long-term backups'
}
```

#### Lifecycle Policy Example

```python
import boto3

s3_client = boto3.client('s3')

# Define lifecycle policy
lifecycle_policy = {
    'Rules': [
        {
            'Id': 'ml-data-lifecycle',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'datasets/'},
            'Transitions': [
                {
                    'Days': 30,
                    'StorageClass': 'STANDARD_IA'
                },
                {
                    'Days': 90,
                    'StorageClass': 'GLACIER_INSTANT_RETRIEVAL'
                },
                {
                    'Days': 365,
                    'StorageClass': 'DEEP_ARCHIVE'
                }
            ],
            'NoncurrentVersionTransitions': [
                {
                    'NoncurrentDays': 30,
                    'StorageClass': 'GLACIER_INSTANT_RETRIEVAL'
                }
            ],
            'NoncurrentVersionExpiration': {
                'NoncurrentDays': 730  # Delete after 2 years
            }
        },
        {
            'Id': 'delete-old-experiments',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'experiments/'},
            'Expiration': {
                'Days': 90  # Auto-delete after 90 days
            }
        },
        {
            'Id': 'production-models',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'models/production/'},
            'Transitions': [
                {
                    'Days': 90,
                    'StorageClass': 'STANDARD_IA'
                }
            ]
            # Never auto-delete production models
        }
    ]
}

# Apply lifecycle policy
s3_client.put_bucket_lifecycle_configuration(
    Bucket='ml-data-bucket',
    LifecycleConfiguration=lifecycle_policy
)

print("Lifecycle policy applied successfully")
```

### Versioning for Reproducibility

```python
import boto3
from datetime import datetime

s3_client = boto3.client('s3')

# Enable versioning
s3_client.put_bucket_versioning(
    Bucket='ml-data-bucket',
    VersioningConfiguration={'Status': 'Enabled'}
)

# Upload versioned dataset
def upload_dataset_version(local_path, dataset_name, version=None):
    """
    Upload dataset with automatic versioning

    Args:
        local_path: Local file path
        dataset_name: Dataset identifier
        version: Optional version tag

    Returns:
        Version ID
    """
    if version is None:
        version = datetime.now().strftime('%Y%m%d_%H%M%S')

    key = f'datasets/{dataset_name}/data.parquet'

    # Upload with metadata
    response = s3_client.upload_file(
        local_path,
        'ml-data-bucket',
        key,
        ExtraArgs={
            'Metadata': {
                'version': version,
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat()
            }
        }
    )

    # Get version ID
    obj = s3_client.head_object(Bucket='ml-data-bucket', Key=key)
    version_id = obj['VersionId']

    print(f"Uploaded {dataset_name} version {version}: {version_id}")
    return version_id

# Download specific version
def download_dataset_version(dataset_name, version_id, local_path):
    """Download specific version of dataset"""
    key = f'datasets/{dataset_name}/data.parquet'

    s3_client.download_file(
        'ml-data-bucket',
        key,
        local_path,
        ExtraArgs={'VersionId': version_id}
    )

    print(f"Downloaded {dataset_name} version {version_id}")

# List all versions
def list_dataset_versions(dataset_name):
    """List all versions of a dataset"""
    key = f'datasets/{dataset_name}/data.parquet'

    response = s3_client.list_object_versions(
        Bucket='ml-data-bucket',
        Prefix=key
    )

    versions = []
    for version in response.get('Versions', []):
        versions.append({
            'version_id': version['VersionId'],
            'last_modified': version['LastModified'],
            'size': version['Size']
        })

    return versions

# Usage
version_id = upload_dataset_version('./imagenet_v2.parquet', 'imagenet', 'v2.0')
download_dataset_version('imagenet', version_id, './downloaded.parquet')
print(list_dataset_versions('imagenet'))
```

### Multi-Part Upload for Large Files

```python
import boto3
from boto3.s3.transfer import TransferConfig

s3_client = boto3.client('s3')

# Configure multi-part upload
config = TransferConfig(
    multipart_threshold=1024 * 25,      # 25MB
    max_concurrency=10,                  # 10 parallel uploads
    multipart_chunksize=1024 * 25,      # 25MB chunks
    use_threads=True
)

def upload_large_file(local_path, bucket, key):
    """
    Upload large file with progress tracking

    Optimized for datasets > 100MB
    """
    import os
    from tqdm import tqdm

    file_size = os.path.getsize(local_path)

    with tqdm(total=file_size, unit='B', unit_scale=True, desc='Uploading') as pbar:
        def callback(bytes_transferred):
            pbar.update(bytes_transferred)

        s3_client.upload_file(
            local_path,
            bucket,
            key,
            Config=config,
            Callback=callback
        )

    print(f"Upload complete: s3://{bucket}/{key}")

# Upload 10GB dataset
upload_large_file(
    './imagenet_full.tar.gz',
    'ml-data-bucket',
    'datasets/imagenet/full.tar.gz'
)
```

---

## Block Storage for ML

Block storage provides high-performance, low-latency storage for training workloads.

### Provider Comparison

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature         │ AWS EBS      │ GCP PD       │ Azure Disk   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ SSD (gp3)       │ $0.08/GB     │ $0.17/GB     │ $0.15/GB     │
│ IOPS (SSD)      │ 16,000       │ 100,000      │ 20,000       │
│ Throughput      │ 1,000 MB/s   │ 1,200 MB/s   │ 900 MB/s     │
│ Max Size        │ 16 TB        │ 64 TB        │ 32 TB        │
│                 │              │              │              │
│ NVMe (io2)      │ $0.125/GB    │ $0.17/GB     │ $0.40/GB     │
│ IOPS (NVMe)     │ 64,000       │ 100,000      │ 160,000      │
│ Throughput      │ 4,000 MB/s   │ 1,200 MB/s   │ 4,000 MB/s   │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

### Volume Types for ML

#### AWS EBS Volume Selection

```python
"""
EBS Volume Types for Different ML Workloads
"""

# General Purpose SSD (gp3) - Most ML workloads
gp3_config = {
    'type': 'gp3',
    'size_gb': 1000,
    'iops': 3000,              # Baseline: 3,000
    'throughput_mbps': 125,    # Baseline: 125 MB/s
    'cost_per_gb': 0.08,
    'use_case': [
        'Standard training datasets',
        'Model checkpoints',
        'General ML workloads'
    ]
}

# Provisioned IOPS SSD (io2) - High-performance training
io2_config = {
    'type': 'io2',
    'size_gb': 1000,
    'iops': 10000,             # Up to 64,000
    'throughput_mbps': 1000,   # Up to 4,000 MB/s
    'cost_per_gb': 0.125,
    'cost_per_iops': 0.065,
    'use_case': [
        'Large-scale training (ImageNet, COCO)',
        'High-throughput data pipelines',
        'Real-time inference with many small files'
    ]
}

# Throughput Optimized HDD (st1) - Sequential reads
st1_config = {
    'type': 'st1',
    'size_gb': 1000,
    'throughput_mbps': 500,    # Max: 500 MB/s
    'cost_per_gb': 0.045,
    'use_case': [
        'Large sequential datasets',
        'Video/audio processing',
        'Data preprocessing'
    ]
}
```

#### Creating Optimized EBS Volumes

```python
import boto3

ec2_client = boto3.client('ec2')

def create_ml_training_volume(size_gb=1000, availability_zone='us-east-1a'):
    """
    Create optimized EBS volume for ML training

    Args:
        size_gb: Volume size in GB
        availability_zone: AZ for the volume

    Returns:
        Volume ID
    """
    response = ec2_client.create_volume(
        AvailabilityZone=availability_zone,
        Size=size_gb,
        VolumeType='gp3',
        Iops=3000,              # Baseline
        Throughput=125,         # MB/s
        Encrypted=True,         # Always encrypt
        TagSpecifications=[
            {
                'ResourceType': 'volume',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ml-training-volume'},
                    {'Key': 'Purpose', 'Value': 'ml-training'},
                    {'Key': 'Project', 'Value': 'image-classification'}
                ]
            }
        ]
    )

    volume_id = response['VolumeId']
    print(f"Created volume: {volume_id}")

    # Wait for volume to be available
    waiter = ec2_client.get_waiter('volume_available')
    waiter.wait(VolumeIds=[volume_id])

    return volume_id

# Attach volume to instance
def attach_volume(volume_id, instance_id, device='/dev/sdf'):
    """Attach EBS volume to EC2 instance"""
    ec2_client.attach_volume(
        VolumeId=volume_id,
        InstanceId=instance_id,
        Device=device
    )

    # Wait for attachment
    waiter = ec2_client.get_waiter('volume_in_use')
    waiter.wait(VolumeIds=[volume_id])

    print(f"Attached {volume_id} to {instance_id} at {device}")

# Create snapshot for backup
def create_volume_snapshot(volume_id, description):
    """Create snapshot of EBS volume"""
    response = ec2_client.create_snapshot(
        VolumeId=volume_id,
        Description=description,
        TagSpecifications=[
            {
                'ResourceType': 'snapshot',
                'Tags': [
                    {'Key': 'Name', 'Value': f'snapshot-{volume_id}'},
                    {'Key': 'Type', 'Value': 'ml-data-backup'}
                ]
            }
        ]
    )

    snapshot_id = response['SnapshotId']
    print(f"Created snapshot: {snapshot_id}")

    return snapshot_id
```

### Local NVMe Storage

For maximum performance, use local NVMe storage (ephemeral):

```bash
# AWS: Instance storage (i3, i4i instances)
# - i3.2xlarge: 1x 1.9TB NVMe SSD
# - i4i.4xlarge: 1x 3.75TB NVMe SSD
# - Throughput: Up to 16 GB/s

# Create EC2 instance with instance storage
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type i3.2xlarge \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/sdb",
      "VirtualName": "ephemeral0"
    }
  ]'

# Mount NVMe storage
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir -p /data
sudo mount /dev/nvme0n1 /data
sudo chown -R ubuntu:ubuntu /data

# Performance test
sudo fio --name=randwrite --ioengine=libaio --iodepth=32 \
  --rw=randwrite --bs=4k --direct=1 --size=1G \
  --numjobs=4 --runtime=60 --group_reporting \
  --filename=/data/test

# Expected results: 50,000+ IOPS, 200+ MB/s
```

**Important**: Instance storage is ephemeral - data is lost on instance stop/termination. Always backup to S3.

```python
import subprocess
import schedule
import time

def backup_nvme_to_s3():
    """
    Backup NVMe storage to S3 (for ephemeral instance storage)
    """
    print("Starting backup...")

    # Create tarball
    subprocess.run([
        'tar', '-czf', '/tmp/data_backup.tar.gz',
        '-C', '/data', '.'
    ])

    # Upload to S3
    subprocess.run([
        'aws', 's3', 'cp',
        '/tmp/data_backup.tar.gz',
        's3://ml-data-bucket/backups/data_backup.tar.gz'
    ])

    # Cleanup
    subprocess.run(['rm', '/tmp/data_backup.tar.gz'])

    print("Backup complete")

# Schedule hourly backups
schedule.every().hour.do(backup_nvme_to_s3)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## File Storage Systems

Shared file systems enable multiple compute instances to access the same data.

### Provider Comparison

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature         │ AWS EFS      │ GCP Filestore│ Azure Files  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Performance     │ Up to 10GB/s │ Up to 1.6GB/s│ Up to 10GB/s │
│ Capacity        │ Petabytes    │ 100TB        │ 100TB        │
│ Protocol        │ NFS v4.1     │ NFS v3       │ SMB 3.0/NFS  │
│                 │              │              │              │
│ Standard Price  │ $0.30/GB     │ $0.20/GB     │ $0.10/GB     │
│ Performance     │ $0.60/GB     │ $0.20/GB     │ $0.20/GB     │
│ IOPS            │ 500K+        │ 100K         │ 100K         │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

### AWS EFS for Shared ML Datasets

```python
import boto3

efs_client = boto3.client('efs')
ec2_client = boto3.client('ec2')

def create_efs_filesystem(name='ml-shared-storage'):
    """
    Create EFS filesystem for shared ML data

    Returns:
        File system ID
    """
    response = efs_client.create_file_system(
        CreationToken=name,
        PerformanceMode='generalPurpose',  # or 'maxIO' for >1000 instances
        ThroughputMode='bursting',          # or 'provisioned'
        Encrypted=True,
        Tags=[
            {'Key': 'Name', 'Value': name},
            {'Key': 'Purpose', 'Value': 'ml-training'}
        ]
    )

    filesystem_id = response['FileSystemId']
    print(f"Created EFS: {filesystem_id}")

    # Wait for filesystem to be available
    waiter = efs_client.get_waiter('file_system_available')
    waiter.wait(FileSystemId=filesystem_id)

    return filesystem_id

def create_mount_target(filesystem_id, subnet_id, security_group_id):
    """Create EFS mount target in subnet"""
    response = efs_client.create_mount_target(
        FileSystemId=filesystem_id,
        SubnetId=subnet_id,
        SecurityGroups=[security_group_id]
    )

    mount_target_id = response['MountTargetId']
    print(f"Created mount target: {mount_target_id}")

    return mount_target_id

# Mount EFS on EC2 instance
"""
# Install EFS utilities
sudo yum install -y amazon-efs-utils

# Create mount point
sudo mkdir -p /mnt/efs

# Mount filesystem
sudo mount -t efs -o tls fs-12345678:/ /mnt/efs

# Verify
df -h /mnt/efs

# Add to /etc/fstab for automatic mounting
echo "fs-12345678:/ /mnt/efs efs defaults,_netdev 0 0" | sudo tee -a /etc/fstab
"""
```

### Use Cases for Shared File Storage

```python
"""
Shared File Storage Use Cases for ML
"""

# Use Case 1: Collaborative Jupyter Notebooks
shared_notebooks = {
    'path': '/mnt/efs/notebooks',
    'use_case': 'Multiple data scientists accessing same notebooks',
    'benefit': 'Real-time collaboration, shared experiments',
    'users': ['data-scientist-1', 'data-scientist-2', 'data-scientist-3']
}

# Use Case 2: Shared Training Datasets
shared_datasets = {
    'path': '/mnt/efs/datasets',
    'use_case': 'Multiple training jobs accessing same data',
    'benefit': 'No data duplication, consistent dataset version',
    'consumers': ['training-job-1', 'training-job-2', 'validation-job']
}

# Use Case 3: Model Repository
model_repository = {
    'path': '/mnt/efs/models',
    'use_case': 'Centralized model storage for inference fleet',
    'benefit': 'Single source of truth, instant updates',
    'structure': {
        'production/': 'Production models',
        'staging/': 'Staging models',
        'experimental/': 'Experimental models'
    }
}

# Use Case 4: Distributed Training Checkpoints
distributed_checkpoints = {
    'path': '/mnt/efs/checkpoints',
    'use_case': 'Multi-node training checkpoints',
    'benefit': 'Fault tolerance across nodes',
    'pattern': 'Save checkpoints to shared storage for resumption'
}
```

---

## Data Lakes for ML

Data lakes provide centralized repositories for structured and unstructured data at scale.

### Data Lake Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        ML Data Lake                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Raw Zone (Object Storage)                                     │
│  ─────────────────────────                                     │
│  ├── images/                                                   │
│  │   ├── train/          (100GB)                               │
│  │   ├── validation/     (20GB)                                │
│  │   └── test/           (20GB)                                │
│  ├── videos/              (500GB)                              │
│  └── logs/                (1TB)                                │
│                                                                │
│  Processed Zone (Parquet/ORC)                                  │
│  ─────────────────────────────                                 │
│  ├── features/                                                 │
│  │   ├── image_embeddings.parquet      (10GB)                 │
│  │   └── video_features.parquet        (50GB)                 │
│  ├── labels/                                                   │
│  │   └── annotations.parquet           (1GB)                  │
│  └── preprocessed/                                             │
│      └── normalized_images/            (80GB)                  │
│                                                                │
│  Curated Zone (Training-Ready)                                 │
│  ──────────────────────────────                                │
│  ├── train.tfrecord       (50GB)                               │
│  ├── val.tfrecord         (10GB)                               │
│  └── test.tfrecord        (10GB)                               │
│                                                                │
│  Models Zone (Versioned)                                       │
│  ────────────────────────                                      │
│  ├── resnet50/                                                 │
│  │   ├── v1.0/           (100MB)                               │
│  │   ├── v1.1/           (100MB)                               │
│  │   └── v2.0/           (100MB)                               │
│  └── experiments/                                              │
│      └── exp_*/           (50MB each)                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Data Lake Organization Strategy

```python
"""
Data Lake Organization for ML Projects
"""

import os
from pathlib import Path

class MLDataLake:
    """
    Organize ML data in a structured data lake
    """

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.zones = {
            'raw': self.base_path / 'raw',
            'processed': self.base_path / 'processed',
            'curated': self.base_path / 'curated',
            'models': self.base_path / 'models',
            'experiments': self.base_path / 'experiments'
        }

        # Create directory structure
        for zone in self.zones.values():
            zone.mkdir(parents=True, exist_ok=True)

    def get_raw_path(self, data_type, split=None):
        """
        Get path for raw data

        Args:
            data_type: Type of data (images, videos, text)
            split: Optional split (train, val, test)

        Returns:
            Path object
        """
        path = self.zones['raw'] / data_type
        if split:
            path = path / split
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_processed_path(self, feature_type):
        """Get path for processed features"""
        path = self.zones['processed'] / feature_type
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_curated_path(self, dataset_name):
        """Get path for curated training data"""
        path = self.zones['curated'] / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_path(self, model_name, version):
        """Get path for model artifacts"""
        path = self.zones['models'] / model_name / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_experiment_path(self, experiment_id):
        """Get path for experiment artifacts"""
        path = self.zones['experiments'] / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

# Usage
data_lake = MLDataLake('s3://ml-data-lake')

# Organize raw data
train_images = data_lake.get_raw_path('images', 'train')
print(f"Store training images in: {train_images}")

# Processed features
embeddings = data_lake.get_processed_path('image_embeddings')
print(f"Store embeddings in: {embeddings}")

# Curated training data
training_data = data_lake.get_curated_path('imagenet_processed')
print(f"Store training-ready data in: {training_data}")

# Model versioning
model_v1 = data_lake.get_model_path('resnet50', 'v1.0')
print(f"Store model v1.0 in: {model_v1}")
```

### Metadata Management

```python
import json
from datetime import datetime
import boto3

s3_client = boto3.client('s3')

class DatasetMetadata:
    """
    Manage metadata for datasets in data lake
    """

    def __init__(self, bucket_name):
        self.bucket = bucket_name
        self.metadata_prefix = 'metadata/'

    def register_dataset(self, dataset_name, metadata):
        """
        Register dataset with metadata

        Args:
            dataset_name: Unique dataset identifier
            metadata: Dictionary with dataset information
        """
        metadata_key = f"{self.metadata_prefix}{dataset_name}.json"

        # Enrich metadata
        metadata.update({
            'registered_at': datetime.now().isoformat(),
            'dataset_name': dataset_name
        })

        # Upload metadata
        s3_client.put_object(
            Bucket=self.bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )

        print(f"Registered dataset: {dataset_name}")

    def get_dataset_metadata(self, dataset_name):
        """Retrieve dataset metadata"""
        metadata_key = f"{self.metadata_prefix}{dataset_name}.json"

        response = s3_client.get_object(
            Bucket=self.bucket,
            Key=metadata_key
        )

        metadata = json.loads(response['Body'].read())
        return metadata

    def list_datasets(self):
        """List all registered datasets"""
        response = s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.metadata_prefix
        )

        datasets = []
        for obj in response.get('Contents', []):
            dataset_name = obj['Key'].replace(self.metadata_prefix, '').replace('.json', '')
            datasets.append(dataset_name)

        return datasets

# Usage
metadata_mgr = DatasetMetadata('ml-data-lake')

# Register new dataset
metadata_mgr.register_dataset('imagenet_2024', {
    'description': 'ImageNet dataset for image classification',
    'size_gb': 150,
    'num_samples': 1281167,
    'num_classes': 1000,
    'format': 'JPEG',
    'location': 's3://ml-data-lake/raw/images/imagenet/',
    'splits': {
        'train': 1281167,
        'val': 50000,
        'test': 100000
    },
    'preprocessing': {
        'resize': [224, 224],
        'normalization': 'imagenet',
        'augmentation': ['random_crop', 'horizontal_flip']
    },
    'version': '2024.1',
    'license': 'Academic use only',
    'citation': 'Deng et al., ImageNet, 2009'
})

# Retrieve metadata
metadata = metadata_mgr.get_dataset_metadata('imagenet_2024')
print(f"Dataset: {metadata['description']}")
print(f"Size: {metadata['size_gb']} GB")
print(f"Samples: {metadata['num_samples']}")
```

---

## Caching Strategies

Caching reduces latency and cost by storing frequently accessed data closer to compute.

### Multi-Level Caching Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Caching Hierarchy                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Level 1: In-Memory Cache (Redis)                         │
│  ─────────────────────────────────                         │
│  - Hot features (embeddings)           <1ms latency       │
│  - Recent predictions                  Size: 1-10 GB      │
│  - Session data                        Cost: High         │
│                                                            │
│  Level 2: Local SSD Cache                                 │
│  ─────────────────────────                                 │
│  - Preprocessed data                   1-5ms latency      │
│  - Model weights                       Size: 100-1000 GB  │
│  - Recent datasets                     Cost: Medium       │
│                                                            │
│  Level 3: Network Storage (EFS/NFS)                       │
│  ───────────────────────────────────                       │
│  - Shared datasets                     10-50ms latency    │
│  - Model repository                    Size: 1-10 TB      │
│  - Checkpoints                         Cost: Medium       │
│                                                            │
│  Level 4: Object Storage (S3)                             │
│  ─────────────────────────────                             │
│  - Full datasets                       100-500ms latency  │
│  - Archives                            Size: Unlimited    │
│  - Backups                             Cost: Low          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Redis Cache for ML Features

```python
import redis
import numpy as np
import pickle
from typing import Optional

class FeatureCache:
    """
    Redis-based cache for ML features (embeddings, predictions)
    """

    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False  # Store binary data
        )
        self.default_ttl = ttl  # seconds

    def set_embedding(self, key: str, embedding: np.ndarray, ttl: Optional[int] = None):
        """
        Cache embedding vector

        Args:
            key: Unique identifier (e.g., 'img_123_embedding')
            embedding: Numpy array
            ttl: Time to live in seconds
        """
        ttl = ttl or self.default_ttl

        # Serialize numpy array
        serialized = pickle.dumps(embedding)

        # Store with expiration
        self.redis_client.setex(
            name=key,
            time=ttl,
            value=serialized
        )

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding"""
        serialized = self.redis_client.get(key)

        if serialized:
            embedding = pickle.loads(serialized)
            return embedding

        return None

    def set_prediction(self, image_id: str, prediction: dict, ttl: Optional[int] = None):
        """Cache prediction result"""
        ttl = ttl or self.default_ttl
        key = f"pred:{image_id}"

        self.redis_client.setex(
            name=key,
            time=ttl,
            value=pickle.dumps(prediction)
        )

    def get_prediction(self, image_id: str) -> Optional[dict]:
        """Retrieve cached prediction"""
        key = f"pred:{image_id}"
        serialized = self.redis_client.get(key)

        if serialized:
            return pickle.loads(serialized)

        return None

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        info = self.redis_client.info('stats')
        memory = self.redis_client.info('memory')

        return {
            'total_keys': self.redis_client.dbsize(),
            'hits': info.get('keyspace_hits', 0),
            'misses': info.get('keyspace_misses', 0),
            'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1),
            'memory_used_mb': memory.get('used_memory', 0) / 1024 / 1024,
            'memory_peak_mb': memory.get('used_memory_peak', 0) / 1024 / 1024
        }

# Usage
cache = FeatureCache(host='redis.example.com', ttl=3600)

# Cache embedding
embedding = np.random.rand(512)  # 512-dim embedding
cache.set_embedding('img_12345_embedding', embedding, ttl=7200)

# Retrieve embedding
cached_embedding = cache.get_embedding('img_12345_embedding')
if cached_embedding is not None:
    print(f"Cache hit! Embedding shape: {cached_embedding.shape}")
else:
    print("Cache miss, need to compute embedding")

# Cache prediction
prediction = {
    'class': 'cat',
    'confidence': 0.95,
    'top_5': ['cat', 'dog', 'bird', 'fish', 'hamster']
}
cache.set_prediction('img_12345', prediction)

# Get cache stats
stats = cache.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Memory used: {stats['memory_used_mb']:.2f} MB")
```

### Local SSD Caching

```python
import os
import hashlib
import shutil
from pathlib import Path
import boto3

class LocalDiskCache:
    """
    Local SSD cache for training data
    """

    def __init__(self, cache_dir='/mnt/nvme/cache', max_size_gb=100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.s3_client = boto3.client('s3')

    def _get_cache_path(self, s3_key):
        """Generate cache path from S3 key"""
        # Use hash to avoid filesystem issues with long paths
        key_hash = hashlib.md5(s3_key.encode()).hexdigest()
        return self.cache_dir / key_hash

    def get(self, bucket, s3_key, local_path):
        """
        Get file from cache or download from S3

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Destination path

        Returns:
            True if cached, False if downloaded
        """
        cache_path = self._get_cache_path(s3_key)

        if cache_path.exists():
            # Cache hit
            shutil.copy(cache_path, local_path)
            # Update access time
            os.utime(cache_path, None)
            return True
        else:
            # Cache miss - download from S3
            self.s3_client.download_file(bucket, s3_key, local_path)

            # Add to cache
            shutil.copy(local_path, cache_path)

            # Evict if needed
            self._evict_if_needed()

            return False

    def _evict_if_needed(self):
        """Evict old files if cache is full (LRU)"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*') if f.is_file())

        if total_size > self.max_size_bytes:
            # Get files sorted by access time (LRU)
            files = [(f, f.stat().st_atime) for f in self.cache_dir.glob('*') if f.is_file()]
            files.sort(key=lambda x: x[1])

            # Remove oldest files
            bytes_to_free = total_size - (self.max_size_bytes * 0.8)  # Free to 80%
            freed = 0

            for file_path, _ in files:
                if freed >= bytes_to_free:
                    break

                file_size = file_path.stat().st_size
                file_path.unlink()
                freed += file_size
                print(f"Evicted: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")

    def get_stats(self):
        """Get cache statistics"""
        files = list(self.cache_dir.glob('*'))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        return {
            'num_files': len(files),
            'total_size_gb': total_size / 1024 / 1024 / 1024,
            'utilization': total_size / self.max_size_bytes
        }

# Usage
cache = LocalDiskCache(cache_dir='/mnt/nvme/cache', max_size_gb=100)

# Download with caching
was_cached = cache.get(
    bucket='ml-data-bucket',
    s3_key='datasets/imagenet/train/img_001.jpg',
    local_path='/tmp/img_001.jpg'
)

if was_cached:
    print("Served from local cache (fast!)")
else:
    print("Downloaded from S3 (slower)")

# Check cache stats
stats = cache.get_stats()
print(f"Cache: {stats['num_files']} files, {stats['total_size_gb']:.2f} GB ({stats['utilization']:.1%} full)")
```

---

## Data Versioning

Version control for datasets ensures reproducibility and enables experimentation.

### DVC (Data Version Control)

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC in project
cd ml-project
git init
dvc init

# Configure S3 remote
dvc remote add -d myremote s3://ml-data-bucket/dvc-storage
dvc remote modify myremote region us-east-1

# Track dataset with DVC
dvc add data/imagenet/train
# This creates: data/imagenet/train.dvc

# Commit to git (only the .dvc file, not the data)
git add data/imagenet/train.dvc .gitignore
git commit -m "Add training dataset"

# Push data to S3
dvc push

# On another machine, clone and get data
git clone https://github.com/myorg/ml-project.git
cd ml-project
dvc pull  # Downloads data from S3

# Create a new version
# ... modify data ...
dvc add data/imagenet/train
git add data/imagenet/train.dvc
git commit -m "Update training dataset v2"
dvc push

# Switch between versions
git checkout <commit-hash>
dvc checkout  # Gets the data version for that commit
```

### Custom Versioning System

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path
import boto3

class DatasetVersioning:
    """
    Custom dataset versioning system
    """

    def __init__(self, bucket_name, project_name):
        self.bucket = bucket_name
        self.project = project_name
        self.s3_client = boto3.client('s3')
        self.versions_prefix = f'versions/{project_name}/'

    def compute_dataset_hash(self, dataset_path):
        """Compute hash of dataset directory"""
        hash_md5 = hashlib.md5()

        for filepath in sorted(Path(dataset_path).rglob('*')):
            if filepath.is_file():
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def create_version(self, dataset_path, version_name, metadata=None):
        """
        Create a new dataset version

        Args:
            dataset_path: Local path to dataset
            version_name: Version identifier (e.g., 'v1.0', 'experiment-20240101')
            metadata: Optional metadata dictionary

        Returns:
            Version ID (hash)
        """
        # Compute dataset hash
        dataset_hash = self.compute_dataset_hash(dataset_path)

        # Create version metadata
        version_metadata = {
            'version_name': version_name,
            'version_id': dataset_hash,
            'created_at': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'project': self.project
        }

        if metadata:
            version_metadata.update(metadata)

        # Save metadata
        metadata_key = f'{self.versions_prefix}{version_name}/metadata.json'
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=metadata_key,
            Body=json.dumps(version_metadata, indent=2)
        )

        # Upload dataset files
        for filepath in Path(dataset_path).rglob('*'):
            if filepath.is_file():
                relative_path = filepath.relative_to(dataset_path)
                s3_key = f'{self.versions_prefix}{version_name}/data/{relative_path}'

                self.s3_client.upload_file(
                    str(filepath),
                    self.bucket,
                    s3_key
                )

        print(f"Created version {version_name} (ID: {dataset_hash})")
        return dataset_hash

    def get_version(self, version_name, local_path):
        """Download specific version"""
        version_prefix = f'{self.versions_prefix}{version_name}/data/'

        # Download all files
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=version_prefix):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                relative_path = s3_key.replace(version_prefix, '')
                local_file = Path(local_path) / relative_path

                local_file.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(self.bucket, s3_key, str(local_file))

        print(f"Downloaded version {version_name} to {local_path}")

    def list_versions(self):
        """List all versions"""
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.versions_prefix,
            Delimiter='/'
        )

        versions = []
        for prefix in response.get('CommonPrefixes', []):
            version_name = prefix['Prefix'].replace(self.versions_prefix, '').rstrip('/')
            versions.append(version_name)

        return versions

    def compare_versions(self, version1, version2):
        """Compare two versions"""
        # Get metadata for both versions
        meta1_key = f'{self.versions_prefix}{version1}/metadata.json'
        meta2_key = f'{self.versions_prefix}{version2}/metadata.json'

        meta1 = json.loads(self.s3_client.get_object(Bucket=self.bucket, Key=meta1_key)['Body'].read())
        meta2 = json.loads(self.s3_client.get_object(Bucket=self.bucket, Key=meta2_key)['Body'].read())

        return {
            'version1': version1,
            'version2': version2,
            'same_data': meta1['version_id'] == meta2['version_id'],
            'created_at_diff': meta2['created_at'] + ' vs ' + meta1['created_at']
        }

# Usage
versioning = DatasetVersioning('ml-data-bucket', 'image-classification')

# Create version
versioning.create_version(
    dataset_path='./data/imagenet',
    version_name='v1.0',
    metadata={
        'description': 'Initial ImageNet dataset',
        'num_samples': 1281167,
        'preprocessing': 'resize_224x224'
    }
)

# Create another version with changes
versioning.create_version(
    dataset_path='./data/imagenet_augmented',
    version_name='v1.1',
    metadata={
        'description': 'ImageNet with augmentation',
        'num_samples': 1281167,
        'preprocessing': 'resize_224x224 + random_crop + flip'
    }
)

# List versions
versions = versioning.list_versions()
print(f"Available versions: {versions}")

# Get specific version
versioning.get_version('v1.0', './data/downloaded')

# Compare versions
comparison = versioning.compare_versions('v1.0', 'v1.1')
print(f"Same data: {comparison['same_data']}")
```

---

## Performance Optimization

### Benchmarking Storage Performance

```python
import time
import numpy as np
import boto3
from io import BytesIO

def benchmark_s3_performance(bucket, num_files=100, file_size_mb=10):
    """
    Benchmark S3 read/write performance

    Args:
        bucket: S3 bucket name
        num_files: Number of files to test
        file_size_mb: Size of each file in MB

    Returns:
        Performance metrics
    """
    s3_client = boto3.client('s3')

    # Generate test data
    data = np.random.bytes(file_size_mb * 1024 * 1024)

    # Upload benchmark
    upload_times = []
    for i in range(num_files):
        key = f'benchmark/upload_{i}.bin'

        start = time.time()
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        elapsed = time.time() - start

        upload_times.append(elapsed)

    # Download benchmark
    download_times = []
    for i in range(num_files):
        key = f'benchmark/upload_{i}.bin'

        start = time.time()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        _ = response['Body'].read()
        elapsed = time.time() - start

        download_times.append(elapsed)

    # Calculate metrics
    avg_upload = np.mean(upload_times)
    avg_download = np.mean(download_times)
    upload_throughput = file_size_mb / avg_upload
    download_throughput = file_size_mb / avg_download

    # Cleanup
    for i in range(num_files):
        s3_client.delete_object(Bucket=bucket, Key=f'benchmark/upload_{i}.bin')

    return {
        'avg_upload_time': avg_upload,
        'avg_download_time': avg_download,
        'upload_throughput_mbps': upload_throughput,
        'download_throughput_mbps': download_throughput,
        'p95_upload': np.percentile(upload_times, 95),
        'p95_download': np.percentile(download_times, 95)
    }

# Run benchmark
results = benchmark_s3_performance('ml-data-bucket', num_files=50, file_size_mb=10)
print(f"Upload: {results['upload_throughput_mbps']:.2f} MB/s (p95: {results['p95_upload']:.2f}s)")
print(f"Download: {results['download_throughput_mbps']:.2f} MB/s (p95: {results['p95_download']:.2f}s)")
```

### Data Pipeline Optimization

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3

def download_file_worker(args):
    """Worker function for parallel downloads"""
    bucket, key, local_path = args
    s3_client = boto3.client('s3')

    try:
        s3_client.download_file(bucket, key, local_path)
        return local_path, True
    except Exception as e:
        return local_path, False

def parallel_download(bucket, keys, local_dir, max_workers=10):
    """
    Download multiple files in parallel

    Args:
        bucket: S3 bucket name
        keys: List of S3 keys
        local_dir: Local directory
        max_workers: Number of parallel workers

    Returns:
        Number of successful downloads
    """
    from pathlib import Path

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Prepare download tasks
    tasks = []
    for key in keys:
        filename = key.split('/')[-1]
        local_path = local_dir / filename
        tasks.append((bucket, key, str(local_path)))

    # Execute in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file_worker, task) for task in tasks]

        for future in as_completed(futures):
            local_path, success = future.result()
            if success:
                successful += 1

    return successful

# Usage
keys = [f'datasets/images/img_{i:05d}.jpg' for i in range(1000)]
successful = parallel_download('ml-data-bucket', keys, './data', max_workers=20)
print(f"Downloaded {successful}/{len(keys)} files")
```

---

## Cost Optimization

### Cost Comparison: 1TB Dataset Over 1 Year

```
Scenario: 1TB dataset stored for 1 year

┌────────────────────────┬────────────┬─────────────┬─────────────┐
│ Storage Strategy       │ AWS Cost   │ GCP Cost    │ Azure Cost  │
├────────────────────────┼────────────┼─────────────┼─────────────┤
│ All Standard           │ $276       │ $240        │ $221        │
│                        │            │             │             │
│ Lifecycle (30/90/365)  │ $96        │ $72         │ $78         │
│ Savings                │ 65%        │ 70%         │ 65%         │
│                        │            │             │             │
│ Aggressive Archive     │ $48        │ $14         │ $12         │
│ (90% to archive)       │            │             │             │
│ Savings                │ 83%        │ 94%         │ 95%         │
└────────────────────────┴────────────┴─────────────┴─────────────┘

Recommendations:
- Active training data: Standard storage
- Validation data: Move to Cool/Nearline after 30 days
- Old experiments: Archive after 90 days
- Compliance data: Deep Archive immediately
```

### Cost Optimization Script

```python
import boto3
from datetime import datetime, timedelta

s3_client = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

def analyze_storage_costs(bucket_name):
    """
    Analyze storage costs and provide optimization recommendations

    Returns:
        Cost analysis and recommendations
    """
    # Get bucket size by storage class
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/S3',
        MetricName='BucketSizeBytes',
        Dimensions=[
            {'Name': 'BucketName', 'Value': bucket_name},
            {'Name': 'StorageType', 'Value': 'StandardStorage'}
        ],
        StartTime=datetime.now() - timedelta(days=1),
        EndTime=datetime.now(),
        Period=86400,
        Statistics=['Average']
    )

    standard_bytes = response['Datapoints'][0]['Average'] if response['Datapoints'] else 0
    standard_gb = standard_bytes / 1024 / 1024 / 1024

    # Calculate costs
    current_monthly_cost = standard_gb * 0.023  # Standard storage

    # Estimate savings with lifecycle
    optimized_cost = (
        standard_gb * 0.3 * 0.023 +  # 30% stays in Standard
        standard_gb * 0.4 * 0.0125 +  # 40% moves to IA
        standard_gb * 0.3 * 0.004     # 30% moves to Glacier
    )

    savings = current_monthly_cost - optimized_cost
    savings_pct = (savings / current_monthly_cost * 100) if current_monthly_cost > 0 else 0

    return {
        'bucket': bucket_name,
        'size_gb': standard_gb,
        'current_monthly_cost': current_monthly_cost,
        'optimized_monthly_cost': optimized_cost,
        'monthly_savings': savings,
        'savings_percentage': savings_pct,
        'annual_savings': savings * 12
    }

# Usage
analysis = analyze_storage_costs('ml-data-bucket')
print(f"Bucket: {analysis['bucket']}")
print(f"Size: {analysis['size_gb']:.2f} GB")
print(f"Current cost: ${analysis['current_monthly_cost']:.2f}/month")
print(f"Optimized cost: ${analysis['optimized_monthly_cost']:.2f}/month")
print(f"Savings: ${analysis['monthly_savings']:.2f}/month ({analysis['savings_percentage']:.1f}%)")
print(f"Annual savings: ${analysis['annual_savings']:.2f}")
```

---

## Hands-on Exercise

### Exercise: Build an Optimized ML Data Pipeline

**Objective**: Create a complete data pipeline with:
- S3 data lake with lifecycle management
- Local SSD caching for training
- Redis caching for inference features
- Data versioning with DVC
- Cost optimization with tiering

**Steps**:

1. **Set up data lake structure**
2. **Configure lifecycle policies**
3. **Implement local caching**
4. **Add Redis feature cache**
5. **Set up DVC versioning**
6. **Benchmark performance**
7. **Calculate cost savings**

**Expected Results**:
- 70% faster training (with caching)
- 65% cost reduction (with lifecycle)
- Full reproducibility (with versioning)

---

## Summary

In this lesson, you learned:

✅ Compare object, block, and file storage types
✅ Implement data lakes with proper organization
✅ Configure lifecycle policies for cost optimization (65-70% savings)
✅ Design multi-level caching strategies (Redis, local SSD)
✅ Implement data versioning for reproducibility
✅ Optimize storage performance for ML workloads
✅ Benchmark and compare cloud storage providers
✅ Apply security best practices

**Key Takeaways**:
- Object storage is ideal for large datasets (S3, GCS, Blob)
- Block storage provides low latency for training (EBS, Persistent Disk)
- Lifecycle management saves 65-70% on storage costs
- Multi-level caching dramatically improves training speed
- Data versioning is essential for reproducibility

**Next Steps**:
- Complete hands-on exercise
- Implement data lake for your project
- Set up lifecycle policies
- Proceed to Lesson 06: Cloud Networking for ML

---

**Estimated Time to Complete**: 6 hours (including hands-on exercise)
**Difficulty**: Intermediate
**Next Lesson**: [06-cloud-networking.md](./06-cloud-networking.md)
