# Lesson 02: AWS for ML Infrastructure

**Duration:** 7 hours
**Objectives:** Master AWS services for building, deploying, and managing ML infrastructure

## Introduction

Amazon Web Services (AWS) is the most widely used cloud platform, offering the broadest set of services for ML infrastructure. This lesson covers the essential AWS services you need to build production ML systems, from compute and storage to managed ML services.

## AWS ML Services Landscape

```
┌──────────────────────────────────────────────────────────────┐
│                   AWS ML Services Stack                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Managed ML Services                                          │
│  ┌────────────┬─────────────┬──────────────┐                │
│  │ SageMaker  │   Bedrock   │  Rekognition │                │
│  │ (Training/ │  (Gen AI)   │  (Computer   │                │
│  │  Serving)  │             │   Vision)    │                │
│  └────────────┴─────────────┴──────────────┘                │
│                       ↓                                       │
│  Compute Layer                                                │
│  ┌────────────┬─────────────┬──────────────┐                │
│  │    EC2     │     ECS     │     EKS      │                │
│  │ (VMs+GPU)  │ (Containers)│ (Kubernetes) │                │
│  └────────────┴─────────────┴──────────────┘                │
│                       ↓                                       │
│  Storage Layer                                                │
│  ┌────────────┬─────────────┬──────────────┐                │
│  │     S3     │     EBS     │     EFS      │                │
│  │  (Object)  │   (Block)   │    (File)    │                │
│  └────────────┴─────────────┴──────────────┘                │
│                       ↓                                       │
│  Foundation Layer                                             │
│  ┌────────────┬─────────────┬──────────────┐                │
│  │    VPC     │     IAM     │  CloudWatch  │                │
│  │ (Network)  │  (Security) │ (Monitoring) │                │
│  └────────────┴─────────────┴──────────────┘                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Part 1: AWS Account Setup and IAM

### Setting Up Your AWS Account

**Step 1: Create AWS Account**
```bash
# Sign up at: https://aws.amazon.com/
# You'll need:
# - Email address
# - Credit card (won't be charged for free tier)
# - Phone number for verification

# Free Tier includes (12 months):
# - 750 hours/month t2.micro EC2 (1 vCPU, 1 GB RAM)
# - 5 GB S3 storage
# - 30 GB EBS storage
# - 15 GB data transfer out
```

**Step 2: Secure Root Account**
```bash
# CRITICAL: Never use root account for daily operations!

# 1. Enable MFA (Multi-Factor Authentication)
#    Account → Security Credentials → MFA → Activate MFA
#    Use: Google Authenticator, Authy, or hardware token

# 2. Create billing alarm
#    CloudWatch → Alarms → Create Alarm → Billing
#    Alert when: EstimatedCharges > $10
```

**Step 3: Install AWS CLI**
```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
# Output: aws-cli/2.x.x Python/3.x.x Linux/x.x.x

# Configure AWS CLI
aws configure
# AWS Access Key ID: [Your key]
# AWS Secret Access Key: [Your secret]
# Default region name: us-east-1
# Default output format: json

# Test configuration
aws sts get-caller-identity
```

### IAM (Identity and Access Management)

**Understand IAM Concepts:**
```
IAM Components:

1. Users: Individual people
2. Groups: Collections of users
3. Roles: Permissions for AWS services
4. Policies: JSON documents defining permissions

Principle of Least Privilege:
Grant only the permissions needed for the task
```

**Create IAM User for ML Development:**
```bash
# Create user
aws iam create-user --user-name ml-engineer

# Attach policy for ML development
aws iam attach-user-policy \
    --user-name ml-engineer \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-user-policy \
    --user-name ml-engineer \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess

aws iam attach-user-policy \
    --user-name ml-engineer \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create access key
aws iam create-access-key --user-name ml-engineer
```

**Example IAM Policy for ML:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "ec2:DescribeInstances",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    }
  ]
}
```

## Part 2: EC2 for ML Compute

### EC2 Instance Types for ML

**General Purpose (t3, m5):**
- Light inference workloads
- Development and testing
- Cost-effective

**Compute Optimized (c5, c6i):**
- CPU-based training
- Batch inference
- High-performance computing

**GPU Instances:**
```
┌──────────────┬─────────────┬──────────┬─────────────┬───────────┐
│ Instance     │ GPU         │ GPU RAM  │ vCPUs       │ Cost/hour │
├──────────────┼─────────────┼──────────┼─────────────┼───────────┤
│ g4dn.xlarge  │ 1x T4       │ 16 GB    │ 4           │ $0.526    │
│ g4dn.12xl    │ 4x T4       │ 64 GB    │ 48          │ $3.912    │
│ p3.2xlarge   │ 1x V100     │ 16 GB    │ 8           │ $3.06     │
│ p3.8xlarge   │ 4x V100     │ 64 GB    │ 32          │ $12.24    │
│ p4d.24xlarge │ 8x A100     │ 320 GB   │ 96          │ $32.77    │
│ g5.xlarge    │ 1x A10G     │ 24 GB    │ 4           │ $1.006    │
│ g5.48xlarge  │ 8x A10G     │ 192 GB   │ 192         │ $16.288   │
└──────────────┴─────────────┴──────────┴─────────────┴───────────┘

Recommendations:
- Development: g4dn.xlarge (T4, affordable)
- Small models: g5.xlarge (A10G, good value)
- Medium models: p3.2xlarge (V100, proven)
- Large models: p3.8xlarge or p4d.24xlarge (A100)
```

### Launching an EC2 Instance for ML

**Using AWS Console:**
```
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Choose AMI: Deep Learning AMI (Ubuntu)
   - Pre-installed: PyTorch, TensorFlow, CUDA
4. Choose Instance Type: g4dn.xlarge
5. Configure:
   - Network: Default VPC
   - Storage: 50 GB gp3
   - Security Group: Allow SSH (22), HTTP (8000)
6. Review and Launch
7. Create/Select Key Pair
8. Launch
```

**Using AWS CLI:**
```bash
# Find Deep Learning AMI
aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI (Ubuntu*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
    --output text

# Create security group
aws ec2 create-security-group \
    --group-name ml-training \
    --description "ML training security group"

# Allow SSH
aws ec2 authorize-security-group-ingress \
    --group-name ml-training \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# Allow custom port for Jupyter/API
aws ec2 authorize-security-group-ingress \
    --group-name ml-training \
    --protocol tcp \
    --port 8000-8888 \
    --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
    --image-id ami-xxxxx \
    --instance-type g4dn.xlarge \
    --key-name my-key-pair \
    --security-groups ml-training \
    --block-device-mappings '[
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }
    ]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-training}]'

# Get instance details
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ml-training" \
    --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress,State.Name]' \
    --output table
```

**Connect to Instance:**
```bash
# SSH into instance
ssh -i my-key-pair.pem ubuntu@<public-ip>

# Verify GPU
nvidia-smi

# Activate conda environment
conda activate pytorch

# Test PyTorch with GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Spot Instances for Cost Savings

**What are Spot Instances?**
- Unused EC2 capacity at up to 90% discount
- Can be interrupted with 2-minute warning
- Perfect for training (with checkpointing)

**Launch Spot Instance:**
```bash
# Create spot instance request
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-xxxxx",
        "InstanceType": "g4dn.xlarge",
        "KeyName": "my-key-pair",
        "SecurityGroups": ["ml-training"]
    }'

# Monitor spot price history
aws ec2 describe-spot-price-history \
    --instance-types g4dn.xlarge \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --product-descriptions "Linux/UNIX" \
    --query 'SpotPriceHistory[*].[Timestamp,SpotPrice]' \
    --output table
```

**Best Practices for Spot Instances:**
```python
# Implement checkpointing in training script
import torch

def train_with_checkpoints(model, train_loader, epochs=10):
    checkpoint_dir = "/data/checkpoints"

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(f"{checkpoint_dir}/latest.pth"):
        checkpoint = torch.load(f"{checkpoint_dir}/latest.pth")
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # Training loop
        for batch in train_loader:
            # ... training code ...
            pass

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, f"{checkpoint_dir}/latest.pth")

        print(f"Checkpoint saved at epoch {epoch + 1}")
```

## Part 3: S3 for Data and Model Storage

### S3 Concepts

```
S3 Hierarchy:

Bucket (globally unique name)
  ├─ folder/ (prefix, not real directory)
  │   ├─ file1.txt (object)
  │   └─ file2.txt (object)
  └─ models/
      ├─ model-v1.pth
      └─ model-v2.pth

Key Features:
- Unlimited storage
- 99.999999999% durability (11 nines)
- Versioning support
- Lifecycle policies
- Access control (IAM, bucket policies)
```

### Creating and Using S3 Buckets

**Create Bucket:**
```bash
# Create bucket
aws s3 mb s3://my-ml-datasets-12345

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-ml-datasets-12345 \
    --versioning-configuration Status=Enabled

# Set lifecycle policy (archive old data)
aws s3api put-bucket-lifecycle-configuration \
    --bucket my-ml-datasets-12345 \
    --lifecycle-configuration file://lifecycle.json
```

**lifecycle.json:**
```json
{
  "Rules": [
    {
      "Id": "Archive old models",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "models/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

**Upload/Download Data:**
```bash
# Upload file
aws s3 cp model.pth s3://my-ml-datasets-12345/models/model-v1.pth

# Upload directory
aws s3 cp datasets/ s3://my-ml-datasets-12345/datasets/ --recursive

# Download file
aws s3 cp s3://my-ml-datasets-12345/models/model-v1.pth ./

# Sync directory (efficient, only changed files)
aws s3 sync s3://my-ml-datasets-12345/datasets/ ./datasets/

# List objects
aws s3 ls s3://my-ml-datasets-12345/models/

# Delete object
aws s3 rm s3://my-ml-datasets-12345/models/old-model.pth
```

**Python SDK (boto3):**
```python
import boto3

# Create S3 client
s3 = boto3.client('s3')

# Upload file
s3.upload_file('model.pth', 'my-ml-datasets-12345', 'models/model-v1.pth')

# Download file
s3.download_file('my-ml-datasets-12345', 'models/model-v1.pth', 'model.pth')

# List objects
response = s3.list_objects_v2(Bucket='my-ml-datasets-12345', Prefix='models/')
for obj in response['Contents']:
    print(obj['Key'], obj['Size'])

# Generate pre-signed URL (temporary access)
url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-ml-datasets-12345', 'Key': 'models/model-v1.pth'},
    ExpiresIn=3600  # 1 hour
)
print(f"Download URL: {url}")
```

### S3 Storage Classes and Costs

```
┌────────────────────┬──────────────┬─────────────┬──────────────┐
│ Storage Class      │ Cost/GB/mo   │ Retrieval   │ Use Case     │
├────────────────────┼──────────────┼─────────────┼──────────────┤
│ Standard           │ $0.023       │ Free        │ Active data  │
│ Intelligent-Tier   │ $0.023-0.015 │ Free        │ Unknown      │
│ Standard-IA        │ $0.0125      │ $0.01/GB    │ Infrequent   │
│ One Zone-IA        │ $0.01        │ $0.01/GB    │ Backups      │
│ Glacier Instant    │ $0.004       │ $0.03/GB    │ Archive      │
│ Glacier Flexible   │ $0.0036      │ Variable    │ Long-term    │
│ Glacier Deep       │ $0.00099     │ $0.02/GB    │ Compliance   │
└────────────────────┴──────────────┴─────────────┴──────────────┘

Recommendations:
- Training data: Standard (frequent access)
- Trained models: Standard-IA (occasional access)
- Old experiments: Glacier (archive)
```

## Part 4: EKS for Kubernetes

### Amazon EKS Overview

**What is EKS?**
- Managed Kubernetes service
- AWS handles control plane
- You manage worker nodes
- Integrates with AWS services

**Why EKS for ML?**
- Scalable model serving
- Multi-model deployment
- Resource isolation
- Auto-scaling
- High availability

### Setting Up EKS Cluster

**Prerequisites:**
```bash
# Install eksctl
curl --silent --location "https://github.com/weksctl-io/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify
eksctl version
kubectl version --client
```

**Create EKS Cluster:**
```bash
# Create cluster with eksctl (simple)
eksctl create cluster \
    --name ml-serving-cluster \
    --region us-east-1 \
    --nodegroup-name ml-nodes \
    --node-type m5.large \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 4 \
    --managed

# This takes ~15 minutes
# Creates:
# - EKS control plane
# - VPC with subnets
# - Security groups
# - IAM roles
# - Node group (EC2 instances)

# Verify cluster
kubectl get nodes
```

**cluster-config.yaml (Advanced):**
```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ml-serving-cluster
  region: us-east-1
  version: "1.28"

nodeGroups:
  - name: cpu-nodes
    instanceType: m5.xlarge
    desiredCapacity: 2
    minSize: 1
    maxSize: 5
    volumeSize: 50
    labels:
      workload: cpu
    tags:
      Environment: production
      Team: ml-infrastructure

  - name: gpu-nodes
    instanceType: g4dn.xlarge
    desiredCapacity: 0
    minSize: 0
    maxSize: 3
    volumeSize: 100
    labels:
      workload: gpu
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule

# Create with config
eksctl create cluster -f cluster-config.yaml
```

### Deploying ML Models on EKS

**Example: Deploy Model Serving API:**
```yaml
# ml-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
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
        image: <your-ecr-repo>/ml-model:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "s3://my-ml-datasets-12345/models/model-v1.pth"
        - name: AWS_REGION
          value: "us-east-1"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
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
  - port: 80
    targetPort: 8000
```

**Deploy:**
```bash
# Apply deployment
kubectl apply -f ml-deployment.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Get service URL
kubectl get service ml-model-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Test endpoint
curl http://<load-balancer-url>/health
```

## Part 5: SageMaker for Managed ML

### SageMaker Components

```
SageMaker Services:

1. Training Jobs: Managed training infrastructure
2. Endpoints: Managed model hosting
3. Feature Store: Centralized feature management
4. Model Registry: Version control for models
5. Pipelines: ML workflow orchestration
6. Studio: IDE for ML development
```

### Training with SageMaker

**Python SDK Example:**
```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Get SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # IAM role for SageMaker

# Define training job
pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir='scripts',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # 1x V100 GPU
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.001
    },
    output_path='s3://my-ml-datasets-12345/models/',
    base_job_name='resnet-training'
)

# Start training
pytorch_estimator.fit({
    'training': 's3://my-ml-datasets-12345/datasets/imagenet/',
    'validation': 's3://my-ml-datasets-12345/datasets/imagenet-val/'
})

# Training job runs on managed infrastructure
# Model artifacts saved to S3 automatically
```

**train.py (SageMaker Training Script):**
```python
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def train(args):
    # Training code here
    model = create_model()
    train_loader = DataLoader(...)

    for epoch in range(args.epochs):
        for batch in train_loader:
            # Training step
            pass

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker-specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    args = parser.parse_args()
    train(args)
```

### Deploying Models with SageMaker

**Deploy trained model:**
```python
# Deploy model to endpoint
predictor = pytorch_estimator.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='resnet-endpoint'
)

# Make predictions
import numpy as np

input_data = np.random.rand(1, 3, 224, 224).astype('float32')
prediction = predictor.predict(input_data)

print(f"Prediction: {prediction}")

# Update endpoint (rolling deployment)
predictor.update_endpoint(
    initial_instance_count=3,
    instance_type='ml.m5.2xlarge'
)

# Delete endpoint (important for cost savings!)
predictor.delete_endpoint()
```

### SageMaker Pricing

```
Training Instances:
- ml.p3.2xlarge (1x V100):  $3.825/hour
- ml.p3.8xlarge (4x V100):  $14.688/hour
- ml.g4dn.xlarge (1x T4):   $0.736/hour

Hosting Instances:
- ml.t3.medium:             $0.065/hour
- ml.m5.xlarge:             $0.269/hour
- ml.c5.2xlarge:            $0.476/hour

Cost Example (ResNet-50 Training):
- Training: ml.p3.2xlarge × 20 hours = $76.50
- Hosting: ml.m5.xlarge × 720 hours/month = $193.68/month
```

## Part 6: Networking with VPC

### VPC Basics

```
VPC Architecture:

AWS Cloud
├─ Region (us-east-1)
│   ├─ VPC (10.0.0.0/16)
│   │   ├─ Public Subnet (10.0.1.0/24) [AZ-1]
│   │   │   ├─ Internet Gateway
│   │   │   ├─ NAT Gateway
│   │   │   └─ Load Balancer
│   │   ├─ Private Subnet (10.0.2.0/24) [AZ-1]
│   │   │   ├─ EC2 Instances
│   │   │   └─ EKS Nodes
│   │   ├─ Public Subnet (10.0.3.0/24) [AZ-2]
│   │   └─ Private Subnet (10.0.4.0/24) [AZ-2]
│   └─ Security Groups & NACLs
```

**Create VPC for ML:**
```bash
# Create VPC
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=ml-vpc}]'

# Create subnets (public and private in each AZ)
aws ec2 create-subnet \
    --vpc-id vpc-xxxxx \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=ml-public-1a}]'

# Create internet gateway
aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=ml-igw}]'

# Attach to VPC
aws ec2 attach-internet-gateway \
    --vpc-id vpc-xxxxx \
    --internet-gateway-id igw-xxxxx
```

## Key Takeaways

1. **IAM Security**: Use MFA, principle of least privilege, never use root
2. **EC2 for Compute**: GPU instances for training, spot for savings
3. **S3 for Storage**: Versioning, lifecycle policies, appropriate storage class
4. **EKS for Orchestration**: Managed Kubernetes for scalable serving
5. **SageMaker**: Fully managed training and deployment
6. **VPC for Networking**: Isolate resources, use private subnets
7. **Cost Optimization**: Spot instances, right-sizing, auto-shutdown

## Practical Exercise

**Deploy a complete ML system on AWS:**

1. Create S3 bucket for datasets and models
2. Launch EC2 GPU instance with Deep Learning AMI
3. Train a simple model, save to S3
4. Deploy model to EKS cluster
5. Expose via Application Load Balancer
6. Set up CloudWatch monitoring
7. Calculate monthly cost

## Self-Check Questions

1. What are the benefits of using Spot instances for training?
2. When would you use SageMaker vs EC2 for training?
3. What S3 storage class would you use for archived experiment data?
4. How do you ensure high availability for model serving on EKS?
5. What IAM permissions are needed for SageMaker?
6. How do you optimize S3 costs for large datasets?
7. What's the difference between public and private subnets?

## Additional Resources

- [AWS ML Services Overview](https://aws.amazon.com/machine-learning/)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [AWS CLI Reference](https://docs.aws.amazon.com/cli/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---

**Next Lesson:** [03-gcp-ml-infrastructure.md](./03-gcp-ml-infrastructure.md) - Google Cloud Platform for ML
