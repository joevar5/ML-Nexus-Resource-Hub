# Lesson 04: Building AI Infrastructure with Terraform

## Learning Objectives

By the end of this lesson, you will:

- Provision GPU instances on AWS (p3, g4dn, p4d) and GCP (A100, V100)
- Set up EKS/GKE clusters optimized for ML workloads
- Configure S3/GCS buckets for datasets, models, and artifacts
- Design VPC, subnets, and security groups for ML infrastructure
- Create IAM roles and policies for ML services
- Build a complete end-to-end ML training platform
- Optimize costs using spot instances and auto-scaling
- Implement auto-scaling for inference workloads

## GPU Instance Provisioning

### AWS GPU Instance Types

**GPU Instance Families:**
- **p3**: Previous-gen NVIDIA V100 GPUs (good price/performance)
- **p4d**: NVIDIA A100 GPUs (latest, most powerful)
- **g4dn**: NVIDIA T4 GPUs (inference, light training)
- **g5**: NVIDIA A10G GPUs (mid-range option)

**Comparison:**

| Instance Type | GPUs | GPU Memory | vCPUs | RAM | Network | Cost/hr (approx) | Best For |
|--------------|------|------------|-------|-----|---------|------------------|----------|
| p3.2xlarge | 1x V100 | 16 GB | 8 | 61 GB | 10 Gbps | $3.06 | Small models, learning |
| p3.8xlarge | 4x V100 | 64 GB | 32 | 244 GB | 10 Gbps | $12.24 | Distributed training |
| p3.16xlarge | 8x V100 | 128 GB | 64 | 488 GB | 25 Gbps | $24.48 | Large-scale training |
| p4d.24xlarge | 8x A100 | 320 GB | 96 | 1152 GB | 400 Gbps | $32.77 | Enterprise training |
| g4dn.xlarge | 1x T4 | 16 GB | 4 | 16 GB | 25 Gbps | $0.526 | Inference |
| g4dn.12xlarge | 4x T4 | 64 GB | 48 | 192 GB | 50 Gbps | $3.912 | Batch inference |

### Basic GPU Instance

```hcl
# gpu-instance.tf

provider "aws" {
  region = "us-west-2"  # Good GPU availability
}

# Data source: Find latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

# GPU training instance
resource "aws_instance" "gpu_training" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "p3.2xlarge"

  # Use spot for 70% cost savings (can be interrupted)
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price                      = "1.50"  # Max price per hour
      spot_instance_type            = "persistent"
      instance_interruption_behavior = "stop"  # Stop, don't terminate
    }
  }

  # Storage for datasets and models
  root_block_device {
    volume_size = 500  # 500 GB
    volume_type = "gp3"
    iops        = 3000
    throughput  = 125
  }

  # EBS volume for large datasets
  ebs_block_device {
    device_name = "/dev/sdf"
    volume_size = 1000  # 1 TB
    volume_type = "gp3"
  }

  # Startup script
  user_data = <<-EOF
              #!/bin/bash
              # Install additional ML tools
              pip install --upgrade torch torchvision torchaudio
              pip install transformers datasets accelerate
              pip install wandb mlflow

              # Mount EBS volume
              mkfs -t ext4 /dev/nvme1n1
              mkdir -p /data
              mount /dev/nvme1n1 /data
              echo "/dev/nvme1n1 /data ext4 defaults,nofail 0 2" >> /etc/fstab

              # Setup complete
              echo "GPU instance ready!" > /tmp/setup_complete.txt
              EOF

  # IAM role for S3 access
  iam_instance_profile = aws_iam_instance_profile.ml_training.name

  # Networking
  subnet_id              = aws_subnet.private_a.id
  vpc_security_group_ids = [aws_security_group.gpu_training.id]

  tags = {
    Name        = "GPU-Training-Server"
    Environment = "production"
    Workload    = "ML-Training"
    ManagedBy   = "Terraform"
  }
}

# Elastic IP for consistent access
resource "aws_eip" "gpu_training" {
  instance = aws_instance.gpu_training.id
  domain   = "vpc"

  tags = {
    Name = "GPU-Training-EIP"
  }
}
```

### GPU Cluster with Auto Scaling

```hcl
# gpu-cluster.tf

# Launch template for GPU instances
resource "aws_launch_template" "gpu_training" {
  name_prefix   = "gpu-training-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = "p3.8xlarge"

  iam_instance_profile {
    name = aws_iam_instance_profile.ml_training.name
  }

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [aws_security_group.gpu_training.id]
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 500
      volume_type = "gp3"
      iops        = 3000
      throughput  = 125
    }
  }

  # Use spot instances
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "5.00"
    }
  }

  user_data = base64encode(templatefile("${path.module}/scripts/gpu-init.sh", {
    cluster_name = "ml-training-cluster"
    s3_bucket    = aws_s3_bucket.ml_data.bucket
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name      = "GPU-Training-Node"
      Cluster   = "ml-training-cluster"
      ManagedBy = "Terraform"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "gpu_training" {
  name                = "gpu-training-asg"
  vpc_zone_identifier = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  min_size         = 0  # Scale to zero when not in use
  max_size         = 10
  desired_capacity = 4

  launch_template {
    id      = aws_launch_template.gpu_training.id
    version = "$Latest"
  }

  # Health checks
  health_check_type         = "EC2"
  health_check_grace_period = 300

  # Termination policies (terminate oldest instances first)
  termination_policies = ["OldestInstance"]

  tag {
    key                 = "Name"
    value               = "GPU-Training-ASG-Node"
    propagate_at_launch = true
  }

  tag {
    key                 = "Cluster"
    value               = "ml-training-cluster"
    propagate_at_launch = true
  }
}

# Scheduled scaling (scale up during business hours)
resource "aws_autoscaling_schedule" "scale_up" {
  scheduled_action_name  = "scale-up-business-hours"
  min_size               = 4
  max_size               = 10
  desired_capacity       = 4
  recurrence             = "0 8 * * MON-FRI"  # 8 AM weekdays
  autoscaling_group_name = aws_autoscaling_group.gpu_training.name
}

# Scale down after hours
resource "aws_autoscaling_schedule" "scale_down" {
  scheduled_action_name  = "scale-down-after-hours"
  min_size               = 0
  max_size               = 10
  desired_capacity       = 0
  recurrence             = "0 20 * * *"  # 8 PM daily
  autoscaling_group_name = aws_autoscaling_group.gpu_training.name
}
```

### GCP GPU Instances

```hcl
# gcp-gpu.tf

provider "google" {
  project = "my-ml-project"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# Data source: Latest Deep Learning image
data "google_compute_image" "deep_learning" {
  family  = "pytorch-latest-gpu"
  project = "deeplearning-platform-release"
}

# GPU instance with A100
resource "google_compute_instance" "gpu_training" {
  name         = "gpu-training-a100"
  machine_type = "a2-highgpu-1g"  # 1x A100 GPU

  boot_disk {
    initialize_params {
      image = data.google_compute_image.deep_learning.self_link
      size  = 500  # GB
      type  = "pd-ssd"
    }
  }

  # Additional disk for data
  attached_disk {
    source = google_compute_disk.training_data.id
  }

  # GPU configuration
  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = 1
  }

  # Required for GPU instances
  scheduling {
    on_host_maintenance = "TERMINATE"
    preemptible         = true  # Use preemptible for cost savings
    automatic_restart   = false
  }

  network_interface {
    network    = google_compute_network.ml_vpc.name
    subnetwork = google_compute_subnetwork.ml_subnet.name

    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.ml_training.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets accelerate
    echo "GPU setup complete" > /tmp/ready
  EOF

  tags = ["gpu-training", "ml-workload"]
}

# Persistent disk for training data
resource "google_compute_disk" "training_data" {
  name = "training-data-disk"
  type = "pd-ssd"
  size = 1000  # 1 TB
  zone = "us-central1-a"
}

# Service account for ML workloads
resource "google_service_account" "ml_training" {
  account_id   = "ml-training-sa"
  display_name = "ML Training Service Account"
}

# Grant storage access
resource "google_project_iam_member" "ml_training_storage" {
  project = "my-ml-project"
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_training.email}"
}
```

## EKS Cluster for ML Workloads

### Complete EKS Setup

```hcl
# eks-cluster.tf

# EKS cluster
resource "aws_eks_cluster" "ml_platform" {
  name     = "ml-platform-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = [aws_subnet.private_a.id, aws_subnet.private_b.id, aws_subnet.public_a.id, aws_subnet.public_b.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]  # Restrict in production
    security_group_ids      = [aws_security_group.eks_cluster.id]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_service_policy,
  ]

  tags = {
    Name        = "ML-Platform-Cluster"
    Environment = "production"
  }
}

# EKS cluster IAM role
resource "aws_iam_role" "eks_cluster" {
  name = "ml-platform-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role_policy_attachment" "eks_service_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.eks_cluster.name
}

# KMS key for cluster encryption
resource "aws_kms_key" "eks" {
  description             = "EKS cluster encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}

# CPU node group (for control plane tasks, data processing)
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = "cpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 2
  }

  instance_types = ["c5.2xlarge"]  # CPU-optimized
  capacity_type  = "ON_DEMAND"     # For reliability

  labels = {
    workload = "cpu"
    role     = "general"
  }

  tags = {
    Name = "ML-Platform-CPU-Nodes"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# GPU node group (for training and inference)
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 0  # Can scale to zero when not needed
  }

  instance_types = ["p3.2xlarge"]
  capacity_type  = "SPOT"  # Use spot for cost savings

  labels = {
    workload      = "gpu"
    role          = "ml-training"
    gpu-type      = "v100"
    nvidia.com/gpu = "true"
  }

  taints {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Name = "ML-Platform-GPU-Nodes"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# Inference node group (g4dn instances)
resource "aws_eks_node_group" "inference_nodes" {
  cluster_name    = aws_eks_cluster.ml_platform.name
  node_group_name = "inference-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  scaling_config {
    desired_size = 2
    max_size     = 20
    min_size     = 1
  }

  instance_types = ["g4dn.xlarge"]  # T4 GPUs, good for inference
  capacity_type  = "SPOT"

  labels = {
    workload      = "inference"
    role          = "model-serving"
    gpu-type      = "t4"
    nvidia.com/gpu = "true"
  }

  taints {
    key    = "workload"
    value  = "inference"
    effect = "NO_SCHEDULE"
  }

  tags = {
    Name = "ML-Platform-Inference-Nodes"
    "k8s.io/cluster-autoscaler/enabled" = "true"
    "k8s.io/cluster-autoscaler/${aws_eks_cluster.ml_platform.name}" = "owned"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

# EKS node IAM role
resource "aws_iam_role" "eks_nodes" {
  name = "ml-platform-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes.name
}

# Custom policy for S3 access
resource "aws_iam_role_policy" "eks_s3_access" {
  name = "eks-s3-access"
  role = aws_iam_role.eks_nodes.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*",
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      }
    ]
  })
}

# Cluster autoscaler IAM policy
resource "aws_iam_role_policy" "cluster_autoscaler" {
  name = "cluster-autoscaler"
  role = aws_iam_role.eks_nodes.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeScalingActivities",
          "autoscaling:DescribeTags",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup",
          "ec2:DescribeImages",
          "ec2:GetInstanceTypesFromInstanceRequirements",
          "eks:DescribeNodegroup"
        ]
        Resource = "*"
      }
    ]
  })
}

# OIDC provider for service accounts
data "tls_certificate" "eks" {
  url = aws_eks_cluster.ml_platform.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.ml_platform.identity[0].oidc[0].issuer

  tags = {
    Name = "ML-Platform-OIDC"
  }
}

# Output kubeconfig
output "eks_cluster_endpoint" {
  value = aws_eks_cluster.ml_platform.endpoint
}

output "eks_cluster_name" {
  value = aws_eks_cluster.ml_platform.name
}

output "configure_kubectl" {
  value = "aws eks update-kubeconfig --region us-west-2 --name ${aws_eks_cluster.ml_platform.name}"
}
```

## Storage: S3 Buckets

```hcl
# s3-storage.tf

# Bucket for raw datasets
resource "aws_s3_bucket" "ml_datasets" {
  bucket = "ml-platform-datasets-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "ML Datasets"
    Purpose     = "Raw and processed training data"
    Environment = "production"
  }
}

# Versioning for datasets (track changes)
resource "aws_s3_bucket_versioning" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle rules (move old data to cheaper storage)
resource "aws_s3_bucket_lifecycle_configuration" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    id     = "archive-old-datasets"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "STANDARD_IA"  # Infrequent Access
    }

    transition {
      days          = 180
      storage_class = "GLACIER"  # Long-term archive
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }
  }

  rule {
    id     = "cleanup-temp-data"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 7  # Delete temp data after 7 days
    }
  }
}

# Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_encryption.arn
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket for trained models
resource "aws_s3_bucket" "ml_models" {
  bucket = "ml-platform-models-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "ML Models"
    Purpose     = "Trained model artifacts"
    Environment = "production"
  }
}

# Versioning for models (critical for rollbacks)
resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Encryption for models
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_encryption.arn
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket for experiment artifacts (MLflow, Weights & Biases)
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "ml-platform-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "ML Artifacts"
    Purpose     = "Experiment tracking and metadata"
    Environment = "production"
  }
}

# Lifecycle for artifacts (cleanup old experiments)
resource "aws_s3_bucket_lifecycle_configuration" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  rule {
    id     = "cleanup-old-experiments"
    status = "Enabled"

    filter {
      prefix = "experiments/"
    }

    expiration {
      days = 90  # Keep experiments for 90 days
    }
  }
}

# KMS key for S3 encryption
resource "aws_kms_key" "s3_encryption" {
  description             = "KMS key for ML S3 buckets"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "ML-S3-Encryption-Key"
  }
}

resource "aws_kms_alias" "s3_encryption" {
  name          = "alias/ml-s3-encryption"
  target_key_id = aws_kms_key.s3_encryption.key_id
}

# Bucket policies
resource "aws_s3_bucket_policy" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnforcedTLS"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "AllowEKSNodes"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.eks_nodes.arn
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*"
        ]
      }
    ]
  })
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}
```

## VPC and Networking

```hcl
# networking.tf

# VPC for ML infrastructure
resource "aws_vpc" "ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "ML-Platform-VPC"
    Environment = "production"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Public subnets (for load balancers, NAT gateways)
resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.ml_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-west-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "ML-Platform-Public-A"
    Type = "Public"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.ml_vpc.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "us-west-2b"
  map_public_ip_on_launch = true

  tags = {
    Name = "ML-Platform-Public-B"
    Type = "Public"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                      = "1"
  }
}

# Private subnets (for GPU instances, EKS nodes)
resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "us-west-2a"

  tags = {
    Name = "ML-Platform-Private-A"
    Type = "Private"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"            = "1"
  }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = "10.0.11.0/24"
  availability_zone = "us-west-2b"

  tags = {
    Name = "ML-Platform-Private-B"
    Type = "Private"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"            = "1"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ml_igw" {
  vpc_id = aws_vpc.ml_vpc.id

  tags = {
    Name = "ML-Platform-IGW"
  }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat_a" {
  domain = "vpc"

  tags = {
    Name = "ML-Platform-NAT-EIP-A"
  }

  depends_on = [aws_internet_gateway.ml_igw]
}

resource "aws_eip" "nat_b" {
  domain = "vpc"

  tags = {
    Name = "ML-Platform-NAT-EIP-B"
  }

  depends_on = [aws_internet_gateway.ml_igw]
}

# NAT Gateways (for private subnet internet access)
resource "aws_nat_gateway" "nat_a" {
  allocation_id = aws_eip.nat_a.id
  subnet_id     = aws_subnet.public_a.id

  tags = {
    Name = "ML-Platform-NAT-A"
  }

  depends_on = [aws_internet_gateway.ml_igw]
}

resource "aws_nat_gateway" "nat_b" {
  allocation_id = aws_eip.nat_b.id
  subnet_id     = aws_subnet.public_b.id

  tags = {
    Name = "ML-Platform-NAT-B"
  }

  depends_on = [aws_internet_gateway.ml_igw]
}

# Route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ml_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ml_igw.id
  }

  tags = {
    Name = "ML-Platform-Public-RT"
  }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# Route tables for private subnets
resource "aws_route_table" "private_a" {
  vpc_id = aws_vpc.ml_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_a.id
  }

  tags = {
    Name = "ML-Platform-Private-RT-A"
  }
}

resource "aws_route_table" "private_b" {
  vpc_id = aws_vpc.ml_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_b.id
  }

  tags = {
    Name = "ML-Platform-Private-RT-B"
  }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private_a.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private_b.id
}

# VPC Flow Logs (for network monitoring)
resource "aws_flow_log" "ml_vpc" {
  iam_role_arn    = aws_iam_role.vpc_flow_logs.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_logs.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.ml_vpc.id

  tags = {
    Name = "ML-Platform-VPC-Flow-Logs"
  }
}

resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  name              = "/aws/vpc/ml-platform"
  retention_in_days = 30

  tags = {
    Name = "ML-Platform-VPC-Logs"
  }
}

resource "aws_iam_role" "vpc_flow_logs" {
  name = "ml-platform-vpc-flow-logs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "vpc-flow-logs.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "vpc_flow_logs" {
  name = "vpc-flow-logs-policy"
  role = aws_iam_role.vpc_flow_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}
```

## Security Groups

```hcl
# security-groups.tf

# Security group for GPU training instances
resource "aws_security_group" "gpu_training" {
  name        = "gpu-training-sg"
  description = "Security group for GPU training instances"
  vpc_id      = aws_vpc.ml_vpc.id

  # SSH access (restrict to bastion or VPN in production)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # VPC only
  }

  # Jupyter notebook
  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # TensorBoard
  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # Distributed training communication
  ingress {
    description = "PyTorch DDP"
    from_port   = 29500
    to_port     = 29500
    protocol    = "tcp"
    self        = true  # Only from other instances in this SG
  }

  # All outbound traffic
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "GPU-Training-SG"
  }
}

# Security group for EKS cluster
resource "aws_security_group" "eks_cluster" {
  name        = "eks-cluster-sg"
  description = "Security group for EKS cluster control plane"
  vpc_id      = aws_vpc.ml_vpc.id

  tags = {
    Name = "EKS-Cluster-SG"
  }
}

# Allow nodes to communicate with cluster
resource "aws_security_group_rule" "cluster_inbound_node_https" {
  description              = "Allow nodes to communicate with cluster API"
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.eks_cluster.id
  source_security_group_id = aws_security_group.eks_nodes.id
}

# Security group for EKS nodes
resource "aws_security_group" "eks_nodes" {
  name        = "eks-nodes-sg"
  description = "Security group for EKS worker nodes"
  vpc_id      = aws_vpc.ml_vpc.id

  # Allow nodes to communicate with each other
  ingress {
    description = "Node to node"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # Allow cluster to communicate with nodes
  ingress {
    description              = "Cluster to node"
    from_port                = 1025
    to_port                  = 65535
    protocol                 = "tcp"
    security_group_id        = aws_security_group.eks_cluster.id
    source_security_group_id = aws_security_group.eks_cluster.id
  }

  # All outbound
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "EKS-Nodes-SG"
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
  }
}

# Security group for load balancers
resource "aws_security_group" "alb" {
  name        = "ml-platform-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.ml_vpc.id

  # HTTPS from anywhere
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP from anywhere (redirect to HTTPS)
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "ALB-SG"
  }
}
```

## IAM Roles and Policies

```hcl
# iam.tf

# IAM role for ML training instances
resource "aws_iam_role" "ml_training" {
  name = "ml-training-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "ML-Training-Role"
  }
}

# Custom policy for ML training
resource "aws_iam_role_policy" "ml_training" {
  name = "ml-training-policy"
  role = aws_iam_role.ml_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*",
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*",
          aws_s3_bucket.ml_artifacts.arn,
          "${aws_s3_bucket.ml_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:*:*:secret:ml/*"
      }
    ]
  })
}

# Instance profile
resource "aws_iam_instance_profile" "ml_training" {
  name = "ml-training-profile"
  role = aws_iam_role.ml_training.name
}

# IAM role for model serving / inference
resource "aws_iam_role" "ml_inference" {
  name = "ml-inference-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

# Inference policy (read-only for models)
resource "aws_iam_role_policy" "ml_inference" {
  name = "ml-inference-policy"
  role = aws_iam_role.ml_inference.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ml_inference" {
  name = "ml-inference-profile"
  role = aws_iam_role.ml_inference.name
}
```

## Complete End-to-End Example

Create a complete ML training platform with all components:

**main.tf:**
```hcl
# main.tf - Complete ML Platform

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "ml-platform/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "ML-Platform"
      ManagedBy   = "Terraform"
      Environment = var.environment
      CostCenter  = "AI-Research"
    }
  }
}

# Variables
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "ml-platform-cluster"
}

# Include all the resources defined above:
# - VPC and networking (networking.tf)
# - Security groups (security-groups.tf)
# - S3 buckets (s3-storage.tf)
# - EKS cluster (eks-cluster.tf)
# - GPU instances (gpu-instance.tf)
# - IAM roles (iam.tf)

# Outputs
output "eks_cluster_name" {
  value       = aws_eks_cluster.ml_platform.name
  description = "EKS cluster name"
}

output "eks_cluster_endpoint" {
  value       = aws_eks_cluster.ml_platform.endpoint
  description = "EKS cluster endpoint"
}

output "datasets_bucket" {
  value       = aws_s3_bucket.ml_datasets.bucket
  description = "S3 bucket for datasets"
}

output "models_bucket" {
  value       = aws_s3_bucket.ml_models.bucket
  description = "S3 bucket for models"
}

output "gpu_instance_ip" {
  value       = aws_eip.gpu_training.public_ip
  description = "GPU training instance public IP"
}

output "configure_kubectl" {
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${aws_eks_cluster.ml_platform.name}"
  description = "Command to configure kubectl"
}
```

**Deploy the platform:**
```bash
# Initialize
terraform init

# Plan
terraform plan -out=tfplan

# Review plan carefully
terraform show tfplan

# Apply
terraform apply tfplan

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name ml-platform-cluster

# Verify cluster
kubectl get nodes

# Deploy NVIDIA device plugin for GPU support
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -l nvidia.com/gpu=true

# When done (be careful!)
terraform destroy
```

## Cost Optimization

### Spot Instances

Spot instances can save 70-90% on GPU costs:

```hcl
# Use spot for training (can be interrupted)
resource "aws_instance" "gpu_training_spot" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "p3.8xlarge"

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price                      = "3.00"  # Max $3/hr (vs $12.24 on-demand)
      spot_instance_type            = "persistent"
      instance_interruption_behavior = "stop"  # Stop and resume when capacity returns
    }
  }

  # Checkpointing in user_data to handle interruptions
  user_data = <<-EOF
              #!/bin/bash
              # Training script with checkpointing
              python train.py \
                --checkpoint-dir s3://models/checkpoints/ \
                --checkpoint-frequency 100 \
                --resume-from-checkpoint true
              EOF

  tags = {
    Name = "GPU-Training-Spot"
  }
}
```

### Auto-Scaling for Inference

```hcl
# Auto-scaling based on request rate
resource "aws_appautoscaling_target" "inference" {
  max_capacity       = 20
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.inference.name}/${aws_ecs_service.model_serving.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "inference_scale_up" {
  name               = "inference-scale-up"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.inference.resource_id
  scalable_dimension = aws_appautoscaling_target.inference.scalable_dimension
  service_namespace  = aws_appautoscaling_target.inference.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0  # Target 70% CPU utilization

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```

### Scheduled Scaling

```hcl
# Scale down during off-hours
resource "aws_autoscaling_schedule" "scale_down_night" {
  scheduled_action_name  = "scale-down-night"
  min_size               = 0
  max_size               = 10
  desired_capacity       = 0
  recurrence             = "0 20 * * *"  # 8 PM daily
  autoscaling_group_name = aws_autoscaling_group.gpu_training.name
}

resource "aws_autoscaling_schedule" "scale_up_morning" {
  scheduled_action_name  = "scale-up-morning"
  min_size               = 2
  max_size               = 10
  desired_capacity       = 4
  recurrence             = "0 8 * * MON-FRI"  # 8 AM weekdays
  autoscaling_group_name = aws_autoscaling_group.gpu_training.name
}
```

## Key Takeaways

✅ GPU instances (p3, p4d, g4dn) are provisioned based on workload requirements
✅ EKS clusters provide scalable, containerized ML infrastructure
✅ S3 buckets with versioning and lifecycle policies manage ML data
✅ VPC design isolates ML workloads with public/private subnets
✅ Security groups control network access between components
✅ IAM roles follow least privilege principle
✅ Spot instances save 70-90% on GPU costs
✅ Auto-scaling handles variable inference workloads
✅ Complete infrastructure deployed with single `terraform apply`

## Next Steps

Now that you've built AI infrastructure with Terraform, explore:

- **Lesson 05**: Pulumi for Python-based infrastructure code
- **Lesson 06**: Advanced IaC patterns and reusable modules
- **Lesson 07**: GitOps and CI/CD for infrastructure

---

**Next Lesson**: [05-pulumi-infrastructure-as-software.md](05-pulumi-infrastructure-as-software.md)
