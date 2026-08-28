# Exercise 01: Production ML Infrastructure with Terraform

**Estimated Time**: 32-40 hours

## Business Context

Your AI startup is scaling rapidly and currently manages infrastructure manually through the AWS console. The problems are mounting:

**Current Pain Points**:
- **Environment inconsistency**: Dev, staging, and prod environments drift over time
- **Slow provisioning**: Takes 2-3 days to spin up new ML training environment
- **No version control**: Infrastructure changes aren't tracked or reviewable
- **High costs**: Forgot to shut down $15,000 GPU cluster over weekend
- **Onboarding nightmare**: New engineers take 1 week to understand infrastructure setup
- **Disaster recovery**: No documented way to rebuild if AWS region fails

**Recent Incident**:
A junior engineer accidentally deleted the production database (RDS) while trying to clean up dev resources. Recovery took 8 hours from backups, costing $200K in downtime and lost customer trust.

The CTO has mandated **Infrastructure as Code (IaC)** with these requirements:

1. **All infrastructure in code** (Terraform) for reproducibility
2. **Multi-environment support** (dev, staging, production) from single codebase
3. **Cost controls** with automatic shutdown of expensive resources
4. **GitOps workflow** - all changes via pull requests
5. **Complete ML platform**: GPU training, Kubernetes inference, data storage, monitoring

## Learning Objectives

After completing this exercise, you will be able to:

1. Design and implement production ML infrastructure using Terraform
2. Manage multiple environments (dev/staging/prod) with workspaces and modules
3. Implement remote state management with S3 + DynamoDB for team collaboration
4. Provision GPU instances, Kubernetes clusters, and data storage
5. Apply Terraform best practices (modules, variables, outputs, data sources)
6. Implement cost controls and resource tagging
7. Create secure infrastructure with proper IAM roles and security groups
8. Build a complete GitOps workflow with automated testing

## Prerequisites

- AWS account with admin access (or ability to create IAM users/roles)
- Terraform installed (v1.5+)
- AWS CLI configured
- Git fundamentals
- Module 04 (Kubernetes) recommended
- Basic understanding of cloud resources

## Problem Statement

Build a **Production ML Infrastructure Platform** using Terraform that:

1. **Provisions complete ML environment** with:
   - GPU training instances (EC2 p3.2xlarge)
   - Kubernetes cluster for model inference (EKS)
   - S3 buckets for datasets, models, artifacts
   - RDS PostgreSQL for metadata/features
   - VPC with proper networking and security

2. **Supports multiple environments** (dev, staging, production) with:
   - Shared modules for consistency
   - Environment-specific configurations
   - Cost optimization per environment

3. **Implements team collaboration** with:
   - Remote state in S3
   - State locking with DynamoDB
   - Module versioning

4. **Enables GitOps** with:
   - All changes via Git
   - Automated plan/apply in CI/CD
   - Pull request reviews required

### Success Metrics

- Complete infrastructure deployed in <15 minutes (down from 2-3 days)
- 100% infrastructure reproducibility across environments
- Zero manual AWS console changes (all via Terraform)
- Cost reduction of 40% through automated resource scheduling
- Infrastructure changes reviewed via pull requests (100% coverage)
- Recovery time objective (RTO) <1 hour for complete infrastructure rebuild

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Production ML Infrastructure (Terraform)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    VPC (10.0.0.0/16)                     │  │
│  │                                                           │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐     │  │
│  │  │  Public Subnet      │    │  Private Subnet     │     │  │
│  │  │  (10.0.1.0/24)      │    │  (10.0.10.0/24)     │     │  │
│  │  │                     │    │                     │     │  │
│  │  │  - NAT Gateway      │    │  - EKS Cluster      │     │  │
│  │  │  - Bastion Host     │    │  - ML Inference     │     │  │
│  │  └─────────────────────┘    │  - GPU Training     │     │  │
│  │                              │    (p3.2xlarge)     │     │  │
│  │                              └─────────────────────┘     │  │
│  │                                                           │  │
│  │  ┌─────────────────────┐    ┌─────────────────────┐     │  │
│  │  │  Data Subnet        │    │  Database Subnet    │     │  │
│  │  │  (10.0.20.0/24)     │    │  (10.0.30.0/24)     │     │  │
│  │  │                     │    │                     │     │  │
│  │  │  - S3 Gateway EP    │    │  - RDS PostgreSQL   │     │  │
│  │  └─────────────────────┘    │  - Multi-AZ         │     │  │
│  │                              └─────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Storage & Data Services                  │  │
│  │                                                           │  │
│  │  - S3: ml-datasets-{env}                                 │  │
│  │  - S3: ml-models-{env}                                   │  │
│  │  - S3: ml-artifacts-{env}                                │  │
│  │  - ECR: Docker registry                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Terraform State Management                  │  │
│  │                                                           │  │
│  │  - S3: terraform-state-{account-id}                      │  │
│  │  - DynamoDB: terraform-locks                             │  │
│  │  - Workspaces: dev, staging, production                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Terraform Project Setup (6-8 hours)

Set up Terraform project structure with remote state and modules.

#### 1.1 Project Structure

Create the following directory structure:

```
terraform-ml-infrastructure/
├── README.md
├── .gitignore
├── backend.tf              # Remote state configuration
├── main.tf                 # Root module - calls child modules
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── terraform.tfvars        # Default variable values (not committed)
├── versions.tf             # Terraform and provider versions
│
├── environments/           # Environment-specific configs
│   ├── dev.tfvars
│   ├── staging.tfvars
│   └── production.tfvars
│
├── modules/               # Reusable modules
│   ├── networking/        # VPC, subnets, security groups
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── compute/           # EC2, ASG for GPU training
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── eks/               # Kubernetes cluster
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── storage/           # S3 buckets
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── database/          # RDS PostgreSQL
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
└── scripts/
    ├── plan.sh            # Terraform plan wrapper
    ├── apply.sh           # Terraform apply wrapper
    └── destroy.sh         # Terraform destroy wrapper
```

#### 1.2 Backend Configuration

Create `backend.tf`:

```hcl
# TODO: Configure S3 backend for remote state

terraform {
  backend "s3" {
    # Backend configuration - don't hardcode values here
    # Instead, pass via -backend-config flags or backend.hcl file
    #
    # bucket         = "terraform-state-${account_id}"
    # key            = "ml-infrastructure/terraform.tfstate"
    # region         = "us-west-2"
    # encrypt        = true
    # dynamodb_table = "terraform-locks"

    # Benefits of remote state:
    # 1. Team collaboration (shared state)
    # 2. State locking (prevents concurrent modifications)
    # 3. Encryption at rest
    # 4. Versioning (rollback capability)
  }
}

# Note: Backend initialization
# First time setup:
# 1. Create S3 bucket manually (or use bootstrap script)
# 2. Create DynamoDB table with LockID as partition key
# 3. Initialize backend: terraform init -backend-config=backend.hcl
```

Create `backend.hcl`:

```hcl
# TODO: Backend configuration file (not committed to Git)

bucket         = "terraform-state-123456789012"  # Replace with your account ID
key            = "ml-infrastructure/terraform.tfstate"
region         = "us-west-2"
encrypt        = true
dynamodb_table = "terraform-locks"
```

Create `scripts/bootstrap-backend.sh`:

```bash
#!/bin/bash
# TODO: Bootstrap remote state backend

set -e

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-west-2"
BUCKET_NAME="terraform-state-${ACCOUNT_ID}"
TABLE_NAME="terraform-locks"

echo "Creating Terraform backend resources..."

# Create S3 bucket for state
aws s3api create-bucket \
  --bucket "${BUCKET_NAME}" \
  --region "${REGION}" \
  --create-bucket-configuration LocationConstraint="${REGION}"

# Enable versioning (allows state rollback)
aws s3api put-bucket-versioning \
  --bucket "${BUCKET_NAME}" \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket "${BUCKET_NAME}" \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Block public access
aws s3api put-public-access-block \
  --bucket "${BUCKET_NAME}" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name "${TABLE_NAME}" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "${REGION}"

echo "Backend resources created successfully!"
echo "Bucket: ${BUCKET_NAME}"
echo "DynamoDB Table: ${TABLE_NAME}"
```

#### 1.3 Root Configuration

Create `versions.tf`:

```hcl
# TODO: Define Terraform and provider versions

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "ml-infrastructure"
      ManagedBy   = "Terraform"
      CostCenter  = var.cost_center
    }
  }
}

# Kubernetes provider (configured after EKS creation)
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_ca_cert)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = [
      "eks",
      "get-token",
      "--cluster-name",
      module.eks.cluster_name
    ]
  }
}
```

Create `variables.tf`:

```hcl
# TODO: Define input variables

variable "aws_region" {
  description = "AWS region for infrastructure"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "ml-platform"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "ml-engineering"
}

# Networking variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Compute variables
variable "gpu_instance_type" {
  description = "GPU instance type for training"
  type        = string
  default     = "p3.2xlarge"  # 1x V100 GPU
}

variable "gpu_instance_count" {
  description = "Number of GPU instances"
  type        = number
  default     = 1
}

variable "enable_gpu_autoscaling" {
  description = "Enable autoscaling for GPU instances"
  type        = bool
  default     = false
}

# EKS variables
variable "eks_cluster_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.28"
}

variable "eks_node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    desired_size   = number
    min_size       = number
    max_size       = number
  }))

  default = {
    general = {
      instance_types = ["t3.xlarge"]
      desired_size   = 2
      min_size       = 1
      max_size       = 5
    }
  }
}

# Database variables
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS (GB)"
  type        = number
  default     = 100
}

variable "db_multi_az" {
  description = "Enable Multi-AZ for RDS"
  type        = bool
  default     = false
}

# Tags
variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

Create `environments/production.tfvars`:

```hcl
# TODO: Production environment configuration

environment = "production"
aws_region  = "us-west-2"

# Production uses more robust resources
gpu_instance_type     = "p3.8xlarge"  # 4x V100 GPUs
gpu_instance_count    = 2
enable_gpu_autoscaling = true

eks_node_groups = {
  general = {
    instance_types = ["t3.2xlarge"]
    desired_size   = 3
    min_size       = 3
    max_size       = 10
  }
  gpu_inference = {
    instance_types = ["g4dn.xlarge"]  # T4 GPUs for inference
    desired_size   = 2
    min_size       = 1
    max_size       = 5
  }
}

# Production database - Multi-AZ for high availability
db_instance_class     = "db.r5.xlarge"
db_allocated_storage  = 500
db_multi_az           = true

tags = {
  Compliance = "HIPAA"
  Backup     = "Daily"
}
```

Create `environments/dev.tfvars`:

```hcl
# TODO: Development environment configuration

environment = "dev"
aws_region  = "us-west-2"

# Dev uses smaller, cheaper resources
gpu_instance_type      = "g4dn.xlarge"  # T4 GPU (cheaper than V100)
gpu_instance_count     = 1
enable_gpu_autoscaling = false

eks_node_groups = {
  general = {
    instance_types = ["t3.medium"]
    desired_size   = 1
    min_size       = 1
    max_size       = 2
  }
}

# Dev database - single AZ, smaller instance
db_instance_class    = "db.t3.small"
db_allocated_storage = 20
db_multi_az          = false

tags = {
  AutoShutdown = "True"  # Auto-shutdown after hours to save costs
}
```

### Part 2: Networking Module (6-8 hours)

Create VPC with public/private subnets, NAT gateways, security groups.

#### 2.1 Networking Module

Create `modules/networking/main.tf`:

```hcl
# TODO: Create VPC and networking resources

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-vpc"
    }
  )
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-igw"
    }
  )
}

# Public Subnets (for NAT gateways, bastion hosts)
resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-public-${count.index + 1}"
      Type = "Public"
    }
  )
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? length(var.availability_zones) : 0
  domain = "vpc"

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-nat-eip-${count.index + 1}"
    }
  )

  depends_on = [aws_internet_gateway.main]
}

# NAT Gateways (one per AZ for high availability)
resource "aws_nat_gateway" "main" {
  count = var.enable_nat_gateway ? length(var.availability_zones) : 0

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-nat-${count.index + 1}"
    }
  )

  depends_on = [aws_internet_gateway.main]
}

# Private Subnets (for EKS, GPU instances, RDS)
resource "aws_subnet" "private" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]

  tags = merge(
    var.tags,
    {
      Name                              = "${var.project_name}-${var.environment}-private-${count.index + 1}"
      Type                              = "Private"
      "kubernetes.io/role/internal-elb" = "1"  # For EKS internal load balancers
    }
  )
}

# Route Table for Public Subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-public-rt"
    }
  )
}

# Route Table Associations for Public Subnets
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Route Tables for Private Subnets (one per AZ for NAT gateway redundancy)
resource "aws_route_table" "private" {
  count = length(var.availability_zones)

  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = var.enable_nat_gateway ? aws_nat_gateway.main[count.index].id : null
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-private-rt-${count.index + 1}"
    }
  )
}

# Route Table Associations for Private Subnets
resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Group for GPU Training Instances
resource "aws_security_group" "gpu_training" {
  name_prefix = "${var.project_name}-${var.environment}-gpu-training-"
  description = "Security group for GPU training instances"
  vpc_id      = aws_vpc.main.id

  # SSH access from bastion host only
  ingress {
    description     = "SSH from bastion"
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-gpu-training-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Security Group for Bastion Host
resource "aws_security_group" "bastion" {
  name_prefix = "${var.project_name}-${var.environment}-bastion-"
  description = "Security group for bastion host"
  vpc_id      = aws_vpc.main.id

  # SSH access from office IP or VPN
  ingress {
    description = "SSH from office"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-bastion-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Security Group for RDS
resource "aws_security_group" "database" {
  name_prefix = "${var.project_name}-${var.environment}-database-"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.main.id

  # PostgreSQL access from EKS and GPU instances
  ingress {
    description     = "PostgreSQL from EKS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id, aws_security_group.gpu_training.id]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-database-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Security Group for EKS Cluster
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${var.project_name}-${var.environment}-eks-cluster-"
  description = "Security group for EKS cluster control plane"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-eks-cluster-sg"
    }
  )
}

# Security Group for EKS Worker Nodes
resource "aws_security_group" "eks_nodes" {
  name_prefix = "${var.project_name}-${var.environment}-eks-nodes-"
  description = "Security group for EKS worker nodes"
  vpc_id      = aws_vpc.main.id

  # Allow nodes to communicate with each other
  ingress {
    description = "Node to node"
    from_port   = 0
    to_port     = 65535
    protocol    = "-1"
    self        = true
  }

  # Allow nodes to receive communication from cluster
  ingress {
    description     = "Cluster to node"
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-eks-nodes-sg"
    }
  )

  lifecycle {
    create_before_destroy = true
  }
}

# VPC Endpoint for S3 (allows private S3 access without NAT gateway costs)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  route_table_ids = concat(
    [aws_route_table.public.id],
    aws_route_table.private[*].id
  )

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-s3-endpoint"
    }
  )
}
```

Create `modules/networking/variables.tf`:

```hcl
# TODO: Define networking module variables

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateways for private subnets"
  type        = bool
  default     = true
}

variable "allowed_ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH to bastion"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # TODO: Restrict to office IP in production
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

Create `modules/networking/outputs.tf`:

```hcl
# TODO: Define networking module outputs

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "security_group_gpu_training_id" {
  description = "Security group ID for GPU training instances"
  value       = aws_security_group.gpu_training.id
}

output "security_group_database_id" {
  description = "Security group ID for database"
  value       = aws_security_group.database.id
}

output "security_group_eks_cluster_id" {
  description = "Security group ID for EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "security_group_eks_nodes_id" {
  description = "Security group ID for EKS nodes"
  value       = aws_security_group.eks_nodes.id
}
```

### Part 3: Compute, Storage, and Database Modules (8-10 hours)

Implement GPU compute, S3 storage, and RDS database modules.

**Due to length constraints, I'll provide the structure and key sections:**

Create `modules/compute/main.tf`:

```hcl
# TODO: GPU Training Instances with Auto Scaling

# Launch Template for GPU instances
resource "aws_launch_template" "gpu_training" {
  name_prefix   = "${var.project_name}-${var.environment}-gpu-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu_training.name
  }

  vpc_security_group_ids = var.security_group_ids

  # User data script for ML environment setup
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    environment = var.environment
    s3_bucket   = var.ml_datasets_bucket
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      var.tags,
      {
        Name = "${var.project_name}-${var.environment}-gpu-training"
      }
    )
  }
}

# Auto Scaling Group (optional, for cost optimization)
resource "aws_autoscaling_group" "gpu_training" {
  count = var.enable_autoscaling ? 1 : 0

  name                = "${var.project_name}-${var.environment}-gpu-asg"
  desired_capacity    = var.desired_capacity
  min_size            = var.min_size
  max_size            = var.max_size
  vpc_zone_identifier = var.subnet_ids

  launch_template {
    id      = aws_launch_template.gpu_training.id
    version = "$Latest"
  }

  # Scale down to 0 after hours to save costs (dev/staging only)
  dynamic "tag" {
    for_each = var.enable_scheduled_scaling ? [1] : []
    content {
      key                 = "AutoShutdown"
      value               = "True"
      propagate_at_launch = true
    }
  }
}

# Scheduled Action: Scale down at night (save costs)
resource "aws_autoscaling_schedule" "scale_down" {
  count = var.enable_scheduled_scaling ? 1 : 0

  scheduled_action_name  = "scale-down-evening"
  autoscaling_group_name = aws_autoscaling_group.gpu_training[0].name
  recurrence             = "0 20 * * *"  # 8 PM daily
  desired_capacity       = 0
  min_size               = 0
  max_size               = var.max_size
}

# Scheduled Action: Scale up in morning
resource "aws_autoscaling_schedule" "scale_up" {
  count = var.enable_scheduled_scaling ? 1 : 0

  scheduled_action_name  = "scale-up-morning"
  autoscaling_group_name = aws_autoscaling_group.gpu_training[0].name
  recurrence             = "0 8 * * 1-5"  # 8 AM weekdays
  desired_capacity       = var.desired_capacity
  min_size               = var.min_size
  max_size               = var.max_size
}

# IAM Role for GPU instances
resource "aws_iam_role" "gpu_training" {
  name = "${var.project_name}-${var.environment}-gpu-training"

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

# IAM Policy: S3 access for datasets/models
resource "aws_iam_role_policy" "s3_access" {
  name = "s3-access"
  role = aws_iam_role.gpu_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${var.ml_datasets_bucket}/*",
        "arn:aws:s3:::${var.ml_models_bucket}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "gpu_training" {
  name = "${var.project_name}-${var.environment}-gpu-training"
  role = aws_iam_role.gpu_training.name
}

# Find latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
```

Create `modules/storage/main.tf`:

```hcl
# TODO: S3 Buckets for ML Datasets, Models, Artifacts

# ML Datasets Bucket
resource "aws_s3_bucket" "ml_datasets" {
  bucket = "${var.project_name}-ml-datasets-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = merge(
    var.tags,
    {
      Name    = "ML Datasets"
      Purpose = "Training/validation datasets"
    }
  )
}

# Enable versioning (protect against accidental deletion)
resource "aws_s3_bucket_versioning" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle policy: Move to cheaper storage after 90 days
resource "aws_s3_bucket_lifecycle_configuration" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    id     = "archive-old-datasets"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "INTELLIGENT_TIERING"  # Automatically moves between tiers
    }

    transition {
      days          = 365
      storage_class = "GLACIER"  # Long-term archive
    }
  }
}

# Block public access (security best practice)
resource "aws_s3_bucket_public_access_block" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Similar for ML Models and Artifacts buckets
# ... (repeat for ml_models and ml_artifacts)
```

Create `modules/database/main.tf`:

```hcl
# TODO: RDS PostgreSQL for feature store and metadata

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db-subnet"
  subnet_ids = var.subnet_ids

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-db-subnet-group"
    }
  )
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier     = "${var.project_name}-${var.environment}-postgres"
  engine         = "postgres"
  engine_version = var.engine_version

  instance_class        = var.instance_class
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage

  db_name  = var.database_name
  username = var.master_username
  password = random_password.db_password.result

  vpc_security_group_ids = var.security_group_ids
  db_subnet_group_name   = aws_db_subnet_group.main.name

  multi_az               = var.multi_az
  publicly_accessible    = false
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"  # 3-4 AM UTC
  maintenance_window     = "sun:04:00-sun:05:00"

  # Enable encryption at rest
  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn

  # Enable automated backups
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  # Performance Insights
  performance_insights_enabled    = var.performance_insights_enabled
  performance_insights_retention_period = 7

  skip_final_snapshot       = var.environment != "production"
  final_snapshot_identifier = "${var.project_name}-${var.environment}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-${var.environment}-postgres"
    }
  )
}

# Random password for database (stored in Secrets Manager)
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Store password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "db_password" {
  name = "${var.project_name}-${var.environment}-db-password"

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = var.master_username
    password = random_password.db_password.result
    host     = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = var.database_name
  })
}

# KMS Key for RDS encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = var.tags
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${var.project_name}-${var.environment}-rds"
  target_key_id = aws_kms_key.rds.key_id
}
```

### Part 4: Root Module Integration and GitOps (8-10 hours)

Integrate all modules and implement GitOps workflow.

Create `main.tf`:

```hcl
# TODO: Root module - integrate all child modules

# Local values for common configurations
locals {
  common_tags = merge(
    var.tags,
    {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "Terraform"
    }
  )
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  project_name     = var.project_name
  environment      = var.environment
  vpc_cidr         = var.vpc_cidr
  aws_region       = var.aws_region
  enable_nat_gateway = var.environment == "production" ? true : false  # NAT only in prod

  tags = local.common_tags
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  project_name = var.project_name
  environment  = var.environment

  tags = local.common_tags
}

# GPU Compute Module
module "compute" {
  source = "./modules/compute"

  project_name   = var.project_name
  environment    = var.environment
  instance_type  = var.gpu_instance_type
  desired_capacity = var.gpu_instance_count
  min_size       = 0
  max_size       = var.gpu_instance_count * 2

  subnet_ids         = module.networking.private_subnet_ids
  security_group_ids = [module.networking.security_group_gpu_training_id]

  ml_datasets_bucket = module.storage.ml_datasets_bucket_name
  ml_models_bucket   = module.storage.ml_models_bucket_name

  enable_autoscaling      = var.enable_gpu_autoscaling
  enable_scheduled_scaling = var.environment != "production"  # Auto-shutdown in dev/staging

  tags = local.common_tags
}

# Database Module
module "database" {
  source = "./modules/database"

  project_name    = var.project_name
  environment     = var.environment
  instance_class  = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  multi_az        = var.db_multi_az

  subnet_ids         = module.networking.private_subnet_ids
  security_group_ids = [module.networking.security_group_database_id]

  tags = local.common_tags
}

# EKS Module (TODO: Implement in Part 5 if time allows)
# module "eks" {
#   source = "./modules/eks"
#   ...
# }
```

Create `.github/workflows/terraform.yml`:

```yaml
# TODO: GitOps workflow for infrastructure changes

name: Terraform CI/CD

on:
  pull_request:
    paths:
      - '**.tf'
      - '**.tfvars'
  push:
    branches:
      - main

env:
  TF_VERSION: 1.5.7
  AWS_REGION: us-west-2

jobs:
  validate:
    name: Validate Terraform
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Format Check
        run: terraform fmt -check -recursive

      - name: Terraform Init
        run: terraform init -backend=false

      - name: Terraform Validate
        run: terraform validate

      - name: Run tflint
        uses: terraform-linters/setup-tflint@v3
        with:
          tflint_version: latest

      - name: Init tflint
        run: tflint --init

      - name: Run tflint
        run: tflint -f compact

      - name: Security Scan with tfsec
        uses: aquasecurity/tfsec-action@v1.0.0

  plan:
    name: Terraform Plan
    runs-on: ubuntu-latest
    needs: validate

    strategy:
      matrix:
        environment: [dev, staging, production]

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init

      - name: Terraform Plan
        run: |
          terraform plan \
            -var-file=environments/${{ matrix.environment }}.tfvars \
            -out=${{ matrix.environment }}.tfplan

      - name: Upload Plan
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.environment }}-tfplan
          path: ${{ matrix.environment }}.tfplan

      - name: Comment PR with Plan
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('${{ matrix.environment }}.tfplan', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Terraform Plan: ${{ matrix.environment }}\n\`\`\`\n${plan}\n\`\`\``
            });

  apply:
    name: Terraform Apply
    runs-on: ubuntu-latest
    needs: plan
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    strategy:
      matrix:
        environment: [dev]  # Auto-apply only dev, manual for staging/prod

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: ${{ env.TF_VERSION }}

      - name: Terraform Init
        run: terraform init

      - name: Download Plan
        uses: actions/download-artifact@v3
        with:
          name: ${{ matrix.environment }}-tfplan

      - name: Terraform Apply
        run: terraform apply -auto-approve ${{ matrix.environment }}.tfplan
```

## Acceptance Criteria

### Functional Requirements

- [ ] Complete ML infrastructure deployed via Terraform
- [ ] Multi-environment support (dev, staging, production)
- [ ] Remote state in S3 with locking (DynamoDB)
- [ ] All modules are reusable and parameterized
- [ ] GitOps workflow with PR-based changes
- [ ] Automated cost controls (scheduled scaling for dev/staging)
- [ ] Security best practices (encryption, IAM, security groups)

### Performance Requirements

- [ ] Infrastructure deployment completes in <15 minutes
- [ ] Terraform plan executes in <2 minutes
- [ ] State operations complete in <5 seconds

### Operational Requirements

- [ ] 100% infrastructure reproducibility
- [ ] Zero manual AWS console changes
- [ ] All infrastructure changes via pull requests
- [ ] Cost reduction of 40% through automation
- [ ] RTO <1 hour for complete rebuild

### Code Quality

- [ ] All Terraform code passes `terraform fmt`
- [ ] All code passes `terraform validate`
- [ ] Security scan (tfsec) passes with no high/critical issues
- [ ] Comprehensive documentation in README
- [ ] All variables have descriptions and validation

## Testing Strategy

### Local Testing

```bash
# Format check
terraform fmt -check -recursive

# Validate configuration
terraform init -backend=false
terraform validate

# Security scan
tfsec .

# Plan (dry run)
terraform plan -var-file=environments/dev.tfvars
```

### Integration Testing

```bash
# Deploy to dev environment
./scripts/plan.sh dev
./scripts/apply.sh dev

# Verify resources created
aws ec2 describe-instances --filters "Name=tag:Environment,Values=dev"
aws s3 ls | grep ml-datasets-dev
aws rds describe-db-instances --query "DBInstances[?DBInstanceIdentifier contains(@, 'dev')]"

# Destroy dev environment
./scripts/destroy.sh dev
```

## Deliverables

1. **Terraform Code** (all modules and root configuration)
2. **Documentation** (`README.md`, `ARCHITECTURE.md`)
3. **GitHub Actions Workflow** (`.github/workflows/terraform.yml`)
4. **Helper Scripts** (`scripts/*.sh`)
5. **Environment Configurations** (`environments/*.tfvars`)

## Bonus Challenges

1. **EKS Module** (+8 hours): Complete EKS cluster with node groups
2. **Cost Estimation** (+4 hours): Integrate Infracost for PR cost estimates
3. **Atlantis** (+6 hours): Self-hosted Terraform automation

## Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)

## Submission

```bash
git add .
git commit -m "Complete Exercise 01: Terraform ML Infrastructure"
git push origin exercise-01-terraform-ml-infrastructure
```

---

**Estimated Time Breakdown**:
- Part 1 (Project Setup): 6-8 hours
- Part 2 (Networking Module): 6-8 hours
- Part 3 (Compute/Storage/Database): 8-10 hours
- Part 4 (Integration & GitOps): 8-10 hours
- Part 5 (Testing & Documentation): 4-6 hours
- **Total**: 32-40 hours
