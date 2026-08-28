# Lesson 06: Advanced IaC Patterns

## Learning Objectives

By the end of this lesson, you will:

- Create reusable Terraform modules for ML infrastructure
- Design module inputs, outputs, and composition patterns
- Implement multi-environment strategies (dev/staging/prod)
- Deploy multi-cloud infrastructure (AWS + GCP)
- Use dynamic configuration with external data sources
- Master conditional resource creation
- Apply advanced for_each and count patterns
- Control resource dependencies and targeting
- Publish and version modules in registries

## Terraform Modules

Modules are reusable packages of Terraform configuration. Think of them as functions for infrastructure.

### Why Modules?

**Without modules (copy-paste hell):**
```
projects/
├── dev/
│   ├── vpc.tf          # 200 lines
│   ├── instances.tf    # 150 lines
│   └── storage.tf      # 100 lines
├── staging/
│   ├── vpc.tf          # 200 lines (duplicated!)
│   ├── instances.tf    # 150 lines (duplicated!)
│   └── storage.tf      # 100 lines (duplicated!)
└── production/
    ├── vpc.tf          # 200 lines (duplicated!)
    ├── instances.tf    # 150 lines (duplicated!)
    └── storage.tf      # 100 lines (duplicated!)

# Total: 1350 lines (900 duplicated!)
# Any change requires updating 3 places
```

**With modules (DRY principle):**
```
projects/
├── modules/
│   └── ml-infrastructure/
│       ├── main.tf       # 450 lines (single source of truth)
│       ├── variables.tf
│       └── outputs.tf
├── dev/
│   └── main.tf           # 20 lines (uses module)
├── staging/
│   └── main.tf           # 20 lines (uses module)
└── production/
    └── main.tf           # 20 lines (uses module)

# Total: 510 lines (no duplication!)
# Changes in one place affect all environments
```

### Creating a Module

**Directory structure:**
```
modules/
└── gpu-cluster/
    ├── main.tf         # Resources
    ├── variables.tf    # Input variables
    ├── outputs.tf      # Output values
    ├── README.md       # Documentation
    └── examples/       # Usage examples
        └── basic/
            └── main.tf
```

**modules/gpu-cluster/variables.tf:**
```hcl
variable "cluster_name" {
  description = "Name of the GPU cluster"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type for GPU nodes"
  type        = string
  default     = "p3.2xlarge"

  validation {
    condition     = can(regex("^(p3|p4d|g4dn|g5)\\.", var.instance_type))
    error_message = "Instance type must be a GPU instance (p3, p4d, g4dn, or g5)."
  }
}

variable "instance_count" {
  description = "Number of GPU instances in cluster"
  type        = number
  default     = 2

  validation {
    condition     = var.instance_count > 0 && var.instance_count <= 100
    error_message = "Instance count must be between 1 and 100."
  }
}

variable "disk_size_gb" {
  description = "Root volume size in GB"
  type        = number
  default     = 500
}

variable "use_spot_instances" {
  description = "Use spot instances for cost savings"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum spot price per hour (only if use_spot_instances = true)"
  type        = string
  default     = "3.00"
}

variable "vpc_id" {
  description = "VPC ID where cluster will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for instance placement"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "s3_buckets" {
  description = "S3 buckets for data access"
  type = object({
    datasets = string
    models   = string
  })
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
```

**modules/gpu-cluster/main.tf:**
```hcl
# Get latest Deep Learning AMI
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
}

# Security group
resource "aws_security_group" "cluster" {
  name_prefix = "${var.cluster_name}-"
  description = "Security group for ${var.cluster_name} GPU cluster"
  vpc_id      = var.vpc_id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Jupyter
  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Inter-instance communication
  ingress {
    description = "Inter-instance"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "${var.cluster_name}-sg"
    },
    var.tags
  )

  lifecycle {
    create_before_destroy = true
  }
}

# IAM role
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-role"

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

  tags = var.tags
}

# IAM policy for S3 access
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.cluster_name}-s3-policy"
  role = aws_iam_role.cluster.id

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
        "arn:aws:s3:::${var.s3_buckets.datasets}",
        "arn:aws:s3:::${var.s3_buckets.datasets}/*",
        "arn:aws:s3:::${var.s3_buckets.models}",
        "arn:aws:s3:::${var.s3_buckets.models}/*"
      ]
    }]
  })
}

# Instance profile
resource "aws_iam_instance_profile" "cluster" {
  name = "${var.cluster_name}-profile"
  role = aws_iam_role.cluster.name
}

# Launch template
resource "aws_launch_template" "cluster" {
  name_prefix   = "${var.cluster_name}-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.cluster.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.cluster.id]
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = var.disk_size_gb
      volume_type = "gp3"
      iops        = 3000
      throughput  = 125
    }
  }

  # Spot instance configuration
  dynamic "instance_market_options" {
    for_each = var.use_spot_instances ? [1] : []

    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price
        spot_instance_type            = "persistent"
        instance_interruption_behavior = "stop"
      }
    }
  }

  user_data = base64encode(templatefile("${path.module}/user-data.sh", {
    cluster_name    = var.cluster_name
    datasets_bucket = var.s3_buckets.datasets
    models_bucket   = var.s3_buckets.models
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      {
        Name    = "${var.cluster_name}-node"
        Cluster = var.cluster_name
      },
      var.tags
    )
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "cluster" {
  name                = "${var.cluster_name}-asg"
  vpc_zone_identifier = var.subnet_ids
  min_size            = 0
  max_size            = var.instance_count * 2
  desired_capacity    = var.instance_count

  launch_template {
    id      = aws_launch_template.cluster.id
    version = "$Latest"
  }

  health_check_type         = "EC2"
  health_check_grace_period = 300

  tag {
    key                 = "Name"
    value               = "${var.cluster_name}-asg-node"
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = var.tags

    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.cluster_name}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors GPU cluster CPU utilization"

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.cluster.name
  }
}
```

**modules/gpu-cluster/outputs.tf:**
```hcl
output "cluster_name" {
  description = "Name of the GPU cluster"
  value       = var.cluster_name
}

output "security_group_id" {
  description = "Security group ID for the cluster"
  value       = aws_security_group.cluster.id
}

output "iam_role_arn" {
  description = "IAM role ARN for cluster instances"
  value       = aws_iam_role.cluster.arn
}

output "autoscaling_group_name" {
  description = "Auto Scaling Group name"
  value       = aws_autoscaling_group.cluster.name
}

output "autoscaling_group_arn" {
  description = "Auto Scaling Group ARN"
  value       = aws_autoscaling_group.cluster.arn
}

output "launch_template_id" {
  description = "Launch template ID"
  value       = aws_launch_template.cluster.id
}
```

**modules/gpu-cluster/user-data.sh:**
```bash
#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting GPU cluster initialization..."

# Update system
yum update -y

# Install additional ML tools
pip install --upgrade torch torchvision torchaudio
pip install transformers datasets accelerate
pip install wandb mlflow tensorboard

# Configure AWS CLI
aws configure set default.region $(ec2-metadata --availability-zone | cut -d' ' -f2 | sed 's/.$//')

# Create data directories
mkdir -p /data/datasets /data/models /data/cache

# Sync S3 data
aws s3 sync s3://${datasets_bucket} /data/datasets --quiet &
aws s3 sync s3://${models_bucket} /data/models --quiet &

# Setup NVIDIA container runtime (if using Docker)
if command -v docker &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
        sudo tee /etc/yum.repos.d/nvidia-docker.repo
    yum install -y nvidia-container-toolkit
    systemctl restart docker
fi

# Create cluster info file
cat > /etc/cluster-info <<EOF
CLUSTER_NAME=${cluster_name}
DATASETS_BUCKET=${datasets_bucket}
MODELS_BUCKET=${models_bucket}
INITIALIZED_AT=$(date)
EOF

echo "GPU cluster initialization complete!"
touch /tmp/cluster-ready
```

**modules/gpu-cluster/README.md:**
```markdown
# GPU Cluster Module

Creates an auto-scaling GPU cluster for ML training workloads.

## Features

- Auto Scaling Group with GPU instances
- Spot instance support for cost optimization
- S3 integration for datasets and models
- CloudWatch monitoring and alarms
- Customizable security groups
- IAM roles with least privilege

## Usage

\`\`\`hcl
module "training_cluster" {
  source = "./modules/gpu-cluster"

  cluster_name       = "ml-training"
  instance_type      = "p3.8xlarge"
  instance_count     = 4
  use_spot_instances = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  s3_buckets = {
    datasets = aws_s3_bucket.datasets.id
    models   = aws_s3_bucket.models.id
  }

  tags = {
    Project     = "ML-Platform"
    Environment = "production"
  }
}
\`\`\`

## Requirements

- Terraform >= 1.0
- AWS Provider >= 5.0

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|----------|
| cluster_name | Name of the GPU cluster | string | - | yes |
| instance_type | EC2 instance type | string | p3.2xlarge | no |
| instance_count | Number of instances | number | 2 | no |
| vpc_id | VPC ID | string | - | yes |
| subnet_ids | Subnet IDs | list(string) | - | yes |

## Outputs

| Name | Description |
|------|-------------|
| cluster_name | GPU cluster name |
| security_group_id | Security group ID |
| autoscaling_group_name | ASG name |
```

### Using Modules

**environment/production/main.tf:**
```hcl
module "training_cluster" {
  source = "../../modules/gpu-cluster"

  cluster_name       = "production-training"
  instance_type      = "p3.8xlarge"
  instance_count     = 10
  use_spot_instances = false  # Production uses on-demand

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  s3_buckets = {
    datasets = aws_s3_bucket.datasets.id
    models   = aws_s3_bucket.models.id
  }

  tags = {
    Project     = "ML-Platform"
    Environment = "production"
    CostCenter  = "AI-Research"
  }
}

module "inference_cluster" {
  source = "../../modules/gpu-cluster"

  cluster_name       = "production-inference"
  instance_type      = "g4dn.xlarge"  # T4 GPUs for inference
  instance_count     = 5
  use_spot_instances = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnet_ids

  s3_buckets = {
    datasets = aws_s3_bucket.datasets.id
    models   = aws_s3_bucket.models.id
  }

  tags = {
    Project     = "ML-Platform"
    Environment = "production"
    CostCenter  = "AI-Research"
  }
}

# Use module outputs
output "training_cluster_asg" {
  value = module.training_cluster.autoscaling_group_name
}

output "inference_cluster_asg" {
  value = module.inference_cluster.autoscaling_group_name
}
```

## Multi-Environment Strategies

### Strategy 1: Workspaces

```bash
# Create workspaces
terraform workspace new dev
terraform workspace new staging
terraform workspace new production

# Switch between workspaces
terraform workspace select dev
terraform apply

terraform workspace select production
terraform apply
```

**main.tf with workspaces:**
```hcl
locals {
  environment = terraform.workspace

  # Environment-specific configuration
  config = {
    dev = {
      instance_type  = "t3.large"
      instance_count = 1
      disk_size      = 50
    }
    staging = {
      instance_type  = "p3.2xlarge"
      instance_count = 2
      disk_size      = 100
    }
    production = {
      instance_type  = "p3.8xlarge"
      instance_count = 10
      disk_size      = 500
    }
  }

  env_config = local.config[local.environment]
}

module "gpu_cluster" {
  source = "./modules/gpu-cluster"

  cluster_name   = "${local.environment}-cluster"
  instance_type  = local.env_config.instance_type
  instance_count = local.env_config.instance_count
  disk_size_gb   = local.env_config.disk_size

  # ...
}
```

### Strategy 2: Separate Directories (Recommended)

```
infrastructure/
├── modules/
│   ├── gpu-cluster/
│   ├── vpc/
│   └── storage/
├── environments/
│   ├── dev/
│   │   ├── backend.tf
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── backend.tf
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   └── production/
│       ├── backend.tf
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars
└── README.md
```

**environments/dev/main.tf:**
```hcl
module "gpu_cluster" {
  source = "../../modules/gpu-cluster"

  cluster_name   = "dev-cluster"
  instance_type  = var.instance_type
  instance_count = var.instance_count

  # ...
}
```

**environments/dev/terraform.tfvars:**
```hcl
instance_type  = "t3.large"
instance_count = 1
disk_size_gb   = 50
```

**environments/production/terraform.tfvars:**
```hcl
instance_type  = "p3.8xlarge"
instance_count = 10
disk_size_gb   = 500
```

### Strategy 3: Terragrunt (Advanced)

Terragrunt is a thin wrapper for Terraform that provides DRY configuration.

**terragrunt.hcl (root):**
```hcl
remote_state {
  backend = "s3"
  config = {
    bucket         = "my-terraform-state"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

**environments/dev/terragrunt.hcl:**
```hcl
include "root" {
  path = find_in_parent_folders()
}

inputs = {
  environment    = "dev"
  instance_type  = "t3.large"
  instance_count = 1
}
```

**environments/production/terragrunt.hcl:**
```hcl
include "root" {
  path = find_in_parent_folders()
}

inputs = {
  environment    = "production"
  instance_type  = "p3.8xlarge"
  instance_count = 10
}
```

## Multi-Cloud Deployments

### AWS + GCP Configuration

```hcl
# providers.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

provider "google" {
  project = "my-ml-project"
  region  = "us-central1"
}

# AWS GPU cluster
module "aws_training_cluster" {
  source = "./modules/aws-gpu-cluster"

  cluster_name   = "aws-training"
  instance_type  = "p3.8xlarge"
  instance_count = 5

  # ...
}

# GCP GPU cluster
module "gcp_training_cluster" {
  source = "./modules/gcp-gpu-cluster"

  cluster_name   = "gcp-training"
  machine_type   = "a2-highgpu-1g"  # A100 GPU
  instance_count = 3

  # ...
}

# Cross-cloud networking (VPN or direct connect)
resource "aws_vpn_gateway" "main" {
  vpc_id = module.aws_vpc.vpc_id
}

resource "google_compute_vpn_gateway" "main" {
  name    = "ml-vpn-gateway"
  network = module.gcp_vpc.network_name
}

# Configure VPN tunnels
resource "aws_vpn_connection" "gcp" {
  vpn_gateway_id      = aws_vpn_gateway.main.id
  customer_gateway_id = aws_customer_gateway.gcp.id
  type                = "ipsec.1"
  static_routes_only  = true
}
```

## Dynamic Configuration

### External Data Sources

```hcl
# Get instance pricing from AWS API
data "aws_ec2_spot_price" "gpu" {
  instance_type     = "p3.8xlarge"
  availability_zone = "us-west-2a"

  filter {
    name   = "product-description"
    values = ["Linux/UNIX"]
  }
}

# Use dynamic pricing
resource "aws_spot_instance_request" "gpu" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = "p3.8xlarge"
  spot_price    = data.aws_ec2_spot_price.gpu.spot_price * 1.1  # 10% buffer

  # ...
}

# HTTP data source (external API)
data "http" "gpu_availability" {
  url = "https://api.example.com/gpu-availability?region=us-west-2"

  request_headers = {
    Accept = "application/json"
  }
}

locals {
  gpu_available = jsondecode(data.http.gpu_availability.body).available
}

# Only create if GPUs are available
resource "aws_instance" "gpu" {
  count = local.gpu_available ? var.instance_count : 0

  # ...
}
```

### Template Files

```hcl
# Read template file
data "template_file" "user_data" {
  template = file("${path.module}/templates/user-data.sh.tpl")

  vars = {
    cluster_name    = var.cluster_name
    datasets_bucket = aws_s3_bucket.datasets.id
    models_bucket   = aws_s3_bucket.models.id
    mlflow_uri      = var.mlflow_tracking_uri
    wandb_api_key   = var.wandb_api_key
  }
}

resource "aws_instance" "gpu" {
  user_data = data.template_file.user_data.rendered
  # ...
}
```

**templates/user-data.sh.tpl:**
```bash
#!/bin/bash
export CLUSTER_NAME="${cluster_name}"
export DATASETS_BUCKET="${datasets_bucket}"
export MODELS_BUCKET="${models_bucket}"
export MLFLOW_TRACKING_URI="${mlflow_uri}"
export WANDB_API_KEY="${wandb_api_key}"

# Setup script
pip install mlflow wandb
mlflow server --host 0.0.0.0 &
```

## Conditional Resources

### Simple Conditionals

```hcl
# Create resource only in production
resource "aws_instance" "monitoring" {
  count = var.environment == "production" ? 1 : 0

  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"

  tags = {
    Name = "monitoring-server"
  }
}

# Create multiple resources conditionally
resource "aws_instance" "gpu_cluster" {
  count = var.enable_gpu ? var.gpu_count : 0

  instance_type = "p3.8xlarge"
  # ...
}
```

### Complex Conditionals with Dynamic Blocks

```hcl
resource "aws_security_group" "ml" {
  name   = "ml-sg"
  vpc_id = var.vpc_id

  # Conditionally add rules
  dynamic "ingress" {
    for_each = var.enable_ssh ? [1] : []

    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.ssh_allowed_cidrs
    }
  }

  dynamic "ingress" {
    for_each = var.enable_jupyter ? [1] : []

    content {
      from_port   = 8888
      to_port     = 8888
      protocol    = "tcp"
      cidr_blocks = var.jupyter_allowed_cidrs
    }
  }

  # Always include egress
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

## Advanced for_each and count Patterns

### for_each with Maps

```hcl
# Create different instance types based on workload
variable "workloads" {
  type = map(object({
    instance_type = string
    count         = number
    disk_size     = number
  }))

  default = {
    training = {
      instance_type = "p3.8xlarge"
      count         = 4
      disk_size     = 500
    }
    inference = {
      instance_type = "g4dn.xlarge"
      count         = 10
      disk_size     = 100
    }
    preprocessing = {
      instance_type = "c5.4xlarge"
      count         = 5
      disk_size     = 200
    }
  }
}

# Create launch template for each workload
resource "aws_launch_template" "workload" {
  for_each = var.workloads

  name_prefix   = "${each.key}-"
  instance_type = each.value.instance_type

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = each.value.disk_size
    }
  }

  tags = {
    Workload = each.key
  }
}

# Create ASG for each workload
resource "aws_autoscaling_group" "workload" {
  for_each = var.workloads

  name             = "${each.key}-asg"
  min_size         = 0
  max_size         = each.value.count * 2
  desired_capacity = each.value.count

  launch_template {
    id      = aws_launch_template.workload[each.key].id
    version = "$Latest"
  }

  # ...
}
```

### Nested for_each

```hcl
# Create buckets in multiple regions
variable "regions" {
  default = ["us-west-2", "us-east-1", "eu-west-1"]
}

variable "bucket_types" {
  default = ["datasets", "models", "artifacts"]
}

locals {
  # Flatten into list of {region, bucket_type} combinations
  region_bucket_combinations = flatten([
    for region in var.regions : [
      for bucket_type in var.bucket_types : {
        region      = region
        bucket_type = bucket_type
        key         = "${region}-${bucket_type}"
      }
    ]
  ])

  # Convert to map for for_each
  region_buckets = {
    for combo in local.region_bucket_combinations :
    combo.key => combo
  }
}

# Create bucket in each region for each type
resource "aws_s3_bucket" "multi_region" {
  for_each = local.region_buckets

  bucket = "ml-${each.value.bucket_type}-${each.value.region}-${data.aws_caller_identity.current.account_id}"

  # Note: Region must be set via provider alias
  provider = aws.${each.value.region}

  tags = {
    Region     = each.value.region
    BucketType = each.value.bucket_type
  }
}
```

### count with Computed Values

```hcl
# Scale instances based on time of day (example)
data "external" "business_hours" {
  program = ["python", "${path.module}/scripts/check_business_hours.py"]
}

locals {
  is_business_hours = data.external.business_hours.result.is_business_hours == "true"
  instance_count    = local.is_business_hours ? 10 : 2
}

resource "aws_instance" "auto_scale" {
  count = local.instance_count

  instance_type = "p3.2xlarge"
  # ...
}
```

## Resource Dependencies

### Implicit Dependencies

Terraform automatically detects dependencies when you reference resources:

```hcl
resource "aws_security_group" "ml" {
  vpc_id = aws_vpc.main.id  # Implicit dependency on vpc
}

resource "aws_instance" "gpu" {
  vpc_security_group_ids = [aws_security_group.ml.id]  # Implicit dependency on SG
}

# Terraform will create in order: VPC → Security Group → Instance
```

### Explicit Dependencies

Use `depends_on` when Terraform can't detect dependencies:

```hcl
resource "aws_iam_role_policy_attachment" "ml_policy" {
  role       = aws_iam_role.ml.name
  policy_arn = aws_iam_policy.ml.arn
}

resource "aws_instance" "gpu" {
  iam_instance_profile = aws_iam_instance_profile.ml.name

  # Explicit dependency: ensure policy is attached before creating instance
  depends_on = [
    aws_iam_role_policy_attachment.ml_policy
  ]
}
```

### Resource Targeting

Apply or destroy specific resources:

```bash
# Apply only specific resource
terraform apply -target=aws_instance.gpu_training

# Apply module and its dependencies
terraform apply -target=module.gpu_cluster

# Destroy specific resource
terraform destroy -target=aws_instance.temp_server

# Multiple targets
terraform apply \
  -target=module.vpc \
  -target=module.gpu_cluster
```

## Module Versioning and Registry

### Git-based Versioning

```hcl
# Use specific git tag
module "gpu_cluster" {
  source = "git::https://github.com/myorg/terraform-modules.git//gpu-cluster?ref=v1.2.0"

  # ...
}

# Use specific branch
module "gpu_cluster" {
  source = "git::https://github.com/myorg/terraform-modules.git//gpu-cluster?ref=main"

  # ...
}

# Use specific commit
module "gpu_cluster" {
  source = "git::https://github.com/myorg/terraform-modules.git//gpu-cluster?ref=abc123def"

  # ...
}
```

### Terraform Registry

```hcl
# Public registry
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  # ...
}

# Private registry
module "gpu_cluster" {
  source  = "app.terraform.io/myorg/gpu-cluster/aws"
  version = "~> 2.1"

  # ...
}
```

### Semantic Versioning

```hcl
# Specific version
version = "1.2.3"

# Any version in 1.x series
version = "~> 1.0"

# Any version >= 1.2.3 but < 2.0.0
version = "~> 1.2"

# Any version >= 1.2.3
version = ">= 1.2.3"

# Version range
version = ">= 1.0.0, < 2.0.0"
```

## Complete Advanced Example

**Project structure:**
```
ml-infrastructure/
├── modules/
│   ├── vpc/
│   ├── gpu-cluster/
│   ├── storage/
│   └── monitoring/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── production/
├── shared/
│   ├── data.tf
│   └── locals.tf
└── README.md
```

**environments/production/main.tf:**
```hcl
terraform {
  required_version = ">= 1.0"

  backend "s3" {
    bucket         = "ml-platform-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = var.common_tags
  }
}

# VPC Module
module "vpc" {
  source = "../../modules/vpc"

  name               = "${var.project_name}-vpc"
  cidr               = "10.0.0.0/16"
  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]

  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false

  tags = var.common_tags
}

# Storage Module
module "storage" {
  source = "../../modules/storage"

  project_name = var.project_name
  environment  = var.environment

  bucket_types = ["datasets", "models", "artifacts", "logs"]

  lifecycle_rules = {
    datasets = {
      transition_days    = 90
      transition_class   = "STANDARD_IA"
      expiration_enabled = false
    }
    models = {
      transition_days    = 180
      transition_class   = "GLACIER"
      expiration_enabled = false
    }
    artifacts = {
      transition_days    = 30
      transition_class   = "STANDARD_IA"
      expiration_enabled = true
      expiration_days    = 90
    }
  }

  tags = var.common_tags
}

# Training Cluster Module
module "training_cluster" {
  source = "../../modules/gpu-cluster"

  cluster_name       = "${var.project_name}-training"
  instance_type      = "p3.8xlarge"
  instance_count     = 10
  disk_size_gb       = 500
  use_spot_instances = false

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids

  s3_buckets = {
    datasets = module.storage.bucket_names["datasets"]
    models   = module.storage.bucket_names["models"]
  }

  allowed_cidr_blocks = concat(
    module.vpc.private_subnets_cidr_blocks,
    [var.office_cidr]
  )

  tags = merge(
    var.common_tags,
    {
      Workload = "Training"
    }
  )
}

# Inference Cluster Module
module "inference_cluster" {
  source = "../../modules/gpu-cluster"

  cluster_name       = "${var.project_name}-inference"
  instance_type      = "g4dn.xlarge"
  instance_count     = 20
  disk_size_gb       = 100
  use_spot_instances = true
  spot_max_price     = "0.75"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnet_ids

  s3_buckets = {
    datasets = module.storage.bucket_names["datasets"]
    models   = module.storage.bucket_names["models"]
  }

  allowed_cidr_blocks = ["0.0.0.0/0"]

  tags = merge(
    var.common_tags,
    {
      Workload = "Inference"
    }
  )
}

# Monitoring Module
module "monitoring" {
  source = "../../modules/monitoring"

  project_name = var.project_name

  alarm_targets = {
    training_asg  = module.training_cluster.autoscaling_group_name
    inference_asg = module.inference_cluster.autoscaling_group_name
  }

  alarm_email = var.alarm_email

  tags = var.common_tags
}

# Outputs
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "storage_buckets" {
  value = module.storage.bucket_names
}

output "training_cluster_asg" {
  value = module.training_cluster.autoscaling_group_name
}

output "inference_cluster_asg" {
  value = module.inference_cluster.autoscaling_group_name
}
```

## Best Practices

✅ **Use modules** for reusable infrastructure components
✅ **Version your modules** with git tags or registry versions
✅ **Document modules** with README and examples
✅ **Separate environments** with directories (not just workspaces)
✅ **Use for_each over count** for more flexibility
✅ **Leverage dynamic blocks** for conditional configuration
✅ **Explicit dependencies** when Terraform can't detect them
✅ **Resource targeting** carefully (can cause drift)
✅ **Test modules** in dev before production
✅ **Semantic versioning** for module releases

## Key Takeaways

✅ Modules promote code reuse and DRY principles
✅ Multi-environment strategies: workspaces, directories, or Terragrunt
✅ Multi-cloud requires careful provider configuration
✅ Dynamic configuration with data sources and templates
✅ Conditional resources with count, for_each, and dynamic blocks
✅ Advanced patterns enable complex infrastructure scenarios
✅ Module versioning ensures stability and reproducibility
✅ Dependencies control resource creation order

## Next Steps

Now that you've mastered advanced IaC patterns, explore:

- **Lesson 07**: GitOps and CI/CD workflows for infrastructure
- **Lesson 08**: Security best practices and compliance

---

**Next Lesson**: [07-gitops-iac-cicd.md](07-gitops-iac-cicd.md)
