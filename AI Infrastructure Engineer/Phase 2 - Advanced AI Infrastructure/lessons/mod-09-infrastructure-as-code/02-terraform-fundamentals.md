# Lesson 02: Terraform Fundamentals

## Learning Objectives

By the end of this lesson, you will:

- Install and configure Terraform on your system
- Understand HCL (HashiCorp Configuration Language) syntax
- Work with providers, resources, and data sources
- Use variables, outputs, and local values effectively
- Master essential Terraform CLI commands
- Create your first complete Terraform project

## Installing Terraform

### macOS (Homebrew)

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Verify installation
terraform version
# Terraform v1.6.0
```

### Linux (Ubuntu/Debian)

```bash
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list

sudo apt update
sudo apt install terraform

# Verify
terraform -version
```

### Windows (Chocolatey)

```powershell
choco install terraform

# Verify
terraform -version
```

### Using tfenv (Version Manager)

Manage multiple Terraform versions:

```bash
# Install tfenv
git clone https://github.com/tfutils/tfenv.git ~/.tfenv

# Add to PATH
echo 'export PATH="$HOME/.tfenv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install specific version
tfenv install 1.6.0
tfenv use 1.6.0

# Install latest
tfenv install latest
tfenv use latest
```

## Terraform Configuration Structure

### Basic Project Layout

```
ml-infrastructure/
├── main.tf              # Primary resources
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── versions.tf          # Provider version constraints
├── terraform.tfvars     # Variable values (don't commit secrets!)
└── .gitignore           # Ignore .terraform/ and *.tfstate
```

### Minimal Working Example

**main.tf**:
```hcl
# Configure AWS Provider
provider "aws" {
  region = "us-west-2"
}

# Create S3 bucket for ML datasets
resource "aws_s3_bucket" "ml_data" {
  bucket = "my-ml-datasets-unique-name"

  tags = {
    Name        = "ML Datasets"
    Environment = "development"
  }
}
```

## HCL Syntax Basics

### Comments

```hcl
# Single-line comment

/*
Multi-line
comment
*/

// Also works (C-style)
```

### Blocks

```hcl
<BLOCK_TYPE> "<BLOCK_LABEL>" "<BLOCK_LABEL>" {
  # Block body
  <IDENTIFIER> = <EXPRESSION>
}

# Example:
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"
}
```

### Arguments and Attributes

```hcl
resource "aws_instance" "ml_server" {
  ami           = "ami-12345678"  # Argument
  instance_type = "p3.2xlarge"   # Argument

  # After creation, you can reference attributes:
  # aws_instance.ml_server.id
  # aws_instance.ml_server.public_ip
  # aws_instance.ml_server.private_ip
}
```

### Data Types

```hcl
# String
variable "region" {
  type    = string
  default = "us-west-2"
}

# Number
variable "instance_count" {
  type    = number
  default = 3
}

# Boolean
variable "enable_monitoring" {
  type    = bool
  default = true
}

# List
variable "availability_zones" {
  type    = list(string)
  default = ["us-west-2a", "us-west-2b"]
}

# Map
variable "instance_types" {
  type = map(string)
  default = {
    dev  = "t3.micro"
    prod = "p3.8xlarge"
  }
}

# Object
variable "server_config" {
  type = object({
    instance_type = string
    disk_size     = number
    monitoring    = bool
  })
  default = {
    instance_type = "p3.2xlarge"
    disk_size     = 100
    monitoring    = true
  }
}
```

### Expressions

```hcl
# References
var.region
aws_instance.ml_server.public_ip

# String interpolation
"The server IP is ${aws_instance.ml_server.public_ip}"

# String template
"ml-bucket-${var.environment}-${var.region}"

# Arithmetic
instance_count * 2
disk_size + 50

# Comparison
instance_count > 5
var.environment == "production"

# Logical
var.enable_gpu && var.environment == "prod"
```

## Providers

Providers are plugins that interact with cloud APIs.

### Provider Configuration

**versions.tf**:
```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # Any 5.x version
    }
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0, < 6.0"
    }
  }
}
```

**main.tf**:
```hcl
# AWS Provider
provider "aws" {
  region  = "us-west-2"
  profile = "default"  # AWS CLI profile

  default_tags {
    tags = {
      ManagedBy = "Terraform"
      Project   = "ML-Platform"
    }
  }
}

# GCP Provider
provider "google" {
  project = "my-ml-project"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# Multiple provider configurations (aliases)
provider "aws" {
  alias  = "us-east"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu-west"
  region = "eu-west-1"
}

# Use aliased provider
resource "aws_instance" "east_server" {
  provider = aws.us-east
  # ...
}
```

### Provider Authentication

```hcl
# ❌ DON'T: Hardcode credentials
provider "aws" {
  access_key = "AKIAIOSFODNN7EXAMPLE"  # Never do this!
  secret_key = "wJalrXUtnFEMI..."
}

# ✅ DO: Use environment variables
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
provider "aws" {
  region = "us-west-2"
}

# ✅ DO: Use AWS CLI profile
provider "aws" {
  region  = "us-west-2"
  profile = "ml-admin"
}

# ✅ DO: Use instance profile / IAM role (when running on EC2)
provider "aws" {
  region = "us-west-2"
  # Automatically uses instance metadata
}
```

## Resources

Resources are the infrastructure components you want to create.

### Basic Resource

```hcl
resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "p3.2xlarge"

  tags = {
    Name = "ML Training Server"
  }
}
```

### Resource with Dependencies

```hcl
# Security group (created first)
resource "aws_security_group" "ml_sg" {
  name        = "ml-security-group"
  description = "Security group for ML servers"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Instance (created second, depends on security group)
resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "p3.2xlarge"

  # Implicit dependency (Terraform detects automatically)
  vpc_security_group_ids = [aws_security_group.ml_sg.id]

  tags = {
    Name = "ML Training Server"
  }
}
```

### Explicit Dependencies

```hcl
resource "aws_instance" "ml_server" {
  ami           = "ami-12345678"
  instance_type = "p3.2xlarge"

  # Explicit dependency (when implicit doesn't work)
  depends_on = [
    aws_iam_role_policy_attachment.ml_policy
  ]
}
```

### Resource Meta-Arguments

#### count

```hcl
# Create 3 identical GPU instances
resource "aws_instance" "ml_cluster" {
  count = 3

  ami           = "ami-12345678"
  instance_type = "p3.2xlarge"

  tags = {
    Name = "ML Server ${count.index}"  # ML Server 0, 1, 2
  }
}

# Reference specific instance
output "first_server_ip" {
  value = aws_instance.ml_cluster[0].public_ip
}

# Reference all instances
output "all_server_ips" {
  value = aws_instance.ml_cluster[*].public_ip
}
```

#### for_each

```hcl
# Create buckets for different environments
variable "environments" {
  type    = set(string)
  default = ["dev", "staging", "prod"]
}

resource "aws_s3_bucket" "ml_data" {
  for_each = var.environments

  bucket = "ml-datasets-${each.value}"

  tags = {
    Environment = each.value
  }
}

# Reference specific bucket
output "prod_bucket" {
  value = aws_s3_bucket.ml_data["prod"].bucket
}
```

#### lifecycle

```hcl
resource "aws_instance" "ml_server" {
  ami           = "ami-12345678"
  instance_type = "p3.2xlarge"

  lifecycle {
    # Prevent accidental destruction
    prevent_destroy = true

    # Create new before destroying old (zero downtime)
    create_before_destroy = true

    # Ignore changes to specific attributes
    ignore_changes = [
      tags,
      user_data
    ]
  }
}
```

## Data Sources

Data sources fetch information about existing resources.

### Fetching Existing Resources

```hcl
# Get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Use the AMI in a resource
resource "aws_instance" "ml_server" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "p3.2xlarge"
}
```

### Getting Account Information

```hcl
# Get current AWS account ID and region
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Use in configurations
locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

output "account_info" {
  value = "Account: ${local.account_id}, Region: ${local.region}"
}
```

### Fetching Availability Zones

```hcl
# Get all available AZs in current region
data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# Use in resource
resource "aws_subnet" "ml_subnet" {
  count = length(data.aws_availability_zones.available.names)

  vpc_id            = aws_vpc.main.id
  availability_zone = data.aws_availability_zones.available.names[count.index]
  cidr_block        = cidrsubnet("10.0.0.0/16", 8, count.index)
}
```

## Variables

Variables make your configuration reusable and flexible.

### Defining Variables

**variables.tf**:
```hcl
# Basic variable
variable "region" {
  description = "AWS region for ML infrastructure"
  type        = string
  default     = "us-west-2"
}

# Variable with validation
variable "instance_type" {
  description = "EC2 instance type for ML training"
  type        = string
  default     = "p3.2xlarge"

  validation {
    condition     = can(regex("^p3\\.", var.instance_type))
    error_message = "Instance type must be a p3 GPU instance."
  }
}

# Variable without default (must be provided)
variable "project_name" {
  description = "Name of the ML project"
  type        = string
}

# Complex variable
variable "training_cluster" {
  description = "Configuration for ML training cluster"
  type = object({
    instance_count = number
    instance_type  = string
    disk_size_gb   = number
    enable_spot    = bool
  })
  default = {
    instance_count = 4
    instance_type  = "p3.8xlarge"
    disk_size_gb   = 500
    enable_spot    = true
  }
}
```

### Providing Variable Values

**Method 1: terraform.tfvars**
```hcl
# terraform.tfvars
region       = "us-east-1"
project_name = "image-classification"

training_cluster = {
  instance_count = 8
  instance_type  = "p3.16xlarge"
  disk_size_gb   = 1000
  enable_spot    = false
}
```

**Method 2: Command line**
```bash
terraform apply \
  -var="region=us-east-1" \
  -var="project_name=nlp-project"
```

**Method 3: Environment variables**
```bash
export TF_VAR_region="us-east-1"
export TF_VAR_project_name="computer-vision"
terraform apply
```

**Method 4: Variable files**
```bash
# dev.tfvars
terraform apply -var-file="dev.tfvars"

# prod.tfvars
terraform apply -var-file="prod.tfvars"
```

### Using Variables

```hcl
resource "aws_instance" "ml_server" {
  ami           = data.aws_ami.ml_ami.id
  instance_type = var.instance_type

  tags = {
    Name    = "${var.project_name}-training"
    Project = var.project_name
  }
}

resource "aws_instance" "training_cluster" {
  count = var.training_cluster.instance_count

  ami           = data.aws_ami.ml_ami.id
  instance_type = var.training_cluster.instance_type

  root_block_device {
    volume_size = var.training_cluster.disk_size_gb
  }
}
```

## Outputs

Outputs display values after Terraform runs.

### Defining Outputs

**outputs.tf**:
```hcl
# Simple output
output "ml_server_public_ip" {
  description = "Public IP of ML training server"
  value       = aws_instance.ml_server.public_ip
}

# Output with multiple values
output "cluster_ips" {
  description = "Public IPs of all training cluster nodes"
  value       = aws_instance.training_cluster[*].public_ip
}

# Sensitive output (won't show in CLI, but available programmatically)
output "database_password" {
  description = "RDS database password"
  value       = aws_db_instance.ml_db.password
  sensitive   = true
}

# Structured output
output "infrastructure_info" {
  description = "Complete infrastructure information"
  value = {
    server_ip      = aws_instance.ml_server.public_ip
    s3_bucket      = aws_s3_bucket.ml_data.bucket
    cluster_size   = length(aws_instance.training_cluster)
    total_cost_est = "$${aws_instance.ml_server.instance_type == "p3.2xlarge" ? 3.06 : 12.24}/hour"
  }
}
```

### Using Outputs

```bash
# After terraform apply
terraform output
# ml_server_public_ip = "54.123.45.67"
# cluster_ips = ["54.123.45.68", "54.123.45.69", "54.123.45.70"]

# Get specific output
terraform output ml_server_public_ip
# 54.123.45.67

# Get output in JSON
terraform output -json
# {"ml_server_public_ip": {"sensitive": false, "value": "54.123.45.67"}}

# Use in scripts
SERVER_IP=$(terraform output -raw ml_server_public_ip)
ssh ubuntu@$SERVER_IP
```

## Local Values

Local values are internal variables for DRY (Don't Repeat Yourself) code.

```hcl
locals {
  # Common tags for all resources
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    CostCenter  = "AI-Research"
  }

  # Computed values
  cluster_name = "${var.project_name}-${var.environment}-cluster"

  # Conditional logic
  use_gpu = var.environment == "prod" ? true : false

  instance_type = local.use_gpu ? "p3.8xlarge" : "t3.large"

  # Data transformations
  availability_zones = slice(
    data.aws_availability_zones.available.names,
    0,
    min(3, length(data.aws_availability_zones.available.names))
  )
}

# Use locals in resources
resource "aws_instance" "ml_server" {
  ami           = data.aws_ami.ml_ami.id
  instance_type = local.instance_type

  tags = merge(
    local.common_tags,
    {
      Name = local.cluster_name
    }
  )
}
```

## Terraform CLI Commands

### terraform init

Initialize a Terraform working directory.

```bash
# Initialize (downloads providers)
terraform init

# Upgrade provider versions
terraform init -upgrade

# Migrate state backend
terraform init -migrate-state
```

### terraform fmt

Format configuration files to canonical style.

```bash
# Format all .tf files in current directory
terraform fmt

# Format recursively
terraform fmt -recursive

# Check if files are formatted (CI/CD)
terraform fmt -check
```

### terraform validate

Validate configuration syntax.

```bash
# Validate configuration
terraform validate

# Example output:
# Success! The configuration is valid.

# Or if there are errors:
# Error: Unsupported argument
#   on main.tf line 5, in resource "aws_instance" "ml_server":
#   5:   invalid_arg = "value"
```

### terraform plan

Preview changes before applying.

```bash
# Create execution plan
terraform plan

# Save plan to file
terraform plan -out=tfplan

# Show what will be destroyed
terraform plan -destroy

# Target specific resource
terraform plan -target=aws_instance.ml_server

# Use variable file
terraform plan -var-file="prod.tfvars"
```

### terraform apply

Apply changes to infrastructure.

```bash
# Apply changes (will prompt for confirmation)
terraform apply

# Auto-approve (dangerous, for automation only)
terraform apply -auto-approve

# Apply saved plan
terraform apply tfplan

# Apply with variables
terraform apply -var="instance_count=5"

# Target specific resource
terraform apply -target=aws_instance.ml_server
```

### terraform destroy

Destroy all managed infrastructure.

```bash
# Destroy everything (prompts for confirmation)
terraform destroy

# Auto-approve destruction
terraform destroy -auto-approve

# Destroy specific resource
terraform destroy -target=aws_instance.ml_server
```

### terraform show

Show current state or plan.

```bash
# Show current state
terraform show

# Show saved plan
terraform show tfplan

# Output in JSON
terraform show -json
```

### terraform output

Display output values.

```bash
# Show all outputs
terraform output

# Show specific output
terraform output ml_server_ip

# Output in JSON
terraform output -json

# Raw output (no quotes, for scripts)
terraform output -raw ml_server_ip
```

### terraform state

Advanced state management.

```bash
# List all resources in state
terraform state list

# Show specific resource
terraform state show aws_instance.ml_server

# Remove resource from state (doesn't destroy)
terraform state rm aws_instance.ml_server

# Move/rename resource
terraform state mv aws_instance.old aws_instance.new

# Pull remote state
terraform state pull
```

### terraform console

Interactive console for testing expressions.

```bash
terraform console

# Try expressions:
> var.region
"us-west-2"

> aws_instance.ml_server.public_ip
"54.123.45.67"

> length(aws_instance.training_cluster)
3

> [for s in aws_instance.training_cluster : s.id]
["i-0abc123", "i-0def456", "i-0ghi789"]
```

## Your First Terraform Project

Let's create a complete ML infrastructure project.

### Step 1: Project Structure

```bash
mkdir ml-infrastructure
cd ml-infrastructure

# Create files
touch main.tf variables.tf outputs.tf versions.tf terraform.tfvars
touch .gitignore
```

### Step 2: versions.tf

```hcl
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
```

### Step 3: variables.tf

```hcl
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name for tagging"
  type        = string
  default     = "ml-training"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "p3.2xlarge"
}
```

### Step 4: main.tf

```hcl
provider "aws" {
  region = var.region
}

# Data source: Latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }
}

# S3 bucket for datasets
resource "aws_s3_bucket" "datasets" {
  bucket = "${var.project_name}-datasets-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name    = "ML Datasets"
    Project = var.project_name
  }
}

# S3 bucket for models
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name    = "ML Models"
    Project = var.project_name
  }
}

# Security group
resource "aws_security_group" "ml_sg" {
  name        = "${var.project_name}-sg"
  description = "Security group for ML training server"

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Restrict this in production!
  }

  # Jupyter notebook
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# IAM role for EC2
resource "aws_iam_role" "ml_role" {
  name = "${var.project_name}-role"

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

# Attach S3 read/write policy
resource "aws_iam_role_policy_attachment" "s3_policy" {
  role       = aws_iam_role.ml_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Instance profile
resource "aws_iam_instance_profile" "ml_profile" {
  name = "${var.project_name}-profile"
  role = aws_iam_role.ml_role.name
}

# Get current account ID
data "aws_caller_identity" "current" {}

# EC2 instance for ML training
resource "aws_instance" "ml_server" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  iam_instance_profile   = aws_iam_instance_profile.ml_profile.name
  vpc_security_group_ids = [aws_security_group.ml_sg.id]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              echo "Setup complete" > /tmp/setup.log
              EOF

  tags = {
    Name    = "${var.project_name}-server"
    Project = var.project_name
  }
}
```

### Step 5: outputs.tf

```hcl
output "ml_server_public_ip" {
  description = "Public IP of ML training server"
  value       = aws_instance.ml_server.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to server"
  value       = "ssh -i your-key.pem ubuntu@${aws_instance.ml_server.public_ip}"
}

output "datasets_bucket" {
  description = "S3 bucket for datasets"
  value       = aws_s3_bucket.datasets.bucket
}

output "models_bucket" {
  description = "S3 bucket for models"
  value       = aws_s3_bucket.models.bucket
}
```

### Step 6: .gitignore

```
# Local .terraform directories
**/.terraform/*

# .tfstate files
*.tfstate
*.tfstate.*

# Crash log files
crash.log
crash.*.log

# Exclude all .tfvars files with secrets
*.tfvars
*.tfvars.json

# Ignore override files
override.tf
override.tf.json
*_override.tf
*_override.tf.json

# Ignore CLI configuration files
.terraformrc
terraform.rc

# Lock files (some teams commit this)
.terraform.lock.hcl
```

### Step 7: Deploy

```bash
# Initialize
terraform init

# Format code
terraform fmt

# Validate
terraform validate

# Plan
terraform plan

# Apply
terraform apply

# Get outputs
terraform output ml_server_public_ip

# Connect
ssh -i your-key.pem ubuntu@$(terraform output -raw ml_server_public_ip)

# When done, destroy
terraform destroy
```

## Best Practices

✅ Use version control (Git) for all Terraform code
✅ Always run `terraform fmt` before committing
✅ Use `terraform validate` to catch syntax errors
✅ Always review `terraform plan` before `apply`
✅ Use remote state backends (S3, Terraform Cloud)
✅ Never commit `.tfstate` files or `terraform.tfvars` with secrets
✅ Use variables for reusability
✅ Tag all resources consistently
✅ Use data sources to reference existing infrastructure
✅ Break large configurations into modules

## Common Errors and Solutions

### Error: Provider not found

```bash
# Error: Provider "aws" not found
# Solution:
terraform init
```

### Error: Resource already exists

```bash
# Error: resource already exists
# Solution: Import existing resource
terraform import aws_instance.ml_server i-1234567890abcdef0
```

### Error: Invalid syntax

```bash
# Error: Argument or block definition required
# Solution: Run validate
terraform validate
# Fix syntax based on error message
```

## Next Steps

Now that you've mastered Terraform fundamentals, you're ready to:

- **Lesson 03**: Learn state management and team collaboration
- **Lesson 04**: Build complete AI infrastructure with Terraform
- **Lesson 05**: Explore Pulumi for Python-based IaC

## Key Takeaways

✅ Terraform uses HCL (HashiCorp Configuration Language)
✅ Providers connect Terraform to cloud APIs
✅ Resources are infrastructure components to create
✅ Data sources fetch information about existing resources
✅ Variables make configurations reusable
✅ Outputs display important values
✅ Always: init → fmt → validate → plan → apply → destroy
✅ Never commit secrets or state files to version control

---

**Next Lesson**: [03-terraform-state-management.md](03-terraform-state-management.md)
