# Module 09: Infrastructure as Code - Hands-On Exercises

Practical exercises to reinforce Infrastructure as Code concepts and build real-world ML infrastructure skills.

---

## Exercise Overview

| Exercise | Topic | Difficulty | Time | Prerequisites |
|----------|-------|------------|------|---------------|
| 01 | Deploy EC2 Instance | Beginner | 1-2 hours | AWS account, Terraform installed |
| 02 | S3 Buckets for ML Data | Beginner | 1 hour | Exercise 01 |
| 03 | GPU Cluster with Terraform | Intermediate | 2-3 hours | Exercise 01-02 |
| 04 | Reusable Terraform Modules | Intermediate | 2-3 hours | Exercise 01-03 |
| 05 | Remote State with S3 Backend | Intermediate | 1-2 hours | Exercise 01-04 |
| 06 | ML Infrastructure with Pulumi | Intermediate | 2-3 hours | Python, Pulumi installed |
| 07 | GitOps Workflow with GitHub Actions | Advanced | 3-4 hours | GitHub account, Git knowledge |
| 08 | Secrets Management | Advanced | 2-3 hours | AWS Secrets Manager knowledge |

**Total estimated time**: 15-21 hours

---

## Prerequisites

### Required Accounts

- **AWS Account** - Free tier eligible (sign up at https://aws.amazon.com/free/)
- **GitHub Account** - For Exercise 07 (free at https://github.com/)

### Required Tools

```bash
# Terraform
brew install terraform  # macOS
# or download from https://www.terraform.io/downloads

# AWS CLI
brew install awscli  # macOS
# or download from https://aws.amazon.com/cli/

# Git
brew install git  # macOS

# (Optional) Pulumi for Exercise 06
brew install pulumi  # macOS
# or download from https://www.pulumi.com/docs/install/

# (Optional) Pre-commit for Exercise 08
pip install pre-commit
```

### AWS Configuration

```bash
# Configure AWS credentials
aws configure

# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-west-2)
# - Output format (json)

# Verify
aws sts get-caller-identity
```

---

## Exercise 01: Deploy Your First EC2 Instance

**Objective**: Create a basic EC2 instance using Terraform to understand the core workflow.

### Learning Goals

- Understand Terraform project structure
- Write basic HCL configuration
- Use terraform init, plan, apply, destroy
- Reference data sources

### Steps

1. **Create project directory**
   ```bash
   mkdir exercise-01-ec2
   cd exercise-01-ec2
   ```

2. **Create versions.tf**
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

3. **Create main.tf**
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   # Data source: Latest Amazon Linux 2 AMI
   data "aws_ami" "amazon_linux" {
     most_recent = true
     owners      = ["amazon"]

     filter {
       name   = "name"
       values = ["amzn2-ami-hvm-*-x86_64-gp2"]
     }
   }

   # Security group
   resource "aws_security_group" "allow_ssh" {
     name        = "allow-ssh"
     description = "Allow SSH inbound traffic"

     ingress {
       description = "SSH from anywhere"
       from_port   = 22
       to_port     = 22
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]  # TODO: Restrict this in production!
     }

     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }

     tags = {
       Name = "allow-ssh"
     }
   }

   # EC2 instance
   resource "aws_instance" "web" {
     ami           = data.aws_ami.amazon_linux.id
     instance_type = "t3.micro"  # Free tier eligible

     vpc_security_group_ids = [aws_security_group.allow_ssh.id]

     tags = {
       Name = "Exercise-01-Instance"
     }
   }
   ```

4. **Create outputs.tf**
   ```hcl
   output "instance_id" {
     description = "ID of the EC2 instance"
     value       = aws_instance.web.id
   }

   output "instance_public_ip" {
     description = "Public IP address of the EC2 instance"
     value       = aws_instance.web.public_ip
   }

   output "ssh_command" {
     description = "SSH command to connect to instance"
     value       = "ssh ec2-user@${aws_instance.web.public_ip}"
   }
   ```

5. **Create .gitignore**
   ```
   .terraform/
   *.tfstate
   *.tfstate.backup
   .terraform.lock.hcl
   ```

6. **Deploy**
   ```bash
   # Initialize Terraform
   terraform init

   # Format code
   terraform fmt

   # Validate syntax
   terraform validate

   # Preview changes
   terraform plan

   # Apply changes
   terraform apply
   # Type 'yes' when prompted

   # View outputs
   terraform output

   # SSH to instance (if you have a key pair configured)
   ssh ec2-user@$(terraform output -raw instance_public_ip)

   # Destroy when done
   terraform destroy
   # Type 'yes' to confirm
   ```

### Challenge Tasks

- [ ] Add a key pair to enable SSH access
- [ ] Change instance type to t3.small and re-apply
- [ ] Add user data to install nginx on boot
- [ ] Add multiple security group rules (HTTP, HTTPS)
- [ ] Tag resources with your name and date

### Expected Results

✅ EC2 instance created successfully
✅ Security group attached
✅ Public IP address outputted
✅ Instance can be SSH'd into (if key configured)
✅ Terraform state file created locally

---

## Exercise 02: S3 Buckets for ML Data

**Objective**: Create S3 buckets with proper configuration for ML datasets and models.

### Learning Goals

- Configure S3 buckets with lifecycle policies
- Enable versioning for model storage
- Set up bucket encryption
- Use variables for configuration

### Steps

1. **Create project directory**
   ```bash
   mkdir exercise-02-s3
   cd exercise-02-s3
   ```

2. **Create variables.tf**
   ```hcl
   variable "project_name" {
     description = "Name of the ML project"
     type        = string
     default     = "ml-project"
   }

   variable "environment" {
     description = "Environment (dev, staging, prod)"
     type        = string
     default     = "dev"
   }
   ```

3. **Create main.tf**
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   data "aws_caller_identity" "current" {}

   locals {
     account_id = data.aws_caller_identity.current.account_id
   }

   # Bucket for raw datasets
   resource "aws_s3_bucket" "datasets" {
     bucket = "${var.project_name}-datasets-${local.account_id}"

     tags = {
       Name        = "ML Datasets"
       Environment = var.environment
       Purpose     = "raw-data"
     }
   }

   # Enable versioning for datasets
   resource "aws_s3_bucket_versioning" "datasets" {
     bucket = aws_s3_bucket.datasets.id

     versioning_configuration {
       status = "Enabled"
     }
   }

   # Lifecycle policy for datasets
   resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
     bucket = aws_s3_bucket.datasets.id

     rule {
       id     = "archive-old-data"
       status = "Enabled"

       transition {
         days          = 30
         storage_class = "STANDARD_IA"  # Infrequent Access
       }

       transition {
         days          = 90
         storage_class = "GLACIER"
       }
     }
   }

   # Bucket for trained models
   resource "aws_s3_bucket" "models" {
     bucket = "${var.project_name}-models-${local.account_id}"

     tags = {
       Name        = "ML Models"
       Environment = var.environment
       Purpose     = "model-artifacts"
     }
   }

   # Enable versioning for models (critical!)
   resource "aws_s3_bucket_versioning" "models" {
     bucket = aws_s3_bucket.models.id

     versioning_configuration {
       status = "Enabled"
     }
   }

   # Server-side encryption
   resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
     bucket = aws_s3_bucket.models.id

     rule {
       apply_server_side_encryption_by_default {
         sse_algorithm = "AES256"
       }
     }
   }

   # Block public access
   resource "aws_s3_bucket_public_access_block" "models" {
     bucket = aws_s3_bucket.models.id

     block_public_acls       = true
     block_public_policy     = true
     ignore_public_acls      = true
     restrict_public_buckets = true
   }
   ```

4. **Create outputs.tf**
   ```hcl
   output "datasets_bucket_name" {
     description = "Name of the datasets S3 bucket"
     value       = aws_s3_bucket.datasets.bucket
   }

   output "models_bucket_name" {
     description = "Name of the models S3 bucket"
     value       = aws_s3_bucket.models.bucket
   }

   output "datasets_bucket_arn" {
     value = aws_s3_bucket.datasets.arn
   }

   output "models_bucket_arn" {
     value = aws_s3_bucket.models.arn
   }
   ```

5. **Deploy**
   ```bash
   terraform init
   terraform plan
   terraform apply

   # Test bucket access
   echo "test data" > test.txt
   aws s3 cp test.txt s3://$(terraform output -raw datasets_bucket_name)/

   # List objects
   aws s3 ls s3://$(terraform output -raw datasets_bucket_name)/

   # Cleanup
   aws s3 rm s3://$(terraform output -raw datasets_bucket_name)/test.txt
   terraform destroy
   ```

### Challenge Tasks

- [ ] Add a third bucket for processed/features data
- [ ] Implement S3 bucket logging
- [ ] Add CORS configuration for web access
- [ ] Create IAM policy document for bucket access
- [ ] Add lifecycle rule to delete old model versions after 180 days

### Expected Results

✅ Two S3 buckets created (datasets, models)
✅ Versioning enabled on both buckets
✅ Encryption enabled on models bucket
✅ Lifecycle policies applied
✅ Public access blocked

---

## Exercise 03: GPU Cluster with Terraform

**Objective**: Provision a small GPU cluster for ML training with proper networking and IAM.

### Learning Goals

- Configure VPC and subnets
- Create GPU instances (p3 family)
- Set up IAM roles and instance profiles
- Use count for multiple instances

### Steps

1. **Create project directory**
   ```bash
   mkdir exercise-03-gpu-cluster
   cd exercise-03-gpu-cluster
   ```

2. **Create variables.tf**
   ```hcl
   variable "region" {
     type    = string
     default = "us-west-2"  # p3 instances available
   }

   variable "cluster_size" {
     description = "Number of GPU instances"
     type        = number
     default     = 2
   }

   variable "instance_type" {
     description = "GPU instance type"
     type        = string
     default     = "p3.2xlarge"  # 1x V100 GPU
   }
   ```

3. **Create main.tf** (partial - complete in exercise)
   ```hcl
   provider "aws" {
     region = var.region
   }

   # Get latest Deep Learning AMI
   data "aws_ami" "deep_learning" {
     most_recent = true
     owners      = ["amazon"]

     filter {
       name   = "name"
       values = ["Deep Learning AMI GPU PyTorch*"]
     }
   }

   # VPC
   resource "aws_vpc" "ml_vpc" {
     cidr_block           = "10.0.0.0/16"
     enable_dns_hostnames = true

     tags = {
       Name = "ml-training-vpc"
     }
   }

   # TODO: Add subnet, internet gateway, route table
   # TODO: Add security group
   # TODO: Add IAM role for S3 access
   # TODO: Add GPU instances with count

   # GPU instances
   resource "aws_instance" "gpu_cluster" {
     count = var.cluster_size

     ami           = data.aws_ami.deep_learning.id
     instance_type = var.instance_type

     # TODO: Add subnet_id, security_groups, iam_instance_profile

     tags = {
       Name = "GPU-Cluster-${count.index + 1}"
     }
   }
   ```

4. **Complete the implementation**
   - Add subnet creation
   - Add internet gateway and routing
   - Create security group (SSH, inter-cluster communication)
   - Create IAM role with S3 access
   - Attach IAM instance profile to instances

5. **Deploy and test**
   ```bash
   terraform init
   terraform plan
   terraform apply

   # SSH to first instance
   ssh -i your-key.pem ubuntu@$(terraform output -raw cluster_ips | jq -r '.[0]')

   # Test GPU
   nvidia-smi

   # Destroy (expensive resources!)
   terraform destroy
   ```

### Challenge Tasks

- [ ] Use spot instances to reduce cost
- [ ] Add EFS for shared storage between cluster nodes
- [ ] Configure instances to auto-shutdown after 2 hours
- [ ] Add CloudWatch alarms for GPU utilization
- [ ] Create a bastion host for secure SSH access

### Expected Results

✅ VPC and networking created
✅ 2 GPU instances running
✅ IAM roles attached
✅ Instances can communicate with each other
✅ SSH access configured

**⚠️ Cost Warning**: p3.2xlarge costs ~$3.06/hour. Remember to destroy resources!

---

## Exercise 04: Reusable Terraform Modules

**Objective**: Create reusable modules for common ML infrastructure patterns.

### Learning Goals

- Understand module structure
- Create module inputs and outputs
- Compose modules in root configuration
- Version and document modules

### Steps

1. **Create project structure**
   ```bash
   mkdir -p exercise-04-modules/{modules/{s3-ml-bucket,gpu-instance},environments/{dev,prod}}
   cd exercise-04-modules
   ```

2. **Create S3 module** (`modules/s3-ml-bucket/main.tf`)
   ```hcl
   variable "bucket_name" {
     type = string
   }

   variable "versioning_enabled" {
     type    = bool
     default = true
   }

   variable "lifecycle_days_ia" {
     type    = number
     default = 30
   }

   variable "tags" {
     type    = map(string)
     default = {}
   }

   resource "aws_s3_bucket" "this" {
     bucket = var.bucket_name
     tags   = var.tags
   }

   resource "aws_s3_bucket_versioning" "this" {
     bucket = aws_s3_bucket.this.id

     versioning_configuration {
       status = var.versioning_enabled ? "Enabled" : "Suspended"
     }
   }

   # TODO: Add lifecycle, encryption, public access block

   output "bucket_name" {
     value = aws_s3_bucket.this.bucket
   }

   output "bucket_arn" {
     value = aws_s3_bucket.this.arn
   }
   ```

3. **Create GPU instance module** (`modules/gpu-instance/main.tf`)
   ```hcl
   variable "instance_type" {
     type = string
   }

   variable "instance_name" {
     type = string
   }

   # TODO: Add more variables

   data "aws_ami" "deep_learning" {
     most_recent = true
     owners      = ["amazon"]
     # ...
   }

   resource "aws_instance" "gpu" {
     ami           = data.aws_ami.deep_learning.id
     instance_type = var.instance_type
     # ...
   }

   output "instance_id" {
     value = aws_instance.gpu.id
   }

   output "public_ip" {
     value = aws_instance.gpu.public_ip
   }
   ```

4. **Use modules in environments**

   **environments/dev/main.tf**:
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   module "datasets_bucket" {
     source = "../../modules/s3-ml-bucket"

     bucket_name         = "ml-datasets-dev-${data.aws_caller_identity.current.account_id}"
     versioning_enabled  = true
     lifecycle_days_ia   = 30

     tags = {
       Environment = "dev"
       Purpose     = "datasets"
     }
   }

   module "gpu_server" {
     source = "../../modules/gpu-instance"

     instance_type = "t3.micro"  # Cheap for dev
     instance_name = "dev-ml-server"
   }

   data "aws_caller_identity" "current" {}
   ```

   **environments/prod/main.tf**:
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   module "datasets_bucket" {
     source = "../../modules/s3-ml-bucket"

     bucket_name        = "ml-datasets-prod-${data.aws_caller_identity.current.account_id}"
     versioning_enabled = true
     lifecycle_days_ia  = 90  # Keep longer in prod

     tags = {
       Environment = "prod"
       Purpose     = "datasets"
     }
   }

   module "gpu_server" {
     source = "../../modules/gpu-instance"

     instance_type = "p3.8xlarge"  # Powerful for prod
     instance_name = "prod-ml-server"
   }

   data "aws_caller_identity" "current" {}
   ```

5. **Deploy**
   ```bash
   # Dev environment
   cd environments/dev
   terraform init
   terraform apply

   # Prod environment
   cd ../prod
   terraform init
   terraform apply

   # Destroy both
   cd ../dev && terraform destroy
   cd ../prod && terraform destroy
   ```

### Challenge Tasks

- [ ] Add module versioning with Git tags
- [ ] Create README.md for each module with usage examples
- [ ] Publish modules to private Terraform Registry
- [ ] Add validation for module inputs
- [ ] Create module for complete ML platform (VPC + GPU + S3)

### Expected Results

✅ Two reusable modules created
✅ Modules used in multiple environments
✅ Different configurations per environment
✅ DRY principle applied

---

## Exercise 05: Remote State with S3 Backend

**Objective**: Configure remote state storage for team collaboration.

### Learning Goals

- Set up S3 backend for state storage
- Configure state locking with DynamoDB
- Enable state encryption
- Migrate from local to remote state

### Steps

1. **Create state backend infrastructure**

   **backend-setup/main.tf**:
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   # S3 bucket for state
   resource "aws_s3_bucket" "terraform_state" {
     bucket = "my-terraform-state-${data.aws_caller_identity.current.account_id}"

     tags = {
       Name    = "Terraform State"
       Purpose = "terraform-state"
     }
   }

   # Enable versioning
   resource "aws_s3_bucket_versioning" "terraform_state" {
     bucket = aws_s3_bucket.terraform_state.id

     versioning_configuration {
       status = "Enabled"
     }
   }

   # Enable encryption
   resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
     bucket = aws_s3_bucket.terraform_state.id

     rule {
       apply_server_side_encryption_by_default {
         sse_algorithm = "AES256"
       }
     }
   }

   # Block public access
   resource "aws_s3_bucket_public_access_block" "terraform_state" {
     bucket = aws_s3_bucket.terraform_state.id

     block_public_acls       = true
     block_public_policy     = true
     ignore_public_acls      = true
     restrict_public_buckets = true
   }

   # DynamoDB table for state locking
   resource "aws_dynamodb_table" "terraform_locks" {
     name         = "terraform-state-locks"
     billing_mode = "PAY_PER_REQUEST"
     hash_key     = "LockID"

     attribute {
       name = "LockID"
       type = "S"
     }

     tags = {
       Name    = "Terraform State Locks"
       Purpose = "terraform-state-locking"
     }
   }

   data "aws_caller_identity" "current" {}

   output "state_bucket" {
     value = aws_s3_bucket.terraform_state.bucket
   }

   output "dynamodb_table" {
     value = aws_dynamodb_table.terraform_locks.name
   }
   ```

2. **Deploy backend infrastructure**
   ```bash
   cd backend-setup
   terraform init
   terraform apply
   # Note the output values
   ```

3. **Create project using remote state**

   **project/backend.tf**:
   ```hcl
   terraform {
     backend "s3" {
       bucket         = "my-terraform-state-ACCOUNT_ID"  # Replace with your bucket
       key            = "ml-project/terraform.tfstate"
       region         = "us-west-2"
       dynamodb_table = "terraform-state-locks"
       encrypt        = true
     }
   }
   ```

   **project/main.tf**:
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   resource "aws_s3_bucket" "example" {
     bucket = "example-bucket-${data.aws_caller_identity.current.account_id}"
   }

   data "aws_caller_identity" "current" {}
   ```

4. **Initialize and test**
   ```bash
   cd project
   terraform init  # Will configure S3 backend
   terraform apply

   # Verify state is in S3
   aws s3 ls s3://my-terraform-state-ACCOUNT_ID/ml-project/

   # Test state locking by running apply in two terminals simultaneously
   # Second terminal should wait for lock
   ```

### Challenge Tasks

- [ ] Migrate existing local state to remote backend
- [ ] Set up backend for multiple environments (dev, staging, prod)
- [ ] Configure workspaces with remote state
- [ ] Add S3 bucket lifecycle policy for old state versions
- [ ] Create IAM policy for least-privilege state access

### Expected Results

✅ S3 bucket created for state storage
✅ DynamoDB table created for locking
✅ State encryption enabled
✅ State stored remotely
✅ State locking prevents concurrent modifications

---

## Exercise 06: ML Infrastructure with Pulumi (Python)

**Objective**: Build the same ML infrastructure using Pulumi with Python instead of Terraform.

### Learning Goals

- Install and configure Pulumi
- Write infrastructure as Python code
- Compare Pulumi vs Terraform workflow
- Use Python language features (loops, functions)

### Steps

1. **Install Pulumi**
   ```bash
   brew install pulumi  # macOS
   # or: curl -fsSL https://get.pulumi.com | sh

   # Login to Pulumi (free tier)
   pulumi login
   ```

2. **Create new Pulumi project**
   ```bash
   mkdir exercise-06-pulumi
   cd exercise-06-pulumi

   pulumi new aws-python
   # Follow prompts:
   # - project name: ml-infrastructure
   # - description: ML infrastructure with Pulumi
   # - stack name: dev
   # - aws:region: us-west-2
   ```

3. **Edit `__main__.py`**
   ```python
   import pulumi
   import pulumi_aws as aws

   # Get latest Deep Learning AMI
   deep_learning_ami = aws.ec2.get_ami(
       most_recent=True,
       owners=["amazon"],
       filters=[aws.ec2.GetAmiFilterArgs(
           name="name",
           values=["Deep Learning AMI GPU PyTorch*"]
       )]
   )

   # S3 bucket for datasets
   datasets_bucket = aws.s3.Bucket(
       "ml-datasets",
       bucket=f"ml-datasets-{aws.get_caller_identity().account_id}",
       versioning=aws.s3.BucketVersioningArgs(
           enabled=True
       ),
       server_side_encryption_configuration=aws.s3.BucketServerSideEncryptionConfigurationArgs(
           rule=aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
               apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                   sse_algorithm="AES256"
               )
           )
       ),
       tags={
           "Name": "ML Datasets",
           "Purpose": "datasets"
       }
   )

   # Security group
   ml_security_group = aws.ec2.SecurityGroup(
       "ml-sg",
       description="Security group for ML training",
       ingress=[
           aws.ec2.SecurityGroupIngressArgs(
               protocol="tcp",
               from_port=22,
               to_port=22,
               cidr_blocks=["0.0.0.0/0"]
           )
       ],
       egress=[
           aws.ec2.SecurityGroupEgressArgs(
               protocol="-1",
               from_port=0,
               to_port=0,
               cidr_blocks=["0.0.0.0/0"]
           )
       ]
   )

   # Create multiple GPU instances using Python loop
   gpu_instances = []
   for i in range(2):  # Python loop!
       instance = aws.ec2.Instance(
           f"gpu-instance-{i}",
           ami=deep_learning_ami.id,
           instance_type="t3.micro",  # Use t3.micro for testing
           vpc_security_group_ids=[ml_security_group.id],
           tags={
               "Name": f"GPU-Instance-{i+1}",
               "ManagedBy": "Pulumi"
           }
       )
       gpu_instances.append(instance)

   # Exports
   pulumi.export("bucket_name", datasets_bucket.bucket)
   pulumi.export("instance_ips", [inst.public_ip for inst in gpu_instances])
   ```

4. **Deploy**
   ```bash
   # Preview changes
   pulumi preview

   # Deploy
   pulumi up
   # Select 'yes'

   # View outputs
   pulumi stack output

   # Destroy
   pulumi destroy
   ```

### Challenge Tasks

- [ ] Add IAM role using Pulumi
- [ ] Create a Python function to generate tags
- [ ] Use Python dictionary to configure multiple environments
- [ ] Compare Pulumi state management vs Terraform
- [ ] Convert Exercise 03 (GPU cluster) to Pulumi

### Expected Results

✅ Pulumi project created
✅ Infrastructure defined in Python
✅ S3 bucket and EC2 instances created
✅ Pulumi state stored in Pulumi Service
✅ Comparison of Pulumi vs Terraform workflow

---

## Exercise 07: GitOps Workflow with GitHub Actions

**Objective**: Implement automated Terraform workflow with GitHub Actions for infrastructure CI/CD.

### Learning Goals

- Set up GitHub Actions for Terraform
- Implement automated plan on PRs
- Add security scanning (tfsec)
- Automate apply on merge to main

### Steps

1. **Create GitHub repository**
   ```bash
   mkdir exercise-07-gitops
   cd exercise-07-gitops
   git init
   gh repo create exercise-07-gitops --private
   ```

2. **Add Terraform code** (reuse Exercise 02)

3. **Create GitHub Actions workflow**

   **.github/workflows/terraform.yml**:
   ```yaml
   name: Terraform CI/CD

   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   env:
     AWS_REGION: us-west-2

   jobs:
     terraform:
       name: Terraform
       runs-on: ubuntu-latest

       permissions:
         pull-requests: write
         contents: read

       steps:
         - name: Checkout
           uses: actions/checkout@v3

         - name: Setup Terraform
           uses: hashicorp/setup-terraform@v2
           with:
             terraform_version: 1.6.0

         - name: Configure AWS Credentials
           uses: aws-actions/configure-aws-credentials@v2
           with:
             aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
             aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
             aws-region: ${{ env.AWS_REGION }}

         - name: Terraform Format
           id: fmt
           run: terraform fmt -check
           continue-on-error: true

         - name: Terraform Init
           id: init
           run: terraform init

         - name: Terraform Validate
           id: validate
           run: terraform validate -no-color

         - name: tfsec Security Scan
           uses: aquasecurity/tfsec-action@v1.0.0
           with:
             soft_fail: true

         - name: Terraform Plan
           id: plan
           if: github.event_name == 'pull_request'
           run: terraform plan -no-color
           continue-on-error: true

         - name: Comment PR with Plan
           uses: actions/github-script@v6
           if: github.event_name == 'pull_request'
           with:
             github-token: ${{ secrets.GITHUB_TOKEN }}
             script: |
               const output = `#### Terraform Format 📝\`${{ steps.fmt.outcome }}\`
               #### Terraform Validation 🤖\`${{ steps.validate.outcome }}\`
               #### Terraform Plan 📖\`${{ steps.plan.outcome }}\`

               <details><summary>Show Plan</summary>

               \`\`\`terraform
               ${{ steps.plan.outputs.stdout }}
               \`\`\`

               </details>

               *Pushed by: @${{ github.actor }}, Action: \`${{ github.event_name }}\`*`;

               github.rest.issues.createComment({
                 issue_number: context.issue.number,
                 owner: context.repo.owner,
                 repo: context.repo.repo,
                 body: output
               })

         - name: Terraform Apply
           if: github.ref == 'refs/heads/main' && github.event_name == 'push'
           run: terraform apply -auto-approve
   ```

4. **Add AWS credentials to GitHub Secrets**
   - Go to repository Settings → Secrets → Actions
   - Add `AWS_ACCESS_KEY_ID`
   - Add `AWS_SECRET_ACCESS_KEY`

5. **Test workflow**
   ```bash
   # Create feature branch
   git checkout -b add-bucket

   # Make change to main.tf
   # Add a new S3 bucket

   # Commit and push
   git add .
   git commit -m "Add new S3 bucket"
   git push origin add-bucket

   # Create PR
   gh pr create --title "Add new S3 bucket" --body "Adds bucket for feature data"

   # Check Actions tab - should see terraform plan in PR comments

   # Merge PR
   gh pr merge --squash

   # Check Actions tab - should see terraform apply on main
   ```

### Challenge Tasks

- [ ] Add Infracost to show cost estimates in PRs
- [ ] Add Checkov for additional security scanning
- [ ] Implement approval requirement for production deployments
- [ ] Add drift detection scheduled workflow
- [ ] Create separate workflows for multiple environments

### Expected Results

✅ GitHub Actions workflow created
✅ Terraform plan runs on PRs
✅ Plan output commented on PR
✅ Security scanning with tfsec
✅ Terraform apply runs on merge to main

---

## Exercise 08: Secrets Management with AWS Secrets Manager

**Objective**: Securely manage secrets in Terraform using AWS Secrets Manager and IAM roles.

### Steps

1. **Create project**
   ```bash
   mkdir exercise-08-secrets
   cd exercise-08-secrets
   ```

2. **Create main.tf with secrets**
   ```hcl
   provider "aws" {
     region = "us-west-2"
   }

   # Generate random password for database
   resource "random_password" "db_password" {
     length  = 32
     special = true
   }

   # Store in Secrets Manager
   resource "aws_secretsmanager_secret" "db_password" {
     name        = "ml-db-password"
     description = "Database password for ML application"
   }

   resource "aws_secretsmanager_secret_version" "db_password" {
     secret_id     = aws_secretsmanager_secret.db_password.id
     secret_string = random_password.db_password.result
   }

   # IAM role for EC2 to read secrets
   resource "aws_iam_role" "ml_app" {
     name = "ml-app-role"

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

   # Policy to read specific secret
   resource "aws_iam_role_policy" "read_secret" {
     role = aws_iam_role.ml_app.id

     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [{
         Action   = "secretsmanager:GetSecretValue"
         Effect   = "Allow"
         Resource = aws_secretsmanager_secret.db_password.arn
       }]
     })
   }

   # Instance profile
   resource "aws_iam_instance_profile" "ml_app" {
     name = "ml-app-profile"
     role = aws_iam_role.ml_app.name
   }

   # EC2 instance with IAM role
   resource "aws_instance" "ml_app" {
     ami                  = data.aws_ami.amazon_linux.id
     instance_type        = "t3.micro"
     iam_instance_profile = aws_iam_instance_profile.ml_app.name

     user_data = <<-EOF
       #!/bin/bash
       # Install AWS CLI
       yum install -y aws-cli

       # Retrieve secret at runtime
       DB_PASSWORD=$(aws secretsmanager get-secret-value \
         --secret-id ${aws_secretsmanager_secret.db_password.id} \
         --query SecretString \
         --output text \
         --region us-west-2)

       # Use password in application
       echo "export DB_PASSWORD='$DB_PASSWORD'" >> /etc/environment
     EOF

     tags = {
       Name = "ML-App-Server"
     }
   }

   data "aws_ami" "amazon_linux" {
     most_recent = true
     owners      = ["amazon"]
     filter {
       name   = "name"
       values = ["amzn2-ami-hvm-*-x86_64-gp2"]
     }
   }

   output "secret_arn" {
     description = "ARN of the secret"
     value       = aws_secretsmanager_secret.db_password.arn
   }
   ```

3. **Add pre-commit hook to prevent secret commits**

   **.pre-commit-config.yaml**:
   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
         - id: detect-aws-credentials
         - id: detect-private-key

     - repo: https://github.com/gitleaks/gitleaks
       rev: v8.16.1
       hooks:
         - id: gitleaks
   ```

   Install and test:
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

4. **Deploy and test**
   ```bash
   terraform init
   terraform apply

   # Verify secret exists
   aws secretsmanager get-secret-value \
     --secret-id ml-db-password \
     --query SecretString \
     --output text

   # SSH to instance and verify secret is accessible
   # (Instance should be able to retrieve secret via IAM role)
   ```

### Challenge Tasks

- [ ] Rotate secrets automatically with Lambda
- [ ] Use HashiCorp Vault instead of AWS Secrets Manager
- [ ] Implement secret versioning strategy
- [ ] Add monitoring for secret access
- [ ] Create secrets for multi-environment setup

### Expected Results

✅ Secrets created in AWS Secrets Manager
✅ IAM role grants least-privilege access
✅ Secrets never hardcoded in Terraform
✅ Pre-commit hooks prevent secret commits
✅ Application retrieves secrets at runtime

---

## Final Project: Complete ML Platform

**Objective**: Combine all exercises into a production-ready ML platform.

### Requirements

- [ ] Multi-environment setup (dev, staging, prod)
- [ ] GPU training cluster with auto-scaling
- [ ] S3 buckets for data, models, logs
- [ ] Proper VPC, subnets, security groups
- [ ] IAM roles with least privilege
- [ ] Secrets management
- [ ] Remote state with locking
- [ ] Reusable modules
- [ ] GitOps workflow with GitHub Actions
- [ ] Security scanning (tfsec, Checkov)
- [ ] Cost estimation (Infracost)
- [ ] Monitoring and logging
- [ ] Documentation

**Estimated time**: 10-15 hours

---

## Additional Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [Pulumi Documentation](https://www.pulumi.com/docs/)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)

---

**Congratulations on completing the Infrastructure as Code exercises!**

Return to [Module README](../README.md) for the quiz and next steps.
