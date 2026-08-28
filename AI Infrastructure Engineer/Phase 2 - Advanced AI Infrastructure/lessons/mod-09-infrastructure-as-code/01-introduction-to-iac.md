# Lesson 01: Introduction to Infrastructure as Code

## Learning Objectives

By the end of this lesson, you will:

- Understand what Infrastructure as Code (IaC) is and why it's essential
- Recognize the problems IaC solves in AI/ML infrastructure
- Distinguish between declarative and imperative approaches
- Compare popular IaC tools (Terraform, Pulumi, CloudFormation, Ansible)
- Understand special considerations for AI/ML infrastructure

## What is Infrastructure as Code?

**Infrastructure as Code (IaC)** is the practice of managing and provisioning infrastructure through machine-readable definition files, rather than manual processes or interactive configuration tools.

### Traditional Infrastructure Management

```
Developer → Opens AWS Console → Clicks buttons → Creates EC2 instance
           → Configures security groups manually
           → Sets up networking in UI
           → Forgets exactly what was configured
           → Cannot easily reproduce
```

**Problems:**
- Manual, error-prone, time-consuming
- No version control or audit trail
- Difficult to reproduce
- Hard to collaborate
- Inconsistent environments (dev vs prod)

### Infrastructure as Code Approach

```python
# infrastructure/main.tf
resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "p3.2xlarge"  # GPU instance

  tags = {
    Name        = "ML-Training-Server"
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}
```

**Benefits:**
- Version controlled (Git)
- Reproducible and consistent
- Automated deployment
- Collaborative (code reviews)
- Self-documenting
- Easily cloned for different environments

## Why IaC Matters for AI Infrastructure

AI/ML infrastructure has unique challenges that make IaC especially valuable:

### 1. **Cost Management**

GPU instances are expensive ($3-$24+ per hour). IaC enables:

```hcl
# Provision GPU cluster only when needed
resource "aws_instance" "training_cluster" {
  count         = var.training_active ? 3 : 0  # Conditional creation
  instance_type = "p3.8xlarge"
  # ... configuration
}
```

Spin up expensive resources for training, destroy when done, saving thousands of dollars.

### 2. **Reproducible Experiments**

ML experiments must be reproducible. IaC ensures identical infrastructure:

```python
# Pulumi example - same infrastructure every time
training_env = aws.ec2.Instance(
    "ml-training",
    instance_type="p3.2xlarge",
    ami="ami-tensorflow-gpu-2.12",
    user_data="""#!/bin/bash
        pip install tensorflow==2.12.0
        python /opt/ml/train.py
    """
)
```

### 3. **Multi-Environment Management**

Easily create dev, staging, production environments:

```
infrastructure/
├── environments/
│   ├── dev/
│   │   └── terraform.tfvars    # instance_type = "t3.medium"
│   ├── staging/
│   │   └── terraform.tfvars    # instance_type = "p3.2xlarge"
│   └── prod/
│       └── terraform.tfvars    # instance_type = "p3.8xlarge"
└── main.tf                     # Same code, different configs
```

### 4. **Disaster Recovery**

If infrastructure fails, rebuild it in minutes:

```bash
# Disaster strikes - entire cluster lost
terraform apply  # Rebuild everything from code
# ✓ GPU cluster
# ✓ Storage buckets
# ✓ Networking
# ✓ Load balancers
# ✓ Monitoring
# All recreated in <10 minutes
```

### 5. **Collaboration**

Infrastructure changes go through code review:

```
Pull Request: Add GPU cluster for new model training
Files changed:
  + gpu_cluster.tf      (new GPU instances)
  + storage.tf          (add model storage bucket)

Reviewers: @ml-team, @infra-team
Status: 2 approvals required
```

## Declarative vs Imperative Approaches

### Imperative (How)

Specify **how** to achieve the desired state, step by step:

```bash
# Imperative - shell script
aws ec2 run-instances \
  --image-id ami-12345678 \
  --instance-type p3.2xlarge \
  --key-name my-key

aws ec2 create-security-group \
  --group-name ml-sg \
  --description "ML security group"

aws ec2 authorize-security-group-ingress \
  --group-name ml-sg \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
```

**Problems:**
- Must handle current state manually
- Re-running creates duplicates
- Hard to maintain
- Error recovery is complex

### Declarative (What)

Specify **what** the desired end state should be:

```hcl
# Declarative - Terraform
resource "aws_instance" "ml_server" {
  ami           = "ami-12345678"
  instance_type = "p3.2xlarge"
  key_name      = "my-key"
}

resource "aws_security_group" "ml_sg" {
  name        = "ml-sg"
  description = "ML security group"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

**Benefits:**
- Idempotent (safe to re-run)
- Self-correcting (detects drift)
- Easier to understand
- Automatic dependency resolution

## Popular IaC Tools

### 1. Terraform

**Developer**: HashiCorp
**Language**: HCL (HashiCorp Configuration Language)
**Approach**: Declarative
**Cloud Support**: Multi-cloud (AWS, GCP, Azure, 3000+ providers)

```hcl
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "ml_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "p3.2xlarge"
}
```

**Pros:**
- Industry standard
- Huge provider ecosystem
- Strong community
- Excellent state management
- Multi-cloud support

**Cons:**
- DSL to learn (HCL)
- Less flexible than real programming languages

**Best for**: Multi-cloud environments, large teams, standardization

### 2. Pulumi

**Developer**: Pulumi Corp
**Language**: Python, TypeScript, Go, C#, Java
**Approach**: Declarative
**Cloud Support**: Multi-cloud

```python
import pulumi
import pulumi_aws as aws

ml_server = aws.ec2.Instance(
    "ml-server",
    instance_type="p3.2xlarge",
    ami="ami-0c55b159cbfafe1f0"
)

pulumi.export("instance_id", ml_server.id)
```

**Pros:**
- Use familiar programming languages (Python!)
- Full language features (loops, conditionals, functions)
- Type safety and IDE support
- Easier for developers

**Cons:**
- Smaller community than Terraform
- Newer (less mature)
- Less provider coverage

**Best for**: Python-heavy teams, complex logic, software engineers

### 3. AWS CloudFormation

**Developer**: AWS
**Language**: YAML or JSON
**Approach**: Declarative
**Cloud Support**: AWS only

```yaml
Resources:
  MLServer:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: p3.2xlarge
      ImageId: ami-0c55b159cbfafe1f0
```

**Pros:**
- Native AWS integration
- No extra tools required
- AWS support
- StackSets for multi-account

**Cons:**
- AWS only (vendor lock-in)
- Verbose YAML
- Limited abstraction
- Slower iteration

**Best for**: AWS-only shops, compliance requirements

### 4. Ansible

**Developer**: Red Hat
**Language**: YAML
**Approach**: Imperative (but can be declarative)
**Cloud Support**: Multi-cloud + configuration management

```yaml
- name: Create ML EC2 instance
  amazon.aws.ec2_instance:
    name: ml-server
    instance_type: p3.2xlarge
    image_id: ami-0c55b159cbfafe1f0
    state: present
```

**Pros:**
- Configuration management + provisioning
- Agentless
- Simple YAML
- Great for existing servers

**Cons:**
- Imperative nature can cause issues
- State management not as robust
- Slower for large infrastructures

**Best for**: Configuration management, hybrid cloud/on-prem

### Comparison Matrix

| Feature | Terraform | Pulumi | CloudFormation | Ansible |
|---------|-----------|--------|----------------|---------|
| **Multi-cloud** | ✅ Excellent | ✅ Excellent | ❌ AWS only | ✅ Good |
| **State Management** | ✅ Excellent | ✅ Excellent | ✅ Good | ⚠️ Limited |
| **Learning Curve** | Medium | Low (if you know Python) | Medium | Low |
| **Community** | ✅ Huge | Growing | Large (AWS) | ✅ Huge |
| **Language** | HCL (DSL) | Python/TS/Go | YAML/JSON | YAML |
| **Cost** | Free (OSS) | Free + paid tiers | Free (AWS) | Free (OSS) |
| **Best for AI/ML** | ✅✅ | ✅✅ | ✅ | ⚠️ |

## Key IaC Concepts

### 1. **Providers**

Plugins that interact with APIs (AWS, GCP, Azure, etc.):

```hcl
provider "aws" {
  region = "us-west-2"
}

provider "google" {
  project = "my-ml-project"
  region  = "us-central1"
}
```

### 2. **Resources**

Infrastructure components to create:

```hcl
resource "aws_s3_bucket" "ml_data" {
  bucket = "my-ml-datasets"
}

resource "aws_instance" "gpu_server" {
  ami           = "ami-12345"
  instance_type = "p3.8xlarge"
}
```

### 3. **State**

Current infrastructure status tracked by IaC tool:

```
terraform.tfstate
{
  "resources": [
    {
      "type": "aws_instance",
      "name": "gpu_server",
      "instances": [{
        "attributes": {
          "id": "i-0abc123",
          "public_ip": "54.123.45.67"
        }
      }]
    }
  ]
}
```

### 4. **Variables**

Parameterize infrastructure for reusability:

```hcl
variable "instance_type" {
  description = "EC2 instance type for ML training"
  type        = string
  default     = "p3.2xlarge"
}

resource "aws_instance" "ml_server" {
  instance_type = var.instance_type
}
```

### 5. **Outputs**

Export values for use elsewhere:

```hcl
output "ml_server_ip" {
  value = aws_instance.ml_server.public_ip
}

output "s3_bucket_name" {
  value = aws_s3_bucket.ml_data.bucket
}
```

## IaC Workflow

```
1. Write Code
   ├── Define infrastructure in .tf or .py files
   └── Specify providers, resources, variables

2. Version Control
   ├── Commit to Git
   └── Create pull request for review

3. Plan
   ├── terraform plan / pulumi preview
   └── Review changes before applying

4. Apply
   ├── terraform apply / pulumi up
   └── Infrastructure is created/modified

5. Monitor State
   ├── Check terraform.tfstate
   └── Verify infrastructure matches code

6. Update
   ├── Modify code
   └── Repeat plan → apply cycle

7. Destroy (when done)
   ├── terraform destroy / pulumi destroy
   └── Remove all infrastructure (save money!)
```

## AI Infrastructure IaC Patterns

### Pattern 1: Training Environment

```hcl
# GPU cluster for training
module "training_cluster" {
  source = "./modules/gpu-cluster"

  instance_count = 4
  instance_type  = "p3.8xlarge"
  dataset_bucket = "s3://my-datasets"

  # Automatically shut down after 8 hours
  auto_shutdown_hours = 8
}
```

### Pattern 2: Model Serving Infrastructure

```hcl
# Kubernetes cluster for model serving
module "inference_cluster" {
  source = "./modules/eks-cluster"

  node_groups = {
    cpu_nodes = {
      instance_type = "c5.2xlarge"
      min_size      = 2
      max_size      = 10
    }
    gpu_nodes = {
      instance_type = "p3.2xlarge"
      min_size      = 1
      max_size      = 5
    }
  }
}
```

### Pattern 3: Data Pipeline Infrastructure

```hcl
# S3 buckets for ML data
resource "aws_s3_bucket" "raw_data" {
  bucket = "ml-raw-data-${var.environment}"
}

resource "aws_s3_bucket" "processed_data" {
  bucket = "ml-processed-data-${var.environment}"
}

resource "aws_s3_bucket" "models" {
  bucket = "ml-models-${var.environment}"

  versioning {
    enabled = true  # Track model versions
  }
}
```

## Special Considerations for AI/ML Infrastructure

### 1. **GPU Availability**

GPUs are limited and region-specific:

```hcl
# Check multiple regions for GPU availability
variable "gpu_regions" {
  default = ["us-west-2", "us-east-1", "eu-west-1"]
}

# Use availability zones with GPU instances
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}
```

### 2. **Cost Optimization**

```hcl
# Use spot instances for training (70% cheaper)
resource "aws_spot_instance_request" "ml_training" {
  ami           = "ami-12345"
  instance_type = "p3.8xlarge"
  spot_price    = "3.00"  # Max price per hour

  # Persistent request - auto-retry if terminated
  spot_type = "persistent"
}
```

### 3. **Data Locality**

```hcl
# Ensure compute and storage are in same region
variable "region" {
  default = "us-west-2"
}

resource "aws_s3_bucket" "datasets" {
  bucket = "ml-datasets"
  region = var.region  # Same region as GPU instances
}

resource "aws_instance" "gpu_server" {
  region = var.region
  # ...
}
```

### 4. **Autoscaling for Inference**

```hcl
# Auto-scale inference servers based on load
resource "aws_autoscaling_group" "inference" {
  min_size         = 2
  max_size         = 20
  desired_capacity = 5

  target_group_arns = [aws_lb_target_group.inference.arn]

  # Scale based on CPU or custom metrics
  tag {
    key                 = "ModelVersion"
    value               = "v2.3"
    propagate_at_launch = true
  }
}
```

## Best Practices

### 1. **Start Small**

Begin with simple resources, gradually add complexity:

```
Week 1: Single EC2 instance
Week 2: Add S3 buckets
Week 3: Add networking
Week 4: Complete ML platform
```

### 2. **Use Version Control**

```bash
git init
git add *.tf
git commit -m "Initial ML infrastructure"
git push origin main
```

### 3. **Always Run Plan First**

```bash
terraform plan   # See what will change
# Review output carefully
terraform apply  # Only if plan looks good
```

### 4. **Use Remote State**

Store state in S3, not locally:

```hcl
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "ml-infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}
```

### 5. **Tag Everything**

```hcl
locals {
  common_tags = {
    Project     = "ML-Platform"
    Environment = var.environment
    ManagedBy   = "Terraform"
    CostCenter  = "AI-Research"
    Owner       = "ml-team@company.com"
  }
}

resource "aws_instance" "ml_server" {
  # ...
  tags = local.common_tags
}
```

## Common Pitfalls to Avoid

### ❌ Don't: Edit Infrastructure Manually

```
DON'T: Log into AWS Console and modify instance
DO:    Update Terraform code and apply
```

### ❌ Don't: Commit Secrets

```hcl
# ❌ DON'T DO THIS
provider "aws" {
  access_key = "AKIAIOSFODNN7EXAMPLE"  # Secret in code!
  secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
}

# ✅ DO THIS
provider "aws" {
  # Credentials from environment variables or AWS CLI config
}
```

### ❌ Don't: Skip Planning

```bash
# ❌ DON'T
terraform apply -auto-approve  # Dangerous!

# ✅ DO
terraform plan
# Review output
terraform apply
```

### ❌ Don't: Mix Manual and IaC

Choose IaC or manual, not both. Mixed approaches lead to state drift and confusion.

## Getting Started Checklist

- [ ] Install Terraform or Pulumi
- [ ] Set up cloud provider CLI (AWS CLI, gcloud, etc.)
- [ ] Configure credentials securely
- [ ] Create first simple resource (S3 bucket)
- [ ] Version control your code
- [ ] Learn to read plan output
- [ ] Practice apply and destroy cycle
- [ ] Set up remote state backend
- [ ] Create reusable modules

## Next Steps

Now that you understand IaC fundamentals, you're ready to:

- **Lesson 02**: Learn Terraform syntax and commands in depth
- **Lesson 03**: Master state management for team collaboration
- **Lesson 04**: Build real AI infrastructure with Terraform

## Key Takeaways

✅ IaC manages infrastructure through code, not manual processes
✅ Declarative approaches (Terraform, Pulumi) are preferred for infrastructure
✅ IaC is essential for AI/ML due to cost, reproducibility, and scale requirements
✅ Terraform and Pulumi are the leading multi-cloud IaC tools
✅ Always use version control, plan before applying, and avoid manual changes
✅ AI infrastructure has special needs: GPUs, cost optimization, data locality

## Additional Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [Pulumi Documentation](https://www.pulumi.com/docs)
- [HashiCorp Learn - Terraform](https://learn.hashicorp.com/terraform)
- [AWS Well-Architected Framework - IaC](https://aws.amazon.com/architecture/well-architected/)

---

**Next Lesson**: [02-terraform-fundamentals.md](02-terraform-fundamentals.md)
