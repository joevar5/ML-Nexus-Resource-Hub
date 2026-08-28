# Lesson 03: Terraform State Management

## Learning Objectives

By the end of this lesson, you will:

- Understand what Terraform state is and why it's critical
- Differentiate between local and remote state backends
- Configure S3 + DynamoDB backend for team collaboration
- Implement state locking to prevent concurrent modifications
- Master state manipulation commands (list, show, mv, rm, pull, push)
- Use workspaces for managing multiple environments
- Secure state files with encryption
- Plan for backup and disaster recovery scenarios
- Apply state management best practices to ML infrastructure

## What is Terraform State?

### The State File

Terraform stores information about your managed infrastructure in a **state file** (`terraform.tfstate`). This is a JSON file that maps your Terraform configuration to real-world resources.

**Example state file snippet:**
```json
{
  "version": 4,
  "terraform_version": "1.6.0",
  "serial": 3,
  "lineage": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "outputs": {
    "ml_server_ip": {
      "value": "54.123.45.67",
      "type": "string"
    }
  },
  "resources": [
    {
      "mode": "managed",
      "type": "aws_instance",
      "name": "ml_server",
      "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
      "instances": [
        {
          "schema_version": 1,
          "attributes": {
            "id": "i-0abc123def456789",
            "instance_type": "p3.2xlarge",
            "ami": "ami-0c55b159cbfafe1f0",
            "public_ip": "54.123.45.67",
            "private_ip": "10.0.1.42",
            "tags": {
              "Name": "ML-Training-Server"
            }
          }
        }
      ]
    }
  ]
}
```

### Why State Matters

**1. Mapping Configuration to Reality**

Terraform needs to know which real-world resources correspond to your configuration:

```hcl
# Your configuration
resource "aws_instance" "ml_server" {
  instance_type = "p3.2xlarge"
  ami           = "ami-12345"
}
```

State file says: "This configuration maps to EC2 instance `i-0abc123def456789`"

**2. Tracking Metadata**

State stores important metadata not available from the cloud API:
- Resource dependencies
- Output values
- Provider configurations
- Resource attributes

**3. Performance Optimization**

For large infrastructures (100s of resources), querying the cloud provider API for every resource is slow. State provides a local cache.

**4. Enabling Collaboration**

Multiple team members need to work on the same infrastructure. State provides the single source of truth.

### What Happens Without State?

```bash
# First apply
terraform apply
# Creates: EC2 instance i-0abc123

# Delete state file
rm terraform.tfstate

# Second apply
terraform apply
# Terraform doesn't know instance exists!
# Creates: ANOTHER EC2 instance i-0def456

# Result: Duplicate resources, no management, chaos!
```

## Local vs Remote State

### Local State (Default)

**Location**: `terraform.tfstate` in your working directory

**Pros:**
- Simple for single-user scenarios
- No setup required
- Fast access

**Cons:**
- ❌ Can't collaborate (file conflicts)
- ❌ No locking (concurrent apply causes corruption)
- ❌ Lost if laptop dies
- ❌ Secrets stored in plain text on disk
- ❌ No versioning or history

**Use case:** Personal learning, proof-of-concepts, single-developer projects

### Remote State

**Location**: Cloud storage (S3, Azure Blob, GCS, Terraform Cloud)

**Pros:**
- ✅ Team collaboration
- ✅ State locking (prevents conflicts)
- ✅ Versioning and backup
- ✅ Encryption at rest
- ✅ Access control
- ✅ Disaster recovery

**Cons:**
- Requires initial setup
- Potential costs (usually negligible)

**Use case:** ALL production infrastructure, team projects, ML platforms

## Configuring Remote State: S3 + DynamoDB

The most common remote backend for AWS users combines:
- **S3**: Stores the state file
- **DynamoDB**: Provides state locking

### Step 1: Create S3 Bucket for State

```hcl
# bootstrap/s3-backend.tf
# Run this first to create the backend infrastructure

provider "aws" {
  region = "us-west-2"
}

# S3 bucket for Terraform state
resource "aws_s3_bucket" "terraform_state" {
  bucket = "my-company-terraform-state"  # Must be globally unique

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name        = "Terraform State Bucket"
    Purpose     = "Infrastructure State"
    ManagedBy   = "Terraform"
  }
}

# Enable versioning (keep history of state changes)
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable encryption (protect secrets in state)
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access (security best practice)
resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable logging (audit access to state)
resource "aws_s3_bucket" "terraform_logs" {
  bucket = "my-company-terraform-state-logs"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_logging" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  target_bucket = aws_s3_bucket.terraform_logs.id
  target_prefix = "state-access-logs/"
}

# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-state-locks"
  billing_mode = "PAY_PER_REQUEST"  # On-demand pricing
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name      = "Terraform State Locks"
    Purpose   = "State Locking"
    ManagedBy = "Terraform"
  }
}

# Outputs
output "s3_bucket_name" {
  value       = aws_s3_bucket.terraform_state.id
  description = "Name of the S3 bucket for Terraform state"
}

output "dynamodb_table_name" {
  value       = aws_dynamodb_table.terraform_locks.name
  description = "Name of DynamoDB table for state locking"
}
```

**Deploy backend infrastructure:**
```bash
cd bootstrap
terraform init
terraform apply

# Note the outputs
# s3_bucket_name = "my-company-terraform-state"
# dynamodb_table_name = "terraform-state-locks"
```

### Step 2: Configure Backend in Your Project

**backend.tf** (in your main project):
```hcl
terraform {
  backend "s3" {
    bucket         = "my-company-terraform-state"
    key            = "ml-infrastructure/production/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-state-locks"
    encrypt        = true

    # Optional: Use assume role for cross-account access
    # role_arn = "arn:aws:iam::123456789012:role/TerraformRole"
  }
}
```

**Key parameter explanation:**
- `bucket`: S3 bucket name
- `key`: Path to state file (allows multiple projects in same bucket)
- `region`: AWS region of the bucket
- `dynamodb_table`: Table for locking
- `encrypt`: Enable server-side encryption

### Step 3: Migrate from Local to Remote State

```bash
# Your project with local state
cd ml-infrastructure

# Add backend.tf configuration (as shown above)

# Initialize with backend migration
terraform init -migrate-state

# Terraform will prompt:
# Do you want to copy existing state to the new backend?
#   Pre-existing state was found while migrating the previous "local" backend to the
#   newly configured "s3" backend. No existing state was found in the newly
#   configured "s3" backend. Do you want to copy this state to the new "s3"
#   backend? Enter "yes" to copy and "no" to start with an empty state.

# Enter: yes

# Verify migration
terraform state list

# Old local state is now in S3
# Local terraform.tfstate is backed up as terraform.tfstate.backup
```

### Multiple Projects/Environments Pattern

```hcl
# Use different state file paths per environment

# Development environment
terraform {
  backend "s3" {
    bucket = "my-company-terraform-state"
    key    = "ml-platform/dev/terraform.tfstate"
    region = "us-west-2"
    dynamodb_table = "terraform-state-locks"
    encrypt = true
  }
}

# Staging environment
terraform {
  backend "s3" {
    bucket = "my-company-terraform-state"
    key    = "ml-platform/staging/terraform.tfstate"
    region = "us-west-2"
    dynamodb_table = "terraform-state-locks"
    encrypt = true
  }
}

# Production environment
terraform {
  backend "s3" {
    bucket = "my-company-terraform-state"
    key    = "ml-platform/production/terraform.tfstate"
    region = "us-west-2"
    dynamodb_table = "terraform-state-locks"
    encrypt = true
  }
}
```

**Directory structure:**
```
ml-platform/
├── environments/
│   ├── dev/
│   │   ├── backend.tf      # dev state path
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── backend.tf      # staging state path
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   └── production/
│       ├── backend.tf      # production state path
│       ├── main.tf
│       └── terraform.tfvars
└── modules/
    └── gpu-cluster/
```

## State Locking

State locking prevents multiple users from running `terraform apply` simultaneously, which would corrupt the state.

### How Locking Works

```
User 1: terraform apply
  ↓
  Acquires lock in DynamoDB (LockID: "ml-platform/prod/terraform.tfstate")
  ↓
  Reading state from S3...
  ↓
  Making changes...
  ↓
  Writing state to S3...
  ↓
  Releases lock

User 2: terraform apply (while User 1 is still running)
  ↓
  Attempts to acquire lock
  ↓
  Lock is already held!
  ↓
  Error: Error acquiring the state lock
```

**Error message:**
```
Error: Error acquiring the state lock

Error message: ConditionalCheckFailedException: The conditional request failed
Lock Info:
  ID:        a1b2c3d4-e5f6-7890-abcd-ef1234567890
  Path:      my-company-terraform-state/ml-platform/prod/terraform.tfstate
  Operation: OperationTypeApply
  Who:       user1@host1
  Version:   1.6.0
  Created:   2024-01-15 10:30:00 UTC
  Info:

Terraform acquires a state lock to protect the state from being written
by multiple users at the same time. Please resolve the issue above and try
again. For most commands, you can disable locking with the "-lock=false"
flag, but this is not recommended.
```

### Viewing Lock Status

DynamoDB table entry while locked:
```json
{
  "LockID": "my-company-terraform-state/ml-platform/prod/terraform.tfstate",
  "Info": "{\"ID\":\"a1b2c3d4-e5f6-7890-abcd-ef1234567890\",\"Operation\":\"OperationTypeApply\",\"Info\":\"\",\"Who\":\"user1@host1\",\"Version\":\"1.6.0\",\"Created\":\"2024-01-15T10:30:00Z\",\"Path\":\"my-company-terraform-state/ml-platform/prod/terraform.tfstate\"}"
}
```

### Force Unlock (Emergency Only)

If a lock is stuck (e.g., process crashed):

```bash
# Get lock ID from error message
terraform force-unlock a1b2c3d4-e5f6-7890-abcd-ef1234567890

# Terraform will warn:
# Do not unlock the state unless you are certain another process is not holding the lock.
# Unlocking the state when another process holds it can cause corruption and data loss.

# Only proceed if you're ABSOLUTELY SURE no other process is running
```

**Safer approach:**
```bash
# Check if process is actually running
ps aux | grep terraform

# Or check DynamoDB table
aws dynamodb get-item \
  --table-name terraform-state-locks \
  --key '{"LockID":{"S":"my-company-terraform-state/ml-platform/prod/terraform.tfstate"}}'

# If legitimately stuck, force unlock
terraform force-unlock <lock-id>
```

## State Commands

### terraform state list

List all resources in the state.

```bash
# List all resources
terraform state list

# Output:
# aws_instance.ml_server
# aws_s3_bucket.datasets
# aws_s3_bucket.models
# aws_security_group.ml_sg
# aws_iam_role.ml_role
# aws_iam_instance_profile.ml_profile

# Filter resources
terraform state list | grep aws_instance
# aws_instance.ml_server
# aws_instance.gpu_cluster[0]
# aws_instance.gpu_cluster[1]
# aws_instance.gpu_cluster[2]
```

### terraform state show

Show detailed information about a specific resource.

```bash
# Show resource details
terraform state show aws_instance.ml_server

# Output:
# resource "aws_instance" "ml_server" {
#     ami                          = "ami-0c55b159cbfafe1f0"
#     arn                          = "arn:aws:ec2:us-west-2:123456789012:instance/i-0abc123def456789"
#     associate_public_ip_address  = true
#     availability_zone            = "us-west-2a"
#     id                           = "i-0abc123def456789"
#     instance_type                = "p3.2xlarge"
#     private_ip                   = "10.0.1.42"
#     public_ip                    = "54.123.45.67"
#     tags                         = {
#         "Name" = "ML-Training-Server"
#     }
#     vpc_security_group_ids       = [
#         "sg-0def456ghi789jkl0",
#     ]
# }

# Use in scripts
PUBLIC_IP=$(terraform state show aws_instance.ml_server | grep public_ip | head -1 | awk '{print $3}' | tr -d '"')
echo $PUBLIC_IP
```

### terraform state mv

Move or rename resources in state (doesn't change actual infrastructure).

**Use cases:**
1. Renaming resources in configuration
2. Moving resources between modules
3. Refactoring infrastructure

**Example 1: Rename resource**
```hcl
# Old configuration
resource "aws_instance" "old_name" {
  ami           = "ami-12345"
  instance_type = "p3.2xlarge"
}

# New configuration (you want to rename)
resource "aws_instance" "new_name" {
  ami           = "ami-12345"
  instance_type = "p3.2xlarge"
}
```

Without `state mv`, Terraform would destroy `old_name` and create `new_name` (downtime!).

```bash
# Rename in state first
terraform state mv aws_instance.old_name aws_instance.new_name

# Then update configuration
# Now terraform plan shows no changes
```

**Example 2: Move to module**
```bash
# Before: resource at root level
# resource "aws_instance" "ml_server"

# Move to module
terraform state mv aws_instance.ml_server module.gpu_cluster.aws_instance.ml_server

# Update configuration to use module
# module "gpu_cluster" {
#   source = "./modules/gpu-cluster"
# }
```

**Example 3: Split resources**
```bash
# You have a count-based resource
# resource "aws_instance" "cluster" {
#   count = 3
# }

# Access: aws_instance.cluster[0], [1], [2]

# Split into individual resources
terraform state mv 'aws_instance.cluster[0]' aws_instance.cluster_node_1
terraform state mv 'aws_instance.cluster[1]' aws_instance.cluster_node_2
terraform state mv 'aws_instance.cluster[2]' aws_instance.cluster_node_3
```

### terraform state rm

Remove resource from state (doesn't destroy actual infrastructure).

**Use cases:**
1. Stop managing a resource with Terraform
2. Manually created resource that you imported by mistake
3. Handing resource to another team/system

```bash
# Remove resource from state
terraform state rm aws_instance.legacy_server

# The EC2 instance still exists in AWS
# But Terraform no longer manages it

# Next terraform apply won't try to modify or destroy it
```

**Complete workflow for handing off resources:**
```bash
# Scenario: Moving ML inference cluster to another team

# 1. Export resource configuration
terraform state show aws_instance.inference_cluster > inference_cluster.txt

# 2. Remove from your state
terraform state rm aws_instance.inference_cluster

# 3. Remove from your .tf files
# (delete the resource block)

# 4. Verify
terraform plan
# Should show no changes for this resource

# 5. Give inference_cluster.txt and resource ID to other team
# They can import it: terraform import aws_instance.their_name i-0abc123
```

### terraform state pull

Download remote state to stdout.

```bash
# View remote state
terraform state pull

# Save to file
terraform state pull > state-backup.json

# Use in scripts
STATE=$(terraform state pull)
echo $STATE | jq '.resources[] | select(.type=="aws_instance")'
```

### terraform state push

Upload local state to remote backend (DANGEROUS - rarely needed).

```bash
# Restore from backup
terraform state push state-backup.json

# Warning: This will overwrite remote state!
# Only use for disaster recovery
```

**Safer alternatives:**
- S3 versioning (restore previous version)
- Terraform Cloud/Enterprise state backups

### terraform state replace-provider

Update provider in state (useful for provider migrations).

```bash
# Scenario: Migrating from community provider to official one

# Old provider: registry.terraform.io/terraform-providers/aws
# New provider: registry.terraform.io/hashicorp/aws

terraform state replace-provider \
  registry.terraform.io/terraform-providers/aws \
  registry.terraform.io/hashicorp/aws
```

## Workspaces

Workspaces allow managing multiple instances of the same infrastructure with a single configuration.

### Default Workspace

Every Terraform project starts with a `default` workspace.

```bash
# Show current workspace
terraform workspace show
# default

# List all workspaces
terraform workspace list
# * default
```

### Creating and Using Workspaces

```bash
# Create development workspace
terraform workspace new dev
# Created and switched to workspace "dev"!

# Create staging workspace
terraform workspace new staging

# Create production workspace
terraform workspace new production

# List workspaces
terraform workspace list
#   default
#   dev
# * production
#   staging

# Switch workspace
terraform workspace select dev
# Switched to workspace "dev".

# Delete workspace (must be empty)
terraform workspace delete staging
```

### Workspace-Aware Configuration

```hcl
# Use workspace name in resource naming
resource "aws_instance" "ml_server" {
  ami           = "ami-12345"
  instance_type = terraform.workspace == "production" ? "p3.8xlarge" : "p3.2xlarge"

  tags = {
    Name        = "ml-server-${terraform.workspace}"
    Environment = terraform.workspace
  }
}

# Different instance counts per workspace
resource "aws_instance" "training_cluster" {
  count = terraform.workspace == "production" ? 10 : 2

  ami           = "ami-12345"
  instance_type = "p3.8xlarge"

  tags = {
    Name        = "training-${terraform.workspace}-${count.index}"
    Environment = terraform.workspace
  }
}

# Load workspace-specific variables
locals {
  env_config = {
    dev = {
      instance_count = 1
      instance_type  = "t3.large"
      disk_size      = 50
    }
    staging = {
      instance_count = 2
      instance_type  = "p3.2xlarge"
      disk_size      = 100
    }
    production = {
      instance_count = 10
      instance_type  = "p3.8xlarge"
      disk_size      = 500
    }
  }

  config = local.env_config[terraform.workspace]
}

resource "aws_instance" "ml_cluster" {
  count         = local.config.instance_count
  instance_type = local.config.instance_type

  root_block_device {
    volume_size = local.config.disk_size
  }
}
```

### Workspaces with Remote State

With S3 backend, each workspace gets its own state file:

```
my-company-terraform-state/
└── ml-platform/
    └── env:/
        ├── default/
        │   └── terraform.tfstate
        ├── dev/
        │   └── terraform.tfstate
        ├── staging/
        │   └── terraform.tfstate
        └── production/
            └── terraform.tfstate
```

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "my-company-terraform-state"
    key            = "ml-platform/terraform.tfstate"  # Base path
    region         = "us-west-2"
    dynamodb_table = "terraform-state-locks"
    encrypt        = true
  }
}

# Actual paths:
# dev:        ml-platform/env:/dev/terraform.tfstate
# staging:    ml-platform/env:/staging/terraform.tfstate
# production: ml-platform/env:/production/terraform.tfstate
```

### Workspaces vs Separate Directories

**Workspaces:**
- ✅ Same code for all environments
- ✅ Easy to switch between environments
- ✅ Less code duplication
- ❌ Risk of applying to wrong environment
- ❌ All environments must use same backend

**Separate directories:**
- ✅ Complete isolation
- ✅ Different backends per environment
- ✅ Easier to enforce separation of concerns
- ❌ Code duplication
- ❌ More complex to keep in sync

**Recommendation for ML infrastructure:**
Use separate directories for production vs non-production, workspaces within non-production:

```
ml-infrastructure/
├── production/           # Separate directory for production
│   ├── backend.tf       # Production-specific backend
│   ├── main.tf
│   └── terraform.tfvars
└── non-production/      # Separate directory for dev/staging
    ├── backend.tf       # Non-prod backend
    ├── main.tf
    └── terraform.tfvars # Use workspaces: dev, staging
```

## State Security and Encryption

### Encryption at Rest

**S3 encryption (already configured in Step 1):**
```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"  # AWS-managed keys
    }
  }
}

# Or use KMS for more control
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.terraform_state.arn
    }
  }
}

resource "aws_kms_key" "terraform_state" {
  description             = "KMS key for Terraform state encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true
}
```

### Encryption in Transit

Always use HTTPS for S3 backend (enabled by default).

### Sensitive Data in State

**The problem:** State files contain sensitive information in plain text.

```json
{
  "resources": [
    {
      "type": "aws_db_instance",
      "name": "ml_database",
      "instances": [{
        "attributes": {
          "password": "super-secret-password-123",  // ❌ Visible in state!
          "username": "admin"
        }
      }]
    }
  ]
}
```

**Best practices:**

1. **Use remote state with encryption**
2. **Restrict access to state files**
3. **Use secrets management services**

```hcl
# ❌ DON'T: Hardcode secrets
resource "aws_db_instance" "ml_db" {
  password = "super-secret-password"  # Will be in state!
}

# ✅ DO: Use AWS Secrets Manager
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "ml-database-password"
}

resource "aws_db_instance" "ml_db" {
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
  # Still in state, but at least not in code
}

# ✅ BETTER: Use IAM authentication (no passwords)
resource "aws_db_instance" "ml_db" {
  iam_database_authentication_enabled = true
  # No password needed!
}
```

### Access Control

**S3 bucket policy (restrict state access):**
```hcl
resource "aws_s3_bucket_policy" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EnforcedTLS"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.terraform_state.arn,
          "${aws_s3_bucket.terraform_state.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "RestrictAccess"
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::123456789012:role/TerraformRole",
            "arn:aws:iam::123456789012:user/ml-admin"
          ]
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.terraform_state.arn}/*"
      }
    ]
  })
}
```

**IAM policy for Terraform users:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-company-terraform-state",
        "arn:aws:s3:::my-company-terraform-state/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem"
      ],
      "Resource": "arn:aws:dynamodb:us-west-2:123456789012:table/terraform-state-locks"
    }
  ]
}
```

## Backup and Disaster Recovery

### S3 Versioning (Automatic Backups)

Already configured in Step 1. S3 keeps all versions of state file.

**View versions:**
```bash
# List all versions
aws s3api list-object-versions \
  --bucket my-company-terraform-state \
  --prefix ml-platform/production/terraform.tfstate

# Output shows all versions with VersionId
```

**Restore previous version:**
```bash
# 1. Download specific version
aws s3api get-object \
  --bucket my-company-terraform-state \
  --key ml-platform/production/terraform.tfstate \
  --version-id <VERSION_ID> \
  state-backup.json

# 2. Restore using state push (CAREFUL!)
terraform state push state-backup.json
```

### Manual Backups

**Automated backup script:**
```bash
#!/bin/bash
# backup-terraform-state.sh

BUCKET="my-company-terraform-state"
STATE_KEY="ml-platform/production/terraform.tfstate"
BACKUP_DIR="./state-backups"
DATE=$(date +%Y%m%d-%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Download current state
aws s3 cp s3://${BUCKET}/${STATE_KEY} \
  ${BACKUP_DIR}/terraform.tfstate.${DATE}

# Keep only last 30 days
find $BACKUP_DIR -name "terraform.tfstate.*" -mtime +30 -delete

echo "State backed up to ${BACKUP_DIR}/terraform.tfstate.${DATE}"
```

**Schedule with cron:**
```bash
# Run backup daily at 2 AM
0 2 * * * /path/to/backup-terraform-state.sh
```

### Disaster Recovery Procedure

**Scenario: Corrupted state file**

```bash
# Step 1: Don't panic! State is versioned.

# Step 2: List recent versions
aws s3api list-object-versions \
  --bucket my-company-terraform-state \
  --prefix ml-platform/production/terraform.tfstate \
  --max-items 10

# Step 3: Download last known good version
aws s3api get-object \
  --bucket my-company-terraform-state \
  --key ml-platform/production/terraform.tfstate \
  --version-id <GOOD_VERSION_ID> \
  recovered-state.json

# Step 4: Verify the recovered state
cat recovered-state.json | jq '.resources | length'
# Should show expected number of resources

# Step 5: Push recovered state (CAREFUL!)
terraform state push recovered-state.json

# Step 6: Verify infrastructure matches
terraform plan
# Should show minimal or no changes

# Step 7: If plan looks good, you're recovered!
```

**Scenario: Lost state file entirely**

If you lose state AND backups (very rare with remote state), you can rebuild:

```bash
# Step 1: Start with empty state
rm terraform.tfstate terraform.tfstate.backup

# Step 2: Import existing resources one by one
terraform import aws_instance.ml_server i-0abc123def456789
terraform import aws_s3_bucket.datasets my-ml-datasets-bucket
terraform import aws_security_group.ml_sg sg-0def456ghi789jkl0
# ... import all resources

# Step 3: Verify
terraform plan
# Should show "No changes" or only minor differences

# This is tedious - that's why remote state + backups are critical!
```

## Real-World ML Infrastructure State Management

### Multi-Region GPU Training Platform

```hcl
# State organized by region and environment
# backend.tf

terraform {
  backend "s3" {
    bucket         = "ml-platform-terraform-state"
    key            = "${var.region}/${terraform.workspace}/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "ml-platform-state-locks"
    encrypt        = true
  }
}

# State file paths:
# us-west-2/production/terraform.tfstate
# us-east-1/production/terraform.tfstate
# eu-west-1/production/terraform.tfstate
# us-west-2/dev/terraform.tfstate
```

### Handling Large State Files

**Problem:** State file with 500+ resources is slow.

**Solution 1: Split into multiple state files**
```
ml-platform/
├── networking/           # VPC, subnets, etc.
│   └── terraform.tfstate
├── compute/             # EC2, EKS clusters
│   └── terraform.tfstate
├── storage/             # S3, EFS, RDS
│   └── terraform.tfstate
└── monitoring/          # CloudWatch, Prometheus
    └── terraform.tfstate
```

**Solution 2: Use data sources for cross-state references**
```hcl
# In compute/ directory
data "terraform_remote_state" "networking" {
  backend = "s3"

  config = {
    bucket = "ml-platform-terraform-state"
    key    = "networking/production/terraform.tfstate"
    region = "us-west-2"
  }
}

# Use outputs from networking state
resource "aws_instance" "ml_server" {
  subnet_id              = data.terraform_remote_state.networking.outputs.private_subnet_id
  vpc_security_group_ids = [data.terraform_remote_state.networking.outputs.ml_sg_id]
}
```

### State Management in CI/CD

**GitHub Actions workflow:**
```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  pull_request:
    paths:
      - 'infrastructure/**'
  push:
    branches:
      - main

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2

      - name: Terraform Init
        run: terraform init
        working-directory: infrastructure

      - name: Terraform Plan
        id: plan
        run: terraform plan -no-color
        working-directory: infrastructure
        continue-on-error: true

      - name: Comment PR with Plan
        uses: actions/github-script@v6
        if: github.event_name == 'pull_request'
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const output = `#### Terraform Plan 📖\`${{ steps.plan.outcome }}\`

            <details><summary>Show Plan</summary>

            \`\`\`
            ${{ steps.plan.outputs.stdout }}
            \`\`\`

            </details>`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: output
            })

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: terraform apply -auto-approve
        working-directory: infrastructure
```

## Best Practices Checklist

- ✅ Always use remote state for team projects
- ✅ Enable S3 versioning for state files
- ✅ Enable encryption at rest (S3, DynamoDB)
- ✅ Use DynamoDB for state locking
- ✅ Restrict access to state files with IAM
- ✅ Never commit state files to version control
- ✅ Use unique state file paths per environment
- ✅ Back up state files regularly
- ✅ Test disaster recovery procedures
- ✅ Use workspaces OR separate directories for environments
- ✅ Monitor state file access (S3 logging)
- ✅ Use `terraform state` commands instead of manual edits
- ✅ Review state file changes in pull requests
- ✅ Document state file organization
- ✅ Plan disaster recovery procedures

## Common Mistakes to Avoid

### ❌ Mistake 1: Committing State to Git

```bash
# .gitignore
*.tfstate
*.tfstate.*
*.tfstate.backup

# ALWAYS ignore state files!
```

### ❌ Mistake 2: Manually Editing State

```bash
# ❌ NEVER DO THIS
vim terraform.tfstate  # Editing state file directly

# ✅ DO THIS
terraform state rm aws_instance.old_server
terraform state mv aws_instance.a aws_instance.b
```

### ❌ Mistake 3: Using Same State for Multiple Environments

```hcl
# ❌ WRONG: Same state file for dev and prod
terraform {
  backend "s3" {
    bucket = "terraform-state"
    key    = "infrastructure.tfstate"  # Same for all!
  }
}

# ✅ CORRECT: Different state per environment
terraform {
  backend "s3" {
    bucket = "terraform-state"
    key    = "infrastructure/${var.environment}.tfstate"
  }
}
```

### ❌ Mistake 4: Ignoring Locking Errors

```bash
# ❌ DON'T
terraform apply -lock=false  # Skips locking!

# ✅ DO
# Wait for other user to finish, or investigate stuck lock
```

### ❌ Mistake 5: No Backups

Set up automated backups and test restoration regularly!

## Key Takeaways

✅ State is Terraform's memory of your infrastructure
✅ Remote state enables team collaboration and safety
✅ S3 + DynamoDB is the standard backend for AWS
✅ State locking prevents concurrent modifications
✅ Use `terraform state` commands, never edit state manually
✅ Workspaces manage multiple environments with one config
✅ Encrypt state files (S3 encryption, KMS)
✅ Always have backups and disaster recovery plan
✅ Restrict access to state files (IAM, bucket policies)
✅ Use separate state files for production isolation

## Next Steps

Now that you've mastered state management, you're ready to:

- **Lesson 04**: Build complete AI infrastructure with Terraform
- **Lesson 05**: Explore Pulumi for Python-based infrastructure
- **Lesson 06**: Learn advanced IaC patterns and modules

---

**Next Lesson**: [04-building-ai-infrastructure-terraform.md](04-building-ai-infrastructure-terraform.md)
