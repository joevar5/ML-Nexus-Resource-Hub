# Lesson 08: Security Best Practices for IaC

## Learning Objectives

By the end of this lesson, you will:

- Manage secrets securely (AWS Secrets Manager, HashiCorp Vault)
- Prevent credential leaks in version control
- Encrypt sensitive data in Terraform state
- Scan infrastructure for security issues (tfsec, Checkov, Terrascan)
- Implement least privilege IAM policies
- Estimate and optimize infrastructure costs (Infracost)
- Apply effective tagging strategies for cost allocation
- Document infrastructure effectively
- Plan disaster recovery procedures
- Establish team collaboration and code review guidelines

## Secret Management

### The Problem

```hcl
# ❌ NEVER DO THIS - Secrets in code
resource "aws_db_instance" "main" {
  username = "admin"
  password = "SuperSecret123!"  # Hardcoded password!
}

# This will be:
# 1. Visible in Git history forever
# 2. Stored in Terraform state (plain text)
# 3. Visible in plan output
# 4. Accessible to anyone with repo access
```

### AWS Secrets Manager

**Create secret:**
```bash
# Create secret in AWS
aws secretsmanager create-secret \
  --name ml-platform/db-password \
  --secret-string "$(openssl rand -base64 32)"

# Output:
# {
#   "ARN": "arn:aws:secretsmanager:us-west-2:123456789012:secret:ml-platform/db-password-AbCdEf",
#   "Name": "ml-platform/db-password",
#   "VersionId": "abc123-def456-ghi789"
# }
```

**Use in Terraform:**
```hcl
# Data source to retrieve secret
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = "ml-platform/db-password"
}

resource "aws_db_instance" "main" {
  identifier = "ml-database"
  engine     = "postgres"
  
  username = "ml_admin"
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
  
  # Secret still ends up in state, but not in code!
}

# Better: Use IAM authentication (no passwords)
resource "aws_db_instance" "main" {
  identifier = "ml-database"
  engine     = "postgres"
  
  iam_database_authentication_enabled = true
  # No password needed!
}
```

**Rotate secrets automatically:**
```hcl
resource "aws_secretsmanager_secret" "db_password" {
  name                    = "ml-platform/db-password"
  recovery_window_in_days = 30
}

resource "aws_secretsmanager_secret_rotation" "db_password" {
  secret_id           = aws_secretsmanager_secret.db_password.id
  rotation_lambda_arn = aws_lambda_function.rotate_secret.arn

  rotation_rules {
    automatically_after_days = 30
  }
}

# Lambda function to rotate secret
resource "aws_lambda_function" "rotate_secret" {
  filename      = "rotate_secret.zip"
  function_name = "rotate-db-secret"
  role          = aws_iam_role.lambda_rotation.arn
  handler       = "index.handler"
  runtime       = "python3.11"
}
```

### HashiCorp Vault

**Setup Vault provider:**
```hcl
provider "vault" {
  address = "https://vault.company.com"
  token   = var.vault_token  # From environment variable
}

# Read secret from Vault
data "vault_generic_secret" "db_credentials" {
  path = "secret/ml-platform/database"
}

resource "aws_db_instance" "main" {
  username = data.vault_generic_secret.db_credentials.data["username"]
  password = data.vault_generic_secret.db_credentials.data["password"]
}

# Dynamic secrets (Vault generates temporary credentials)
data "vault_aws_access_credentials" "ml_app" {
  backend = "aws"
  role    = "ml-application"
}

provider "aws" {
  access_key = data.vault_aws_access_credentials.ml_app.access_key
  secret_key = data.vault_aws_access_credentials.ml_app.secret_key
  # Credentials expire automatically!
}
```

### Environment Variables

```bash
# Set secrets as environment variables
export TF_VAR_db_password="$(aws secretsmanager get-secret-value --secret-id ml-platform/db-password --query SecretString --output text)"
export TF_VAR_api_key="$(cat ~/.secrets/api_key)"

# Terraform automatically uses TF_VAR_* variables
terraform apply
```

**variables.tf:**
```hcl
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true  # Masks value in output
}

variable "api_key" {
  description = "API key"
  type        = string
  sensitive   = true
}
```

### Preventing Credential Leaks

**.gitignore:**
```
# Terraform files
*.tfstate
*.tfstate.*
*.tfstate.backup
.terraform/
.terraform.lock.hcl

# Variable files with secrets
*.tfvars
*.tfvars.json
secrets.auto.tfvars

# Environment files
.env
.env.*
!.env.example

# Keys
*.pem
*.key
*.crt

# Vault tokens
.vault-token

# AWS credentials
.aws/credentials

# Backup files
*.backup
```

**pre-commit hooks (detect secrets):**

Install:
```bash
pip install pre-commit detect-secrets
```

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package.lock.json

  - repo: https://github.com/antonbabenko/pre-commit-terraform
    rev: v1.83.0
    hooks:
      - id: terraform_fmt
      - id: terraform_validate
      - id: terraform_tflint
```

Setup:
```bash
# Initialize pre-commit
pre-commit install

# Create baseline of existing secrets
detect-secrets scan > .secrets.baseline

# Now commits are blocked if secrets detected
git add main.tf
git commit -m "Add database"
# If password in code: commit blocked!
```

**git-secrets (AWS):**
```bash
# Install git-secrets
brew install git-secrets

# Setup in repo
cd ml-infrastructure
git secrets --install
git secrets --register-aws

# Scan for secrets
git secrets --scan

# Scan entire history
git secrets --scan-history
```

## State File Encryption

### S3 Backend Encryption

```hcl
# Backend configuration with encryption
terraform {
  backend "s3" {
    bucket         = "ml-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true  # Encrypt at rest
    kms_key_id     = "arn:aws:kms:us-west-2:123456789012:key/abc123"
    dynamodb_table = "terraform-locks"
  }
}

# S3 bucket configuration
resource "aws_s3_bucket" "terraform_state" {
  bucket = "ml-terraform-state"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.terraform_state.arn
    }
  }
}

# KMS key for state encryption
resource "aws_kms_key" "terraform_state" {
  description             = "Terraform state encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::123456789012:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Terraform to use the key"
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::123456789012:role/TerraformRole",
            "arn:aws:iam::123456789012:user/terraform"
          ]
        }
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
}
```

### Terraform Cloud Encryption

```hcl
terraform {
  cloud {
    organization = "my-company"

    workspaces {
      name = "ml-production"
    }
  }
}

# Terraform Cloud features:
# - State encryption at rest (AES-256)
# - State encryption in transit (TLS)
# - State versioning
# - Access controls
# - Audit logs
```

### Sensitive Outputs

```hcl
# Mark outputs as sensitive
output "db_password" {
  description = "Database password"
  value       = random_password.db.result
  sensitive   = true  # Won't show in console output
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = aws_api_gateway_stage.prod.invoke_url
  # Not sensitive, will show in output
}

# Access sensitive outputs
terraform output db_password
# (sensitive value)

# Show sensitive output
terraform output -raw db_password
# actual-password-here
```

## Security Scanning

### tfsec Configuration

**.tfsec/config.yml:**
```yaml
severity_overrides:
  aws-s3-enable-bucket-encryption: HIGH
  aws-ec2-no-public-ingress-sgr: CRITICAL

exclude:
  - aws-s3-enable-bucket-logging  # We use CloudTrail instead

minimum_severity: MEDIUM
```

**Custom checks:**
```yaml
checks:
  - code: ML-001
    description: Ensure GPU instances have monitoring enabled
    impact: Cannot track GPU utilization
    resolution: Enable detailed monitoring
    requiredTypes:
      - resource
    requiredLabels:
      - aws_instance
    severity: WARNING
    matchSpec:
      name: instance_type
      action: startsWith
      value: p3
    checkBlock:
      - or:
        - name: monitoring
          value: true
```

### Checkov Custom Policies

**checkov/gpu_monitoring.py:**
```python
from checkov.common.models.enums import CheckResult
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

class GPUMonitoringEnabled(BaseResourceCheck):
    def __init__(self):
        name = "Ensure GPU instances have detailed monitoring enabled"
        id = "CKV_AWS_ML_1"
        supported_resources = ['aws_instance']
        categories = ['monitoring']
        super().__init__(name=name, id=id, categories=categories, 
                         supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        # Check if instance type is GPU
        instance_type = conf.get('instance_type', [''])[0]
        if not any(x in instance_type for x in ['p3', 'p4', 'g4', 'g5']):
            return CheckResult.UNKNOWN
        
        # Check if monitoring is enabled
        monitoring = conf.get('monitoring', [False])[0]
        if monitoring:
            return CheckResult.PASSED
        return CheckResult.FAILED

check = GPUMonitoringEnabled()
```

### Terrascan

**Install:**
```bash
brew install terrascan

# Or
curl -L "$(curl -s https://api.github.com/repos/tenable/terrascan/releases/latest | grep -o -E 'https://.+?_Darwin_x86_64.tar.gz')" > terrascan.tar.gz
tar -xf terrascan.tar.gz terrascan && rm terrascan.tar.gz
sudo mv terrascan /usr/local/bin/
```

**Run:**
```bash
# Scan Terraform
terrascan scan -t terraform

# Scan specific directory
terrascan scan -d infrastructure/

# Output formats
terrascan scan -o json
terrascan scan -o junit-xml

# Skip rules
terrascan scan --skip-rules="AWS.S3Bucket.DS.High.1043,AWS.Instance.NetworkSecurity.Medium.0506"

# Scan with specific policies
terrascan scan --policy-path ./policies/
```

**terrascan-config.toml:**
```toml
[rules]
  skip-rules = [
    "AWS.S3Bucket.LM.LOW.0078",
    "AWS.CloudTrail.Logging.Medium.007"
  ]

[severity]
  high = ["AWS.*.DS.High.*"]
  medium = ["AWS.*.NetworkSecurity.Medium.*"]
  low = ["AWS.*.LM.LOW.*"]
```

## IAM Least Privilege

### Granular IAM Policies

```hcl
# ❌ Too permissive
resource "aws_iam_policy" "bad_policy" {
  name = "ml-training-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:*"  # All S3 actions!
      Resource = "*"     # All resources!
    }]
  })
}

# ✅ Least privilege
resource "aws_iam_policy" "good_policy" {
  name = "ml-training-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadDatasets"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.datasets.arn,
          "${aws_s3_bucket.datasets.arn}/*"
        ]
      },
      {
        Sid    = "WriteModels"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl"
        ]
        Resource = "${aws_s3_bucket.models.arn}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-server-side-encryption" = "AES256"
          }
        }
      },
      {
        Sid    = "WriteMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "ML/Training"
          }
        }
      }
    ]
  })
}
```

### Assume Role Pattern

```hcl
# Cross-account access with assume role
resource "aws_iam_role" "ml_training_cross_account" {
  name = "ml-training-cross-account"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        AWS = "arn:aws:iam::111111111111:root"  # Source account
      }
      Action = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "sts:ExternalId" = var.external_id
        }
      }
    }]
  })
}

# Attach permissions
resource "aws_iam_role_policy_attachment" "ml_training" {
  role       = aws_iam_role.ml_training_cross_account.name
  policy_arn = aws_iam_policy.ml_training_policy.arn
}
```

### Service Control Policies (SCPs)

```hcl
# Prevent deletion of GPU instances
resource "aws_organizations_policy" "prevent_gpu_deletion" {
  name        = "PreventGPUDeletion"
  description = "Prevent deletion of GPU training instances"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Deny"
      Action = [
        "ec2:TerminateInstances"
      ]
      Resource = "*"
      Condition = {
        StringLike = {
          "ec2:InstanceType" = [
            "p3.*",
            "p4d.*",
            "g4dn.*"
          ]
        }
        StringNotEquals = {
          "aws:PrincipalArn" = [
            "arn:aws:iam::*:role/AdminRole"
          ]
        }
      }
    }]
  })
}

resource "aws_organizations_policy_attachment" "ml_ou" {
  policy_id = aws_organizations_policy.prevent_gpu_deletion.id
  target_id = aws_organizations_organizational_unit.ml.id
}
```

## Cost Optimization

### Infracost Integration

**Install:**
```bash
brew install infracost

# Or
curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh | sh

# Register (free)
infracost auth login
```

**infracost.yml:**
```yaml
version: 0.1

projects:
  - path: infrastructure/environments/production
    name: ml-platform-production
    usage_file: usage-production.yml

  - path: infrastructure/environments/staging
    name: ml-platform-staging
    usage_file: usage-staging.yml
```

**usage-production.yml (accurate cost estimation):**
```yaml
version: 0.1

resource_usage:
  aws_instance.gpu_training:
    operating_system: linux
    monthly_hrs: 730  # 24/7 operation
    vcpu_count: 32
    
  aws_instance.gpu_training[0]:
    monthly_hrs: 500  # Only during business hours
    
  aws_s3_bucket.datasets:
    storage_gb: 10000  # 10 TB
    monthly_tier1_requests: 1000000  # 1M requests
    monthly_tier2_requests: 500000
    monthly_select_data_scanned_gb: 5000
    monthly_data_at_rest_gb: 10000
```

**Run Infracost:**
```bash
# Breakdown
infracost breakdown --path infrastructure/

# Output:
# Name                                      Monthly Qty  Unit   Monthly Cost
# 
# aws_instance.gpu_training
# └─ Instance usage (Linux/UNIX, on-demand)         730  hours      $2,236.80
# └─ root_block_device
#    └─ Storage (general purpose SSD, gp3)          500  GB            $40.00
#
# aws_s3_bucket.datasets
# └─ Standard
#    ├─ Storage                                  10,000  GB           $230.00
#    ├─ PUT, COPY, POST, LIST requests        1,000,000  requests       $5.00
#    └─ GET, SELECT requests                    500,000  requests       $0.20
#
# OVERALL TOTAL                                                    $2,512.00

# Compare
infracost diff --path infrastructure/

# Output shows cost changes
```

**GitHub Actions:**
```yaml
      - name: Setup Infracost
        uses: infracost/actions/setup@v2
        with:
          api-key: ${{ secrets.INFRACOST_API_KEY }}

      - name: Generate Infracost diff
        run: |
          infracost breakdown --path=. \
            --format=json \
            --out-file=/tmp/infracost-base.json
          
          infracost diff --path=. \
            --compare-to=/tmp/infracost-base.json \
            --format=json \
            --out-file=/tmp/infracost.json

      - name: Post Infracost comment
        uses: infracost/actions/comment@v1
        with:
          path: /tmp/infracost.json
          behavior: update
```

### Cost Allocation Tags

```hcl
# Consistent tagging strategy
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    CostCenter  = var.cost_center
    Owner       = var.owner_email
    Application = "ML-Platform"
  }

  # Resource-specific tags
  training_tags = merge(local.common_tags, {
    Workload = "Training"
    Purpose  = "Model-Training"
  })

  inference_tags = merge(local.common_tags, {
    Workload = "Inference"
    Purpose  = "Model-Serving"
  })
}

# Apply tags
resource "aws_instance" "training" {
  count = 10
  
  instance_type = "p3.8xlarge"
  
  tags = merge(
    local.training_tags,
    {
      Name  = "training-${count.index}"
      Index = tostring(count.index)
    }
  )
}

# Default tags (provider level)
provider "aws" {
  region = "us-west-2"

  default_tags {
    tags = local.common_tags
  }
}
```

**Cost allocation report:**
```bash
# AWS CLI
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=TAG,Key=Workload

# Output:
# Training: $15,230.45
# Inference: $8,456.23
# Total: $23,686.68
```

### Budget Alerts

```hcl
resource "aws_budgets_budget" "ml_platform" {
  name         = "ml-platform-monthly"
  budget_type  = "COST"
  limit_amount = "10000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filters = {
    TagKeyValue = [
      "Project$ML-Platform",
      "ManagedBy$Terraform"
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
```

## Documentation Best Practices

### Module Documentation

**README.md template:**
```markdown
# GPU Training Cluster Module

## Overview

Creates an auto-scaling GPU training cluster for machine learning workloads.

## Features

- Auto Scaling Group with GPU instances (p3, p4d, g4dn)
- Spot instance support for 70% cost savings
- Automatic S3 data synchronization
- CloudWatch monitoring and alerting
- Cost-optimized with scheduled scaling

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

| Name | Version |
|------|---------|
| terraform | >= 1.0 |
| aws | >= 5.0 |

## Providers

| Name | Version |
|------|---------|
| aws | >= 5.0 |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| cluster_name | Name of the GPU cluster | string | - | yes |
| instance_type | EC2 instance type | string | p3.2xlarge | no |
| instance_count | Number of instances | number | 2 | no |
| use_spot_instances | Use spot instances | bool | true | no |
| vpc_id | VPC ID | string | - | yes |
| subnet_ids | Subnet IDs | list(string) | - | yes |

## Outputs

| Name | Description |
|------|-------------|
| cluster_name | GPU cluster name |
| security_group_id | Security group ID |
| autoscaling_group_name | ASG name |

## Examples

See [examples/](./examples/) for complete usage examples.

## Cost Estimate

~$2,450/month (4x p3.8xlarge instances, spot pricing)

## Security Considerations

- Instances in private subnets
- Security group restricts SSH to VPC
- IAM role follows least privilege
- EBS volumes encrypted at rest

## Maintenance

- AMI updates: Monthly
- Security patches: Automated via user-data
- Scaling review: Quarterly

## Support

Contact: ml-platform-team@company.com
```

### terraform-docs

**Install:**
```bash
brew install terraform-docs

# Or
go install github.com/terraform-docs/terraform-docs@latest
```

**Generate documentation:**
```bash
# Generate README
terraform-docs markdown table . > README.md

# Custom template
terraform-docs markdown document \
  --output-file README.md \
  --output-mode inject \
  .
```

**.terraform-docs.yml:**
```yaml
formatter: markdown table

sections:
  show:
    - header
    - requirements
    - providers
    - inputs
    - outputs
    - resources

output:
  file: README.md
  mode: inject
  template: |-
    <!-- BEGIN_TF_DOCS -->
    {{ .Content }}
    <!-- END_TF_DOCS -->

sort:
  enabled: true
  by: required
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup-infrastructure.sh

DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="./backups/$DATE"

mkdir -p $BACKUP_DIR

# 1. Export Terraform state
terraform state pull > $BACKUP_DIR/terraform.tfstate

# 2. Export all .tf files
tar -czf $BACKUP_DIR/terraform-code.tar.gz *.tf

# 3. Export AWS resources
aws ec2 describe-instances > $BACKUP_DIR/ec2-instances.json
aws s3api list-buckets > $BACKUP_DIR/s3-buckets.json
aws rds describe-db-instances > $BACKUP_DIR/rds-instances.json

# 4. Export configurations
aws secretsmanager list-secrets > $BACKUP_DIR/secrets.json

# 5. Upload to S3
aws s3 cp $BACKUP_DIR s3://disaster-recovery-backups/$DATE/ --recursive

echo "Backup complete: $BACKUP_DIR"
```

### Recovery Procedure

**disaster-recovery.md:**
```markdown
# Disaster Recovery Procedure

## Scenario 1: Lost State File

1. Check S3 versioning for state file
2. Download previous version
3. terraform state push <backup.tfstate>
4. Verify: terraform plan (should show no changes)

## Scenario 2: Accidental Resource Deletion

1. Import deleted resource:
   terraform import aws_instance.gpu i-0abc123def456
2. Verify configuration matches
3. terraform plan (should show no changes)

## Scenario 3: Complete Infrastructure Loss

1. Restore code from Git
2. Restore state from backup
3. Review and update provider versions
4. terraform init
5. terraform plan (review changes)
6. terraform apply

## Scenario 4: Corrupted State

1. terraform state pull > current.tfstate
2. Review with: jq . current.tfstate
3. If corrupted, restore from S3 version
4. terraform state push backup.tfstate
5. Verify: terraform plan

## Emergency Contacts

- Infrastructure Lead: john@company.com
- DevOps Team: devops@company.com
- On-Call: PagerDuty #ML-Platform

## RTO/RPO

- Recovery Time Objective (RTO): 4 hours
- Recovery Point Objective (RPO): 1 hour
```

## Team Collaboration

### Code Review Guidelines

**.github/PULL_REQUEST_TEMPLATE.md:**
```markdown
## Infrastructure Change Request

### Description
Brief description of the infrastructure changes

### Type of Change
- [ ] New resource
- [ ] Resource modification
- [ ] Resource deletion
- [ ] Configuration change
- [ ] Security update

### Checklist
- [ ] terraform fmt applied
- [ ] terraform validate passed
- [ ] tflint passed
- [ ] Security scan passed (tfsec/Checkov)
- [ ] Cost impact reviewed (Infracost)
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Reviewed blast radius

### Cost Impact
<!-- Infracost will comment with cost changes -->

### Security Impact
- [ ] No new security groups rules
- [ ] No public access added
- [ ] No secrets in code
- [ ] IAM follows least privilege

### Testing
How was this tested?
- [ ] Applied in dev environment
- [ ] Applied in staging environment
- [ ] Verified resource creation
- [ ] Verified application functionality

### Rollback Plan
Describe rollback procedure if this change causes issues

### Screenshots
If applicable, add screenshots of terraform plan output

### Additional Context
Any other context about the changes
```

### CODEOWNERS

**.github/CODEOWNERS:**
```
# Infrastructure team reviews all changes
* @ml-platform/infrastructure-team

# Security team reviews security-related changes
**/security-groups.tf @ml-platform/security-team
**/iam.tf @ml-platform/security-team
**/kms.tf @ml-platform/security-team

# Cost team reviews expensive resources
**/gpu-*.tf @ml-platform/cost-optimization-team
**/eks-*.tf @ml-platform/cost-optimization-team

# Production requires senior approval
/environments/production/ @ml-platform/senior-engineers
```

### Branch Protection Rules

Required via GitHub UI:
```
Branch: main

Required:
☑ Require pull request reviews (2)
☑ Dismiss stale reviews
☑ Require review from Code Owners
☑ Require status checks to pass
  - terraform-validate
  - terraform-plan
  - tfsec
  - checkov
☑ Require branches to be up to date
☑ Require conversation resolution
☑ Restrict who can push
☑ Require signed commits
```

## Complete Security Checklist

### Code Security
- [ ] No secrets in version control
- [ ] .gitignore configured properly
- [ ] pre-commit hooks installed
- [ ] git-secrets configured

### Secret Management
- [ ] Secrets in AWS Secrets Manager/Vault
- [ ] Automatic secret rotation enabled
- [ ] State files encrypted (KMS)
- [ ] Sensitive outputs marked

### IAM Security
- [ ] Least privilege policies
- [ ] No wildcard permissions
- [ ] MFA required for production
- [ ] Assume role for cross-account
- [ ] Service Control Policies applied

### Resource Security
- [ ] Security groups restrictive
- [ ] Resources in private subnets
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enforced
- [ ] Logging enabled

### Scanning
- [ ] tfsec runs on every PR
- [ ] Checkov runs on every PR
- [ ] Terrascan in CI/CD
- [ ] OPA policies enforced
- [ ] Dependency scanning

### Cost Management
- [ ] Infracost in PR reviews
- [ ] Tags for cost allocation
- [ ] Budget alerts configured
- [ ] Spot instances where appropriate
- [ ] Regular cost reviews

### Documentation
- [ ] README for each module
- [ ] Architecture diagrams
- [ ] Disaster recovery procedures
- [ ] Runbooks for operations
- [ ] Change log maintained

### Compliance
- [ ] SOC 2 requirements met
- [ ] HIPAA compliance (if applicable)
- [ ] GDPR compliance (if applicable)
- [ ] Audit logs enabled
- [ ] Retention policies configured

## Key Takeaways

✅ Never commit secrets to version control
✅ Use AWS Secrets Manager or Vault for secrets
✅ Encrypt state files with KMS
✅ Security scan every PR (tfsec, Checkov)
✅ Apply least privilege IAM policies
✅ Use Infracost for cost visibility
✅ Tag everything for cost allocation
✅ Document infrastructure thoroughly
✅ Test disaster recovery procedures
✅ Enforce code reviews and approvals

## Conclusion

Security and best practices are not optional—they're essential for production infrastructure. Following these guidelines will help you build secure, cost-effective, and maintainable ML infrastructure.

---

**Course Complete!** You've mastered Infrastructure as Code for AI/ML platforms.

**Next Steps:**
- Apply these concepts to real projects
- Contribute to open-source IaC modules
- Explore advanced topics (service mesh, GitOps operators)
- Stay updated with cloud provider changes
