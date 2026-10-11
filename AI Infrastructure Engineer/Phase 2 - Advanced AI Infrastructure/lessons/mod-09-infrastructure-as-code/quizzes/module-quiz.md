# Module 09: Infrastructure as Code - Quiz

Test your understanding of Infrastructure as Code concepts, Terraform, state management, GitOps, and IaC security for AI/ML infrastructure for AI/ML infrastructure.

**Time:** 40 minutes
**Questions:** 21 multiple choice + 2 scenario-based questions
**Passing Score:** 70% (17/23 correct)

---

## Multiple Choice Questions

### Question 1: Terraform State

What is the purpose of Terraform state?

A) To store API credentials securely
B) To track the current state of managed infrastructure
C) To define what resources should be created
D) To execute shell commands during deployment

<details>
<summary>Answer</summary>

**B) To track the current state of managed infrastructure**

Explanation: Terraform state is a JSON file that maps your configuration to real-world resources. It tracks resource IDs, attributes, and dependencies to enable Terraform to plan updates correctly.
</details>

---

### Question 2: Remote State Backend

Why should you use a remote state backend (like S3) instead of local state?

A) It's faster to run terraform commands
B) It enables team collaboration and prevents concurrent modifications
C) It's required for all Terraform projects
D) It automatically backs up your infrastructure

<details>
<summary>Answer</summary>

**B) It enables team collaboration and prevents concurrent modifications**

Explanation: Remote state backends enable multiple team members to work on the same infrastructure safely. With state locking (e.g., DynamoDB), concurrent modifications are prevented. Local state only works for individual developers.
</details>

---

### Question 3: Terraform Resource Dependencies

How does Terraform determine the order in which to create resources?

A) Alphabetically by resource name
B) In the order they appear in the .tf files
C) By analyzing implicit and explicit dependencies
D) Randomly, then retries if there are errors

<details>
<summary>Answer</summary>

**C) By analyzing implicit and explicit dependencies**

Explanation: Terraform builds a dependency graph by analyzing resource references (implicit) and `depends_on` declarations (explicit). It creates resources in the correct order automatically.
</details>

---

### Question 4: HCL Variables

What is the correct way to reference a variable named `instance_type` in a Terraform resource?

A) `${instance_type}`
B) `var.instance_type`
C) `variable.instance_type`
D) `$instance_type`

<details>
<summary>Answer</summary>

**B) var.instance_type**

Explanation: In HCL, variables are referenced using the `var.` prefix followed by the variable name. The `${}` interpolation syntax is used within strings, but the correct reference is still `var.instance_type`.
</details>

---

### Question 5: count vs for_each

When should you use `for_each` instead of `count` in Terraform?

A) When you need exactly one resource
B) When you want to create resources from a map or set and need stable identifiers
C) When you want resources created in alphabetical order
D) Only when using modules

<details>
<summary>Answer</summary>

**B) When you want to create resources from a map or set and need stable identifiers**

Explanation: `for_each` is preferred over `count` when you need resources to be identified by keys rather than indices. This prevents unintended resource destruction when lists change order.
</details>

---

### Question 6: GPU Instance Provisioning

Which Terraform resource meta-argument would you use to provision expensive GPU instances only in production?

A) `lifecycle { prevent_destroy = true }`
B) `count = var.environment == "prod" ? 3 : 0`
C) `depends_on = [var.environment]`
D) `for_each = var.gpu_instances`

<details>
<summary>Answer</summary>

**B) count = var.environment == "prod" ? 3 : 0**

Explanation: Using a conditional expression with `count` allows you to create resources conditionally. When count is 0, no resources are created, saving costs in non-production environments.
</details>

---

### Question 7: Terraform Plan

What does `terraform plan` do?

A) Creates infrastructure resources
B) Shows a preview of changes without making them
C) Formats Terraform files
D) Initializes the working directory

<details>
<summary>Answer</summary>

**B) Shows a preview of changes without making them**

Explanation: `terraform plan` generates an execution plan showing what actions Terraform will take to reach the desired state. It's a dry-run that doesn't modify actual infrastructure.
</details>

---

### Question 8: Terraform State Locking

What is the purpose of state locking with DynamoDB in an S3 backend?

A) To encrypt the state file
B) To prevent concurrent terraform operations that could corrupt state
C) To backup state automatically
D) To version the state file

<details>
<summary>Answer</summary>

**B) To prevent concurrent terraform operations that could corrupt state**

Explanation: State locking prevents multiple team members from running `terraform apply` simultaneously, which could corrupt the state file. DynamoDB provides the locking mechanism for S3 backends.
</details>

---

### Question 9: Data Sources

What is a Terraform data source used for?

A) To create new infrastructure resources
B) To fetch information about existing resources
C) To store sensitive data securely
D) To define input variables

<details>
<summary>Answer</summary>

**B) To fetch information about existing resources**

Explanation: Data sources allow Terraform to query existing infrastructure (like AMIs, VPCs, or availability zones) and use that information in your configuration. They're read-only.
</details>

---

### Question 10: Spot Instances for ML Training

Why are spot instances beneficial for ML training workloads in Terraform?

A) They have better GPU performance
B) They're up to 90% cheaper than on-demand instances
C) They're guaranteed to never be interrupted
D) They have faster network speeds

<details>
<summary>Answer</summary>

**B) They're up to 90% cheaper than on-demand instances**

Explanation: Spot instances offer significant cost savings (up to 90% off) by using spare cloud capacity. They can be interrupted, but for fault-tolerant ML training with checkpointing, they're ideal.
</details>

---

### Question 11: Sensitive Outputs

How do you mark a Terraform output as sensitive to prevent it from being displayed in the CLI?

A) `output "password" { value = var.password }`
B) `output "password" { value = var.password, sensitive = true }`
C) `output "password" { value = var.password, hidden = true }`
D) `output "password" { value = var.password, encrypted = true }`

<details>
<summary>Answer</summary>

**B) output "password" { value = var.password, sensitive = true }**

Explanation: The `sensitive = true` argument prevents the output value from being displayed in the CLI or logs, protecting sensitive information like passwords or API keys.
</details>

---

### Question 12: Terraform Workspaces

What is the purpose of Terraform workspaces?

A) To run Terraform in different cloud providers
B) To manage multiple environments (dev, staging, prod) with the same configuration
C) To parallelize terraform operations
D) To version control Terraform code

<details>
<summary>Answer</summary>

**B) To manage multiple environments (dev, staging, prod) with the same configuration**

Explanation: Workspaces allow you to use the same Terraform configuration for multiple environments by maintaining separate state files. Each workspace has its own state.
</details>

---

### Question 13: GitOps for Infrastructure

What is a key principle of GitOps for infrastructure?

A) Infrastructure changes are made directly in the cloud console
B) Infrastructure changes go through Git pull requests and automated pipelines
C) Git is not used for infrastructure code
D) Only developers can modify infrastructure

<details>
<summary>Answer</summary>

**B) Infrastructure changes go through Git pull requests and automated pipelines**

Explanation: GitOps treats Git as the single source of truth. All infrastructure changes are proposed via pull requests, reviewed, and automatically applied through CI/CD pipelines.
</details>

---

### Question 14: terraform fmt

What does the `terraform fmt` command do?

A) Validates the configuration syntax
B) Formats code to canonical style
C) Creates an execution plan
D) Initializes the working directory

<details>
<summary>Answer</summary>

**B) Formats code to canonical style**

Explanation: `terraform fmt` automatically formats Terraform configuration files to a canonical style, ensuring consistency across your codebase. It's useful for code reviews and CI/CD.
</details>

---

### Question 15: Preventing Resource Destruction

Which lifecycle meta-argument prevents accidental destruction of critical resources?

A) `create_before_destroy = true`
B) `prevent_destroy = true`
C) `ignore_changes = all`
D) `destroy = false`

<details>
<summary>Answer</summary>

**B) prevent_destroy = true**

Explanation: Setting `prevent_destroy = true` in the lifecycle block causes Terraform to reject any plan that would destroy the resource, protecting critical infrastructure from accidental deletion.
</details>

---

### Question 16: IAM Roles for EC2

Why should ML training EC2 instances use IAM roles instead of hardcoded credentials?

A) IAM roles are faster
B) IAM roles provide temporary, automatically rotated credentials
C) IAM roles are required by AWS
D) IAM roles are cheaper

<details>
<summary>Answer</summary>

**B) IAM roles provide temporary, automatically rotated credentials**

Explanation: IAM roles for EC2 (via instance profiles) provide temporary credentials that are automatically rotated. This is more secure than hardcoding access keys and follows AWS best practices.
</details>

---

### Question 17: State File Security

Terraform state files can contain sensitive data. What's the best practice?

A) Commit state files to Git for backup
B) Use remote state with encryption (e.g., S3 with KMS)
C) Store state files on your local machine only
D) Email state files to team members

<details>
<summary>Answer</summary>

**B) Use remote state with encryption (e.g., S3 with KMS)**

Explanation: State files often contain sensitive information. Using remote state with encryption (S3 + KMS, Terraform Cloud, etc.) ensures security, backup, and team collaboration.
</details>

---

### Question 18: tfsec and Checkov

What do tfsec and Checkov do in an IaC workflow?

A) Format Terraform code
B) Scan infrastructure code for security issues and misconfigurations
C) Generate documentation
D) Deploy infrastructure

<details>
<summary>Answer</summary>

**B) Scan infrastructure code for security issues and misconfigurations**

Explanation: tfsec and Checkov are security scanning tools that analyze Terraform code for potential security issues, compliance violations, and best practice violations before deployment.
</details>

---

### Question 19: Auto-Scaling for Inference

Which Terraform resource would you use to auto-scale ML inference servers based on load?

A) `aws_instance`
B) `aws_autoscaling_group`
C) `aws_lambda_function`
D) `aws_ecs_task`

<details>
<summary>Answer</summary>

**B) aws_autoscaling_group**

Explanation: Auto Scaling Groups automatically adjust the number of EC2 instances based on demand. For ML inference, you can scale based on CPU, request count, or custom metrics.
</details>

---

### Question 20: Terraform Import

What does `terraform import` do?

A) Imports provider plugins
B) Brings existing infrastructure under Terraform management
C) Imports modules from the registry
D) Loads variable files

<details>
<summary>Answer</summary>

**B) Brings existing infrastructure under Terraform management**

Explanation: `terraform import` allows you to import existing cloud resources into Terraform state, so they can be managed by Terraform going forward. You still need to write the configuration manually.
</details>

---

### Question 21: Cost Optimization

Which tool can estimate the cost of infrastructure before applying Terraform changes?

A) terraform plan
B) Infracost
C) terraform validate
D) tfsec

<details>
<summary>Answer</summary>

**B) Infracost**

Explanation: Infracost analyzes Terraform code and provides cost estimates before deployment. It integrates with CI/CD to show cost impact of infrastructure changes in pull requests.
</details>

---

## Scenario-Based Questions

### Scenario 1: State File Corruption

**Context:**
Your team is working on ML infrastructure with Terraform. Two engineers accidentally ran `terraform apply` simultaneously, and now your state file is corrupted. Your production GPU cluster state is inconsistent with reality.

**Question:**
What steps should you take to recover and prevent this in the future?

**Your answer:**

<details>
<summary>Recommended Answer</summary>

**Recovery Steps:**

1. **Stop all Terraform operations immediately**
   ```bash
   # Communicate to team to halt all terraform commands
   ```

2. **Restore state from backup**
   ```bash
   # If using S3 backend with versioning
   aws s3api list-object-versions --bucket my-terraform-state --prefix prod/terraform.tfstate

   # Restore previous version
   aws s3api get-object --bucket my-terraform-state \
     --key prod/terraform.tfstate \
     --version-id <previous-version-id> \
     terraform.tfstate.restored
   ```

3. **Verify infrastructure**
   ```bash
   terraform plan
   # Check if plan matches expected state
   ```

4. **Refresh state if needed**
   ```bash
   terraform refresh
   # Updates state to match real infrastructure
   ```

**Prevention:**

1. **Enable state locking**
   ```hcl
   terraform {
     backend "s3" {
       bucket         = "my-terraform-state"
       key            = "prod/terraform.tfstate"
       region         = "us-west-2"
       dynamodb_table = "terraform-state-lock"  # State locking
       encrypt        = true
     }
   }
   ```

2. **Enable S3 versioning**
   ```hcl
   resource "aws_s3_bucket_versioning" "state" {
     bucket = aws_s3_bucket.terraform_state.id
     versioning_configuration {
       status = "Enabled"
     }
   }
   ```

3. **Implement CI/CD**
   - Use Atlantis or GitHub Actions
   - Only allow terraform apply through automated pipelines
   - Require PR approvals

4. **Team training**
   - Educate team on state management
   - Establish runbooks for common scenarios
</details>

---

### Scenario 2: Secret Management

**Context:**
You need to deploy an ML inference service that requires:
- Database password for storing predictions
- API key for external ML model API
- AWS access keys for S3 bucket access

The service runs on EC2 instances managed by Terraform.

**Question:**
How should you securely manage these secrets in your Terraform configuration?

**Your answer:**

<details>
<summary>Recommended Answer</summary>

**Secure Secret Management Approach:**

1. **Use AWS Secrets Manager for sensitive data**
   ```hcl
   # Create secrets in Secrets Manager
   resource "aws_secretsmanager_secret" "db_password" {
     name = "ml-inference-db-password"
   }

   resource "aws_secretsmanager_secret_version" "db_password" {
     secret_id     = aws_secretsmanager_secret.db_password.id
     secret_string = var.db_password  # Provided via TF_VAR or secure CI/CD
   }

   resource "aws_secretsmanager_secret" "api_key" {
     name = "ml-model-api-key"
   }

   resource "aws_secretsmanager_secret_version" "api_key" {
     secret_id     = aws_secretsmanager_secret.api_key.id
     secret_string = var.ml_api_key
   }
   ```

2. **Use IAM roles instead of access keys**
   ```hcl
   # Create IAM role with S3 permissions
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

   resource "aws_iam_role_policy" "s3_access" {
     role = aws_iam_role.ml_inference.id

     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [{
         Action   = ["s3:GetObject", "s3:PutObject"]
         Effect   = "Allow"
         Resource = "arn:aws:s3:::ml-predictions/*"
       }]
     })
   }

   # Grant permission to read secrets
   resource "aws_iam_role_policy" "secrets_access" {
     role = aws_iam_role.ml_inference.id

     policy = jsonencode({
       Version = "2012-10-17"
       Statement = [{
         Action   = ["secretsmanager:GetSecretValue"]
         Effect   = "Allow"
         Resource = [
           aws_secretsmanager_secret.db_password.arn,
           aws_secretsmanager_secret.api_key.arn
         ]
       }]
     })
   }

   # Attach role to instance
   resource "aws_iam_instance_profile" "ml_inference" {
     name = "ml-inference-profile"
     role = aws_iam_role.ml_inference.name
   }

   resource "aws_instance" "ml_inference" {
     ami                  = data.aws_ami.ml_ami.id
     instance_type        = "c5.2xlarge"
     iam_instance_profile = aws_iam_instance_profile.ml_inference.name

     user_data = <<-EOF
       #!/bin/bash
       # Application retrieves secrets at runtime
       DB_PASSWORD=$(aws secretsmanager get-secret-value \
         --secret-id ${aws_secretsmanager_secret.db_password.id} \
         --query SecretString --output text)

       API_KEY=$(aws secretsmanager get-secret-value \
         --secret-id ${aws_secretsmanager_secret.api_key.id} \
         --query SecretString --output text)

       # Start application with secrets from environment
       export DB_PASSWORD
       export API_KEY
       /opt/ml-service/start.sh
     EOF
   }
   ```

3. **Never commit secrets**
   ```hcl
   # .gitignore
   *.tfvars
   terraform.tfstate
   terraform.tfstate.backup
   .terraform/
   ```

4. **Use encrypted state**
   ```hcl
   terraform {
     backend "s3" {
       bucket  = "terraform-state"
       key     = "ml-inference/terraform.tfstate"
       region  = "us-west-2"
       encrypt = true
       kms_key_id = "arn:aws:kms:us-west-2:123456789:key/..."
     }
   }
   ```

**What NOT to do:**
- ❌ Hardcode secrets in .tf files
- ❌ Commit terraform.tfvars with secrets
- ❌ Use access keys when IAM roles are available
- ❌ Store secrets in plain text environment variables
- ❌ Leave state files unencrypted
</details>

---

## Scoring Guide

- 21 questions and 2 scenarios, 1 point each (23 points)
- Passing score: 70% (17/23)
