# Lab 03: Remote State Backend with S3 + DynamoDB Locking

**Duration:** 45 min  **Prerequisites:** AWS CLI, Terraform

## Objective
Move Terraform state from local disk to S3, add DynamoDB-based state locking, and verify two concurrent applies are properly serialized.

## Steps

### 1. Bootstrap the backend resources
First time, you can't yet store the bootstrap state in S3 (chicken-and-egg). Apply with local state, then migrate.
```hcl
# bootstrap/main.tf
resource "aws_s3_bucket" "tfstate" { bucket = "company-tfstate-${random_string.suffix.result}" }
resource "random_string" "suffix"  { length = 6 special = false upper = false }
resource "aws_s3_bucket_versioning" "v" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "enc" {
  bucket = aws_s3_bucket.tfstate.id
  rule { apply_server_side_encryption_by_default { sse_algorithm = "AES256" } }
}
resource "aws_dynamodb_table" "locks" {
  name = "tfstate-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key = "LockID"
  attribute { name = "LockID" type = "S" }
}
output "bucket" { value = aws_s3_bucket.tfstate.id }
```
```bash
cd bootstrap && terraform init && terraform apply
BUCKET=$(terraform output -raw bucket)
```

### 2. Configure backend in your project
```hcl
# main.tf
terraform {
  backend "s3" {
    bucket         = "company-tfstate-abc123"      # from bootstrap output
    key            = "ml/lab/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "tfstate-locks"
    encrypt        = true
  }
}
```

### 3. Migrate existing local state
```bash
terraform init -migrate-state
# Confirm: yes
ls -la terraform.tfstate     # gone — state now in S3
```

### 4. Test locking
In terminal A:
```bash
terraform apply              # type "yes" but don't hit enter yet
```
In terminal B, in another checkout of the same project:
```bash
terraform plan               # should fail with "Error: Error acquiring the state lock"
```

### 5. Inspect the lock
```bash
aws dynamodb scan --table-name tfstate-locks
```
Releases when terminal A finishes.

### 6. Verify versioning
```bash
aws s3api list-object-versions --bucket $BUCKET --prefix ml/lab/
```
Each apply produces a new version. To roll back, restore an older version.

## Validation
- [ ] State file in S3, not local disk.
- [ ] Concurrent apply attempts are blocked with a clear error.
- [ ] Multiple state versions visible in S3.

## Cleanup
```bash
# In your project:
terraform destroy
# In bootstrap (only after no projects use the state any more):
cd bootstrap && terraform destroy
```

## Troubleshooting
- **`Failed to get existing workspaces: AccessDenied`** — IAM permissions; need `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` on bucket and `dynamodb:GetItem`/`PutItem`/`DeleteItem` on table.
- **`Error: state snapshot was created by Terraform vX.Y, which is newer`** — Local TF version older than the version that created the state. Upgrade.
- **State versioning not on** — Add `aws_s3_bucket_versioning` to bootstrap. Once on, every save creates a version.
