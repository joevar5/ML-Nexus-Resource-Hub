# Exercise 04: Terraform State Management at Scale

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 03; lab 03

## Objective

Design a state management strategy for 50+ Terraform projects across 4 environments. Implement remote backends with locking, per-project isolation, workspace patterns, and a recovery procedure for state-corruption incidents.

## Why this matters

State management is where Terraform deployments most often break catastrophically — overwriting state, losing state, conflicting concurrent applies. A planned strategy prevents the 3-day cleanup most teams have lived through.

## Requirements

1. S3 + DynamoDB backend, one bucket per environment.
2. Per-project state file with a clear key convention.
3. Workspace strategy decided (per-workspace vs per-directory).
4. State migration runbook (move resource between projects).
5. State recovery procedure (restore from S3 versioning).
6. Drift detection scheduled job.

## Step-by-step

### Step 1 — Backend bootstrap (30 min)
Create the backend infrastructure (chicken-and-egg: bootstrap uses local state, then migrates).

Per env: `company-tfstate-{dev,staging,prod}` S3 bucket with versioning + encryption + KMS key.
Single `tfstate-locks` DynamoDB table (per region).

### Step 2 — Key convention (15 min)
```
s3://company-tfstate-prod/
  ml-platform/vpc/terraform.tfstate
  ml-platform/eks/terraform.tfstate
  ml-platform/rds/terraform.tfstate
  data-platform/...
```
Hierarchical: org → team → project. Predictable, audit-friendly.

### Step 3 — Backend config (15 min)
```hcl
terraform {
  backend "s3" {
    bucket         = "company-tfstate-prod"
    key            = "ml-platform/vpc/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "tfstate-locks"
    encrypt        = true
    kms_key_id     = "alias/tfstate"
  }
}
```

### Step 4 — Workspace vs directory (30 min)
**Workspaces** (one state per workspace): great for "many similar things" (per-tenant, per-region).
```bash
terraform workspace new us-west-2
terraform workspace select us-west-2
```

**Directories** (one project = one state): better for "different things". Most teams pick this.

For multi-env: directory per env, NOT workspace per env. Workspaces blur the boundary; directories make differences explicit.

### Step 5 — IAM model (30 min)
Per backend bucket, per role:
- **Reader**: `s3:GetObject`, `s3:ListBucket` on read-only paths.
- **Engineer**: `s3:GetObject` + `s3:PutObject` + `dynamodb:GetItem`/`PutItem`/`DeleteItem`.
- **Restore-only**: `s3:GetObjectVersion` for state recovery.

CI service account has Engineer role.

### Step 6 — State migration runbook (30 min)
Moving a resource between two states (e.g., RDS from `data-platform` to `ml-platform` ownership):

```bash
# In source project
terraform state pull > /tmp/old-state.tfstate
terraform state rm 'module.rds'

# In destination project
terraform state push -force /tmp/source-state.tfstate  # carefully
# Then:
terraform state mv 'aws_db_instance.this' 'module.rds.aws_db_instance.this'

# Verify both ends with: terraform plan (should show no changes)
```

Document this as `STATE_MIGRATION.md`.

### Step 7 — Recovery procedure (15 min)
```bash
# List state file versions
aws s3api list-object-versions --bucket company-tfstate-prod \
  --prefix ml-platform/vpc/terraform.tfstate

# Restore a specific version
aws s3api copy-object \
  --bucket company-tfstate-prod \
  --copy-source 'company-tfstate-prod/ml-platform/vpc/terraform.tfstate?versionId=<vid>' \
  --key ml-platform/vpc/terraform.tfstate

# Verify
terraform init -reconfigure
terraform plan
```

### Step 8 — Drift detection (15 min)
Daily CI job per project: `terraform plan -detailed-exitcode`. Exit code 2 = drift. Send Slack notification.

## Deliverables

1. Backend bootstrap module.
2. `STATE_MIGRATION.md` runbook.
3. `STATE_RECOVERY.md` runbook.
4. Drift detection job per project.

## Validation

- [ ] Backend exists, encrypted, versioned.
- [ ] Two engineers cannot apply same state simultaneously.
- [ ] State migration tested in a sandbox.
- [ ] Recovery from a deliberately-corrupted state succeeds.

## Common pitfalls

- **Backend block with interpolation** — Not allowed. Use `-backend-config` or environment variables.
- **Same key for two projects** — Concurrent applies corrupt. Strict key convention prevents this.
- **`terraform state push -force`** — Loaded weapon. Always have a backup.
- **No versioning** — Corruption is permanent without it. Always enable S3 versioning.
