# Exercise 07: Multi-Account / Multi-Project Security Architecture

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** AWS Organizations or GCP Folders; admin access

## Objective

Design and implement a multi-account (AWS) or multi-project (GCP) security architecture for an ML platform: separate prod/staging/dev/sandbox accounts, central audit logging, SCP/Org Policy guardrails, cross-account IAM, and a least-privilege deployment role used by CI to push to prod from a separate account.

## Why this matters

Production blast radius is the single biggest source of cloud incidents. Single-account setups give engineers root access to everything to do their work. Multi-account architectures separate environments at the strongest boundary cloud providers offer. Every team that scales past ~10 engineers ends up here; doing it deliberately is cheaper than retroactively migrating.

## Requirements

### Topology

```
Management account
├── Audit account              # CloudTrail logs, S3 access logs, Config history
├── Security account           # GuardDuty hub, Security Hub, IAM identity center
├── Networking account         # Transit Gateway, central DNS, prefix lists
├── Shared services account    # ECR, container registry, CI/CD runners
└── Workload OU
    ├── Production
    ├── Staging
    ├── Development
    └── Sandbox (per-user)
```

Even if you can only create 2-3 accounts in your test sandbox, model all of them in code and apply to the subset you have.

### Controls

1. **Service Control Policies (SCPs)** at OU level:
   - Workload OU: deny disabling CloudTrail, deny launching unsupported regions, deny launching unapproved instance types.
   - Production OU: deny `iam:CreateUser`, deny root account usage.
   - Sandbox OU: spend cap via budget action that stops EC2 instances.
2. **Centralized CloudTrail** organization trail logging to the audit account.
3. **AWS Config** recording in every account, conforming pack: encrypted-at-rest, public-access blocked, MFA on root.
4. **GuardDuty** enabled org-wide, central hub in security account.
5. **IAM Identity Center (SSO)** with permission sets per role:
   - `DataScientist`: read-only on prod data + admin on dev/sandbox.
   - `MLEngineer`: admin on dev/sandbox, write on staging, read on prod.
   - `Operator`: admin on staging+prod, read on dev.
6. **Cross-account IAM role** `cicd-deployer` in prod, assumable only from the shared services account, scoped to ECS/EKS/S3 deploys for the ML workload.

## Step-by-step

### Step 1 — Inventory current state (15 min)
List accounts, identify what runs where. If only 1 account, plan the migration.

### Step 2 — Provision Organizations (30 min)
Terraform module creating the OUs and a few accounts. Use `aws_organizations_organizational_unit` and `aws_organizations_account`.

### Step 3 — SCPs (45 min)
Author 4-6 SCPs covering the controls above. Attach to OUs. Test each by trying the forbidden action from a workload account.
```hcl
resource "aws_organizations_policy" "deny_root_in_prod" {
  name = "deny-root-in-prod"
  type = "SERVICE_CONTROL_POLICY"
  content = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Deny",
      Action = "*",
      Resource = "*",
      Condition = { StringLike = { "aws:PrincipalArn" = "arn:aws:iam::*:root" } }
    }]
  })
}
```

### Step 4 — Centralized logging (30 min)
- Create the organization trail in management account.
- Encrypt with KMS key in audit account.
- Apply S3 bucket policy denying cross-account deletes.

### Step 5 — IAM Identity Center (30 min)
- Enable Identity Center in management account.
- Define permission sets.
- Assign users to accounts via groups.
- Verify SSO login from `aws sso login --profile ml-prod`.

### Step 6 — CI/CD cross-account role (30 min)
```hcl
# In the prod account
resource "aws_iam_role" "cicd_deployer" {
  assume_role_policy = jsonencode({
    Statement = [{
      Effect = "Allow",
      Action = "sts:AssumeRole",
      Principal = { AWS = "arn:aws:iam::<shared-svc-account>:role/github-actions" },
      Condition = { StringEquals = { "sts:ExternalId" = "ml-cicd-prod" } }
    }]
  })
  inline_policy {
    name = "deploy"
    policy = jsonencode({...scoped to model-api ECR + ECS + S3 only...})
  }
}
```
GitHub Actions workflow assumes role via OIDC.

### Step 7 — Verify guardrails (30 min)
Each SCP must be testable. Write `scripts/test-controls.sh` that attempts the forbidden action from each account type and verifies the denial.

### Step 8 — Document (30 min)
`ARCHITECTURE.md` with the topology, role matrix, and on-call escalation paths for "I'm locked out" and "we suspect a credential leak".

## Deliverables

1. Terraform modules for Organizations + SCPs + Identity Center.
2. Cross-account CI/CD role in code.
3. `ARCHITECTURE.md` + `RUNBOOK.md` for security incidents.
4. Working SSO login + working CI/CD deploy from shared services to prod.

## Validation

- [ ] Attempting `aws s3 mb s3://test` as a Sandbox user from outside the allowed region fails due to SCP.
- [ ] CloudTrail events from a workload account appear in the audit account's S3 bucket within ~15 minutes.
- [ ] `aws sso login` works for at least one permission set in at least one workload account.
- [ ] GitHub Actions can assume the cicd-deployer role and push to prod ECR.
- [ ] Removing the SCP allows the previously-blocked action (sanity check that SCPs were what blocked it).

## Stretch goals

- Add **AWS Control Tower** as the higher-level abstraction over the manual Organizations setup.
- Implement **break-glass access**: a special account with audit-only-on-use, time-boxed admin role for incidents.
- Add **Backup vault** per account with cross-region copies for DR.
- Integrate **Detective** or **Lake Formation** for analytic queries over the audit log.

## Common pitfalls

- **SCPs don't grant; they only deny** — IAM policies still required to grant access. New folks get confused when they have SCP-allow but no IAM-allow.
- **Detaching SCPs in prod** — Don't, unless via a documented break-glass. Most outages from SCP changes are in the "we needed to do one thing real quick" category.
- **Sandbox without budget caps** — A misconfigured GPU workload can run up $10k in a weekend. Always have spend-stop budget actions.
- **CI deploys from a developer account** — Use a dedicated shared services account; never let prod creds live alongside developer creds.

## Solutions

Reference Terraform in the engineer-solutions repo.
