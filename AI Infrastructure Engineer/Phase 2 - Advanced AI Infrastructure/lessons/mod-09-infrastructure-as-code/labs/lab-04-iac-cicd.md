# Lab 05: IaC in CI/CD with PR-Based Plan

**Duration:** 60 min  **Prerequisites:** GitHub repo with Terraform code; AWS credentials in GH Secrets

## Objective
Set up a CI/CD workflow that runs `terraform plan` on every PR, posts the plan as a PR comment, and runs `terraform apply` only on merge to `main`.

## Steps

### 1. Workflow
```yaml
# .github/workflows/terraform.yml
name: terraform
on:
  pull_request:
    paths: ['terraform/**']
  push:
    branches: [main]
    paths: ['terraform/**']

permissions:
  contents: read
  pull-requests: write
  id-token: write

jobs:
  plan:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: terraform } }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions
          aws-region: us-west-2
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.7.5 }
      - run: terraform fmt -check -recursive
      - run: terraform init
      - run: terraform validate
      - id: plan
        run: terraform plan -no-color -out=plan.tfplan
        continue-on-error: true
      - name: Show plan
        run: terraform show -no-color plan.tfplan > plan.txt
      - name: Comment PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('terraform/plan.txt', 'utf8');
            const body = `### Terraform Plan\n\`\`\`\n${plan.slice(0, 60000)}\n\`\`\``;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body
            });
      - if: steps.plan.outcome == 'failure'
        run: exit 1

  apply:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production           # gates on env approval
    defaults: { run: { working-directory: terraform } }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions
          aws-region: us-west-2
      - uses: hashicorp/setup-terraform@v3
      - run: terraform init
      - run: terraform apply -auto-approve
```

### 2. Set up OIDC trust on AWS side (one-time)
Create an IAM role trusted by GitHub's OIDC provider; attach least-privilege policies. (Out of scope of this lab — see HashiCorp's guide.)

### 3. Configure GH environment
Settings → Environments → `production` → Required reviewers (yourself or team).

### 4. Open a PR
```bash
git checkout -b lab-05
echo '# trivial change' >> terraform/README.md
git add terraform/README.md && git commit -m "lab: trigger plan"
git push -u origin lab-05
gh pr create --fill
```
Watch the workflow; plan should comment on the PR.

### 5. Merge
On merge, the apply workflow runs (waiting for environment approval).

### 6. Diff alerting
Sentinel/OPA/conftest can fail the plan if it includes destructive resource changes (see lab 07).

## Validation
- [ ] PR opens → plan workflow comments on PR with the plan output.
- [ ] Plan failure causes PR check to fail.
- [ ] Merge to main → apply workflow pauses for approval, then applies.

## Cleanup
```bash
git checkout main && git pull && git branch -D lab-05
```

## Troubleshooting
- **`Permission denied (publickey)`** — Use HTTPS clone or set up SSH.
- **OIDC `not authorized to perform sts:AssumeRoleWithWebIdentity`** — Trust policy on AWS role doesn't match GH org/repo claims.
- **Plan comment cut off** — GitHub's comment max is 65,536 chars; truncate or upload as artifact.
