# Exercise 10: Multi-Environment Promotion Pipeline

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 03 (modules)

## Objective

Build a promotion pipeline: code → dev → staging → prod with PR-based approvals, environment-specific values, and an auto-revert on regression. Both for Terraform IaC and the application deployment.

## Requirements

1. Single Terraform codebase with dev/staging/prod environments.
2. PR triggers plan for affected environments.
3. Merge to main applies to dev automatically.
4. Staging applies on tag.
5. Prod applies on tag + manual approval.
6. Each environment has a `values.tfvars` file under version control.

## Step-by-step

### Step 1 — Project layout (15 min)
```
infra/
├── modules/             (from exercise 03)
└── envs/
    ├── dev/
    │   ├── main.tf       # composes modules
    │   ├── backend.tf    # state for dev
    │   └── terraform.tfvars
    ├── staging/
    └── prod/
```

### Step 2 — Environment-specific tfvars (30 min)
```hcl
# envs/dev/terraform.tfvars
region              = "us-west-2"
cluster_size        = "small"
node_count          = 2
db_instance_type    = "db.t4g.small"
multi_az            = false

# envs/prod/terraform.tfvars
region              = "us-west-2"
cluster_size        = "large"
node_count          = 6
db_instance_type    = "db.m6g.xlarge"
multi_az            = true
```

### Step 3 — Branch + tag strategy (15 min)
- `main` branch always reflects `dev` state.
- Tags `staging-v*` deploy to staging.
- Tags `prod-v*` deploy to prod (after approval).

### Step 4 — Plan-on-PR (45 min)
```yaml
on: pull_request
jobs:
  plan:
    strategy:
      matrix: { env: [dev, staging, prod] }
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: envs/${{ matrix.env }} } }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123:role/gh-actions-${{ matrix.env }}
          aws-region: us-west-2
      - uses: hashicorp/setup-terraform@v3
      - run: terraform init
      - run: terraform plan -no-color -out=plan.tfplan
      - uses: actions/upload-artifact@v4
        with: { name: plan-${{ matrix.env }}, path: envs/${{ matrix.env }}/plan.tfplan }
      - name: Comment PR
        run: |
          terraform show -no-color plan.tfplan > plan.txt
          gh pr comment ${{ github.event.pull_request.number }} \
            --body "### Plan for ${{ matrix.env }}\n\`\`\`\n$(cat plan.txt | head -200)\n\`\`\`"
```

### Step 5 — Apply-on-merge (dev) (30 min)
```yaml
on:
  push: { branches: [main], paths: ['envs/dev/**', 'modules/**'] }
jobs:
  apply-dev:
    runs-on: ubuntu-latest
    environment: dev    # has 0 reviewers required
    steps:
      - ...
      - run: cd envs/dev && terraform init && terraform apply -auto-approve
```

### Step 6 — Apply-on-tag (staging) (30 min)
```yaml
on:
  push: { tags: ['staging-v*'] }
jobs:
  apply-staging:
    runs-on: ubuntu-latest
    environment: staging  # 1 reviewer required
    steps:
      - ...
      - run: cd envs/staging && terraform init && terraform apply -auto-approve
```

### Step 7 — Apply with approval (prod) (15 min)
```yaml
on:
  push: { tags: ['prod-v*'] }
jobs:
  apply-prod:
    runs-on: ubuntu-latest
    environment: prod  # 2 reviewers + 1 hour wait
    steps:
      - ...
      - run: cd envs/prod && terraform init && terraform apply -auto-approve
```
Configure in GitHub Settings → Environments → prod → required reviewers + wait timer.

### Step 8 — Auto-revert on failure (15 min)
If post-apply smoke test fails:
```yaml
- name: Smoke test
  run: ./scripts/smoke-test.sh ${{ matrix.env }}
- name: Revert on failure
  if: failure()
  run: |
    git revert HEAD --no-edit
    git push origin main
    terraform plan -out=revert.tfplan
    terraform apply -auto-approve revert.tfplan
```

## Deliverables

1. Three envs sharing modules.
2. PR workflow generating per-env plans.
3. Apply workflows per env with appropriate approvals.
4. Smoke test + auto-revert.
5. `PROMOTION_FLOW.md` documenting the lifecycle of a change.

## Validation

- [ ] PR comment shows plan for each env.
- [ ] Merge to main applies to dev only.
- [ ] Staging tag triggers staging apply with one reviewer gate.
- [ ] Prod tag requires two reviewers + 1h wait.
- [ ] Synthetic smoke-test failure auto-reverts.

## Common pitfalls

- **Same state for two envs** — Catastrophic. Strict per-env backend config.
- **Promotion drift** — Staging and prod diverge over time. Quarterly review.
- **No smoke test** — Apply succeeds, app broken. Always test from outside.
- **Approval bypass** — Engineers add themselves as reviewer. Use a separate approver role.
