# Exercise 12: Disaster Recovery via IaC

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercises 03, 04, 10

## Objective

Design and rehearse a disaster recovery procedure where a complete cluster + dependencies are reconstructed from IaC + S3-backed state + Vault + S3-backed model artifacts. Target RTO < 4 hours, RPO < 1 hour.

## Why this matters

DR plans that aren't rehearsed are wishful thinking. IaC plus regular DR rehearsals turn DR from a 3-day fire drill into a 4-hour controlled event.

## Requirements

1. Document what's restored from where.
2. Rehearse: deliberately destroy a sandbox env; restore from backups.
3. Measure actual RTO (recovery time) and RPO (data loss).
4. Document lessons learned.

## Step-by-step

### Step 1 — Inventory (30 min)
What must come back, and from where?

| Component | Source of truth | Backup mechanism |
|---|---|---|
| Cluster (EKS) | Terraform module | S3 state file (versioned) |
| Cluster apps (manifests) | Git (ArgoCD) | Git clone |
| Vault | Vault data | Vault snapshot to S3 (daily) |
| Postgres data | CloudNativePG cluster | WAL archive to S3 (continuous) |
| Object storage data | S3 | Cross-region replication |
| Container images | ECR / GHCR | Replicated to backup registry |
| Secrets | Vault | Vault snapshot includes |
| Model artifacts | S3 | S3 versioning + cross-region |

### Step 2 — Restore script (45 min)
```bash
#!/usr/bin/env bash
set -euo pipefail
ENV=${1:?usage: restore.sh <env>}
REGION=${2:-us-west-2}

# 1. Restore Terraform state (if lost)
aws s3 cp s3://company-tfstate-backup-${ENV}/latest.tfstate ./envs/${ENV}/terraform.tfstate

# 2. Provision infrastructure
cd envs/${ENV}
terraform init -reconfigure
terraform apply -auto-approve

# 3. Wait for cluster ready
aws eks update-kubeconfig --name ml-${ENV} --region ${REGION}
kubectl wait --for=condition=Ready node --all --timeout=10m

# 4. Bootstrap ArgoCD (if not running)
kubectl apply -n argocd -f manifests/argocd/install.yaml
kubectl wait --for=condition=Available deploy/argocd-server -n argocd --timeout=10m

# 5. Apply root Application — pulls all app manifests
kubectl apply -f manifests/argocd/root-app-${ENV}.yaml

# 6. Restore Vault from snapshot
kubectl exec -n vault vault-0 -- vault operator raft snapshot restore /backup/latest.snap
# Unseal Vault (manual; humans hold keys)

# 7. Restore Postgres (CNPG auto-recovers from WAL archive)
# Verify with: kubectl get cluster pg -n iris

# 8. Smoke test
./scripts/smoke-test.sh ${ENV}
```

### Step 3 — Run the rehearsal (60 min)
Pick a sandbox env. Note start time. Destroy:
```bash
cd envs/sandbox
terraform destroy -auto-approve
# Also delete the state file from S3 to simulate full disaster:
aws s3 rm s3://company-tfstate-prod/sandbox/terraform.tfstate
```

Restore:
```bash
./restore.sh sandbox
```

Time it. Record what failed, what was unclear, what was missing.

### Step 4 — Lessons learned (30 min)
Document in `DR_LESSONS.md`:
- Where did the procedure deviate from plan?
- What was the actual RTO vs target?
- What was the RPO vs target?
- What gaps were discovered?
- What follow-up work?

### Step 5 — Update runbook (15 min)
Edit the runbook with fixes for every issue. The next rehearsal should be smoother.

## Deliverables

1. Inventory document.
2. `restore.sh` script.
3. Rehearsal log with timings.
4. `DR_LESSONS.md`.
5. Updated runbook.

## Validation

- [ ] Restore completes within RTO target (< 4h).
- [ ] Data loss is within RPO target (< 1h).
- [ ] Smoke test passes post-restore.
- [ ] Runbook is updated with lessons learned.

## Stretch goals

- **Cross-region failover**: restore into a different region.
- **Active-active DR**: traffic auto-fails over to the secondary region.
- **Chaos engineering**: schedule quarterly random destruction in sandbox.

## Common pitfalls

- **State file in same account as infra** — Lose account → lose state. Store state cross-account.
- **Vault keys lost** — No way to unseal. Use Shamir's secret sharing across humans.
- **Backups never tested** — A backup you've never restored is a hope.
- **Documentation drift** — Procedure works in your head, fails for the on-call engineer. Test with someone else running it.
