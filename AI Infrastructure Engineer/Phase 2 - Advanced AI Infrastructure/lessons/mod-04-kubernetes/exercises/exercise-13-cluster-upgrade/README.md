# Exercise 13: Cluster Upgrade Procedure

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** kind cluster

## Objective

Plan and execute a Kubernetes cluster upgrade from 1.29 to 1.30, validating API compatibility, draining nodes one-by-one, and verifying no application disruption.

## Why this matters

Cluster upgrades are the riskiest routine operation an SRE does. A well-rehearsed procedure makes them boring; an ad-hoc one makes them outages.

## Requirements

1. Pre-upgrade audit: deprecated APIs, custom controllers, addons.
2. Workloads tolerant to a node going away (PDBs, replicas > 1).
3. Control plane upgrade.
4. Node pool upgrade with rolling drain.
5. Post-upgrade validation: all apps healthy, no rollback needed.

## Step-by-step

### Step 1 — Start with a 1.29 cluster (10 min)
```bash
kind create cluster --image kindest/node:v1.29.4 --name upgrade-lab
```

### Step 2 — Deploy diverse workloads (30 min)
- iris-api Deployment (4 replicas, PDB minAvailable=2)
- pg cluster (3 replicas)
- A Job
- A CronJob
- ingress-nginx

### Step 3 — Pre-upgrade audit (30 min)
```bash
# Deprecated APIs check
kubectl get -A --all-resources | grep -E "extensions/v1beta1|apps/v1beta1"
pluto detect-files -d .              # https://github.com/FairwindsOps/pluto

# Workloads without PDBs
kubectl get pdb -A
kubectl get deploy -A -o json | jq '.items[] | select(.spec.replicas > 1) | .metadata' 
# (cross-reference with PDBs)
```
List anything that would block the upgrade.

### Step 4 — Workload readiness checks (15 min)
For each Deployment with `replicas > 1`, ensure:
- PDB protecting it
- PodDisruptionBudget allows drain
- App tolerates restart

### Step 5 — Test in a copy cluster first (real world); skipped in lab.

### Step 6 — Upgrade control plane (30 min)
For kind:
```bash
# Stop cluster
kind delete cluster --name upgrade-lab
# Recreate at new version
kind create cluster --image kindest/node:v1.30.0 --name upgrade-lab
# (For real clusters: kubeadm upgrade plan / apply; cloud providers have their own commands)
```

Verify:
```bash
kubectl version
kubectl get nodes      # all Ready, v1.30.0
```

### Step 7 — Re-deploy workloads (15 min)
Apply the same manifests. Verify all Ready.

### Step 8 — Validate no API drift (30 min)
```bash
# Any deprecated API warnings in apiserver logs?
kubectl logs -n kube-system kube-apiserver-... | grep -i deprecated

# Run pluto again
pluto detect-files -d . --target-versions=k8s=v1.30.0
```

### Step 9 — Performance / functional validation (15 min)
Smoke-test each application; verify SLOs still met.

### Step 10 — Post-upgrade docs (15 min)
`UPGRADE_RUNBOOK.md` documenting:
- Pre-checks
- Sequence
- Validation
- Rollback plan (snapshot-based)

## Deliverables

1. Upgraded cluster.
2. All workloads healthy post-upgrade.
3. `UPGRADE_RUNBOOK.md`.
4. List of any deprecated APIs found and fixed.

## Validation

- [ ] `kubectl get nodes` shows the new version.
- [ ] No 5xx during the upgrade (verify with curl in a loop during cordon/drain).
- [ ] All workloads return to Ready state.
- [ ] pluto reports clean.

## Stretch goals

- Perform the upgrade with **zero downtime**: actually verify p95 latency unchanged during the upgrade window.
- Practice on EKS / GKE / AKS where upgrade UX differs significantly.
- Add a **canary cluster** upgrade pattern: upgrade staging first, soak for 2 weeks, then prod.

## Common pitfalls

- **Skipping pre-checks** — Discover deprecated APIs only after upgrade fails.
- **No PDBs on critical apps** — Drain evicts all replicas, brief outage.
- **Upgrading > 1 minor version** — Not supported in one step; do n+1 at a time.
- **Forgetting addons** — Calico/Cilium/coredns versions may need bumps too.
