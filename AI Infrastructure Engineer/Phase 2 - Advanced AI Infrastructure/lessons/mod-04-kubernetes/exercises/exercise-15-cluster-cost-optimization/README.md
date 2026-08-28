# Exercise 15: Kubernetes Cluster Cost Optimization

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 14 + mod-02 exercise 08

## Objective

Audit a cluster for cost waste, apply 5 optimization techniques (rightsizing, spot, cluster autoscaling, idle resource hunting, storage class tiering), measure savings.

## Requirements

1. Baseline cost (or simulated).
2. Apply each of 5 techniques.
3. Measure result.
4. Build a quarterly review process.

## Step-by-step

### Step 1 — Inventory (30 min)
- Total CPU + memory requested vs available
- Total cost (use AWS / GCP cost API or estimate from instance types)
- Per-namespace breakdown

```bash
kubectl top nodes
kubectl top pods -A --containers
```

Compute utilization: actual usage / requested capacity. Below 30% = waste.

### Step 2 — Rightsizing (45 min)
For each over-requested deployment:
```bash
kubectl get deploy -A -o json | jq -r '.items[] | "\(.metadata.namespace)/\(.metadata.name) \(.spec.template.spec.containers[0].resources.requests)"'
```
Compare to actual usage over past 7 days. Reduce requests where actual << requested.

Tool: **goldilocks** suggests rightsizing:
```bash
helm install goldilocks fairwinds-stable/goldilocks -n goldilocks --create-namespace
kubectl label ns team-a goldilocks.fairwinds.com/enabled=true
```

### Step 3 — Spot instances (45 min)
Add a spot node pool (per cloud):
```yaml
# Karpenter NodePool example
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata: { name: spot }
spec:
  template:
    spec:
      requirements:
        - { key: karpenter.sh/capacity-type, operator: In, values: [spot] }
        - { key: kubernetes.io/arch, operator: In, values: [amd64] }
```
Mark batch workloads to prefer spot:
```yaml
nodeSelector: { karpenter.sh/capacity-type: spot }
tolerations: [{ key: spot, operator: Exists }]
```

### Step 4 — Cluster Autoscaler / Karpenter (30 min)
Without an autoscaler, you pay for max capacity 24/7. With Karpenter, nodes provision on demand and de-provision when idle.

Install per cloud's docs. Set ttlSecondsAfterEmpty so empty nodes go away in 60s.

Test: scale down all workloads; node count should drop within minutes.

### Step 5 — Idle resource hunt (30 min)
Find:
- Pods with restart count > 100 (probably crashlooping; either fix or remove)
- Deployments scaled to 0 (leftover from old projects)
- PVs/PVCs not mounted
- LoadBalancers without traffic
- Unused namespaces

```bash
kubectl get pvc -A | awk '$3 == ""'
kubectl get svc -A --field-selector spec.type=LoadBalancer
```

### Step 6 — Storage class tiering (15 min)
- Hot data (databases): SSD / gp3
- Cold data (logs, backups): HDD / sc1 / S3
Migrate cold PVCs to cheaper class.

### Step 7 — Build cost reports (30 min)
Install **OpenCost** (Kubecost OSS) or **Kubecost**:
```bash
helm install opencost opencost/opencost -n opencost --create-namespace
```
Gives per-namespace / per-pod / per-deployment cost.

Set up weekly Slack digest with top 10 spenders.

## Deliverables

1. `COST_AUDIT.md` with before/after numbers.
2. Goldilocks recommendations applied.
3. Spot pool in place.
4. Karpenter (or CA) provisioning on demand.
5. Weekly digest configured.

## Validation

- [ ] Cluster utilization improved (actual/requested up by ≥ 20pp).
- [ ] Total monthly cost down ≥ 30%.
- [ ] No production workload affected (verify SLOs).
- [ ] Weekly digest delivers to Slack.

## Stretch goals

- Add **cost-aware autoscaling**: prefer spot for HPA scale-ups, fall back to on-demand when spot unavailable.
- Implement **quotas as cost budgets**: ResourceQuota → equivalent $/month.
- Add **per-tenant chargeback**: each team sees their own cost line.

## Common pitfalls

- **Rightsizing too aggressively** — Pods OOM under load. Use p99 utilization, not average.
- **All-spot for prod** — Spot interruption mid-prediction = user error. Mix with on-demand or use spot only for batch.
- **Karpenter without consolidation enabled** — Doesn't merge under-utilized nodes.
- **Idle hunt killing things in use** — Verify ownership before deleting; always 7-day grace.
