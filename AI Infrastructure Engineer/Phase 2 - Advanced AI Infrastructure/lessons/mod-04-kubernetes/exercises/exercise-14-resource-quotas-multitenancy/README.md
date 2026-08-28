# Exercise 14: Resource Quotas and Multi-Tenancy

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 03 (core resources)

## Objective

Make a single cluster safely host 3 teams' workloads: namespace per team, ResourceQuota + LimitRange + NetworkPolicy + RBAC per team. Demonstrate that team A can't accidentally consume team B's capacity or read team B's secrets.

## Requirements

1. 3 namespaces (`team-a`, `team-b`, `team-c`).
2. ResourceQuota per namespace (CPU, memory, pod count, PVC count).
3. LimitRange enforcing default + max per pod.
4. NetworkPolicy denying cross-namespace traffic by default.
5. RBAC: each team admin scoped to own namespace; no cross-team access.
6. PriorityClass per tier (gold preempts silver preempts bronze).

## Step-by-step

### Step 1 — Namespaces (15 min)
```bash
for ns in team-a team-b team-c; do
  kubectl create ns $ns
  kubectl label ns $ns tier=silver
done
```

### Step 2 — ResourceQuota (30 min)
```yaml
apiVersion: v1
kind: ResourceQuota
metadata: { name: team-quota, namespace: team-a }
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    pods: "30"
    persistentvolumeclaims: "10"
    services.loadbalancers: "1"
```

### Step 3 — LimitRange (15 min)
```yaml
apiVersion: v1
kind: LimitRange
metadata: { name: defaults, namespace: team-a }
spec:
  limits:
    - type: Container
      default:        { cpu: 500m, memory: 256Mi }
      defaultRequest: { cpu: 100m, memory: 128Mi }
      max:            { cpu: "4", memory: 8Gi }
```

### Step 4 — Default deny NetworkPolicy + same-namespace allow (30 min)
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny, namespace: team-a }
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: allow-same-namespace, namespace: team-a }
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress: [{ from: [{ podSelector: {} }] }]
  egress:  [{ to: [{ podSelector: {} }] }]
---
# Plus allow DNS (per mod-04 lab 07)
```

### Step 5 — RBAC (30 min)
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { name: team-admin, namespace: team-a }
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: { name: team-a-admin, namespace: team-a }
subjects: [{ kind: User, name: alice }]
roleRef: { kind: Role, name: team-admin, apiGroup: rbac.authorization.k8s.io }
```

### Step 6 — PriorityClasses (15 min)
```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata: { name: gold }
value: 1000
description: Critical workload
---
kind: PriorityClass
metadata: { name: silver }
value: 500
---
kind: PriorityClass
metadata: { name: bronze }
value: 100
```

### Step 7 — Validate isolation (45 min)
```bash
# Team A user tries to read team B secret → forbidden
kubectl --as=alice get secret -n team-b
# error: User "alice" cannot list resource "secrets" in namespace "team-b"

# Team A pod tries to reach team B pod → blocked by NetworkPolicy
kubectl exec -n team-a pod-a -- curl team-b-service.team-b -m 3
# timeout

# Team A overspends quota → admission denied
kubectl create deploy big -n team-a --image=nginx --replicas=100
# Error from server (Forbidden): ... exceeded quota
```

### Step 8 — Demo preemption (15 min)
Fill cluster with team-A bronze workloads. Schedule a gold-priority workload from team-B. Observe team-A pods preempted to make room.

## Deliverables

1. All 3 team setups committed as manifests.
2. `MULTITENANCY.md` documenting the isolation guarantees + their limits.
3. Validation script that proves isolation (CI-runnable).

## Validation

- [ ] Team A user cannot read Team B secrets.
- [ ] Team A pod cannot reach Team B pods.
- [ ] Quota blocks over-provisioning.
- [ ] Preemption works across priority classes.

## Stretch goals

- Add **Hierarchical Namespaces** (HNC) for nested team structures.
- Add **OPA Gatekeeper** for org-wide policies (e.g., "all images must come from approved registry").
- Add **vCluster** for stronger isolation (each tenant gets a virtual control plane).

## Common pitfalls

- **NetworkPolicy without CNI support** — flannel doesn't enforce. Use Calico or Cilium.
- **Quota requires limits AND requests on every container** — Pods without them fail admission. Use LimitRange defaults.
- **Cross-namespace Service references** — Don't forget the FQDN: `svc.namespace.svc.cluster.local`.
- **Default-deny breaks DNS** — Always allow DNS egress to kube-system first.
