# Exercise 10: Multi-Cluster Federation

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Two K8s clusters (use 2 kind clusters or a real + a kind)

## Objective

Connect two Kubernetes clusters for workload portability: cross-cluster service discovery via a service mesh (Istio multi-primary, Linkerd, or Cilium Cluster Mesh) and cross-cluster ArgoCD-managed deployments.

## Requirements

1. Two clusters with bidirectional service discovery.
2. A service in cluster A can call a service in cluster B by name.
3. ArgoCD installed in a "hub" cluster, managing deployments to both.
4. Failover: redirect traffic when one cluster fails.

## Step-by-step

### Step 1 — Two kind clusters (15 min)
```bash
kind create cluster --name a --config a.yaml
kind create cluster --name b --config b.yaml
```
Use different pod/service CIDRs to avoid collisions.

### Step 2 — Install Cilium Cluster Mesh (45 min)
Best for kind:
```bash
cilium install --context kind-a --cluster-name a --cluster-id 1
cilium install --context kind-b --cluster-name b --cluster-id 2
cilium clustermesh enable --context kind-a
cilium clustermesh enable --context kind-b
cilium clustermesh connect --context kind-a --destination-context kind-b
```

### Step 3 — Deploy cross-cluster services (30 min)
In cluster A: a frontend.
In cluster B: a backend, annotated `service.cilium.io/global: "true"`.
Frontend in A calls `backend.default.svc.cluster.local` — resolves to B.

### Step 4 — ArgoCD setup (45 min)
```bash
kubectl --context kind-a create ns argocd
kubectl --context kind-a apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Add cluster B as a target
argocd cluster add kind-b
argocd app create iris-on-b --repo <git> --path ./manifests --dest-server <kind-b-api> --dest-namespace default
```

### Step 5 — Failover demo (30 min)
- Pause cluster B; frontend health check fails for backend.
- Use ArgoCD ApplicationSet to deploy backend to A as well; service mesh load-balances.
- Resume B; ApplicationSet keeps both in sync.

## Deliverables

1. Two clusters with mesh.
2. Cross-cluster service call demo.
3. ArgoCD managing both.
4. `FEDERATION_TRADEOFFS.md`: cluster mesh vs single-cluster-multiple-namespaces.

## Validation

- [ ] Service in A reaches service in B by DNS name.
- [ ] ArgoCD shows healthy sync to both clusters.
- [ ] Killing cluster B doesn't break A's frontend (because B's backend pods exist in A too).

## Common pitfalls

- **CIDR overlap** — Clusters can't mesh if pod/service ranges overlap.
- **Bidirectional control plane access** — Each cluster needs network access to the other's apiserver.
- **Federation complexity** — Often overkill; multiple namespaces in one cluster suffices.
