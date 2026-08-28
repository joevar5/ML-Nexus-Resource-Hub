# Exercise 06: GitOps with ArgoCD

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Kubernetes cluster

## Objective

Implement GitOps with ArgoCD: cluster manifests live in Git, ArgoCD continuously reconciles to match Git. Add ApplicationSet for multi-cluster + multi-env deployments, App-of-Apps pattern, and progressive sync waves.

## Why this matters

GitOps reverses the deploy paradigm: instead of pushing changes to the cluster, you push to Git and the cluster pulls. This makes audit trivial, rollback a `git revert`, and disaster recovery a fresh cluster + `argocd app sync`.

## Requirements

1. ArgoCD installed.
2. Single-cluster app deployed via ArgoCD.
3. App-of-Apps pattern managing 5+ apps.
4. ApplicationSet generating apps from a Git directory.
5. Multi-env via ApplicationSet matrix generator.
6. Sync waves ensuring CRDs deploy before resources that use them.

## Step-by-step

### Step 1 — Install ArgoCD (15 min)
```bash
kubectl create ns argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl port-forward -n argocd svc/argocd-server 8080:443 &
# Get initial admin password:
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
# Login at https://localhost:8080
```

### Step 2 — Single Application (30 min)
Create a Git repo `argocd-apps/` containing manifests:
```
argocd-apps/
└── iris-api/
    ├── deployment.yaml
    └── service.yaml
```
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: iris-api, namespace: argocd }
spec:
  project: default
  source:
    repoURL: https://github.com/me/argocd-apps
    targetRevision: HEAD
    path: iris-api
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated: { prune: true, selfHeal: true }
```
```bash
kubectl apply -f apps/iris-api.yaml
argocd app sync iris-api
argocd app get iris-api
```
Edit a manifest in Git → push → watch ArgoCD auto-sync.

### Step 3 — App-of-Apps (45 min)
One Application that points to a directory of Applications:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata: { name: root, namespace: argocd }
spec:
  source:
    repoURL: https://github.com/me/argocd-apps
    path: bootstrap   # contains other Application manifests
  destination: { server: https://kubernetes.default.svc, namespace: argocd }
  syncPolicy: { automated: { prune: true, selfHeal: true } }
```

`bootstrap/` contains Applications for iris-api, monitoring, ingress, etc. Apply root once; ArgoCD bootstraps everything.

### Step 4 — ApplicationSet from directory (45 min)
Generate one Application per directory:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata: { name: services, namespace: argocd }
spec:
  generators:
    - git:
        repoURL: https://github.com/me/argocd-apps
        revision: HEAD
        directories:
          - path: services/*
  template:
    metadata: { name: '{{path.basename}}' }
    spec:
      project: default
      source:
        repoURL: https://github.com/me/argocd-apps
        path: '{{path}}'
      destination: { server: https://kubernetes.default.svc, namespace: '{{path.basename}}' }
      syncPolicy: { automated: { prune: true, selfHeal: true } }
```

Add `services/<name>/` directory → ApplicationSet auto-creates an Application.

### Step 5 — Multi-env via matrix generator (30 min)
```yaml
generators:
  - matrix:
      generators:
        - list:
            elements:
              - { cluster: dev,     url: https://dev-cluster.api }
              - { cluster: staging, url: https://stg-cluster.api }
              - { cluster: prod,    url: https://prod-cluster.api }
        - git:
            directories: [{ path: services/* }]
template:
  metadata: { name: '{{path.basename}}-{{cluster}}' }
  spec:
    source:
      path: '{{path}}'
      helm: { valueFiles: [values-{{cluster}}.yaml] }
    destination: { server: '{{url}}', namespace: '{{path.basename}}' }
```

### Step 6 — Sync waves (15 min)
```yaml
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "-1"   # CRDs deploy first
```
CRDs in wave -1, namespace+RBAC in wave 0, deployments in wave 1.

### Step 7 — Drift detection demo (15 min)
Manually `kubectl edit deploy iris-api`. ArgoCD detects drift, displays OutOfSync, auto-heals.

## Deliverables

1. ArgoCD running.
2. Git repo with manifests.
3. App-of-Apps managing 5+ Applications.
4. ApplicationSet generating apps from a directory.
5. Multi-env ApplicationSet for at least 2 envs.
6. `GITOPS_PHILOSOPHY.md`: why your team uses GitOps + when not to.

## Validation

- [ ] Pushing to Git triggers a sync within 3 minutes.
- [ ] Manually changing a resource is reverted by self-heal.
- [ ] Adding a service directory creates a new Application automatically.
- [ ] Sync waves enforce ordering (CRDs first).

## Stretch goals

- Add **Argo Rollouts** for progressive delivery via the same Git workflow.
- Implement **PR-based environments**: opening a PR creates a temp namespace via ApplicationSet.
- Add **OPA Gatekeeper** as part of the GitOps pipeline.

## Common pitfalls

- **Self-heal turned off** — Forgot `selfHeal: true`. Manual changes persist; defeats GitOps.
- **Sync without waves** — Random order; some resources fail because prerequisites not yet applied.
- **Secret in Git** — Use sealed-secrets, external-secrets, or sops; never commit plaintext.
- **No drift alerts** — Drift goes unnoticed; auto-heal reverts user attempts at hotfixes.
