# Exercise 03: Core Resources Mastery

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01

## Objective

Build hands-on with every core resource: Pod, ReplicaSet, Deployment, StatefulSet, DaemonSet, Job, CronJob, ConfigMap, Secret, PersistentVolume, PersistentVolumeClaim, Service, Ingress, ServiceAccount, Role, RoleBinding. Demonstrate each with a real use case.

## Requirements

For each of 15 resources, write a manifest, apply it, and document one realistic use case in your work.

## Step-by-step

### Step 1 — Pod (raw) (10 min)
```yaml
apiVersion: v1
kind: Pod
metadata: { name: standalone }
spec:
  containers: [{ name: app, image: nginx }]
```
Use case: **debug image** — quick `kubectl run` for one-off tests.

### Step 2 — ReplicaSet (skip; rarely written directly; Deployments wrap them) (5 min)
Just observe one created by a Deployment.

### Step 3 — Deployment (covered in ex-01)

### Step 4 — StatefulSet (covered in lab 04)

### Step 5 — DaemonSet (15 min)
Pod on every node:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata: { name: node-exporter }
spec:
  selector: { matchLabels: { app: node-exporter } }
  template:
    metadata: { labels: { app: node-exporter } }
    spec:
      containers:
        - name: node-exporter
          image: prom/node-exporter:v1.7.0
          ports: [{ containerPort: 9100, hostPort: 9100 }]
      hostNetwork: true
```
Use case: per-node agents (logging, monitoring, networking).

### Step 6 — Job (15 min)
```yaml
apiVersion: batch/v1
kind: Job
metadata: { name: train-model }
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - { name: train, image: my-trainer:latest }
  backoffLimit: 2
```
Use case: model training, data migration.

### Step 7 — CronJob (15 min)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata: { name: nightly-retrain }
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers: [{ name: train, image: my-trainer:latest }]
```
Use case: scheduled retraining, cleanup tasks.

### Step 8 — ConfigMap + Secret (covered in ex-01 + lab 03)

### Step 9 — PV + PVC (covered in lab 04)

### Step 10 — Service types (30 min)
ClusterIP, NodePort, LoadBalancer, ExternalName, Headless. One manifest each. Use cases:
- ClusterIP: internal
- NodePort: lab / dev exposure
- LoadBalancer: production external
- ExternalName: alias to external service (DNS)
- Headless: per-pod DNS for StatefulSet

### Step 11 — Ingress with TLS (30 min)
Add cert-manager + Let's Encrypt or self-signed:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: iris-tls
  annotations: { cert-manager.io/cluster-issuer: letsencrypt-prod }
spec:
  ingressClassName: nginx
  tls: [{ hosts: [iris.example.com], secretName: iris-tls }]
  rules:
    - host: iris.example.com
      http: { paths: [{ path: /, pathType: Prefix, backend: { service: { name: iris-api, port: { number: 80 } } } }] }
```

### Step 12 — ServiceAccount + Role + RoleBinding (30 min)
Allow a pod to read pods in its namespace:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata: { name: pod-reader }
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { name: pod-reader }
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: { name: pod-reader }
subjects:
  - kind: ServiceAccount
    name: pod-reader
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: pod-reader
```

Test:
```bash
kubectl run probe --rm -it --image=bitnami/kubectl --serviceaccount=pod-reader -- get pods
```

### Step 13 — Namespace + ResourceQuota (30 min)
```yaml
apiVersion: v1
kind: Namespace
metadata: { name: tenant-a }
---
apiVersion: v1
kind: ResourceQuota
metadata: { name: tenant-a-quota, namespace: tenant-a }
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    pods: "30"
```

## Deliverables

1. Manifests for each resource (one file per resource).
2. `USE_CASES.md` with one realistic use case per resource.
3. `RBAC_DIAGRAM.md` showing the SA-Role-Binding relationship.

## Validation

- [ ] Each resource type applied successfully.
- [ ] DaemonSet has one pod per node.
- [ ] CronJob creates a Job at the scheduled time.
- [ ] ServiceAccount-based access works (and is denied without binding).
- [ ] Quota blocks exceeding limits in a tenant namespace.

## Stretch goals

- Add **LimitRange** to set default requests/limits.
- Add **PriorityClass** to influence pod scheduling order.
- Use **kubectl explain** on each resource to document the spec.

## Common pitfalls

- **DaemonSet on tainted nodes** — Needs explicit tolerations.
- **Job that never completes** — Forgot `restartPolicy: Never` or `OnFailure`.
- **RoleBinding cross-namespace** — Role is namespaced; for cross-ns use ClusterRole + RoleBinding.
- **ResourceQuota requires both requests AND limits** — If pods don't set them, they fail admission.
