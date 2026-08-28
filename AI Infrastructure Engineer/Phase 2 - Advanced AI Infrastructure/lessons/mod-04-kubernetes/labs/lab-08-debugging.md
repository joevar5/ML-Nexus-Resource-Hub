# Lab 08: Debug a Broken Deployment from Scratch

**Duration:** 90 min  **Prerequisites:** Any cluster

## Objective
Practice systematic Kubernetes debugging on three deliberately broken workloads. Find the root cause and fix each without looking ahead.

## Steps

### 1. Apply the broken manifests
```yaml
# broken.yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: bug-a }
spec:
  replicas: 1
  selector: { matchLabels: { app: bug-a } }
  template:
    metadata: { labels: { app: bug-a } }
    spec:
      containers:
        - name: app
          image: nginx:badtag           # bug A: nonexistent tag
          ports: [{ containerPort: 80 }]
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: bug-b }
spec:
  replicas: 1
  selector: { matchLabels: { app: bug-b } }
  template:
    metadata: { labels: { app: bug-b } }
    spec:
      containers:
        - name: app
          image: nginx:1.27
          resources:
            requests: { cpu: "200", memory: "100Gi" }    # bug B: impossible requests
---
apiVersion: apps/v1
kind: Deployment
metadata: { name: bug-c }
spec:
  replicas: 1
  selector: { matchLabels: { app: bug-c } }
  template:
    metadata: { labels: { app: bug-c-WRONG } }            # bug C: selector/label mismatch
    spec:
      containers:
        - name: app
          image: nginx:1.27
```
```bash
kubectl apply -f broken.yaml
```

### 2. Debug methodology
For each broken deployment:
1. `kubectl get pods`  → what phase / restart count?
2. `kubectl describe pod <name>` → Events at the bottom usually tell you why.
3. `kubectl logs <name>` → app-level errors.
4. `kubectl get deploy <name> -o yaml | grep -A 10 status` → controller-level state.

### 3. Bug A — ImagePullBackOff
```
Events:
  Failed to pull image "nginx:badtag": ... manifest not found
```
**Fix:** correct the tag.

### 4. Bug B — Unschedulable
```
Events:
  0/1 nodes are available: 1 Insufficient cpu, 1 Insufficient memory.
```
**Fix:** realistic requests.

### 5. Bug C — Deployment never reaches Ready
`kubectl describe deploy bug-c` shows replicas 0. Or it's stuck — selector says `bug-c` but template labels say `bug-c-WRONG`. **Fix:** Match the labels.

### 6. Bonus: simulate a CrashLoop
```yaml
containers:
  - name: app
    image: busybox
    command: ["sh", "-c", "exit 1"]
```
Investigate with `kubectl logs --previous` and `kubectl describe pod`.

## Validation
- [ ] All three deployments become Ready 1/1.
- [ ] You can articulate the root cause of each bug without scrolling up.

## Cleanup
```bash
kubectl delete -f broken.yaml
```

## Debug shortcuts
- `kubectl get events --sort-by=.metadata.creationTimestamp`
- `kubectl exec -it <pod> -- sh` — for inside-pod investigation
- `kubectl debug -it <pod> --image=busybox --target=<container>` — ephemeral debug container
- `kubectl get pod <name> -o yaml` — full spec + status

## Troubleshooting (about debugging itself)
- **`describe` shows no Events** — Events are namespaced and time out after ~1h. Don't rely on them solely for older failures.
- **`logs` returns nothing** — Container never started. Check `describe` for the init phase or pull failures.
