# Lab 01: Your First Pod and Deployment

**Duration:** 60 min  **Prerequisites:** kind/minikube cluster running

## Objective
Run a bare Pod, then graduate to a Deployment with replicas, rolling updates, and rollback.

## Steps

### 1. Run a bare Pod
```bash
kubectl run nginx --image=nginx:1.27 --port=80
kubectl get pods
kubectl describe pod nginx | tail -20
kubectl port-forward pod/nginx 8080:80 &
curl http://localhost:8080
kill %1
kubectl delete pod nginx
```
Notice the Pod stays gone — no controller is recreating it.

### 2. Promote to a Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: web }
spec:
  replicas: 3
  selector: { matchLabels: { app: web } }
  template:
    metadata: { labels: { app: web } }
    spec:
      containers:
        - name: nginx
          image: nginx:1.27
          ports: [{ containerPort: 80 }]
          resources:
            requests: { cpu: 50m, memory: 64Mi }
            limits:   { cpu: 200m, memory: 128Mi }
```
```bash
kubectl apply -f deployment.yaml
kubectl get pods -l app=web
kubectl delete pod -l app=web --field-selector=status.phase=Running --grace-period=0  # delete one
kubectl get pods -l app=web                                                            # replaced
```

### 3. Roll out a new image
```bash
kubectl set image deployment/web nginx=nginx:1.27-alpine
kubectl rollout status deployment/web
kubectl rollout history deployment/web
```

### 4. Rollback
```bash
kubectl rollout undo deployment/web
kubectl rollout history deployment/web
```

### 5. Tweak the rollout strategy
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```
Re-apply, redeploy with a new image, observe stricter rollout: at most 4 pods at any time, at least 3 Ready.

## Validation
- [ ] `kubectl get deployment web` shows 3/3 Ready.
- [ ] After deleting one pod, a replacement appears within seconds.
- [ ] `kubectl rollout history` shows at least 3 revisions.
- [ ] Rollback restores the previous image (verify with `kubectl describe pod -l app=web | grep Image`).

## Cleanup
```bash
kubectl delete -f deployment.yaml
```

## Troubleshooting
- **Pods stuck Pending** — `kubectl describe pod` shows scheduling reasons (insufficient CPU/memory, taints).
- **Rollout hangs** — `kubectl rollout status --timeout=30s` then `kubectl describe` the new pods.
