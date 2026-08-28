# Lab 06: Horizontal Pod Autoscaler under Load

**Duration:** 60 min  **Prerequisites:** Lab 05 chart deployed; metrics-server installed

## Objective
Configure an HPA on CPU and custom metrics, generate load, and observe scale-up + scale-down behavior.

## Steps

### 1. Install metrics-server (kind needs `--kubelet-insecure-tls`)
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl top nodes
```

### 2. HPA on CPU
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: iris }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: iris }
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 50 } }
```
```bash
kubectl apply -f hpa.yaml
kubectl get hpa iris -w
```

### 3. Generate load
```bash
kubectl run loadgen --rm -it --image=williamyeh/hey --restart=Never -- \
  -z 3m -c 50 -m POST -T 'application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
  http://iris-iris-chart/predict
```

### 4. Observe scale-up
```bash
watch -n 2 'kubectl get hpa,deploy,pods -l app=iris-chart'
```
Expect replicas to climb from 2 toward 10 over ~30-60s.

### 5. Stop load and observe scale-down
After load ends, replicas should fall back to 2 after the stabilization window (default 300s).

### 6. HPA behavior tuning
```yaml
spec:
  behavior:
    scaleUp:   { stabilizationWindowSeconds: 30 }
    scaleDown: { stabilizationWindowSeconds: 120 }
```

### 7. Custom metric (Prometheus Adapter sketch)
Install `prometheus-adapter`, expose `iris_predictions_per_second` from your app, write a `Pods` metric in the HPA. Scale on requests/sec instead of CPU.

## Validation
- [ ] `kubectl top pods` shows actual CPU usage.
- [ ] During load, pods scale up to at least replicaCount × 2.
- [ ] After load, pods scale back down within stabilization window.
- [ ] `kubectl describe hpa iris` shows current vs target utilization.

## Cleanup
```bash
kubectl delete hpa iris
helm uninstall iris 2>/dev/null
```

## Troubleshooting
- **HPA shows `unknown / 50%`** — metrics-server not running or not authorized. Check `kubectl top pods` first.
- **HPA never scales up** — Resource requests aren't set on the deployment; HPA needs them to compute utilization.
- **Scales up but not down** — Stabilization window (5 min default) hasn't elapsed; that's intentional anti-flap behavior.
