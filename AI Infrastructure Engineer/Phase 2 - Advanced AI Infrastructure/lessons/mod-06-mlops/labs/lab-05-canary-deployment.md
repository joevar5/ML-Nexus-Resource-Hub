# Lab 05: Canary Deployment with Argo Rollouts

**Duration:** 75 min  **Prerequisites:** kind cluster

## Objective
Replace a Deployment with an Argo Rollouts canary that progressively shifts traffic 10% → 25% → 50% → 100% with pauses, automatic analysis on real metrics, and automatic rollback on failure.

## Steps

### 1. Install Argo Rollouts
```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
brew install argoproj/tap/kubectl-argo-rollouts
```

### 2. Define a Rollout resource
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata: { name: iris }
spec:
  replicas: 4
  selector: { matchLabels: { app: iris } }
  template:
    metadata: { labels: { app: iris } }
    spec:
      containers:
        - name: api
          image: iris-api:0.2
          ports: [{ containerPort: 8000 }]
  strategy:
    canary:
      canaryService: iris-canary
      stableService: iris-stable
      steps:
        - setWeight: 10
        - pause: { duration: 30s }
        - setWeight: 25
        - pause: { duration: 30s }
        - analysis:
            templates:
              - templateName: http-success-rate
        - setWeight: 50
        - pause: { duration: 30s }
        - setWeight: 100
```

Plus two Services pointing at the same selector with stable/canary subsets, and an Ingress that splits via Istio/NGINX (out of scope here — use `setWeight` with a default ALB and trust the rollout controller).

### 3. AnalysisTemplate
```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata: { name: http-success-rate }
spec:
  args: [{ name: service-name }]
  metrics:
    - name: success-rate
      interval: 15s
      successCondition: result[0] >= 0.99
      failureLimit: 2
      provider:
        prometheus:
          address: http://monitoring-kube-prometheus-prometheus.monitoring.svc:9090
          query: |
            sum(rate(http_requests_total{job="iris",status=~"2.."}[2m]))
            /
            sum(rate(http_requests_total{job="iris"}[2m]))
```

### 4. Apply
```bash
kubectl apply -f rollout.yaml
kubectl argo rollouts get rollout iris -w
```

### 5. Trigger a canary
```bash
kubectl argo rollouts set image iris api=iris-api:0.3
```
Watch progression in the CLI tool (live updates).

### 6. Simulate a failure
Deploy `iris-api:0.4` that returns 500s. AnalysisTemplate detects success-rate drop, rolls back automatically.

## Validation
- [ ] Rollout progresses through all weight steps when the new image is healthy.
- [ ] AnalysisTemplate query returns values in `kubectl argo rollouts get`.
- [ ] Bad image triggers auto-rollback within ~1 min of analysis failure.

## Cleanup
```bash
kubectl delete rollout iris
kubectl delete -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
kubectl delete ns argo-rollouts
```

## Troubleshooting
- **`Rollout stuck at 10%`** — Analysis succeeded but auto-promotion disabled; check `progressDeadlineSeconds`.
- **AnalysisTemplate "no datapoints"** — Prometheus query returns empty. Verify `http_requests_total` exists and has matching labels.
- **Canary and stable services not separating traffic** — Without a traffic-management tool (Istio, NGINX), Rollouts only adjusts replica counts; HTTP-level splitting needs a controller integration.
