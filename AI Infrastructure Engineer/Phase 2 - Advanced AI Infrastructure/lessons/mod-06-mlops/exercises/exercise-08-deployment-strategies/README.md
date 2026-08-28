# Exercise 08: Model Deployment Strategies

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 03, 07; Kubernetes cluster

## Objective

Implement and compare four model deployment strategies: rolling update, blue-green, canary, and shadow. Run each on a real model-serving service; measure rollout time, blast radius if something goes wrong, and the operational complexity of each.

## Why this matters

Choosing the wrong deployment strategy for the risk profile of your model is how teams have outages. Engineers who know all four — and when each is right — ship updates faster with smaller blast radii.

## The four strategies

| Strategy | Risk | Speed | Complexity |
|---|---|---|---|
| Rolling | Medium (some users see new immediately) | Fast | Low |
| Blue-Green | Low (instant switch, instant rollback) | Slow (2× capacity) | Medium |
| Canary | Low (gradual) | Slow | Medium |
| Shadow | Zero (no user impact) | N/A — validation only | High |

## Requirements

For each strategy, implement working Kubernetes manifests + a script that triggers the rollout and lets you observe progress + rollback.

## Step-by-step

### Step 1 — Setup (15 min)
Have iris-api v1 and v2 images in a registry. v2 has a deliberate-but-subtle behavior change (e.g., always returns class 0).

### Step 2 — Rolling update (30 min)
```yaml
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```
```bash
kubectl set image deployment/iris-api api=iris-api:v2
kubectl rollout status deployment/iris-api
```

Observe: at any moment some pods serve v1, some v2. Old pods drained gradually.

### Step 3 — Blue-Green (45 min)
Two deployments, one Service.
```yaml
# Service points at app=iris-api, color=blue
apiVersion: v1
kind: Service
metadata: { name: iris-api }
spec:
  selector: { app: iris-api, color: blue }
  ports: [{ port: 80, targetPort: 8000 }]

---
# Blue deployment (v1)
spec:
  replicas: 4
  template:
    metadata: { labels: { app: iris-api, color: blue } }
    spec: { containers: [{ image: iris-api:v1, ... }] }

---
# Green deployment (v2)
spec:
  replicas: 4
  template:
    metadata: { labels: { app: iris-api, color: green } }
    spec: { containers: [{ image: iris-api:v2, ... }] }
```
Switch traffic atomically:
```bash
kubectl patch svc iris-api --type=merge -p '{"spec":{"selector":{"app":"iris-api","color":"green"}}}'
# Verify; if bad:
kubectl patch svc iris-api --type=merge -p '{"spec":{"selector":{"app":"iris-api","color":"blue"}}}'
```

### Step 4 — Canary with Argo Rollouts (per mod-06 lab 05) (60 min)
Replace Deployment with `Rollout`:
```yaml
spec:
  replicas: 4
  strategy:
    canary:
      canaryService: iris-canary
      stableService: iris-stable
      steps:
        - setWeight: 10
        - pause: { duration: 60s }
        - setWeight: 25
        - pause: { duration: 60s }
        - analysis: { templates: [{templateName: success-rate}] }
        - setWeight: 50
        - pause: { duration: 60s }
        - setWeight: 100
```

### Step 5 — Shadow deployment (45 min)
Route 100% real traffic to v1 (live) AND copy 100% of requests to v2 (shadow). Compare predictions but only serve v1's response.

Sketch via a Python proxy:
```python
@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    primary, shadow = await asyncio.gather(
        forward_to("http://iris-v1:8000/predict", body),
        forward_to("http://iris-v2:8000/predict", body, ignore_errors=True),
    )
    # Log shadow result for offline comparison
    metrics_log(body, primary, shadow)
    return primary
```
Or use Istio mirroring:
```yaml
http:
  - route:
      - destination: { host: iris-api, subset: v1 }
        weight: 100
    mirror: { host: iris-api, subset: v2 }
    mirror_percentage: { value: 100 }
```

### Step 6 — Measure (30 min)
For each strategy, record:
- Time from `apply` to "fully rolled out"
- Blast radius if v2 is broken: how many real users see the bug before you notice
- Rollback time
- Operational complexity (lines of YAML, services involved)

Fill `STRATEGY_COMPARISON.md`.

## Deliverables

1. Working manifests for all 4 strategies.
2. Rollback procedure documented per strategy.
3. `STRATEGY_COMPARISON.md` with measurements.
4. `WHEN_TO_USE.md`: decision tree (high-risk model → use shadow then canary; routine bugfix → rolling; etc.).

## Validation

- [ ] Each of the 4 strategies works end-to-end on your cluster.
- [ ] Blue-green switch is instantaneous from user perspective.
- [ ] Canary auto-pauses at each step.
- [ ] Shadow logs both predictions for offline comparison without user impact.
- [ ] Rollback procedure verified for each.

## Stretch goals

- Add **automatic rollback** based on real-time SLO metrics (per Argo Rollouts AnalysisTemplate).
- Implement **multi-region** blue-green with DNS-based traffic switch.
- Add **persisted shadow comparison**: weekly report showing where v2 disagrees with v1.

## Common pitfalls

- **Rolling update with `maxUnavailable > 0`** — Brief capacity dip during rollout; latency spike for users.
- **Blue-green with stateful workloads** — Two databases? Schema migrations? Painful. Stick to stateless for blue-green.
- **Canary without metrics-based gates** — Just slow rolling. Need analysis or you're using complexity for no benefit.
- **Shadow that affects v1 performance** — Shadow requests share resources; cap concurrency; circuit-break v2 if it gets slow.
