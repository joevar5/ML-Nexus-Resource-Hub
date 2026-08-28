# Lab 06: A/B Test with Traffic Splitting

**Duration:** 60 min  **Prerequisites:** kind cluster; Istio installed (`istioctl install -y`)

## Objective
Deploy two model versions side-by-side, split traffic 50/50, log per-version prediction outcomes, and compute the winner with a simple statistical test.

## Steps

### 1. Deploy v1 and v2
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: iris-v1, labels: { app: iris, version: v1 } }
spec:
  replicas: 2
  selector: { matchLabels: { app: iris, version: v1 } }
  template:
    metadata: { labels: { app: iris, version: v1 } }
    spec: { containers: [{ name: api, image: iris-api:0.2, ports: [{ containerPort: 8000 }] }] }
---
# Same for iris-v2 with image iris-api:0.3 and label version: v2
```

### 2. Single Service spanning both versions
```yaml
apiVersion: v1
kind: Service
metadata: { name: iris }
spec:
  selector: { app: iris }
  ports: [{ port: 80, targetPort: 8000 }]
```

### 3. Istio VirtualService for 50/50 split
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata: { name: iris }
spec:
  host: iris
  subsets:
    - { name: v1, labels: { version: v1 } }
    - { name: v2, labels: { version: v2 } }
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata: { name: iris }
spec:
  hosts: [iris]
  http:
    - route:
        - destination: { host: iris, subset: v1 }
          weight: 50
        - destination: { host: iris, subset: v2 }
          weight: 50
```

### 4. Drive traffic and collect outcomes
The API includes the model version in responses. Capture them:
```bash
for _ in {1..500}; do
  curl -s -X POST http://iris.default/predict -H 'content-type: application/json' \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
done | jq -r .model_version | sort | uniq -c
```
Expect ~50/50 split.

### 5. Log outcomes (clickthrough, conversion) per version
In a real test, you'd log a downstream business metric tied to each request. For the lab, simulate with a synthetic outcome generator:
```python
import requests, random, csv
rows = []
for _ in range(2000):
    r = requests.post("http://iris.default/predict", json={...}).json()
    success = random.random() < (0.85 if r["model_version"]=="v2" else 0.80)
    rows.append([r["model_version"], int(success)])
csv.writer(open("results.csv","w")).writerows(rows)
```

### 6. Analyze
```python
import pandas as pd
from scipy import stats
df = pd.read_csv("results.csv", names=["version","success"])
pivot = df.groupby("version").agg(["mean","count"])
print(pivot)
res = stats.proportions_ztest(
    df.groupby("version")["success"].sum(),
    df.groupby("version")["success"].count(),
)
print("p-value:", res[1])
```

## Validation
- [ ] Traffic split is ~50/50 (within ±5% over 500+ requests).
- [ ] Outcome capture identifies which version each request hit.
- [ ] Z-test produces a p-value; with simulated 5pp lift over 2000 trials, expect p < 0.05.

## Cleanup
```bash
kubectl delete deploy iris-v1 iris-v2
kubectl delete svc iris
kubectl delete destinationrule iris
kubectl delete virtualservice iris
```

## Troubleshooting
- **All traffic to v1** — VirtualService not applied or Istio not injected. Label namespace: `kubectl label ns default istio-injection=enabled`.
- **`p-value` always tiny** — You generated correlated synthetic outcomes; use independent random sampling.
- **Split is 70/30 not 50/50** — Connection pooling pins traffic per client. Use a fresh client per request or increase n.
