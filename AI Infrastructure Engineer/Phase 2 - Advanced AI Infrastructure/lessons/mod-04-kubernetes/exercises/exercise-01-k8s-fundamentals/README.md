# Exercise 01: Kubernetes Fundamentals — Build a Working Service

**Duration:** 3 hours
**Difficulty:** Beginner+
**Prerequisites:** kind cluster (lab 01); kubectl basics

## Objective

Deploy a real ML serving service end-to-end on Kubernetes from scratch (no Helm): Deployment + Service + ConfigMap + Secret + Ingress + HPA. By the end you'll be able to read and write these manifests fluently.

## Requirements

1. Deploy `iris-api` (from earlier exercises) with 3 replicas.
2. Externalize ML config via ConfigMap and credentials via Secret.
3. Expose via NGINX Ingress.
4. Add HPA scaling between 2-10 replicas based on CPU.
5. Add liveness + readiness probes.
6. Apply resource requests + limits.
7. Validate end-to-end with `curl` + load test.

## Step-by-step

### Step 1 — ConfigMap and Secret (15 min)
```yaml
apiVersion: v1
kind: ConfigMap
metadata: { name: iris-config }
data:
  app.yaml: |
    log_level: INFO
    feature_count: 4
    batch_max: 128
---
apiVersion: v1
kind: Secret
metadata: { name: iris-secrets }
type: Opaque
stringData:
  api_token: "test-token-12345"
```

### Step 2 — Deployment (30 min)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: iris-api, labels: { app: iris } }
spec:
  replicas: 3
  selector: { matchLabels: { app: iris } }
  template:
    metadata: { labels: { app: iris } }
    spec:
      containers:
        - name: api
          image: iris-api:0.2
          imagePullPolicy: Never
          ports: [{ containerPort: 8000 }]
          env:
            - { name: API_TOKEN, valueFrom: { secretKeyRef: { name: iris-secrets, key: api_token } } }
          volumeMounts:
            - { name: config, mountPath: /etc/app, readOnly: true }
          resources:
            requests: { cpu: 100m, memory: 256Mi }
            limits:   { cpu: 500m, memory: 512Mi }
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet: { path: /ready, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 3
      volumes:
        - name: config
          configMap: { name: iris-config }
```

### Step 3 — Service + Ingress (15 min)
```yaml
apiVersion: v1
kind: Service
metadata: { name: iris-api }
spec:
  selector: { app: iris }
  ports: [{ port: 80, targetPort: 8000 }]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: iris-api
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$1
spec:
  ingressClassName: nginx
  rules:
    - host: iris.localtest.me
      http:
        paths:
          - path: /(.*)
            pathType: ImplementationSpecific
            backend:
              service: { name: iris-api, port: { number: 80 } }
```

### Step 4 — HPA (15 min)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: iris-api }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: iris-api }
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
```

### Step 5 — Apply + verify (30 min)
```bash
kubectl apply -f .
kubectl get pods,svc,ingress,hpa
kubectl rollout status deployment/iris-api

curl http://iris.localtest.me/health
curl -X POST http://iris.localtest.me/v1/predict -H 'content-type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

### Step 6 — Generate load + observe HPA (45 min)
```bash
kubectl run loadgen --rm -it --image=williamyeh/hey --restart=Never -- \
  -z 5m -c 50 -m POST -T 'application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}' http://iris-api/v1/predict
watch -n 2 'kubectl get hpa,pods'
```
Replicas should climb from 2 to 5+; after load stops, scale back to 2.

### Step 7 — Deliberate misconfigure + fix (30 min)
Set readiness probe to wrong path. Re-apply. Observe 0/3 Ready, Service has no endpoints, requests fail. Fix → recover.

## Deliverables

1. All 5 manifests committed.
2. Working `curl` to Ingress.
3. Demonstration of HPA scale-up.
4. `OPERATIONS.md`: how to deploy, roll back, debug this service.

## Validation

- [ ] `kubectl get pods -l app=iris` shows ≥ 2 Ready.
- [ ] Ingress URL returns 200.
- [ ] HPA scales up under load, down when idle.
- [ ] Bad readiness probe → 0 endpoints → requests fail.

## Stretch goals

- Add a NetworkPolicy that only allows ingress traffic from the nginx-ingress namespace.
- Add a PodDisruptionBudget ensuring at least 2 replicas during rollouts.
- Add a separate Service for a canary subset.

## Common pitfalls

- **Resource requests missing → HPA broken** — HPA needs requests to compute utilization.
- **Liveness probe too aggressive** — Kills pods during normal startup. `initialDelaySeconds` matters.
- **Ingress controller not installed** — Apply the kind-specific NGINX manifest first.
