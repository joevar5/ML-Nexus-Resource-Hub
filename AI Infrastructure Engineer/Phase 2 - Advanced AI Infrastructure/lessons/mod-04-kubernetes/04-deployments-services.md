# Lesson 04: Deployments and Services Deep Dive

**Duration:** 8 hours
**Objectives:** Master advanced deployment strategies, service patterns, and health checks for ML workloads

## Introduction

You've learned the basics of Deployments and Services. Now we'll explore advanced patterns critical for production ML infrastructure: rolling updates, blue-green deployments, canary releases, and sophisticated health checks.

## Advanced Deployment Strategies

### 1. Rolling Updates (Default)

**Zero-downtime deployments** by gradually replacing pods.

**Configuration:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2          # Max 2 extra pods (20%)
      maxUnavailable: 1    # Max 1 unavailable (10%)
  template:
    spec:
      containers:
      - name: model-server
        image: ml-model:v2
```

**How it works:**
```
Initial state:     [v1] [v1] [v1] [v1] [v1]
maxSurge +2:       [v1] [v1] [v1] [v1] [v1] [v2] [v2]
maxUnavailable 1:  [v1] [v1] [v1] [v1] [X]  [v2] [v2]
Replace:           [v1] [v1] [v1] [v1] [v2] [v2] [v2]
Continue:          [v2] [v2] [v2] [v2] [v2] [v2] [v2]
```

**Best practices for ML models:**
```yaml
# Conservative rollout for critical models
rollingUpdate:
  maxSurge: 1
  maxUnavailable: 0    # No downtime tolerance

# Aggressive for non-critical
rollingUpdate:
  maxSurge: 3
  maxUnavailable: 2
```

### 2. Blue-Green Deployment

**Two complete environments**: blue (current) and green (new).

**Implementation with labels:**
```yaml
# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: blue
  template:
    metadata:
      labels:
        app: ml-model
        version: blue
    spec:
      containers:
      - name: model
        image: ml-model:v1

---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: green
  template:
    metadata:
      labels:
        app: ml-model
        version: green
    spec:
      containers:
      - name: model
        image: ml-model:v2

---
# Service initially points to blue
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Switch to 'green' to cutover
  ports:
  - port: 80
    targetPort: 8080
```

**Cutover process:**
```bash
# 1. Deploy green alongside blue
kubectl apply -f green-deployment.yaml

# 2. Test green independently
kubectl port-forward deployment/ml-model-green 8080:8080

# 3. Switch service to green (instant cutover)
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"green"}}}'

# 4. Monitor for issues

# 5. If OK, delete blue
kubectl delete deployment ml-model-blue

# 6. If issues, rollback to blue
kubectl patch service ml-model-service -p '{"spec":{"selector":{"version":"blue"}}}'
```

**Advantages:**
- Instant cutover and rollback
- Full testing before cutover
- Two complete environments

**Disadvantages:**
- Double resources during transition
- Higher cost

### 3. Canary Deployment

**Gradual rollout**: route small % of traffic to new version.

**Using multiple deployments:**
```yaml
# Stable deployment (90% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-model
      track: stable
  template:
    metadata:
      labels:
        app: ml-model
        track: stable
    spec:
      containers:
      - name: model
        image: ml-model:v1

---
# Canary deployment (10% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model
      track: canary
  template:
    metadata:
      labels:
        app: ml-model
        track: canary
    spec:
      containers:
      - name: model
        image: ml-model:v2

---
# Service balances across both
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model  # Matches both stable and canary
  ports:
  - port: 80
    targetPort: 8080
```

**Traffic distribution:**
```
9 stable pods + 1 canary pod = ~10% to canary
```

**Gradual rollout:**
```bash
# Start: 10% canary
kubectl scale deployment ml-model-stable --replicas=9
kubectl scale deployment ml-model-canary --replicas=1

# If metrics OK: 25% canary
kubectl scale deployment ml-model-stable --replicas=6
kubectl scale deployment ml-model-canary --replicas=2

# If metrics OK: 50% canary
kubectl scale deployment ml-model-stable --replicas=4
kubectl scale deployment ml-model-canary --replicas=4

# If metrics OK: 100% canary
kubectl scale deployment ml-model-stable --replicas=0
kubectl scale deployment ml-model-canary --replicas=8

# Cleanup: rename canary to stable
```

**Advanced canary with Flagger (GitOps):**
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: ml-model
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model
  service:
    port: 8080
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      threshold: 99
    - name: request-duration
      threshold: 500
```

### 4. Recreate Strategy

**Delete all old pods before creating new ones** - causes downtime.

```yaml
strategy:
  type: Recreate
```

**Use cases:**
- Database migrations requiring downtime
- Incompatible versions cannot coexist
- Resource constraints (cannot run both versions)

**Process:**
```
[v1] [v1] [v1] → (delete all) → [v2] [v2] [v2]
         ↓ Downtime
```

## Advanced Health Checks

### Comprehensive Health Check Configuration

**For ML model serving:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-model-server
spec:
  containers:
  - name: model
    image: ml-model:v1
    ports:
    - containerPort: 8080

    # Startup probe: large model loading (60 seconds)
    startupProbe:
      httpGet:
        path: /startup
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      failureThreshold: 12  # 60 seconds (12 * 5s)

    # Liveness probe: is container alive?
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 60
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3     # Restart after 30s (3 * 10s)
      successThreshold: 1

    # Readiness probe: ready for traffic?
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 3
      successThreshold: 2  # Must succeed twice
```

### Health Check Implementations

**Flask example:**
```python
from flask import Flask, jsonify
import time
import threading

app = Flask(__name__)

# State tracking
startup_complete = False
model_loaded = False
healthy = True

def load_model():
    global model_loaded, startup_complete
    time.sleep(30)  # Simulate slow model loading
    model_loaded = True
    startup_complete = True

# Start model loading in background
threading.Thread(target=load_model, daemon=True).start()

@app.route('/startup', methods=['GET'])
def startup():
    """Startup probe: has initial setup completed?"""
    if startup_complete:
        return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'loading'}), 503

@app.route('/health', methods=['GET'])
def health():
    """Liveness probe: is application alive?"""
    if healthy:
        return jsonify({'status': 'healthy'}), 200
    return jsonify({'status': 'unhealthy'}), 503

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness probe: ready to serve traffic?"""
    if model_loaded and healthy:
        # Additional checks
        if check_dependencies():  # DB, cache, etc.
            return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'not_ready'}), 503
```

### Custom Probe Handlers

**TCP socket probe:**
```yaml
livenessProbe:
  tcpSocket:
    port: 8080
  initialDelaySeconds: 15
  periodSeconds: 10
```

**Exec command probe:**
```yaml
livenessProbe:
  exec:
    command:
    - /bin/sh
    - -c
    - "curl -f http://localhost:8080/health || exit 1"
  initialDelaySeconds: 30
  periodSeconds: 10
```

**gRPC probe (Kubernetes 1.24+):**
```yaml
livenessProbe:
  grpc:
    port: 9090
  initialDelaySeconds: 10
```

## Service Patterns

### 1. Headless Service for StatefulSets

**For distributed ML training** where pods need stable DNS names.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-training
spec:
  clusterIP: None  # Headless
  selector:
    app: ml-training
  ports:
  - port: 8080
    name: training

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ml-training
spec:
  serviceName: ml-training  # Links to headless service
  replicas: 3
  selector:
    matchLabels:
      app: ml-training
  template:
    metadata:
      labels:
        app: ml-training
    spec:
      containers:
      - name: trainer
        image: distributed-trainer:v1
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
```

**DNS names:**
```
ml-training-0.ml-training.default.svc.cluster.local
ml-training-1.ml-training.default.svc.cluster.local
ml-training-2.ml-training.default.svc.cluster.local
```

### 2. Session Affinity (Sticky Sessions)

**Route requests from same client to same pod.**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-sticky
spec:
  selector:
    app: ml-model
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
  ports:
  - port: 80
    targetPort: 8080
```

**Use cases:**
- Stateful ML applications
- Model caching per pod
- WebSocket connections

### 3. External Services

**Access external databases/APIs:**

```yaml
# External service (outside cluster)
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: mysql-instance.cxxxx.us-west-2.rds.amazonaws.com

---
# Or with endpoints for IP-based
apiVersion: v1
kind: Service
metadata:
  name: external-api
spec:
  ports:
  - port: 443
    targetPort: 443

---
apiVersion: v1
kind: Endpoints
metadata:
  name: external-api
subsets:
- addresses:
  - ip: 203.0.113.42
  ports:
  - port: 443
```

### 4. Service Topology (Locality)

**Route traffic to closest pods** (reduce latency).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-local
spec:
  selector:
    app: ml-model
  topologyKeys:
  - "kubernetes.io/hostname"    # Same node
  - "topology.kubernetes.io/zone"  # Same zone
  - "*"                         # Any node
  ports:
  - port: 80
    targetPort: 8080
```

## Resource Management for ML Workloads

### Quality of Service (QoS) Classes

**1. Guaranteed (highest priority):**
```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
```

**2. Burstable:**
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "2"
  limits:
    memory: "16Gi"
    cpu: "4"
```

**3. BestEffort (lowest priority):**
```yaml
# No resources specified
# Evicted first under pressure
```

### Resource Quotas and Limits

**Namespace-level quotas:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    requests.nvidia.com/gpu: "8"
    limits.cpu: "200"
    limits.memory: "400Gi"
    pods: "50"
    services.loadbalancers: "5"
```

**Limit ranges (defaults):**
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-limits
  namespace: ml-team
spec:
  limits:
  - max:
      cpu: "16"
      memory: "64Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "1"
      memory: "2Gi"
    defaultRequest:
      cpu: "500m"
      memory: "1Gi"
    type: Container
```

## Hands-On: Canary Deployment for ML Model

**Setup:**
```bash
# Create namespace
kubectl create namespace ml-canary

# Deploy stable version
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-stable
  namespace: ml-canary
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-model
      track: stable
  template:
    metadata:
      labels:
        app: ml-model
        track: stable
        version: v1
    spec:
      containers:
      - name: model
        image: hashicorp/http-echo:latest
        args:
        - "-text=Model v1 (stable)"
        - "-listen=:8080"
        ports:
        - containerPort: 8080
EOF

# Deploy canary version
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-canary
  namespace: ml-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model
      track: canary
  template:
    metadata:
      labels:
        app: ml-model
        track: canary
        version: v2
    spec:
      containers:
      - name: model
        image: hashicorp/http-echo:latest
        args:
        - "-text=Model v2 (canary)"
        - "-listen=:8080"
        ports:
        - containerPort: 8080
EOF

# Create service
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: ml-model
  namespace: ml-canary
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8080
EOF
```

**Test traffic distribution:**
```bash
# Port forward service
kubectl port-forward -n ml-canary svc/ml-model 8080:80 &

# Send 100 requests and count distribution
for i in {1..100}; do curl -s http://localhost:8080; done | sort | uniq -c

# Expected:
# ~90 Model v1 (stable)
# ~10 Model v2 (canary)
```

**Promote canary:**
```bash
# Gradually increase canary traffic
kubectl scale deployment ml-model-stable --replicas=6 -n ml-canary
kubectl scale deployment ml-model-canary --replicas=4 -n ml-canary

# Test again
for i in {1..100}; do curl -s http://localhost:8080; done | sort | uniq -c

# If metrics good, full cutover
kubectl scale deployment ml-model-stable --replicas=0 -n ml-canary
kubectl scale deployment ml-model-canary --replicas=10 -n ml-canary
```

**Cleanup:**
```bash
kubectl delete namespace ml-canary
```

## Production Best Practices

### 1. Always Use Readiness Probes

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

Without readiness probes, pods receive traffic immediately, potentially causing errors during startup.

### 2. Set Resource Requests and Limits

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"
```

### 3. Use Deployment Annotations

```yaml
metadata:
  annotations:
    kubernetes.io/change-cause: "Update to model v2.1 with improved accuracy"
```

View history:
```bash
kubectl rollout history deployment/ml-model
```

### 4. PodDisruptionBudgets

**Ensure availability during voluntary disruptions:**

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-model-pdb
spec:
  minAvailable: 2  # At least 2 pods always available
  selector:
    matchLabels:
      app: ml-model
```

Or:
```yaml
spec:
  maxUnavailable: 1  # Max 1 pod can be down
```

### 5. Anti-Affinity Rules

**Spread pods across nodes/zones:**

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: ml-model
      topologyKey: kubernetes.io/hostname  # Different nodes
```

Or across zones:
```yaml
topologyKey: topology.kubernetes.io/zone
```

## Summary

In this lesson, you've mastered:

- ✅ Advanced deployment strategies: rolling, blue-green, canary
- ✅ Comprehensive health check configuration
- ✅ Service patterns: headless, session affinity, external
- ✅ Resource management and QoS classes
- ✅ Production best practices for ML workloads
- ✅ Hands-on canary deployment

## What's Next?

**Lesson 05: Networking and Ingress** - Deep dive into Kubernetes networking, Ingress controllers, and TLS termination.

## Self-Check Questions

1. What's the difference between blue-green and canary deployments?
2. When would you use a headless service?
3. What's the purpose of each type of health probe?
4. How do resource requests affect scheduling?
5. What is a PodDisruptionBudget and why use it?

---

**Next Lesson:** [05-networking-ingress.md](./05-networking-ingress.md)
