# Lesson 03: Core Kubernetes Resources

**Duration:** 7 hours
**Objectives:** Master pods, ReplicaSets, Deployments, Services, ConfigMaps, Secrets, and Namespaces

## Introduction

In previous lessons, you learned about Kubernetes architecture. Now it's time to work directly with the core resources that make up Kubernetes applications.

This lesson covers the fundamental building blocks you'll use daily as an ML infrastructure engineer.

## Pods: The Atomic Unit

**A Pod is the smallest deployable unit in Kubernetes** - a wrapper around one or more containers.

### What is a Pod?

```
┌─────────────────────────────────────┐
│            POD                      │
│  ┌──────────────┐  ┌────────────┐  │
│  │ Container 1  │  │Container 2 │  │
│  │ (ML Model)   │  │ (Sidecar)  │  │
│  └──────────────┘  └────────────┘  │
│                                     │
│  Shared:                            │
│  - Network namespace (IP address)   │
│  - IPC namespace                    │
│  - Volumes                          │
│  - Lifecycle                        │
└─────────────────────────────────────┘
```

### Pod Characteristics

1. **Shared Network:** All containers share the same IP and port space
2. **Shared Storage:** Containers can share volumes
3. **Atomic Scheduling:** Containers in a pod scheduled together
4. **Ephemeral:** Pods are disposable and replaceable

### Basic Pod Manifest

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference-pod
  labels:
    app: ml-api
    version: v1
spec:
  containers:
  - name: model-server
    image: tensorflow/serving:latest
    ports:
    - containerPort: 8501
    env:
    - name: MODEL_NAME
      value: "resnet"
```

### Multi-Container Pods

**Common patterns for ML workloads:**

**1. Sidecar Pattern:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-with-logging
spec:
  containers:
  # Main container: ML model
  - name: model-server
    image: my-ml-model:v1
    ports:
    - containerPort: 8080
    volumeMounts:
    - name: logs
      mountPath: /var/log/ml

  # Sidecar: Log collector
  - name: log-collector
    image: fluent/fluentd:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/ml

  volumes:
  - name: logs
    emptyDir: {}
```

**2. Ambassador Pattern:**
```yaml
# ML model + proxy for metrics/auth
spec:
  containers:
  - name: model-server
    image: my-ml-model:v1

  - name: envoy-proxy
    image: envoyproxy/envoy:latest
    # Handles authentication, rate limiting, metrics
```

**3. Adapter Pattern:**
```yaml
# ML model + format converter
spec:
  containers:
  - name: model-server
    image: pytorch-model:v1
    # Outputs custom format

  - name: format-adapter
    image: output-converter:v1
    # Converts to standard API format
```

### Init Containers

**Run before main containers** - useful for setup tasks.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-model-with-init
spec:
  # Init containers run sequentially before main containers
  initContainers:
  - name: download-model
    image: google/cloud-sdk:latest
    command:
    - gsutil
    - cp
    - gs://my-models/resnet-v1.pb
    - /models/
    volumeMounts:
    - name: model-storage
      mountPath: /models

  - name: validate-model
    image: model-validator:v1
    command: ["python", "validate.py", "/models/resnet-v1.pb"]
    volumeMounts:
    - name: model-storage
      mountPath: /models

  # Main container starts after init containers succeed
  containers:
  - name: model-server
    image: tensorflow/serving:latest
    volumeMounts:
    - name: model-storage
      mountPath: /models

  volumes:
  - name: model-storage
    emptyDir: {}
```

### Pod Lifecycle Phases

```
Pending → ContainerCreating → Running → Succeeded/Failed
                                    ↓
                              CrashLoopBackOff
```

**Phases:**
- **Pending:** Accepted but not yet scheduled
- **ContainerCreating:** Pulling images, starting containers
- **Running:** At least one container running
- **Succeeded:** All containers terminated successfully
- **Failed:** At least one container failed
- **Unknown:** Cannot determine state

### Resource Requests and Limits

**Critical for ML workloads** to prevent resource contention.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: trainer
    image: pytorch-trainer:v1
    resources:
      # Requests: Guaranteed resources for scheduling
      requests:
        memory: "16Gi"    # Minimum memory
        cpu: "4"          # Minimum CPUs (4 cores)
      # Limits: Maximum resources allowed
      limits:
        memory: "32Gi"    # Pod killed if exceeds
        cpu: "8"          # Throttled if exceeds
```

**QoS Classes:**

**1. Guaranteed (highest priority):**
```yaml
# Requests == Limits
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
  limits:
    memory: "16Gi"
    cpu: "4"
```

**2. Burstable:**
```yaml
# Requests < Limits
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
# No requests or limits
# Pod evicted first under pressure
```

### Health Probes

**Kubernetes checks container health** using probes.

**1. Liveness Probe:** Is container alive?
```yaml
spec:
  containers:
  - name: model-server
    image: ml-api:v1
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30  # Wait before first check
      periodSeconds: 10        # Check every 10s
      timeoutSeconds: 5        # Timeout after 5s
      failureThreshold: 3      # Restart after 3 failures
```

**2. Readiness Probe:** Is container ready for traffic?
```yaml
spec:
  containers:
  - name: model-server
    image: ml-api:v1
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5
      successThreshold: 1      # Consider ready after 1 success
```

**3. Startup Probe:** Has container started? (new in 1.16)
```yaml
# For slow-starting applications (large ML models)
spec:
  containers:
  - name: llm-server
    image: llama-70b:v1
    startupProbe:
      httpGet:
        path: /startup
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 10
      failureThreshold: 30      # 5 minutes (30 * 10s)
```

**Probe Types:**
```yaml
# HTTP GET
livenessProbe:
  httpGet:
    path: /health
    port: 8080

# TCP Socket
livenessProbe:
  tcpSocket:
    port: 8080

# Command Execution
livenessProbe:
  exec:
    command:
    - cat
    - /tmp/healthy
```

## ReplicaSets

**Ensures a specified number of pod replicas are running.**

### ReplicaSet Manifest

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: ml-api-replicaset
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: model-server
        image: ml-api:v1
        ports:
        - containerPort: 8080
```

**How ReplicaSets Work:**
```
1. ReplicaSet controller watches API server
2. Counts pods matching selector
3. If count < replicas: Create pods
4. If count > replicas: Delete extra pods
5. Updates status
```

**Label Selectors:**
```yaml
# matchLabels (AND logic)
selector:
  matchLabels:
    app: ml-api
    env: production

# matchExpressions (more flexible)
selector:
  matchExpressions:
  - key: app
    operator: In
    values: [ml-api, ml-api-v2]
  - key: env
    operator: NotIn
    values: [dev, test]
```

**Note:** Typically you don't create ReplicaSets directly - Deployments manage them.

## Deployments

**The standard way to deploy applications** in Kubernetes.

### Why Deployments?

Deployments provide:
- **Declarative updates** for pods and ReplicaSets
- **Rollout history** and versioning
- **Rollback** to previous versions
- **Scaling**
- **Pause and resume** rollouts

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-deployment
  labels:
    app: ml-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max extra pods during update
      maxUnavailable: 0  # Max unavailable pods
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        version: v1
    spec:
      containers:
      - name: model-server
        image: ml-api:v1
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Deployment Strategies

**1. RollingUpdate (default):**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1        # Create 1 extra pod
    maxUnavailable: 0  # No downtime
```

Process:
```
Initial: [v1] [v1] [v1]
Step 1:  [v1] [v1] [v1] [v2]  (maxSurge: +1)
Step 2:  [v1] [v1] [v2] [v2]  (delete old, create new)
Step 3:  [v1] [v2] [v2] [v2]
Final:   [v2] [v2] [v2]
```

**2. Recreate:**
```yaml
strategy:
  type: Recreate
```

Process:
```
Initial: [v1] [v1] [v1]
Step 1:  (delete all)
Step 2:  [v2] [v2] [v2]  (create new)
# Downtime during transition
```

### Deployment Operations

**Create deployment:**
```bash
kubectl apply -f deployment.yaml

# Or imperative
kubectl create deployment ml-api --image=ml-api:v1 --replicas=3
```

**Update deployment (new model version):**
```bash
# Update image
kubectl set image deployment/ml-api-deployment model-server=ml-api:v2

# Or edit directly
kubectl edit deployment ml-api-deployment

# Or apply updated YAML
kubectl apply -f deployment.yaml
```

**Watch rollout:**
```bash
kubectl rollout status deployment/ml-api-deployment

# Output:
# Waiting for deployment "ml-api-deployment" rollout to finish: 1 out of 3 new replicas have been updated...
# Waiting for deployment "ml-api-deployment" rollout to finish: 2 out of 3 new replicas have been updated...
# deployment "ml-api-deployment" successfully rolled out
```

**Rollout history:**
```bash
kubectl rollout history deployment/ml-api-deployment

# REVISION  CHANGE-CAUSE
# 1         <none>
# 2         kubectl set image deployment/ml-api-deployment model-server=ml-api:v2
# 3         kubectl set image deployment/ml-api-deployment model-server=ml-api:v3
```

**Rollback:**
```bash
# Rollback to previous version
kubectl rollout undo deployment/ml-api-deployment

# Rollback to specific revision
kubectl rollout undo deployment/ml-api-deployment --to-revision=2
```

**Pause/Resume:**
```bash
# Pause rollout (for canary testing)
kubectl rollout pause deployment/ml-api-deployment

# Make changes
kubectl set image deployment/ml-api-deployment model-server=ml-api:v4

# Resume rollout
kubectl rollout resume deployment/ml-api-deployment
```

**Scale:**
```bash
# Imperative scaling
kubectl scale deployment/ml-api-deployment --replicas=5

# Declarative scaling
kubectl patch deployment ml-api-deployment -p '{"spec":{"replicas":5}}'
```

## Services

**Stable networking abstraction** for accessing pods.

### Why Services?

**Problem:** Pods are ephemeral
- Pod IPs change when pods restart
- Multiple pod replicas have different IPs
- How do clients find and connect?

**Solution:** Service provides:
- Stable virtual IP (ClusterIP)
- DNS name
- Load balancing across pods

### Service Types

**1. ClusterIP (default):**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  type: ClusterIP
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80        # Service port
    targetPort: 8080  # Container port
```

Accessible only within cluster:
```bash
curl http://ml-api-service.default.svc.cluster.local/predict
```

**2. NodePort:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-nodeport
spec:
  type: NodePort
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
    nodePort: 30080  # Exposed on all nodes (30000-32767)
```

Accessible via any node:
```bash
curl http://<node-ip>:30080/predict
```

**3. LoadBalancer:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-lb
spec:
  type: LoadBalancer
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

Creates cloud load balancer (ELB, GCE LB, Azure LB):
```bash
kubectl get svc ml-api-lb
# EXTERNAL-IP: 203.0.113.42

curl http://203.0.113.42/predict
```

**4. ExternalName:**
```yaml
# Map to external service
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: my-database.cxxxxxxxxxxxx.us-west-2.rds.amazonaws.com
```

### Service Discovery

**1. Environment Variables:**
```bash
# Kubernetes injects service info
ML_API_SERVICE_HOST=10.96.0.42
ML_API_SERVICE_PORT=80
```

**2. DNS (preferred):**
```bash
# Format: <service-name>.<namespace>.svc.cluster.local

# Same namespace
curl http://ml-api-service/predict

# Different namespace
curl http://ml-api-service.production.svc.cluster.local/predict

# Fully qualified
curl http://ml-api-service.production.svc.cluster.local/predict
```

### Endpoints

Services route traffic to **endpoints** (pod IPs).

```bash
# View service endpoints
kubectl get endpoints ml-api-service

# NAME             ENDPOINTS                           AGE
# ml-api-service   10.244.1.5:8080,10.244.2.6:8080    5m

# If no endpoints, service selector doesn't match any pods
```

### Headless Services

**No load balancing, return pod IPs directly.**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-training-headless
spec:
  clusterIP: None  # Headless
  selector:
    app: ml-training
  ports:
  - port: 8080
```

DNS returns all pod IPs:
```bash
nslookup ml-training-headless.default.svc.cluster.local

# Returns:
# 10.244.1.5
# 10.244.1.6
# 10.244.1.7
```

**Use case:** Distributed training where workers need direct communication.

## ConfigMaps

**Store non-sensitive configuration** separate from code.

### Create ConfigMap

**From literal values:**
```bash
kubectl create configmap ml-config \
  --from-literal=MODEL_NAME=resnet \
  --from-literal=BATCH_SIZE=32
```

**From file:**
```bash
# config.properties
MODEL_NAME=resnet
BATCH_SIZE=32
NUM_WORKERS=4

kubectl create configmap ml-config --from-file=config.properties
```

**From YAML:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
data:
  MODEL_NAME: "resnet"
  BATCH_SIZE: "32"
  config.yaml: |
    model:
      name: resnet
      version: v1
      batch_size: 32
    inference:
      timeout: 30
      max_concurrent: 100
```

### Use ConfigMap

**1. Environment variables:**
```yaml
spec:
  containers:
  - name: model-server
    image: ml-api:v1
    env:
    - name: MODEL_NAME
      valueFrom:
        configMapKeyRef:
          name: ml-config
          key: MODEL_NAME

    # Or all keys as env vars
    envFrom:
    - configMapRef:
        name: ml-config
```

**2. Volume mount:**
```yaml
spec:
  containers:
  - name: model-server
    image: ml-api:v1
    volumeMounts:
    - name: config
      mountPath: /etc/config
      readOnly: true
  volumes:
  - name: config
    configMap:
      name: ml-config
```

Files created:
```
/etc/config/MODEL_NAME (contains: resnet)
/etc/config/BATCH_SIZE (contains: 32)
/etc/config/config.yaml (contains: full YAML)
```

## Secrets

**Store sensitive data** (passwords, tokens, keys).

### Create Secret

**From literal:**
```bash
kubectl create secret generic ml-api-secret \
  --from-literal=API_KEY=sk-1234567890abcdef \
  --from-literal=DB_PASSWORD=my-secure-password
```

**From file:**
```bash
kubectl create secret generic tls-secret \
  --from-file=tls.crt=server.crt \
  --from-file=tls.key=server.key
```

**From YAML:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-api-secret
type: Opaque
data:
  # Base64 encoded values
  API_KEY: c2stMTIzNDU2Nzg5MGFiY2RlZg==
  DB_PASSWORD: bXktc2VjdXJlLXBhc3N3b3Jk
```

**Encode/decode:**
```bash
# Encode
echo -n "my-password" | base64
# bXktcGFzc3dvcmQ=

# Decode
echo "bXktcGFzc3dvcmQ=" | base64 --decode
# my-password
```

### Use Secret

**1. Environment variable:**
```yaml
spec:
  containers:
  - name: model-server
    env:
    - name: API_KEY
      valueFrom:
        secretKeyRef:
          name: ml-api-secret
          key: API_KEY
```

**2. Volume mount:**
```yaml
spec:
  containers:
  - name: model-server
    volumeMounts:
    - name: secrets
      mountPath: /etc/secrets
      readOnly: true
  volumes:
  - name: secrets
    secret:
      secretName: ml-api-secret
```

### Secret Types

```yaml
# Generic (Opaque)
type: Opaque

# Docker registry credentials
type: kubernetes.io/dockerconfigjson

# TLS certificate
type: kubernetes.io/tls

# Service account token
type: kubernetes.io/service-account-token
```

**Docker registry secret:**
```bash
kubectl create secret docker-registry regcred \
  --docker-server=myregistry.azurecr.io \
  --docker-username=myuser \
  --docker-password=mypassword \
  --docker-email=myemail@example.com
```

**Use in pod:**
```yaml
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: model-server
    image: myregistry.azurecr.io/ml-api:v1
```

## Namespaces

**Logical isolation** within a cluster.

### Default Namespaces

```bash
# List namespaces
kubectl get namespaces

# NAME              STATUS   AGE
# default           Active   10d
# kube-system       Active   10d (system components)
# kube-public       Active   10d (publicly readable)
# kube-node-lease   Active   10d (node heartbeats)
```

### Create Namespace

```bash
# Imperative
kubectl create namespace ml-team-a

# Declarative
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team-b
EOF
```

### Use Namespace

```bash
# Deploy to namespace
kubectl apply -f deployment.yaml -n ml-team-a

# Set default namespace
kubectl config set-context --current --namespace=ml-team-a

# View resources in namespace
kubectl get pods -n ml-team-a

# View all namespaces
kubectl get pods --all-namespaces
```

### Namespace Use Cases

**1. Multi-tenancy:**
```
- namespace: team-a
- namespace: team-b
- namespace: team-c
```

**2. Environment separation:**
```
- namespace: dev
- namespace: staging
- namespace: production
```

**3. Resource isolation:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team-a
spec:
  hard:
    requests.cpu: "100"
    requests.memory: "200Gi"
    requests.nvidia.com/gpu: "4"
    pods: "50"
```

**4. Network policies:**
```yaml
# Only allow traffic within namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-external
  namespace: ml-team-a
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector: {}
```

## Labels and Selectors

**Labels organize resources**, selectors query them.

### Labels

```yaml
metadata:
  labels:
    app: ml-api
    version: v1
    env: production
    team: ml-platform
    cost-center: engineering
```

### Label Operations

```bash
# Add label
kubectl label pod my-pod version=v2

# Update label
kubectl label pod my-pod version=v3 --overwrite

# Remove label
kubectl label pod my-pod version-

# View labels
kubectl get pods --show-labels

# Filter by label
kubectl get pods -l app=ml-api
kubectl get pods -l app=ml-api,version=v1
kubectl get pods -l 'app in (ml-api, ml-batch)'
kubectl get pods -l 'version!=v1'
```

### Label Selectors in Resources

```yaml
# Service selects pods
spec:
  selector:
    app: ml-api
    version: v1

# Deployment manages pods
spec:
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api  # Must match selector
```

## Hands-On: Deploy ML API with ConfigMap

**Step 1: Create ConfigMap**

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
data:
  MODEL_NAME: "resnet50"
  BATCH_SIZE: "32"
  LOG_LEVEL: "INFO"
EOF
```

**Step 2: Create Secret**

```bash
kubectl create secret generic ml-secret \
  --from-literal=API_KEY=sk-test-key-12345
```

**Step 3: Create Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        version: v1
    spec:
      containers:
      - name: model-server
        image: ml-api:v1
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: ml-config
              key: MODEL_NAME
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: ml-secret
              key: API_KEY
        envFrom:
        - configMapRef:
            name: ml-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

**Step 4: Create Service**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  type: LoadBalancer
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

**Step 5: Test**

```bash
# Get service URL (minikube)
URL=$(minikube service ml-api-service --url)

# Test endpoint
curl $URL/predict

# View logs
kubectl logs -l app=ml-api --tail=20

# Exec into pod
kubectl exec -it $(kubectl get pod -l app=ml-api -o name | head -1) -- /bin/bash

# Inside pod, check environment
echo $MODEL_NAME
echo $BATCH_SIZE
```

## Summary

In this lesson, you've mastered:

- ✅ Pods: atomic unit, multi-container patterns, init containers
- ✅ Resource requests and limits for ML workloads
- ✅ Health probes: liveness, readiness, startup
- ✅ ReplicaSets: maintain desired replicas
- ✅ Deployments: declarative updates, rolling updates, rollbacks
- ✅ Services: stable networking, types (ClusterIP, NodePort, LoadBalancer)
- ✅ ConfigMaps: non-sensitive configuration
- ✅ Secrets: sensitive data management
- ✅ Namespaces: logical isolation and multi-tenancy
- ✅ Labels and selectors: resource organization

## What's Next?

In **Lesson 04: Deployments and Services**, you'll learn:
- Advanced deployment strategies
- Blue-green and canary deployments
- Service mesh introduction
- Advanced health checks

## Self-Check Questions

1. What's the difference between a pod and a container?
2. When would you use multi-container pods?
3. What's the difference between requests and limits?
4. How do rolling updates work?
5. What are the different service types and when to use each?
6. What's the difference between ConfigMaps and Secrets?

## Additional Resources

- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Services](https://kubernetes.io/docs/concepts/services-networking/service/)
- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

---

**Next Lesson:** [04-deployments-services.md](./04-deployments-services.md) - Advanced deployment strategies and service patterns
