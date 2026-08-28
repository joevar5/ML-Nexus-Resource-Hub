# Lesson 01: Kubernetes Introduction

**Duration:** 6 hours
**Objectives:** Understand why Kubernetes is critical for ML infrastructure, when to use it, and deploy your first K8s cluster

## Welcome to Kubernetes!

You've mastered Docker and can containerize ML applications. Now it's time to orchestrate those containers at scale with Kubernetes - the industry-standard platform for container orchestration.

In this lesson, you'll understand why Kubernetes has become essential for ML infrastructure, explore the Kubernetes ecosystem, and get hands-on with your first local cluster.

## The Container Orchestration Challenge

### The Problem: Managing Containers at Scale

Imagine you've containerized an ML model serving application. Great! But now:

**Scenario 1: Scaling**
- Traffic spikes from 100 to 10,000 requests/second
- You need to manually start 50 more containers across 10 servers
- How do you distribute them? How do you load balance?

**Scenario 2: Reliability**
- One server crashes, taking down 5 containers
- How do you detect failures and restart containers automatically?
- How do you maintain desired state?

**Scenario 3: Updates**
- New model version needs deployment
- How do you deploy without downtime?
- How do you rollback if something breaks?

**Scenario 4: Resource Management**
- You have 100 different ML models to serve
- Each has different CPU/GPU/memory requirements
- How do you optimally pack containers onto servers?

**Without orchestration, you'd need:**
- Custom scripts for container management
- Manual intervention for failures
- Complex load balancing configuration
- Home-grown scheduling logic
- Manual capacity planning

**This is where Kubernetes excels.**

## What is Kubernetes?

**Kubernetes (K8s)** is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.

### Etymology
The name comes from Greek κυβερνήτης (kubernetes), meaning "helmsman" or "pilot." The "K8s" abbreviation replaces the 8 letters "ubernete" with "8".

### Created by Google, Now CNCF

- **Origins:** Google's internal system "Borg" (ran for 15+ years)
- **Open-sourced:** 2014
- **Donated to CNCF:** 2015 (Cloud Native Computing Foundation)
- **Current state:** De facto standard for container orchestration

### Core Capabilities

Kubernetes provides:

1. **Automated deployment and scaling** - Declare desired state, K8s maintains it
2. **Self-healing** - Automatically replaces and reschedules containers
3. **Service discovery and load balancing** - Built-in networking
4. **Storage orchestration** - Automatic mounting of storage systems
5. **Automated rollouts and rollbacks** - Safe deployments with zero downtime
6. **Secret and configuration management** - Secure handling of sensitive data
7. **Resource optimization** - Efficient bin-packing of workloads

## Why Kubernetes for ML Infrastructure?

### ML-Specific Benefits

**1. GPU Scheduling**
```yaml
# Request GPU in pod specification
resources:
  limits:
    nvidia.com/gpu: 1
```
Kubernetes automatically schedules pods on GPU nodes and manages GPU allocation.

**2. Scalable Model Serving**
- Handle variable traffic patterns
- Auto-scale from 1 to 100+ replicas
- Load balance across model instances
- Rolling updates for new model versions

**3. Multi-Model Management**
- Deploy 100+ different models
- Each with independent scaling
- Resource isolation between models
- Different resource requirements per model

**4. Training Job Orchestration**
- Run distributed training jobs
- Automatic retry on failure
- Priority-based scheduling
- Preemptible instances for cost savings

**5. Reproducibility**
- Infrastructure as code (YAML manifests)
- Version control for deployments
- Consistent dev/staging/production environments

### Real-World ML Infrastructure on Kubernetes

**Uber: Michelangelo ML Platform**
- 1,000+ ML models deployed on Kubernetes
- Automatic scaling based on traffic patterns
- GPU scheduling for training workloads
- Multi-tenant platform serving multiple teams
- **Result:** 95% reduction in deployment time

**Spotify: ML Feature Platform**
- 200+ Kubernetes clusters globally
- Serving billions of predictions daily
- Auto-scaling during peak hours (3x capacity)
- Cost savings: 70% reduction through efficient resource utilization

**OpenAI: LLM Infrastructure**
- Large-scale GPU clusters on Kubernetes
- Distributed training for GPT models
- Inference serving with thousands of requests/second
- Dynamic resource allocation for research vs. production

**Airbnb: Search and Recommendation**
- 500+ microservices on Kubernetes
- ML models for search ranking, pricing, recommendations
- Canary deployments for safe model updates
- Global deployment across 10+ regions

## Kubernetes vs. Alternatives

### Docker Compose

**Docker Compose:**
```yaml
# docker-compose.yml
services:
  model-server:
    image: my-ml-model:latest
    ports:
      - "8080:8080"
    deploy:
      replicas: 3
```

**Limitations:**
- Single-host only (cannot span multiple servers)
- No built-in self-healing
- Manual scaling
- Limited load balancing
- No rolling updates

**Use Docker Compose when:**
- Development and testing
- Single-machine deployments
- Simple multi-container apps
- Local ML experiments

### Docker Swarm

**Docker Swarm:**
- Built into Docker Engine
- Simpler than Kubernetes
- Multi-host orchestration

**Limitations:**
- Smaller ecosystem
- Less community support
- Fewer advanced features
- Not industry standard for ML

**Market Reality:** Kubernetes has ~90% market share in container orchestration.

### Cloud-Native Services (AWS ECS, Google Cloud Run)

**Pros:**
- Managed service (less operational overhead)
- Tight cloud integration
- Simpler for basic use cases

**Cons:**
- Vendor lock-in
- Limited customization
- Not portable across clouds
- Higher costs at scale

**Use K8s when:**
- Multi-cloud or cloud portability needed
- Complex ML workflows
- Large scale (100+ services)
- Need full control and customization

### Comparison Matrix

| Feature | Docker Compose | Docker Swarm | Kubernetes | Cloud-Native (ECS/Run) |
|---------|---------------|--------------|------------|------------------------|
| Multi-host | No | Yes | Yes | Yes |
| Auto-healing | No | Basic | Advanced | Yes |
| Load balancing | Basic | Yes | Advanced | Yes |
| GPU support | Manual | Manual | Native | Limited |
| Ecosystem | Small | Medium | Massive | Cloud-specific |
| Learning curve | Easy | Medium | Steep | Easy |
| ML use cases | Dev/test | Small prod | Enterprise | Simple prod |
| Cost (managed) | Free | Free | Free (self-hosted) | Pay per use |

## When to Use Kubernetes (and When Not To)

### ✅ Use Kubernetes When:

**1. Scale Requirements**
- Serving 1000+ requests/second
- Need to scale from 1 to 100+ instances
- Variable traffic patterns

**2. High Availability Needs**
- 99.9%+ uptime requirements
- Automatic failover critical
- Multi-region deployments

**3. Complex Infrastructure**
- 10+ microservices
- Multiple ML models
- Mix of training and serving workloads

**4. Team Size**
- 5+ engineers who can learn K8s
- Dedicated infrastructure team
- Organization committed to cloud-native

**5. Multi-Cloud or Portability**
- Running on multiple clouds
- Avoiding vendor lock-in
- On-premises + cloud hybrid

**6. Resource Optimization**
- High GPU costs (efficient scheduling needed)
- Variable workloads
- Multi-tenant platform

### ❌ Don't Use Kubernetes When:

**1. Simple Applications**
- Single model, low traffic (<100 req/s)
- No scaling requirements
- Prototypes and experiments

**2. Small Teams**
- 1-2 person team
- No time to learn K8s complexity
- Limited operational resources

**3. Cloud-Native Alternatives Sufficient**
- AWS Lambda for simple inference
- Google Cloud Run for containerized apps
- SageMaker for managed ML

**4. Serverless Works**
- Event-driven workloads
- Intermittent usage
- Cold-start latency acceptable

**5. Development/Testing**
- Docker Compose is simpler
- Local development environments
- Individual experimentation

### Decision Framework

```
Is your app containerized?
├─ No → Containerize first (Module 03)
└─ Yes
    ├─ Need multi-host orchestration?
    │   ├─ No → Use Docker Compose
    │   └─ Yes
    │       ├─ Simple use case + cloud-native acceptable?
    │       │   ├─ Yes → Consider Cloud Run, ECS, etc.
    │       │   └─ No → Kubernetes
    │       └─ Need multi-cloud or advanced features?
    │           └─ Yes → Kubernetes
```

## The Kubernetes Ecosystem (CNCF Landscape)

Kubernetes is the core, but the ecosystem is vast:

### Core Infrastructure
- **Kubernetes:** Orchestration
- **containerd/CRI-O:** Container runtimes
- **etcd:** Distributed key-value store

### Networking
- **Calico, Cilium, Flannel:** CNI plugins
- **Istio, Linkerd:** Service meshes
- **Ingress Nginx, Traefik:** Ingress controllers

### Storage
- **Rook, Longhorn:** Cloud-native storage
- **MinIO:** Object storage
- **CSI drivers:** Cloud storage integration

### Observability
- **Prometheus:** Metrics
- **Grafana:** Visualization
- **Jaeger, Zipkin:** Distributed tracing
- **Fluentd, Loki:** Log aggregation

### CI/CD
- **ArgoCD, Flux:** GitOps
- **Tekton:** Cloud-native CI/CD
- **Spinnaker:** Multi-cloud deployment

### ML-Specific
- **Kubeflow:** End-to-end ML platform
- **KServe:** Model serving
- **Seldon Core:** ML deployment
- **Argo Workflows:** DAG-based workflows

### Security
- **Falco:** Runtime security
- **OPA (Open Policy Agent):** Policy enforcement
- **cert-manager:** Certificate management
- **Vault:** Secrets management

## Kubernetes Distributions

Different flavors for different needs:

### Local Development
- **minikube:** Single-node cluster for learning
- **kind (Kubernetes in Docker):** Multi-node clusters in Docker
- **k3s:** Lightweight K8s (great for edge/IoT)
- **Docker Desktop:** Built-in K8s (macOS/Windows)
- **MicroK8s:** Lightweight, minimal K8s

### Production (Cloud)
- **GKE (Google Kubernetes Engine):** Google Cloud
- **EKS (Elastic Kubernetes Service):** AWS
- **AKS (Azure Kubernetes Service):** Microsoft Azure
- **DOKS (DigitalOcean Kubernetes):** DigitalOcean
- **LKE (Linode Kubernetes Engine):** Linode

### On-Premises
- **OpenShift:** Red Hat's enterprise Kubernetes
- **Rancher:** Multi-cluster management
- **Tanzu:** VMware's Kubernetes platform
- **Charmed Kubernetes:** Canonical's distribution

### For ML Workloads
- **NVIDIA DGX Cloud:** GPU-optimized K8s
- **Run:ai:** GPU virtualization on K8s
- **Determined AI:** ML training platform on K8s

## Installation: Local Kubernetes Cluster

For learning and development, we'll use **minikube** (most popular) or **kind** (faster, more like production).

### Option 1: Minikube (Recommended for Beginners)

**Prerequisites:**
- Docker installed
- 2 CPUs, 4GB RAM available
- Virtualization support

**Installation (macOS):**
```bash
# Install minikube
brew install minikube

# Start cluster
minikube start

# Verify
kubectl cluster-info
kubectl get nodes
```

**Installation (Linux):**
```bash
# Download and install
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster
minikube start

# Verify
kubectl cluster-info
```

**Installation (Windows):**
```powershell
# Using Chocolatey
choco install minikube

# Or download installer from GitHub
# https://github.com/kubernetes/minikube/releases

# Start cluster
minikube start
```

### Option 2: kind (Kubernetes in Docker)

**Prerequisites:**
- Docker installed and running
- kubectl installed

**Installation:**
```bash
# macOS
brew install kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Windows
choco install kind

# Create cluster
kind create cluster --name ml-learning

# Verify
kubectl cluster-info --context kind-ml-learning
```

### Option 3: Docker Desktop Kubernetes

**Easiest for macOS/Windows users:**

1. Install Docker Desktop
2. Settings → Kubernetes → Enable Kubernetes
3. Apply & Restart
4. Verify: `kubectl cluster-info`

### Verify Installation

```bash
# Check kubectl version
kubectl version --client

# Check cluster info
kubectl cluster-info

# Check nodes
kubectl get nodes

# Expected output:
# NAME       STATUS   ROLES           AGE   VERSION
# minikube   Ready    control-plane   1m    v1.28.3
```

## Hands-On: Deploy Your First Application to Kubernetes

Let's deploy a simple ML inference API to Kubernetes.

### Step 1: Create a Simple Flask ML API

**app.py:**
```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simulate ML prediction
    input_data = np.array(data['features'])
    prediction = float(np.sum(input_data))  # Dummy model

    return jsonify({
        'prediction': prediction,
        'model_version': 'v1.0'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

**requirements.txt:**
```
flask==3.0.0
numpy==1.26.0
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
```

### Step 2: Build and Push Docker Image

```bash
# Build image
docker build -t ml-api:v1 .

# Test locally
docker run -p 8080:8080 ml-api:v1

# Test endpoint (in another terminal)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4, 5]}'

# Stop container
docker stop $(docker ps -q --filter ancestor=ml-api:v1)

# Load image into minikube (if using minikube)
minikube image load ml-api:v1

# Or for kind
kind load docker-image ml-api:v1 --name ml-learning
```

### Step 3: Create Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  replicas: 3  # Run 3 copies
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:v1
        imagePullPolicy: Never  # Use local image
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
```

**Apply deployment:**
```bash
kubectl apply -f deployment.yaml

# Check deployment
kubectl get deployments

# Check pods
kubectl get pods

# Expected output:
# NAME                      READY   STATUS    RESTARTS   AGE
# ml-api-xxxxx-xxxxx        1/1     Running   0          10s
# ml-api-xxxxx-xxxxx        1/1     Running   0          10s
# ml-api-xxxxx-xxxxx        1/1     Running   0          10s
```

### Step 4: Create Service for Load Balancing

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  type: LoadBalancer  # Expose externally
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

**Apply service:**
```bash
kubectl apply -f service.yaml

# Check service
kubectl get services

# For minikube, expose service
minikube service ml-api-service --url

# For kind, use port-forward
kubectl port-forward service/ml-api-service 8080:80
```

### Step 5: Test the Deployed Application

```bash
# Get service URL (minikube)
URL=$(minikube service ml-api-service --url)

# Test prediction
curl -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4, 5]}'

# Expected output:
# {"prediction": 15.0, "model_version": "v1.0"}

# Test health check
curl $URL/health

# Expected output:
# {"status": "healthy"}
```

### Step 6: Explore and Scale

```bash
# View pod details
kubectl describe pod ml-api-xxxxx-xxxxx

# View logs
kubectl logs -f ml-api-xxxxx-xxxxx

# Scale up to 5 replicas
kubectl scale deployment ml-api --replicas=5

# Watch scaling in action
kubectl get pods -w

# Scale back down
kubectl scale deployment ml-api --replicas=3
```

### Step 7: Update Deployment (Rolling Update)

Update **app.py** to change version:
```python
return jsonify({
    'prediction': prediction,
    'model_version': 'v2.0'  # Changed
})
```

```bash
# Rebuild image
docker build -t ml-api:v2 .

# Load into cluster
minikube image load ml-api:v2
# or: kind load docker-image ml-api:v2

# Update deployment
kubectl set image deployment/ml-api ml-api=ml-api:v2

# Watch rolling update
kubectl rollout status deployment/ml-api

# Test new version
curl -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3]}'

# Rollback if needed
kubectl rollout undo deployment/ml-api
```

### Step 8: Cleanup

```bash
# Delete resources
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml

# Or delete everything with label
kubectl delete all -l app=ml-api

# Stop minikube (optional)
minikube stop

# Delete cluster
minikube delete
```

## Key Concepts Introduced

Through this hands-on exercise, you've encountered:

1. **Deployments:** Manage replicated pods
2. **Pods:** Smallest deployable unit (container wrapper)
3. **Services:** Load balancing and networking
4. **ReplicaSets:** Maintain desired pod count (managed by Deployment)
5. **Rolling updates:** Zero-downtime deployments
6. **Scaling:** Horizontal scaling with replicas
7. **kubectl:** Command-line tool for K8s

We'll dive deep into each in upcoming lessons.

## Common Issues and Troubleshooting

### Issue 1: Pods Not Starting

```bash
# Check pod status
kubectl get pods

# If status is ImagePullBackOff or ErrImagePull
# Solution: Ensure image is loaded into cluster
minikube image load your-image:tag
# or
kind load docker-image your-image:tag

# If status is CrashLoopBackOff
# Check logs
kubectl logs pod-name

# Check events
kubectl describe pod pod-name
```

### Issue 2: Service Not Accessible

```bash
# Check service
kubectl get svc

# For minikube, use service URL
minikube service service-name --url

# For kind/other, use port-forward
kubectl port-forward service/service-name local-port:service-port
```

### Issue 3: Insufficient Resources

```bash
# Check node resources
kubectl describe node

# Adjust resource requests in deployment
# Lower memory/cpu requests if needed
```

## Summary

In this lesson, you've learned:

- ✅ Why Kubernetes is essential for ML infrastructure at scale
- ✅ How Kubernetes compares to Docker Compose and alternatives
- ✅ When to use Kubernetes (and when not to)
- ✅ The Kubernetes ecosystem and CNCF landscape
- ✅ How to install local Kubernetes (minikube/kind)
- ✅ Deploy your first ML application to Kubernetes
- ✅ Scale deployments and perform rolling updates

## What's Next?

In **Lesson 02: Kubernetes Architecture**, you'll learn:
- Control plane components and how they work
- Node components and pod lifecycle
- How Kubernetes makes scheduling decisions
- API server and etcd internals

## Self-Check Questions

Before proceeding, ensure you can answer:

1. What problems does Kubernetes solve for ML infrastructure?
2. When would you choose Kubernetes over simpler alternatives?
3. What is a pod? What is a deployment?
4. How do you scale a deployment?
5. What is a service and why is it needed?
6. How do rolling updates work?

## Additional Resources

**Official Documentation:**
- [Kubernetes Concepts](https://kubernetes.io/docs/concepts/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Interactive Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)

**Tutorials:**
- [Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way)
- [CNCF Kubernetes Fundamentals](https://www.cncf.io/certification/training/)

**Books:**
- "Kubernetes in Action" by Marko Lukša
- "Kubernetes Patterns" by Bilgin Ibryam and Roland Huß

**ML-Specific:**
- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [NVIDIA GPU Operator for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/getting-started.html)

---

**Next Lesson:** [02-k8s-architecture.md](./02-k8s-architecture.md) - Understanding Kubernetes architecture and components
