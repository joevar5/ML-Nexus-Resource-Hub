# Lesson 02: Kubernetes Architecture

**Duration:** 8 hours
**Objectives:** Understand Kubernetes architecture, control plane components, node components, and how they work together

## Introduction

In Lesson 01, you deployed your first application to Kubernetes. Now it's time to understand what's happening under the hood.

Kubernetes is a distributed system with multiple components working together to orchestrate containers. Understanding this architecture is crucial for:
- **Troubleshooting** production issues
- **Optimizing** performance
- **Designing** scalable ML infrastructure
- **Passing** CKA/CKAD certifications

## High-Level Architecture

Kubernetes follows a **master-worker architecture** (now called **control plane-worker nodes**):

```
┌─────────────────────────────────────────────────────────────────┐
│                     KUBERNETES CLUSTER                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              CONTROL PLANE (Master)                      │  │
│  │  ┌────────────┐  ┌─────────┐  ┌───────────┐            │  │
│  │  │ API Server │  │  etcd   │  │ Scheduler │            │  │
│  │  └────────────┘  └─────────┘  └───────────┘            │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐    │  │
│  │  │ Controller Manager   │  │ Cloud Controller     │    │  │
│  │  └──────────────────────┘  └──────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ (API calls)                      │
│                              │                                  │
│  ┌───────────────┬───────────┴───────────┬───────────────┐    │
│  │  Worker Node 1 │   Worker Node 2      │  Worker Node 3 │    │
│  │ ┌────────────┐ │  ┌────────────┐      │ ┌────────────┐│    │
│  │ │  kubelet   │ │  │  kubelet   │      │ │  kubelet   ││    │
│  │ │ kube-proxy │ │  │ kube-proxy │      │ │ kube-proxy ││    │
│  │ │ Container  │ │  │ Container  │      │ │ Container  ││    │
│  │ │  Runtime   │ │  │  Runtime   │      │ │  Runtime   ││    │
│  │ └────────────┘ │  └────────────┘      │ └────────────┘│    │
│  │  [Pods...]     │   [Pods...]          │  [Pods...]    │    │
│  └───────────────┴───────────────────────┴───────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Declarative Configuration:** You declare desired state, K8s maintains it
2. **Controller Pattern:** Controllers watch state and reconcile differences
3. **API-Driven:** All components communicate via the API server
4. **Distributed and Resilient:** No single point of failure
5. **Pluggable:** Many components can be swapped (CNI, CSI, CRI)

## Control Plane Components

The **control plane** makes global decisions about the cluster (scheduling, responding to events, maintaining desired state).

### 1. API Server (kube-apiserver)

**The brain of Kubernetes** - central management entity.

**Purpose:**
- Exposes Kubernetes API (REST API)
- Front-end for control plane
- All cluster communication goes through API server
- Validates and configures data for API objects

**Key Responsibilities:**
- Authentication and authorization
- Admission control (webhooks, policies)
- API resource validation
- Serves as gateway to etcd
- Provides watch mechanism for changes

**How it works:**
```
kubectl → API Server → etcd (store)
                     ↓
              Controllers watch API
                     ↓
              Take actions
```

**Interaction Example:**
```bash
# When you run:
kubectl apply -f deployment.yaml

# What happens:
# 1. kubectl sends HTTP POST to API server
# 2. API server authenticates and authorizes request
# 3. API server validates deployment spec
# 4. API server runs admission controllers
# 5. API server writes to etcd
# 6. Deployment controller watches API, sees new deployment
# 7. Controller creates ReplicaSet
# 8. ReplicaSet controller creates Pods
# 9. Scheduler assigns Pods to nodes
# 10. Kubelet on nodes start containers
```

**API Server as Bottleneck:**
For large ML platforms (100+ nodes, 1000+ pods):
- API server can become a bottleneck
- Horizontal scaling: run multiple API server instances
- Use caching and reduce API calls

**Production Configuration:**
```yaml
# High-availability API server
apiVersion: v1
kind: Pod
metadata:
  name: kube-apiserver
spec:
  containers:
  - name: kube-apiserver
    command:
    - kube-apiserver
    - --etcd-servers=https://etcd1:2379,https://etcd2:2379,https://etcd3:2379
    - --enable-admission-plugins=NodeRestriction,PodSecurityPolicy
    - --audit-log-path=/var/log/kubernetes/audit.log
    - --audit-log-maxage=30
```

### 2. etcd

**Distributed key-value store** - the cluster's database.

**Purpose:**
- Stores all cluster state
- Consistent and highly-available
- Source of truth for cluster

**What's stored in etcd:**
- Cluster configuration
- Pod specifications
- Secrets and ConfigMaps
- Service endpoints
- Resource quotas
- All Kubernetes objects

**Key Properties:**
- **Consistency:** Uses Raft consensus algorithm
- **Reliability:** Requires 3 or 5 instances for HA
- **Watch:** Clients can watch for changes
- **Performance:** Critical path for cluster operations

**Architecture:**
```
┌──────────────────────────────────────┐
│         etcd Cluster (HA)            │
│  ┌──────┐    ┌──────┐    ┌──────┐   │
│  │etcd-1│ ←→ │etcd-2│ ←→ │etcd-3│   │
│  └──────┘    └──────┘    └──────┘   │
│     ↑            ↑           ↑       │
└─────┼────────────┼───────────┼───────┘
      └────────────┴───────────┘
               │
         API Server (reads/writes)
```

**Backup and Disaster Recovery:**
```bash
# Backup etcd (CRITICAL for production)
ETCDCTL_API=3 etcdctl snapshot save snapshot.db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# Verify snapshot
ETCDCTL_API=3 etcdctl snapshot status snapshot.db

# Restore (disaster recovery)
ETCDCTL_API=3 etcdctl snapshot restore snapshot.db \
  --data-dir=/var/lib/etcd-restore
```

**Performance Considerations for ML:**
- etcd has ~8GB default storage limit
- Large ConfigMaps (datasets) can fill etcd
- Store large data externally (S3, NFS)
- Monitor etcd size and performance

### 3. Scheduler (kube-scheduler)

**Assigns pods to nodes** based on resource requirements and constraints.

**Purpose:**
- Watch for newly created pods with no assigned node
- Select best node for each pod
- Update pod with node assignment

**Scheduling Process:**
```
1. Filter: Remove nodes that don't meet requirements
   - Insufficient CPU/memory
   - GPU not available
   - Node selectors don't match
   - Taints/tolerations

2. Score: Rank remaining nodes
   - Resource balance
   - Pod affinity/anti-affinity
   - Spread across zones
   - Custom scoring

3. Bind: Assign pod to highest-scored node
```

**Scheduling Factors:**
```yaml
# Example: ML training pod with GPU requirement
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  # Node selector: only GPU nodes
  nodeSelector:
    accelerator: nvidia-tesla-v100

  # Resource requirements
  containers:
  - name: trainer
    resources:
      requests:
        memory: "32Gi"
        cpu: "8"
        nvidia.com/gpu: "1"
      limits:
        memory: "64Gi"
        cpu: "16"
        nvidia.com/gpu: "1"

  # Affinity: prefer nodes in same zone as data
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: topology.kubernetes.io/zone
            operator: In
            values:
            - us-west-1a

  # Toleration: allow scheduling on tainted GPU nodes
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

**Custom Schedulers:**
For specialized ML workloads:
- **Gang scheduling:** Schedule all pods of distributed training job together
- **Bin packing:** Maximize node utilization
- **Priority scheduling:** High-priority inference over batch training

Example: Using custom scheduler
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  schedulerName: gpu-aware-scheduler  # Custom scheduler
  containers:
  - name: trainer
    image: training-image:latest
```

### 4. Controller Manager (kube-controller-manager)

**Runs controller processes** that regulate cluster state.

**Purpose:**
- Watch desired state (from API server)
- Observe current state
- Make changes to reach desired state

**Built-in Controllers:**

**Node Controller:**
- Monitor node health
- Respond to node failures
- Evict pods from unhealthy nodes

**ReplicaSet Controller:**
- Ensure correct number of pod replicas
- Create/delete pods as needed

**Deployment Controller:**
- Manage rolling updates
- Handle rollbacks

**Job Controller:**
- Create pods for batch jobs
- Track job completion

**Service Controller:**
- Create/update load balancers
- Manage endpoint objects

**Namespace Controller:**
- Delete all objects when namespace deleted

**Example: ReplicaSet Controller Logic**
```
Loop:
  1. Watch ReplicaSet objects
  2. For each ReplicaSet:
     a. Count current pods with matching labels
     b. Compare to desired replica count
     c. If current < desired: Create pods
     d. If current > desired: Delete extra pods
     e. Update ReplicaSet status
  3. Sleep/wait for changes
```

**Custom Controllers for ML:**
```python
# Pseudocode: Model Deployment Controller
while True:
    for model_deployment in watch_model_deployments():
        desired_version = model_deployment.spec.version
        current_version = get_current_version(model_deployment)

        if desired_version != current_version:
            # Rolling update to new model version
            create_new_pods(desired_version)
            wait_for_health_checks()
            delete_old_pods(current_version)
            update_status(model_deployment)
```

### 5. Cloud Controller Manager

**Integrates with cloud provider APIs** (AWS, GCP, Azure).

**Purpose:**
- Manage cloud-specific resources
- Route traffic
- Provision storage
- Manage load balancers

**Cloud-Specific Controllers:**

**Node Controller:**
- Check if node deleted in cloud
- Get node metadata (instance type, zone, IP)

**Route Controller:**
- Set up routes in cloud VPC
- Enable pod-to-pod networking across nodes

**Service Controller:**
- Create cloud load balancers for LoadBalancer services
- Update load balancer backends

**Volume Controller:**
- Create/attach/mount cloud volumes
- Handle volume snapshots

**Example: AWS Integration**
```yaml
# Service creates AWS ELB
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:..."
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8080
  selector:
    app: ml-api
```

Cloud controller automatically:
1. Creates Network Load Balancer in AWS
2. Configures SSL certificate
3. Updates target groups with pod IPs
4. Manages health checks

## Worker Node Components

Worker nodes run containerized applications.

### 1. kubelet

**Primary node agent** - ensures containers are running.

**Purpose:**
- Register node with API server
- Watch for pod assignments to node
- Ensure pod containers are running and healthy
- Report node and pod status

**Key Responsibilities:**
```
1. Pod Lifecycle Management:
   - Pull container images
   - Start containers via CRI
   - Monitor container health
   - Restart failed containers

2. Volume Management:
   - Mount volumes into pods
   - Manage volume lifecycles

3. Health Monitoring:
   - Run liveness probes
   - Run readiness probes
   - Report status to API server

4. Resource Management:
   - Enforce resource limits
   - Monitor resource usage
```

**kubelet Configuration for ML Workloads:**
```yaml
# /var/lib/kubelet/config.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration

# Increase image pull timeout for large ML images
imageGCHighThresholdPercent: 85
imageGCLowThresholdPercent: 80

# Resource reservation for system
systemReserved:
  cpu: "1000m"
  memory: "2Gi"

# Kubelet resource reservation
kubeReserved:
  cpu: "500m"
  memory: "1Gi"

# Eviction thresholds (prevent OOM)
evictionHard:
  memory.available: "500Mi"
  nodefs.available: "10%"

# GPU support
featureGates:
  DevicePlugins: true
```

**Interaction with Container Runtime:**
```
kubelet → CRI (Container Runtime Interface)
            ↓
      containerd / CRI-O
            ↓
      Container (running app)
```

### 2. kube-proxy

**Network proxy** running on each node.

**Purpose:**
- Implement Kubernetes Service abstraction
- Maintain network rules for pod communication
- Enable load balancing across pod replicas

**How Services Work:**
```
Client → Service IP (virtual) → kube-proxy rules → Pod IP (actual)
```

**Proxy Modes:**

**1. iptables mode (most common):**
```bash
# kube-proxy creates iptables rules
# Example: Service at 10.96.0.1:80 → Pods at 10.244.1.5:8080, 10.244.2.6:8080

# iptables rules (simplified):
-A KUBE-SERVICES -d 10.96.0.1/32 -p tcp -m tcp --dport 80 -j KUBE-SVC-XXXXX
-A KUBE-SVC-XXXXX -m statistic --mode random --probability 0.5 -j KUBE-SEP-POD1
-A KUBE-SVC-XXXXX -j KUBE-SEP-POD2
-A KUBE-SEP-POD1 -p tcp -j DNAT --to-destination 10.244.1.5:8080
-A KUBE-SEP-POD2 -p tcp -j DNAT --to-destination 10.244.2.6:8080
```

**2. IPVS mode (better performance):**
- Uses Linux IPVS (IP Virtual Server)
- Better for 1000+ services
- More load balancing algorithms

**3. userspace mode (legacy):**
- kube-proxy proxies traffic
- Slower, but works everywhere

**Configuration for ML:**
```yaml
# kube-proxy config for high-performance ML serving
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
mode: "ipvs"  # Better for many services
ipvs:
  scheduler: "lc"  # Least connection (good for long ML requests)
  minSyncPeriod: 5s
  syncPeriod: 30s
conntrack:
  maxPerCore: 131072  # Increase for high traffic
```

### 3. Container Runtime

**Runs containers** on the node.

**Container Runtime Interface (CRI):**
- Standardized interface between kubelet and runtime
- Pluggable architecture

**Popular Runtimes:**

**containerd (most common):**
- CNCF graduated project
- Used by GKE, EKS, AKS
- Lightweight and efficient

**CRI-O:**
- Designed specifically for Kubernetes
- OCI-compliant
- Used by OpenShift

**Docker (via dockershim, deprecated):**
- Kubernetes 1.24+ removed dockershim
- Use containerd instead

**GPU Support:**
```bash
# NVIDIA Container Runtime integration
# Allows containers to access GPUs

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure containerd
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

## Add-ons

Optional components that provide cluster features.

### DNS (CoreDNS)

**Purpose:**
- Service discovery via DNS
- Pods can find services by name

**How it works:**
```bash
# Pod can access service by name
curl http://ml-api-service.default.svc.cluster.local/predict

# DNS resolution:
# ml-api-service → Service IP → Pod IPs
```

**CoreDNS Configuration:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
```

### Dashboard

**Web UI** for Kubernetes cluster.

```bash
# Deploy dashboard
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Access dashboard
kubectl proxy
# Visit: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

**Alternative: Lens (Desktop App)**
- More user-friendly
- Better visualizations
- Multi-cluster management

### Metrics Server

**Collects resource metrics** (CPU, memory) from kubelets.

**Purpose:**
- Enable `kubectl top nodes` and `kubectl top pods`
- Required for Horizontal Pod Autoscaler (HPA)

```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# View node metrics
kubectl top nodes

# View pod metrics
kubectl top pods --all-namespaces

# View pod metrics with containers
kubectl top pods --containers
```

### Ingress Controller

**HTTP/HTTPS routing** to services.

Popular options:
- **NGINX Ingress:** Most popular
- **Traefik:** Cloud-native, automatic HTTPS
- **Istio:** Service mesh with ingress

## How Components Work Together

### Example: Deploying an ML Model

**Step-by-step flow:**

**1. You run:**
```bash
kubectl apply -f ml-deployment.yaml
```

**2. kubectl → API Server:**
- Sends HTTP POST with deployment spec
- API server authenticates and authorizes

**3. API Server → etcd:**
- Stores deployment object in etcd

**4. Deployment Controller (watching API):**
- Sees new deployment
- Creates ReplicaSet object
- Writes to API server

**5. ReplicaSet Controller (watching API):**
- Sees new ReplicaSet
- Creates 3 Pod objects (replicas: 3)
- Writes to API server

**6. Scheduler (watching for unscheduled pods):**
- Sees 3 new pods without node assignment
- Filters nodes (GPU available? Memory sufficient?)
- Scores remaining nodes
- Assigns each pod to best node
- Updates pod spec with node name

**7. kubelet on Node-1 (watching for pods assigned to it):**
- Sees new pod assignment
- Pulls container image
- Tells container runtime to start container
- Reports status to API server

**8. kube-proxy on Node-1:**
- Sees new pod IP
- Updates iptables rules
- Enables service routing

**9. Service Controller (if LoadBalancer service):**
- Creates cloud load balancer
- Configures backend pool with pod IPs

**10. CoreDNS:**
- Service name → Service IP mapping available

**Result:** ML model is serving traffic!

### Example: Pod Failure and Recovery

**1. Container crashes:**
```
Application error → Container exits
```

**2. kubelet detects:**
- Liveness probe fails
- kubelet restarts container

**3. If restart fails repeatedly:**
- Pod enters CrashLoopBackOff
- kubelet backs off restart attempts

**4. If node fails:**
- Node Controller detects (no heartbeat)
- Marks node as NotReady
- Evicts pods after timeout (default: 5 minutes)

**5. ReplicaSet Controller:**
- Sees replica count < desired
- Creates replacement pod

**6. Scheduler:**
- Assigns new pod to healthy node

**7. New pod starts:**
- Service automatically includes new pod
- Traffic routes to healthy pods

## Hands-On: Explore Cluster Components

### View Control Plane Pods

```bash
# List all system pods
kubectl get pods -n kube-system

# Expected output:
# NAME                               READY   STATUS    RESTARTS   AGE
# coredns-xxxxx                      1/1     Running   0          1h
# etcd-minikube                      1/1     Running   0          1h
# kube-apiserver-minikube            1/1     Running   0          1h
# kube-controller-manager-minikube   1/1     Running   0          1h
# kube-proxy-xxxxx                   1/1     Running   0          1h
# kube-scheduler-minikube            1/1     Running   0          1h
# storage-provisioner                1/1     Running   0          1h
```

### View Component Logs

```bash
# API server logs
kubectl logs -n kube-system kube-apiserver-minikube

# Scheduler logs
kubectl logs -n kube-system kube-scheduler-minikube

# Controller manager logs
kubectl logs -n kube-system kube-controller-manager-minikube

# CoreDNS logs
kubectl logs -n kube-system coredns-xxxxx
```

### Check Component Health

```bash
# Component status (deprecated in newer versions)
kubectl get componentstatuses

# Alternative: check pod health
kubectl get pods -n kube-system

# Check API server health
kubectl get --raw /healthz

# Check API server livez
kubectl get --raw /livez

# Check API server readyz
kubectl get --raw /readyz
```

### Explore Node Details

```bash
# List nodes
kubectl get nodes

# Detailed node information
kubectl describe node minikube

# Key sections:
# - Conditions: Ready, MemoryPressure, DiskPressure
# - Capacity: CPU, memory, pods
# - Allocatable: Available for pods
# - Allocated resources: What's in use
# - Events: Recent node events
```

### View Cluster Info

```bash
# Cluster information
kubectl cluster-info

# Cluster configuration
kubectl config view

# API server address
kubectl cluster-info | grep master

# Kubernetes version
kubectl version
```

## Architecture Best Practices for ML

### 1. High Availability Control Plane

For production ML infrastructure:
```
- 3 or 5 control plane nodes (odd number for quorum)
- 3 or 5 etcd instances
- Load balancer in front of API servers
- Distributed across availability zones
```

### 2. Dedicated Node Pools

```yaml
# Example: Separate node pools for different workloads

# CPU-only inference nodes
- name: cpu-inference
  machineType: n1-standard-8
  taints:
    - key: workload-type
      value: inference
      effect: NoSchedule

# GPU training nodes
- name: gpu-training
  machineType: n1-standard-16
  accelerators:
    - type: nvidia-tesla-v100
      count: 4
  taints:
    - key: workload-type
      value: training
      effect: NoSchedule

# Spot/preemptible nodes for batch jobs
- name: spot-batch
  machineType: n1-standard-16
  spot: true
  taints:
    - key: workload-type
      value: batch
      effect: NoSchedule
```

### 3. Resource Quotas and Limits

```yaml
# Namespace-level quotas for teams
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
    pods: "50"
```

### 4. Monitoring and Alerting

- Prometheus for metrics
- Grafana for dashboards
- Alert on control plane health
- Alert on node resource pressure

## Summary

In this lesson, you've learned:

- ✅ Kubernetes architecture (control plane + worker nodes)
- ✅ Control plane components: API server, etcd, scheduler, controllers
- ✅ Worker node components: kubelet, kube-proxy, container runtime
- ✅ Add-ons: DNS, metrics server, dashboard
- ✅ How components work together to orchestrate containers
- ✅ Production architecture patterns for ML workloads

## What's Next?

In **Lesson 03: Core Resources**, you'll learn:
- Pods in depth
- ReplicaSets and Deployments
- Services for networking
- ConfigMaps and Secrets
- Namespaces for organization

## Self-Check Questions

1. What is the role of the API server?
2. What does etcd store and why is it critical?
3. How does the scheduler decide where to place pods?
4. What's the difference between kubelet and kube-proxy?
5. Describe the flow when you create a deployment.
6. Why is high availability important for the control plane?

## Additional Resources

- [Kubernetes Components Documentation](https://kubernetes.io/docs/concepts/overview/components/)
- [Kubernetes Architecture Deep Dive](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/architecture/architecture.md)
- [etcd Documentation](https://etcd.io/docs/)
- [The Raft Consensus Algorithm](https://raft.github.io/)

---

**Next Lesson:** [03-core-resources.md](./03-core-resources.md) - Pods, Deployments, Services, and ConfigMaps
