# Module 006: Kubernetes Introduction - Assessment Quiz

## Instructions

This quiz contains 25 questions covering all aspects of Module 006. Answer each question to the best of your ability.

**Time Limit**: 60 minutes (recommended)
**Passing Score**: 80% (20/25 correct)

---

## Section 1: Kubernetes Architecture

### Question 1
Which Kubernetes component is responsible for scheduling Pods to nodes?

A) kubelet
B) kube-scheduler
C) kube-controller-manager
D) kube-proxy

**Answer**: B
**Explanation**: kube-scheduler is responsible for assigning Pods to nodes.

---

### Question 2
Where does Kubernetes store all cluster state and configuration?

A) In memory on the API server
B) In etcd
C) In a relational database
D) On each worker node

**Answer**: B
**Explanation**: etcd stores all cluster state using Raft consensus.

---

### Question 3
What is the purpose of the kubelet?

A) Schedule Pods to nodes
B) Run controllers that manage cluster state
C) Manage containers on a specific node
D) Implement Service networking

**Answer**: C
**Explanation**: kubelet manages containers and Pod lifecycle on each node.

---

### Question 4
Which component implements Service load balancing on each node?

A) kube-proxy
B) kubelet
C) Container runtime
D) CoreDNS

**Answer**: A
**Explanation**: kube-proxy implements Service networking and load balancing.

---

### Question 5
What communication protocol does the Kubernetes API Server use?

A) gRPC
B) REST (HTTP/HTTPS)
C) WebSockets
D) MQTT

**Answer**: B
**Explanation**: The API server uses REST (HTTP/HTTPS) for communication.

---

### Question 6
In Helm 3, where does Helm store release information?

A) In a separate Tiller service
B) In Kubernetes Secrets
C) In etcd directly
D) On the client machine only

**Answer**: B
**Explanation**: Helm 3 stores releases as Secrets in Kubernetes.

---

## Section 2: Pods and Deployments

### Question 7
What is the smallest deployable unit in Kubernetes?

A) Container
B) Pod
C) Deployment
D) ReplicaSet

**Answer**: B
**Explanation**: Pod is the smallest deployable unit.

---

### Question 8
How many containers can a Pod contain?

A) Exactly one
B) One or more
C) At least two
D) Up to five

**Answer**: B
**Explanation**: Pods can contain one or more containers (usually one main application container and optional sidecar/init containers).

---

### Question 9
What happens when you delete a Deployment?

A) Only the Deployment object is deleted
B) The Deployment and ReplicaSets are deleted, but Pods remain
C) The Deployment, ReplicaSets, and Pods are all deleted
D) Nothing happens; Deployments cannot be deleted

**Answer**: C
**Explanation**: Deleting a Deployment cascades to automatically clean up and delete its ReplicaSets and Pods.

---

### Question 10
What is the purpose of a liveness probe?

A) Determine if a container should receive traffic
B) Determine if Kubernetes should restart a container
C) Check if a container has started successfully
D) Monitor resource usage

**Answer**: B
**Explanation**: Liveness probes determine if a container should be restarted by kubelet.

---

### Question 11
In a rolling update with maxUnavailable=1 and maxSurge=1, starting with 3 replicas, what is the maximum number of Pods during the update?

A) 3
B) 4
C) 5
D) 6

**Answer**: B
**Explanation**: maxSurge=1 allows up to 4 Pods (3 + 1), maxUnavailable=1 means at least 2 running.

---

## Section 3: Services and Networking

### Question 12
What is the default Service type in Kubernetes?

A) NodePort
B) LoadBalancer
C) ClusterIP
D) ExternalName

**Answer**: C
**Explanation**: ClusterIP is the default Service type.

---

### Question 13
How does a Service find its backend Pods?

A) By IP address
B) By name
C) By label selectors
D) By namespace

**Answer**: C
**Explanation**: Services use label selectors to identify and load balance across backend Pods.

---

### Question 14
What port range is used for NodePort services?

A) 1-1024
B) 8000-9000
C) 30000-32767
D) 40000-50000

**Answer**: C
**Explanation**: NodePort uses the default port range 30000-32767.

---

### Question 15
What is the full DNS name for a service named "api" in the "production" namespace?

A) api.production
B) api.production.svc
C) api.production.svc.cluster.local
D) production.api.cluster.local

**Answer**: C
**Explanation**: Full DNS name format is `<service>.<namespace>.svc.cluster.local`.

---

## Section 4: Configuration and Storage

### Question 16
What is the difference between a ConfigMap and a Secret?

A) Secrets support encryption at rest, ConfigMaps do not (by default)
B) ConfigMaps can be larger than Secrets
C) Secrets can only store strings, ConfigMaps can store any data
D) There is no difference; they're interchangeable

**Answer**: A
**Explanation**: Secrets support encryption at rest (when enabled), whereas ConfigMaps do not.

---

### Question 17
What happens when you update a ConfigMap that's mounted as a volume in a running Pod?

A) Pod automatically restarts
B) File is updated in the container (eventually)
C) Nothing; Pods don't see ConfigMap updates
D) Container crashes

**Answer**: B
**Explanation**: Mounted ConfigMaps are eventually updated inside the running container.

---

### Question 18
What access mode allows multiple nodes to mount a volume for reading and writing?

A) ReadWriteOnce (RWO)
B) ReadOnlyMany (ROX)
C) ReadWriteMany (RWX)
D) ReadWriteMulti (RWM)

**Answer**: C
**Explanation**: ReadWriteMany (RWX) allows multi-node read-write access.

---

### Question 19
What creates PersistentVolumes automatically when a PersistentVolumeClaim is created?

A) kubelet
B) StorageClass with dynamic provisioner
C) PersistentVolume controller
D) Volume plugin

**Answer**: B
**Explanation**: StorageClass with dynamic provisioner automatically creates PVs on-demand when a PVC is created.

---

## Section 5: Helm

### Question 20
What file in a Helm chart contains default configuration values?

A) Chart.yaml
B) values.yaml
C) defaults.yaml
D) config.yaml

**Answer**: B
**Explanation**: values.yaml contains default configuration values.

---

### Question 21
What is a Helm release?

A) A version of Helm software
B) An instance of a chart installed in a cluster
C) A packaged Helm chart
D) A Helm repository

**Answer**: B
**Explanation**: A release is an instance of a chart installed in a cluster.

---

### Question 22
How do you roll back a Helm release to the previous version?

A) `helm undo`
B) `helm revert`
C) `helm rollback`
D) `helm restore`

**Answer**: C
**Explanation**: `helm rollback` rolls back to the previous version.

---

## Section 6: Operations and Debugging

### Question 23
What kubectl command shows detailed information about a resource including events?

A) kubectl get
B) kubectl explain
C) kubectl describe
D) kubectl logs

**Answer**: C
**Explanation**: `kubectl describe` shows detailed information about a resource, including events.

---

### Question 24
A Pod is in "CrashLoopBackOff" status. Which command shows why the container crashed?

A) `kubectl logs <pod>`
B) `kubectl logs <pod> --previous`
C) `kubectl describe pod <pod>`
D) Both B and C

**Answer**: D
**Explanation**: Both `kubectl logs <pod> --previous` and `kubectl describe pod <pod>` help identify container crash causes.

---

### Question 25
What does the command `kubectl top pods` display?

A) The first few Pods in the list
B) Current CPU and memory usage
C) Pod priority levels
D) Pods with most restarts

**Answer**: B
**Explanation**: `kubectl top pods` shows current CPU and memory usage of running pods.
