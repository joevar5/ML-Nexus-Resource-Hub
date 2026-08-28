# Module 04 Quiz: Kubernetes Fundamentals

**Total Questions:** 25
**Passing Score:** 70% (18/25 correct)
**Time Limit:** 50 minutes
**Type:** Mixed (Multiple Choice, True/False, Short Answer)

---

## Section 1: Kubernetes Architecture (Questions 1-6)

### Question 1
**Which component of the Kubernetes control plane stores the cluster state?**

A) API Server
B) etcd
C) Scheduler
D) Controller Manager

<details>
<summary>Answer</summary>
B) etcd - A distributed key-value store that stores all cluster data, including configuration and state
</details>

---

### Question 2
**What is the smallest deployable unit in Kubernetes?**

A) Container
B) Deployment
C) Pod
D) Node

<details>
<summary>Answer</summary>
C) Pod - A pod can contain one or more containers that share network and storage
</details>

---

### Question 3
**Which component on each worker node is responsible for communicating with the API server?**

A) kube-proxy
B) kubelet
C) Container runtime
D) etcd

<details>
<summary>Answer</summary>
B) kubelet - The node agent that ensures containers are running in pods as specified
</details>

---

### Question 4
**True or False: The kube-scheduler is responsible for deciding which node a pod runs on.**

<details>
<summary>Answer</summary>
True - The scheduler watches for newly created pods with no assigned node and selects a node for them to run on based on resource requirements and constraints.
</details>

---

### Question 5
**What tool is used to interact with the Kubernetes API server from the command line?**

A) kubeadm
B) kubectl
C) kubelet
D) helm

<details>
<summary>Answer</summary>
B) kubectl - The Kubernetes command-line tool for running commands against Kubernetes clusters
</details>

---

### Question 6
**Which control plane component runs controller processes like ReplicaSet controller and Deployment controller?**

A) API Server
B) Scheduler
C) Controller Manager
D) kubelet

<details>
<summary>Answer</summary>
C) Controller Manager (kube-controller-manager) - Runs various controllers that regulate the state of the cluster
</details>

---

## Section 2: Core Resources (Questions 7-12)

### Question 7
**What is the primary purpose of a Deployment in Kubernetes?**

A) Store configuration data
B) Manage stateful applications
C) Provide declarative updates for Pods and ReplicaSets
D) Expose services to external traffic

<details>
<summary>Answer</summary>
C) Provide declarative updates for Pods and ReplicaSets - Deployments manage the desired state of pods and enable rolling updates and rollbacks
</details>

---

### Question 8
**Which resource type would you use to ensure exactly one pod runs on each node in the cluster?**

A) Deployment
B) StatefulSet
C) DaemonSet
D) ReplicaSet

<details>
<summary>Answer</summary>
C) DaemonSet - Ensures a copy of a pod runs on all (or some) nodes, commonly used for logging agents and monitoring
</details>

---

### Question 9
**What is the difference between a ConfigMap and a Secret?**

A) ConfigMaps are for configuration; Secrets are for sensitive data (base64 encoded)
B) ConfigMaps are larger than Secrets
C) Secrets are faster to access
D) There is no difference

<details>
<summary>Answer</summary>
A) ConfigMaps are for configuration; Secrets are for sensitive data (base64 encoded) - Secrets are designed to hold sensitive information and are handled more carefully by Kubernetes
</details>

---

### Question 10
**True or False: Labels are key-value pairs attached to Kubernetes objects used for organizing and selecting resources.**

<details>
<summary>Answer</summary>
True - Labels enable users to map organizational structures onto system objects and are used by selectors to identify groups of objects
</details>

---

### Question 11
**Which field in a pod spec defines the minimum resources (CPU, memory) a container needs?**

A) resources.limits
B) resources.requests
C) resources.minimum
D) resources.required

<details>
<summary>Answer</summary>
B) resources.requests - Requests define the minimum guaranteed resources; limits define the maximum
</details>

---

### Question 12
**What is the purpose of namespaces in Kubernetes?**

A) To organize network policies
B) To provide logical isolation and scope for resources
C) To control pod scheduling
D) To manage storage classes

<details>
<summary>Answer</summary>
B) To provide logical isolation and scope for resources - Namespaces allow dividing cluster resources between multiple users/teams
</details>

---

## Section 3: Networking and Services (Questions 13-18)

### Question 13
**Which Service type exposes the service on each node's IP at a static port?**

A) ClusterIP
B) NodePort
C) LoadBalancer
D) ExternalName

<details>
<summary>Answer</summary>
B) NodePort - Makes the service accessible from outside the cluster using <NodeIP>:<NodePort>
</details>

---

### Question 14
**What is the default Service type in Kubernetes?**

A) NodePort
B) LoadBalancer
C) ClusterIP
D) ExternalName

<details>
<summary>Answer</summary>
C) ClusterIP - Exposes the service on a cluster-internal IP, making it only reachable from within the cluster
</details>

---

### Question 15
**True or False: An Ingress controller is required for Ingress resources to function.**

<details>
<summary>Answer</summary>
True - Ingress resources define routing rules, but an Ingress controller (like Nginx, Traefik) is needed to implement those rules
</details>

---

### Question 16
**How do Kubernetes services discover which pods to route traffic to?**

A) By pod name
B) By pod IP address
C) By label selectors matching pod labels
D) By namespace only

<details>
<summary>Answer</summary>
C) By label selectors matching pod labels - Services use selectors to dynamically find pods with matching labels
</details>

---

### Question 17
**What Kubernetes resource would you use to expose an HTTP/HTTPS route from outside the cluster to services within?**

A) Service
B) Ingress
C) NetworkPolicy
D) Endpoint

<details>
<summary>Answer</summary>
B) Ingress - Manages external access to services, typically HTTP/HTTPS, and can provide SSL termination, name-based virtual hosting, etc.
</details>

---

### Question 18
**Short Answer: Explain the difference between a Service of type LoadBalancer and an Ingress.**

<details>
<summary>Sample Answer</summary>

**LoadBalancer Service:**
- Creates a cloud load balancer (AWS ELB, GCP Load Balancer, etc.)
- One load balancer per service (costly if you have many services)
- Works at L4 (TCP/UDP level)
- Simple, straightforward for single service exposure

**Ingress:**
- Single entry point for multiple services (HTTP/HTTPS routing)
- Works at L7 (application level)
- One load balancer for many services (cost-effective)
- Supports path-based routing, host-based routing, TLS termination
- Requires Ingress controller to be installed

**Use Case:**
- LoadBalancer: Single service needing external access, non-HTTP protocols
- Ingress: Multiple HTTP/HTTPS services, need path-based routing or SSL termination

(Core differences acceptable)
</details>

---

## Section 4: Storage and Configuration (Questions 19-22)

### Question 19
**What is the relationship between PersistentVolume (PV) and PersistentVolumeClaim (PVC)?**

A) They are the same thing
B) PV is a request for storage; PVC is the actual storage
C) PV is cluster storage; PVC is a user's request for that storage
D) PV is cloud storage; PVC is local storage

<details>
<summary>Answer</summary>
C) PV is cluster storage (provisioned by admin); PVC is a user's request for that storage - PVCs bind to PVs based on storage requirements
</details>

---

### Question 20
**Which Kubernetes resource enables dynamic provisioning of PersistentVolumes?**

A) StorageClass
B) VolumeClass
C) PersistentVolume
D) VolumeProvisioner

<details>
<summary>Answer</summary>
A) StorageClass - Defines classes of storage with different properties (e.g., SSD vs HDD, different performance tiers)
</details>

---

### Question 21
**True or False: ConfigMaps can be mounted as volumes or exposed as environment variables in pods.**

<details>
<summary>Answer</summary>
True - ConfigMaps can be consumed in both ways, providing flexibility in how configuration is provided to applications
</details>

---

### Question 22
**Which Kubernetes resource is designed for deploying stateful applications that require stable network identities and persistent storage?**

A) Deployment
B) StatefulSet
C) DaemonSet
D) Job

<details>
<summary>Answer</summary>
B) StatefulSet - Manages stateful applications with unique network identifiers, stable persistent storage, and ordered deployment/scaling
</details>

---

## Section 5: GPU and Advanced Topics (Questions 23-25)

### Question 23
**How do you request GPU resources in a Kubernetes pod specification?**

A) Add `nvidia.com/gpu: 1` to resources.requests
B) Add `gpu: 1` to resources.requests
C) Add `cuda: 1` to resources.requests
D) GPUs are allocated automatically

<details>
<summary>Answer</summary>
A) Add `nvidia.com/gpu: 1` to resources.requests (or limits) - Example:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```
</details>

---

### Question 24
**What is Helm in the Kubernetes ecosystem?**

A) A monitoring tool
B) A package manager for Kubernetes applications
C) A container runtime
D) A networking plugin

<details>
<summary>Answer</summary>
B) A package manager for Kubernetes applications - Helm helps you define, install, and upgrade Kubernetes applications using Charts
</details>

---

### Question 25
**Short Answer: What is the Horizontal Pod Autoscaler (HPA) and when would you use it for ML workloads?**

<details>
<summary>Sample Answer</summary>

**Horizontal Pod Autoscaler (HPA):**
- Automatically scales the number of pods in a Deployment/ReplicaSet based on observed metrics
- Monitors CPU utilization, memory, or custom metrics
- Increases/decreases pod replicas to match demand

**For ML Workloads:**

**Use Cases:**
1. **Inference Services:** Scale model serving pods based on request rate or latency
2. **Variable Traffic:** Handle traffic spikes automatically (e.g., 10 pods during peak, 2 pods off-peak)
3. **Cost Optimization:** Reduce pods during low-demand periods to save costs

**Example:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**When NOT to use for ML:**
- Training jobs (use fixed replicas or Job resources)
- Stateful applications requiring stable pod identities

(Core concept + example use case acceptable)
</details>

---

## Scoring Guide

### Grading Rubric

- **23-25 correct (92-100%):** Excellent - You have mastered Kubernetes fundamentals
- **20-22 correct (80-88%):** Very Good - Strong understanding with minor gaps
- **18-19 correct (72-76%):** Passing - Adequate knowledge, review areas where you struggled
- **Below 18 (< 72%):** Not Passing - Review module material and retake quiz

### What to Do Next

**If you passed (≥ 70%):**
1. Review any questions you got wrong
2. Complete the Module 04 practical exercises
3. Work on the practical assessment (production K8s deployment)
4. Proceed to Module 05: Data Pipelines and Orchestration

**If you didn't pass (< 70%):**
1. Review lessons corresponding to questions you missed
2. Practice with kubectl and local cluster
3. Deploy sample applications and troubleshoot issues
4. Retake quiz after additional study

---

## Answer Key Summary

1. B (etcd)
2. C (Pod)
3. B (kubelet)
4. True
5. B (kubectl)
6. C (Controller Manager)
7. C (Declarative updates)
8. C (DaemonSet)
9. A (ConfigMaps for config, Secrets for sensitive)
10. True
11. B (resources.requests)
12. B (Logical isolation)
13. B (NodePort)
14. C (ClusterIP)
15. True
16. C (Label selectors)
17. B (Ingress)
18. Short Answer (see details)
19. C (PV is storage, PVC is request)
20. A (StorageClass)
21. True
22. B (StatefulSet)
23. A (nvidia.com/gpu in requests)
24. B (Package manager)
25. Short Answer (see details)

---

## Key Concepts to Review

### If you struggled with Section 1 (Architecture):
- Review Lesson 02: Kubernetes Architecture
- Explore your cluster: `kubectl get nodes`, `kubectl get componentstatuses`
- Understand: Control plane, worker nodes, etcd, scheduler, controller manager

### If you struggled with Section 2 (Core Resources):
- Review Lesson 03: Core Resources
- Review Lesson 04: Deployments and Services
- Practice: Create Deployments, scale replicas, use ConfigMaps/Secrets
- Understand: Pod lifecycle, labels and selectors, resource requests/limits

### If you struggled with Section 3 (Networking):
- Review Lesson 05: Networking and Ingress
- Practice: Create Services (ClusterIP, NodePort), set up Ingress
- Understand: Service types, selectors, Ingress routing, DNS

### If you struggled with Section 4 (Storage):
- Review Lesson 06: Storage and Persistence
- Practice: Create PVCs, use volumes in pods, deploy StatefulSets
- Understand: PV/PVC binding, StorageClasses, volume types

### If you struggled with Section 5 (Advanced):
- Review Lesson 08: Helm Package Manager
- Review Lesson 09: GPU Scheduling
- Review Lesson 10: Monitoring and Observability
- Practice: Install Helm charts, request GPU resources, set up HPA
- Understand: Helm charts, GPU device plugins, auto-scaling

---

## Practical Application

After passing the quiz, demonstrate skills by:

1. **Deploy multi-tier ML application** (frontend, model server, database)
2. **Configure auto-scaling** with HPA based on CPU/memory
3. **Set up Ingress** with SSL/TLS for external access
4. **Use Helm** to package and deploy your application
5. **Request GPU resources** and deploy GPU-accelerated workload
6. **Implement monitoring** with kubectl top and basic Prometheus
7. **Document** your deployment in README with troubleshooting guide

---

## Additional Practice

### Hands-On Labs
- [Kubernetes Basics Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/) - Free browser-based K8s playground
- [Katacoda Kubernetes Scenarios](https://www.katacoda.com/courses/kubernetes) - Interactive tutorials

### Certification Prep
- [Certified Kubernetes Application Developer (CKAD)](https://www.cncf.io/certification/ckad/) - Application deployment focus
- [Certified Kubernetes Administrator (CKA)](https://www.cncf.io/certification/cka/) - Cluster administration focus

---

**Time to Review:** 30-40 minutes to review answers and understand mistakes

**Ready for Module 05?** If you scored ≥ 70% and completed exercises, you're ready for data pipelines!
