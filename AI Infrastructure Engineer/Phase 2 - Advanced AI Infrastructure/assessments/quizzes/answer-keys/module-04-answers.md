# Module 04: Kubernetes — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-04-kubernetes/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** Which component of the Kubernetes control plane stores the cluster state?

**Answer:** B) etcd

**Explanation:**
etcd is a distributed, strongly-consistent key-value store that serves as the single source of truth for all cluster state, including desired and observed state for every API object (Pods, Deployments, Secrets, ConfigMaps, etc.). Every other control plane component reads and writes through the API Server, which is the *only* component that talks directly to etcd. Losing etcd without a backup means losing the cluster's memory, which is why it is typically run as a quorum of 3 or 5 members.

**Common Mistakes:**
- Choosing A) API Server: the API Server is the front door for all reads/writes, but it is stateless and persists nothing on its own; it delegates storage to etcd.
- Choosing C) Scheduler or D) Controller Manager: these are reconciliation loops that consume state from the API Server; they do not store it.

**Related Material:** `lessons/mod-04-kubernetes/02-k8s-architecture.md`

---

## Question 2
**Q:** What is the smallest deployable unit in Kubernetes?

**Answer:** C) Pod

**Explanation:**
Kubernetes never schedules a bare container. The Pod is the atomic unit of scheduling and the smallest object you can create in the API. A Pod wraps one or more tightly coupled containers that share a network namespace (same IP, localhost connectivity) and can share volumes, so they are always co-located on the same node and scaled together.

**Common Mistakes:**
- Choosing A) Container: containers run *inside* a Pod, but Kubernetes' scheduling, networking, and lifecycle primitives operate on the Pod, not the container.
- Choosing B) Deployment: a Deployment is a higher-level controller that *manages* Pods through ReplicaSets, not a deployable unit itself.
- Choosing D) Node: a Node is a worker machine that *hosts* Pods, not something you deploy.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 3
**Q:** Which component on each worker node is responsible for communicating with the API server?

**Answer:** B) kubelet

**Explanation:**
The kubelet is the node agent that runs on every worker. It watches the API Server for Pods assigned to its node, instructs the container runtime to start/stop containers, mounts volumes, runs probes, and reports node and Pod status back up to the API Server. It is the bridge between the control plane's desired state and the actual state on the node.

**Common Mistakes:**
- Choosing A) kube-proxy: kube-proxy programs node networking (iptables/IPVS/eBPF) for Services; it does not manage Pod lifecycle.
- Choosing C) Container runtime: containerd/CRI-O actually run the containers, but the kubelet drives them via the CRI — the runtime does not talk to the API Server.
- Choosing D) etcd: etcd is part of the control plane, not a per-node component.

**Related Material:** `lessons/mod-04-kubernetes/02-k8s-architecture.md`

---

## Question 4
**Q:** True or False: The kube-scheduler is responsible for deciding which node a pod runs on.

**Answer:** True

**Explanation:**
The scheduler watches for Pods whose `spec.nodeName` is empty and selects a suitable node by running filtering (predicates) and scoring (priorities) over the cluster: it checks resource requests, node selectors, taints/tolerations, affinity/anti-affinity, topology spread, and more. Once it picks a node, it writes the binding back to the API Server, and the target node's kubelet takes it from there.

**Common Mistakes:**
- Answering False because you assumed the kubelet picks the node: the kubelet only acts on Pods *already* bound to its node by the scheduler.
- Confusing scheduling with placement constraints: nodeSelectors and affinity rules influence the scheduler's decision, but the scheduler is still the component that decides.

**Related Material:** `lessons/mod-04-kubernetes/02-k8s-architecture.md`

---

## Question 5
**Q:** What tool is used to interact with the Kubernetes API server from the command line?

**Answer:** B) kubectl

**Explanation:**
kubectl is the official CLI for the Kubernetes API. It reads a kubeconfig file for cluster endpoints and credentials, then issues authenticated HTTPS requests to the API Server for verbs like `get`, `apply`, `describe`, `logs`, and `exec`. It is the day-to-day tool for both operators and developers.

**Common Mistakes:**
- Choosing A) kubeadm: kubeadm bootstraps and upgrades clusters; it is not used for routine API interaction.
- Choosing C) kubelet: the kubelet is a node-side daemon, not a CLI.
- Choosing D) helm: Helm is a package manager built *on top of* kubectl's API; it doesn't replace general-purpose API access.

**Related Material:** `lessons/mod-04-kubernetes/01-k8s-introduction.md`

---

## Question 6
**Q:** Which control plane component runs controller processes like ReplicaSet controller and Deployment controller?

**Answer:** C) Controller Manager

**Explanation:**
kube-controller-manager is a single binary that hosts dozens of built-in controllers (Deployment, ReplicaSet, Node, Job, EndpointSlice, ServiceAccount, etc.). Each controller runs a reconciliation loop: it observes the current state via the API Server, compares it to the desired state in the spec, and takes action to converge the two. Bundling them in one process simplifies leader election and configuration.

**Common Mistakes:**
- Choosing A) API Server: it serves the API but does not run business logic for resources.
- Choosing B) Scheduler: the scheduler only handles Pod-to-Node binding, not ReplicaSet/Deployment reconciliation.
- Choosing D) kubelet: the kubelet is per-node and acts on Pods, not on higher-level controllers.

**Related Material:** `lessons/mod-04-kubernetes/02-k8s-architecture.md`

---

## Question 7
**Q:** What is the primary purpose of a Deployment in Kubernetes?

**Answer:** C) Provide declarative updates for Pods and ReplicaSets

**Explanation:**
A Deployment lets you declare the desired state (image, replica count, labels, strategy) and the controller drives the cluster toward it. It manages a ReplicaSet for each revision, enabling rolling updates, surge/unavailable controls, pause/resume, history, and rollback to a previous revision — all without you imperatively shutting down Pods.

**Common Mistakes:**
- Choosing A) Store configuration data: that is the role of ConfigMaps and Secrets.
- Choosing B) Manage stateful applications: stateful workloads with stable identity and per-Pod storage belong in a StatefulSet.
- Choosing D) Expose services to external traffic: exposing traffic is the job of Services and Ingress, not Deployments.

**Related Material:** `lessons/mod-04-kubernetes/04-deployments-services.md`

---

## Question 8
**Q:** Which resource type would you use to ensure exactly one pod runs on each node in the cluster?

**Answer:** C) DaemonSet

**Explanation:**
A DaemonSet guarantees that a copy of a Pod runs on every matching node, and it automatically schedules onto new nodes as they join the cluster (and cleans up when nodes leave). Typical uses are node-level agents: log shippers (Fluent Bit), metrics collectors (node-exporter), CNI plugins, and GPU device plugins.

**Common Mistakes:**
- Choosing A) Deployment: Deployments target a replica count distributed across the cluster, not one-per-node.
- Choosing B) StatefulSet: StatefulSets give stable identity and ordered rollout but do not pin one Pod per node.
- Choosing D) ReplicaSet: a ReplicaSet just maintains N Pods total; placement is up to the scheduler.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 9
**Q:** What is the difference between a ConfigMap and a Secret?

**Answer:** A) ConfigMaps are for configuration; Secrets are for sensitive data (base64 encoded)

**Explanation:**
Both objects expose key/value data into Pods (as env vars, files, or CLI args), but Secrets are intended for sensitive material such as passwords, tokens, and TLS keys. Secrets are stored base64-encoded in etcd (which is *encoding*, not encryption — enable encryption-at-rest for real protection), can be encrypted via KMS providers, and are handled with extra care by the kubelet (tmpfs mounts, not written to disk by default).

**Common Mistakes:**
- Choosing B) ConfigMaps are larger: both have the same ~1 MiB object size limit; size is not the distinguishing factor.
- Choosing C) Secrets are faster to access: there is no performance difference at consumption time.
- Choosing D) There is no difference: this misses RBAC, encryption-at-rest, and mounting semantics that treat Secrets differently.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 10
**Q:** True or False: Labels are key-value pairs attached to Kubernetes objects used for organizing and selecting resources.

**Answer:** True

**Explanation:**
Labels are arbitrary key/value pairs attached to objects' metadata. They are the mechanism Kubernetes uses to wire components together: a Service selects Pods by label, a Deployment manages a ReplicaSet whose Pods carry specific labels, and NetworkPolicies match traffic by label selector. They are intended for identifying and grouping objects, unlike annotations which carry non-identifying metadata.

**Common Mistakes:**
- Answering False because you confused labels with annotations: annotations are also key/value pairs but are *not* selectable and are meant for tooling metadata.
- Assuming labels are only for humans: selectors make them load-bearing for the control plane.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 11
**Q:** Which field in a pod spec defines the minimum resources (CPU, memory) a container needs?

**Answer:** B) resources.requests

**Explanation:**
`resources.requests` is what the scheduler uses to find a node with enough free capacity, and it is the amount the kubelet/cgroups guarantee the container. `resources.limits` is the upper bound — CPU above the limit is throttled, memory above the limit triggers an OOM kill. Setting requests too low risks eviction under pressure; setting them equal to limits gives a Pod a Guaranteed QoS class.

**Common Mistakes:**
- Choosing A) resources.limits: limits are the maximum, not the minimum guarantee.
- Choosing C) resources.minimum or D) resources.required: neither field exists in the Pod spec.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 12
**Q:** What is the purpose of namespaces in Kubernetes?

**Answer:** B) To provide logical isolation and scope for resources

**Explanation:**
Namespaces partition a single cluster into virtual sub-clusters so multiple teams or environments can share it. They scope names (you can have two Services called `api` in two namespaces), are the unit for RBAC bindings, ResourceQuotas, LimitRanges, and NetworkPolicies, and they let you delete a whole tenant's objects in one shot. They are not a hard security boundary on their own — pair them with NetworkPolicies and RBAC.

**Common Mistakes:**
- Choosing A) To organize network policies: NetworkPolicies live inside namespaces but namespaces are not *for* organizing them.
- Choosing C) To control pod scheduling: scheduling is driven by node selectors, affinity, taints, and resources — not namespaces.
- Choosing D) To manage storage classes: StorageClasses are cluster-scoped and unrelated to namespaces.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 13
**Q:** Which Service type exposes the service on each node's IP at a static port?

**Answer:** B) NodePort

**Explanation:**
A NodePort Service allocates a port from the configured range (default 30000–32767) and opens it on every node's IP. Traffic arriving at `<NodeIP>:<NodePort>` is forwarded by kube-proxy to a backend Pod. NodePort is typically used as the plumbing under a cloud LoadBalancer or for on-prem clusters fronted by an external load balancer.

**Common Mistakes:**
- Choosing A) ClusterIP: ClusterIP is only reachable from inside the cluster.
- Choosing C) LoadBalancer: LoadBalancer also opens a NodePort under the hood, but the question specifically describes the per-node IP + static port pattern, which is NodePort's defining behavior.
- Choosing D) ExternalName: ExternalName is just a DNS CNAME alias; it doesn't expose any port.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 14
**Q:** What is the default Service type in Kubernetes?

**Answer:** C) ClusterIP

**Explanation:**
If you omit `spec.type` on a Service, Kubernetes assigns ClusterIP. The Service gets a stable virtual IP from the Service CIDR that is only routable inside the cluster, and kube-proxy load-balances connections across the Endpoints/EndpointSlices that match the selector. It is the right default because most service-to-service traffic stays internal.

**Common Mistakes:**
- Choosing A) NodePort or B) LoadBalancer: these expose the Service externally and must be requested explicitly.
- Choosing D) ExternalName: ExternalName is a niche DNS alias type, never a default.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 15
**Q:** True or False: An Ingress controller is required for Ingress resources to function.

**Answer:** True

**Explanation:**
An Ingress object is a piece of *configuration* — a set of host/path routing rules. Nothing in the cluster acts on it until you install an Ingress controller (NGINX, Traefik, HAProxy, AWS ALB, GCE, Istio, etc.) that watches Ingress resources and programs the actual reverse proxy or cloud load balancer. Cloud distributions like GKE and EKS may ship one by default, but it still has to exist.

**Common Mistakes:**
- Answering False because your managed cluster "just worked": a controller was pre-installed for you.
- Confusing Ingress with Service of type LoadBalancer: LoadBalancers do not require a controller because the cloud provider integration ships with the cluster.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 16
**Q:** How do Kubernetes services discover which pods to route traffic to?

**Answer:** C) By label selectors matching pod labels

**Explanation:**
A Service's `spec.selector` is a label matcher. The endpoints controller continuously watches Pods and writes the IPs of matching, ready Pods into an EndpointSlice (or Endpoints) object. kube-proxy reads that and programs node-local load balancing. Because the binding is dynamic, Pods can come and go and the Service automatically tracks the current set.

**Common Mistakes:**
- Choosing A) By pod name: Pod names are unstable for Deployments (random suffix per ReplicaSet rollout).
- Choosing B) By pod IP address: Pod IPs are ephemeral and change with each restart, so static IP wiring would break constantly.
- Choosing D) By namespace only: the namespace scopes the selector but does not by itself select Pods.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 17
**Q:** What Kubernetes resource would you use to expose an HTTP/HTTPS route from outside the cluster to services within?

**Answer:** B) Ingress

**Explanation:**
Ingress is the API for L7 (HTTP/HTTPS) routing: host-based and path-based rules, TLS termination, and a single external entry point that fans out to many backend Services. This lets you front many microservices with one external IP/load balancer instead of provisioning a LoadBalancer per Service.

**Common Mistakes:**
- Choosing A) Service: a Service alone can expose traffic externally (NodePort/LoadBalancer) but is L4 and does not do host/path routing or shared TLS.
- Choosing C) NetworkPolicy: NetworkPolicies *restrict* traffic between Pods; they do not expose anything.
- Choosing D) Endpoint: Endpoints/EndpointSlices are internal plumbing that list backing Pod IPs; they are not user-facing routing.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 18
**Q:** Short Answer: Explain the difference between a Service of type LoadBalancer and an Ingress.

**Answer:** A LoadBalancer Service provisions one external L4 load balancer (TCP/UDP) per Service via the cloud provider, while an Ingress is an L7 (HTTP/HTTPS) routing layer that can fan a single external entry point out to many backing Services with host- and path-based rules plus TLS termination.

**Explanation:**
- **LoadBalancer Service:** cloud-provider integration creates an external LB (AWS NLB/ELB, GCP LB, Azure LB) bound to one Service. L4, protocol-agnostic, simple, but you pay for one LB per Service. Good for non-HTTP protocols (gRPC over arbitrary ports, raw TCP, UDP) or single-service exposure.
- **Ingress:** an API + controller pair that programs an L7 reverse proxy (NGINX, Envoy, ALB). One external IP can route by Host header and URL path to many Services, terminate TLS centrally, and apply rewrites/auth. Cost-effective when you have many HTTP services; requires an Ingress controller.
- **Rule of thumb:** Ingress for many HTTP services behind one entry point; LoadBalancer for a single service or non-HTTP traffic. In practice, the Ingress controller itself is usually exposed via a LoadBalancer Service.

**Common Mistakes:**
- Saying "Ingress replaces LoadBalancer": Ingress controllers are normally *fronted* by a LoadBalancer Service.
- Conflating L4 vs L7: a LoadBalancer Service cannot do host/path routing or shared TLS termination across services.
- Forgetting that Ingress requires a controller to do anything.

**Related Material:** `lessons/mod-04-kubernetes/05-networking-ingress.md`

---

## Question 19
**Q:** What is the relationship between PersistentVolume (PV) and PersistentVolumeClaim (PVC)?

**Answer:** C) PV is cluster storage; PVC is a user's request for that storage

**Explanation:**
A PersistentVolume is a cluster-scoped storage resource — either statically provisioned by an admin or dynamically created by a StorageClass — that represents a real piece of storage (an EBS volume, NFS export, Ceph RBD image, etc.). A PersistentVolumeClaim is a namespaced request from a user/Pod that specifies size, access modes, and StorageClass. Kubernetes binds the PVC to a matching PV (or provisions a new one), and the Pod mounts the PVC, decoupling Pods from the underlying storage backend.

**Common Mistakes:**
- Choosing A) They are the same thing: they are deliberately separated to decouple consumption from provisioning.
- Choosing B) PV is a request; PVC is the actual storage: this reverses the roles.
- Choosing D) PV is cloud storage; PVC is local storage: the cloud-vs-local distinction is about volume drivers, not PV vs PVC.

**Related Material:** `lessons/mod-04-kubernetes/06-storage-persistence.md`

---

## Question 20
**Q:** Which Kubernetes resource enables dynamic provisioning of PersistentVolumes?

**Answer:** A) StorageClass

**Explanation:**
A StorageClass describes a "class" of storage (provisioner plugin, parameters such as disk type or IOPS, reclaim policy, volume binding mode). When a PVC references a StorageClass and no matching PV exists, the CSI driver named in the StorageClass provisions a new volume on demand and binds it to the PVC. This eliminates the need for an admin to pre-create PVs for every claim.

**Common Mistakes:**
- Choosing B) VolumeClass or D) VolumeProvisioner: neither resource exists in the Kubernetes API.
- Choosing C) PersistentVolume: a PV represents an *already provisioned* unit of storage and does not itself trigger provisioning.

**Related Material:** `lessons/mod-04-kubernetes/06-storage-persistence.md`

---

## Question 21
**Q:** True or False: ConfigMaps can be mounted as volumes or exposed as environment variables in pods.

**Answer:** True

**Explanation:**
ConfigMaps support multiple consumption patterns: individual keys can be exposed as `env` variables, all keys can be loaded with `envFrom`, or the entire ConfigMap can be projected as files in a volume mount. The volume-mount form has the advantage that updates to the ConfigMap propagate to the Pod's files (eventually), while environment variables are read once at process start.

**Common Mistakes:**
- Answering False because you only used the env-var form: both forms are first-class.
- Assuming env-var updates propagate live: they do not — the Pod must be restarted to see new values.

**Related Material:** `lessons/mod-04-kubernetes/03-core-resources.md`

---

## Question 22
**Q:** Which Kubernetes resource is designed for deploying stateful applications that require stable network identities and persistent storage?

**Answer:** B) StatefulSet

**Explanation:**
StatefulSets give each replica a stable, ordinal identity (`web-0`, `web-1`, …), a stable DNS name via a headless Service, and a dedicated PVC provisioned from `volumeClaimTemplates` that follows the Pod across rescheduling. They roll out and scale in order, which is essential for clustered databases, message brokers, and consensus systems where peers need stable identity.

**Common Mistakes:**
- Choosing A) Deployment: Deployments treat Pods as interchangeable and give them random names — no stable identity.
- Choosing C) DaemonSet: DaemonSets pin one Pod per node and are not designed for stateful clustered workloads.
- Choosing D) Job: Jobs run a Pod to completion; they are for batch work, not long-running stateful services.

**Related Material:** `lessons/mod-04-kubernetes/06-storage-persistence.md`

---

## Question 23
**Q:** How do you request GPU resources in a Kubernetes pod specification?

**Answer:** A) Add `nvidia.com/gpu: 1` to resources.requests (or limits)

**Explanation:**
GPUs are exposed through the Device Plugin API, and NVIDIA's plugin advertises the extended resource name `nvidia.com/gpu`. You request whole GPUs as an integer under `resources.limits` (and/or `requests`); fractional GPUs are not supported by the default plugin, though MIG and time-slicing options exist. The scheduler then places the Pod only on nodes that advertise enough of that resource.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

**Common Mistakes:**
- Choosing B) `gpu: 1` or C) `cuda: 1`: these are not registered extended resource names; the scheduler will ignore them.
- Choosing D) GPUs are allocated automatically: you must install the device plugin *and* explicitly request the resource.

**Related Material:** `lessons/mod-04-kubernetes/08-gpu-scheduling.md`

---

## Question 24
**Q:** What is Helm in the Kubernetes ecosystem?

**Answer:** B) A package manager for Kubernetes applications

**Explanation:**
Helm packages a set of related Kubernetes manifests into a *Chart* — a versioned, parameterized template with a `values.yaml` for configuration. `helm install`/`upgrade`/`rollback` apply or revert the rendered manifests as a single release, and Charts can be published to repositories (OCI registries, ChartMuseum) for reuse. It is the standard way to share and version complex K8s applications.

**Common Mistakes:**
- Choosing A) A monitoring tool: monitoring is Prometheus/Grafana/OpenTelemetry territory.
- Choosing C) A container runtime: runtimes are containerd, CRI-O, etc.
- Choosing D) A networking plugin: CNIs (Calico, Cilium, Flannel) handle networking.

**Related Material:** `lessons/mod-04-kubernetes/07-helm-package-manager.md`

---

## Question 25
**Q:** Short Answer: What is the Horizontal Pod Autoscaler (HPA) and when would you use it for ML workloads?

**Answer:** The HPA is a controller that automatically scales the replica count of a Deployment, ReplicaSet, or StatefulSet up and down based on observed metrics (CPU, memory, or custom/external metrics) against a target. For ML workloads, it fits inference services that see variable request load — not training jobs.

**Explanation:**
- **What it does:** the HPA controller polls metrics (CPU/memory by default, plus custom or external metrics via the metrics adapter API), computes the desired replicas to bring usage to the target, and updates the workload's `replicas` field within configured `minReplicas`/`maxReplicas` bounds.
- **Good fits for ML:**
  - Online inference services where QPS, latency, or queue depth varies through the day — autoscale model-server Pods on CPU/GPU utilization, request rate, or in-flight requests.
  - Batch-style request queues where you scale workers on queue depth via an external metric.
  - Cost control: shrink to `minReplicas` overnight and absorb traffic spikes by scaling out.
- **Poor fits:**
  - Training jobs — use `Job`/`Indexed Job` or specialized training operators with fixed parallelism.
  - Stateful systems needing stable identity — StatefulSets can be HPA targets, but scaling shards/replicas of a database has correctness implications that the HPA does not understand.
  - GPU-bound inference where the bottleneck is GPU memory: pair HPA with custom GPU metrics rather than CPU.

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

**Common Mistakes:**
- Confusing HPA (replica count) with VPA (per-Pod resource sizing) or the Cluster Autoscaler (node count).
- Using HPA on CPU for GPU inference: CPU utilization is rarely the true bottleneck — scale on a custom GPU or latency metric instead.
- Applying HPA to training Jobs: Jobs target a completion count, not a steady-state replica count.

**Related Material:** `lessons/mod-04-kubernetes/09-monitoring-troubleshooting.md`

---
