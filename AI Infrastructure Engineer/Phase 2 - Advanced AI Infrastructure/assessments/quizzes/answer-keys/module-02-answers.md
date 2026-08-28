# Module 02: Cloud Computing for ML — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-02-cloud-computing/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz. Do not use as a primary reference during the quiz.

---

## Question 1
**Q:** Which cloud service model provides the most control over the underlying infrastructure?

**Answer:** C) Infrastructure as a Service (IaaS)

**Explanation:**
IaaS exposes the lowest-level cloud primitives — virtual machines, virtual disks, virtual networks, and security groups — and leaves the OS, runtime, and application stack to you. That maximal control is exactly why ML teams pick IaaS when they need to install custom CUDA drivers, tune kernels, or run non-standard frameworks. PaaS, SaaS, and FaaS all trade away progressively more control in exchange for higher-level abstractions.

**Common Mistakes:**
- Choosing A (SaaS) by confusing "most managed" with "most control" — SaaS gives you the least control because the entire application is operated by the provider.
- Choosing D (FaaS) because serverless feels "powerful" — in reality FaaS hides the OS, runtime, and scaling entirely, giving you almost no infrastructure control.

**Related Material:** [lessons/mod-02-cloud-computing/01-cloud-architecture.md](../../../lessons/mod-02-cloud-computing/01-cloud-architecture.md)

---

## Question 2
**Q:** True or False: Reserved instances are always cheaper than spot instances for ML training workloads.

**Answer:** False

**Explanation:**
Spot/preemptible instances typically run 70–90% cheaper than on-demand, which is usually well below the 40–60% discount that 1- or 3-year reserved instances provide. Reserved instances win on *guaranteed capacity*, not raw price. For interruptible training jobs with checkpointing, spot is almost always the cheaper option; reserved is better for steady-state inference where uptime matters more than absolute cost.

**Common Mistakes:**
- Answering True because reserved instances *sound* like the most committed (and therefore cheapest) option — commitment buys availability, not the lowest unit price.
- Forgetting that "always" in a True/False statement is a strong claim that fails the moment a single counter-example (spot training jobs) exists.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 3
**Q:** What is the primary benefit of using object storage (S3/GCS/Blob) for ML datasets?

**Answer:** C) Unlimited scalability and durability at low cost

**Explanation:**
Object stores are designed for effectively unlimited capacity, ~11 nines of durability, and per-GB pricing that beats block or file storage for cold/warm data. That makes them the natural home for large training datasets, model artifacts, and checkpoints. They are *not* optimized for low-latency random access or transactional querying — those are the trade-offs you accept in exchange for scale and cost.

**Common Mistakes:**
- Choosing A (fastest read/write) — object storage has higher per-request latency than block storage; throughput scales with parallel requests, not single-request speed.
- Choosing B (lowest latency for random access) — block storage and local NVMe are the right pick when random access latency matters.
- Choosing D (built-in query) — query engines like Athena/BigQuery sit *on top* of object storage; the storage tier itself is not a database.

**Related Material:** [lessons/mod-02-cloud-computing/05-cloud-storage.md](../../../lessons/mod-02-cloud-computing/05-cloud-storage.md)

---

## Question 4
**Q:** Which GPU instance type would you choose for cost-effective training of a large language model that can tolerate interruptions?

**Answer:** C) Spot P4 instances (AWS)

**Explanation:**
The question hands you the two key constraints — *cost-effective* and *tolerates interruptions* — which is the textbook fit for spot pricing. P4 instances (A100 GPUs) on spot are 70–90% cheaper than on-demand, and a job with regular checkpointing simply resumes from the last checkpoint if reclaimed. On-demand and reserved leave that discount on the table; T2 has no GPU at all.

**Common Mistakes:**
- Choosing B (reserved P4) because reservations sound "committed and cheap" — reserved is better for steady inference, not bursty training.
- Choosing D (T2 CPU-only) because it looks cheapest per-hour — training an LLM on CPU is so slow that total job cost dwarfs any per-hour savings.

**Related Material:** [lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md](../../../lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md)

---

## Question 5
**Q:** What is the primary purpose of a VPC (Virtual Private Cloud)?

**Answer:** B) To isolate and secure network resources

**Explanation:**
A VPC is a logically isolated slice of the cloud provider's network where you control IP ranges, subnets, route tables, security groups, and gateway access. It is fundamentally a *security and isolation* construct. Cost, transfer speed, and multi-cloud are all unrelated concerns.

**Common Mistakes:**
- Choosing A (reduce costs) — VPC itself is generally free; it is not a cost-control mechanism.
- Choosing C (speed up transfers) — placement and instance type drive throughput, not the existence of a VPC.

**Related Material:** [lessons/mod-02-cloud-computing/06-cloud-networking.md](../../../lessons/mod-02-cloud-computing/06-cloud-networking.md)

---

## Question 6
**Q:** Which managed ML service belongs to which cloud provider? Match correctly:
1. SageMaker  2. Vertex AI  3. Azure Machine Learning

**Answer:** A) 1=AWS, 2=GCP, 3=Azure

**Explanation:**
SageMaker is AWS's flagship managed ML platform, Vertex AI is GCP's unified ML platform (the successor to AI Platform), and Azure Machine Learning is Microsoft's. Memorizing the trio is foundational to comparing managed ML offerings later in the module.

**Common Mistakes:**
- Choosing B or D because of confusion about Vertex AI — it is GCP, not AWS or Azure.
- Choosing C because the name "Azure Machine Learning" gets swapped with Azure OpenAI Service in memory; the two are distinct products.

**Related Material:** [lessons/mod-02-cloud-computing/07-managed-ml-services.md](../../../lessons/mod-02-cloud-computing/07-managed-ml-services.md)

---

## Question 7
**Q:** What is unique to Google Cloud Platform compared to AWS and Azure?

**Answer:** B) TPUs (Tensor Processing Units)

**Explanation:**
TPUs are Google-designed ASICs for matrix-heavy ML workloads and are only available through GCP (Cloud TPU and TPU VMs). All three major clouds offer Kubernetes, object storage, and GPU VMs, so those options are not unique. TPU access is one of the strongest reasons teams reach for GCP specifically.

**Common Mistakes:**
- Choosing A (Kubernetes) — EKS, AKS, and GKE all exist; managed Kubernetes is table stakes, not a GCP differentiator.
- Choosing C or D — both object storage and GPU VMs exist on every major provider.

**Related Material:** [lessons/mod-02-cloud-computing/03-gcp-ml-infrastructure.md](../../../lessons/mod-02-cloud-computing/03-gcp-ml-infrastructure.md)

---

## Question 8
**Q:** In AWS, what service would you use to orchestrate containerized ML workloads at scale?

**Answer:** C) EKS (Elastic Kubernetes Service)

**Explanation:**
EKS is AWS's managed Kubernetes control plane and is the standard way to run containerized ML workloads (training jobs, inference services, Kubeflow, Ray) at scale. EC2 alone gives you VMs but no orchestration, Lambda is for short-lived stateless functions, and S3 is storage — none of those are container orchestrators.

**Common Mistakes:**
- Choosing A (EC2) — EC2 provides the compute substrate but no scheduling, scaling, or service discovery for containers.
- Choosing B (Lambda) — Lambda has hard time and memory limits and is unsuited to long-running GPU training or large model serving.

**Related Material:** [lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md](../../../lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md)

---

## Question 9
**Q:** Which Azure service integrates directly with OpenAI models for LLM deployments?

**Answer:** B) Azure OpenAI Service

**Explanation:**
Azure OpenAI Service is the Microsoft-hosted, enterprise-grade gateway to GPT-4, GPT-3.5, DALL-E, and embeddings models, including SLAs, private networking, and data residency guarantees. AKS, Azure ML, and Functions are general-purpose services that *can* host models but don't provide first-party OpenAI integration.

**Common Mistakes:**
- Choosing C (Azure Machine Learning) because it is Azure's general ML platform — it does not, by itself, give you OpenAI model access.
- Choosing A (AKS) because container orchestration is involved in serving LLMs — but AKS is generic infrastructure, not an OpenAI integration.

**Related Material:** [lessons/mod-02-cloud-computing/04-azure-ml-infrastructure.md](../../../lessons/mod-02-cloud-computing/04-azure-ml-infrastructure.md)

---

## Question 10
**Q:** What is the GCP equivalent of AWS S3?

**Answer:** A) Cloud Storage (GCS)

**Explanation:**
Google Cloud Storage is GCP's object storage service and the direct analogue to AWS S3 and Azure Blob Storage. Persistent Disk is block storage (analogous to EBS), Filestore is managed NFS file storage (analogous to EFS), and Cloud SQL is a managed relational database.

**Common Mistakes:**
- Choosing B (Persistent Disk) — block storage attached to a single VM is the EBS equivalent, not the S3 equivalent.
- Choosing C (Filestore) — NFS-style shared file systems are the EFS equivalent.

**Related Material:** [lessons/mod-02-cloud-computing/05-cloud-storage.md](../../../lessons/mod-02-cloud-computing/05-cloud-storage.md)

---

## Question 11
**Q:** Which storage type provides the best performance for high-throughput ML training data access?

**Answer:** B) Block storage (EBS/Persistent Disk)

**Explanation:**
Block storage delivers the highest IOPS and lowest per-request latency of the managed options, which matters for tight training loops that re-read shards quickly. Object storage scales throughput with parallelism but pays per-request overhead; NFS adds protocol overhead and metadata bottlenecks; databases are simply the wrong tool. For even higher performance, ephemeral local NVMe on the instance beats network block storage.

**Common Mistakes:**
- Choosing A (object storage) because "S3 scales infinitely" — aggregate throughput can be high, but per-file latency hurts active training loops without a caching layer.
- Choosing C (NFS) thinking shared file semantics are inherently fast — NFS adds metadata and locking overhead that constrains throughput at scale.

**Related Material:** [lessons/mod-02-cloud-computing/05-cloud-storage.md](../../../lessons/mod-02-cloud-computing/05-cloud-storage.md)

---

## Question 12
**Q:** What is the purpose of a security group in cloud networking?

**Answer:** B) To act as a virtual firewall controlling inbound/outbound traffic

**Explanation:**
Security groups are stateful virtual firewalls attached to instances or ENIs that define allow rules for inbound and outbound traffic by port, protocol, and source/destination. They do not encrypt data, balance load, or perform threat detection — those are KMS, ELB, and GuardDuty/Defender concerns respectively.

**Common Mistakes:**
- Choosing A (encrypt data at rest) — encryption is handled by KMS, EBS encryption, or S3 SSE, not by network rules.
- Choosing D (monitor security threats) — confusing access control with detection; threat monitoring is GuardDuty, Defender, or Security Command Center.

**Related Material:** [lessons/mod-02-cloud-computing/06-cloud-networking.md](../../../lessons/mod-02-cloud-computing/06-cloud-networking.md)

---

## Question 13
**Q:** True or False: Data transfer within the same availability zone is typically free.

**Answer:** True

**Explanation:**
Major providers generally do not charge for traffic between resources in the same AZ over private IPs. Cross-AZ traffic typically costs a few cents per GB in each direction, and cross-region or internet egress costs significantly more. Co-locating compute and storage in the same AZ is therefore one of the simplest cost wins for data-heavy ML workloads.

**Common Mistakes:**
- Answering False because of a vague recollection that "cloud data transfer always costs money" — egress costs are real, but same-AZ private traffic is the exception.
- Conflating cross-AZ (charged) with same-AZ (free) — the AZ boundary is where the meter starts.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 14
**Q:** What is the recommended approach for storing model artifacts and checkpoints during training?

**Answer:** B) Object storage (S3/GCS) with periodic checkpointing

**Explanation:**
Object storage is durable (~11 nines), cheap per GB, accessible from any instance, and survives instance termination — all critical when spot/preemptible nodes can disappear mid-training. Periodic checkpointing to object storage is the established pattern that lets you resume training without data loss. Local instance storage is ephemeral, databases are not designed for large binary blobs, and in-memory storage is lost on any failure.

**Common Mistakes:**
- Choosing A (local instance storage) — fast, but checkpoints vanish when the instance is reclaimed, which is the whole reason you checkpoint in the first place.
- Choosing C (database) — RDS/Cloud SQL are expensive, row-oriented, and a poor fit for multi-GB binary artifacts.

**Related Material:** [lessons/mod-02-cloud-computing/05-cloud-storage.md](../../../lessons/mod-02-cloud-computing/05-cloud-storage.md)

---

## Question 15
**Q:** Which networking component would you use to distribute incoming ML inference requests across multiple instances?

**Answer:** C) Load Balancer

**Explanation:**
Load balancers (ALB/NLB on AWS, Cloud Load Balancing on GCP, Azure Load Balancer/Application Gateway) terminate client connections and distribute requests across a pool of backends, providing health checks and high availability for inference endpoints. Security groups filter traffic, VPCs define network boundaries, and NAT gateways enable outbound internet from private subnets — none of those distribute incoming traffic.

**Common Mistakes:**
- Choosing A (Security Group) — security groups decide *whether* traffic is allowed, not how to spread it across backends.
- Choosing D (NAT Gateway) — NAT handles outbound private-to-internet traffic, the opposite direction from inbound inference requests.

**Related Material:** [lessons/mod-02-cloud-computing/06-cloud-networking.md](../../../lessons/mod-02-cloud-computing/06-cloud-networking.md)

---

## Question 16
**Q:** Which strategy would MOST reduce costs for ML training workloads?

**Answer:** B) Use spot instances with checkpointing and automatic restart

**Explanation:**
Spot/preemptible instances cut compute cost by 70–90%, and pairing them with checkpoint-and-resume logic makes interruptions a non-event for batch training. That is a larger discount than reserved instances offer, and far better than running everything on the most expensive GPU at on-demand prices. The keyword in the question is "MOST" — spot wins on raw savings.

**Common Mistakes:**
- Choosing C (reserved for all workloads) — reservations help steady-state usage but lock you in and still cost more than spot for interruptible jobs.
- Choosing D (always run on the most powerful GPU) — over-provisioning hardware is one of the largest sources of waste; right-sizing beats brute force.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 17
**Q:** What is the typical cost difference between ingress (data in) and egress (data out) for major cloud providers?

**Answer:** C) Ingress is free, egress is expensive

**Explanation:**
All three major clouds let you upload data for free but charge per GB for egress to the internet and (at a lower rate) cross-region. This pricing asymmetry is deliberate — it lowers the barrier to onboarding data and raises the cost of leaving, which is a key driver of vendor lock-in. Architecting to keep compute close to data and minimize egress is a core cost-optimization habit.

**Common Mistakes:**
- Choosing A (ingress expensive, egress free) — this is the exact inverse of how cloud pricing works.
- Choosing D (both free) — a common myth, especially among engineers who have only worked in the free tier where small egress quotas hide the real cost curve.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 18
**Q:** Short Answer: Name THREE strategies to optimize cloud costs for ML workloads.

**Answer:** Any three of the following are acceptable:
1. **Use spot/preemptible instances** for interruptible training with checkpointing (70–90% savings).
2. **Right-size instances** by monitoring utilization and matching instance type to actual GPU/CPU/memory demand.
3. **Auto-scaling** to scale inference fleets down at off-peak and up at peak load.
4. **Storage lifecycle policies** that move cold data to cheaper tiers (S3 Glacier, GCS Coldline, Azure Archive).
5. **Reserved instances or savings plans** for predictable, steady-state workloads (40–60% savings).
6. **Minimize data transfer costs** by co-locating compute and data and avoiding cross-region/egress traffic.

**Explanation:**
A strong answer names three distinct levers, ideally spanning compute (spot/reserved/right-sizing), storage (lifecycle tiering), and network (data locality/egress). Listing three flavors of the same lever — for example "spot, preemptible, and interruptible instances" — should not score full credit because it shows only one underlying strategy.

**Common Mistakes:**
- Listing vague platitudes like "use less compute" or "monitor your bill" without naming a concrete mechanism.
- Naming three variants of the same concept (e.g., spot + preemptible + interruptible) instead of three distinct strategies.
- Omitting any mention of data-transfer/egress cost, which is often the largest hidden line item for ML workloads.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 19
**Q:** What is the primary reason to implement a multi-cloud strategy?

**Answer:** B) To avoid vendor lock-in and improve resilience

**Explanation:**
Multi-cloud reduces dependence on any single provider's pricing power, outages, and roadmap, and can satisfy regulatory or data-residency requirements that one cloud cannot. It generally *increases* operational complexity and cost in exchange for flexibility — so the trade-off is deliberate, not accidental. The other options either invert the rationale or describe side effects.

**Common Mistakes:**
- Choosing C (more complex architecture) — added complexity is a *cost* of multi-cloud, not a goal.
- Choosing D (use free tiers from multiple providers) — stitching free tiers together is a hobbyist tactic, not a strategic reason serious teams go multi-cloud.

**Related Material:** [lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md](../../../lessons/mod-02-cloud-computing/08-multi-cloud-cost-optimization.md)

---

## Question 20
**Q:** True or False: It's a best practice to hard-code cloud credentials directly in application code for convenience.

**Answer:** False

**Explanation:**
Hard-coded credentials inevitably leak — into git history, container images, logs, or screenshots — and they bypass the audit, rotation, and least-privilege controls the cloud provides. The correct patterns are IAM roles (instance/pod identity), service accounts with workload identity, secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault), and at minimum environment variables sourced from a secret store. This rule has no exceptions for "convenience."

**Common Mistakes:**
- Answering True because hard-coding "works" in development — convenience now translates directly into a credential rotation incident later.
- Assuming a private repo is safe enough — once a secret is committed, it must be treated as compromised even if the repo is private.

**Related Material:** [lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md](../../../lessons/mod-02-cloud-computing/02-aws-ml-infrastructure.md)

---
