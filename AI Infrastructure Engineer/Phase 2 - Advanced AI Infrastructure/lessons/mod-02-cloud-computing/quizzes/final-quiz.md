# Module 02: Cloud Computing - Final Quiz

**Time Limit:** 75 minutes
**Passing Score:** 80% (40/50 questions)
**Coverage:** All Lessons (01-08)

---

## Section 1: Cloud Architecture for ML (Lesson 01, 5 questions)

### Question 1
**What is the primary advantage of cloud-native ML architectures?**

A) Lower total cost than on-premises
B) Faster training times guaranteed
C) Elasticity and scalability on demand
D) No need for monitoring

<details>
<summary>Answer</summary>
C) Elasticity and scalability on demand - Cloud-native architectures let compute grow and shrink with actual load, rather than being sized for peak demand year-round.
</details>

---

### Question 2
**Which storage type is best for training datasets in the cloud?**

A) Block storage (EBS)
B) Relational database
C) Local SSD only
D) Object storage (S3, GCS)

<details>
<summary>Answer</summary>
D) Object storage (S3, GCS) - Durable, cheap at scale, and the natural home for large training datasets accessed by many jobs.
</details>

---

### Question 3
**What is the purpose of a VPC in cloud ML infrastructure?**

A) To provide isolated network environment
B) To reduce costs
C) To speed up training
D) To enable multi-cloud deployments

<details>
<summary>Answer</summary>
A) To provide isolated network environment - A VPC gives your resources a private, software-defined network boundary, separate from other tenants and the public internet.
</details>

---

### Question 4
**Which compute option is most cost-effective for batch training jobs?**

A) On-demand instances
B) Spot/preemptible instances
C) Reserved instances
D) Dedicated hosts

<details>
<summary>Answer</summary>
B) Spot/preemptible instances - Batch training tolerates interruption (with checkpointing), which is exactly what spot capacity trades for a 60-90% discount.
</details>

---

### Question 5
**What is the trade-off of using serverless for ML inference?**

A) Lower cost vs higher complexity
B) Better security vs worse performance
C) Higher latency vs lower operational overhead
D) No trade-offs exist

<details>
<summary>Answer</summary>
C) Higher latency vs lower operational overhead - Serverless removes server management, but cold starts add latency that a warm, always-on endpoint doesn't have.
</details>

---

## Section 2: AWS for ML Infrastructure (Lesson 02, 5 questions)

### Question 6
**Which AWS service provides managed Kubernetes?**

A) ECS
B) Fargate
C) Lambda
D) EKS

<details>
<summary>Answer</summary>
D) EKS - Elastic Kubernetes Service is AWS's managed control plane for Kubernetes; ECS is AWS's own (non-Kubernetes) container orchestrator.
</details>

---

### Question 7
**What is the primary use of AWS SageMaker?**

A) Managed ML training and deployment
B) Object storage
C) Container orchestration
D) Networking

<details>
<summary>Answer</summary>
A) Managed ML training and deployment - SageMaker is AWS's end-to-end managed ML platform, covering training, tuning, and endpoint deployment.
</details>

---

### Question 8
**Which AWS storage service is best for large-scale object storage?**

A) EBS
B) S3
C) EFS
D) Instance Store

<details>
<summary>Answer</summary>
B) S3 - Simple Storage Service is AWS's object store; EBS is block storage and EFS is a shared file system, neither built for this scale of unstructured data.
</details>

---

### Question 9
**What does IAM stand for in AWS?**

A) Internet Access Management
B) Infrastructure Automation Manager
C) Identity and Access Management
D) Image Asset Manager

<details>
<summary>Answer</summary>
C) Identity and Access Management - IAM is AWS's system for controlling who (or what) can do what to which resources.
</details>

---

### Question 10
**Which EC2 instance type is optimized for GPU workloads?**

A) t3.large
B) m5.xlarge
C) c5.4xlarge
D) p3.2xlarge

<details>
<summary>Answer</summary>
D) p3.2xlarge - The p-series is AWS's GPU-accelerated instance family; t3/m5/c5 are general-purpose/compute-optimized with no GPU.
</details>

---

## Section 3: GCP for ML Infrastructure (Lesson 03, 5 questions)

### Question 11
**What is Google's managed ML platform called?**

A) Vertex AI
B) AI Platform
C) Cloud ML Engine
D) ML Studio

<details>
<summary>Answer</summary>
A) Vertex AI - Google's current unified ML platform; "AI Platform" and "Cloud ML Engine" were its predecessor names before the Vertex AI rebrand.
</details>

---

### Question 12
**Which GCP service provides managed Kubernetes?**

A) GCE
B) GKE
C) GAE
D) Cloud Run

<details>
<summary>Answer</summary>
B) GKE - Google Kubernetes Engine is GCP's managed Kubernetes; GCE is raw VMs, GAE is App Engine, Cloud Run is serverless containers (not Kubernetes).
</details>

---

### Question 13
**What is a TPU in Google Cloud?**

A) Total Performance Upgrade
B) Training Pipeline Utility
C) Tensor Processing Unit for ML acceleration
D) Temporary Processing Unit

<details>
<summary>Answer</summary>
C) Tensor Processing Unit for ML acceleration - Custom silicon Google designed specifically to accelerate tensor operations in ML training and inference.
</details>

---

### Question 14
**Which GCP storage is equivalent to AWS S3?**

A) Persistent Disk
B) Filestore
C) Cloud SQL
D) Cloud Storage

<details>
<summary>Answer</summary>
D) Cloud Storage - GCS is GCP's object store; Persistent Disk is block storage, Filestore is a managed file system, Cloud SQL is a relational database.
</details>

---

### Question 15
**What is the advantage of TPUs over GPUs for certain ML workloads?**

A) Optimized for TensorFlow operations
B) Cheaper always
C) Better for gaming
D) No advantages

<details>
<summary>Answer</summary>
A) Optimized for TensorFlow operations - TPUs are purpose-built for the matrix-multiplication-heavy operations common in TensorFlow/JAX training, not a general-purpose GPU replacement.
</details>

---

## Section 4: Azure for ML Infrastructure (Lesson 04, 5 questions)

### Question 16
**What is Azure's managed ML service called?**

A) Azure AI
B) Azure Machine Learning
C) Azure ML Studio
D) Azure Cognitive Services

<details>
<summary>Answer</summary>
B) Azure Machine Learning - Azure's end-to-end managed ML platform; Cognitive Services is Azure's pre-built AI APIs, a different product.
</details>

---

### Question 17
**Which Azure service provides managed Kubernetes?**

A) Azure Container Instances
B) Azure Functions
C) AKS (Azure Kubernetes Service)
D) Azure App Service

<details>
<summary>Answer</summary>
C) AKS (Azure Kubernetes Service) - Azure's managed Kubernetes control plane; Container Instances runs single containers without orchestration.
</details>

---

### Question 18
**What unique capability does Azure OpenAI Service provide?**

A) Free GPU access
B) Automated model training
C) Unlimited storage
D) Access to GPT models through Azure

<details>
<summary>Answer</summary>
D) Access to GPT models through Azure - Azure OpenAI is the only major cloud with first-party, enterprise-governed access to OpenAI's models.
</details>

---

### Question 19
**Which Azure storage is equivalent to AWS S3?**

A) Blob Storage
B) Azure Files
C) Azure Disk
D) Table Storage

<details>
<summary>Answer</summary>
A) Blob Storage - Azure's object store; Azure Files is a managed file share, Azure Disk is block storage, Table Storage is a NoSQL key-value store.
</details>

---

### Question 20
**What is a key advantage of Azure for enterprise customers?**

A) Lowest cost
B) Integration with Microsoft ecosystem
C) Fastest performance
D) No security features

<details>
<summary>Answer</summary>
B) Integration with Microsoft ecosystem - Native ties to Active Directory, Office 365, and enterprise compliance tooling are Azure's strongest differentiator, not raw price or speed.
</details>

---

## Section 5: Cloud Storage for ML (Lesson 05, 6 questions)

### Question 21
**What is the main advantage of object storage (S3, GCS) for ML datasets?**

A) Fastest access speed
B) Best for frequent updates
C) Scalability and cost-effectiveness
D) Requires no configuration

<details>
<summary>Answer</summary>
C) Scalability and cost-effectiveness - Object storage trades raw speed for durability and near-unlimited, cheap capacity, which is what large ML datasets need.
</details>

---

### Question 22
**When should you use block storage (EBS) instead of object storage?**

A) For archival data
B) For infrequently accessed data
C) Never, always use object storage
D) For high-IOPS database workloads

<details>
<summary>Answer</summary>
D) For high-IOPS database workloads - Block storage is attached directly to one instance and optimized for low-latency, high-IOPS random access that object storage can't match.
</details>

---

### Question 23
**What is a data lake?**

A) Centralized repository for structured and unstructured data
B) A relational database
C) A caching system
D) A type of VPN

<details>
<summary>Answer</summary>
A) Centralized repository for structured and unstructured data - A data lake holds raw and processed data of any format in one place, typically on object storage.
</details>

---

### Question 24
**Which caching strategy can improve ML inference performance?**

A) Never cache anything
B) Cache predictions for frequent inputs
C) Cache training data only
D) Cache models in RAM only

<details>
<summary>Answer</summary>
B) Cache predictions for frequent inputs - Caching repeated inference results avoids recomputation for the same input, cutting latency and compute cost.
</details>

---

### Question 25
**What is the difference between data lakes and data warehouses?**

A) No difference
B) Lakes are faster
C) Lakes store raw data, warehouses store processed data
D) Warehouses are cheaper

<details>
<summary>Answer</summary>
C) Lakes store raw data, warehouses store processed data - Lakes hold data in its native form for flexibility; warehouses hold cleaned, schema-enforced data optimized for querying.
</details>

---

### Question 26
**Which storage tier is most cost-effective for ML training data accessed monthly?**

A) Hot/Standard tier
B) Archive tier
C) Premium tier
D) Cool/Infrequent Access tier

<details>
<summary>Answer</summary>
D) Cool/Infrequent Access tier - Monthly access is too frequent for Archive (which has high retrieval cost/latency) but too infrequent to justify Hot pricing.
</details>

---

## Section 6: Cloud Networking for ML (Lesson 06, 6 questions)

### Question 27
**What is the purpose of a load balancer in ML systems?**

A) Distribute traffic across multiple instances
B) Store models
C) Train models faster
D) Reduce storage costs

<details>
<summary>Answer</summary>
A) Distribute traffic across multiple instances - Load balancers spread inference requests across model server replicas for scalability and availability.
</details>

---

### Question 28
**Why use a CDN for model serving?**

A) To train models faster
B) To reduce latency for global users
C) To increase security only
D) To save on storage costs

<details>
<summary>Answer</summary>
B) To reduce latency for global users - A CDN caches responses at edge locations near users, cutting the round-trip to a distant origin server.
</details>

---

### Question 29
**What is a service mesh like Istio used for?**

A) Training models
B) Storing data
C) Managing microservice communication
D) Monitoring only

<details>
<summary>Answer</summary>
C) Managing microservice communication - A service mesh handles traffic routing, retries, and observability between microservices, such as different model versions.
</details>

---

### Question 30
**What is the purpose of a VPN in hybrid cloud setups?**

A) Speed up training
B) Reduce costs
C) Store models
D) Secure connection between on-prem and cloud

<details>
<summary>Answer</summary>
D) Secure connection between on-prem and cloud - A site-to-site VPN tunnels traffic between an on-premises network and a cloud VPC over an encrypted connection.
</details>

---

### Question 31
**Which subnet design is recommended for production ML systems?**

A) Public subnet for load balancers, private for compute
B) All resources in one public subnet
C) No subnets needed
D) Random assignment

<details>
<summary>Answer</summary>
A) Public subnet for load balancers, private for compute - Only internet-facing components (load balancers, NAT, bastion) belong in a public subnet; training and inference compute stay private.
</details>

---

### Question 32
**What is the benefit of using private endpoints for cloud services?**

A) Cheaper costs
B) Traffic doesn't traverse public internet
C) Faster training
D) No benefits

<details>
<summary>Answer</summary>
B) Traffic doesn't traverse public internet - A private endpoint routes traffic to a cloud service entirely within the provider's network, reducing exposure.
</details>

---

## Section 7: Managed ML Services (Lesson 07, 6 questions)

### Question 33
**What is the main advantage of using SageMaker over building custom infrastructure?**

A) Always cheaper
B) Better model accuracy
C) Reduced operational overhead
D) Unlimited free tier

<details>
<summary>Answer</summary>
C) Reduced operational overhead - SageMaker takes over provisioning, tracking, and deployment plumbing; it doesn't change what the model itself can achieve, and often costs more than raw compute.
</details>

---

### Question 34
**What is a Feature Store?**

A) App store for ML models
B) Storage for datasets
C) Model registry
D) Centralized repository for ML features

<details>
<summary>Answer</summary>
D) Centralized repository for ML features - A feature store serves the same computed features consistently to both training (offline) and serving (online).
</details>

---

### Question 35
**What does a Model Registry provide?**

A) Versioning and metadata for models
B) Free models
C) Automatic model training
D) GPU access

<details>
<summary>Answer</summary>
A) Versioning and metadata for models - A registry tracks which model version is deployed where, along with lineage and approval status.
</details>

---

### Question 36
**Which managed service feature automatically tunes hyperparameters?**

A) Manual tuning
B) AutoML
C) Feature engineering
D) Data labeling

<details>
<summary>Answer</summary>
B) AutoML - AutoML searches architectures and hyperparameters automatically, without a human-written training loop.
</details>

---

### Question 37
**What is the trade-off of using managed ML services?**

A) No trade-offs
B) Always more expensive
C) Less control vs easier operations
D) Worse performance always

<details>
<summary>Answer</summary>
C) Less control vs easier operations - You give up some low-level configurability in exchange for not having to build and maintain the underlying plumbing yourself.
</details>

---

### Question 38
**Which cloud provider offers TPU access through their managed ML service?**

A) AWS only
B) Azure only
C) None of them
D) Google Cloud (Vertex AI)

<details>
<summary>Answer</summary>
D) Google Cloud (Vertex AI) - TPUs are Google's proprietary ML accelerator, available through Vertex AI; AWS and Azure don't offer TPU access.
</details>

---

## Section 8: Multi-Cloud and Cost Optimization (Lesson 08, 12 questions)

### Question 39
**What is a multi-cloud strategy?**

A) Using multiple cloud providers
B) Using multiple regions in one cloud
C) Multiple teams using cloud
D) Multiple accounts in one cloud

<details>
<summary>Answer</summary>
A) Using multiple cloud providers - Multi-cloud specifically means running workloads across more than one cloud vendor, not just multiple regions or accounts within one.
</details>

---

### Question 40
**What is the main advantage of Reserved Instances?**

A) Better performance
B) Significant cost savings (up to 75%)
C) More flexibility
D) Faster deployment

<details>
<summary>Answer</summary>
B) Significant cost savings (up to 75%) - Committing to 1-3 years of usage buys a steep discount over on-demand pricing, at the cost of flexibility, not performance.
</details>

---

### Question 41
**What are Spot Instances best used for?**

A) Production databases
B) Real-time inference
C) Fault-tolerant batch processing
D) Critical services

<details>
<summary>Answer</summary>
C) Fault-tolerant batch processing - Spot capacity can be reclaimed with short notice, so it only fits workloads (like checkpointed training) that tolerate interruption.
</details>

---

### Question 42
**What is the typical discount for Spot Instances?**

A) 10-20%
B) 5%
C) No discount
D) 50-90%

<details>
<summary>Answer</summary>
D) 50-90% - Spot/preemptible pricing reflects unused capacity, which is why the discount is so much steeper than Reserved Instances' 40-60%.
</details>

---

### Question 43
**Why set up billing alerts in cloud platforms?**

A) Prevent unexpected charges
B) Required by law
C) Get better performance
D) Access more services

<details>
<summary>Answer</summary>
A) Prevent unexpected charges - A billing alert catches a runaway cost trend (a misconfigured autoscaler, an orphaned GPU instance) before it becomes a surprise invoice.
</details>

---

### Question 44
**What is the purpose of cloud cost tagging?**

A) Make resources colorful
B) Track spending by project/team/environment
C) Speed up resources
D) No real purpose

<details>
<summary>Answer</summary>
B) Track spending by project/team/environment - Tags are what let a bill be attributed back to a specific team or project instead of showing up as one undifferentiated total.
</details>

---

### Question 45
**Which metric is important for ML cost optimization?**

A) Code quality only
B) Team size
C) Cost per training job / inference request
D) Office location

<details>
<summary>Answer</summary>
C) Cost per training job / inference request - Normalizing cost to a unit of work is what makes cost comparable across instance types, providers, and time.
</details>

---

### Question 46
**What is the benefit of auto-scaling for ML inference?**

A) Better accuracy
B) Faster training
C) No benefits
D) Match capacity to demand, optimize costs

<details>
<summary>Answer</summary>
D) Match capacity to demand, optimize costs - Auto-scaling adds replicas under load and removes them when idle, instead of paying for peak capacity around the clock.
</details>

---

### Question 47
**How can you reduce data transfer costs?**

A) Keep data and compute in same region
B) Never transfer data
C) Use slowest network
D) Transfer more frequently

<details>
<summary>Answer</summary>
A) Keep data and compute in same region - Cross-region and egress transfers carry per-GB fees that same-region traffic mostly avoids.
</details>

---

### Question 48
**What is the typical cost breakdown for ML cloud infrastructure?**

A) 90% compute, 10% storage
B) Compute, storage, and networking all significant
C) 100% storage
D) No compute costs

<details>
<summary>Answer</summary>
B) Compute, storage, and networking all significant - GPU compute usually dominates, but storage and data-transfer costs compound at scale and shouldn't be ignored.
</details>

---

### Question 49
**What is a Savings Plan in cloud billing?**

A) Free tier extension
B) One-time discount
C) Commitment to consistent usage for discounts
D) Loyalty program

<details>
<summary>Answer</summary>
C) Commitment to consistent usage for discounts - Unlike a Reserved Instance, a Savings Plan commits to a spend level rather than a specific instance type, trading some specificity for flexibility.
</details>

---

### Question 50
**Why use multiple availability zones for production ML systems?**

A) Better model accuracy
B) Cheaper costs
C) Faster training
D) High availability and fault tolerance

<details>
<summary>Answer</summary>
D) High availability and fault tolerance - Spreading replicas across AZs means a single AZ outage doesn't take the whole service down.
</details>

---

## Answer Key

1. C   2. D   3. A   4. B   5. C
6. D   7. A   8. B   9. C   10. D
11. A  12. B  13. C  14. D  15. A
16. B  17. C  18. D  19. A  20. B
21. C  22. D  23. A  24. B  25. C
26. D  27. A  28. B  29. C  30. D
31. A  32. B  33. C  34. D  35. A
36. B  37. C  38. D  39. A  40. B
41. C  42. D  43. A  44. B  45. C
46. D  47. A  48. B  49. C  50. D

---

## Scoring

- **45-50 correct (90-100%)**: Excellent! Mastered cloud computing for ML
- **40-44 correct (80-89%)**: Good! Ready for next module
- **35-39 correct (70-79%)**: Fair. Review weak areas
- **Below 35 (< 70%)**: Review module materials before continuing

---

## Areas for Review

If you scored poorly in a section, review:

- **Section 1**: Lesson 01 - Cloud Architecture for ML
- **Section 2**: Lesson 02 - AWS for ML Infrastructure
- **Section 3**: Lesson 03 - GCP for ML Infrastructure
- **Section 4**: Lesson 04 - Azure for ML Infrastructure
- **Section 5**: Lesson 05 - Cloud Storage for ML
- **Section 6**: Lesson 06 - Cloud Networking for ML
- **Section 7**: Lesson 07 - Managed ML Services
- **Section 8**: Lesson 08 - Multi-Cloud and Cost Optimization

---

## Next Steps

After passing this quiz:

1. Complete the Module 02 Capstone Project
2. Proceed to Module 03: Kubernetes Deep Dive
3. Or explore Module 04: Monitoring and Observability

---

**Congratulations on completing Module 02!**
