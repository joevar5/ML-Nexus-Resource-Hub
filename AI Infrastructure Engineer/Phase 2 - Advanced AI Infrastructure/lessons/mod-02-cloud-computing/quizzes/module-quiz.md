# Module 02 Quiz: Cloud Computing for ML

**Total Questions:** 20
**Passing Score:** 70% (14/20 correct)
**Time Limit:** 45 minutes
**Type:** Mixed (Multiple Choice, True/False, Short Answer)

---

## Section 1: Cloud Computing Fundamentals (Questions 1-5)

### Question 1
**Which cloud service model provides the most control over the underlying infrastructure?**

A) Software as a Service (SaaS)
B) Platform as a Service (PaaS)
C) Infrastructure as a Service (IaaS)
D) Function as a Service (FaaS)

<details>
<summary>Answer</summary>
C) Infrastructure as a Service (IaaS) - Provides maximum control over VMs, storage, and networking
</details>

---

### Question 2
**True or False: Reserved instances are always cheaper than spot instances for ML training workloads.**

<details>
<summary>Answer</summary>
False - Spot instances can be 70-90% cheaper than reserved instances, but they can be interrupted. Reserved instances provide guaranteed capacity at a discount compared to on-demand, but spot is generally cheaper for interruptible workloads.
</details>

---

### Question 3
**What is the primary benefit of using object storage (S3/GCS/Blob) for ML datasets?**

A) Fastest read/write performance
B) Lowest latency for random access
C) Unlimited scalability and durability at low cost
D) Built-in database query capabilities

<details>
<summary>Answer</summary>
C) Unlimited scalability and durability at low cost - Object storage is designed for massive scale, 11 9's of durability, and cost-effective storage of large datasets.
</details>

---

### Question 4
**Which GPU instance type would you choose for cost-effective training of a large language model that can tolerate interruptions?**

A) On-demand P4 instances (AWS)
B) Reserved P4 instances (AWS)
C) Spot P4 instances (AWS)
D) T2 instances with CPU only (AWS)

<details>
<summary>Answer</summary>
C) Spot P4 instances (AWS) - Spot instances offer 70-90% cost savings and are ideal for interruptible training workloads with checkpointing.
</details>

---

### Question 5
**What is the primary purpose of a VPC (Virtual Private Cloud)?**

A) To reduce cloud costs
B) To isolate and secure network resources
C) To speed up data transfers
D) To enable multi-cloud deployments

<details>
<summary>Answer</summary>
B) To isolate and secure network resources - VPCs provide network isolation, security groups, and controlled access to cloud resources.
</details>

---

## Section 2: Cloud Platform Specifics (Questions 6-10)

### Question 6
**Which managed ML service belongs to which cloud provider? Match correctly:**

1. SageMaker
2. Vertex AI
3. Azure Machine Learning

A) 1=AWS, 2=GCP, 3=Azure
B) 1=GCP, 2=AWS, 3=Azure
C) 1=Azure, 2=GCP, 3=AWS
D) 1=AWS, 2=Azure, 3=GCP

<details>
<summary>Answer</summary>
A) 1=AWS (SageMaker), 2=GCP (Vertex AI), 3=Azure (Azure Machine Learning)
</details>

---

### Question 7
**What is unique to Google Cloud Platform compared to AWS and Azure?**

A) Kubernetes service (GKE)
B) TPUs (Tensor Processing Units)
C) Object storage service
D) Virtual machines with GPUs

<details>
<summary>Answer</summary>
B) TPUs (Tensor Processing Units) - Google Cloud is the only major cloud provider offering TPUs, which are specialized hardware for ML training and inference.
</details>

---

### Question 8
**In AWS, what service would you use to orchestrate containerized ML workloads at scale?**

A) EC2 (Elastic Compute Cloud)
B) Lambda (Serverless Functions)
C) EKS (Elastic Kubernetes Service)
D) S3 (Simple Storage Service)

<details>
<summary>Answer</summary>
C) EKS (Elastic Kubernetes Service) - EKS is AWS's managed Kubernetes service for orchestrating containerized applications at scale.
</details>

---

### Question 9
**Which Azure service integrates directly with OpenAI models for LLM deployments?**

A) Azure Kubernetes Service
B) Azure OpenAI Service
C) Azure Machine Learning
D) Azure Functions

<details>
<summary>Answer</summary>
B) Azure OpenAI Service - Provides managed access to OpenAI models (GPT-4, GPT-3.5, DALL-E) with enterprise features.
</details>

---

### Question 10
**What is the GCP equivalent of AWS S3?**

A) Cloud Storage (GCS)
B) Persistent Disk
C) Filestore
D) Cloud SQL

<details>
<summary>Answer</summary>
A) Cloud Storage (GCS) - Google Cloud Storage is GCP's object storage service, equivalent to AWS S3.
</details>

---

## Section 3: Storage and Networking (Questions 11-15)

### Question 11
**Which storage type provides the best performance for high-throughput ML training data access?**

A) Object storage (S3/GCS)
B) Block storage (EBS/Persistent Disk)
C) File storage with NFS
D) Database storage (RDS/Cloud SQL)

<details>
<summary>Answer</summary>
B) Block storage (EBS/Persistent Disk) - Block storage provides the highest IOPS and throughput, ideal for training workloads requiring fast random access. For even better performance, ephemeral SSD or local NVMe is best.
</details>

---

### Question 12
**What is the purpose of a security group in cloud networking?**

A) To encrypt data at rest
B) To act as a virtual firewall controlling inbound/outbound traffic
C) To load balance traffic across instances
D) To monitor security threats

<details>
<summary>Answer</summary>
B) To act as a virtual firewall controlling inbound/outbound traffic - Security groups define rules for allowed network traffic to/from resources.
</details>

---

### Question 13
**True or False: Data transfer within the same availability zone is typically free.**

<details>
<summary>Answer</summary>
True - Most cloud providers don't charge for data transfer within the same availability zone, but charge for cross-AZ and cross-region transfers.
</details>

---

### Question 14
**What is the recommended approach for storing model artifacts and checkpoints during training?**

A) Local instance storage only
B) Object storage (S3/GCS) with periodic checkpointing
C) Database (RDS/Cloud SQL)
D) In-memory only

<details>
<summary>Answer</summary>
B) Object storage (S3/GCS) with periodic checkpointing - Object storage is durable, cost-effective, and prevents data loss if instances are interrupted.
</details>

---

### Question 15
**Which networking component would you use to distribute incoming ML inference requests across multiple instances?**

A) Security Group
B) VPC
C) Load Balancer
D) NAT Gateway

<details>
<summary>Answer</summary>
C) Load Balancer - Load balancers (ALB, NLB, or cloud equivalents) distribute traffic across multiple backend instances for high availability and scalability.
</details>

---

## Section 4: Cost Optimization and Best Practices (Questions 16-20)

### Question 16
**Which strategy would MOST reduce costs for ML training workloads?**

A) Use only on-demand instances for reliability
B) Use spot instances with checkpointing and automatic restart
C) Use reserved instances for all workloads
D) Run training on the most powerful GPU available

<details>
<summary>Answer</summary>
B) Use spot instances with checkpointing and automatic restart - Spot instances can save 70-90% compared to on-demand, and with proper checkpointing, interruptions are manageable.
</details>

---

### Question 17
**What is the typical cost difference between ingress (data in) and egress (data out) for major cloud providers?**

A) Ingress is expensive, egress is free
B) Both ingress and egress are expensive
C) Ingress is free, egress is expensive
D) Both ingress and egress are free

<details>
<summary>Answer</summary>
C) Ingress is free, egress is expensive - Cloud providers typically don't charge for data uploads (ingress) but charge for data downloads (egress), especially cross-region or to the internet.
</details>

---

### Question 18
**Short Answer: Name THREE strategies to optimize cloud costs for ML workloads.**

<details>
<summary>Sample Answer</summary>

1. **Use Spot Instances:** For interruptible training workloads with checkpointing (70-90% savings)
2. **Right-size Instances:** Monitor utilization and choose appropriate instance types (avoid over-provisioning)
3. **Auto-scaling:** Scale down during low-demand periods, scale up during peak inference loads
4. **Storage Lifecycle Policies:** Move infrequently accessed data to cheaper storage tiers (Glacier, Coldline)
5. **Reserved Instances:** For predictable, long-running workloads (40-60% savings vs on-demand)
6. **Optimize Data Transfer:** Minimize cross-region and egress costs by co-locating data and compute

(Any 3 valid strategies acceptable)
</details>

---

### Question 19
**What is the primary reason to implement a multi-cloud strategy?**

A) To maximize costs by using multiple providers
B) To avoid vendor lock-in and improve resilience
C) To make architecture more complex
D) To use free tiers from multiple providers

<details>
<summary>Answer</summary>
B) To avoid vendor lock-in and improve resilience - Multi-cloud strategies provide flexibility, reduce dependency on a single vendor, and can improve disaster recovery and compliance capabilities.
</details>

---

### Question 20
**True or False: It's a best practice to hard-code cloud credentials directly in application code for convenience.**

<details>
<summary>Answer</summary>
False - Credentials should NEVER be hard-coded. Use IAM roles, service accounts, secret management services (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault), or environment variables instead.
</details>

---

## Scoring Guide

### Grading Rubric

- **18-20 correct (90-100%):** Excellent - You have mastered cloud computing concepts for ML
- **16-17 correct (80-89%):** Very Good - Strong understanding with minor gaps
- **14-15 correct (70-79%):** Passing - Adequate knowledge, review areas where you struggled
- **Below 14 (< 70%):** Not Passing - Review module material and retake quiz

### What to Do Next

**If you passed (≥ 70%):**
1. Review any questions you got wrong
2. Complete the Module 02 practical exercises
3. Optionally work on the capstone project
4. Proceed to Module 03: Containerization with Docker

**If you didn't pass (< 70%):**
1. Review the lessons corresponding to questions you missed
2. Go through hands-on exercises again
3. Retake the quiz after additional study
4. Reach out in GitHub Discussions if you need clarification

---

## Answer Key Summary

1. C
2. False
3. C
4. C
5. B
6. A
7. B
8. C
9. B
10. A
11. B
12. B
13. True
14. B
15. C
16. B
17. C
18. Short Answer (see details)
19. B
20. False

---

## Self-Assessment Questions

After completing the quiz, reflect on:

1. Which cloud platform am I most comfortable with?
2. Do I understand the cost implications of different storage and compute choices?
3. Can I design a secure network architecture for ML workloads?
4. Am I confident in comparing and choosing between cloud services?
5. What topics should I review before proceeding to Module 03?

---

**Time to Review:** 15-30 minutes to review answers and understand mistakes

**Ready for Module 03?** If you scored ≥ 70% and completed the exercises, you're ready to move forward!
