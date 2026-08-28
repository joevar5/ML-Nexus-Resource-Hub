# Lesson 01: Cloud Architecture for ML

**Duration:** 6 hours
**Objectives:** Understand cloud architecture patterns and design principles for ML systems

## Introduction

Architecting ML systems in the cloud requires understanding both general cloud architecture principles and ML-specific considerations. This lesson covers fundamental cloud architecture patterns, components, and design decisions that impact ML infrastructure.

## Cloud Architecture Fundamentals

### The Three Pillars of Cloud Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Cloud Architecture Pillars                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. COMPUTE                2. STORAGE        3. NETWORK │
│  ┌──────────┐             ┌──────────┐     ┌─────────┐ │
│  │   VMs    │             │  Object  │     │   VPC   │ │
│  │ Containers│            │  Block   │     │  Subnet │ │
│  │Serverless│             │   File   │     │   LB    │ │
│  └──────────┘             └──────────┘     └─────────┘ │
│       ↓                        ↓                 ↓      │
│  Run ML code           Store data/models    Connect it  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 1. Compute Resources

**Virtual Machines (VMs)**
- Full control over OS and software
- Suitable for complex ML pipelines
- Can run 24/7 or on-demand
- Examples: AWS EC2, GCP Compute Engine, Azure VMs

**Containers**
- Portable and consistent environments
- Faster startup than VMs
- Ideal for ML model serving
- Examples: AWS ECS/EKS, GCP GKE, Azure AKS

**Serverless**
- No server management
- Auto-scaling
- Pay per execution
- Examples: AWS Lambda, GCP Cloud Functions, Azure Functions

**When to use each:**
```
VMs:           Long-running training jobs, complex dependencies
Containers:    Model serving, microservices, portability
Serverless:    Lightweight inference, event-driven processing
```

### 2. Storage Resources

**Object Storage**
- Unlimited scalability
- Store datasets, models, logs
- HTTP/S access
- Examples: AWS S3, GCP Cloud Storage, Azure Blob Storage
- Cost: ~$0.02/GB/month

**Block Storage**
- Attached to compute instances
- Low latency, high IOPS
- Databases, application data
- Examples: AWS EBS, GCP Persistent Disk, Azure Managed Disks
- Cost: ~$0.10/GB/month

**File Storage**
- Shared file systems
- NFS protocol
- Multi-instance access
- Examples: AWS EFS, GCP Filestore, Azure Files
- Cost: ~$0.30/GB/month

**When to use each:**
```
Object Storage:  Datasets, trained models, backups, logs
Block Storage:   Databases, training data cache, checkpoints
File Storage:    Shared training data, home directories
```

### 3. Networking Resources

**Virtual Private Cloud (VPC)**
- Isolated network environment
- Define IP ranges, subnets
- Control traffic flow

**Load Balancers**
- Distribute traffic across instances
- Health checking
- SSL termination

**Content Delivery Network (CDN)**
- Cache model predictions
- Reduce latency for users
- Examples: AWS CloudFront, GCP Cloud CDN, Azure CDN

## ML-Specific Architecture Patterns

### Pattern 1: Training Architecture

```
┌──────────────────────────────────────────────────────────┐
│              ML Training Architecture                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Data Storage          Compute              Output       │
│  ┌──────────┐         ┌────────┐         ┌──────────┐   │
│  │ S3/GCS   │ ──────► │  GPU   │ ──────► │  Model   │   │
│  │ Dataset  │         │Instance│         │  Storage │   │
│  │  (1TB)   │         │Training│         │  (S3/GCS)│   │
│  └──────────┘         └────────┘         └──────────┘   │
│       │                    │                   │         │
│       │                    ↓                   │         │
│       │              ┌────────┐                │         │
│       │              │Metrics │                │         │
│       │              │Tracking│                │         │
│       │              │(MLflow)│                │         │
│       │              └────────┘                │         │
│       │                                        │         │
│       └────────────── Clean up ───────────────┘         │
│                                                           │
└──────────────────────────────────────────────────────────┘

Components:
1. Data Storage: S3/GCS bucket with versioned datasets
2. Training Instance: GPU instance (p3.2xlarge, n1-highmem-8)
3. Experiment Tracking: MLflow on separate instance
4. Model Registry: Versioned models in object storage
5. Monitoring: CloudWatch/Stackdriver for GPU utilization
```

**Key Considerations:**
- Use spot instances for cost savings (up to 90% off)
- Implement checkpointing for fault tolerance
- Auto-shutdown after training completes
- Version datasets and models
- Track experiments with MLflow/W&B

**Cost Estimation:**
```
Training Job (ResNet-50 on ImageNet):
- GPU Instance (p3.2xlarge): $3.06/hour
- Training Time: 24 hours
- Storage (1TB dataset): $20/month
- Data Transfer: $50
──────────────────────────────────────
Total: ~$150 per training run
```

### Pattern 2: Inference Architecture

```
┌──────────────────────────────────────────────────────────┐
│             ML Inference Architecture                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────┐       ┌──────────┐       ┌────────────┐     │
│  │  CDN   │       │   Load   │       │  Inference │     │
│  │ Cache  │ ◄───  │ Balancer │  ───► │  Instances │     │
│  └────────┘       └──────────┘       │  (Auto-    │     │
│                         │             │   scaled)  │     │
│                         │             └────────────┘     │
│                         │                   │            │
│                         ↓                   ↓            │
│                   ┌──────────┐       ┌────────────┐     │
│                   │  Metrics │       │   Model    │     │
│                   │ (Prom/   │       │  Storage   │     │
│                   │  Stack)  │       │  (S3/GCS)  │     │
│                   └──────────┘       └────────────┘     │
│                                                           │
└──────────────────────────────────────────────────────────┘

Components:
1. CDN: Cache responses at edge locations
2. Load Balancer: Distribute traffic, SSL termination
3. Auto-Scaling Group: 2-10 instances based on load
4. Model Storage: Versioned models in S3/GCS
5. Monitoring: Prometheus + Grafana or CloudWatch
6. Caching: Redis for frequent predictions
```

**Key Considerations:**
- Use auto-scaling for variable load
- Implement caching at multiple levels
- Use CDN for static assets
- Monitor latency (p50, p95, p99)
- Implement circuit breakers

**Cost Estimation:**
```
Inference Service (1M requests/month):
- Load Balancer: $18/month
- Compute (3x t3.medium): $90/month
- Data Transfer: $80/month
- CDN: $40/month
──────────────────────────────────────
Total: ~$230/month
```

### Pattern 3: Data Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Data Pipeline Architecture                   │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Ingestion         Processing        Storage             │
│  ┌─────────┐       ┌─────────┐     ┌─────────┐          │
│  │ API/    │ ───► │ Spark/  │ ──► │  Data   │          │
│  │ Stream  │       │ Airflow │     │  Lake   │          │
│  └─────────┘       └─────────┘     │ (S3/GCS)│          │
│       │                 │           └─────────┘          │
│       │                 │                 │              │
│       │                 ↓                 ↓              │
│       │           ┌─────────┐       ┌─────────┐         │
│       │           │Transform│       │ Feature │         │
│       │           │  Jobs   │       │  Store  │         │
│       │           └─────────┘       └─────────┘         │
│       │                                   │              │
│       └──────────────────────────────────┘              │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Architecture Design Principles

### 1. High Availability

**Definition:** System remains operational even when components fail

**Strategies:**
- **Multi-AZ Deployment**: Distribute across availability zones
- **Load Balancing**: Distribute traffic across instances
- **Health Checks**: Automatic failover to healthy instances
- **Redundancy**: Multiple instances of critical components

**Example:**
```
Single AZ (99.5% uptime):
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
   ┌───┴───────┐
   │  2 VMs    │  (Single point of failure: AZ outage)
   │ (Same AZ) │
   └───────────┘

Multi-AZ (99.99% uptime):
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
   ┌───┴────────────────┐
   │                    │
┌──┴────┐          ┌────┴──┐
│ AZ-1  │          │ AZ-2  │
│ 2 VMs │          │ 2 VMs │
└───────┘          └───────┘
```

### 2. Scalability

**Vertical Scaling (Scale Up)**
- Increase instance size (more CPU/RAM/GPU)
- Simpler but has limits
- Requires restart

**Horizontal Scaling (Scale Out)**
- Add more instances
- Unlimited scaling potential
- No downtime

**Auto-Scaling Example:**
```yaml
# AWS Auto Scaling Configuration
MinSize: 2              # Minimum instances
MaxSize: 10             # Maximum instances
DesiredCapacity: 3      # Initial instances
TargetCPU: 70%          # Scale when CPU > 70%

Scaling Policies:
- Scale Up: Add 2 instances when CPU > 70% for 5 minutes
- Scale Down: Remove 1 instance when CPU < 30% for 10 minutes
```

### 3. Cost Optimization

**Right-Sizing**
- Start small, scale up as needed
- Monitor actual usage
- Avoid over-provisioning

**Reserved Instances**
- 1-year: ~40% savings
- 3-year: ~60% savings
- Good for steady workloads

**Spot Instances**
- 70-90% savings
- Can be terminated with 2-minute notice
- Perfect for training, batch processing

**Cost Optimization Hierarchy:**
```
1. Spot Instances (Training, Batch)       90% savings
2. Reserved Instances (Production)        60% savings
3. Auto-Scaling (Variable load)           40% savings
4. Right-Sizing (Match actual needs)      30% savings
5. Storage Lifecycle (Archive old data)   50% savings
6. Data Transfer Optimization             20% savings
```

### 4. Security

**Defense in Depth:**
```
┌────────────────────────────────────────────────┐
│ Layer 1: Network (VPC, Subnets, Firewalls)    │
│ Layer 2: Access (IAM, RBAC, MFA)              │
│ Layer 3: Data (Encryption at rest & transit)  │
│ Layer 4: Application (Input validation)       │
│ Layer 5: Monitoring (Logs, alerts, audit)     │
└────────────────────────────────────────────────┘
```

**Best Practices:**
- Principle of least privilege
- Encrypt data at rest and in transit
- Network isolation (private subnets)
- Regular security audits
- Automated patching

### 5. Observability

**Three Pillars:**
1. **Metrics**: Quantitative measurements (CPU, latency, errors)
2. **Logs**: Event records for debugging
3. **Traces**: Request flow through system

**Monitoring Stack:**
```
Application ──► Metrics   ──► Prometheus ──► Grafana
            │
            ├─► Logs      ──► ELK Stack  ──► Kibana
            │
            └─► Traces    ──► Jaeger     ──► UI
```

## Architecting for ML Workloads

### Training vs Inference Architecture

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Compute** | GPU-heavy, spot instances | CPU, auto-scaling |
| **Storage** | High-throughput (EBS/Persistent SSD) | Low-latency (caching) |
| **Network** | Internal only | Public-facing, CDN |
| **Cost Model** | Temporary (hours/days) | Continuous (24/7) |
| **Optimization** | Throughput | Latency |
| **Scaling** | Vertical (bigger GPU) | Horizontal (more instances) |

### Data Flow Architecture

```
Data Ingestion → Preprocessing → Training → Evaluation → Deployment → Monitoring
       ↓              ↓              ↓           ↓            ↓           ↓
   S3/GCS        Spark/EMR      GPU VMs     MLflow      K8s/ECS    Prometheus
   Kinesis       Airflow        Spot         W&B         ECS         CloudWatch
   Kafka         Glue           Reserved     TensorB     Fargate     Grafana
```

## Practical Exercise: Design an ML Architecture

### Scenario
Design a cloud architecture for an image classification service:

**Requirements:**
- 1 million predictions per day
- 99.9% availability
- < 500ms latency (p95)
- Global user base
- $500/month budget
- Support model updates without downtime

### Your Task

1. **Choose compute resources:**
   - How many instances?
   - What instance types?
   - Auto-scaling strategy?

2. **Design storage:**
   - Where to store models?
   - How to version them?
   - Caching strategy?

3. **Plan networking:**
   - Load balancer configuration?
   - CDN usage?
   - Multi-region deployment?

4. **Estimate costs:**
   - Compute costs
   - Storage costs
   - Network costs
   - Total monthly cost

5. **Plan for high availability:**
   - Multi-AZ deployment?
   - Health checks?
   - Failover strategy?

### Sample Solution

**Architecture:**
```
Global Users
     │
     ↓
┌────────────┐
│ CloudFront │ (CDN, caching)
│   (CDN)    │
└─────┬──────┘
      │
      ↓
┌─────────────┐
│ Application │ (SSL termination, routing)
│Load Balancer│
└─────┬───────┘
      │
   ┌──┴──────────┐
   │             │
┌──┴───┐    ┌───┴──┐
│ AZ-1 │    │ AZ-2 │
│ 2x   │    │ 2x   │
│ t3.  │    │ t3.  │
│medium│    │medium│
└──┬───┘    └───┬──┘
   │            │
   └─────┬──────┘
         │
    ┌────┴────┐
    │ S3/GCS  │ (Model storage)
    │ (Models)│
    └─────────┘
```

**Cost Breakdown:**
```
Load Balancer:           $18/month
CloudFront (CDN):        $50/month (1M requests)
Compute (4x t3.medium):  $120/month
S3 Storage (10GB):       $0.23/month
Data Transfer:           $80/month
Monitoring:              $30/month
────────────────────────────────────
Total:                   ~$298/month (Under budget!)
```

**Scaling Strategy:**
```
Normal Load (0-6am):    2 instances
Peak Load (12-8pm):     4-6 instances
High Load (events):     8-10 instances
```

## Key Architecture Decisions

### 1. Region Selection

**Factors:**
- User location (latency)
- Data residency requirements
- Service availability
- Cost (varies by region)

**Example:**
```
US Users:      us-east-1 (Virginia) or us-west-2 (Oregon)
EU Users:      eu-west-1 (Ireland) or eu-central-1 (Frankfurt)
Asia Users:    ap-southeast-1 (Singapore) or ap-northeast-1 (Tokyo)
```

### 2. Instance Selection

**CPU-Based (Model Serving):**
- Small model: t3.medium (2 vCPU, 4GB)
- Medium model: c5.xlarge (4 vCPU, 8GB)
- Large model: c5.4xlarge (16 vCPU, 32GB)

**GPU-Based (Training):**
- Small model: g4dn.xlarge (1x T4, 4 vCPU)
- Medium model: p3.2xlarge (1x V100, 8 vCPU)
- Large model: p3.8xlarge (4x V100, 32 vCPU)

### 3. Storage Strategy

**Hot Data** (accessed frequently):
- Standard S3/GCS
- SSD-backed block storage
- In-memory caching (Redis)

**Warm Data** (accessed occasionally):
- Infrequent Access storage class
- Standard block storage

**Cold Data** (archived):
- Glacier/Archive storage
- Lifecycle policies

## Common Architecture Patterns

### Pattern: Lambda Architecture

```
Real-time Layer:    Stream processing → Serving Layer
                    (Kafka/Kinesis)
                           │
Batch Layer:        ─────────► Model Training
                    Historical Data
```

Use for: Real-time predictions with periodic retraining

### Pattern: Microservices

```
API Gateway
    │
    ├─► Preprocessing Service
    ├─► Model Service A (v1)
    ├─► Model Service B (v2)
    └─► Post-processing Service
```

Use for: Multiple models, independent scaling

### Pattern: Monolith

```
Single Application
├─ API endpoints
├─ Model inference
├─ Pre/post processing
└─ Monitoring
```

Use for: Simple use cases, getting started

## Key Takeaways

1. **Three pillars**: Compute, Storage, Network
2. **Training ≠ Inference**: Different architectures for different workloads
3. **High Availability**: Multi-AZ, load balancing, health checks
4. **Cost Optimization**: Spot instances, auto-scaling, right-sizing
5. **Security**: Defense in depth, principle of least privilege
6. **Observability**: Metrics, logs, traces
7. **Design for failure**: Assume components will fail
8. **Start simple**: Add complexity as needed

## Self-Check Questions

1. What are the three pillars of cloud architecture?
2. When would you use object storage vs block storage?
3. What's the difference between vertical and horizontal scaling?
4. Why use spot instances for training but not serving?
5. What are the three pillars of observability?
6. How does multi-AZ deployment improve availability?
7. What's the cost difference between reserved and on-demand instances?

## Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Google Cloud Architecture Center](https://cloud.google.com/architecture)
- [Azure Architecture Center](https://docs.microsoft.com/azure/architecture/)
- [Martin Fowler's Architecture Patterns](https://martinfowler.com/)
- [Cloud Architecture Patterns (Book)](https://www.oreilly.com/library/view/cloud-architecture-patterns/9781449357979/)

---

**Next Lesson:** [02-aws-ml-infrastructure.md](./02-aws-ml-infrastructure.md) - Deep dive into AWS
