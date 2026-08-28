# Exercise 02: Multi-Cloud ML Infrastructure with Pulumi

**Estimated Time**: 28-36 hours

## Business Context

Your company has committed to a **multi-cloud strategy** to avoid vendor lock-in and leverage best-of-breed cloud services:

- **AWS**: Strong in general compute, storage (S3), and mature ML services
- **GCP**: Best-in-class TPUs for ML training, BigQuery for analytics
- **Azure**: Enterprise integration, hybrid cloud capabilities

**Current Challenge**: Managing infrastructure across multiple clouds with Terraform HCL is complex:
- Different provider syntax and patterns
- Hard to share logic between providers (copy-paste code)
- Limited abstraction capabilities
- No type safety or IDE support

**The Solution**: The CTO wants to explore **Pulumi** - Infrastructure as Software using real programming languages (Python, TypeScript, Go).

**Pulumi Benefits**:
1. **Real programming languages**: Use Python with all its libraries, logic, and tooling
2. **Type safety**: Catch errors before deployment
3. **Better abstractions**: Create reusable components with classes and functions
4. **Multi-cloud patterns**: Share logic across AWS, GCP, Azure
5. **Testing**: Unit test infrastructure code like application code

**Your Mission**: Build a **multi-cloud ML training platform** that:
- Trains models on **GCP TPUs** (best price/performance for LLMs)
- Stores data in **AWS S3** (lowest storage costs)
- Serves models on **AWS EKS** (existing Kubernetes expertise)
- Monitors with **Azure Monitor** (enterprise requirement)

All managed through **Pulumi Python** code.

## Learning Objectives

After completing this exercise, you will be able to:

1. Build infrastructure with Pulumi using Python
2. Manage multi-cloud deployments (AWS, GCP, Azure)
3. Create reusable infrastructure components with Python classes
4. Implement complex logic with conditionals, loops, and functions
5. Test infrastructure code with unit tests
6. Integrate with existing Python ML pipelines
7. Compare Pulumi vs Terraform trade-offs

## Prerequisites

- Module 09 Exercise 01 (Terraform) - for comparison
- Python programming (intermediate level)
- AWS, GCP, and Azure accounts (trial accounts sufficient)
- Pulumi CLI installed
- Cloud provider CLIs (aws, gcloud, az)

## Problem Statement

Build a **Multi-Cloud ML Training Platform** using Pulumi that:

1. **Data Layer** (AWS):
   - S3 buckets for datasets and models
   - DynamoDB for metadata
   - CloudFront CDN for model distribution

2. **Training Layer** (GCP):
   - TPU VMs for large model training
   - GCS buckets for training artifacts
   - Vertex AI for experiment tracking

3. **Serving Layer** (AWS):
   - EKS cluster for model inference
   - Application Load Balancer
   - Auto-scaling based on traffic

4. **Monitoring Layer** (Azure):
   - Azure Monitor for unified observability
   - Log Analytics workspace
   - Application Insights

5. **Cross-Cloud Connectivity**:
   - VPC peering / VPN connections
   - Shared secrets management
   - Unified IAM/RBAC

### Success Metrics

- Infrastructure deployable to 3 clouds with single command
- Code reuse >60% across cloud providers
- Type safety catches 100% of configuration errors pre-deployment
- Infrastructure code testable with pytest
- Deployment time <20 minutes for complete stack
- Cost optimization: Use cheapest cloud for each workload

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│         Multi-Cloud ML Platform (Pulumi + Python)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AWS (Data + Serving)                   │  │
│  │                                                           │  │
│  │  ┌─────────────────┐         ┌─────────────────┐        │  │
│  │  │  S3 Buckets     │         │  EKS Cluster    │        │  │
│  │  │  - Datasets     │────────▶│  - Inference    │        │  │
│  │  │  - Models       │         │  - Autoscaling  │        │  │
│  │  │  - Artifacts    │         │  - ALB          │        │  │
│  │  └─────────────────┘         └─────────────────┘        │  │
│  │          │                            │                   │  │
│  │          │                            │                   │  │
│  │          ▼                            ▼                   │  │
│  │  ┌─────────────────┐         ┌─────────────────┐        │  │
│  │  │  DynamoDB       │         │  CloudFront CDN │        │  │
│  │  │  - Metadata     │         │  - Model Dist.  │        │  │
│  │  └─────────────────┘         └─────────────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               │ Cross-Cloud VPN                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    GCP (Training)                         │  │
│  │                                                           │  │
│  │  ┌─────────────────┐         ┌─────────────────┐        │  │
│  │  │  TPU VMs        │────────▶│  GCS Buckets    │        │  │
│  │  │  - v4-8 (8 chip)│         │  - Checkpoints  │        │  │
│  │  │  - Preemptible  │         │  - Logs         │        │  │
│  │  └─────────────────┘         └─────────────────┘        │  │
│  │          │                                                │  │
│  │          ▼                                                │  │
│  │  ┌─────────────────┐                                     │  │
│  │  │  Vertex AI      │                                     │  │
│  │  │  - Experiments  │                                     │  │
│  │  │  - Metadata     │                                     │  │
│  │  └─────────────────┘                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               │ Metrics & Logs                  │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Azure (Monitoring)                        │  │
│  │                                                           │  │
│  │  ┌─────────────────┐         ┌─────────────────┐        │  │
│  │  │  Azure Monitor  │◀────────│  Log Analytics  │        │  │
│  │  │  - Dashboards   │         │  - Query Engine │        │  │
│  │  │  - Alerts       │         └─────────────────┘        │  │
│  │  └─────────────────┘                                     │  │
│  │          ▲                                                │  │
│  │          │                                                │  │
│  │  ┌───────┴──────────┐                                    │  │
│  │  │ App Insights     │                                    │  │
│  │  │ - Distributed    │                                    │  │
│  │  │   tracing        │                                    │  │
│  │  └──────────────────┘                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Pulumi Project Setup (5-7 hours)

Set up Pulumi project with Python and configure providers.

#### 1.1 Initialize Pulumi Project

```bash
# Create project directory
mkdir pulumi-multicloud-ml
cd pulumi-multicloud-ml

# Initialize Pulumi project with Python
pulumi new python --name multicloud-ml-platform

# Install cloud provider SDKs
pip install pulumi-aws pulumi-gcp pulumi-azure pulumi-kubernetes
```

#### 1.2 Project Structure

Create directory structure:

```
pulumi-multicloud-ml/
├── Pulumi.yaml              # Pulumi project configuration
├── Pulumi.dev.yaml          # Dev stack configuration
├── Pulumi.prod.yaml         # Production stack configuration
├── requirements.txt         # Python dependencies
├── __main__.py              # Entry point
│
├── components/              # Reusable components
│   ├── __init__.py
│   ├── aws/
│   │   ├── __init__.py
│   │   ├── data_storage.py      # S3, DynamoDB
│   │   ├── ml_serving.py        # EKS cluster
│   │   └── cdn.py               # CloudFront
│   ├── gcp/
│   │   ├── __init__.py
│   │   ├── tpu_training.py      # TPU VMs
│   │   └── vertex_ai.py         # Vertex AI
│   ├── azure/
│   │   ├── __init__.py
│   │   └── monitoring.py        # Azure Monitor
│   └── shared/
│       ├── __init__.py
│       ├── networking.py        # Cross-cloud VPN
│       └── secrets.py           # Unified secrets
│
├── config/
│   ├── __init__.py
│   ├── dev.py                   # Dev environment config
│   └── prod.py                  # Production config
│
├── tests/
│   ├── __init__.py
│   ├── test_aws_components.py
│   ├── test_gcp_components.py
│   └── test_integration.py
│
└── scripts/
    ├── deploy.sh
    └── destroy.sh
```

#### 1.3 Main Configuration

Create `__main__.py`:

```python
"""
TODO: Main Pulumi program - multi-cloud ML infrastructure.

This demonstrates Pulumi's power:
- Real Python code (not HCL)
- Type safety with IDE autocomplete
- Reusable components
- Complex logic (loops, conditionals)
"""

import pulumi
import pulumi_aws as aws
import pulumi_gcp as gcp
import pulumi_azure as azure

from components.aws.data_storage import DataStorageStack
from components.aws.ml_serving import MLServingStack
from components.gcp.tpu_training import TPUTrainingStack
from components.azure.monitoring import MonitoringStack
from components.shared.networking import CrossCloudVPN

# Get configuration
config = pulumi.Config()
environment = pulumi.get_stack()
project_name = pulumi.get_project()

# Configuration values (with defaults)
aws_region = config.get("aws_region") or "us-west-2"
gcp_region = config.get("gcp_region") or "us-central1"
azure_region = config.get("azure_region") or "eastus"

enable_tpu_training = config.get_bool("enable_tpu_training") or False
enable_monitoring = config.get_bool("enable_monitoring") or True

# TODO: AWS Data Layer
aws_data = DataStorageStack(
    f"{project_name}-data",
    environment=environment,
    region=aws_region,
    enable_versioning=environment == "prod",  # Versioning only in prod
    enable_cdn=environment == "prod"
)

# TODO: GCP Training Layer (optional, expensive)
if enable_tpu_training:
    gcp_training = TPUTrainingStack(
        f"{project_name}-training",
        environment=environment,
        region=gcp_region,
        tpu_type="v4-8",  # 8-chip TPU pod
        preemptible=environment != "prod"  # Use spot instances in dev
    )

# TODO: AWS Serving Layer
aws_serving = MLServingStack(
    f"{project_name}-serving",
    environment=environment,
    region=aws_region,
    model_bucket=aws_data.models_bucket.id,
    cluster_version="1.28",
    node_instance_types=["t3.xlarge"] if environment == "dev" else ["m5.2xlarge"]
)

# TODO: Azure Monitoring (if enabled)
if enable_monitoring:
    azure_monitoring = MonitoringStack(
        f"{project_name}-monitoring",
        environment=environment,
        region=azure_region
    )

# TODO: Cross-cloud VPN (connect AWS and GCP)
if enable_tpu_training:
    vpn = CrossCloudVPN(
        f"{project_name}-vpn",
        aws_vpc_id=aws_serving.vpc_id,
        gcp_vpc_name=gcp_training.vpc_name,
        aws_region=aws_region,
        gcp_region=gcp_region
    )

# Exports (outputs)
pulumi.export("aws_datasets_bucket", aws_data.datasets_bucket.bucket)
pulumi.export("aws_models_bucket", aws_data.models_bucket.bucket)
pulumi.export("eks_cluster_name", aws_serving.cluster_name)
pulumi.export("eks_cluster_endpoint", aws_serving.cluster_endpoint)

if enable_tpu_training:
    pulumi.export("gcp_tpu_name", gcp_training.tpu_vm_name)
    pulumi.export("gcp_tpu_ip", gcp_training.tpu_internal_ip)

if enable_monitoring:
    pulumi.export("azure_workspace_id", azure_monitoring.workspace_id)
```

Create `Pulumi.yaml`:

```yaml
# TODO: Pulumi project configuration

name: multicloud-ml-platform
description: Multi-cloud ML infrastructure with AWS, GCP, and Azure
runtime: python

# Configuration schema (validates stack configs)
config:
  aws_region:
    type: string
    description: AWS region for data and serving layers
    default: us-west-2

  gcp_region:
    type: string
    description: GCP region for TPU training
    default: us-central1

  azure_region:
    type: string
    description: Azure region for monitoring
    default: eastus

  enable_tpu_training:
    type: boolean
    description: Enable expensive GCP TPU training resources
    default: false

  enable_monitoring:
    type: boolean
    description: Enable Azure Monitor integration
    default: true
```

Create `Pulumi.dev.yaml`:

```yaml
# TODO: Development stack configuration

config:
  multicloud-ml-platform:aws_region: us-west-2
  multicloud-ml-platform:gcp_region: us-central1
  multicloud-ml-platform:azure_region: eastus
  multicloud-ml-platform:enable_tpu_training: false  # Expensive, disable in dev
  multicloud-ml-platform:enable_monitoring: false    # Save costs in dev
```

### Part 2: AWS Components (7-9 hours)

Build reusable AWS components for data storage and ML serving.

#### 2.1 Data Storage Component

Create `components/aws/data_storage.py`:

```python
"""
TODO: AWS data storage component.

Demonstrates Pulumi Python features:
- Component resources (grouping related resources)
- Type hints
- Python classes for encapsulation
"""

import pulumi
import pulumi_aws as aws
from typing import Optional

class DataStorageStack(pulumi.ComponentResource):
    """
    AWS data storage for ML datasets and models.

    Includes:
    - S3 buckets (datasets, models, artifacts)
    - DynamoDB for metadata
    - CloudFront CDN (optional)
    """

    def __init__(
        self,
        name: str,
        environment: str,
        region: str,
        enable_versioning: bool = False,
        enable_cdn: bool = False,
        opts: Optional[pulumi.ResourceOptions] = None
    ):
        super().__init__("custom:aws:DataStorageStack", name, {}, opts)

        # Child resource options (make this component the parent)
        child_opts = pulumi.ResourceOptions(parent=self)

        # TODO: Create S3 bucket for datasets
        self.datasets_bucket = aws.s3.Bucket(
            f"{name}-datasets",
            bucket=f"{name}-datasets-{environment}",
            acl="private",
            versioning=aws.s3.BucketVersioningArgs(
                enabled=enable_versioning
            ),
            server_side_encryption_configuration=aws.s3.BucketServerSideEncryptionConfigurationArgs(
                rule=aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                    apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                        sse_algorithm="AES256"
                    )
                )
            ),
            lifecycle_rules=[
                aws.s3.BucketLifecycleRuleArgs(
                    enabled=True,
                    transitions=[
                        aws.s3.BucketLifecycleRuleTransitionArgs(
                            days=90,
                            storage_class="INTELLIGENT_TIERING"
                        ),
                        aws.s3.BucketLifecycleRuleTransitionArgs(
                            days=365,
                            storage_class="GLACIER"
                        )
                    ]
                )
            ],
            tags={
                "Environment": environment,
                "Purpose": "ML Datasets",
                "ManagedBy": "Pulumi"
            },
            opts=child_opts
        )

        # TODO: Create S3 bucket for trained models
        self.models_bucket = aws.s3.Bucket(
            f"{name}-models",
            bucket=f"{name}-models-{environment}",
            acl="private",
            versioning=aws.s3.BucketVersioningArgs(
                enabled=True  # Always version models
            ),
            server_side_encryption_configuration=aws.s3.BucketServerSideEncryptionConfigurationArgs(
                rule=aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                    apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                        sse_algorithm="AES256"
                    )
                )
            ),
            tags={
                "Environment": environment,
                "Purpose": "ML Models",
                "ManagedBy": "Pulumi"
            },
            opts=child_opts
        )

        # TODO: Create DynamoDB table for metadata
        self.metadata_table = aws.dynamodb.Table(
            f"{name}-metadata",
            name=f"{name}-metadata-{environment}",
            billing_mode="PAY_PER_REQUEST",  # Auto-scaling, no capacity planning
            hash_key="model_id",
            range_key="version",
            attributes=[
                aws.dynamodb.TableAttributeArgs(
                    name="model_id",
                    type="S"  # String
                ),
                aws.dynamodb.TableAttributeArgs(
                    name="version",
                    type="N"  # Number
                )
            ],
            tags={
                "Environment": environment,
                "Purpose": "Model Metadata",
                "ManagedBy": "Pulumi"
            },
            opts=child_opts
        )

        # TODO: Create CloudFront distribution (if enabled)
        if enable_cdn:
            self.cdn = aws.cloudfront.Distribution(
                f"{name}-cdn",
                enabled=True,
                default_cache_behavior=aws.cloudfront.DistributionDefaultCacheBehaviorArgs(
                    target_origin_id=self.models_bucket.arn,
                    viewer_protocol_policy="redirect-to-https",
                    allowed_methods=["GET", "HEAD"],
                    cached_methods=["GET", "HEAD"],
                    forwarded_values=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesArgs(
                        query_string=False,
                        cookies=aws.cloudfront.DistributionDefaultCacheBehaviorForwardedValuesCookiesArgs(
                            forward="none"
                        )
                    ),
                    min_ttl=0,
                    default_ttl=3600,
                    max_ttl=86400
                ),
                origins=[
                    aws.cloudfront.DistributionOriginArgs(
                        origin_id=self.models_bucket.arn,
                        domain_name=self.models_bucket.bucket_regional_domain_name,
                        s3_origin_config=aws.cloudfront.DistributionOriginS3OriginConfigArgs(
                            origin_access_identity=""
                        )
                    )
                ],
                restrictions=aws.cloudfront.DistributionRestrictionsArgs(
                    geo_restriction=aws.cloudfront.DistributionRestrictionsGeoRestrictionArgs(
                        restriction_type="none"
                    )
                ),
                viewer_certificate=aws.cloudfront.DistributionViewerCertificateArgs(
                    cloudfront_default_certificate=True
                ),
                tags={
                    "Environment": environment,
                    "ManagedBy": "Pulumi"
                },
                opts=child_opts
            )

        # Register outputs
        self.register_outputs({
            "datasets_bucket": self.datasets_bucket.id,
            "models_bucket": self.models_bucket.id,
            "metadata_table": self.metadata_table.id
        })
```

#### 2.2 ML Serving Component

Create `components/aws/ml_serving.py`:

```python
"""
TODO: AWS ML serving component (EKS cluster).

Demonstrates:
- Complex resource dependencies
- Using outputs from other resources
- Kubernetes provider configuration
"""

import pulumi
import pulumi_aws as aws
import pulumi_eks as eks
from typing import List, Optional

class MLServingStack(pulumi.ComponentResource):
    """
    EKS cluster for ML model serving.

    Includes:
    - EKS cluster
    - Node groups (CPU and optional GPU)
    - Application Load Balancer
    - Auto-scaling configuration
    """

    def __init__(
        self,
        name: str,
        environment: str,
        region: str,
        model_bucket: pulumi.Output[str],
        cluster_version: str = "1.28",
        node_instance_types: List[str] = None,
        opts: Optional[pulumi.ResourceOptions] = None
    ):
        super().__init__("custom:aws:MLServingStack", name, {}, opts)

        child_opts = pulumi.ResourceOptions(parent=self)

        if node_instance_types is None:
            node_instance_types = ["t3.xlarge"]

        # TODO: Create EKS cluster using eks.Cluster (simplified)
        self.cluster = eks.Cluster(
            f"{name}-eks",
            version=cluster_version,
            instance_type=node_instance_types[0],
            desired_capacity=2 if environment == "prod" else 1,
            min_size=1,
            max_size=5 if environment == "prod" else 2,
            create_oidc_provider=True,  # For IAM roles for service accounts
            tags={
                "Environment": environment,
                "Purpose": "ML Inference",
                "ManagedBy": "Pulumi"
            },
            opts=child_opts
        )

        # TODO: Create IAM role for model access
        # EKS pods need to read from S3 models bucket
        model_access_policy = aws.iam.Policy(
            f"{name}-model-access",
            policy=pulumi.Output.all(model_bucket).apply(
                lambda args: pulumi.Output.json_dumps({
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{args[0]}",
                            f"arn:aws:s3:::{args[0]}/*"
                        ]
                    }]
                })
            ),
            opts=child_opts
        )

        # Store cluster details
        self.cluster_name = self.cluster.eks_cluster.name
        self.cluster_endpoint = self.cluster.eks_cluster.endpoint
        self.vpc_id = self.cluster.eks_cluster.vpc_config.vpc_id

        # Register outputs
        self.register_outputs({
            "cluster_name": self.cluster_name,
            "cluster_endpoint": self.cluster_endpoint,
            "vpc_id": self.vpc_id
        })
```

### Part 3: GCP TPU Training Component (6-8 hours)

Create GCP TPU training infrastructure.

Create `components/gcp/tpu_training.py`:

```python
"""
TODO: GCP TPU training component.

Demonstrates:
- Multi-cloud patterns
- Conditional resource creation
- Integration with Python ML libraries
"""

import pulumi
import pulumi_gcp as gcp
from typing import Optional

class TPUTrainingStack(pulumi.ComponentResource):
    """
    GCP TPU VMs for large model training.

    TPU benefits:
    - 10× faster than GPUs for large models
    - Better cost/performance for transformers
    - Optimized for TensorFlow and JAX
    """

    def __init__(
        self,
        name: str,
        environment: str,
        region: str,
        tpu_type: str = "v4-8",  # 8-chip TPU pod
        preemptible: bool = False,  # Spot instances (70% cheaper)
        opts: Optional[pulumi.ResourceOptions] = None
    ):
        super().__init__("custom:gcp:TPUTrainingStack", name, {}, opts)

        child_opts = pulumi.ResourceOptions(parent=self)

        # TODO: Create VPC for TPU VMs
        self.vpc = gcp.compute.Network(
            f"{name}-vpc",
            auto_create_subnetworks=False,
            opts=child_opts
        )

        # TODO: Create subnet
        self.subnet = gcp.compute.Subnetwork(
            f"{name}-subnet",
            network=self.vpc.id,
            ip_cidr_range="10.1.0.0/24",
            region=region,
            opts=child_opts
        )

        # TODO: Create firewall rules (allow internal + SSH)
        self.firewall_internal = gcp.compute.Firewall(
            f"{name}-allow-internal",
            network=self.vpc.self_link,
            allows=[
                gcp.compute.FirewallAllowArgs(
                    protocol="tcp",
                    ports=["0-65535"]
                ),
                gcp.compute.FirewallAllowArgs(
                    protocol="udp",
                    ports=["0-65535"]
                ),
                gcp.compute.FirewallAllowArgs(
                    protocol="icmp"
                )
            ],
            source_ranges=["10.1.0.0/24"],
            opts=child_opts
        )

        # TODO: Create TPU VM
        self.tpu_vm = gcp.tpu.V2Vm(
            f"{name}-tpu",
            zone=f"{region}-a",
            runtime_version="tpu-vm-tf-2.14.0",  # TensorFlow 2.14
            accelerator_config=gcp.tpu.V2VmAcceleratorConfigArgs(
                type=tpu_type,
                topology="2x2x1" if tpu_type == "v4-8" else "2x2x2"
            ),
            network_config=gcp.tpu.V2VmNetworkConfigArgs(
                network=self.vpc.self_link,
                subnetwork=self.subnet.self_link,
                enable_external_ips=True
            ),
            scheduling_config=gcp.tpu.V2VmSchedulingConfigArgs(
                preemptible=preemptible
            ),
            tags={
                "environment": environment,
                "purpose": "ml-training",
                "managed-by": "pulumi"
            },
            opts=child_opts
        )

        # TODO: Create GCS bucket for training artifacts
        self.training_bucket = gcp.storage.Bucket(
            f"{name}-training",
            location=region,
            uniform_bucket_level_access=True,
            versioning=gcp.storage.BucketVersioningArgs(
                enabled=True
            ),
            lifecycle_rules=[
                gcp.storage.BucketLifecycleRuleArgs(
                    action=gcp.storage.BucketLifecycleRuleActionArgs(
                        type="Delete"
                    ),
                    condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                        age=30  # Delete after 30 days
                    )
                )
            ],
            opts=child_opts
        )

        # Store TPU details
        self.tpu_vm_name = self.tpu_vm.name
        self.tpu_internal_ip = self.tpu_vm.network_endpoints.apply(
            lambda endpoints: endpoints[0].ip_address if endpoints else None
        )
        self.vpc_name = self.vpc.name

        # Register outputs
        self.register_outputs({
            "tpu_vm_name": self.tpu_vm_name,
            "tpu_internal_ip": self.tpu_internal_ip,
            "vpc_name": self.vpc_name,
            "training_bucket": self.training_bucket.name
        })
```

### Part 4: Azure Monitoring Component (4-6 hours)

Create Azure monitoring infrastructure.

Create `components/azure/monitoring.py`:

```python
"""
TODO: Azure monitoring component.

Demonstrates:
- Third cloud provider integration
- Unified observability across clouds
"""

import pulumi
import pulumi_azure as azure
from typing import Optional

class MonitoringStack(pulumi.ComponentResource):
    """
    Azure Monitor for unified observability.

    Collects metrics and logs from:
    - AWS EKS (via Azure Monitor agent)
    - GCP TPU VMs (via custom exporters)
    - All application telemetry
    """

    def __init__(
        self,
        name: str,
        environment: str,
        region: str,
        opts: Optional[pulumi.ResourceOptions] = None
    ):
        super().__init__("custom:azure:MonitoringStack", name, {}, opts)

        child_opts = pulumi.ResourceOptions(parent=self)

        # TODO: Create Resource Group
        self.resource_group = azure.core.ResourceGroup(
            f"{name}-rg",
            location=region,
            tags={
                "environment": environment,
                "purpose": "monitoring",
                "managed-by": "pulumi"
            },
            opts=child_opts
        )

        # TODO: Create Log Analytics Workspace
        self.workspace = azure.operationalinsights.Workspace(
            f"{name}-workspace",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            sku="PerGB2018",  # Pay-per-GB pricing
            retention_in_days=30 if environment == "dev" else 90,
            tags={
                "environment": environment,
                "purpose": "ml-logs"
            },
            opts=child_opts
        )

        # TODO: Create Application Insights
        self.app_insights = azure.appinsights.Insights(
            f"{name}-appinsights",
            resource_group_name=self.resource_group.name,
            location=self.resource_group.location,
            application_type="web",
            workspace_id=self.workspace.id,
            tags={
                "environment": environment,
                "purpose": "ml-telemetry"
            },
            opts=child_opts
        )

        # Store monitoring details
        self.workspace_id = self.workspace.id
        self.instrumentation_key = self.app_insights.instrumentation_key

        # Register outputs
        self.register_outputs({
            "workspace_id": self.workspace_id,
            "instrumentation_key": self.instrumentation_key
        })
```

### Part 5: Testing and Deployment (6-8 hours)

Implement unit tests and deployment automation.

#### 5.1 Unit Tests

Create `tests/test_aws_components.py`:

```python
"""
TODO: Unit tests for AWS components.

Pulumi allows testing infrastructure code with pytest!
"""

import unittest
import pulumi

class TestDataStorage(unittest.TestCase):
    """Test AWS data storage component."""

    @pulumi.runtime.test
    def test_s3_bucket_encryption(self):
        """Verify S3 buckets have encryption enabled."""

        # TODO: Import component and test
        from components.aws.data_storage import DataStorageStack

        def check_encryption(args):
            bucket, encryption = args
            # Verify encryption is enabled
            self.assertEqual(encryption["rule"]["apply_server_side_encryption_by_default"]["sse_algorithm"], "AES256")

        # Create component
        stack = DataStorageStack(
            "test-storage",
            environment="test",
            region="us-west-2",
            enable_versioning=True
        )

        # Test encryption configuration
        pulumi.Output.all(
            stack.datasets_bucket,
            stack.datasets_bucket.server_side_encryption_configuration
        ).apply(check_encryption)

    @pulumi.runtime.test
    def test_versioning_enabled_prod(self):
        """Verify versioning enabled in production."""

        from components.aws.data_storage import DataStorageStack

        stack = DataStorageStack(
            "test-storage-prod",
            environment="prod",
            region="us-west-2",
            enable_versioning=True
        )

        def check_versioning(versioning):
            self.assertTrue(versioning["enabled"])

        stack.models_bucket.versioning.apply(check_versioning)
```

#### 5.2 Deployment Scripts

Create `scripts/deploy.sh`:

```bash
#!/bin/bash
# TODO: Deploy multi-cloud infrastructure

set -e

STACK=${1:-dev}

echo "Deploying to stack: ${STACK}"

# Select stack
pulumi stack select ${STACK} || pulumi stack init ${STACK}

# Preview changes
pulumi preview --stack ${STACK}

# Ask for confirmation
read -p "Deploy these changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Deploy
    pulumi up --stack ${STACK} --yes

    # Show outputs
    pulumi stack output --stack ${STACK}
fi
```

## Acceptance Criteria

### Functional Requirements

- [ ] Multi-cloud infrastructure (AWS + GCP + Azure) deployed via Pulumi
- [ ] All components implemented as reusable Python classes
- [ ] Type safety enforced (type hints, IDE autocomplete)
- [ ] Unit tests for all components with >80% coverage
- [ ] Configuration management for dev/prod stacks
- [ ] Cross-cloud connectivity (VPN between AWS and GCP)

### Performance Requirements

- [ ] Infrastructure deployment completes in <20 minutes
- [ ] Pulumi preview executes in <30 seconds
- [ ] Code reuse >60% across cloud providers

### Code Quality

- [ ] All Python code passes type checking (mypy)
- [ ] Code formatted with black
- [ ] Comprehensive documentation and examples
- [ ] Unit tests with pytest

## Testing Strategy

```bash
# Type checking
mypy components/

# Unit tests
pytest tests/ -v

# Preview changes
pulumi preview

# Deploy to dev
./scripts/deploy.sh dev
```

## Deliverables

1. **Pulumi Code** (Python components and main program)
2. **Tests** (Unit tests with pytest)
3. **Documentation** (Architecture guide, comparison with Terraform)
4. **Deployment Scripts**

## Bonus Challenges

1. **Policy as Code** (+4 hours): Add Pulumi policy packs
2. **Custom Components** (+6 hours): Package components as reusable library
3. **Integration Tests** (+6 hours): Test actual cloud deployments

## Resources

- [Pulumi Documentation](https://www.pulumi.com/docs/)
- [Pulumi AWS](https://www.pulumi.com/registry/packages/aws/)
- [Pulumi GCP](https://www.pulumi.com/registry/packages/gcp/)
- [Pulumi Azure](https://www.pulumi.com/registry/packages/azure/)

## Submission

```bash
git add .
git commit -m "Complete Exercise 02: Pulumi Multi-Cloud ML"
git push origin exercise-02-pulumi-multicloud-ml
```

---

**Estimated Time Breakdown**:
- Part 1 (Setup): 5-7 hours
- Part 2 (AWS Components): 7-9 hours
- Part 3 (GCP Components): 6-8 hours
- Part 4 (Azure Components): 4-6 hours
- Part 5 (Testing & Deployment): 6-8 hours
- **Total**: 28-36 hours
