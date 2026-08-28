# Exercise 02: Cloud ML Infrastructure Deployment

## Learning Objectives

By completing this exercise, you will:
- Deploy identical ML infrastructure across AWS, GCP, and Azure
- Use infrastructure as code (Terraform) for multi-cloud deployments
- Implement cloud-native ML services
- Compare performance and costs across clouds
- Build portable ML infrastructure

## Overview

Modern ML teams often need to deploy across multiple clouds for redundancy, cost optimization, or customer requirements. This exercise builds a complete ML infrastructure stack that can be deployed identically to AWS, GCP, and Azure using Terraform.

## Prerequisites

- Terraform 1.5+
- Active cloud accounts (AWS, GCP, Azure)
- Understanding of cloud services (compute, storage, networking)
- Basic ML model serving concepts
- Python 3.11+

## Problem Statement

Build infrastructure-as-code that deploys a complete ML serving stack to all three major clouds:

1. **Kubernetes cluster** with GPU support
2. **Object storage** for models and data
3. **Managed database** (PostgreSQL) for metadata
4. **Managed caching** (Redis) for predictions
5. **Load balancer** with SSL termination
6. **Monitoring** (Prometheus + Grafana)
7. **Networking** (VPC, subnets, firewall rules)

All deployed identically across AWS, GCP, and Azure.

## Requirements

### Functional Requirements

#### FR1: Kubernetes Cluster
- **AWS**: EKS (Elastic Kubernetes Service)
- **GCP**: GKE (Google Kubernetes Engine)
- **Azure**: AKS (Azure Kubernetes Service)

Features:
- 3-5 worker nodes
- GPU node pool (optional)
- Autoscaling enabled
- Network policies
- RBAC configured

#### FR2: Object Storage
- **AWS**: S3 bucket
- **GCP**: Cloud Storage bucket
- **Azure**: Blob Storage container

Features:
- Versioning enabled
- Lifecycle policies
- Encryption at rest
- Access logging

#### FR3: Managed Database
- **AWS**: RDS PostgreSQL
- **GCP**: Cloud SQL PostgreSQL
- **Azure**: Azure Database for PostgreSQL

Features:
- Multi-AZ/HA configuration
- Automated backups
- Encryption
- Read replicas (optional)

#### FR4: Managed Cache
- **AWS**: ElastiCache Redis
- **GCP**: Memorystore Redis
- **Azure**: Azure Cache for Redis

Features:
- HA configuration
- Persistence enabled
- Encryption in transit

#### FR5: Networking
- **AWS**: VPC with public/private subnets
- **GCP**: VPC with subnets
- **Azure**: Virtual Network with subnets

Features:
- NAT gateway for private subnets
- Security groups/firewall rules
- Network peering (optional)

#### FR6: Monitoring
- Prometheus for metrics
- Grafana for dashboards
- Cloud-native monitoring integration
- Alerting configured

### Non-Functional Requirements

#### NFR1: Portability
- Same Terraform modules work across clouds
- Cloud-specific configurations externalized
- Easy to switch clouds

#### NFR2: Security
- Principle of least privilege
- Secrets managed securely
- Network isolation
- Encryption everywhere

#### NFR3: Cost Optimization
- Right-sized resources
- Auto-scaling configured
- Spot/preemptible instances where appropriate
- Budget alerts

## Implementation Tasks

### Task 1: Terraform Project Structure (3-4 hours)

Set up multi-cloud Terraform project:

```hcl
# Project structure
terraform/
├── modules/
│   ├── kubernetes/
│   │   ├── aws/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── gcp/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   └── azure/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       └── outputs.tf
│   ├── storage/
│   ├── database/
│   ├── cache/
│   └── networking/
├── environments/
│   ├── dev/
│   │   ├── aws/
│   │   ├── gcp/
│   │   └── azure/
│   └── prod/
│       ├── aws/
│       ├── gcp/
│       └── azure/
└── global/
    ├── variables.tf
    └── versions.tf
```

```hcl
# global/versions.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}
```

```hcl
# global/variables.tf

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ml-infrastructure"
}

variable "region" {
  description = "Primary deployment region"
  type        = string
}

variable "enable_gpu" {
  description = "Enable GPU node pools"
  type        = bool
  default     = false
}

variable "cluster_size" {
  description = "Kubernetes cluster size"
  type        = string
  default     = "small"
  validation {
    condition     = contains(["small", "medium", "large"], var.cluster_size)
    error_message = "Cluster size must be small, medium, or large"
  }
}

locals {
  # Cluster sizing
  cluster_configs = {
    small = {
      min_nodes     = 2
      max_nodes     = 5
      instance_type = "medium"
    }
    medium = {
      min_nodes     = 3
      max_nodes     = 10
      instance_type = "large"
    }
    large = {
      min_nodes     = 5
      max_nodes     = 20
      instance_type = "xlarge"
    }
  }

  # Resource tags
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    CreatedDate = timestamp()
  }
}
```

**Acceptance Criteria**:
- [ ] Clean module structure
- [ ] Reusable across clouds
- [ ] Environment-specific configs
- [ ] Common variables defined

---

### Task 2: AWS EKS Module (5-6 hours)

```hcl
# modules/kubernetes/aws/main.tf

# TODO: Implement AWS EKS cluster

# VPC for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  enable_vpn_gateway = false
  single_nat_gateway = var.environment == "dev" ? true : false

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Kubernetes tags for subnet discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = var.tags
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint access
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # OIDC Provider for IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Node groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name           = "general-${var.environment}"
      instance_types = [var.instance_type]

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.min_nodes

      labels = {
        workload = "general"
      }

      # Use spot instances for dev
      capacity_type = var.environment == "dev" ? "SPOT" : "ON_DEMAND"

      # Enable autoscaling
      enable_monitoring = true

      tags = merge(
        var.tags,
        {
          NodeGroup = "general"
        }
      )
    }

    # GPU nodes (if enabled)
    gpu = var.enable_gpu ? {
      name           = "gpu-${var.environment}"
      instance_types = ["g4dn.xlarge"]

      min_size     = 0
      max_size     = 3
      desired_size = 0

      labels = {
        workload    = "gpu"
        nvidia.com/gpu = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = merge(
        var.tags,
        {
          NodeGroup = "gpu"
        }
      )
    } : {}
  }

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  tags = var.tags
}

# Install AWS Load Balancer Controller
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.6.0"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "true"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  depends_on = [module.eks]
}

# Install Cluster Autoscaler
resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  version    = "9.29.0"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.region
  }

  depends_on = [module.eks]
}
```

```hcl
# modules/kubernetes/aws/variables.tf

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "instance_type" {
  description = "EC2 instance type for nodes"
  type        = string
  default     = "t3.medium"
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 5
}

variable "enable_gpu" {
  description = "Enable GPU node group"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}
```

```hcl
# modules/kubernetes/aws/outputs.tf

output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Cluster CA certificate"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
```

**Acceptance Criteria**:
- [ ] EKS cluster deploys successfully
- [ ] VPC configured with public/private subnets
- [ ] Node groups with autoscaling
- [ ] GPU nodes (optional)
- [ ] AWS Load Balancer Controller installed
- [ ] Cluster Autoscaler configured

---

### Task 3: GCP GKE Module (5-6 hours)

```hcl
# modules/kubernetes/gcp/main.tf

# TODO: Implement GCP GKE cluster

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.project_name}-${var.environment}-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.project_name}-${var.environment}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  project       = var.project_id

  # Enable private IP Google access
  private_ip_google_access = true

  # Secondary IP ranges for pods and services
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }
}

# Cloud NAT for private nodes
resource "google_compute_router" "router" {
  name    = "${var.project_name}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.vpc.id
  project = var.project_id
}

resource "google_compute_router_nat" "nat" {
  name                               = "${var.project_name}-${var.environment}-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  project                            = var.project_id
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = "${var.project_name}-${var.environment}"
  location = var.region
  project  = var.project_id

  # Regional cluster for HA
  node_locations = var.node_zones

  # We can't create a cluster with no node pool
  # So we create smallest possible default pool and immediately delete it
  remove_default_node_pool = true
  initial_node_count       = 1

  # Kubernetes version
  min_master_version = var.kubernetes_version

  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # IP allocation for pods and services
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_cidr
  }

  # Master authorized networks
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Cluster addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
    gcp_filestore_csi_driver_config {
      enabled = true
    }
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Monitoring and logging (Cloud Operations)
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  # Cluster autoscaling
  cluster_autoscaling {
    enabled = true
    resource_limits {
      resource_type = "cpu"
      minimum       = var.min_nodes * 2
      maximum       = var.max_nodes * 4
    }
    resource_limits {
      resource_type = "memory"
      minimum       = var.min_nodes * 4
      maximum       = var.max_nodes * 16
    }
  }
}

# General purpose node pool
resource "google_container_node_pool" "general" {
  name       = "general-${var.environment}"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id
  node_count = var.min_nodes

  # Autoscaling
  autoscaling {
    min_node_count = var.min_nodes
    max_node_count = var.max_nodes
  }

  # Node configuration
  node_config {
    machine_type = var.machine_type
    disk_size_gb = 100
    disk_type    = "pd-standard"

    # Use preemptible VMs for dev
    preemptible = var.environment == "dev" ? true : false

    # OAuth scopes
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      workload = "general"
    }

    tags = ["gke-node", "${var.project_name}-${var.environment}"]

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  # Node pool management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# GPU node pool (if enabled)
resource "google_container_node_pool" "gpu" {
  count = var.enable_gpu ? 1 : 0

  name       = "gpu-${var.environment}"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id
  node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  node_config {
    machine_type = "n1-standard-4"

    # GPU accelerator
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    disk_size_gb = 100
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      workload = "gpu"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    tags = ["gke-gpu-node", "${var.project_name}-${var.environment}"]

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
```

**Acceptance Criteria**:
- [ ] GKE cluster deploys successfully
- [ ] VPC with private cluster
- [ ] Node pools with autoscaling
- [ ] GPU nodes (optional)
- [ ] Workload Identity enabled
- [ ] Cloud Operations (logging/monitoring)

---

### Task 4: Azure AKS Module (5-6 hours)

```hcl
# modules/kubernetes/azure/main.tf

# TODO: Implement Azure AKS cluster

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.region
  tags     = var.tags
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

# Subnet for AKS
resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = [var.aks_subnet_cidr]
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.project_name}-${var.environment}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "${var.project_name}-${var.environment}"

  kubernetes_version = var.kubernetes_version

  # Default node pool
  default_node_pool {
    name                = "general"
    node_count          = var.min_nodes
    vm_size             = var.vm_size
    vnet_subnet_id      = azurerm_subnet.aks.id
    enable_auto_scaling = true
    min_count           = var.min_nodes
    max_count           = var.max_nodes
    os_disk_size_gb     = 100

    # Use spot instances for dev
    priority        = var.environment == "dev" ? "Spot" : "Regular"
    eviction_policy = var.environment == "dev" ? "Delete" : null
    spot_max_price  = var.environment == "dev" ? -1 : null

    node_labels = {
      workload = "general"
    }

    tags = var.tags
  }

  # Managed Identity
  identity {
    type = "SystemAssigned"
  }

  # Network profile
  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
    outbound_type     = "loadBalancer"
  }

  # Azure Active Directory integration
  azure_active_directory_role_based_access_control {
    managed                = true
    admin_group_object_ids = var.admin_group_ids
  }

  # Monitoring
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
  }

  # Auto-scaler profile
  auto_scaler_profile {
    balance_similar_node_groups      = true
    expander                          = "least-waste"
    max_graceful_termination_sec      = 600
    max_node_provision_time           = "15m"
    scale_down_delay_after_add        = "10m"
    scale_down_unneeded               = "10m"
    scale_down_unready                = "20m"
    scale_down_utilization_threshold  = 0.5
    scan_interval                     = "10s"
    skip_nodes_with_local_storage     = false
    skip_nodes_with_system_pods       = true
  }

  tags = var.tags
}

# GPU Node Pool (if enabled)
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  count = var.enable_gpu ? 1 : 0

  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = "Standard_NC6s_v3"  # Tesla V100
  node_count            = 0

  enable_auto_scaling = true
  min_count           = 0
  max_count           = 3

  os_disk_size_gb = 100
  vnet_subnet_id  = azurerm_subnet.aks.id

  node_labels = {
    workload = "gpu"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = var.tags
}

# Log Analytics Workspace for monitoring
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.project_name}-${var.environment}-logs"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags
}
```

**Acceptance Criteria**:
- [ ] AKS cluster deploys successfully
- [ ] VNet with subnet
- [ ] Node pools with autoscaling
- [ ] GPU nodes (optional)
- [ ] Azure AD RBAC
- [ ] Log Analytics integration

---

### Task 5: Storage, Database, Cache Modules (6-7 hours)

Create modules for storage, database, and cache for each cloud (simplified example):

```hcl
# modules/storage/aws/main.tf

resource "aws_s3_bucket" "ml_data" {
  bucket = "${var.project_name}-${var.environment}-ml-data"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

```hcl
# modules/database/aws/main.tf

resource "aws_db_instance" "metadata" {
  identifier = "${var.project_name}-${var.environment}"

  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.instance_class

  allocated_storage     = var.storage_size
  max_allocated_storage = var.storage_size * 2

  db_name  = "mlmetadata"
  username = var.db_username
  password = var.db_password

  # Multi-AZ for prod
  multi_az               = var.environment == "prod"
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.db.name

  # Backups
  backup_retention_period = var.environment == "prod" ? 7 : 1
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  # Encryption
  storage_encrypted = true

  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  monitoring_interval             = 60
  monitoring_role_arn             = aws_iam_role.rds_monitoring.arn

  tags = var.tags
}
```

Similar modules for GCP and Azure...

**Acceptance Criteria**:
- [ ] Storage buckets/containers created
- [ ] Managed databases deployed
- [ ] Redis cache instances deployed
- [ ] All encrypted and highly available

---

### Task 6: Deployment Script and Testing (4-5 hours)

```python
# deploy.py

import click
import subprocess
import json
from pathlib import Path
from typing import Dict, List

class MultiCloudDeployer:
    """Deploy ML infrastructure across clouds"""

    def __init__(self, environment: str):
        self.environment = environment
        self.terraform_dir = Path("terraform")

    def deploy(
        self,
        clouds: List[str],
        auto_approve: bool = False
    ) -> Dict[str, bool]:
        """
        TODO: Deploy to selected clouds

        ```python
        results = {}

        for cloud in clouds:
            click.echo(f"Deploying to {cloud}...")

            # Initialize Terraform
            self._terraform_init(cloud)

            # Plan
            self._terraform_plan(cloud)

            # Apply
            if auto_approve or click.confirm(f"Apply changes to {cloud}?"):
                success = self._terraform_apply(cloud, auto_approve)
                results[cloud] = success
            else:
                results[cloud] = False

        return results
        ```
        """
        pass

    def destroy(
        self,
        clouds: List[str],
        auto_approve: bool = False
    ) -> Dict[str, bool]:
        """TODO: Destroy infrastructure"""
        pass

    def output(self, cloud: str) -> Dict:
        """TODO: Get Terraform outputs"""
        pass

    def _terraform_init(self, cloud: str) -> None:
        """TODO: Run terraform init"""
        pass

    def _terraform_plan(self, cloud: str) -> None:
        """TODO: Run terraform plan"""
        pass

    def _terraform_apply(self, cloud: str, auto_approve: bool) -> bool:
        """TODO: Run terraform apply"""
        pass

@click.group()
def cli():
    """Multi-cloud ML infrastructure deployment"""
    pass

@cli.command()
@click.option('--clouds', '-c', multiple=True, type=click.Choice(['aws', 'gcp', 'azure']))
@click.option('--environment', '-e', default='dev')
@click.option('--auto-approve', is_flag=True)
def deploy(clouds, environment, auto_approve):
    """Deploy infrastructure to clouds"""
    deployer = MultiCloudDeployer(environment)
    results = deployer.deploy(list(clouds) or ['aws', 'gcp', 'azure'], auto_approve)

    click.echo("\nDeployment Results:")
    for cloud, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        click.echo(f"{cloud}: {status}")

if __name__ == "__main__":
    cli()
```

**Acceptance Criteria**:
- [ ] Deployment script automates Terraform
- [ ] Deploys to multiple clouds
- [ ] Validates deployments
- [ ] Clean error handling

---

## Deliverables

1. **Terraform Modules**
   - Kubernetes (AWS/GCP/Azure)
   - Storage (AWS/GCP/Azure)
   - Database (AWS/GCP/Azure)
   - Cache (AWS/GCP/Azure)
   - Networking (AWS/GCP/Azure)

2. **Deployment Tools**
   - Python deployment script
   - Configuration files
   - Documentation

3. **Tests**
   - Terraform validation tests
   - Integration tests

4. **Documentation**
   - Setup guide
   - Architecture diagrams
   - Cost estimates

---

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| **Completeness** | 30% |
| **Code Quality** | 25% |
| **Security** | 20% |
| **Documentation** | 15% |
| **Testing** | 10% |

**Passing**: 70%+
**Excellence**: 90%+

---

## Estimated Time

- Task 1: 3-4 hours
- Task 2: 5-6 hours
- Task 3: 5-6 hours
- Task 4: 5-6 hours
- Task 5: 6-7 hours
- Task 6: 4-5 hours
- Testing: 4-5 hours
- Documentation: 3-4 hours

**Total**: 35-43 hours

---

**This exercise teaches production multi-cloud infrastructure deployment with IaC best practices.**
