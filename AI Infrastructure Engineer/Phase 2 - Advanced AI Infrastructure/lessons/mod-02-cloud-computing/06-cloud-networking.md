# Lesson 06: Cloud Networking for ML

**Duration:** 6 hours
**Difficulty:** Intermediate
**Prerequisites:** Lessons 01-05 (Cloud providers and storage)

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Design VPC networks** for ML infrastructure
2. **Configure load balancers** for model serving at scale
3. **Implement CDN** for global model delivery
4. **Secure ML systems** with firewalls and security groups
5. **Set up VPN and hybrid connectivity** for on-premises integration
6. **Optimize network performance** for training and inference
7. **Implement service mesh** for microservices architecture
8. **Monitor and troubleshoot** network issues

---

## Table of Contents

1. [Networking Fundamentals for ML](#networking-fundamentals-for-ml)
2. [Virtual Private Cloud (VPC)](#virtual-private-cloud-vpc)
3. [Load Balancing](#load-balancing)
4. [Content Delivery Network (CDN)](#content-delivery-network-cdn)
5. [Network Security](#network-security)
6. [Hybrid and Multi-Cloud Networking](#hybrid-and-multi-cloud-networking)
7. [Service Mesh](#service-mesh)
8. [Network Performance Optimization](#network-performance-optimization)
9. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
10. [Hands-on Exercise](#hands-on-exercise)

---

## Networking Fundamentals for ML

ML workloads have unique networking requirements compared to traditional applications.

### ML Network Traffic Patterns

```
┌────────────────────────────────────────────────────────────────┐
│                    ML Network Architecture                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Training Phase                                                │
│  ──────────────                                                │
│  Data Storage (S3)  ──→  Training Cluster  ──→  Model Storage │
│     1-10 GB/s             (GPU instances)         100 MB/s    │
│     High bandwidth        Low latency             Medium       │
│                                                                │
│  Inference Phase                                               │
│  ───────────────                                               │
│  User Request  ──→  Load Balancer  ──→  Model Server ──→ DB  │
│     1-10 KB          100K+ requests     <50ms latency   <10ms │
│     Low bandwidth    High availability   Low latency   Cache  │
│                                                                │
│  Distributed Training                                          │
│  ────────────────────                                          │
│  Node 1  ←─────→  Node 2  ←─────→  Node 3  ←─────→  Node 4  │
│    ↓                ↓                ↓                ↓       │
│  All-reduce communication (parameter synchronization)         │
│  Requirement: Low latency (<10ms), High bandwidth (25+ Gbps)  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Network Requirements by Workload

```
┌──────────────────────┬──────────────┬──────────────┬─────────────┐
│ Workload             │ Bandwidth    │ Latency      │ Reliability │
├──────────────────────┼──────────────┼──────────────┼─────────────┤
│ Data Loading         │ Very High    │ Medium       │ Medium      │
│ (Training)           │ 1-10 GB/s    │ 100-500ms    │ 99%         │
│                      │              │              │             │
│ Model Inference      │ Low          │ Very Low     │ Very High   │
│ (Real-time)          │ 1-10 KB      │ <50ms        │ 99.99%      │
│                      │              │              │             │
│ Batch Inference      │ High         │ Medium       │ High        │
│                      │ 100 MB-1GB/s │ <1s          │ 99.9%       │
│                      │              │              │             │
│ Distributed Training │ Very High    │ Very Low     │ High        │
│ (Multi-GPU)          │ 10-100 GB/s  │ <10ms        │ 99.9%       │
│                      │              │              │             │
│ Model Upload/Download│ Medium       │ Low          │ Medium      │
│                      │ 10-100 MB/s  │ <5s          │ 99%         │
└──────────────────────┴──────────────┴──────────────┴─────────────┘
```

---

## Virtual Private Cloud (VPC)

VPC provides isolated network environments for ML infrastructure.

### VPC Design Patterns for ML

#### Pattern 1: Simple VPC (Development)

```
┌─────────────────────────────────────────────────────────┐
│              VPC: 10.0.0.0/16                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Public Subnet: 10.0.1.0/24                             │
│  ├── Bastion Host (SSH gateway)                         │
│  ├── NAT Gateway                                        │
│  └── Load Balancer                                      │
│                                                         │
│  Private Subnet: 10.0.2.0/24                            │
│  ├── ML Training VMs (GPU instances)                    │
│  ├── Jupyter Notebooks                                  │
│  └── Development servers                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Pattern 2: Multi-Tier VPC (Production)

```
┌──────────────────────────────────────────────────────────────┐
│              VPC: 10.0.0.0/16 (Multi-AZ)                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Availability Zone 1           Availability Zone 2          │
│  ────────────────────           ────────────────────        │
│                                                              │
│  Public Subnet A: 10.0.1.0/24   Public Subnet B: 10.0.2.0/24│
│  ├── Load Balancer              ├── Load Balancer           │
│  └── NAT Gateway                └── NAT Gateway             │
│                                                              │
│  App Subnet A: 10.0.11.0/24     App Subnet B: 10.0.12.0/24 │
│  ├── Model Servers (AKS)        ├── Model Servers (AKS)    │
│  └── API Gateway                └── API Gateway            │
│                                                              │
│  Data Subnet A: 10.0.21.0/24    Data Subnet B: 10.0.22.0/24│
│  ├── Redis Cache                ├── Redis Cache            │
│  ├── Database                   ├── Database (replica)     │
│  └── Feature Store              └── Feature Store          │
│                                                              │
│  Training Subnet: 10.0.31.0/24  (Single AZ - cost savings) │
│  ├── GPU Training Instances                                 │
│  ├── Distributed Training Cluster                           │
│  └── Data Processing Pipeline                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Creating VPC for ML (AWS)

```python
import boto3

ec2_client = boto3.client('ec2')

def create_ml_vpc():
    """
    Create VPC for ML infrastructure

    Returns:
        VPC ID and subnet IDs
    """
    # Create VPC
    vpc_response = ec2_client.create_vpc(
        CidrBlock='10.0.0.0/16',
        TagSpecifications=[
            {
                'ResourceType': 'vpc',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ml-vpc'},
                    {'Key': 'Environment', 'Value': 'production'}
                ]
            }
        ]
    )
    vpc_id = vpc_response['Vpc']['VpcId']
    print(f"Created VPC: {vpc_id}")

    # Enable DNS hostnames
    ec2_client.modify_vpc_attribute(
        VpcId=vpc_id,
        EnableDnsHostnames={'Value': True}
    )

    # Create Internet Gateway
    igw_response = ec2_client.create_internet_gateway(
        TagSpecifications=[
            {
                'ResourceType': 'internet-gateway',
                'Tags': [{'Key': 'Name', 'Value': 'ml-igw'}]
            }
        ]
    )
    igw_id = igw_response['InternetGateway']['InternetGatewayId']

    # Attach IGW to VPC
    ec2_client.attach_internet_gateway(
        InternetGatewayId=igw_id,
        VpcId=vpc_id
    )

    # Create Public Subnet (for load balancers, NAT)
    public_subnet_response = ec2_client.create_subnet(
        VpcId=vpc_id,
        CidrBlock='10.0.1.0/24',
        AvailabilityZone='us-east-1a',
        TagSpecifications=[
            {
                'ResourceType': 'subnet',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ml-public-subnet'},
                    {'Key': 'Type', 'Value': 'public'}
                ]
            }
        ]
    )
    public_subnet_id = public_subnet_response['Subnet']['SubnetId']

    # Create Private Subnet (for training instances)
    private_subnet_response = ec2_client.create_subnet(
        VpcId=vpc_id,
        CidrBlock='10.0.2.0/24',
        AvailabilityZone='us-east-1a',
        TagSpecifications=[
            {
                'ResourceType': 'subnet',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ml-training-subnet'},
                    {'Key': 'Type', 'Value': 'private'}
                ]
            }
        ]
    )
    private_subnet_id = private_subnet_response['Subnet']['SubnetId']

    # Create App Subnet (for model servers)
    app_subnet_response = ec2_client.create_subnet(
        VpcId=vpc_id,
        CidrBlock='10.0.11.0/24',
        AvailabilityZone='us-east-1a',
        TagSpecifications=[
            {
                'ResourceType': 'subnet',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ml-app-subnet'},
                    {'Key': 'Type', 'Value': 'private'}
                ]
            }
        ]
    )
    app_subnet_id = app_subnet_response['Subnet']['SubnetId']

    # Create NAT Gateway for private subnets
    # First, allocate Elastic IP
    eip_response = ec2_client.allocate_address(Domain='vpc')
    eip_allocation_id = eip_response['AllocationId']

    nat_response = ec2_client.create_nat_gateway(
        SubnetId=public_subnet_id,
        AllocationId=eip_allocation_id,
        TagSpecifications=[
            {
                'ResourceType': 'natgateway',
                'Tags': [{'Key': 'Name', 'Value': 'ml-nat-gateway'}]
            }
        ]
    )
    nat_gateway_id = nat_response['NatGateway']['NatGatewayId']

    # Wait for NAT Gateway to be available
    waiter = ec2_client.get_waiter('nat_gateway_available')
    waiter.wait(NatGatewayIds=[nat_gateway_id])

    # Create Route Tables
    # Public route table
    public_rt_response = ec2_client.create_route_table(
        VpcId=vpc_id,
        TagSpecifications=[
            {
                'ResourceType': 'route-table',
                'Tags': [{'Key': 'Name', 'Value': 'ml-public-rt'}]
            }
        ]
    )
    public_rt_id = public_rt_response['RouteTable']['RouteTableId']

    # Add route to IGW
    ec2_client.create_route(
        RouteTableId=public_rt_id,
        DestinationCidrBlock='0.0.0.0/0',
        GatewayId=igw_id
    )

    # Associate public subnet with public route table
    ec2_client.associate_route_table(
        SubnetId=public_subnet_id,
        RouteTableId=public_rt_id
    )

    # Private route table
    private_rt_response = ec2_client.create_route_table(
        VpcId=vpc_id,
        TagSpecifications=[
            {
                'ResourceType': 'route-table',
                'Tags': [{'Key': 'Name', 'Value': 'ml-private-rt'}]
            }
        ]
    )
    private_rt_id = private_rt_response['RouteTable']['RouteTableId']

    # Add route to NAT Gateway
    ec2_client.create_route(
        RouteTableId=private_rt_id,
        DestinationCidrBlock='0.0.0.0/0',
        NatGatewayId=nat_gateway_id
    )

    # Associate private subnets with private route table
    ec2_client.associate_route_table(
        SubnetId=private_subnet_id,
        RouteTableId=private_rt_id
    )
    ec2_client.associate_route_table(
        SubnetId=app_subnet_id,
        RouteTableId=private_rt_id
    )

    print(f"VPC setup complete!")
    return {
        'vpc_id': vpc_id,
        'public_subnet_id': public_subnet_id,
        'private_subnet_id': private_subnet_id,
        'app_subnet_id': app_subnet_id,
        'nat_gateway_id': nat_gateway_id
    }

# Create ML VPC
vpc_config = create_ml_vpc()
print(vpc_config)
```

### VPC Peering for Multi-Region Training

```python
def create_vpc_peering(vpc_id_1, vpc_id_2, region_1='us-east-1', region_2='us-west-2'):
    """
    Create VPC peering connection between regions

    Use case: Distributed training across regions
    """
    ec2_client_1 = boto3.client('ec2', region_name=region_1)
    ec2_client_2 = boto3.client('ec2', region_name=region_2)

    # Create peering connection
    response = ec2_client_1.create_vpc_peering_connection(
        VpcId=vpc_id_1,
        PeerVpcId=vpc_id_2,
        PeerRegion=region_2,
        TagSpecifications=[
            {
                'ResourceType': 'vpc-peering-connection',
                'Tags': [{'Key': 'Name', 'Value': 'ml-multi-region-peering'}]
            }
        ]
    )

    peering_id = response['VpcPeeringConnection']['VpcPeeringConnectionId']
    print(f"Created VPC peering: {peering_id}")

    # Accept peering connection in peer region
    ec2_client_2.accept_vpc_peering_connection(
        VpcPeeringConnectionId=peering_id
    )

    print(f"VPC peering accepted")
    return peering_id
```

---

## Load Balancing

Load balancers distribute traffic across multiple model servers for scalability and reliability.

### Load Balancer Types

```
┌──────────────────────┬───────────────────┬─────────────────────┐
│ Type                 │ Use Case          │ Performance         │
├──────────────────────┼───────────────────┼─────────────────────┤
│ Application LB (L7)  │ HTTP/HTTPS APIs   │ 100K requests/sec   │
│ - Path-based routing │ Model serving     │ <100ms latency      │
│ - WebSocket support  │ A/B testing       │                     │
│                      │                   │                     │
│ Network LB (L4)      │ TCP/UDP traffic   │ Millions req/sec    │
│ - Ultra-low latency  │ gRPC inference    │ <10ms latency       │
│ - Static IP          │ Batch processing  │                     │
│                      │                   │                     │
│ Gateway LB           │ Security/Firewall │ High throughput     │
│ - Inline inspection  │ Traffic analysis  │                     │
└──────────────────────┴───────────────────┴─────────────────────┘
```

### Application Load Balancer for ML APIs

```python
import boto3

elbv2_client = boto3.client('elbv2')
ec2_client = boto3.client('ec2')

def create_ml_load_balancer(vpc_id, public_subnet_ids):
    """
    Create Application Load Balancer for ML model serving

    Args:
        vpc_id: VPC ID
        public_subnet_ids: List of public subnet IDs

    Returns:
        Load balancer ARN
    """
    # Create security group for load balancer
    sg_response = ec2_client.create_security_group(
        GroupName='ml-alb-sg',
        Description='Security group for ML ALB',
        VpcId=vpc_id
    )
    sg_id = sg_response['GroupId']

    # Allow HTTP/HTTPS from internet
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 80,
                'ToPort': 80,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            },
            {
                'IpProtocol': 'tcp',
                'FromPort': 443,
                'ToPort': 443,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            }
        ]
    )

    # Create load balancer
    lb_response = elbv2_client.create_load_balancer(
        Name='ml-model-alb',
        Subnets=public_subnet_ids,
        SecurityGroups=[sg_id],
        Scheme='internet-facing',
        Type='application',
        IpAddressType='ipv4',
        Tags=[
            {'Key': 'Name', 'Value': 'ml-model-alb'},
            {'Key': 'Environment', 'Value': 'production'}
        ]
    )

    lb_arn = lb_response['LoadBalancers'][0]['LoadBalancerArn']
    lb_dns = lb_response['LoadBalancers'][0]['DNSName']
    print(f"Created ALB: {lb_dns}")

    # Create target group
    tg_response = elbv2_client.create_target_group(
        Name='ml-model-tg',
        Protocol='HTTP',
        Port=8000,
        VpcId=vpc_id,
        HealthCheckEnabled=True,
        HealthCheckProtocol='HTTP',
        HealthCheckPath='/health',
        HealthCheckIntervalSeconds=30,
        HealthCheckTimeoutSeconds=5,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=3,
        Matcher={'HttpCode': '200'},
        TargetType='instance'
    )

    tg_arn = tg_response['TargetGroups'][0]['TargetGroupArn']
    print(f"Created target group: {tg_arn}")

    # Create listener
    listener_response = elbv2_client.create_listener(
        LoadBalancerArn=lb_arn,
        Protocol='HTTP',
        Port=80,
        DefaultActions=[
            {
                'Type': 'forward',
                'TargetGroupArn': tg_arn
            }
        ]
    )

    listener_arn = listener_response['Listeners'][0]['ListenerArn']
    print(f"Created listener: {listener_arn}")

    return {
        'lb_arn': lb_arn,
        'lb_dns': lb_dns,
        'tg_arn': tg_arn,
        'listener_arn': listener_arn
    }

# Usage
lb_config = create_ml_load_balancer(
    vpc_id='vpc-12345678',
    public_subnet_ids=['subnet-1234', 'subnet-5678']
)
```

### Advanced Routing for A/B Testing

```python
def configure_ab_testing(listener_arn, model_v1_tg_arn, model_v2_tg_arn):
    """
    Configure A/B testing with weighted target groups

    90% traffic to v1 (stable)
    10% traffic to v2 (canary)
    """
    elbv2_client = boto3.client('elbv2')

    # Modify listener to use weighted routing
    elbv2_client.modify_listener(
        ListenerArn=listener_arn,
        DefaultActions=[
            {
                'Type': 'forward',
                'ForwardConfig': {
                    'TargetGroups': [
                        {
                            'TargetGroupArn': model_v1_tg_arn,
                            'Weight': 90
                        },
                        {
                            'TargetGroupArn': model_v2_tg_arn,
                            'Weight': 10
                        }
                    ],
                    'TargetGroupStickinessConfig': {
                        'Enabled': True,
                        'DurationSeconds': 3600  # Sticky for 1 hour
                    }
                }
            }
        ]
    )

    print("Configured A/B testing: 90% v1, 10% v2")

# Gradually shift traffic
def shift_traffic(listener_arn, model_v1_tg_arn, model_v2_tg_arn, v2_percentage):
    """
    Gradually shift traffic to new model version
    """
    elbv2_client = boto3.client('elbv2')

    v1_percentage = 100 - v2_percentage

    elbv2_client.modify_listener(
        ListenerArn=listener_arn,
        DefaultActions=[
            {
                'Type': 'forward',
                'ForwardConfig': {
                    'TargetGroups': [
                        {
                            'TargetGroupArn': model_v1_tg_arn,
                            'Weight': v1_percentage
                        },
                        {
                            'TargetGroupArn': model_v2_tg_arn,
                            'Weight': v2_percentage
                        }
                    ]
                }
            }
        ]
    )

    print(f"Traffic split: {v1_percentage}% v1, {v2_percentage}% v2")

# Canary deployment example
# Day 1: 10% traffic to v2
shift_traffic(listener_arn, tg_v1, tg_v2, v2_percentage=10)

# Day 2: If metrics look good, 50%
shift_traffic(listener_arn, tg_v1, tg_v2, v2_percentage=50)

# Day 3: Full rollout
shift_traffic(listener_arn, tg_v1, tg_v2, v2_percentage=100)
```

---

## Content Delivery Network (CDN)

CDN caches model responses and assets globally for low-latency delivery.

### CDN Architecture for ML

```
┌────────────────────────────────────────────────────────────┐
│                    CDN for ML Serving                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  User (US)         User (Europe)       User (Asia)        │
│     ↓                  ↓                   ↓              │
│  Edge Location     Edge Location     Edge Location        │
│  (California)      (Frankfurt)       (Singapore)          │
│     ↓                  ↓                   ↓              │
│     └──────────────────┴───────────────────┘              │
│                        ↓                                   │
│                  Origin Server                             │
│              (Load Balancer + Model Servers)               │
│                                                            │
│  Benefits:                                                 │
│  - Latency: 200ms → 50ms (75% reduction)                  │
│  - Origin load: Reduced by 80% (caching)                  │
│  - Availability: 99.99% (distributed)                     │
│  - Cost: Reduced bandwidth costs                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### CloudFront Setup for ML APIs

```python
import boto3

cloudfront_client = boto3.client('cloudfront')

def create_ml_cdn(origin_domain, origin_path='/predict'):
    """
    Create CloudFront distribution for ML model serving

    Args:
        origin_domain: Load balancer DNS name
        origin_path: API endpoint path

    Returns:
        CloudFront domain name
    """
    import uuid

    # Create distribution configuration
    config = {
        'CallerReference': str(uuid.uuid4()),
        'Comment': 'CDN for ML model serving',
        'Enabled': True,
        'Origins': {
            'Quantity': 1,
            'Items': [
                {
                    'Id': 'ml-origin',
                    'DomainName': origin_domain,
                    'CustomOriginConfig': {
                        'HTTPPort': 80,
                        'HTTPSPort': 443,
                        'OriginProtocolPolicy': 'http-only',
                        'OriginSslProtocols': {
                            'Quantity': 1,
                            'Items': ['TLSv1.2']
                        },
                        'OriginReadTimeout': 30,
                        'OriginKeepaliveTimeout': 5
                    }
                }
            ]
        },
        'DefaultCacheBehavior': {
            'TargetOriginId': 'ml-origin',
            'ViewerProtocolPolicy': 'redirect-to-https',
            'AllowedMethods': {
                'Quantity': 7,
                'Items': ['GET', 'HEAD', 'OPTIONS', 'PUT', 'POST', 'PATCH', 'DELETE'],
                'CachedMethods': {
                    'Quantity': 2,
                    'Items': ['GET', 'HEAD']
                }
            },
            'CachePolicyId': '4135ea2d-6df8-44a3-9df3-4b5a84be39ad',  # CachingDisabled (for POST requests)
            'Compress': True,
            'MinTTL': 0,
            'DefaultTTL': 0,
            'MaxTTL': 0
        },
        'CacheBehaviors': {
            'Quantity': 1,
            'Items': [
                {
                    'PathPattern': '/predict',
                    'TargetOriginId': 'ml-origin',
                    'ViewerProtocolPolicy': 'https-only',
                    'AllowedMethods': {
                        'Quantity': 3,
                        'Items': ['GET', 'HEAD', 'POST'],
                        'CachedMethods': {
                            'Quantity': 2,
                            'Items': ['GET', 'HEAD']
                        }
                    },
                    'CachePolicyId': '4135ea2d-6df8-44a3-9df3-4b5a84be39ad',
                    'Compress': True,
                    'MinTTL': 0
                }
            ]
        },
        'PriceClass': 'PriceClass_All',  # All edge locations
        'ViewerCertificate': {
            'CloudFrontDefaultCertificate': True
        }
    }

    # Create distribution
    response = cloudfront_client.create_distribution(
        DistributionConfig=config
    )

    distribution_id = response['Distribution']['Id']
    domain_name = response['Distribution']['DomainName']

    print(f"Created CloudFront distribution: {domain_name}")
    print(f"Distribution ID: {distribution_id}")

    return {
        'distribution_id': distribution_id,
        'domain_name': domain_name
    }

# Usage
cdn_config = create_ml_cdn('ml-alb-123456.us-east-1.elb.amazonaws.com')
print(f"Access your model at: https://{cdn_config['domain_name']}/predict")
```

### CDN Caching Strategy for ML

```python
"""
CDN Caching Strategy for Different ML Use Cases
"""

# Use Case 1: Static Model Metadata (Cache aggressively)
model_info_cache = {
    'path': '/models/info',
    'cache_ttl': 86400,  # 24 hours
    'cache_key': 'model_id',
    'rationale': 'Model metadata rarely changes'
}

# Use Case 2: Feature Embeddings (Cache for logged-in users)
embeddings_cache = {
    'path': '/embeddings/*',
    'cache_ttl': 3600,  # 1 hour
    'cache_key': 'user_id + resource_id',
    'rationale': 'Embeddings are expensive to compute, safe to cache per user'
}

# Use Case 3: Real-time Predictions (No caching)
prediction_no_cache = {
    'path': '/predict',
    'cache_ttl': 0,  # No caching
    'rationale': 'Predictions must be fresh for each request'
}

# Use Case 4: Batch Inference Results (Short cache)
batch_results_cache = {
    'path': '/batch/results/*',
    'cache_ttl': 300,  # 5 minutes
    'cache_key': 'batch_id',
    'rationale': 'Results don\'t change, safe to cache briefly'
}
```

---

## Network Security

Securing ML infrastructure requires multiple layers of network security.

### Security Group Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Security Group Layers                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Layer 1: Load Balancer SG                                 │
│  ──────────────────────                                    │
│  Inbound:  0.0.0.0/0:80,443 (Public internet)             │
│  Outbound: App Server SG:8000                              │
│                                                            │
│  Layer 2: Application Server SG                            │
│  ───────────────────────────                               │
│  Inbound:  Load Balancer SG:8000                           │
│  Outbound: Database SG:5432, Redis SG:6379, S3 endpoint   │
│                                                            │
│  Layer 3: Database SG                                      │
│  ─────────────────────                                     │
│  Inbound:  App Server SG:5432                              │
│  Outbound: None (no outbound required)                     │
│                                                            │
│  Layer 4: Training Instance SG                             │
│  ──────────────────────────                                │
│  Inbound:  Bastion SG:22 (SSH)                             │
│  Outbound: S3 endpoint, ECR (pull images), Internet (pip) │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Creating Security Groups

```python
import boto3

ec2_client = boto3.client('ec2')

def create_ml_security_groups(vpc_id):
    """
    Create layered security groups for ML infrastructure

    Returns:
        Dictionary of security group IDs
    """
    # 1. Bastion SG (SSH access)
    bastion_sg = ec2_client.create_security_group(
        GroupName='ml-bastion-sg',
        Description='Bastion host security group',
        VpcId=vpc_id
    )
    bastion_sg_id = bastion_sg['GroupId']

    # Allow SSH from specific IP (your office/home)
    ec2_client.authorize_security_group_ingress(
        GroupId=bastion_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{'CidrIp': '1.2.3.4/32', 'Description': 'Office IP'}]
            }
        ]
    )

    # 2. Load Balancer SG
    lb_sg = ec2_client.create_security_group(
        GroupName='ml-lb-sg',
        Description='Load balancer security group',
        VpcId=vpc_id
    )
    lb_sg_id = lb_sg['GroupId']

    # Allow HTTP/HTTPS from internet
    ec2_client.authorize_security_group_ingress(
        GroupId=lb_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 80,
                'ToPort': 80,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            },
            {
                'IpProtocol': 'tcp',
                'FromPort': 443,
                'ToPort': 443,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            }
        ]
    )

    # 3. Application Server SG
    app_sg = ec2_client.create_security_group(
        GroupName='ml-app-sg',
        Description='ML application server security group',
        VpcId=vpc_id
    )
    app_sg_id = app_sg['GroupId']

    # Allow traffic from load balancer
    ec2_client.authorize_security_group_ingress(
        GroupId=app_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 8000,
                'ToPort': 8000,
                'UserIdGroupPairs': [{'GroupId': lb_sg_id}]
            }
        ]
    )

    # 4. Database SG
    db_sg = ec2_client.create_security_group(
        GroupName='ml-db-sg',
        Description='Database security group',
        VpcId=vpc_id
    )
    db_sg_id = db_sg['GroupId']

    # Allow traffic from app servers
    ec2_client.authorize_security_group_ingress(
        GroupId=db_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 5432,
                'ToPort': 5432,
                'UserIdGroupPairs': [{'GroupId': app_sg_id}]
            }
        ]
    )

    # 5. Training Instance SG
    training_sg = ec2_client.create_security_group(
        GroupName='ml-training-sg',
        Description='ML training instance security group',
        VpcId=vpc_id
    )
    training_sg_id = training_sg['GroupId']

    # Allow SSH from bastion
    ec2_client.authorize_security_group_ingress(
        GroupId=training_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'UserIdGroupPairs': [{'GroupId': bastion_sg_id}]
            }
        ]
    )

    # Allow outbound HTTPS (for pip, downloads)
    ec2_client.authorize_security_group_egress(
        GroupId=training_sg_id,
        IpPermissions=[
            {
                'IpProtocol': 'tcp',
                'FromPort': 443,
                'ToPort': 443,
                'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
            }
        ]
    )

    print("Created all security groups")

    return {
        'bastion_sg_id': bastion_sg_id,
        'lb_sg_id': lb_sg_id,
        'app_sg_id': app_sg_id,
        'db_sg_id': db_sg_id,
        'training_sg_id': training_sg_id
    }

# Usage
sg_config = create_ml_security_groups('vpc-12345678')
```

### Network ACLs (Additional Layer)

```python
def create_network_acl(vpc_id, subnet_id):
    """
    Create Network ACL for additional subnet-level security

    Use case: Block specific IP ranges, DDoS protection
    """
    ec2_client = boto3.client('ec2')

    # Create NACL
    nacl_response = ec2_client.create_network_acl(
        VpcId=vpc_id,
        TagSpecifications=[
            {
                'ResourceType': 'network-acl',
                'Tags': [{'Key': 'Name', 'Value': 'ml-subnet-nacl'}]
            }
        ]
    )
    nacl_id = nacl_response['NetworkAcl']['NetworkAclId']

    # Allow inbound HTTP/HTTPS
    ec2_client.create_network_acl_entry(
        NetworkAclId=nacl_id,
        RuleNumber=100,
        Protocol='6',  # TCP
        RuleAction='allow',
        Egress=False,
        CidrBlock='0.0.0.0/0',
        PortRange={'From': 80, 'To': 80}
    )

    ec2_client.create_network_acl_entry(
        NetworkAclId=nacl_id,
        RuleNumber=110,
        Protocol='6',
        RuleAction='allow',
        Egress=False,
        CidrBlock='0.0.0.0/0',
        PortRange={'From': 443, 'To': 443}
    )

    # Deny specific IP range (example: block malicious traffic)
    ec2_client.create_network_acl_entry(
        NetworkAclId=nacl_id,
        RuleNumber=50,
        Protocol='-1',  # All protocols
        RuleAction='deny',
        Egress=False,
        CidrBlock='192.0.2.0/24'  # Example blocked range
    )

    # Allow all outbound
    ec2_client.create_network_acl_entry(
        NetworkAclId=nacl_id,
        RuleNumber=100,
        Protocol='-1',
        RuleAction='allow',
        Egress=True,
        CidrBlock='0.0.0.0/0'
    )

    # Associate with subnet
    ec2_client.replace_network_acl_association(
        AssociationId=subnet_id,
        NetworkAclId=nacl_id
    )

    print(f"Created NACL: {nacl_id}")
```

---

## Hybrid and Multi-Cloud Networking

Connect on-premises infrastructure with cloud for hybrid ML workflows.

### VPN Setup for Secure Access

```python
import boto3

ec2_client = boto3.client('ec2')

def create_vpn_connection(vpc_id, customer_gateway_ip):
    """
    Create VPN connection to on-premises datacenter

    Args:
        vpc_id: VPC ID
        customer_gateway_ip: Public IP of on-premises VPN device

    Returns:
        VPN connection ID
    """
    # Create Virtual Private Gateway
    vpg_response = ec2_client.create_vpn_gateway(
        Type='ipsec.1',
        TagSpecifications=[
            {
                'ResourceType': 'vpn-gateway',
                'Tags': [{'Key': 'Name', 'Value': 'ml-vpn-gateway'}]
            }
        ]
    )
    vpg_id = vpg_response['VpnGateway']['VpnGatewayId']

    # Attach to VPC
    ec2_client.attach_vpn_gateway(
        VpcId=vpc_id,
        VpnGatewayId=vpg_id
    )

    # Create Customer Gateway
    cgw_response = ec2_client.create_customer_gateway(
        Type='ipsec.1',
        PublicIp=customer_gateway_ip,
        BgpAsn=65000,
        TagSpecifications=[
            {
                'ResourceType': 'customer-gateway',
                'Tags': [{'Key': 'Name', 'Value': 'onprem-gateway'}]
            }
        ]
    )
    cgw_id = cgw_response['CustomerGateway']['CustomerGatewayId']

    # Create VPN Connection
    vpn_response = ec2_client.create_vpn_connection(
        Type='ipsec.1',
        CustomerGatewayId=cgw_id,
        VpnGatewayId=vpg_id,
        Options={
            'StaticRoutesOnly': False,  # Use BGP
            'TunnelOptions': [
                {
                    'TunnelInsideCidr': '169.254.10.0/30',
                    'PreSharedKey': 'your-pre-shared-key-here'
                },
                {
                    'TunnelInsideCidr': '169.254.11.0/30',
                    'PreSharedKey': 'your-pre-shared-key-here'
                }
            ]
        },
        TagSpecifications=[
            {
                'ResourceType': 'vpn-connection',
                'Tags': [{'Key': 'Name', 'Value': 'ml-vpn'}]
            }
        ]
    )

    vpn_id = vpn_response['VpnConnection']['VpnConnectionId']

    print(f"Created VPN connection: {vpn_id}")
    print(f"Configure your on-premises device with the provided config")

    return {
        'vpn_id': vpn_id,
        'vpg_id': vpg_id,
        'cgw_id': cgw_id
    }
```

### AWS Direct Connect for High Bandwidth

```python
"""
AWS Direct Connect for ML Workloads

Use cases:
- High-bandwidth data transfer (1-100 Gbps)
- Low-latency training data access from on-prem storage
- Hybrid training (on-prem + cloud)

Benefits:
- Consistent network performance
- Reduced bandwidth costs
- Private connectivity (not over internet)

Setup:
1. Order Direct Connect connection (AWS Console)
2. Configure Virtual Interface
3. Set up BGP routing
4. Connect to your datacenter

Cost:
- Port hour: $0.30/hour (1 Gbps) to $2.25/hour (10 Gbps)
- Data transfer: $0.02/GB (outbound)
"""
```

---

## Service Mesh

Service mesh provides advanced traffic management for microservices-based ML systems.

### Istio for ML Microservices

```yaml
# ML Inference Service Mesh with Istio

# Virtual Service for Traffic Management
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-service
spec:
  hosts:
  - ml-model.prod.svc.cluster.local
  http:
  - match:
    - headers:
        version:
          exact: v2
    route:
    - destination:
        host: ml-model.prod.svc.cluster.local
        subset: v2
  - route:  # Default route
    - destination:
        host: ml-model.prod.svc.cluster.local
        subset: v1
      weight: 90
    - destination:
        host: ml-model.prod.svc.cluster.local
        subset: v2
      weight: 10
---
# Destination Rule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ml-model-dest-rule
spec:
  host: ml-model.prod.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

---

## Network Performance Optimization

### Placement Groups for Distributed Training

```python
def create_cluster_placement_group(name='ml-training-cluster'):
    """
    Create placement group for low-latency distributed training

    Use case: Multi-GPU training with AllReduce
    """
    ec2_client = boto3.client('ec2')

    response = ec2_client.create_placement_group(
        GroupName=name,
        Strategy='cluster',  # Low-latency, high-bandwidth
        TagSpecifications=[
            {
                'ResourceType': 'placement-group',
                'Tags': [{'Key': 'Purpose', 'Value': 'distributed-training'}]
            }
        ]
    )

    print(f"Created placement group: {name}")
    print("Launch instances with: --placement-group {name}")
    return name

# Launch instances in placement group
def launch_training_cluster(placement_group, num_instances=4):
    """Launch GPU instances in placement group"""
    ec2_client = boto3.client('ec2')

    response = ec2_client.run_instances(
        ImageId='ami-12345678',
        InstanceType='p3.8xlarge',  # 4x V100 GPUs
        MinCount=num_instances,
        MaxCount=num_instances,
        Placement={
            'GroupName': placement_group
        },
        NetworkInterfaces=[
            {
                'DeviceIndex': 0,
                'AssociatePublicIpAddress': False,
                'SubnetId': 'subnet-12345678',
                'Groups': ['sg-training']
            }
        ]
    )

    instance_ids = [i['InstanceId'] for i in response['Instances']]
    print(f"Launched {num_instances} instances in placement group")
    return instance_ids
```

### Enhanced Networking (ENA)

```python
"""
Enhanced Networking for ML Workloads

Benefits:
- Higher bandwidth (up to 100 Gbps)
- Higher packet per second (PPS) performance
- Lower latency
- Lower jitter

Supported instances:
- GPU: p3, p4, g4, g5
- Compute: c5, c6i, m5, m6i

Enable ENA:
- Automatically enabled on supported instances
- Verify: ethtool -i eth0 | grep ena

Performance:
- Standard: 5-10 Gbps
- ENA: 25-100 Gbps
"""
```

---

## Monitoring and Troubleshooting

### Network Monitoring with CloudWatch

```python
import boto3
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')

def get_network_metrics(instance_id, hours=1):
    """
    Get network performance metrics for instance

    Returns:
        Network statistics
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Network In
    network_in = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='NetworkIn',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,  # 5 minutes
        Statistics=['Average', 'Maximum']
    )

    # Network Out
    network_out = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='NetworkOut',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average', 'Maximum']
    )

    avg_in = sum(d['Average'] for d in network_in['Datapoints']) / len(network_in['Datapoints']) if network_in['Datapoints'] else 0
    avg_out = sum(d['Average'] for d in network_out['Datapoints']) / len(network_out['Datapoints']) if network_out['Datapoints'] else 0

    return {
        'avg_network_in_mbps': avg_in / 1024 / 1024 * 8,
        'avg_network_out_mbps': avg_out / 1024 / 1024 * 8,
        'max_network_in_mbps': max((d['Maximum'] for d in network_in['Datapoints']), default=0) / 1024 / 1024 * 8,
        'max_network_out_mbps': max((d['Maximum'] for d in network_out['Datapoints']), default=0) / 1024 / 1024 * 8
    }

# Usage
metrics = get_network_metrics('i-1234567890abcdef0')
print(f"Average network in: {metrics['avg_network_in_mbps']:.2f} Mbps")
print(f"Average network out: {metrics['avg_network_out_mbps']:.2f} Mbps")
```

### Troubleshooting Network Issues

```python
"""
Common Network Issues and Solutions

1. High Latency (>100ms)
   - Check: Placement groups, instance type, region
   - Solution: Use placement groups, enable ENA, choose closer region

2. Low Bandwidth (<1 Gbps)
   - Check: Instance type bandwidth limits, security groups
   - Solution: Upgrade instance type, check for throttling

3. Connection Timeouts
   - Check: Security groups, NACLs, route tables
   - Solution: Verify SG rules, check NAT gateway, validate routes

4. Intermittent Failures
   - Check: Load balancer health checks, target health
   - Solution: Adjust health check settings, increase timeout

5. Cross-Region Latency
   - Check: Inter-region latency (50-200ms is normal)
   - Solution: Use CloudFront/CDN, consider data locality
"""
```

---

## Hands-on Exercise

### Exercise: Build Secure Multi-Tier ML Network

**Objective**: Deploy a production-grade ML system with:
- Multi-AZ VPC
- Layered security groups
- Application Load Balancer
- NAT Gateway
- VPN access for training instances
- CloudWatch monitoring

**Requirements**:
1. High availability (99.9%+)
2. Secure access (no direct internet to training)
3. Load balanced inference API
4. <100ms API latency
5. <$500/month cost

**Expected Architecture**:
- 2 Availability Zones
- 6 Subnets (public, app, training per AZ)
- 5 Security Groups
- 1 Application Load Balancer
- 2 NAT Gateways (HA)
- 3 Model Servers (auto-scaling)

---

## Summary

In this lesson, you learned:

✅ Design VPCs for ML infrastructure
✅ Configure load balancers for model serving
✅ Implement CDN for global delivery
✅ Secure networks with layered security groups
✅ Set up VPN and Direct Connect
✅ Optimize network performance (ENA, placement groups)
✅ Implement service mesh for microservices
✅ Monitor and troubleshoot network issues

**Key Takeaways**:
- VPC design impacts security, performance, and cost
- Load balancers enable A/B testing and gradual rollouts
- CDN reduces latency by 75% for global users
- Layered security (SG + NACL) provides defense in depth
- Placement groups reduce training time by 30%

**Next Steps**:
- Complete hands-on exercise
- Implement VPC for your project
- Configure load balancer with A/B testing
- Proceed to Lesson 07: Managed ML Services

---

**Estimated Time to Complete**: 6 hours (including hands-on exercise)
**Difficulty**: Intermediate
**Next Lesson**: [07-managed-ml-services.md](./07-managed-ml-services.md)
