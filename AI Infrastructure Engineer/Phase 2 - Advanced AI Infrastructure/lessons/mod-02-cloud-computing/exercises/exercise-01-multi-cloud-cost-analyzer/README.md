# Exercise 01: Multi-Cloud Cost Analyzer

## Learning Objectives

By completing this exercise, you will:
- Compare costs across AWS, GCP, and Azure
- Understand cloud pricing models and billing APIs
- Build automated cost analysis tools
- Implement cost optimization recommendations
- Create interactive cost dashboards

## Overview

Cloud costs are a major concern for AI/ML infrastructure teams. This exercise builds a comprehensive cost analysis tool that compares pricing across AWS, GCP, and Azure, helping teams make informed decisions and optimize spending.

## Prerequisites

- Python 3.11+ with cloud SDKs (boto3, google-cloud, azure-sdk)
- Active accounts on AWS, GCP, Azure (free tier sufficient)
- Understanding of cloud services (compute, storage, networking)
- Basic knowledge of cloud billing concepts

## Problem Statement

Build `cloudcost`, a multi-cloud cost analyzer that:

1. **Fetches pricing data** from AWS, GCP, and Azure APIs
2. **Compares costs** for equivalent services across clouds
3. **Analyzes actual usage** and spending patterns
4. **Recommends optimizations** (reserved instances, spot instances, right-sizing)
5. **Generates reports** with visualizations and actionable insights

## Requirements

### Functional Requirements

#### FR1: Pricing Data Collection
- Fetch pricing data from cloud provider APIs:
  - AWS: Pricing API, Cost Explorer API
  - GCP: Cloud Billing API, Pricing API
  - Azure: Retail Prices API, Cost Management API
- Cache pricing data locally (updated daily)
- Support multiple regions
- Handle pricing for:
  - Compute (VMs, GPUs)
  - Storage (S3, GCS, Blob)
  - Networking (data transfer)
  - Managed services (RDS, Cloud SQL, etc.)

#### FR2: Cost Comparison
- Compare equivalent instance types:
  - AWS EC2 vs GCP Compute Engine vs Azure VMs
  - GPU instances across clouds
  - Storage costs (per GB/month)
  - Network egress costs
- Calculate total cost of ownership (TCO)
- Support on-demand, reserved, and spot pricing
- Account for sustained use discounts (GCP)
- Factor in commitment discounts

#### FR3: Usage Analysis
- Connect to cloud billing APIs
- Fetch historical spending data
- Analyze usage patterns:
  - Peak vs off-peak usage
  - Underutilized resources
  - Cost trends over time
- Identify cost anomalies and spikes

#### FR4: Cost Optimization
- Recommend optimizations:
  - Right-sizing (reduce instance size)
  - Reserved instances vs on-demand
  - Spot/preemptible instances
  - Storage class optimization
  - Region selection
- Calculate potential savings
- Prioritize recommendations by impact

#### FR5: Reporting
- Generate interactive dashboards
- Export reports (PDF, HTML, JSON, CSV)
- Visualize:
  - Cost trends
  - Cross-cloud comparisons
  - Savings opportunities
  - Budget vs actual spending

### Non-Functional Requirements

#### NFR1: Accuracy
- Pricing data accurate to provider APIs
- Calculations match cloud billing
- Include all hidden costs (data transfer, API calls)

#### NFR2: Performance
- Cache pricing data to reduce API calls
- Asynchronous data fetching
- Process large billing datasets efficiently

#### NFR3: Security
- Never store cloud credentials in code
- Use IAM roles and service accounts
- Support credential rotation
- Audit logging for all operations

## Implementation Tasks

### Task 1: Cloud Provider Abstraction (5-6 hours)

Build a unified interface for all cloud providers:

```python
# src/cloud_providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

class InstanceFamily(Enum):
    """Instance family types"""
    GENERAL_PURPOSE = "general_purpose"
    COMPUTE_OPTIMIZED = "compute_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    GPU = "gpu"
    STORAGE_OPTIMIZED = "storage_optimized"

class PricingModel(Enum):
    """Pricing models"""
    ON_DEMAND = "on_demand"
    RESERVED_1Y = "reserved_1y"
    RESERVED_3Y = "reserved_3y"
    SPOT = "spot"
    PREEMPTIBLE = "preemptible"

@dataclass
class InstanceSpec:
    """Cloud instance specification"""
    provider: str  # "aws", "gcp", "azure"
    instance_type: str  # e.g., "n1-standard-4"
    vcpus: int
    memory_gb: float
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    family: InstanceFamily = InstanceFamily.GENERAL_PURPOSE
    region: str = "us-east-1"

@dataclass
class PricingInfo:
    """Instance pricing information"""
    instance_spec: InstanceSpec
    pricing_model: PricingModel
    price_per_hour: float
    price_per_month: float  # 730 hours
    currency: str = "USD"
    effective_date: datetime = None

@dataclass
class StoragePricing:
    """Storage pricing"""
    provider: str
    storage_class: str  # "standard", "nearline", "coldline", etc.
    region: str
    price_per_gb_month: float
    retrieval_fee_per_gb: Optional[float] = None

class CloudProvider(ABC):
    """Abstract base class for cloud providers"""

    def __init__(self, region: str = None):
        self.region = region
        self.pricing_cache = {}

    @abstractmethod
    def get_instance_pricing(
        self,
        instance_type: str,
        pricing_model: PricingModel = PricingModel.ON_DEMAND,
        region: Optional[str] = None
    ) -> PricingInfo:
        """
        Get pricing for specific instance type

        TODO: Implement for each cloud provider
        - Query pricing API
        - Parse response
        - Cache results
        - Handle API errors
        """
        pass

    @abstractmethod
    def list_instance_types(
        self,
        family: Optional[InstanceFamily] = None,
        min_vcpus: Optional[int] = None,
        min_memory_gb: Optional[float] = None,
        gpu_required: bool = False
    ) -> List[InstanceSpec]:
        """
        TODO: List available instance types matching criteria
        """
        pass

    @abstractmethod
    def get_storage_pricing(
        self,
        storage_class: str,
        region: Optional[str] = None
    ) -> StoragePricing:
        """TODO: Get storage pricing"""
        pass

    @abstractmethod
    def get_network_pricing(
        self,
        from_region: str,
        to_region: Optional[str] = None,
        to_internet: bool = False
    ) -> Dict[str, float]:
        """
        TODO: Get network/egress pricing

        Returns:
            {
                "inter_region_per_gb": 0.02,
                "internet_egress_per_gb": 0.09
            }
        """
        pass

    @abstractmethod
    def get_actual_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        TODO: Get actual spending from billing API

        Args:
            start_date: Start of date range
            end_date: End of date range
            group_by: Group costs by service, region, tag, etc.

        Returns:
            Cost breakdown dict
        """
        pass

    def find_equivalent_instance(
        self,
        target_spec: InstanceSpec
    ) -> Optional[InstanceSpec]:
        """
        TODO: Find equivalent instance on this cloud

        Match by:
        - Similar vCPU count (±25%)
        - Similar memory (±25%)
        - Same GPU type/count if applicable
        """
        pass
```

**Acceptance Criteria**:
- [ ] Abstract interface covers all providers
- [ ] Consistent API across implementations
- [ ] Proper error handling
- [ ] Caching implemented

---

### Task 2: AWS Implementation (4-5 hours)

```python
# src/cloud_providers/aws.py

import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

class AWSProvider(CloudProvider):
    """AWS pricing and billing"""

    def __init__(self, region: str = "us-east-1"):
        super().__init__(region)
        self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        self.ce_client = boto3.client('ce', region_name='us-east-1')
        self.ec2_client = boto3.client('ec2', region_name=region)

    def get_instance_pricing(
        self,
        instance_type: str,
        pricing_model: PricingModel = PricingModel.ON_DEMAND,
        region: Optional[str] = None
    ) -> PricingInfo:
        """
        TODO: Get EC2 instance pricing

        ```python
        region = region or self.region

        # Check cache
        cache_key = f"{instance_type}_{pricing_model.value}_{region}"
        if cache_key in self.pricing_cache:
            return self.pricing_cache[cache_key]

        # Query AWS Pricing API
        filters = [
            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)},
            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
            {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'}
        ]

        response = self.pricing_client.get_products(
            ServiceCode='AmazonEC2',
            Filters=filters
        )

        # Parse response and extract pricing
        for price_item in response['PriceList']:
            price_data = json.loads(price_item)
            # Extract on-demand or reserved pricing
            # ...

        # Get instance specs
        spec = self._get_instance_specs(instance_type, region)

        pricing = PricingInfo(
            instance_spec=spec,
            pricing_model=pricing_model,
            price_per_hour=price_per_hour,
            price_per_month=price_per_hour * 730,
            currency="USD",
            effective_date=datetime.now()
        )

        self.pricing_cache[cache_key] = pricing
        return pricing
        ```
        """
        pass

    def list_instance_types(
        self,
        family: Optional[InstanceFamily] = None,
        min_vcpus: Optional[int] = None,
        min_memory_gb: Optional[float] = None,
        gpu_required: bool = False
    ) -> List[InstanceSpec]:
        """
        TODO: List EC2 instance types

        ```python
        response = self.ec2_client.describe_instance_types()

        instances = []
        for instance_type in response['InstanceTypes']:
            spec = InstanceSpec(
                provider="aws",
                instance_type=instance_type['InstanceType'],
                vcpus=instance_type['VCpuInfo']['DefaultVCpus'],
                memory_gb=instance_type['MemoryInfo']['SizeInMiB'] / 1024,
                region=self.region
            )

            # Check GPU
            if 'GpuInfo' in instance_type:
                spec.gpu_count = instance_type['GpuInfo']['Gpus'][0]['Count']
                spec.gpu_type = instance_type['GpuInfo']['Gpus'][0]['Name']
                spec.family = InstanceFamily.GPU

            # Filter by criteria
            if min_vcpus and spec.vcpus < min_vcpus:
                continue
            if min_memory_gb and spec.memory_gb < min_memory_gb:
                continue
            if gpu_required and spec.gpu_count == 0:
                continue

            instances.append(spec)

        return instances
        ```
        """
        pass

    def get_storage_pricing(
        self,
        storage_class: str,
        region: Optional[str] = None
    ) -> StoragePricing:
        """
        TODO: Get S3 storage pricing

        Storage classes:
        - STANDARD
        - STANDARD_IA (Infrequent Access)
        - INTELLIGENT_TIERING
        - GLACIER
        - GLACIER_DEEP_ARCHIVE
        """
        pass

    def get_network_pricing(
        self,
        from_region: str,
        to_region: Optional[str] = None,
        to_internet: bool = False
    ) -> Dict[str, float]:
        """TODO: Get data transfer pricing"""
        pass

    def get_actual_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        TODO: Get actual costs from AWS Cost Explorer

        ```python
        group_by = group_by or ['SERVICE']

        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': key} for key in group_by]
        )

        costs = {}
        for result in response['ResultsByTime']:
            for group in result['Groups']:
                key = '_'.join(group['Keys'])
                amount = float(group['Metrics']['UnblendedCost']['Amount'])
                costs[key] = costs.get(key, 0) + amount

        return costs
        ```
        """
        pass

    def _get_location_name(self, region: str) -> str:
        """
        TODO: Convert region code to location name

        Example: us-east-1 -> US East (N. Virginia)
        """
        pass

    def _get_instance_specs(self, instance_type: str, region: str) -> InstanceSpec:
        """TODO: Get instance specifications"""
        pass

    def get_spot_pricing(
        self,
        instance_type: str,
        region: Optional[str] = None
    ) -> float:
        """
        TODO: Get current spot instance pricing

        ```python
        response = self.ec2_client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            MaxResults=1,
            ProductDescriptions=['Linux/UNIX']
        )

        if response['SpotPriceHistory']:
            return float(response['SpotPriceHistory'][0]['SpotPrice'])
        return None
        ```
        """
        pass
```

**Acceptance Criteria**:
- [ ] Fetches EC2 pricing from AWS Pricing API
- [ ] Gets actual costs from Cost Explorer
- [ ] Supports on-demand, reserved, and spot pricing
- [ ] Handles pagination and errors
- [ ] Caches pricing data

---

### Task 3: GCP Implementation (4-5 hours)

```python
# src/cloud_providers/gcp.py

from google.cloud import billing_v1
from google.cloud import compute_v1
from typing import List, Dict, Optional
from datetime import datetime
import requests

class GCPProvider(CloudProvider):
    """GCP pricing and billing"""

    def __init__(self, region: str = "us-east1", project_id: Optional[str] = None):
        super().__init__(region)
        self.project_id = project_id
        self.compute_client = compute_v1.InstancesClient()
        self.pricing_url = "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus"

    def get_instance_pricing(
        self,
        instance_type: str,
        pricing_model: PricingModel = PricingModel.ON_DEMAND,
        region: Optional[str] = None
    ) -> PricingInfo:
        """
        TODO: Get Compute Engine pricing

        ```python
        # GCP pricing is via Cloud Billing API
        # Example: n1-standard-4 in us-east1

        # Fetch pricing from Billing API
        response = requests.get(
            self.pricing_url,
            params={'currencyCode': 'USD'}
        )

        skus = response.json().get('skus', [])

        # Find SKU for this instance type and region
        for sku in skus:
            if instance_type in sku['description'].lower():
                if region in sku['serviceRegions']:
                    # Extract pricing from SKU
                    pricing_info = sku['pricingInfo'][0]
                    price_per_hour = self._parse_pricing(pricing_info)

                    spec = self._get_instance_specs(instance_type, region)

                    # GCP has sustained use discounts (automatic)
                    # Adjust pricing based on expected usage
                    sustained_discount = 0.7  # 30% discount for full month
                    price_per_month = price_per_hour * 730 * sustained_discount

                    return PricingInfo(
                        instance_spec=spec,
                        pricing_model=pricing_model,
                        price_per_hour=price_per_hour,
                        price_per_month=price_per_month,
                        currency="USD"
                    )
        ```
        """
        pass

    def list_instance_types(
        self,
        family: Optional[InstanceFamily] = None,
        min_vcpus: Optional[int] = None,
        min_memory_gb: Optional[float] = None,
        gpu_required: bool = False
    ) -> List[InstanceSpec]:
        """
        TODO: List GCP machine types

        GCP families:
        - n1-standard-* (general purpose)
        - n1-highmem-* (memory optimized)
        - n1-highcpu-* (compute optimized)
        - n2-* (newer generation)
        - c2-* (compute optimized)
        - m1-*, m2-* (memory optimized)
        - a2-* (GPU instances)
        """
        pass

    def get_storage_pricing(
        self,
        storage_class: str,
        region: Optional[str] = None
    ) -> StoragePricing:
        """
        TODO: Get Cloud Storage pricing

        Classes:
        - STANDARD
        - NEARLINE (30-day minimum)
        - COLDLINE (90-day minimum)
        - ARCHIVE (365-day minimum)
        """
        pass

    def get_actual_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        TODO: Get actual costs from Cloud Billing

        Requires:
        - Billing account ID
        - BigQuery export enabled
        - Query billing data from BigQuery
        """
        pass

    def get_preemptible_pricing(
        self,
        instance_type: str,
        region: Optional[str] = None
    ) -> float:
        """
        TODO: Get preemptible instance pricing

        Preemptible = ~80% discount from on-demand
        """
        pass
```

**Acceptance Criteria**:
- [ ] Fetches GCP pricing from Cloud Billing API
- [ ] Accounts for sustained use discounts
- [ ] Supports preemptible instances
- [ ] Gets actual costs from BigQuery billing export

---

### Task 4: Azure Implementation (4-5 hours)

```python
# src/cloud_providers/azure.py

from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.costmanagement import CostManagementClient
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Optional
from datetime import datetime
import requests

class AzureProvider(CloudProvider):
    """Azure pricing and billing"""

    def __init__(self, region: str = "eastus", subscription_id: Optional[str] = None):
        super().__init__(region)
        self.subscription_id = subscription_id
        self.credential = DefaultAzureCredential()

        if subscription_id:
            self.compute_client = ComputeManagementClient(
                self.credential,
                subscription_id
            )
            self.cost_client = CostManagementClient(self.credential)

        self.retail_pricing_url = "https://prices.azure.com/api/retail/prices"

    def get_instance_pricing(
        self,
        instance_type: str,
        pricing_model: PricingModel = PricingModel.ON_DEMAND,
        region: Optional[str] = None
    ) -> PricingInfo:
        """
        TODO: Get Azure VM pricing

        Use Azure Retail Prices API:
        https://prices.azure.com/api/retail/prices

        ```python
        region = region or self.region

        # Query retail prices API
        filter_query = (
            f"serviceName eq 'Virtual Machines' "
            f"and armSkuName eq '{instance_type}' "
            f"and armRegionName eq '{region}' "
            f"and priceType eq 'Consumption'"
        )

        response = requests.get(
            self.retail_pricing_url,
            params={'$filter': filter_query}
        )

        items = response.json().get('Items', [])

        for item in items:
            price_per_hour = item['retailPrice']

            spec = self._get_instance_specs(instance_type, region)

            return PricingInfo(
                instance_spec=spec,
                pricing_model=pricing_model,
                price_per_hour=price_per_hour,
                price_per_month=price_per_hour * 730,
                currency=item['currencyCode']
            )
        ```
        """
        pass

    def list_instance_types(
        self,
        family: Optional[InstanceFamily] = None,
        min_vcpus: Optional[int] = None,
        min_memory_gb: Optional[float] = None,
        gpu_required: bool = False
    ) -> List[InstanceSpec]:
        """
        TODO: List Azure VM sizes

        Families:
        - Standard_D* (general purpose)
        - Standard_E* (memory optimized)
        - Standard_F* (compute optimized)
        - Standard_NC*, Standard_ND* (GPU)
        """
        pass

    def get_storage_pricing(
        self,
        storage_class: str,
        region: Optional[str] = None
    ) -> StoragePricing:
        """
        TODO: Get Blob Storage pricing

        Tiers:
        - Hot
        - Cool
        - Archive
        """
        pass

    def get_actual_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        TODO: Get actual costs from Cost Management API

        ```python
        scope = f"/subscriptions/{self.subscription_id}"

        query = {
            "type": "ActualCost",
            "timeframe": "Custom",
            "timePeriod": {
                "from": start_date.strftime('%Y-%m-%dT00:00:00Z'),
                "to": end_date.strftime('%Y-%m-%dT00:00:00Z')
            },
            "dataset": {
                "granularity": "Daily",
                "aggregation": {
                    "totalCost": {
                        "name": "Cost",
                        "function": "Sum"
                    }
                },
                "grouping": [
                    {"type": "Dimension", "name": group}
                    for group in (group_by or ["ServiceName"])
                ]
            }
        }

        result = self.cost_client.query.usage(scope, query)

        # Parse result
        costs = {}
        for row in result.rows:
            # Extract cost data
            pass

        return costs
        ```
        """
        pass

    def get_spot_pricing(
        self,
        instance_type: str,
        region: Optional[str] = None
    ) -> float:
        """TODO: Get Azure Spot VM pricing"""
        pass
```

**Acceptance Criteria**:
- [ ] Fetches Azure pricing from Retail Prices API
- [ ] Gets actual costs from Cost Management API
- [ ] Supports spot VM pricing
- [ ] Handles Azure-specific pricing models

---

### Task 5: Cost Comparison Engine (5-6 hours)

```python
# src/cost_comparator.py

from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

from .cloud_providers.base import (
    CloudProvider, InstanceSpec, PricingInfo, PricingModel
)
from .cloud_providers.aws import AWSProvider
from .cloud_providers.gcp import GCPProvider
from .cloud_providers.azure import AzureProvider

@dataclass
class ComparisonResult:
    """Cross-cloud comparison result"""
    aws_instance: Optional[InstanceSpec] = None
    gcp_instance: Optional[InstanceSpec] = None
    azure_instance: Optional[InstanceSpec] = None

    aws_pricing: Optional[PricingInfo] = None
    gcp_pricing: Optional[PricingInfo] = None
    azure_pricing: Optional[PricingInfo] = None

    cheapest_provider: str = None
    max_savings_percent: float = 0.0
    max_savings_monthly: float = 0.0

class CostComparator:
    """Compare costs across cloud providers"""

    def __init__(self):
        self.providers = {
            'aws': AWSProvider(),
            'gcp': GCPProvider(),
            'azure': AzureProvider()
        }

    def compare_instance(
        self,
        vcpus: int,
        memory_gb: float,
        gpu_count: int = 0,
        region: str = "us-east",
        pricing_model: PricingModel = PricingModel.ON_DEMAND
    ) -> ComparisonResult:
        """
        TODO: Compare equivalent instance across clouds

        Algorithm:
        1. Find equivalent instance on each cloud
           - Match vCPU count (±25%)
           - Match memory (±25%)
           - Match GPU if required
        2. Get pricing for each
        3. Calculate cheapest option
        4. Calculate savings

        ```python
        target_spec = InstanceSpec(
            provider="target",
            instance_type="target",
            vcpus=vcpus,
            memory_gb=memory_gb,
            gpu_count=gpu_count
        )

        result = ComparisonResult()

        # Find equivalent on each cloud
        for provider_name, provider in self.providers.items():
            equivalent = provider.find_equivalent_instance(target_spec)
            if equivalent:
                pricing = provider.get_instance_pricing(
                    equivalent.instance_type,
                    pricing_model
                )

                setattr(result, f"{provider_name}_instance", equivalent)
                setattr(result, f"{provider_name}_pricing", pricing)

        # Determine cheapest
        prices = {}
        for provider in ['aws', 'gcp', 'azure']:
            pricing = getattr(result, f"{provider}_pricing")
            if pricing:
                prices[provider] = pricing.price_per_month

        if prices:
            result.cheapest_provider = min(prices, key=prices.get)
            cheapest_price = prices[result.cheapest_provider]
            most_expensive = max(prices.values())

            result.max_savings_monthly = most_expensive - cheapest_price
            result.max_savings_percent = (
                (most_expensive - cheapest_price) / most_expensive * 100
            )

        return result
        ```
        """
        pass

    def compare_workload(
        self,
        workload_config: Dict[str, any]
    ) -> pd.DataFrame:
        """
        TODO: Compare cost for complete workload

        workload_config example:
        {
            "compute": [
                {"vcpus": 4, "memory_gb": 16, "count": 10},
                {"vcpus": 8, "memory_gb": 32, "gpu_count": 1, "count": 2}
            ],
            "storage": [
                {"type": "standard", "size_gb": 1000},
                {"type": "archive", "size_gb": 10000}
            ],
            "network": {
                "egress_gb_month": 5000
            },
            "duration_months": 12
        }

        Calculate total TCO for each cloud
        """
        pass

    def generate_comparison_table(
        self,
        results: List[ComparisonResult]
    ) -> pd.DataFrame:
        """
        TODO: Generate comparison table

        Columns:
        - Spec (vCPU/Memory/GPU)
        - AWS Instance
        - AWS Price/month
        - GCP Instance
        - GCP Price/month
        - Azure Instance
        - Azure Price/month
        - Cheapest
        - Savings %
        """
        pass

    def recommend_cloud(
        self,
        workload_config: Dict,
        priorities: List[str] = None
    ) -> Dict[str, any]:
        """
        TODO: Recommend cloud provider

        Consider:
        - Cost (primary)
        - Performance requirements
        - Existing cloud presence
        - Geographic requirements
        - Compliance needs

        priorities example: ["cost", "gpu_availability", "region_coverage"]
        """
        pass
```

**Acceptance Criteria**:
- [ ] Compares equivalent instances accurately
- [ ] Calculates total workload costs
- [ ] Generates comparison tables
- [ ] Provides cloud recommendations
- [ ] Handles missing data gracefully

---

### Task 6: Cost Optimization Recommendations (4-5 hours)

```python
# src/optimizer.py

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Optimization:
    """Cost optimization recommendation"""
    resource_id: str
    resource_type: str  # "instance", "storage", "network"
    current_cost_monthly: float
    optimized_cost_monthly: float
    savings_monthly: float
    savings_percent: float
    recommendation: str
    effort: str  # "low", "medium", "high"
    risk: str  # "low", "medium", "high"
    priority: int  # 1-10, higher = more important

class CostOptimizer:
    """Generate cost optimization recommendations"""

    def analyze_usage(
        self,
        provider: CloudProvider,
        start_date: datetime,
        end_date: datetime
    ) -> List[Optimization]:
        """
        TODO: Analyze usage and generate recommendations

        Analyze:
        1. Underutilized instances (< 40% CPU for 7 days)
        2. Stopped instances still incurring costs
        3. Unattached volumes
        4. Old snapshots
        5. On-demand vs reserved opportunities
        6. Spot instance opportunities
        7. Storage class optimization
        8. Over-provisioned instances
        """
        pass

    def recommend_rightsizing(
        self,
        instance_id: str,
        current_type: str,
        avg_cpu_percent: float,
        avg_memory_percent: float,
        provider: CloudProvider
    ) -> Optional[Optimization]:
        """
        TODO: Recommend instance right-sizing

        If CPU < 40% and Memory < 50% for 7+ days:
        - Recommend smaller instance
        - Calculate savings
        - Estimate migration effort
        """
        pass

    def recommend_reserved_instances(
        self,
        usage_history: Dict,
        provider: CloudProvider
    ) -> List[Optimization]:
        """
        TODO: Recommend reserved instances

        If instance runs > 70% of time for 30+ days:
        - Calculate RI savings (up to 72%)
        - Recommend 1-year or 3-year commitment
        - Break-even analysis
        """
        pass

    def recommend_spot_instances(
        self,
        workload_type: str,
        current_instances: List[Dict],
        provider: CloudProvider
    ) -> List[Optimization]:
        """
        TODO: Recommend spot/preemptible instances

        For fault-tolerant workloads:
        - Calculate spot savings (up to 90%)
        - Risk assessment (interruption frequency)
        - Fallback strategy
        """
        pass

    def recommend_storage_optimization(
        self,
        storage_analysis: Dict,
        provider: CloudProvider
    ) -> List[Optimization]:
        """
        TODO: Recommend storage optimizations

        - Move infrequently accessed data to cheaper tiers
        - Delete old snapshots
        - Compress or deduplicate data
        - Lifecycle policies
        """
        pass

    def prioritize_recommendations(
        self,
        recommendations: List[Optimization]
    ) -> List[Optimization]:
        """
        TODO: Prioritize recommendations

        Score based on:
        - Potential savings (50%)
        - Implementation effort (30%)
        - Risk level (20%)

        Return sorted by priority
        """
        pass
```

**Acceptance Criteria**:
- [ ] Identifies underutilized resources
- [ ] Recommends right-sizing opportunities
- [ ] Suggests reserved instance purchases
- [ ] Recommends spot instance usage
- [ ] Prioritizes by ROI and effort

---

### Task 7: Reporting and Visualization (5-6 hours)

```python
# src/reporter.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from typing import List, Dict
from jinja2 import Template

class CostReporter:
    """Generate cost reports and visualizations"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cost_comparison(
        self,
        comparison_df: pd.DataFrame,
        output: str = "cost_comparison.html"
    ) -> None:
        """
        TODO: Plot cross-cloud cost comparison

        ```python
        fig = go.Figure()

        # Add bars for each cloud
        fig.add_trace(go.Bar(
            name='AWS',
            x=comparison_df.index,
            y=comparison_df['aws_price_monthly'],
            marker_color='#FF9900'
        ))

        fig.add_trace(go.Bar(
            name='GCP',
            x=comparison_df.index,
            y=comparison_df['gcp_price_monthly'],
            marker_color='#4285F4'
        ))

        fig.add_trace(go.Bar(
            name='Azure',
            x=comparison_df.index,
            y=comparison_df['azure_price_monthly'],
            marker_color='#00A4EF'
        ))

        fig.update_layout(
            title='Instance Cost Comparison Across Clouds',
            xaxis_title='Instance Configuration',
            yaxis_title='Cost per Month (USD)',
            barmode='group'
        )

        fig.write_html(self.output_dir / output)
        ```
        """
        pass

    def plot_cost_trends(
        self,
        cost_history: pd.DataFrame,
        output: str = "cost_trends.html"
    ) -> None:
        """
        TODO: Plot cost trends over time

        Line chart showing daily/weekly/monthly costs
        """
        pass

    def plot_savings_opportunities(
        self,
        optimizations: List[Optimization],
        output: str = "savings_opportunities.html"
    ) -> None:
        """
        TODO: Plot savings opportunities

        Waterfall chart or bar chart showing potential savings
        """
        pass

    def plot_cost_breakdown(
        self,
        costs_by_service: Dict[str, float],
        output: str = "cost_breakdown.html"
    ) -> None:
        """
        TODO: Plot cost breakdown by service

        Pie chart or treemap
        """
        pass

    def generate_html_report(
        self,
        comparison_results: List,
        optimizations: List[Optimization],
        cost_history: pd.DataFrame
    ) -> None:
        """
        TODO: Generate comprehensive HTML report

        Include:
        - Executive summary
        - Cost comparison tables and charts
        - Optimization recommendations (prioritized)
        - Cost trends
        - Action items
        """
        pass

    def export_csv(
        self,
        comparison_df: pd.DataFrame,
        output: str = "cost_comparison.csv"
    ) -> None:
        """TODO: Export comparison to CSV"""
        pass

    def export_json(
        self,
        data: Dict,
        output: str = "analysis.json"
    ) -> None:
        """TODO: Export analysis to JSON"""
        pass
```

**Acceptance Criteria**:
- [ ] Generates interactive charts with Plotly
- [ ] Creates comprehensive HTML reports
- [ ] Exports data in multiple formats
- [ ] Visualizations are publication-quality

---

## CLI Interface

```bash
# Compare instance pricing
cloudcost compare --vcpus 4 --memory 16 --region us-east

# Compare specific instances
cloudcost compare --aws m5.xlarge --gcp n1-standard-4 --azure Standard_D4s_v3

# Analyze current spending
cloudcost analyze --provider aws --days 30

# Get optimization recommendations
cloudcost optimize --provider aws --min-savings 100

# Generate full report
cloudcost report --config workload.yaml --output report/

# Compare complete workload
cloudcost workload --config workload.yaml
```

---

## Deliverables

1. **Source Code**
   - Cloud provider implementations (AWS, GCP, Azure)
   - Cost comparison engine
   - Optimization recommender
   - Reporter and visualizations
   - CLI application

2. **Configuration**
   - Example workload configurations
   - Pricing cache settings

3. **Tests**
   - Unit tests with mocked APIs
   - Integration tests (optional, requires credentials)

4. **Documentation**
   - README with setup and usage
   - API credentials setup guide
   - Example reports

---

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| **Accuracy** | 30% |
| **Completeness** | 25% |
| **Optimization Quality** | 20% |
| **Reporting** | 15% |
| **Code Quality** | 10% |

**Passing**: 70%+
**Excellence**: 90%+

---

## Estimated Time

- Task 1: 5-6 hours
- Task 2: 4-5 hours
- Task 3: 4-5 hours
- Task 4: 4-5 hours
- Task 5: 5-6 hours
- Task 6: 4-5 hours
- Task 7: 5-6 hours
- Testing: 4-5 hours
- Documentation: 2-3 hours

**Total**: 37-46 hours

---

**This exercise provides critical skills for cloud cost management in AI/ML infrastructure roles.**
