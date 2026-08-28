# Lesson 08: Multi-Cloud & Cost Optimization

**Duration:** 6 hours
**Difficulty:** Advanced
**Prerequisites:** Lessons 01-07 (Complete cloud computing module)

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Design multi-cloud strategies** for ML infrastructure
2. **Implement cost monitoring** and tracking across clouds
3. **Optimize cloud spending** with right-sizing and reservations
4. **Use spot/preemptible instances** effectively for training
5. **Implement FinOps practices** for ML teams
6. **Migrate workloads** between cloud providers
7. **Build cloud-agnostic** ML pipelines
8. **Forecast and budget** cloud costs accurately

---

## Table of Contents

1. [Multi-Cloud Strategies](#multi-cloud-strategies)
2. [Cost Monitoring and Tracking](#cost-monitoring-and-tracking)
3. [Cost Optimization Techniques](#cost-optimization-techniques)
4. [Spot and Preemptible Instances](#spot-and-preemptible-instances)
5. [FinOps for ML Teams](#finops-for-ml-teams)
6. [Cloud-Agnostic Architecture](#cloud-agnostic-architecture)
7. [Migration Between Clouds](#migration-between-clouds)
8. [Budgeting and Forecasting](#budgeting-and-forecasting)
9. [Best Practices](#best-practices)
10. [Hands-on Exercise](#hands-on-exercise)

---

## Multi-Cloud Strategies

Why and how organizations adopt multi-cloud for ML workloads.

### Multi-Cloud Patterns

```
┌────────────────────────────────────────────────────────────────┐
│                 Multi-Cloud Strategy Patterns                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Pattern 1: Active-Passive (Disaster Recovery)                 │
│  ─────────────────────────────────────────                     │
│  AWS (Primary)           Azure (Backup)                        │
│  ├── Training            ├── Cold standby                      │
│  ├── Inference           ├── Data replication                  │
│  └── Data storage        └── DR only                           │
│                                                                │
│  Use case: Risk mitigation, compliance                         │
│  Complexity: Low                                               │
│  Cost: +10-20% overhead                                        │
│                                                                │
│  Pattern 2: Best-of-Breed (Specialized)                        │
│  ──────────────────────────────────────                        │
│  AWS                     GCP                    Azure          │
│  ├── Data lake (S3)      ├── Training (TPU)    ├── Enterprise │
│  ├── General compute     ├── AutoML            ├── OpenAI     │
│  └── Edge (Greengrass)   └── BigQuery ML       └── Compliance │
│                                                                │
│  Use case: Leverage unique strengths                           │
│  Complexity: High                                              │
│  Cost: Optimized per workload                                  │
│                                                                │
│  Pattern 3: Geographic Distribution                            │
│  ──────────────────────────────────                            │
│  AWS (US)                GCP (Europe)          Azure (Asia)    │
│  ├── US customers        ├── EU customers      ├── APAC       │
│  └── Data residency      └── GDPR compliance   └── Latency    │
│                                                                │
│  Use case: Global reach, data sovereignty                      │
│  Complexity: Medium                                            │
│  Cost: Higher (multi-region data transfer)                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

```
┌──────────────────────────┬────────────────┬─────────────────────┐
│ Scenario                 │ Recommended    │ Reasoning           │
├──────────────────────────┼────────────────┼─────────────────────┤
│ Startup (<50 people)     │ Single cloud   │ Reduce complexity   │
│ Growth (50-200)          │ Single + DR    │ Risk mitigation     │
│ Enterprise (>200)        │ Multi-cloud    │ Avoid vendor lock-in│
│ AI-First Company         │ GCP + AWS      │ TPU + ecosystem     │
│ Microsoft Shop           │ Azure + backup │ Enterprise features │
│ Global Company           │ Multi-cloud    │ Data residency      │
│ Cost-Sensitive           │ Multi-cloud    │ Arbitrage           │
│ High Compliance          │ Multi-cloud    │ Redundancy          │
└──────────────────────────┴────────────────┴─────────────────────┘
```

---

## Cost Monitoring and Tracking

Effective cost management starts with visibility.

### Cost Allocation Strategy

```python
"""
Tag Strategy for ML Cost Tracking

Mandatory tags:
- Environment: dev, staging, production
- Team: data-science, ml-engineering, research
- Project: project-name
- CostCenter: engineering, research, operations
- Owner: email@company.com

Optional tags:
- Experiment: experiment-id
- Model: model-name
- Stage: training, inference, experimentation
"""

import boto3

def tag_ml_resources(resource_arn, tags):
    """
    Apply consistent tagging to ML resources

    Args:
        resource_arn: ARN of resource
        tags: Dictionary of tags
    """
    required_tags = ['Environment', 'Team', 'Project', 'CostCenter', 'Owner']

    # Validate required tags
    for tag in required_tags:
        if tag not in tags:
            raise ValueError(f"Missing required tag: {tag}")

    # Convert to AWS tag format
    aws_tags = [{'Key': k, 'Value': v} for k, v in tags.items()]

    # Apply tags (example for S3)
    s3_client = boto3.client('s3')
    if 's3' in resource_arn:
        bucket_name = resource_arn.split(':::')[-1]
        s3_client.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={'TagSet': aws_tags}
        )

# Usage
tag_ml_resources(
    resource_arn='arn:aws:s3:::ml-training-data',
    tags={
        'Environment': 'production',
        'Team': 'ml-engineering',
        'Project': 'image-classification',
        'CostCenter': 'engineering',
        'Owner': 'ml-team@company.com',
        'Stage': 'training'
    }
)
```

### Multi-Cloud Cost Dashboard

```python
import boto3
import google.cloud.billing_v1 as gcpbilling
from azure.mgmt.costmanagement import CostManagementClient
from datetime import datetime, timedelta
import pandas as pd

class MultiCloudCostTracker:
    """
    Track costs across AWS, GCP, and Azure

    Usage:
        tracker = MultiCloudCostTracker()
        costs = tracker.get_monthly_costs()
        print(costs)
    """

    def __init__(self):
        self.aws_client = boto3.client('ce')  # Cost Explorer
        # GCP and Azure clients initialized elsewhere

    def get_aws_costs(self, days=30):
        """Get AWS costs for last N days"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = self.aws_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'TAG', 'Key': 'Project'},
                {'Type': 'TAG', 'Key': 'Team'}
            ],
            Filter={
                'Tags': {
                    'Key': 'Environment',
                    'Values': ['production']
                }
            }
        )

        costs = []
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            for group in result['Groups']:
                costs.append({
                    'date': date,
                    'cloud': 'AWS',
                    'project': group['Keys'][0],
                    'team': group['Keys'][1],
                    'cost': float(group['Metrics']['UnblendedCost']['Amount'])
                })

        return pd.DataFrame(costs)

    def get_gcp_costs(self, days=30):
        """Get GCP costs for last N days"""
        # Similar implementation using GCP Billing API
        # Placeholder
        return pd.DataFrame()

    def get_azure_costs(self, days=30):
        """Get Azure costs for last N days"""
        # Similar implementation using Azure Cost Management API
        # Placeholder
        return pd.DataFrame()

    def get_total_costs(self, days=30):
        """Aggregate costs from all clouds"""
        aws_costs = self.get_aws_costs(days)
        gcp_costs = self.get_gcp_costs(days)
        azure_costs = self.get_azure_costs(days)

        all_costs = pd.concat([aws_costs, gcp_costs, azure_costs], ignore_index=True)

        # Group by project
        summary = all_costs.groupby('project')['cost'].sum().sort_values(ascending=False)

        return {
            'total': all_costs['cost'].sum(),
            'by_project': summary.to_dict(),
            'by_cloud': all_costs.groupby('cloud')['cost'].sum().to_dict(),
            'details': all_costs
        }

# Usage
tracker = MultiCloudCostTracker()
costs = tracker.get_total_costs(days=30)

print(f"Total monthly cost: ${costs['total']:.2f}")
print(f"\nBy cloud:")
for cloud, cost in costs['by_cloud'].items():
    print(f"  {cloud}: ${cost:.2f}")

print(f"\nTop 5 projects by cost:")
for project, cost in list(costs['by_project'].items())[:5]:
    print(f"  {project}: ${cost:.2f}")
```

---

## Cost Optimization Techniques

Proven strategies to reduce ML infrastructure costs.

### 1. Right-Sizing Instances

```python
import boto3
from datetime import datetime, timedelta

class InstanceRightSizer:
    """
    Analyze CloudWatch metrics to recommend right-sizing

    Identifies over-provisioned instances (CPU < 30%, Memory < 50%)
    """

    def __init__(self):
        self.ec2_client = boto3.client('ec2')
        self.cloudwatch = boto3.client('cloudwatch')

    def analyze_instance(self, instance_id, days=7):
        """
        Analyze instance utilization

        Returns:
            Recommendation dict with suggested instance type
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # Get CPU utilization
        cpu_stats = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average', 'Maximum']
        )

        if not cpu_stats['Datapoints']:
            return {'recommendation': 'Insufficient data'}

        avg_cpu = sum(d['Average'] for d in cpu_stats['Datapoints']) / len(cpu_stats['Datapoints'])
        max_cpu = max(d['Maximum'] for d in cpu_stats['Datapoints'])

        # Get instance details
        instance = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        current_type = instance['Reservations'][0]['Instances'][0]['InstanceType']

        # Recommendation logic
        if avg_cpu < 30 and max_cpu < 60:
            recommendation = 'DOWNSIZE'
            suggested_action = 'Consider smaller instance type'
            potential_savings = 0.4  # 40% savings
        elif avg_cpu > 80 or max_cpu > 95:
            recommendation = 'UPSIZE'
            suggested_action = 'Consider larger instance type'
            potential_savings = 0  # No savings, avoid throttling
        else:
            recommendation = 'OPTIMIZED'
            suggested_action = 'Instance is appropriately sized'
            potential_savings = 0

        return {
            'instance_id': instance_id,
            'current_type': current_type,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'recommendation': recommendation,
            'suggested_action': suggested_action,
            'potential_savings_pct': potential_savings * 100
        }

# Usage
rightsizer = InstanceRightSizer()
recommendation = rightsizer.analyze_instance('i-1234567890abcdef0')
print(f"Instance: {recommendation['current_type']}")
print(f"Avg CPU: {recommendation['avg_cpu']:.1f}%")
print(f"Recommendation: {recommendation['recommendation']}")
print(f"Potential savings: {recommendation['potential_savings_pct']:.0f}%")
```

### 2. Reserved Instances / Savings Plans

```python
"""
Reserved Instances vs Savings Plans Comparison

Reserved Instances (RI):
- Commitment: 1 or 3 years
- Savings: 40-60%
- Flexibility: Low (specific instance type)
- Best for: Steady-state workloads

Savings Plans:
- Commitment: 1 or 3 years
- Savings: 40-60%
- Flexibility: High (any instance in family)
- Best for: Dynamic workloads

Recommendation Matrix:
┌──────────────────────────┬─────────────────┬────────────────────┐
│ Workload Type            │ Recommended     │ Reasoning          │
├──────────────────────────┼─────────────────┼────────────────────┤
│ Inference (stable)       │ Reserved Inst   │ Predictable        │
│ Training (varied)        │ Savings Plan    │ Flexibility        │
│ Experimentation          │ On-demand/Spot  │ Variable usage     │
│ Batch processing         │ Spot instances  │ Interruptible      │
└──────────────────────────┴─────────────────┴────────────────────┘
"""

def calculate_ri_savings(instance_type, hours_per_month, months=36):
    """
    Calculate savings with Reserved Instances

    Example: p3.2xlarge (1x V100)
    - On-demand: $3.06/hour
    - 1-year RI: $2.08/hour (32% savings)
    - 3-year RI: $1.37/hour (55% savings)
    """
    pricing = {
        'p3.2xlarge': {
            'on_demand': 3.06,
            'ri_1yr': 2.08,
            'ri_3yr': 1.37
        },
        'p3.8xlarge': {
            'on_demand': 12.24,
            'ri_1yr': 8.32,
            'ri_3yr': 5.48
        }
    }

    if instance_type not in pricing:
        return None

    prices = pricing[instance_type]
    total_hours = hours_per_month * months

    on_demand_cost = prices['on_demand'] * total_hours

    if months <= 12:
        ri_cost = prices['ri_1yr'] * total_hours
        savings_pct = (1 - prices['ri_1yr'] / prices['on_demand']) * 100
    else:
        ri_cost = prices['ri_3yr'] * total_hours
        savings_pct = (1 - prices['ri_3yr'] / prices['on_demand']) * 100

    return {
        'instance_type': instance_type,
        'on_demand_cost': on_demand_cost,
        'ri_cost': ri_cost,
        'savings': on_demand_cost - ri_cost,
        'savings_pct': savings_pct,
        'payback_months': 0  # Immediate with no upfront RI
    }

# Example: Inference server running 24/7
result = calculate_ri_savings('p3.2xlarge', hours_per_month=730, months=36)
print(f"Savings over 3 years: ${result['savings']:,.0f} ({result['savings_pct']:.0f}%)")
```

### 3. Auto-Shutdown Policies

```python
import boto3
from datetime import datetime, time

class AutoShutdownManager:
    """
    Automatically stop instances during off-hours

    Use case: Development/training instances that don't need 24/7 runtime
    Savings: 50-75% (if only running 8-12 hours/day)
    """

    def __init__(self):
        self.ec2_client = boto3.client('ec2')

    def configure_shutdown_schedule(self, instance_ids, shutdown_time='19:00',
                                     startup_time='08:00', timezone='America/Los_Angeles'):
        """
        Configure auto-shutdown/startup schedule

        Args:
            instance_ids: List of instance IDs
            shutdown_time: Time to shutdown (HH:MM)
            startup_time: Time to start (HH:MM)
            timezone: Timezone for schedule
        """
        # Use AWS Instance Scheduler or Lambda
        # This is a simplified example

        # Tag instances for shutdown
        for instance_id in instance_ids:
            self.ec2_client.create_tags(
                Resources=[instance_id],
                Tags=[
                    {'Key': 'AutoShutdown', 'Value': 'true'},
                    {'Key': 'ShutdownTime', 'Value': shutdown_time},
                    {'Key': 'StartupTime', 'Value': startup_time},
                    {'Key': 'Timezone', 'Value': timezone}
                ]
            )

        print(f"Configured auto-shutdown for {len(instance_ids)} instances")
        print(f"Shutdown: {shutdown_time}, Startup: {startup_time} ({timezone})")

    def estimate_savings(self, instance_type, hourly_rate, hours_saved_per_day):
        """
        Estimate savings from auto-shutdown

        Example: Development instance running 12hr/day instead of 24hr/day
        """
        monthly_hours_saved = hours_saved_per_day * 30
        monthly_savings = hourly_rate * monthly_hours_saved
        annual_savings = monthly_savings * 12

        savings_pct = (hours_saved_per_day / 24) * 100

        return {
            'instance_type': instance_type,
            'hourly_rate': hourly_rate,
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'savings_pct': savings_pct
        }

# Example: p3.2xlarge running 12hr/day instead of 24hr/day
manager = AutoShutdownManager()
savings = manager.estimate_savings('p3.2xlarge', 3.06, hours_saved_per_day=12)
print(f"Annual savings: ${savings['annual_savings']:,.0f} ({savings['savings_pct']:.0f}%)")
# Output: Annual savings: $13,391 (50%)
```

---

## Spot and Preemptible Instances

Leverage spare capacity for massive savings on training workloads.

### Spot Instance Strategy

```python
"""
Spot Instance Best Practices for ML Training

Savings: 60-90% vs on-demand
Availability: 70-95% (varies by region/instance type)
Interruption: 2-minute warning

Best practices:
1. Checkpointing (save every epoch)
2. Flexible instance types
3. Multiple AZs
4. Spot Fleet with fallback
"""

import boto3

class SpotTrainingCluster:
    """
    Manage spot instances for distributed training

    Features:
    - Automatic checkpointing
    - Fallback to on-demand if spot unavailable
    - Multi-AZ for higher availability
    """

    def __init__(self):
        self.ec2_client = boto3.client('ec2')

    def create_spot_training_cluster(self, target_capacity=4, max_price=0.5):
        """
        Create spot fleet for training

        Args:
            target_capacity: Number of instances
            max_price: Max price per hour ($ per instance)

        Returns:
            Spot fleet request ID
        """
        # Spot fleet configuration
        spot_fleet_config = {
            'IamFleetRole': 'arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role',
            'AllocationStrategy': 'lowestPrice',
            'TargetCapacity': target_capacity,
            'SpotPrice': str(max_price),
            'LaunchSpecifications': [
                # Multiple instance types for flexibility
                {
                    'ImageId': 'ami-12345678',
                    'InstanceType': 'p3.2xlarge',
                    'KeyName': 'my-key',
                    'SpotPrice': str(max_price),
                    'SubnetId': 'subnet-1,subnet-2,subnet-3',  # Multi-AZ
                    'UserData': self._get_user_data_script(),
                    'TagSpecifications': [
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {'Key': 'Name', 'Value': 'spot-training'},
                                {'Key': 'Workload', 'Value': 'training'},
                                {'Key': 'SpotFleet', 'Value': 'true'}
                            ]
                        }
                    ]
                },
                # Fallback to smaller instance if p3 unavailable
                {
                    'ImageId': 'ami-12345678',
                    'InstanceType': 'p2.xlarge',
                    'KeyName': 'my-key',
                    'SpotPrice': str(max_price * 0.7),
                    'SubnetId': 'subnet-1,subnet-2,subnet-3',
                    'UserData': self._get_user_data_script()
                }
            ],
            'Type': 'maintain',  # Maintain target capacity
            'ReplaceUnhealthyInstances': True,
            'TerminateInstancesWithExpiration': True,
            'InstanceInterruptionBehavior': 'terminate'
        }

        # Request spot fleet
        response = self.ec2_client.request_spot_fleet(
            SpotFleetRequestConfig=spot_fleet_config
        )

        fleet_id = response['SpotFleetRequestId']
        print(f"Created spot fleet: {fleet_id}")

        return fleet_id

    def _get_user_data_script(self):
        """
        User data script with checkpointing and spot handling

        Monitors spot interruption warnings and saves checkpoint
        """
        script = """#!/bin/bash
# Setup training environment
cd /home/ubuntu/training

# Monitor spot interruption
(
  while true; do
    HTTP_CODE=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s -w %{http_code} -o /dev/null http://169.254.169.254/latest/meta-data/spot/instance-action)
    if [[ "$HTTP_CODE" -eq 200 ]]; then
      echo "Spot interruption detected, saving checkpoint..."
      touch /tmp/SPOT_INTERRUPTION
      # Training script will detect this and save checkpoint
      break
    fi
    sleep 5
  done
) &

# Start training with checkpointing
python train.py \\
  --checkpoint-dir s3://my-bucket/checkpoints \\
  --checkpoint-frequency 100 \\
  --resume-from-checkpoint latest
"""
        import base64
        return base64.b64encode(script.encode()).decode()

# Usage
cluster = SpotTrainingCluster()
fleet_id = cluster.create_spot_training_cluster(target_capacity=4, max_price=0.5)
```

### Checkpoint Management

```python
import torch
import os
from pathlib import Path

class CheckpointManager:
    """
    Robust checkpointing for spot instance training

    Features:
    - Automatic save on spot interruption
    - Resume from latest checkpoint
    - Save to S3 for durability
    """

    def __init__(self, checkpoint_dir, save_frequency=100):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency

    def save_checkpoint(self, epoch, model, optimizer, loss, filename='checkpoint.pth'):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        local_path = self.checkpoint_dir / filename
        torch.save(checkpoint, local_path)

        # Upload to S3
        self._upload_to_s3(local_path, f's3://my-bucket/checkpoints/{filename}')

        print(f"Checkpoint saved: epoch {epoch}, loss {loss:.4f}")

    def load_checkpoint(self, model, optimizer, filename='checkpoint.pth'):
        """Load latest checkpoint"""
        # Download from S3
        local_path = self.checkpoint_dir / filename
        self._download_from_s3(f's3://my-bucket/checkpoints/{filename}', local_path)

        if not local_path.exists():
            print("No checkpoint found, starting from scratch")
            return 0

        checkpoint = torch.load(local_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        print(f"Resumed from checkpoint: epoch {epoch}")
        return epoch

    def _upload_to_s3(self, local_path, s3_uri):
        """Upload file to S3"""
        import boto3
        s3_client = boto3.client('s3')
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        s3_client.upload_file(str(local_path), bucket, key)

    def _download_from_s3(self, s3_uri, local_path):
        """Download file from S3"""
        import boto3
        s3_client = boto3.client('s3')
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        try:
            s3_client.download_file(bucket, key, str(local_path))
        except:
            pass  # File doesn't exist yet

# Training loop with checkpointing
def train_with_spot(model, train_loader, epochs=100):
    checkpoint_mgr = CheckpointManager('/data/checkpoints', save_frequency=100)

    # Resume from checkpoint
    start_epoch = checkpoint_mgr.load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Check for spot interruption
            if os.path.exists('/tmp/SPOT_INTERRUPTION'):
                print("Spot interruption detected, saving final checkpoint...")
                checkpoint_mgr.save_checkpoint(epoch, model, optimizer, loss.item())
                return

            # Periodic checkpoint
            if batch_idx % checkpoint_mgr.save_frequency == 0:
                checkpoint_mgr.save_checkpoint(epoch, model, optimizer, loss.item())

        print(f"Epoch {epoch} completed")
```

---

## FinOps for ML Teams

Financial Operations (FinOps) practices for ML infrastructure.

### Cost Allocation and Showback

```python
"""
FinOps Maturity Model for ML Teams

Level 1: Crawl (Basic Visibility)
- Cost tracking by team/project
- Monthly reports
- Basic tagging

Level 2: Walk (Active Management)
- Real-time dashboards
- Budget alerts
- Showback to teams
- Basic optimization (right-sizing)

Level 3: Run (Optimization)
- Automated cost optimization
- Chargeback to teams
- Forecasting
- Reserved instance management
- Continuous optimization
"""

class MLFinOpsManager:
    """
    FinOps management for ML teams

    Features:
    - Cost allocation by team/project
    - Budget tracking and alerts
    - Showback reports
    - Optimization recommendations
    """

    def __init__(self):
        self.cost_tracker = MultiCloudCostTracker()

    def generate_team_report(self, team_name, month):
        """
        Generate monthly cost report for team

        Returns:
            Cost breakdown and recommendations
        """
        costs = self.cost_tracker.get_total_costs(days=30)
        team_costs = costs['details'][costs['details']['team'] == team_name]

        total_cost = team_costs['cost'].sum()

        # Breakdown by workload type
        training_cost = team_costs[team_costs['workload'] == 'training']['cost'].sum()
        inference_cost = team_costs[team_costs['workload'] == 'inference']['cost'].sum()
        storage_cost = team_costs[team_costs['workload'] == 'storage']['cost'].sum()

        report = {
            'team': team_name,
            'month': month,
            'total_cost': total_cost,
            'breakdown': {
                'training': training_cost,
                'inference': inference_cost,
                'storage': storage_cost
            },
            'per_project': team_costs.groupby('project')['cost'].sum().to_dict(),
            'recommendations': self._generate_recommendations(team_costs)
        }

        return report

    def _generate_recommendations(self, costs_df):
        """Generate cost optimization recommendations"""
        recommendations = []

        # Check for underutilized resources
        if costs_df['utilization'].mean() < 0.5:
            recommendations.append({
                'type': 'RIGHT_SIZE',
                'priority': 'HIGH',
                'message': 'Average utilization <50%, consider downsizing instances',
                'potential_savings': costs_df['cost'].sum() * 0.3
            })

        # Check for non-spot training
        training_costs = costs_df[costs_df['workload'] == 'training']
        if not training_costs.empty and (training_costs['spot'] == False).sum() > 0:
            recommendations.append({
                'type': 'USE_SPOT',
                'priority': 'MEDIUM',
                'message': 'Use spot instances for training workloads',
                'potential_savings': training_costs['cost'].sum() * 0.7
            })

        return recommendations

    def set_budget_alert(self, team_name, monthly_budget, threshold_pct=80):
        """
        Set budget alert for team

        Triggers alert when spending exceeds threshold
        """
        # Use CloudWatch or similar for monitoring
        print(f"Budget alert set for {team_name}: ${monthly_budget} ({threshold_pct}%)")

# Usage
finops = MLFinOpsManager()
report = finops.generate_team_report('ml-engineering', '2024-01')

print(f"Team: {report['team']}")
print(f"Total cost: ${report['total_cost']:,.2f}")
print(f"\nBreakdown:")
for workload, cost in report['breakdown'].items():
    print(f"  {workload}: ${cost:,.2f}")

print(f"\nRecommendations:")
for rec in report['recommendations']:
    print(f"  [{rec['priority']}] {rec['message']}")
    print(f"    Potential savings: ${rec['potential_savings']:,.2f}")
```

---

## Cloud-Agnostic Architecture

Build portable ML infrastructure across clouds.

### Abstraction Layer Pattern

```python
"""
Cloud-Agnostic ML Infrastructure

Goals:
- Portability across AWS, GCP, Azure
- Minimize vendor lock-in
- Consistent APIs

Approach:
- Abstraction layers for compute, storage, ML services
- Infrastructure as Code (Terraform)
- Containerization (Docker/Kubernetes)
- Standard ML frameworks (PyTorch, TensorFlow)
"""

from abc import ABC, abstractmethod

class CloudStorageProvider(ABC):
    """Abstract interface for cloud storage"""

    @abstractmethod
    def upload_file(self, local_path, remote_path):
        pass

    @abstractmethod
    def download_file(self, remote_path, local_path):
        pass

    @abstractmethod
    def list_files(self, prefix):
        pass

class AWSStorage(CloudStorageProvider):
    """AWS S3 implementation"""

    def __init__(self):
        import boto3
        self.s3_client = boto3.client('s3')

    def upload_file(self, local_path, remote_path):
        bucket, key = self._parse_path(remote_path)
        self.s3_client.upload_file(local_path, bucket, key)

    def download_file(self, remote_path, local_path):
        bucket, key = self._parse_path(remote_path)
        self.s3_client.download_file(bucket, key, local_path)

    def list_files(self, prefix):
        bucket, key = self._parse_path(prefix)
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=key)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def _parse_path(self, path):
        """Parse s3://bucket/key format"""
        path = path.replace('s3://', '')
        parts = path.split('/', 1)
        return parts[0], parts[1] if len(parts) > 1 else ''

class GCPStorage(CloudStorageProvider):
    """GCP Cloud Storage implementation"""

    def __init__(self):
        from google.cloud import storage
        self.client = storage.Client()

    def upload_file(self, local_path, remote_path):
        bucket_name, blob_name = self._parse_path(remote_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

    def download_file(self, remote_path, local_path):
        bucket_name, blob_name = self._parse_path(remote_path)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)

    def list_files(self, prefix):
        bucket_name, prefix_path = self._parse_path(prefix)
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix_path)
        return [blob.name for blob in blobs]

    def _parse_path(self, path):
        """Parse gs://bucket/key format"""
        path = path.replace('gs://', '')
        parts = path.split('/', 1)
        return parts[0], parts[1] if len(parts) > 1 else ''

class AzureStorage(CloudStorageProvider):
    """Azure Blob Storage implementation"""

    def __init__(self):
        from azure.storage.blob import BlobServiceClient
        self.client = BlobServiceClient.from_connection_string(
            os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        )

    def upload_file(self, local_path, remote_path):
        container, blob_name = self._parse_path(remote_path)
        blob_client = self.client.get_blob_client(container, blob_name)
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_file(self, remote_path, local_path):
        container, blob_name = self._parse_path(remote_path)
        blob_client = self.client.get_blob_client(container, blob_name)
        with open(local_path, 'wb') as f:
            blob_client.download_blob().readinto(f)

    def list_files(self, prefix):
        container, prefix_path = self._parse_path(prefix)
        container_client = self.client.get_container_client(container)
        blobs = container_client.list_blobs(name_starts_with=prefix_path)
        return [blob.name for blob in blobs]

    def _parse_path(self, path):
        """Parse azure://container/blob format"""
        path = path.replace('azure://', '').replace('https://', '')
        parts = path.split('/', 1)
        return parts[0], parts[1] if len(parts) > 1 else ''

# Factory pattern for cloud-agnostic code
class StorageFactory:
    """Create storage provider based on URI scheme"""

    @staticmethod
    def create(uri):
        if uri.startswith('s3://'):
            return AWSStorage()
        elif uri.startswith('gs://'):
            return GCPStorage()
        elif uri.startswith('azure://'):
            return AzureStorage()
        else:
            raise ValueError(f"Unsupported storage URI: {uri}")

# Usage: Cloud-agnostic code
def upload_model(local_path, remote_path):
    """Upload model to any cloud provider"""
    storage = StorageFactory.create(remote_path)
    storage.upload_file(local_path, remote_path)
    print(f"Uploaded {local_path} to {remote_path}")

# Works with any cloud
upload_model('./model.pth', 's3://my-bucket/models/model.pth')      # AWS
upload_model('./model.pth', 'gs://my-bucket/models/model.pth')      # GCP
upload_model('./model.pth', 'azure://container/models/model.pth')   # Azure
```

---

## Migration Between Clouds

Strategies for migrating ML workloads between cloud providers.

### Migration Checklist

```markdown
## Cloud Migration Checklist

### Phase 1: Assessment (1-2 weeks)
- [ ] Inventory current resources
- [ ] Document dependencies
- [ ] Identify data volumes
- [ ] Estimate migration costs
- [ ] Define success criteria

### Phase 2: Planning (2-4 weeks)
- [ ] Choose migration strategy (lift-shift, refactor, rebuild)
- [ ] Design target architecture
- [ ] Plan data migration approach
- [ ] Identify risks and mitigation
- [ ] Create rollback plan

### Phase 3: Data Migration (Varies)
- [ ] Set up data transfer (AWS DataSync, Transfer Appliance)
- [ ] Migrate datasets incrementally
- [ ] Validate data integrity
- [ ] Replicate data for cutover

### Phase 4: Application Migration (2-6 weeks)
- [ ] Containerize applications
- [ ] Update cloud-specific code
- [ ] Migrate models to new registry
- [ ] Update endpoints
- [ ] Test thoroughly

### Phase 5: Cutover (1 week)
- [ ] Final data sync
- [ ] DNS/traffic cutover
- [ ] Monitor closely
- [ ] Validate functionality
- [ ] Decommission old infrastructure

Estimated timeline: 2-3 months for medium-sized ML platform
```

---

## Summary

In this lesson, you learned:

✅ Design multi-cloud strategies (active-passive, best-of-breed, geographic)
✅ Implement cost monitoring across clouds (tagging, dashboards)
✅ Optimize costs (right-sizing 40%, RIs 55%, auto-shutdown 50%)
✅ Use spot instances (60-90% savings with checkpointing)
✅ Apply FinOps practices (showback, budgets, recommendations)
✅ Build cloud-agnostic architectures (abstraction layers)
✅ Migrate workloads between clouds
✅ Forecast and budget accurately

**Key Takeaways**:
- **Right-sizing** saves 30-40% on compute
- **Reserved Instances** save 40-60% for steady workloads
- **Spot instances** save 60-90% for training (with checkpointing)
- **Auto-shutdown** saves 50-75% on dev instances
- **Combined** optimizations can reduce costs by **70-80%**

**Cost Optimization ROI**:
```
Before optimization: $10,000/month
After optimization:
- Right-sizing: -$3,000 (30%)
- Reserved Instances: -$2,000 (20%)
- Spot instances: -$1,500 (15%)
- Auto-shutdown: -$500 (5%)
Total savings: $7,000/month → $3,000/month (70% reduction!)
```

**Next Steps**:
- Complete hands-on exercise
- Implement cost tracking for your project
- Apply optimization techniques
- **Congratulations! Module 02 complete!**

---

**Estimated Time to Complete**: 6 hours (including hands-on exercise)
**Difficulty**: Advanced
**Module Complete**: Ready for Module 03: Kubernetes Deep Dive

---

## Module 02 Completed! 🎉

You've mastered:
- Cloud architecture patterns
- AWS, GCP, and Azure ML infrastructure
- Cloud storage strategies
- Cloud networking
- Managed ML services
- Multi-cloud and cost optimization

**Total Module Time**: 50 hours
**Projects Completed**: 1 (Basic Model Serving)
**Quizzes**: 1 (30 questions)
**Exercises**: 3 (Environment, Docker, Kubernetes)
