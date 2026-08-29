# Lesson 08: Multi-Cloud & Cost Optimization

## Lesson Overview

Lessons 02-07 built and compared ML infrastructure on a single cloud at a time. This closing lesson of the module asks two different questions: when does it make sense to spread that infrastructure across *multiple* clouds, and — regardless of how many clouds you use — how do you keep the bill under control as usage grows? The two are related: multi-cloud adds cost-tracking complexity (now you're reading three billing APIs instead of one), but a deliberate multi-cloud strategy can itself be a cost lever, not just a resilience one.

By the end of this lesson you will be able to choose an appropriate multi-cloud strategy for a given organization, track and allocate cost across providers, apply the standard optimization levers (right-sizing, reservations, spot/preemptible instances, auto-shutdown), run basic FinOps practices for an ML team, build cloud-agnostic abstractions where portability matters, and plan a migration between providers.

---

## Table of Contents

1. [Multi-Cloud Strategies](#1-multi-cloud-strategies)
2. [Cost Monitoring and Tracking](#2-cost-monitoring-and-tracking)
3. [Cost Optimization Techniques](#3-cost-optimization-techniques)
4. [Spot and Preemptible Instances](#4-spot-and-preemptible-instances)
5. [FinOps for ML Teams](#5-finops-for-ml-teams)
6. [Cloud-Agnostic Architecture](#6-cloud-agnostic-architecture)
7. [Migration Between Clouds](#7-migration-between-clouds)
8. [Budgeting and Forecasting](#8-budgeting-and-forecasting)
9. [Best Practices](#9-best-practices)
10. [Putting It All Together: A Cost Optimization Rollout](#10-putting-it-all-together-a-cost-optimization-rollout)
11. [Key Takeaways](#11-key-takeaways)
12. [What's Next?](#whats-next)
13. [Further Reading](#further-reading)

---

## 1. Multi-Cloud Strategies

Multi-cloud is a deliberate architectural choice, not a default — it trades operational simplicity for one of three specific benefits, and the pattern you pick should map directly to which benefit you're actually after:

| Pattern | Structure | Use Case | Complexity | Cost |
|---|---|---|---|---|
| Active-Passive (DR) | AWS primary (training, inference, storage); Azure cold standby (data replication only) | Risk mitigation, compliance | Low | +10-20% overhead |
| Best-of-Breed | AWS (data lake, edge); GCP (TPU training, AutoML, BigQuery ML); Azure (enterprise, OpenAI, compliance) | Leverage each cloud's unique strength | High | Optimized per workload |
| Geographic Distribution | AWS (US customers); GCP (EU, GDPR); Azure (APAC, latency) | Global reach, data sovereignty | Medium | Higher — cross-region data transfer |

### 1.1 Which Pattern for Which Organization

| Scenario | Recommended | Reasoning |
|---|---|---|
| Startup (<50 people) | Single cloud | Minimize operational complexity |
| Growth stage (50-200) | Single cloud + DR | Risk mitigation without full multi-cloud overhead |
| Enterprise (>200) | Multi-cloud | Avoid vendor lock-in |
| AI-first company | GCP + AWS | TPU access plus ecosystem breadth |
| Microsoft shop | Azure + backup | Enterprise features, minimal added complexity |
| Global company | Multi-cloud (geographic) | Data residency requirements |
| Cost-sensitive at scale | Multi-cloud (best-of-breed) | Price arbitrage between providers |
| High compliance | Multi-cloud (active-passive) | Redundancy across independent providers |

The through-line: multi-cloud is worth its complexity tax once you have a *specific* reason (compliance, a hardware advantage, geographic requirements) — "just in case" is rarely reason enough to justify running three billing accounts and three sets of IAM policies.

---

## 2. Cost Monitoring and Tracking

Cost optimization is impossible without visibility first — you can't right-size what you can't attribute to a team or project.

### 2.1 Tagging Strategy

A small set of mandatory tags makes every dollar attributable: `Environment` (dev/staging/production), `Team`, `Project`, `CostCenter`, `Owner`. Optional tags (`Experiment`, `Model`, `Stage`) add finer-grained breakdowns for ML-specific reporting.

```python
import boto3

def tag_ml_resources(bucket_name, tags):
    required = {'Environment', 'Team', 'Project', 'CostCenter', 'Owner'}
    if not required.issubset(tags):
        raise ValueError(f"Missing required tags: {required - tags.keys()}")
    boto3.client('s3').put_bucket_tagging(
        Bucket=bucket_name, Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in tags.items()]})

tag_ml_resources('ml-training-data', {
    'Environment': 'production', 'Team': 'ml-engineering', 'Project': 'image-classification',
    'CostCenter': 'engineering', 'Owner': 'ml-team@company.com', 'Stage': 'training',
})
```

### 2.2 A Unified Multi-Cloud Cost View

Each cloud exposes cost data through its own API (AWS Cost Explorer, GCP Billing, Azure Cost Management) — grouped consistently by the tags above, so they merge into one dataframe:

```python
import boto3, pandas as pd
from datetime import datetime, timedelta

def get_aws_costs(days=30):
    ce = boto3.client('ce')
    start, end = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')
    resp = ce.get_cost_and_usage(
        TimePeriod={'Start': start, 'End': end}, Granularity='DAILY', Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'TAG', 'Key': 'Project'}, {'Type': 'TAG', 'Key': 'Team'}],
        Filter={'Tags': {'Key': 'Environment', 'Values': ['production']}},
    )
    return pd.DataFrame([
        {'date': r['TimePeriod']['Start'], 'cloud': 'AWS', 'project': g['Keys'][0], 'team': g['Keys'][1],
         'cost': float(g['Metrics']['UnblendedCost']['Amount'])}
        for r in resp['ResultsByTime'] for g in r['Groups']
    ])

# get_gcp_costs() / get_azure_costs() follow the same shape against their own billing APIs
all_costs = pd.concat([get_aws_costs(), get_gcp_costs(), get_azure_costs()], ignore_index=True)
print(f"Total: ${all_costs['cost'].sum():,.2f}")
print(all_costs.groupby('cloud')['cost'].sum())
print(all_costs.groupby('project')['cost'].sum().sort_values(ascending=False).head())
```

---

## 3. Cost Optimization Techniques

Four levers cover most of the achievable savings, and they compound rather than compete — most teams apply all four.

### 3.1 Right-Sizing

Compare actual CPU utilization against the provisioned instance size: consistently low utilization (<30% average, <60% peak) means the instance is oversized; the reverse means it's undersized and risking throttling.

```python
import boto3
from datetime import datetime, timedelta

def rightsizing_recommendation(instance_id, days=7):
    cw = boto3.client('cloudwatch')
    stats = cw.get_metric_statistics(
        Namespace='AWS/EC2', MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=datetime.utcnow() - timedelta(days=days), EndTime=datetime.utcnow(),
        Period=3600, Statistics=['Average', 'Maximum'],
    )['Datapoints']
    if not stats:
        return {'recommendation': 'Insufficient data'}

    avg_cpu = sum(d['Average'] for d in stats) / len(stats)
    max_cpu = max(d['Maximum'] for d in stats)

    if avg_cpu < 30 and max_cpu < 60:
        return {'recommendation': 'DOWNSIZE', 'potential_savings_pct': 40}
    if avg_cpu > 80 or max_cpu > 95:
        return {'recommendation': 'UPSIZE', 'potential_savings_pct': 0}
    return {'recommendation': 'OPTIMIZED', 'potential_savings_pct': 0}
```

### 3.2 Reserved Instances vs. Savings Plans

| Option | Commitment | Savings | Flexibility | Best For |
|---|---|---|---|---|
| Reserved Instances | 1-3 years | 40-60% | Low — locked to one instance type | Steady-state workloads (inference) |
| Savings Plans | 1-3 years | 40-60% | High — any instance in the family | Dynamic workloads (varied training) |
| On-Demand/Spot | None | 0% / 60-90% | Highest | Experimentation / interruptible batch |

```python
PRICING = {'p3.2xlarge': {'on_demand': 3.06, 'ri_1yr': 2.08, 'ri_3yr': 1.37}}

def ri_savings(instance_type, hours_per_month, years=3):
    p = PRICING[instance_type]
    total_hours = hours_per_month * 12 * years
    ri_rate = p['ri_3yr'] if years >= 3 else p['ri_1yr']
    return {'savings': (p['on_demand'] - ri_rate) * total_hours,
            'savings_pct': (1 - ri_rate / p['on_demand']) * 100}

# A 24/7 inference server on p3.2xlarge, 3-year commitment
print(ri_savings('p3.2xlarge', hours_per_month=730))  # ~55% savings
```

### 3.3 Auto-Shutdown for Non-Production Instances

Development and training instances rarely need to run 24/7 — tagging them for a shutdown/startup schedule (via AWS Instance Scheduler, or an equivalent Lambda/cron) captures savings proportional to the hours reclaimed:

```python
def estimate_shutdown_savings(hourly_rate, hours_saved_per_day):
    monthly = hourly_rate * hours_saved_per_day * 30
    return {'monthly_savings': monthly, 'annual_savings': monthly * 12,
            'savings_pct': hours_saved_per_day / 24 * 100}

# p3.2xlarge running 12hr/day instead of 24hr/day
print(estimate_shutdown_savings(3.06, hours_saved_per_day=12))  # $13,391/yr, 50%
```

---

## 4. Spot and Preemptible Instances

Spot/preemptible capacity offers the single largest discount available (60-90% off on-demand) in exchange for a ~2-minute eviction warning — the same tradeoff covered in Lessons 02-04's per-cloud sections, generalized here across providers.

### 4.1 Spot Fleet with Fallback

A resilient spot setup spans multiple instance types and AZs, so losing capacity in one doesn't stall the whole job, and falls back toward smaller/cheaper instance types rather than failing outright:

```python
import boto3

def create_spot_training_fleet(target_capacity=4, max_price=0.5):
    ec2 = boto3.client('ec2')
    config = {
        'IamFleetRole': 'arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role',
        'AllocationStrategy': 'lowestPrice', 'TargetCapacity': target_capacity,
        'Type': 'maintain', 'ReplaceUnhealthyInstances': True,
        'LaunchSpecifications': [
            {'ImageId': 'ami-12345678', 'InstanceType': 'p3.2xlarge', 'SpotPrice': str(max_price),
             'SubnetId': 'subnet-1,subnet-2,subnet-3'},  # multi-AZ
            {'ImageId': 'ami-12345678', 'InstanceType': 'p2.xlarge', 'SpotPrice': str(max_price * 0.7),
             'SubnetId': 'subnet-1,subnet-2,subnet-3'},  # fallback if p3 capacity is unavailable
        ],
    }
    return ec2.request_spot_fleet(SpotFleetRequestConfig=config)['SpotFleetRequestId']
```

### 4.2 Checkpointing Against Eviction

The pattern is identical to the Spot VM checkpointing covered in Lessons 02-04: poll the instance metadata endpoint for the eviction notice, and save state the moment it appears rather than waiting for the 2-minute window to run out.

```python
import torch, os

def train_with_checkpointing(model, optimizer, train_loader, epochs, checkpoint_dir, s3_client):
    start_epoch = 0
    if os.path.exists(f'{checkpoint_dir}/checkpoint.pth'):
        ckpt = torch.load(f'{checkpoint_dir}/checkpoint.pth')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']

    for epoch in range(start_epoch, epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = train_step(model, optimizer, data, target)
            if os.path.exists('/tmp/SPOT_INTERRUPTION') or batch_idx % 100 == 0:
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/checkpoint.pth')
                s3_client.upload_file(f'{checkpoint_dir}/checkpoint.pth', 'my-bucket', 'checkpoints/checkpoint.pth')
                if os.path.exists('/tmp/SPOT_INTERRUPTION'):
                    return  # let the fleet replace this instance; it resumes from this checkpoint
```

---

## 5. FinOps for ML Teams

FinOps is the practice of making cost a shared, continuously-managed responsibility rather than something finance discovers at month-end. Maturity typically builds in three stages:

| Stage | Capabilities |
|---|---|
| Crawl (visibility) | Cost tracking by team/project, monthly reports, basic tagging |
| Walk (active management) | Real-time dashboards, budget alerts, showback to teams, right-sizing |
| Run (optimization) | Automated optimization, chargeback, forecasting, continuous RI/Savings Plan management |

### 5.1 Showback Reports and Recommendations

A showback report ties spend back to a team and flags the two most common findings automatically — low utilization and non-spot training:

```python
def generate_team_report(costs_df, team_name):
    team_costs = costs_df[costs_df['team'] == team_name]
    recommendations = []

    if team_costs['utilization'].mean() < 0.5:
        recommendations.append(('RIGHT_SIZE', 'Average utilization <50%', team_costs['cost'].sum() * 0.3))

    training = team_costs[team_costs['workload'] == 'training']
    if not training.empty and (~training['spot']).any():
        recommendations.append(('USE_SPOT', 'Training running on-demand', training['cost'].sum() * 0.7))

    return {'team': team_name, 'total_cost': team_costs['cost'].sum(),
            'by_workload': team_costs.groupby('workload')['cost'].sum().to_dict(),
            'recommendations': recommendations}
```

---

## 6. Cloud-Agnostic Architecture

Portability is worth building deliberately only when you actually expect to move or run across clouds — Section 1 covers when that's true. The standard approach is a thin interface per capability (storage, compute, ML services) with one implementation per provider, selected by a factory at runtime:

```python
from abc import ABC, abstractmethod

class CloudStorageProvider(ABC):
    @abstractmethod
    def upload_file(self, local_path, remote_path): ...
    @abstractmethod
    def download_file(self, remote_path, local_path): ...

class AWSStorage(CloudStorageProvider):
    def __init__(self):
        import boto3
        self.client = boto3.client('s3')

    def upload_file(self, local_path, remote_path):
        bucket, key = remote_path.replace('s3://', '').split('/', 1)
        self.client.upload_file(local_path, bucket, key)

    def download_file(self, remote_path, local_path):
        bucket, key = remote_path.replace('s3://', '').split('/', 1)
        self.client.download_file(bucket, key, local_path)

# GCPStorage / AzureStorage implement the same interface against gs:// / azure:// URIs

class StorageFactory:
    @staticmethod
    def create(uri):
        return {'s3://': AWSStorage, 'gs://': GCPStorage, 'azure://': AzureStorage}[
            next(p for p in ('s3://', 'gs://', 'azure://') if uri.startswith(p))]()

def upload_model(local_path, remote_path):
    StorageFactory.create(remote_path).upload_file(local_path, remote_path)

upload_model('./model.pth', 's3://my-bucket/models/model.pth')     # AWS
upload_model('./model.pth', 'gs://my-bucket/models/model.pth')     # GCP
upload_model('./model.pth', 'azure://container/models/model.pth')  # Azure
```

The cost of this abstraction is real — three implementations to maintain instead of one, and it can only expose the lowest common denominator of features across providers. Reach for it when portability is a stated requirement, not preemptively for every project.

---

## 7. Migration Between Clouds

A cloud migration is a project with real phases and real lead time, not a weekend task — treat it accordingly:

1. **Assessment (1-2 weeks):** inventory current resources, document dependencies, size the data volume, estimate cost, define success criteria.
2. **Planning (2-4 weeks):** pick a strategy (lift-and-shift, refactor, or rebuild), design the target architecture, plan the data-migration approach, write a rollback plan.
3. **Data migration (varies with volume):** set up transfer (DataSync, a Transfer Appliance, or the destination cloud's equivalent), migrate incrementally, validate integrity before cutover.
4. **Application migration (2-6 weeks):** containerize if not already, update cloud-specific code (Lesson 07 §7.1 covers this for the managed ML platforms specifically), migrate the model registry, re-test thoroughly.
5. **Cutover (1 week):** final sync, DNS/traffic switch, close monitoring, decommission the old infrastructure only once the new one is validated.

**Realistic timeline for a medium-sized ML platform: 2-3 months.** The data migration and application-code phases are almost always where estimates go wrong — pad both rather than the assessment phase.

---

## 8. Budgeting and Forecasting

Cost tracking (Section 2) tells you what happened; budgeting and forecasting are about catching a problem *before* the bill arrives.

### 8.1 Setting Budget Alerts

Every provider supports the same shape: a monthly amount, a percentage threshold, and a notification target. AWS Budgets is representative:

```python
import boto3

boto3.client('budgets').create_budget(
    AccountId='123456789012',
    Budget={'BudgetName': 'ml-engineering-monthly', 'BudgetLimit': {'Amount': '10000', 'Unit': 'USD'},
            'TimeUnit': 'MONTHLY', 'BudgetType': 'COST'},
    NotificationsWithSubscribers=[{
        'Notification': {'NotificationType': 'ACTUAL', 'ComparisonOperator': 'GREATER_THAN', 'Threshold': 80},
        'Subscribers': [{'SubscriptionType': 'EMAIL', 'Address': 'ml-team@company.com'}],
    }],
)
```

Set at least two thresholds per budget (e.g. 80% as an early warning, 100% as an escalation) — a single alert at 100% gives no time to react before the month closes.

### 8.2 Trend-Based Forecasting

The simplest useful forecast is a linear trend over the last few months of actuals — good enough to catch a runaway growth trajectory long before it needs anything more sophisticated:

```python
import numpy as np

def forecast_next_month(monthly_costs):
    months = np.arange(len(monthly_costs))
    slope, intercept = np.polyfit(months, monthly_costs, 1)
    return slope * len(monthly_costs) + intercept

# Last 6 months of actuals -> next month's projection
forecast = forecast_next_month([8200, 8600, 9100, 9400, 9800, 10200])
print(f"Projected next month: ${forecast:,.0f}")  # trend continuing upward -> budget conversation now, not next month
```

A forecast that's trending toward a budget breach is the trigger for the optimization pass in Sections 3-4, applied *before* the overage happens rather than after.

---

## 9. Best Practices

**Tag everything before optimizing anything.** Right-sizing and RI purchases are only as good as the cost attribution behind them — untagged spend can't be assigned to a team, which means it never gets optimized by anyone.

**Default new non-production compute to auto-shutdown and Spot.** These are the two highest-savings, lowest-effort levers (Sections 3.3 and 4) — make them the default rather than an opt-in a team has to remember.

**Buy reservations only for measured steady-state load.** Commit to Reserved Instances/Savings Plans after right-sizing (Section 3.1) has already stabilized the workload's baseline — reserving before right-sizing locks in the wrong size.

**Review budgets and forecasts on a fixed cadence, not reactively.** A monthly FinOps review (Section 5) that checks actuals against forecast (Section 8.2) catches drift while it's still cheap to correct.

**Build cloud-agnostic abstractions only when portability is a real requirement.** The abstraction cost in Section 6 is worth paying for an active multi-cloud strategy (Section 1) or a planned migration (Section 7) — not as insurance against a hypothetical future move.

**Treat a cloud migration as a scoped project with a rollback plan**, not an incremental drift — Section 7's phased approach exists specifically to avoid a half-migrated state with no clear path forward.

---

## 10. Putting It All Together: A Cost Optimization Rollout

**Scenario:** An ML team is spending $10,000/month on a single cloud with no tagging, no reservations, and all training running on-demand. Bring that down using the levers from this lesson, in the order that actually matters.

```python
# 1. Tag everything first (Section 2.1) — nothing downstream works without attribution
tag_ml_resources('ml-training-data', {'Environment': 'production', 'Team': 'ml-engineering',
                                       'Project': 'core-models', 'CostCenter': 'engineering',
                                       'Owner': 'ml-team@company.com'})

# 2. Right-size against 7 days of real utilization (Section 3.1)
for instance_id in production_instance_ids:
    rec = rightsizing_recommendation(instance_id)
    if rec['recommendation'] == 'DOWNSIZE':
        apply_recommended_size(instance_id)  # -> ~30% off compute

# 3. Move interruptible training to Spot with checkpointing (Section 4)
fleet_id = create_spot_training_fleet(target_capacity=4, max_price=0.5)  # -> ~70% off training compute

# 4. Buy Reserved Instances/Savings Plans against the now-stable steady-state baseline (Section 3.2)
# purchased via console/CLI once utilization has been right-sized -> ~20% off remaining steady load

# 5. Auto-shutdown all dev/experimentation instances outside business hours (Section 3.3)
# -> ~5% off overall, concentrated entirely on non-production spend

# 6. Set budget alerts and a monthly forecast review going forward (Section 8)
```

**Result, applied in this order:** right-sizing -$3,000 (30%), Reserved Instances -$2,000 (20%), Spot instances -$1,500 (15%), auto-shutdown -$500 (5%) — **$10,000/month → $3,000/month, a 70% reduction**, with budget alerts now in place to catch any regression before it compounds for a full month.

---

## 11. Key Takeaways

1. **Multi-cloud is a deliberate tradeoff, not a default** — pick active-passive, best-of-breed, or geographic distribution based on a specific need (DR, unique hardware, data residency), not as general-purpose insurance.
2. **Tagging is the prerequisite for every other optimization** — untagged spend can't be attributed, and unattributed spend never gets optimized.
3. **Right-sizing, Reserved Instances, Spot, and auto-shutdown compound** — applied together they routinely cut 70-80% off an unoptimized bill, in that rough order of application.
4. **Buy reservations after right-sizing, not before** — committing to a size before confirming it's correct locks in the wrong number for 1-3 years.
5. **FinOps maturity is a ladder (crawl → walk → run)** — visibility has to exist before showback is meaningful, and showback has to exist before automated optimization is trustworthy.
6. **Cloud-agnostic abstractions cost real engineering effort** — worth it for an active multi-cloud strategy or a planned migration, not as blanket future-proofing.
7. **A migration is a phased project with real lead time (2-3 months for a medium platform)** — the data and application-migration phases are where naive estimates go wrong.
8. **A budget alert without a forecast only tells you after the fact** — pairing threshold alerts with trend forecasting (Section 8.2) catches a cost trajectory problem while there's still time to act on it.

---

## What's Next?

This closes **Module 02: Cloud Computing for ML**. Module 03 moves one layer down the stack into **Kubernetes**, covering the orchestration internals (scheduling, networking, storage, operators) that AKS/EKS/GKE — used throughout this module for model serving — are built on top of.

---

## Further Reading

- **AWS Cost Explorer & Budgets**: https://aws.amazon.com/aws-cost-management/
- **GCP Billing and Cost Management**: https://cloud.google.com/billing/docs
- **Azure Cost Management**: https://learn.microsoft.com/azure/cost-management-billing/
- **FinOps Foundation**: https://www.finops.org/
- **Terraform Multi-Cloud Provisioning**: https://developer.hashicorp.com/terraform/language/providers

---

**Estimated Time to Complete**: 6 hours (including hands-on exercise)
**Difficulty**: Advanced
**Module Complete**: Ready for Module 03: Kubernetes Deep Dive

---

## Module 02 Completed!

You've covered:
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
