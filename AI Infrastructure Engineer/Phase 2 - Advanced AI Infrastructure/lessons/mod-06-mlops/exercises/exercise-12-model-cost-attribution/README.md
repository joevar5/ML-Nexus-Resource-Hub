# Exercise 12: Per-Model Cost Attribution

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 02 + mod-02 exercise 08

## Objective

Build a per-model cost attribution system that tells you, daily, "model X cost $Y to train and $Z to serve." Surface in a dashboard, set per-model budgets, alert on overruns.

## Why this matters

When your team owns 20 models and the cloud bill spikes, "which model is responsible?" should take 30 seconds, not 3 days of forensics. Attribution is a one-time-build, ongoing-value system.

## Requirements

1. **Tags**: every Kubernetes resource (training Job, serving Deployment) labeled with `model_name`.
2. **Training cost**: GPU/CPU time × instance hourly rate.
3. **Serving cost**: replica count × instance hourly rate × uptime + per-request cost (if managed service).
4. **Storage cost**: model artifact size × storage class price.
5. **Daily Parquet**: `s3://finops/per-model-daily/<date>.parquet`.
6. **Dashboard**: per-model spend over time, top spenders this month.
7. **Budget alerts**: per model, send Slack when daily spend > budget × 1.2.

## Step-by-step

### Step 1 — Tagging (30 min)
Standardize labels across deployments:
```yaml
metadata:
  labels:
    model: recs-ranker
    model_version: v3.2
    team: ml-platform
    cost_center: rec
```
Enforce via Kyverno (per mod-03 ex-10):
```yaml
kind: ClusterPolicy
metadata: { name: require-model-label }
spec:
  rules:
    - name: model-label
      match: { resources: { kinds: [Deployment, Job] } }
      validate:
        message: "Deployment must have `model` label"
        pattern: { metadata: { labels: { model: "?*" } } }
```

### Step 2 — Cost collection (60 min)
For each model on each day:
```python
# collect_costs.py
import boto3, pandas as pd
from kubernetes import client, config
from datetime import datetime, timedelta

config.load_kube_config()
k8s = client.CoreV1Api()
metrics = client.CustomObjectsApi()

INSTANCE_HOURLY = {"m6i.large": 0.096, "g4dn.xlarge": 0.526, ...}

def daily_costs(date: str):
    rows = []
    # Get pod-level resource usage from Prometheus
    # Tag each by model from labels
    
    # Serving cost: replicas × instance_type × 24 hours
    for deploy in apps.list_deployment_for_all_namespaces().items:
        model = deploy.metadata.labels.get("model")
        if not model: continue
        instance = deploy.spec.template.spec.node_selector.get("instance-type", "m6i.large")
        cost = deploy.spec.replicas * INSTANCE_HOURLY[instance] * 24
        rows.append({"date": date, "model": model, "kind": "serving", "cost": cost})
    
    # Training cost: completed Jobs in the date range, duration × instance type
    for job in batch.list_job_for_all_namespaces(label_selector="model").items:
        if job.status.completion_time and job.status.completion_time.date() == date:
            duration_h = (job.status.completion_time - job.status.start_time).seconds / 3600
            model = job.metadata.labels["model"]
            instance = job.spec.template.spec.node_selector.get("instance-type", "m6i.large")
            cost = duration_h * INSTANCE_HOURLY[instance]
            rows.append({"date": date, "model": model, "kind": "training", "cost": cost})
    
    # S3 model artifact storage
    s3 = boto3.client('s3')
    for obj in s3.list_objects_v2(Bucket="model-artifacts").get("Contents", []):
        model = obj["Key"].split("/")[0]  # convention: <model>/<version>/model.joblib
        size_gb = obj["Size"] / 1e9
        cost = size_gb * 0.023        # S3 Standard
        rows.append({"date": date, "model": model, "kind": "storage", "cost": cost})
    
    return pd.DataFrame(rows)

df = daily_costs(date="2026-05-22")
df.to_parquet("s3://finops/per-model-daily/2026-05-22.parquet")
```

### Step 3 — Per-prediction cost (15 min)
For LLM/managed serving:
```python
TOKEN_COST = {"gpt-4o-mini": (0.00015, 0.0006), ...}  # ($/1K prompt, $/1K completion)

def prediction_cost(model, usage):
    p, c = TOKEN_COST[model]
    return (usage.prompt_tokens / 1000 * p) + (usage.completion_tokens / 1000 * c)
```
Aggregate to daily.

### Step 4 — Daily Airflow DAG (30 min)
```python
with DAG("collect_per_model_costs", schedule="0 6 * * *", ...) as dag:
    @task
    def collect(ds: str): df = daily_costs(ds); df.to_parquet(f"s3://...{ds}.parquet")
    @task
    def alert_budgets(ds: str):
        df = pd.read_parquet(f"s3://...{ds}.parquet")
        budgets = yaml.safe_load(open("budgets.yaml"))
        for model, group in df.groupby("model"):
            spend = group.cost.sum()
            if spend > budgets.get(model, float("inf")) * 1.2:
                slack(f"{model} spent ${spend:.2f} (budget ${budgets[model]})")
    collect("{{ ds }}") >> alert_budgets("{{ ds }}")
```

### Step 5 — Dashboard (30 min)
Grafana via Athena (or just DuckDB on the Parquet):
- Per-model daily cost (stacked area)
- Top 10 spenders this month
- Cost per 1k predictions (for serving models)
- Budget vs actual gauge per model

### Step 6 — Validation (15 min)
Cross-check against AWS Cost Explorer monthly totals; should agree within 5%.

## Deliverables

1. Tagging policy + Kyverno enforcement.
2. `collect_costs.py` collection script.
3. Airflow DAG running daily.
4. `budgets.yaml` with at least 5 model budgets.
5. Grafana dashboard.
6. `INVESTIGATION_GUIDE.md`: how to use the dashboard during a cost spike.

## Validation

- [ ] All production resources have `model` label.
- [ ] Daily Parquet generated.
- [ ] Total matches AWS Cost Explorer ± 5%.
- [ ] Budget alert fires when synthetically exceeded.

## Stretch goals

- Add **forecast**: project monthly spend per model based on past 14 days.
- Add **per-team aggregation** (sum models by team).
- Integrate with **engineering-finance OKRs**: chargeback per team to discourage waste.

## Common pitfalls

- **Untagged resources** — Default to "untagged" bucket; surface as actionable in dashboard.
- **Shared infra (Postgres, Redis)** — Pro-rate by usage (e.g., Redis memory consumed by model's keys).
- **Spot price variability** — Per-day cost varies; use weighted average per period.
- **Cost lag** — AWS Cost Explorer is ~24h delayed; daily script catches up retroactively.
