# Exercise 08: FinOps Automation for ML Infrastructure

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 01-07; AWS Cost Explorer access

## Objective

Build a `mlfinops` CLI that collects daily cloud spend per team/workload, attributes idle GPU/VM cost, surfaces top waste, and sends a weekly digest to Slack. By the end you'll have an opinionated tool that turns "where did our cloud bill go?" from a quarterly fire drill into a Monday morning email.

## Why this matters

ML workloads are unusually expensive: GPUs are the dominant line item, training jobs run for hours, and idle clusters silently burn dollars. Engineers who can answer "what did training run X cost?" and "what's our top waste category this week?" become unusually valuable, fast.

## Requirements

Build `mlfinops` as a Python tool with these commands:

```text
mlfinops collect    # pull yesterday's data from Cost Explorer/BigQuery Billing/etc.
mlfinops report     # weekly digest by team, workload, and waste category
mlfinops digest     # post the report to Slack
mlfinops idle       # find idle resources (GPU < 5% util for 7 days, etc.)
mlfinops budgets    # compare actual vs budget per team
```

### Required attribution dimensions

Each dollar must be attributable to:
1. **Team** (from resource tag `team`)
2. **Workload** (training / inference / batch / dev / shared)
3. **Resource type** (EC2/GPU, S3, network egress, managed-service)
4. **Lifecycle** (active / idle / abandoned)

### Required outputs

- Daily Parquet to S3: `s3://team-finops/daily/<date>.parquet`
- Slack weekly digest: top 3 spenders, top 3 waste items, week-over-week delta
- Grafana data source: query the daily Parquet via Athena, render trend charts

### Required policies enforced

- **Required tags** check: any resource without `team` and `workload` tags is flagged in the report.
- **GPU idle threshold**: any GPU with <5% utilization for 7 days is flagged.
- **Untouched stopped instances**: stopped EC2 > 30 days is flagged (you're still paying for the EBS volume).
- **Old snapshots**: EBS snapshots > 90 days flagged.

## Step-by-step

### Step 1 — Data collection layer (45 min)
Use `boto3.client('ce')` for AWS Cost Explorer. Pull GroupBy=TAG:team and GroupBy=SERVICE for each day. Write to local Parquet.

```python
import boto3, pandas as pd
ce = boto3.client('ce')
res = ce.get_cost_and_usage(
    TimePeriod={'Start': '2026-05-01', 'End': '2026-05-02'},
    Granularity='DAILY',
    Metrics=['UnblendedCost'],
    GroupBy=[{'Type':'TAG','Key':'team'}, {'Type':'DIMENSION','Key':'SERVICE'}],
)
# Flatten to a DataFrame with columns: date, team, service, cost_usd
```

### Step 2 — Cost allocation logic (45 min)
Group rows by (team, workload, service, lifecycle). Map unknown teams to `untagged`. Surface untagged spend as a separate category.

### Step 3 — Idle detection (45 min)
- For EC2: pull CloudWatch metric `CPUUtilization` last 7 days; flag <5%.
- For GPU instances: pull `nvidia_smi` metric or DCGM if you've enabled it; flag idle.
- For RDS / Elasticache: similar with `DatabaseConnections` / cluster CPU.

### Step 4 — Weekly report (45 min)
Templated markdown with: total spend, top 3 teams, top 3 services, top 3 waste items, untagged spend %, deltas vs prior week.

### Step 5 — Slack delivery (15 min)
Use Slack incoming webhook. Post as a code block with the markdown. Add a button linking to your Grafana dashboard.

### Step 6 — Athena + Grafana (30 min)
Crawl the Parquet bucket with Glue. Connect Grafana to Athena. Build a single dashboard with stacked-area chart by team and pie charts for the week.

### Step 7 — Cron / Argo Workflow (15 min)
Schedule `collect` daily at 06:00 UTC; `report` + `digest` weekly Monday 09:00 local time.

## Deliverables

1. `mlfinops` CLI satisfying all 4 commands.
2. Grafana dashboard JSON (commit to repo).
3. Slack webhook integration.
4. A `BUDGETS.yaml` defining per-team monthly budgets.
5. A `RUNBOOK.md` explaining how to investigate a spending anomaly.

## Validation

- [ ] `mlfinops collect` produces a Parquet for yesterday with team-attributed rows.
- [ ] `mlfinops report` matches AWS Cost Explorer totals to within 1%.
- [ ] `mlfinops idle` flags at least one resource on a real account.
- [ ] Slack digest delivers and is readable.
- [ ] Grafana dashboard renders the past 30 days.

## Stretch goals

- Add **forecast**: project month-end spend based on past 14 days; alert if it exceeds budget.
- Add **anomaly detection** (simple z-score) per (team, service) — flag days that are >3σ above baseline.
- Build a `mlfinops auto-stop` that shuts down resources matching idle criteria (with manual approval).
- Add **savings plans / committed-use** recommendations: identify steady-state spend that would benefit from reserved capacity.

## Common pitfalls

- **Tag propagation lag** — Cost Explorer takes up to 24 hours for new tags to appear in cost data. Build in retries.
- **Cost vs usage** — `UnblendedCost` is what you pay; `NetUnblendedCost` includes credits/discounts. Pick deliberately.
- **EBS snapshots accumulate silently** — Free-tier accounts have small allowances; production accounts can have terabytes of forgotten snapshots.
- **GPU idle is not always waste** — Some teams keep GPUs warm for low-latency training restarts. Allow per-team exceptions in policy.
- **Reporting in your own currency** — Cost Explorer returns USD by default; non-USD orgs need FX conversion.

## Solutions

Reference implementation in the engineer-solutions repo.
