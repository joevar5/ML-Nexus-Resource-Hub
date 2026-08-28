# Exercise 09: Drift Detection and Remediation

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 04 (state management)

## Objective

Implement automated drift detection for Terraform-managed infrastructure: scheduled `plan` jobs, drift reports, severity classification (cosmetic vs material), and a remediation workflow (revert vs adopt).

## Why this matters

Configuration drift is the silent killer of IaC discipline. After 6 months without checks, "manual fixes" accumulate and the next `terraform apply` proposes 200 changes you don't understand. Drift detection turns it into a continuous concern.

## Requirements

1. Daily `plan` for every project; output saved to S3.
2. Drift report: per resource, what changed, by whom (from CloudTrail).
3. Severity rules: tags = info, instance type = warning, security group rules = critical.
4. Slack notification per severity.
5. Remediation flow: review → revert or adopt.

## Step-by-step

### Step 1 — Scheduled `plan` (45 min)
For each project, GitHub Actions / GitLab schedule:
```yaml
on:
  schedule:
    - cron: "0 6 * * *"   # daily 6am UTC
jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
      - run: terraform init
      - id: plan
        run: |
          terraform plan -no-color -detailed-exitcode -out=plan.tfplan
          echo "exit_code=$?" >> $GITHUB_OUTPUT
      - if: steps.plan.outputs.exit_code == '2'
        run: |
          terraform show -json plan.tfplan > drift.json
          aws s3 cp drift.json s3://drift-reports/$(date +%Y-%m-%d)/$REPO.json
          # Trigger downstream report
```

`-detailed-exitcode` returns 2 when changes are detected.

### Step 2 — Drift classification (30 min)
```python
# classify_drift.py
import json
RULES = [
    {"resource_type": "aws_iam_role", "attribute": "assume_role_policy", "severity": "critical"},
    {"resource_type": "aws_security_group_rule", "attribute": "*", "severity": "critical"},
    {"resource_type": "aws_instance", "attribute": "instance_type", "severity": "warning"},
    {"resource_type": "*", "attribute": "tags", "severity": "info"},
]

def classify(drift):
    for change in drift["resource_changes"]:
        for attr, _ in change.get("change", {}).get("after", {}).items():
            for rule in RULES:
                if matches(change["type"], rule["resource_type"]) and matches(attr, rule["attribute"]):
                    yield {"resource": change["address"], "attribute": attr,
                           "severity": rule["severity"]}
                    break
```

### Step 3 — CloudTrail attribution (45 min)
For each drifted resource, query CloudTrail to find who made the change:
```python
import boto3
ct = boto3.client('cloudtrail')

def who_changed(resource_arn, since):
    events = ct.lookup_events(
        LookupAttributes=[{"AttributeKey": "ResourceName", "AttributeValue": resource_arn}],
        StartTime=since,
    )
    if events.get("Events"):
        return events["Events"][0]["Username"]
    return "unknown"
```

### Step 4 — Slack notification (15 min)
```python
import requests
def notify(drift_report):
    severity_counts = Counter(d["severity"] for d in drift_report)
    color = "danger" if severity_counts["critical"] > 0 else "warning"
    msg = {
        "attachments": [{
            "color": color,
            "title": f"Drift in {project}",
            "fields": [
                {"title": "Resources drifted", "value": str(len(drift_report)), "short": True},
                {"title": "Critical", "value": str(severity_counts["critical"]), "short": True},
            ],
            "actions": [
                {"name": "view", "type": "button", "text": "View report",
                 "url": f"https://drift-dashboard/{project}/today"},
            ],
        }]
    }
    requests.post(SLACK_WEBHOOK, json=msg)
```

### Step 5 — Remediation workflow (15 min)
Two options:
- **Revert**: `terraform apply` to restore declared state.
- **Adopt**: `terraform refresh` then commit the new state, OR update HCL to match reality.

Decide per drift:
- Info severity → ignore (or fix-on-next-deploy).
- Warning → triage in next standup.
- Critical → page on-call.

Document the decision tree in `DRIFT_RUNBOOK.md`.

## Deliverables

1. Drift-check workflow per project.
2. `classify_drift.py` script.
3. Slack notification working.
4. `DRIFT_RUNBOOK.md`.
5. Demo: introduce manual change → drift detected → notification arrives.

## Validation

- [ ] Daily run produces a report (empty most days).
- [ ] Synthetic drift (manually edit a resource) triggers notification.
- [ ] Severity classification correct.
- [ ] CloudTrail attribution identifies the actor.

## Stretch goals

- Add **automatic remediation** for cosmetic drift (tag fixes auto-applied).
- Add **monthly trend report**: drift frequency per project.
- Integrate with **driftctl** or **CloudCustodian** for richer detection.

## Common pitfalls

- **`plan` against stale state** — Always `terraform refresh` first.
- **Reporting all drift as critical** — Alert fatigue. Filter by severity.
- **Auto-remediating without review** — A drift might be a real emergency fix; reverting it ruins someone's day.
- **CloudTrail not enabled** — Without it, attribution is guesswork.
