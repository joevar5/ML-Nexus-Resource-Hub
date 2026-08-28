# Exercise 03: Model Registry and Promotion Workflow

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 02 (MLflow)

## Objective

Implement a structured model promotion pipeline: Staging → Production with automated quality gates, manual approval, traceable promotion events, and instant rollback. By the end you'll have a registry workflow that survives audit.

## Why this matters

Untracked model promotions ("alice updated it last week, I think") are the #1 cause of production ML incidents. A structured registry + promotion workflow eliminates that class of incident entirely.

## Requirements

1. **Three stages**: Development (auto-registered), Staging (auto-promoted on quality gate), Production (manual approval).
2. **Quality gates**: accuracy delta vs current Production must be ≥ -0.5pp; required slices must not regress > 1pp.
3. **Approval workflow**: Slack message with approve/reject buttons triggering the transition.
4. **Promotion event log**: every transition recorded with actor, timestamp, justification.
5. **Rollback CLI**: `rollback-model <name>` reverts Production to the previously-active version.

## Step-by-step

### Step 1 — Registry setup (15 min)
Per Exercise 02. Ensure a model is registered and has at least one Staging version.

### Step 2 — Auto-registration from training (30 min)
```python
# train.py
import mlflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model", registered_model_name="recs-ranker")
# By default, this creates a new version in the registry.
```

### Step 3 — Automated quality gate (45 min)
```python
# promote_to_staging.py
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
NAME = "recs-ranker"

# Find the latest "None"-stage version (just registered)
candidate = sorted(client.search_model_versions(f"name='{NAME}'"),
                   key=lambda v: int(v.version), reverse=True)[0]
candidate_run = client.get_run(candidate.run_id)
candidate_auc = candidate_run.data.metrics["auc"]

# Find current Production
prod_versions = client.get_latest_versions(NAME, stages=["Production"])
if not prod_versions:
    # No prod yet — promote unconditionally
    client.transition_model_version_stage(NAME, candidate.version, "Staging")
    print(f"v{candidate.version} → Staging (no prior Production)")
else:
    prod_run = client.get_run(prod_versions[0].run_id)
    prod_auc = prod_run.data.metrics["auc"]

    if candidate_auc >= prod_auc - 0.005:
        client.transition_model_version_stage(NAME, candidate.version, "Staging",
                                              archive_existing_versions=True)
        print(f"v{candidate.version} → Staging (auc {candidate_auc:.4f} vs prod {prod_auc:.4f})")
    else:
        print(f"v{candidate.version} REJECTED ({candidate_auc:.4f} < {prod_auc - 0.005:.4f})")
        exit(1)
```

### Step 4 — Slice-aware quality gates (30 min)
Extend the gate to per-slice (country, age range, segment) metrics:
```python
# Run inference on a per-slice evaluation set and check each.
for slice_name, slice_acc in candidate.data.metrics.items():
    if not slice_name.startswith("auc_by_country_"): continue
    prod_slice = prod_metrics.get(slice_name, 0)
    if slice_acc < prod_slice - 0.01:
        print(f"REJECT: {slice_name} regressed: {slice_acc:.3f} vs prod {prod_slice:.3f}")
        exit(1)
```

### Step 5 — Manual approval via Slack (45 min)
```python
# promote_to_prod.py — sends Slack approval request, blocks until response
import requests, json, time

def request_approval(model_name, version, justification):
    payload = {
        "channel": "#ml-approvals",
        "text": f"Promote {model_name} v{version} to Production?",
        "attachments": [{
            "callback_id": "promote",
            "actions": [
                {"name": "approve", "text": "Approve", "type": "button", "value": f"{model_name}:{version}:approve"},
                {"name": "reject",  "text": "Reject",  "type": "button", "value": f"{model_name}:{version}:reject"},
            ]
        }]
    }
    requests.post("https://hooks.slack.com/services/...", json=payload)

# Approver clicks → webhook → calls:
def on_approval(model_name, version, decision, actor):
    client.transition_model_version_stage(model_name, version, "Production",
                                          archive_existing_versions=True)
    client.set_model_version_tag(model_name, version, "promoted_by", actor)
    client.set_model_version_tag(model_name, version, "promoted_at", str(int(time.time())))
    audit_log(actor, model_name, version, decision)
```

In a real setup: Slack interactivity webhook → small FastAPI service → MLflow client call.

### Step 6 — Audit log (15 min)
```python
def audit_log(actor, model_name, version, decision, reason=""):
    with open("audit.jsonl", "a") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "actor": actor,
            "model": model_name,
            "version": version,
            "decision": decision,
            "reason": reason,
        }) + "\n")
```
In production: write to a database / S3 with cross-account write permission only.

### Step 7 — Rollback CLI (15 min)
```python
# rollback.py
def rollback(model_name):
    versions = sorted(client.search_model_versions(f"name='{model_name}'"),
                      key=lambda v: int(v.version), reverse=True)
    archived = [v for v in versions if v.current_stage == "Archived"]
    if not archived:
        print("No archived versions to roll back to")
        return
    prev = archived[0]
    client.transition_model_version_stage(model_name, prev.version, "Production",
                                          archive_existing_versions=True)
    print(f"Rolled back {model_name} to v{prev.version}")
    audit_log("rollback-cli", model_name, prev.version, "rollback")
```

## Deliverables

1. `promote_to_staging.py` — automated.
2. `promote_to_prod.py` — Slack-gated.
3. `rollback.py` — CLI.
4. `audit.jsonl` populated with at least 5 events.
5. `MODEL_LIFECYCLE.md`: documented flow from training to retirement.

## Validation

- [ ] A newly trained model auto-registers and auto-promotes to Staging if quality gate passes.
- [ ] Slack message produced with approve/reject actions.
- [ ] Approve → version moves to Production; reject → no change.
- [ ] Audit log captures every transition with actor.
- [ ] Rollback reverts to the prior version cleanly.

## Stretch goals

- Add **A/B test promotion**: 50/50 traffic between current Production and Staging for N days, auto-promote if Staging wins.
- Add **shadow mode**: route 100% prod traffic + 100% shadow to Staging; never use Staging response.
- Migrate from MLflow's deprecated stages to **aliases** (`@champion`, `@challenger`).

## Common pitfalls

- **Promoting without archiving** — Two Production versions visible; downstream confusion.
- **No quality gate at all** — Easy to promote a worse model accidentally.
- **Approval bypass via direct registry edit** — Production approvers must NOT have edit on the registry; the approval bot does.
- **Stage transitions deprecated** — MLflow 2.9+ prefers aliases (`@champion`); plan migration.
