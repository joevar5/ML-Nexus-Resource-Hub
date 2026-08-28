# Exercise 11: Cost-Aware IaC

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 10

## Objective

Add cost estimation to every Terraform PR using Infracost. Set per-team monthly budgets. Block PRs that would exceed the budget without override.

## Requirements

1. Infracost in CI on every PR.
2. PR comment with cost delta vs main.
3. Monthly budget per team (defined in `budgets.yaml`).
4. CI fails if PR would push monthly cost over budget × 1.1.
5. Override mechanism: PR label `cost-approved` bypasses.

## Step-by-step

### Step 1 — Install Infracost (15 min)
```bash
brew install infracost
infracost auth login
infracost --version
```

### Step 2 — Baseline cost (15 min)
```bash
cd envs/prod
infracost breakdown --path .
```
Output: per-resource monthly cost, total.

### Step 3 — PR diff (30 min)
```bash
git checkout main; cd envs/prod && infracost breakdown --path . --format=json > /tmp/main.json
git checkout PR-branch && infracost diff --path . --compare-to /tmp/main.json
```

### Step 4 — CI workflow (45 min)
```yaml
on: pull_request
jobs:
  cost:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: infracost/actions/setup@v3
        with: { api-key: ${{ secrets.INFRACOST_API_KEY }} }
      - name: Generate Infracost cost estimate baseline
        run: |
          git checkout main
          infracost breakdown --path envs/prod --format=json --out-file=/tmp/baseline.json
      - name: Generate Infracost diff
        run: |
          git checkout ${{ github.event.pull_request.head.sha }}
          infracost diff --path envs/prod --format=json \
              --compare-to=/tmp/baseline.json --out-file=/tmp/diff.json
      - name: Post diff to PR
        uses: infracost/actions/comment@v3
        with: { path: /tmp/diff.json, behavior: update }
```

### Step 5 — Budget enforcement (45 min)
```yaml
# budgets.yaml
ml-platform:
  prod:    50000   # $50K/mo
  staging:  5000
  dev:      2000
```

```python
# enforce_budget.py
import json, yaml, sys, os
budgets = yaml.safe_load(open("budgets.yaml"))
diff = json.load(open("/tmp/diff.json"))

current_total = sum(p["pastTotalMonthlyCost"] or 0 for p in diff["projects"])
proposed_total = sum(p["totalMonthlyCost"] or 0 for p in diff["projects"])
env = os.environ["ENV"]
team = os.environ["TEAM"]
budget = budgets[team][env]

if proposed_total > budget * 1.10:
    print(f"FAIL: would push monthly cost to ${proposed_total:.0f} > budget ${budget * 1.10:.0f}")
    sys.exit(1)
print(f"OK: ${proposed_total:.0f} <= ${budget * 1.10:.0f}")
```

```yaml
      - name: Enforce budget
        if: contains(github.event.pull_request.labels.*.name, 'cost-approved') == false
        env: { ENV: prod, TEAM: ml-platform }
        run: python enforce_budget.py
```

PR can be merged anyway if labeled `cost-approved`.

## Deliverables

1. Infracost in CI.
2. PR comment with cost delta.
3. `budgets.yaml` with per-team limits.
4. Budget enforcement that blocks merges.
5. Override label demonstrated.

## Validation

- [ ] Every PR has a cost comment.
- [ ] PR adding $1K/mo blocked unless labeled.
- [ ] Override label allows merge.

## Stretch goals

- Add **forecast**: project month-end cost based on past 30 days + this PR.
- Add **savings recommendation**: if PR proposes m6i.large, suggest m7g.large (cheaper, same perf).
- Add **per-resource cost trend** dashboard.

## Common pitfalls

- **Infracost without --terraform-cloud-token** — Some resources can't be priced without Tf Cloud integration.
- **Spot prices not factored** — Spot is variable; Infracost uses on-demand by default.
- **Budget too tight** — Blocks legitimate growth. Quarterly budget review.
