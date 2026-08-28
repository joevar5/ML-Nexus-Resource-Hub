# Exercise 04: Grafana Dashboards as Code

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 03 (PromQL); Grafana running

## Objective

Author Grafana dashboards in code (not the clickops UI), version-control them, validate via CI, and provision them automatically into a fresh Grafana instance. By the end, dashboards are pull-request-reviewed like any other code.

## Why this matters

UI-edited dashboards drift, accumulate broken panels, get accidentally overwritten, and don't survive instance rebuilds. Dashboards-as-code treats them like Terraform: explicit, reviewed, reproducible. Every team past ~5 engineers ends up here.

## Tooling options

Pick one (or implement two for comparison):
- **grafonnet** (Jsonnet library) — most mature, used by Grafana Labs themselves.
- **grafanalib** (Python) — most ergonomic for Python-shop teams.
- **terraform-provider-grafana** — fits if your IaC is already Terraform.
- **dashboard-as-code via Grafana's provisioning + JSON** — simplest, no DSL.

This exercise uses **Python + grafanalib** as the primary path.

## Requirements

1. Three dashboards in code: API overview, ML drift, SLI/SLO error budget.
2. Each dashboard has at least 5 panels.
3. Templated variables (datasource, namespace, service).
4. Annotations linking to deploy events.
5. CI workflow that lints + renders dashboards on every PR and posts a screenshot diff.
6. Provisioning config that auto-loads them on Grafana startup.

## Step-by-step

### Step 1 — Setup (15 min)
```bash
pip install grafanalib
mkdir dashboards-as-code && cd dashboards-as-code
```

### Step 2 — Dashboard 1: API overview (60 min)
```python
# api_overview.dashboard.py
from grafanalib.core import *

dashboard = Dashboard(
    title="iris-api · overview",
    tags=["iris", "api", "ml"],
    timezone="utc",
    refresh="30s",
    templating=Templating(list=[
        Template(name="datasource", type="datasource", query="prometheus"),
        Template(name="namespace", type="query", dataSource="$datasource",
                 query="label_values(kube_pod_info, namespace)"),
    ]),
    panels=[
        Graph(title="Request rate",
              dataSource="$datasource",
              targets=[Target(expr='sum(rate(http_requests_total{namespace="$namespace"}[1m]))')]),
        Stat(title="Error rate %",
             dataSource="$datasource",
             targets=[Target(expr='100*sum(rate(http_requests_total{status=~"5..",namespace="$namespace"}[5m]))/sum(rate(http_requests_total{namespace="$namespace"}[5m]))')]),
        Graph(title="p50/p95/p99 latency",
              dataSource="$datasource",
              targets=[
                  Target(expr='histogram_quantile(0.50, sum by(le)(rate(http_request_duration_seconds_bucket[5m])))', legendFormat='p50'),
                  Target(expr='histogram_quantile(0.95, sum by(le)(rate(http_request_duration_seconds_bucket[5m])))', legendFormat='p95'),
                  Target(expr='histogram_quantile(0.99, sum by(le)(rate(http_request_duration_seconds_bucket[5m])))', legendFormat='p99'),
              ]),
        # 2 more panels
    ],
).auto_panel_ids()
```

Build a `Makefile` target that generates JSON:
```python
# build.py
from grafanalib._gen import write_dashboard
from api_overview.dashboard import dashboard
write_dashboard(dashboard, "out/api_overview.json")
```

### Step 3 — Dashboard 2: ML drift (45 min)
Panels for: PSI per feature (top-10 bar chart), prediction distribution over time (heatmap), feature distribution comparison (split histograms), drift alert status.

### Step 4 — Dashboard 3: SLI/SLO (45 min)
Per Google SRE workbook: availability SLI, latency SLI, error budget remaining gauge, burn-rate alert visualization.

### Step 5 — CI workflow (30 min)
```yaml
# .github/workflows/dashboards.yml
on: [pull_request]
jobs:
  build-and-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install grafanalib
      - run: make build
      - name: Validate JSON
        run: |
          for f in out/*.json; do
            python -m json.tool "$f" > /dev/null
            jq -e '.panels | length > 0' "$f"
          done
      - name: PromQL syntax check
        run: |
          # Extract every PromQL string and validate via promtool
          jq -r '.panels[].targets[].expr' out/*.json | \
            while read -r q; do
              echo "$q" | promtool check parse-expressions
            done
```

### Step 6 — Provisioning (15 min)
```yaml
# grafana/provisioning/dashboards/dashboards.yaml
apiVersion: 1
providers:
  - name: code
    folder: ML Platform
    type: file
    options:
      path: /var/lib/grafana/dashboards
```
Volume-mount your `out/` into `/var/lib/grafana/dashboards`. Restart grafana; dashboards appear.

### Step 7 — Screenshot diff in CI (30 min)
Stretch: use `grafana-screenshot-action` or headless Chromium to capture each dashboard against a known-good Prometheus dataset. PR check fails if rendering changes unexpectedly.

## Deliverables

1. Python dashboard definitions for all 3.
2. Rendered JSON in `out/`.
3. CI workflow.
4. Grafana provisioning config.
5. README explaining the developer flow ("how to add a new dashboard").

## Validation

- [ ] `make build` produces 3 JSON files.
- [ ] All JSON files import cleanly into Grafana.
- [ ] CI fails when a PR introduces invalid PromQL.
- [ ] Restarting Grafana auto-loads the dashboards.

## Stretch goals

- Add **alerting rules** also generated from code (via grafonnet `alerts.libsonnet`).
- Build a **dashboard registry** server that serves dashboards by version + Git SHA.
- Implement **dashboard ownership**: every panel tagged with a team's Slack channel.

## Common pitfalls

- **Auto-generated panel IDs collide** — Always call `.auto_panel_ids()` on the Dashboard.
- **Templating variables don't propagate** — Variable names must match exactly (case-sensitive) across panels.
- **Provisioned dashboards become read-only** — That's the point. Edit in code, not the UI.
- **Grafana version mismatch** — `grafanalib` lags grafana releases; some new panel types require manual JSON authoring.
