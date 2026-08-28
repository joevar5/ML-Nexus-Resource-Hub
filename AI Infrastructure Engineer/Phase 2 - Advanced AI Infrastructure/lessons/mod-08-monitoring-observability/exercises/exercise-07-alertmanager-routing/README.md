# Exercise 07: Alertmanager Routing and Notification Strategy

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 03 (PromQL); Alertmanager running

## Objective

Author production-shaped alerting rules (threshold + SLO burn-rate), wire them through an Alertmanager routing tree with severity-based receivers (PagerDuty + Slack + email), implement inhibition + silence + grouping, and demonstrate the end-to-end flow with a synthetic incident.

## Why this matters

Bad alerting trains engineers to ignore alerts. Engineers who design routing thoughtfully (right severity, right channel, right people) are the ones whose pages produce action.

## Requirements

1. **Alert rules** (10 total): threshold-based + 2 multi-window SLO burn-rate.
2. **Routing tree** with at least 3 receivers (PagerDuty, Slack-critical, Slack-warnings).
3. **Inhibition rules** to suppress noise (e.g., `TargetDown` inhibits downstream `HighErrorRate`).
4. **Group_by** rules that bundle related alerts.
5. **Silences** for known issues (CLI + API + UI walk-through).
6. **Templates** for human-readable Slack and PagerDuty messages.
7. End-to-end test: trigger each severity from `amtool` and verify receiver behavior.

## Step-by-step

### Step 1 — Alert rules (45 min)
```yaml
# alerts.yml
groups:
  - name: model-api
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{job="model-api",status=~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="model-api"}[5m]))
          > 0.05
        for: 5m
        labels: { severity: critical, service: model-api, team: ml-platform }
        annotations:
          summary: "5xx error rate > 5% for 5m"
          runbook_url: "https://runbooks/high-error-rate"

      - alert: TargetDown
        expr: up{job="model-api"} == 0
        for: 2m
        labels: { severity: critical, service: model-api }
        annotations: { summary: "Prometheus target down" }

      - alert: ErrorBudgetBurnFast
        expr: |
          (sum(rate(http_requests_total{status=~"5..",job="model-api"}[5m])) / sum(rate(http_requests_total{job="model-api"}[5m]))) > (14.4 * 0.01)
          and
          (sum(rate(http_requests_total{status=~"5..",job="model-api"}[1h])) / sum(rate(http_requests_total{job="model-api"}[1h]))) > (14.4 * 0.01)
        labels: { severity: critical, service: model-api, slo: availability }
        annotations: { summary: "Burning 5% of monthly error budget in 1h" }

      # ... 7 more covering latency, queue depth, restarts, drift, infrastructure
```

### Step 2 — Routing tree (45 min)
```yaml
# alertmanager.yml
route:
  receiver: default-slack
  group_by: [alertname, cluster, service]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  routes:
    - matchers: [severity = "critical"]
      receiver: pagerduty
      continue: true
    - matchers: [severity = "critical"]
      receiver: slack-critical
    - matchers: [severity = "warning"]
      receiver: slack-warnings
    - matchers: [team = "data-platform"]
      receiver: data-platform-slack

inhibit_rules:
  - source_matchers: [severity = "critical"]
    target_matchers: [severity =~ "warning|info"]
    equal: [alertname, service, cluster]
  - source_matchers: [alertname = "TargetDown"]
    target_matchers: [alertname =~ "HighErrorRate|SlowResponse"]
    equal: [service, cluster]

receivers:
  - name: default-slack
    slack_configs:
      - channel: '#alerts'
        send_resolved: true
        title:  '{{ template "slack.title" . }}'
        text:   '{{ template "slack.text" . }}'
  - name: slack-critical
    slack_configs:
      - channel: '#alerts-critical'
        color: danger
        ... 
  - name: slack-warnings
    slack_configs:
      - channel: '#alerts-warnings'
  - name: pagerduty
    pagerduty_configs:
      - service_key_file: /etc/alertmanager/secrets/pagerduty
        severity: '{{ .CommonLabels.severity }}'
        details:
          runbook: '{{ .CommonAnnotations.runbook_url }}'
  - name: data-platform-slack
    slack_configs:
      - channel: '#data-platform-alerts'
```

### Step 3 — Templates (30 min)
```
# templates.tmpl
{{ define "slack.title" }}
  {{- if eq .Status "firing" -}}:fire:{{- else -}}:white_check_mark:{{- end }}
  [{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}]
  {{ .CommonLabels.alertname }} on {{ .CommonLabels.service }}
{{ end }}
{{ define "slack.text" }}
{{ range .Alerts }}
*Summary:* {{ .Annotations.summary }}
*Severity:* {{ .Labels.severity }}
{{ if .Annotations.runbook_url }}*Runbook:* {{ .Annotations.runbook_url }}{{ end }}
{{ end }}
{{ end }}
```

### Step 4 — End-to-end testing (30 min)
```bash
# Fire a fake alert
amtool alert add \
  alertname=HighErrorRate severity=critical service=model-api \
  --annotation=summary="synthetic test from amtool"

# Check delivery in Slack #alerts-critical and PagerDuty.

# Silence it for an hour
amtool silence add alertname=HighErrorRate --duration=1h --comment="testing"

# Verify routing
amtool config routes test --config.file=alertmanager.yml severity=warning team=ml-platform
# Should report: warnings-slack receiver matched
```

### Step 5 — Validate inhibition (15 min)
Fire both `TargetDown` and `HighErrorRate` for the same service. Only `TargetDown` should send notifications; the error alert is inhibited.

### Step 6 — Production-shape grouping (15 min)
Confirm: 50 simultaneous `HighErrorRate` alerts (one per pod) are grouped into a single notification, not 50 separate pages.

## Deliverables

1. `alerts.yml` with all 10 rules.
2. `alertmanager.yml` with routing + inhibition.
3. `templates.tmpl` with branded Slack + PagerDuty messages.
4. `ALERTING_PHILOSOPHY.md`: when to alert, what severity, what runbook standard.
5. Demo recording or screenshots of the end-to-end flow.

## Validation

- [ ] `amtool config routes test` confirms each severity reaches the expected receiver.
- [ ] Synthetic critical alert produces a PagerDuty incident and a Slack message in #alerts-critical.
- [ ] Inhibition demonstrably suppresses the lower-severity alert.
- [ ] Grouping bundles related alerts into one notification.
- [ ] Silence stops notifications without removing the alert.

## Stretch goals

- Add **routing by time window**: certain alerts page only during business hours (`time_intervals`).
- Add **escalation policies** in PagerDuty: if not acknowledged in 15m, escalate to next on-call.
- Integrate with **incident.io** or **Squadcast** for cross-channel incident management.

## Common pitfalls

- **Notification spam** — Symptom of bad `group_by`. Group by labels that genuinely cluster together (alertname + cluster), not by request_id.
- **Critical alerts that aren't actionable** — Every page should have a runbook. If no runbook exists, downgrade to warning.
- **Routing tree shadows** — Sub-routes are matched in order; the first matching `receiver` wins unless `continue: true`. Test with `amtool`.
- **Templates that don't render** — Test with `amtool template render`; a broken template silently sends raw default text.
