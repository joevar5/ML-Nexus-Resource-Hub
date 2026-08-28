# Lab 05: Author Alert Rules with Multi-Window Burn Rates

**Duration:** 60 min  **Prerequisites:** Prometheus + Alertmanager

## Objective
Write production-shaped Prometheus alert rules: classic threshold rules + Google-SRE-style multi-window error budget burn-rate alerts. Route them through Alertmanager with severity-based receivers.

## Steps

### 1. Threshold alerts
```yaml
# alerts.yml
groups:
  - name: model-api
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(predictions_total{status="error"}[5m]))
          /
          sum(rate(predictions_total[5m]))
          > 0.05
        for: 5m
        labels: { severity: critical }
        annotations:
          summary: "Error rate > 5% for 5m"
          runbook_url: "https://runbooks/high-error-rate"

      - alert: SlowResponses
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(inference_latency_seconds_bucket[5m])))
          > 0.5
        for: 10m
        labels: { severity: warning }
        annotations:
          summary: "p95 latency > 500ms for 10m"
```

### 2. SLO error budget burn rate
For an SLO of 99% availability over 30 days, the error budget is 1%. Multi-window alerts catch both fast burns (5% of budget in 1h) and slow burns (10% of budget in 6h).
```yaml
      - alert: ErrorBudgetBurnFast
        expr: |
          (
            sum(rate(predictions_total{status="error"}[5m]))
            /
            sum(rate(predictions_total[5m]))
          ) > (14.4 * 0.01)
          and
          (
            sum(rate(predictions_total{status="error"}[1h]))
            /
            sum(rate(predictions_total[1h]))
          ) > (14.4 * 0.01)
        labels: { severity: critical }
        annotations:
          summary: "Burning 5% of monthly error budget in 1h"

      - alert: ErrorBudgetBurnSlow
        expr: |
          (
            sum(rate(predictions_total{status="error"}[30m]))
            /
            sum(rate(predictions_total[30m]))
          ) > (6 * 0.01)
          and
          (
            sum(rate(predictions_total{status="error"}[6h]))
            /
            sum(rate(predictions_total[6h]))
          ) > (6 * 0.01)
        labels: { severity: warning }
        annotations:
          summary: "Burning 10% of monthly error budget in 6h"
```
Burn rates of 14.4× and 6× from Google SRE workbook.

### 3. Validate syntax
```bash
docker run --rm -v $PWD:/work prom/prometheus:v2.51.0 \
  promtool check rules /work/alerts.yml
```

### 4. Alertmanager config (alertmanager.yml)
```yaml
route:
  receiver: default-slack
  group_by: [alertname, service]
  group_wait: 30s
  routes:
    - matchers: [severity = "critical"]
      receiver: pagerduty
    - matchers: [severity = "warning"]
      receiver: warnings-slack
receivers:
  - name: default-slack
    slack_configs:
      - channel: '#alerts'
        api_url: 'https://hooks.slack.com/services/T.../B.../...'
  - name: warnings-slack
    slack_configs:
      - channel: '#alerts-warnings'
        api_url: '...'
  - name: pagerduty
    pagerduty_configs:
      - service_key: '<routing-key>'
```

### 5. Test alert routing without firing real traffic
```bash
amtool config routes test --config.file=alertmanager.yml severity=critical
# Expected: matches the pagerduty receiver
```

### 6. Silence an alert temporarily
```bash
amtool silence add alertname=HighErrorRate --duration=2h --comment="known issue, fix in flight"
amtool silence list
```

## Validation
- [ ] `promtool check rules` returns OK.
- [ ] `amtool config routes test` returns the expected receiver per severity.
- [ ] Manually firing an alert (via `amtool alert add ...`) appears in the right Slack channel.

## Cleanup
```bash
amtool silence expire $(amtool silence list -q)
```

## Troubleshooting
- **Rule evaluation lag** — `evaluation_interval` default is 1m; can be lowered, at cost of CPU.
- **`for` clause ignored** — Some operators (`absent`, `up`) don't behave intuitively with `for`. Test in Prometheus UI before relying.
- **Burn-rate alerts always firing on low traffic** — Add `and sum(rate(predictions_total[1h])) > 1` to require minimum volume before alerting.
