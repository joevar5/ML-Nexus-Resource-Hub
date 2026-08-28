# Exercise 08: SLI / SLO / Error Budget Implementation

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercises 03 + 07

## Objective

For the iris-api service, define formal SLIs, set SLOs, compute the error budget, set up burn-rate alerts at multiple windows, and produce a quarterly SLO review template. By the end you'll have applied the Google SRE workbook patterns to a real service.

## Why this matters

SLOs are how engineering teams negotiate reliability vs feature velocity with product and leadership. A team without SLOs alternates between fragile (high pressure to fix every alert) and complacent (no clear sense of when things are bad). SLOs give the conversation grounding.

## Background

Three concepts, in order:

- **SLI (Service Level Indicator):** a measured ratio: `good_events / total_events`. Two for iris-api:
  - Availability SLI = `successful_requests / total_requests`
  - Latency SLI = `requests_under_threshold / total_requests`

- **SLO (Service Level Objective):** a target on the SLI:
  - Availability SLO: 99.5% over 30 days.
  - Latency SLO: 95% of requests under 200ms over 30 days.

- **Error budget:** `1 - SLO`. For 99.5%, budget = 0.5% × total. Burn the budget faster than monthly rate → page.

## Requirements

1. Define SLIs as PromQL recording rules.
2. Set quantitative SLOs.
3. Compute error budget remaining.
4. Implement multi-window multi-burn-rate alerts (per Google SRE workbook).
5. Build a Grafana panel showing budget burn.
6. Write a quarterly SLO review template.

## Step-by-step

### Step 1 — Define SLIs as recording rules (30 min)
```yaml
# sli-rules.yml
groups:
  - name: iris-api-sli
    interval: 30s
    rules:
      - record: sli:availability:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{job="iris-api",status!~"5.."}[5m]))
          /
          sum(rate(http_requests_total{job="iris-api"}[5m]))

      - record: sli:latency:ratio_rate5m
        expr: |
          sum(rate(http_request_duration_seconds_bucket{job="iris-api",le="0.200"}[5m]))
          /
          sum(rate(http_request_duration_seconds_count{job="iris-api"}[5m]))
```

### Step 2 — Set SLOs (15 min)
```yaml
slo_availability: 0.995          # 99.5% over 30d
slo_latency:      0.95            # 95% under 200ms over 30d
budget_window:    30d
```

### Step 3 — Compute budget remaining (30 min)
```promql
# Availability error budget remaining (fraction 0..1):
1 - (
  (1 - sum(rate(http_requests_total{job="iris-api",status!~"5.."}[30d])) / sum(rate(http_requests_total{job="iris-api"}[30d])))
  / 0.005
)
```

### Step 4 — Multi-window burn-rate alerts (60 min)
Per SRE workbook, alert when 2 different windows both indicate fast burn:

```yaml
# Critical: 5% of monthly budget in 1h
- alert: ErrorBudgetBurnFast
  expr: |
    (1 - sli:availability:ratio_rate5m) > (14.4 * 0.005)
    and
    (1 - sli:availability:ratio_rate1h) > (14.4 * 0.005)
  labels: { severity: critical, slo: availability }

# Warning: 10% of monthly budget in 6h
- alert: ErrorBudgetBurnSlow
  expr: |
    (1 - sli:availability:ratio_rate30m) > (6 * 0.005)
    and
    (1 - sli:availability:ratio_rate6h) > (6 * 0.005)
  labels: { severity: warning, slo: availability }
```

Burn-rate values:
- 14.4× burn = exhausts a 30d budget in 50h → critical fast (≥1h sustained).
- 6× burn = exhausts in 5d → warning slow (≥6h sustained).

You'll need recording rules for the longer windows too:
```yaml
- record: sli:availability:ratio_rate30m
  expr: sum(rate(http_requests_total{job="iris-api",status!~"5.."}[30m])) / sum(rate(http_requests_total{job="iris-api"}[30m]))
- record: sli:availability:ratio_rate1h
  expr: ...
- record: sli:availability:ratio_rate6h
  expr: ...
```

### Step 5 — Grafana SLO panel (45 min)
Create a dashboard with:
- **Big number**: error budget remaining as a percentage (red < 25%, yellow < 50%, green ≥ 50%).
- **Trend chart**: SLI over the past 30 days with the SLO line.
- **Burn-rate sparkline**: instantaneous burn rate at 5m, 30m, 1h, 6h windows.
- **Forecast**: at current rate, when will budget exhaust?

### Step 6 — Quarterly SLO review template (30 min)
`SLO_REVIEW_TEMPLATE.md`:
```markdown
## Q[N] SLO Review — iris-api

**Period:** YYYY-MM-DD to YYYY-MM-DD

| SLO | Target | Actual | Met? | Budget consumed |
|---|---|---|---|---|
| Availability | 99.5% | 99.62% | ✅ | 76% |
| Latency p95 < 200ms | 95% | 91.4% | ❌ | 172% (exhausted) |

### Incidents this quarter
- 2026-04-15: 22-min outage on auth dep → 0.3% budget hit
- 2026-05-02: latency regression from v3.4 → 1.1% sustained budget hit

### Reliability work prioritized next quarter
- (Drawn from incident postmortem action items + the failing SLO above)

### SLO changes proposed
- Latency target was too aggressive given downstream feature-store p99; propose moving to 95% < 250ms.
```

## Deliverables

1. SLI recording rules (`sli-rules.yml`).
2. Burn-rate alert rules.
3. Grafana SLO dashboard JSON.
4. `SLO_REVIEW_TEMPLATE.md`.
5. `SLO_PHILOSOPHY.md`: how your team uses SLOs to make trade-off decisions.

## Validation

- [ ] All recording rules evaluate without errors (verify with `promtool check rules`).
- [ ] Synthetic sustained failure (e.g., 10% error rate for 30 min) triggers `ErrorBudgetBurnSlow`.
- [ ] SLO dashboard shows accurate budget remaining.
- [ ] Quarterly review template is filled in for at least one quarter's worth of data.

## Stretch goals

- Add **time-windowed SLOs** (e.g., 99.9% during business hours, 99% off-hours).
- Use **Sloth** or **Pyrra** to generate the SLO config from a simpler YAML spec.
- Implement a **policy gate**: refuse to deploy when error budget < 20%.

## Common pitfalls

- **SLI definition disagreements** — Decide once, write down. "Is a 4xx an error?" — usually no; it's the client's fault.
- **SLOs set by aspiration** — Set based on what you can sustain, not what would be nice. Easier to raise SLOs than lower them.
- **Alerting at SLO threshold** — Paging when you've just exhausted the budget is too late. Alert at burn rate, well before exhaustion.
- **Budget exhaustion ignored** — When the budget burns out, reliability work moves to top priority. If your team doesn't do this, the SLO is decorative.
