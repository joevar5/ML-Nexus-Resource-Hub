# Exercise 10: Observability Cost Control

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** All prior exercises

## Objective

For a representative observability stack (Prometheus + Loki + Tempo + Grafana), characterize per-component cost drivers, apply 5 cost-reduction techniques (cardinality control, sampling, retention tiering, downsampling, storage class), and produce a cost-vs-coverage trade-off matrix. Target: 60% cost reduction with < 5% coverage loss.

## Why this matters

Observability bills are often the second-largest line item on a platform team's cloud bill (after compute). At scale, observability can cost more than the systems it observes. Engineers who can reduce it without losing critical signal are unusually valuable.

## Cost drivers per system

| System | Primary driver | Typical line items |
|---|---|---|
| **Prometheus** | Active time-series count × scrape interval | Memory (TSDB head), disk (WAL + blocks), CPU (queries) |
| **Loki** | Bytes ingested × retention | S3 (chunks), index (DynamoDB/BoltDB), compute |
| **Tempo** | Span volume × retention × sampling rate | S3 (blocks), compute (compactors) |
| **Grafana** | Queries per second × dashboard count | Modest compute, large at scale |
| **Cloud APM (Datadog/NewRelic)** | Custom metrics × hosts × log GB | All-of-the-above billed by vendor |

## Requirements

1. Baseline measurement of your stack's current consumption.
2. Apply 5 cost-reduction techniques:
   - Cardinality reduction
   - Trace sampling
   - Metric retention tiering (raw + downsampled)
   - Log retention by level + service tier
   - Object-storage class transitions (S3 IA, Glacier)
3. Measure consumption after each technique.
4. Identify coverage loss (what can you no longer answer?).
5. Final report with recommendations.

## Step-by-step

### Step 1 — Baseline (30 min)
For each component, gather:
```promql
# Prometheus active series
prometheus_tsdb_head_series

# Prometheus storage usage
prometheus_tsdb_storage_blocks_bytes

# Loki bytes ingested
sum(rate(loki_distributor_bytes_received_total[1h])) * 3600 * 24 * 30  # monthly

# Tempo spans ingested
sum(rate(tempo_distributor_spans_received_total[1h])) * 3600 * 24 * 30

# Active series breakdown by label name
topk(20, count by (__name__)({__name__=~".+"}))
```
Record numbers in `BASELINE.md`.

### Step 2 — Cardinality reduction (45 min)
Top offenders are usually:
- Per-user / per-request labels
- High-cardinality URL paths (`user_id=12345`)
- Customer IDs

Strategies:
- **Drop labels** at scrape: `relabel_configs: - action: labeldrop regex: 'request_id'`
- **Replace dynamic paths**: `/users/<id>` → `/users/:id` via metric_relabel_configs
- **Aggregate before exporting** (recording rule with reduced label set)

Apply and re-measure. Common outcome: 40-60% active series reduction.

### Step 3 — Trace sampling (30 min)
Replace head-based 100% sampling with tail-based or probabilistic:
```yaml
# OTel collector config
processors:
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors-and-slow
        type: and
        and:
          and_sub_policy:
            - { name: status_code, type: status_code, status_code: { status_codes: [ERROR] }}
        # 100% of errors + slow requests
      - name: probabilistic
        type: probabilistic
        probabilistic: { sampling_percentage: 1 }   # 1% of normal traffic
```
Trace volume drops 95%+ while keeping all errors and a representative slow-request sample.

### Step 4 — Metric retention tiering (30 min)
Use Thanos or Cortex/Mimir for tiered storage:
- Raw 15s resolution: 14 days
- Downsampled 5m resolution: 90 days
- Downsampled 1h resolution: 2 years

Configure Thanos compactor; verify queries on old data return downsampled results automatically. Storage cost drops > 70% for old data.

### Step 5 — Log retention by level + tier (30 min)
Loki retention policies:
- DEBUG: not ingested (drop in Vector)
- INFO: 7 days
- WARN: 30 days
- ERROR: 90 days
- Audit logs: 1 year (regulatory)

Implement via Vector routing (per exercise 05).

### Step 6 — S3 storage class transitions (15 min)
S3 lifecycle rules:
- 0-30 days: STANDARD
- 30-90 days: STANDARD_IA (50% cheaper)
- 90+ days: GLACIER_FLEXIBLE_RETRIEVAL (75% cheaper)

Applies to Loki chunks, Tempo blocks, Prometheus archived blocks.

### Step 7 — Cost vs coverage matrix (30 min)
For each technique, document:
- Cost saving (%)
- Coverage loss (what queries can you no longer run?)
- Risk (what scenario does this make harder to debug?)

| Technique | Saved | Lost | Risk |
|---|---|---|---|
| Drop request_id label | 35% | Per-request traceability via Prometheus (use Tempo instead) | Low — Tempo retains traces |
| 1% probabilistic trace sampling | 95% trace storage | Trace for unsampled normal requests | Medium — capture errors at 100% |
| INFO retention 7d → 3d | 50% Loki storage | Investigating > 3-day-old INFO events | Low — most investigations are < 24h |
| ... | ... | ... | ... |

## Deliverables

1. `BASELINE.md` with measurements.
2. Config diffs for each technique (Prometheus, Vector, Loki, Tempo).
3. `COST_VS_COVERAGE.md` matrix.
4. `RECOMMENDATIONS.md`: which techniques to apply at your team's stage.

## Validation

- [ ] At least 5 distinct cost-reduction techniques applied.
- [ ] Each technique has before/after measurement.
- [ ] Total cost reduction ≥ 50% on the test stack.
- [ ] Coverage loss explicitly documented; no surprise discoveries later.

## Stretch goals

- Implement **adaptive sampling**: increase sampling rate during incidents.
- Add **cost allocation per team** (per exercise 08 of mod-02).
- Build a **budget alert**: when Loki ingestion this month projects > $X, page the platform team.

## Common pitfalls

- **Cardinality killer = a label introduced 6 months ago** — Active series sneak up. Run cardinality reports monthly.
- **Sampling that drops your incidents** — Tail-based sampling on errors first; never sample errors away.
- **Retention reduction that breaks compliance** — Audit logs often have regulatory retention. Tag those explicitly and exclude from cost-cutting.
- **Saving money by reducing visibility** — Always document what queries become impossible. Your team will ask.

## Solutions

Reference Vector + Thanos configs in the engineer-solutions repo.
