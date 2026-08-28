# Exercise 12: Pipeline Cost Optimization

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 05 + 10

## Objective

For the pipeline you've been building across module 105, apply 6 cost-optimization techniques and measure savings: spot instances, columnar formats + compression, partitioning, file size tuning, query result caching, and serverless vs always-on. Target a 60% cost reduction without losing functionality.

## Why this matters

A data pipeline's cost is typically dominated by 3-5 line items: compute (Spark, dbt warehouse), storage (S3 + Parquet), query (warehouse credits or Athena scans), and movement (egress). Engineers who understand each lever ship pipelines that scale economically.

## Cost drivers per system

| Layer | Cost driver | Typical line item |
|---|---|---|
| Compute (Spark/EMR) | Instance type × runtime × instance count | EC2 |
| Compute (Glue/serverless) | DPU × runtime | Glue jobs |
| Warehouse (Snowflake/BigQuery) | Credits × bytes scanned or seconds | Compute credits |
| Storage (S3) | GB × month × storage class | S3 standard / IA / Glacier |
| Egress | Bytes egressed cross-region or out | Data transfer |
| Format | Parquet vs CSV vs JSON, compression | Storage + scan |

## Requirements

Apply each technique to your pipeline; measure before/after:

### 1. Compression + columnar
Snappy-compressed Parquet vs CSV — typically 5-10× smaller, 5-50× faster to scan for column-projection queries.

### 2. Partitioning
Hive-style partition pruning. Query `WHERE date = '2026-05-22'` scans only that day's data.

### 3. File size tuning
Optimal Parquet file size: 256-1024 MB. Many small files = listing overhead; one huge file = no parallelism. Compact small files via `OPTIMIZE` (Delta) or rewrite jobs.

### 4. Spot instances for batch
EC2 Spot is 60-90% off on-demand. Pipeline must tolerate interruption (checkpoint + resume).

### 5. Query result caching
Materialize hot aggregates (e.g., `dim_users`, `fct_user_daily_activity` from exercise 10) as tables, not views. Recomputed once per dbt run, not per query.

### 6. Serverless vs always-on
For < ~30% utilization, Glue / Athena / BigQuery on-demand beats EMR / Snowflake always-on warehouse. For > 70% utilization, the reverse.

## Step-by-step

### Step 1 — Baseline (30 min)
Compute current monthly cost:
- Spark job: X DPU-hours × $0.44 = $Y/month
- Storage: Z TB × $0.023 = $W/month
- Athena/BigQuery scans: ...
- Egress: ...

Record in `BASELINE.md`.

### Step 2 — Compression + columnar (15 min)
If your data isn't already Parquet+Snappy, convert it. Re-measure storage + query latency.

### Step 3 — Partitioning audit (30 min)
For each table, list query patterns:
- Always queried with `date` predicate → partition by date.
- Queried by user → cardinality too high; don't partition; use bucketing instead.
- Sometimes scanned whole → don't partition (overhead > benefit).

Re-write affected pipelines.

### Step 4 — File size compaction (30 min)
```sql
-- Delta Lake
OPTIMIZE my_table ZORDER BY (user_id);

-- Iceberg
CALL system.rewrite_data_files('my_table');

-- Plain Parquet on S3: use a Spark job that reads-and-writes with target file size
```

### Step 5 — Spot for batch (30 min)
For your nightly Spark job:
- EMR: spot fleet with diversified instance types, 70% spot + 30% on-demand for stability.
- Glue: spot is implicit in serverless.
Add checkpoint: save progress every 5 min so a spot interruption resumes, not restarts.

### Step 6 — Materialization audit (30 min)
For each dbt model, ask:
- Queried often, expensive to compute → `materialized='table'`
- Queried often, cheap to compute → `view` is fine
- Queried rarely → `view` or `ephemeral`
- Incremental compatible → `incremental` is best

Reassign per the matrix. Re-measure warehouse credits.

### Step 7 — Serverless cost test (15 min)
Run your pipeline once on Glue (serverless) and once on EMR (always-on cluster). Compare cost + runtime. Decision matrix in `COST_RECOMMENDATIONS.md`.

### Step 8 — Egress audit (15 min)
List every cross-region or cross-cloud egress. For each: can it be moved to same-region? S3 cross-region replication costs less than per-query egress.

## Deliverables

1. `BASELINE.md` with before-state cost.
2. Config / code diffs for each technique.
3. `COST_VS_PERF.md`: per-technique cost saved, perf impact (positive or negative).
4. `RECOMMENDATIONS.md`: final config + projected monthly cost.

## Validation

- [ ] Total monthly cost ≤ 40% of baseline.
- [ ] No technique breaks an existing query.
- [ ] Spot interruptions tested in staging — pipeline resumes correctly.
- [ ] Cost breakdown documented per pipeline stage.

## Stretch goals

- Add **automated cost reporting**: weekly summary by pipeline, by team, with deltas.
- Implement **tiered storage**: S3 Standard for hot, Standard-IA for warm, Glacier for cold; with lifecycle rules.
- Run a **chaos cost test**: deliberately schedule peak runs at different times to find off-peak pricing arbitrage.

## Common pitfalls

- **Over-partitioning** — Date+hour+user_id makes 1000 partitions per day; small files explode.
- **Spot interruption without checkpoint** — Long-running jobs restart from scratch; net cost rises.
- **Switching to serverless at high utilization** — Per-second billing is cheaper at low duty cycle, more expensive at high.
- **Egress hiding in cross-account S3 reads** — Even same-region cross-account S3 is sometimes billed; verify.
- **dbt full-refresh on incremental models** — One `--full-refresh` flag burns a month's saved credits in an hour.

## Solutions

Reference materialization decisions + Glue vs EMR comparison in the engineer-solutions repo.
