# Exercise 09: Backfill Strategies

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 02 (Airflow) + 08 (lineage)

## Objective

Implement three backfill strategies (sequential, parallel-by-date, parallel-by-shard) and benchmark them against a real workload. Add safety rails — concurrency caps, dependency-aware ordering, dry-run mode, partial-rerun support — to prevent the "I accidentally backfilled prod and DDOS'd the database" incident.

## Why this matters

Backfilling is the most dangerous routine operation in data engineering. Done wrong, it overwrites good data, exhausts downstream APIs, or causes silent gaps. Done right, it's a routine maintenance task; done wrong, it's a quarter-long cleanup.

## Requirements

1. Three backfill strategies implemented and benchmarked.
2. Concurrency cap configurable.
3. Dependency-aware ordering (don't backfill child before parent).
4. Dry-run mode that produces an execution plan without writing anything.
5. Partial-rerun support: backfill failed sub-ranges without re-running everything.
6. Audit log of every backfill operation.

## The scenario

Daily aggregation DAG (`daily_user_activity`) has run since 2025-01-01. A bug in the aggregation logic was fixed yesterday; you need to backfill 6 months of historical data.

Constraints:
- Downstream API rate-limited to 1000 req/s.
- Source database: handle up to 20 concurrent connections.
- Daily aggregation typically takes 15 minutes.
- Cost: roughly $0.10 per day's aggregation.

## Step-by-step

### Step 1 — Sequential backfill (30 min)
```bash
airflow dags backfill daily_user_activity \
  --start-date 2025-01-01 \
  --end-date 2025-06-30 \
  --rerun-failed-tasks
```
This runs days serially. Time: 180 days × 15 min = 45 hours.

### Step 2 — Parallel by date (30 min)
Reduce `--max-active-runs` to safe value:
```bash
airflow dags backfill daily_user_activity \
  --start-date 2025-01-01 \
  --end-date 2025-06-30 \
  --conf '{"max_active_runs": 10}'
```
10 concurrent days. Time: 180 / 10 × 15 min = 4.5 hours.

But: 10 concurrent DAG runs × ~10 DB queries each = 100 connections → exceeds DB limit.
Reduce to `--max-active-runs 2` for connection safety. Better:

### Step 3 — Parallel by date with worker isolation (45 min)
Use Airflow pools to isolate backfill workers from real-time work:
```python
ingest = BashOperator(
    task_id="ingest",
    bash_command="...",
    pool="backfill_pool",
    pool_slots=1,
)
```
Create the pool: 5 slots for backfill, separate from default pool.

### Step 4 — Parallel by shard (60 min)
For a single date, split work across shards (e.g., by user_id range):
```python
@task
def aggregate_shard(date: str, shard_id: int, num_shards: int) -> str:
    df = read_shard(date, shard_id, num_shards)
    return write_aggregate(df)

# Dynamic task mapping
shards = aggregate_shard.partial(date="{{ ds }}", num_shards=8).expand(shard_id=range(8))
```
Now 1 day × 8 shards × parallel = ~2 min per day; 180 days serial = 6 hours; or 5 days parallel × 8 shards = ~36 min total.

### Step 5 — Dry-run mode (30 min)
Add a `--dry-run` to your custom backfill tool that prints the plan without executing:
```python
def backfill(start, end, dry_run=False):
    dates = pd.date_range(start, end)
    plan = []
    for date in dates:
        if dependencies_satisfied(date):
            plan.append({"date": str(date), "shards": list_shards(date), "estimated_cost": 0.10})
    if dry_run:
        print(json.dumps(plan, indent=2))
        return
    # actually execute
```

### Step 6 — Partial rerun (30 min)
Track per-day status in a backfill state file:
```python
# backfill_state.json
{
  "2025-01-01": {"status": "success", "shards_done": 8},
  "2025-01-02": {"status": "failed", "shards_done": 5, "failed_shards": [5,6,7]},
  ...
}
```
Re-running backfill resumes from the failure point.

### Step 7 — Audit log (15 min)
Every backfill operation appends to `audit.jsonl`:
```json
{"ts": "2026-05-22T15:00:00Z", "user": "alice", "command": "backfill daily_user_activity 2025-01-01..2025-06-30", "result": "success", "days_processed": 180}
```

## Deliverables

1. Custom `backfill.py` CLI implementing the strategies.
2. `BENCHMARKS.md` comparing the three strategies' wall-clock times.
3. `PLAYBOOK.md`: pre-backfill checklist (notify channel, double-check date range, dry-run first).
4. Audit log file populated with at least 3 test backfills.

## Validation

- [ ] Sequential backfill works (slow but safe).
- [ ] Parallel-by-date respects max-active-runs cap.
- [ ] Parallel-by-shard achieves the predicted ~5× speedup over sequential.
- [ ] Dry-run produces a plan without side effects.
- [ ] Partial rerun continues from failure without redoing successful days.
- [ ] Audit log captures every operation.

## Stretch goals

- Add **safe-mode**: refuse to backfill the past 24h (might be in-flight production data).
- Add **cost projection**: show estimated $ before executing.
- Integrate with **Marquez** to record backfill operations as lineage events.

## Common pitfalls

- **DDoS your own DB** — Always start with the lowest concurrency and ramp up.
- **Backfill overwrites good data** — Mark backfill outputs distinctly (e.g., `_backfilled_at` column); never silent-overwrite.
- **Time-zone bugs** — `--start-date 2025-01-01` in UTC vs local time produces different date ranges. Always be explicit.
- **Forgotten downstream rebuilds** — Backfilling a source dataset doesn't backfill downstream aggregates. Document the cascade or automate it.
