# Exercise 10: ETL → ELT Migration with dbt

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 02 (Airflow), SQL fundamentals

## Objective

Migrate a Python/Spark ETL job to an ELT pattern using dbt: load raw data into a warehouse (DuckDB locally, or BigQuery/Snowflake for production), perform all transformations in SQL via dbt models, add tests + documentation, and orchestrate with Airflow.

## Why this matters

ETL (Extract-Transform-Load) puts business logic in Python/Spark code; ELT (Extract-Load-Transform) moves it to SQL on the warehouse. ELT is the modern default for ~80% of analytics workloads — better lineage, better governance, more accessible to analysts. Engineers who can fluidly choose between the two scale ML platforms more cleanly.

## Requirements

1. Start with an existing Spark/Python ETL job (use exercise 05's job).
2. Replace transformations with **dbt models** (SQL).
3. Add **dbt tests** (uniqueness, not null, accepted values).
4. Generate **dbt docs** with lineage graph.
5. **Source freshness checks**.
6. Orchestrate via **Airflow** invoking `dbt build`.

## Step-by-step

### Step 1 — Pick warehouse + install (15 min)
```bash
pip install 'dbt-duckdb==1.7.4'      # local DuckDB
# or: dbt-bigquery, dbt-snowflake, dbt-postgres
```

### Step 2 — Init dbt project (15 min)
```bash
dbt init recs_dbt
cd recs_dbt
# Configure ~/.dbt/profiles.yml:
#   recs_dbt:
#     target: dev
#     outputs:
#       dev:
#         type: duckdb
#         path: ./recs.duckdb
```

### Step 3 — Load raw data into the warehouse (30 min)
Use a Python "EL" step (just load):
```python
# load.py
import duckdb, pandas as pd
con = duckdb.connect("recs.duckdb")
con.execute("CREATE SCHEMA IF NOT EXISTS raw")
df = pd.read_parquet("s3://.../events.parquet")
con.register("df", df)
con.execute("CREATE OR REPLACE TABLE raw.events AS SELECT * FROM df")
```

### Step 4 — Source declaration (15 min)
```yaml
# models/sources.yml
version: 2
sources:
  - name: raw
    schema: raw
    tables:
      - name: events
        loaded_at_field: created_at
        freshness:
          warn_after: { count: 6, period: hour }
          error_after: { count: 24, period: hour }
      - name: products
      - name: users
```

### Step 5 — Staging models (45 min)
```sql
-- models/staging/stg_events.sql
{{ config(materialized='view') }}
SELECT
    event_id,
    user_id,
    product_id,
    event_type,
    CAST(ts AS TIMESTAMP) AS event_at,
    DATE(ts) AS event_date,
    LOWER(country) AS country
FROM {{ source('raw', 'events') }}
WHERE event_type IN ('view','click','purchase')
```

Repeat for products and users.

### Step 6 — Marts (45 min)
```sql
-- models/marts/dim_users.sql
{{ config(materialized='table') }}
SELECT user_id, country, tier, signup_date
FROM {{ ref('stg_users') }}

-- models/marts/fct_user_daily_activity.sql
{{ config(materialized='incremental', incremental_strategy='delete+insert', unique_key=['user_id','event_date']) }}
SELECT
    user_id,
    event_date,
    COUNT(*) FILTER (WHERE event_type = 'view') AS views,
    COUNT(*) FILTER (WHERE event_type = 'click') AS clicks,
    COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases,
    SUM(amount) FILTER (WHERE event_type = 'purchase') AS revenue
FROM {{ ref('stg_events') }}
{% if is_incremental() %}
  WHERE event_date >= (SELECT MAX(event_date) FROM {{ this }})
{% endif %}
GROUP BY 1, 2
```

### Step 7 — Tests (30 min)
```yaml
# models/schema.yml
version: 2
models:
  - name: dim_users
    columns:
      - name: user_id
        tests: [unique, not_null]
      - name: country
        tests:
          - accepted_values: { values: ["US","UK","DE","FR","JP"] }
  - name: fct_user_daily_activity
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns: [user_id, event_date]
```
Run:
```bash
dbt deps
dbt test
```

### Step 8 — Run end-to-end (15 min)
```bash
dbt source freshness   # warns if raw is stale
dbt build              # run models + tests in correct order
dbt docs generate && dbt docs serve   # http://localhost:8080
```

### Step 9 — Airflow orchestration (30 min)
```python
from airflow.providers.dbt.cloud.operators.dbt import DbtCloudRunJobOperator
# or just BashOperator:
from airflow.operators.bash import BashOperator

with DAG("dbt_daily", schedule="0 4 * * *", ...) as dag:
    load_raw = PythonOperator(task_id="load_raw", python_callable=load_raw_data)
    dbt = BashOperator(task_id="dbt_build", bash_command="cd /opt/dbt && dbt build --target prod")
    load_raw >> dbt
```

## Deliverables

1. dbt project with staging + marts models.
2. Test coverage including unique, not_null, accepted_values.
3. Generated dbt docs site.
4. Airflow DAG orchestrating `dbt build`.
5. `MIGRATION_NOTES.md` documenting what moved from Python/Spark to dbt and why.

## Validation

- [ ] `dbt build` succeeds with 0 test failures.
- [ ] `dbt docs serve` shows the lineage graph from raw → staging → marts.
- [ ] Source freshness check warns when raw table is artificially aged.
- [ ] Incremental model `fct_user_daily_activity` only processes new partitions on re-run.

## Stretch goals

- Add **dbt-expectations** package for richer data tests (similar to Great Expectations in dbt).
- Add **snapshots** to track slowly-changing dimensions.
- Implement **dbt Cloud-style CI**: PR check that runs `dbt build --select state:modified+` against a sandbox database.

## Common pitfalls

- **Source freshness uses warehouse timestamp** — Make sure the `loaded_at_field` is meaningful, not last-write-from-dbt.
- **`incremental` model that's not idempotent** — `unique_key` + `merge` is the safest pattern; `append` can duplicate.
- **Tests run in production every build** — Costly. Tag tests, run light ones every build, heavy ones nightly.
- **dbt depends on warehouse compute** — A bad model can run away with credits. Set query timeouts on the connection.
