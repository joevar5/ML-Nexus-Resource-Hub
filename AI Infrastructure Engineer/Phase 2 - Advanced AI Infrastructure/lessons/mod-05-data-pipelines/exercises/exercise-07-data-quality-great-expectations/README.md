# Exercise 07: Data Quality Validation with Great Expectations

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, lab 06 (GE overview)

## Objective

For a real dataset that powers an ML training pipeline, build a Great Expectations suite that catches the data quality issues your downstream models actually care about. Integrate as a pipeline gate, wire failures to Slack, and produce a data-quality dashboard.

## Why this matters

Models silently degrade when inputs degrade. Bad data costs money slowly: a 0.5pp accuracy regression for a quarter before anyone notices. Catching it at ingestion time is the cheapest place; tests-as-data-contracts is the discipline.

## Requirements

1. **Expectation suite** with 15+ expectations covering 4 categories:
   - Schema (columns present, types correct)
   - Per-column value rules (ranges, allowed sets, null tolerance)
   - Cross-column rules (foreign keys, business invariants)
   - Distribution rules (drift detection vs baseline)
2. **Checkpoint** runnable as `great_expectations checkpoint run training_data`.
3. **Pipeline integration**: failure blocks the training stage.
4. **Slack notification** on failure with a link to the data docs.
5. **Data docs** auto-generated + accessible (S3 bucket or local HTTP).

## Step-by-step

### Step 1 — Setup (15 min)
```bash
mkdir ge-quality && cd ge-quality
python -m venv venv && source venv/bin/activate
pip install 'great-expectations==0.18.16' pandas pyarrow

great_expectations init
# Choose: pandas, file-system data source, base dir ./data
```

### Step 2 — Generate sample data (15 min)
```python
# create_data.py
import pandas as pd, numpy as np
np.random.seed(42)
df = pd.DataFrame({
    "transaction_id": range(10_000),
    "user_id":        np.random.randint(1, 1000, 10_000),
    "amount":         np.random.gamma(2, 50, 10_000),
    "currency":       np.random.choice(["USD","EUR","GBP","JPY"], 10_000),
    "country":        np.random.choice(["US","UK","DE","FR","JP"], 10_000),
    "is_fraud":       np.random.choice([0,1], 10_000, p=[0.97,0.03]),
})
df.to_parquet("data/transactions.parquet")
```

### Step 3 — Build expectation suite (60 min)
Create via UI or programmatically:
```python
# build_suite.py
import great_expectations as gx

ctx = gx.get_context()
batch_request = ctx.get_datasource("default").get_asset("transactions").build_batch_request()
validator = ctx.get_validator(batch_request=batch_request, expectation_suite_name="training_data")

# Schema
validator.expect_table_columns_to_match_ordered_list([
    "transaction_id","user_id","amount","currency","country","is_fraud"
])

# Row count + uniqueness
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10_000_000)
validator.expect_column_values_to_be_unique("transaction_id")
validator.expect_column_values_to_not_be_null("transaction_id")

# Value ranges + allowed sets
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=1_000_000, mostly=0.999)
validator.expect_column_values_to_be_in_set("currency", ["USD","EUR","GBP","JPY","AUD","CAD"])
validator.expect_column_values_to_be_in_set("is_fraud", [0,1])

# Class balance (catches schema/labeling bugs)
validator.expect_column_mean_to_be_between("is_fraud", min_value=0.005, max_value=0.10)

# Cross-column / business invariants
# (custom: "USD txn from JP user is suspicious; allow only if amount > 1000")
validator.expect_compound_columns_to_be_unique(["transaction_id"])

# Distribution drift (catches upstream pipeline changes)
validator.expect_column_kl_divergence_to_be_less_than(
    "amount", partition_object={"bins":[0,10,50,100,500,1e6],"weights":[0.2,0.4,0.2,0.15,0.05]},
    threshold=0.5,
)

validator.save_expectation_suite()
```

### Step 4 — Checkpoint (15 min)
```bash
great_expectations checkpoint new training_data
# Wire to the asset + the suite; save
```

### Step 5 — Run + observe (10 min)
```bash
great_expectations checkpoint run training_data
# Exit 0: passed. Non-zero: failed.

great_expectations docs build
open great_expectations/uncommitted/data_docs/local_site/index.html
```

### Step 6 — Fail on purpose (15 min)
Corrupt the data: set 5 rows' `is_fraud` to 2 (invalid). Re-run. Observe failure + which expectation tripped.

### Step 7 — Airflow integration (30 min)
```python
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowFailException

validate = BashOperator(
    task_id="validate_training_data",
    bash_command="great_expectations checkpoint run training_data || exit 1",
)
# downstream train task only runs if validate succeeds
validate >> train
```

### Step 8 — Slack on failure + data docs link (30 min)
Add a `SlackAction` to the checkpoint config:
```yaml
action_list:
  - name: store_validation_result
    action: { class_name: StoreValidationResultAction }
  - name: notify_slack
    action:
      class_name: SlackNotificationAction
      slack_webhook: https://hooks.slack.com/services/...
      notify_on: failure
      renderer:
        module_name: great_expectations.render.renderer.slack_renderer
        class_name: SlackRenderer
```

Update with a link to the rendered HTML data docs (S3-hosted in prod).

## Deliverables

1. Expectation suite (`great_expectations/expectations/training_data.json`).
2. Checkpoint config (`great_expectations/checkpoints/training_data.yml`).
3. Airflow DAG (or DVC pipeline) with validate as a gate before train.
4. Slack webhook config + screenshot of an actual failure notification.
5. `DATA_QUALITY_PLAYBOOK.md`: when to add new expectations, how to triage failures.

## Validation

- [ ] Checkpoint passes on clean data.
- [ ] Checkpoint fails on corrupted data with a precise message.
- [ ] Pipeline DAG / DVC stage refuses to proceed on failure.
- [ ] Slack notification arrives within 30s.
- [ ] Data docs accessible and show pass/fail history.

## Stretch goals

- Add **per-segment** expectations (e.g., country=JP rows have different amount distribution).
- Add **profiling**: GE auto-suggests expectations from a "known good" baseline.
- Migrate to GE's **Fluent API** (newer, cleaner) and compare.

## Common pitfalls

- **Suite too strict** — Catches noise rather than signal. Use `mostly=0.99` for soft constraints.
- **Drift expectations on a moving target** — KL divergence to a fixed baseline alerts forever after a real seasonality shift. Periodically refresh the baseline.
- **`notify_on: any`** — Floods Slack on success too; use `notify_on: failure`.
- **Validation result store too large** — GE accumulates results forever. Set a retention policy in object storage.
