# Lab 06: Validate Data with Great Expectations

**Duration:** 60 min  **Prerequisites:** Python 3.11+

## Objective
Build a Great Expectations suite for an ML training dataset, run it as a checkpoint, fail loud on quality violations, and integrate the run into an Airflow DAG.

## Steps

### 1. Install
```bash
python -m venv venv && source venv/bin/activate
pip install 'great-expectations==0.18.16' pandas pyarrow
```

### 2. Initialize a project
```bash
great_expectations init
# Choose: PandasExecutionEngine, InferredAsset filesystem connector, base dir ./data
```

### 3. Create a tiny dataset
```python
# create_data.py
import pandas as pd, numpy as np
df = pd.DataFrame({
    "user_id": np.arange(1000),
    "amount": np.random.gamma(2, 10, 1000),
    "country": np.random.choice(["US","UK","DE","JP"], 1000),
    "label":  np.random.choice([0,1], 1000, p=[0.97,0.03]),
})
df.to_parquet("data/train.parquet")
```
Run it.

### 4. Build a suite
```bash
great_expectations suite new
# Pick the train.parquet asset, name suite "training_data_suite", choose "Profile data".
```

This opens a Jupyter notebook with auto-generated expectations. Review, prune, save.

### 5. Manually add critical expectations
```python
# in the notebook
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_in_set("label", [0, 1])
validator.expect_column_mean_to_be_between("label", min_value=0.01, max_value=0.10)
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=10_000)
validator.save_expectation_suite(discard_failed_expectations=False)
```

### 6. Create a checkpoint and run
```bash
great_expectations checkpoint new training_data_checkpoint
# Wire it to the suite and data asset
great_expectations checkpoint run training_data_checkpoint
```
Exit code: 0 on pass, non-zero on failure.

### 7. Break it
```python
# corrupt some labels
df.loc[df.index[:5], "label"] = 2
df.to_parquet("data/train.parquet")
```
Re-run checkpoint. Expect failure on the `expect_column_values_to_be_in_set` expectation.

### 8. Use in Airflow
```python
from airflow.operators.bash import BashOperator
validate = BashOperator(
    task_id="validate_training_data",
    bash_command="cd /opt/airflow/ge && great_expectations checkpoint run training_data_checkpoint",
)
```
Validate task fails → downstream `train` doesn't run. That's the point.

## Validation
- [ ] Checkpoint passes on clean data.
- [ ] Checkpoint fails (exit code != 0) on corrupted data.
- [ ] HTML data docs in `uncommitted/data_docs/local_site/` show pass/fail history.

## Cleanup
```bash
deactivate && rm -rf venv data great_expectations
```

## Troubleshooting
- **Checkpoint runs but never fails** — `expect_column_mean_to_be_between` has wide bounds. Tighten them.
- **Notebook can't find Jupyter** — `pip install jupyterlab`.
- **`PandasExecutionEngine` not found in newer GE** — Newer versions use Fluent API; the legacy DataContext approach still works in 0.18.
