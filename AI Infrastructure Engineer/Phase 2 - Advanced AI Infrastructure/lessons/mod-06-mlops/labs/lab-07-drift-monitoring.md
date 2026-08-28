# Lab 07: Drift Monitoring with Evidently

**Duration:** 60 min  **Prerequisites:** Python 3.11+; baseline dataset on hand

## Objective
Compute feature and prediction drift between a baseline (training) dataset and a recent production sample. Produce an HTML report and a JSON summary suitable for alerting on.

## Steps

### 1. Install
```bash
pip install 'evidently==0.4.30' pandas pyarrow
```

### 2. Two datasets
```python
import pandas as pd, numpy as np
np.random.seed(42)

ref = pd.DataFrame({
    "feature_1": np.random.normal(0, 1, 5000),
    "feature_2": np.random.exponential(2, 5000),
    "category":  np.random.choice(["a","b","c"], 5000),
    "prediction": np.random.choice([0,1], 5000, p=[0.7,0.3]),
})

prod = pd.DataFrame({
    "feature_1": np.random.normal(0.4, 1.2, 5000),     # shifted
    "feature_2": np.random.exponential(3, 5000),        # scale changed
    "category":  np.random.choice(["a","b","c"], 5000, p=[0.5,0.3,0.2]),
    "prediction": np.random.choice([0,1], 5000, p=[0.6,0.4]),  # class balance shifted
})
```

### 3. Run drift report
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

rpt = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
rpt.run(reference_data=ref, current_data=prod, column_mapping=None)
rpt.save_html("drift_report.html")
res = rpt.as_dict()
```

### 4. Open the HTML
```bash
open drift_report.html
```
See per-feature drift scores, distributions, and a summary.

### 5. Extract pass/fail to JSON
```python
import json
summary = {
    "dataset_drift": res["metrics"][0]["result"]["dataset_drift"],
    "n_drifted":     res["metrics"][0]["result"]["number_of_drifted_columns"],
    "drift_by_column": {
        c["column_name"]: c["drift_score"]
        for c in res["metrics"][0]["result"]["drift_by_columns"].values()
    },
}
json.dump(summary, open("drift_summary.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
```

### 6. Schedule
Wrap in a daily Airflow DAG; alert if `dataset_drift` is True for 3 days in a row.

## Validation
- [ ] HTML report opens and shows distribution overlays for each feature.
- [ ] `n_drifted` ≥ 2 with the test data above.
- [ ] JSON summary is suitable for piping into an alerting system.

## Cleanup
```bash
rm drift_report.html drift_summary.json
```

## Troubleshooting
- **`dataset_drift` always False** — Default test (KS for numerical, chi2 for categorical) needs a real shift; the synthetic data here is calibrated to trigger.
- **HTML report too large to embed in Slack** — Use the JSON summary for alerting, link to the HTML for investigation.
- **Drift on a feature you don't care about** — Use `ColumnMapping` to exclude.
