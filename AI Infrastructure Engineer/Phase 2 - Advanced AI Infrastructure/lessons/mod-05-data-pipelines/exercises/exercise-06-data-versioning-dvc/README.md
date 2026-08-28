# Exercise 06: Data and Model Versioning with DVC

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Git, Python; lab 05 (DVC overview)

## Objective

Build a DVC-managed ML project where data, code, models, and metrics are reproducibly versioned together. Implement a multi-stage pipeline, demonstrate parameter sweeping with `dvc exp run`, and use `dvc repro` to rebuild only what changed.

## Why this matters

"It worked last week" failures in ML are usually data-related. DVC turns data into a first-class Git citizen: you can `git checkout` to a SHA and `dvc checkout` brings back the exact training data + model + metrics that produced the result. This is reproducibility you can audit.

## Requirements

1. **Pipeline** with 4+ stages (ingest, preprocess, train, evaluate).
2. **Parameters** in `params.yaml` consumed by stages.
3. **Metrics** declared per stage.
4. **Experiments** via `dvc exp run` with parameter overrides.
5. **Remote storage** (S3 or local filesystem).
6. **CI workflow** that runs `dvc repro` on PR and compares metrics.

## Step-by-step

### Step 1 — Initialize (15 min)
```bash
mkdir dvc-project && cd dvc-project
git init -q
python -m venv venv && source venv/bin/activate
pip install 'dvc[s3]' scikit-learn pandas joblib pyyaml

dvc init
git add .dvc .dvcignore && git commit -m "init dvc"
```

### Step 2 — Local remote (5 min)
```bash
mkdir -p ~/dvc-remote
dvc remote add -d localstore ~/dvc-remote
git add .dvc/config && git commit -m "dvc: local remote"
```

### Step 3 — Pipeline definition (45 min)
```yaml
# dvc.yaml
stages:
  ingest:
    cmd: python src/ingest.py --output data/raw.parquet
    deps: [src/ingest.py]
    outs: [data/raw.parquet]

  preprocess:
    cmd: python src/preprocess.py
    deps: [src/preprocess.py, data/raw.parquet]
    params: [preprocess.test_size, preprocess.random_seed]
    outs: [data/train.parquet, data/test.parquet]

  train:
    cmd: python src/train.py
    deps: [src/train.py, data/train.parquet]
    params: [train.algorithm, train.hyperparameters]
    outs: [models/model.joblib]
    metrics: [metrics/train.json]

  evaluate:
    cmd: python src/evaluate.py
    deps: [src/evaluate.py, models/model.joblib, data/test.parquet]
    metrics: [metrics/evaluate.json]
    plots:
      - plots/roc.json:
          x: fpr
          y: tpr
```

### Step 4 — Parameter file (15 min)
```yaml
# params.yaml
preprocess:
  test_size: 0.2
  random_seed: 42
train:
  algorithm: gradient_boosting
  hyperparameters:
    learning_rate: 0.05
    n_estimators: 300
    max_depth: 6
```

### Step 5 — Sample sources (30 min)
```python
# src/ingest.py
import pandas as pd
from sklearn.datasets import load_iris
df = load_iris(as_frame=True).frame
df.to_parquet("data/raw.parquet")
```

```python
# src/train.py
import json, joblib, yaml, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["train"]
df = pd.read_parquet("data/train.parquet")
X, y = df.drop(columns=["target"]), df["target"]
model = GradientBoostingClassifier(**params["hyperparameters"]).fit(X, y)
joblib.dump(model, "models/model.joblib")
json.dump({"train_accuracy": float(accuracy_score(y, model.predict(X)))},
          open("metrics/train.json", "w"))
```

(`preprocess.py` and `evaluate.py` similar.)

### Step 6 — Run + commit (10 min)
```bash
mkdir -p data models metrics plots
dvc repro
git add dvc.yaml dvc.lock params.yaml src/ metrics/ plots/
git commit -m "feat: initial pipeline"
dvc push
```

### Step 7 — Experiments (30 min)
```bash
dvc exp run --set-param train.hyperparameters.learning_rate=0.1
dvc exp run --set-param train.hyperparameters.learning_rate=0.01
dvc exp run --set-param train.algorithm=random_forest \
            --set-param train.hyperparameters.n_estimators=500

dvc exp show       # tabular comparison
dvc exp apply <exp-name>     # promote an experiment to working dir
```

### Step 8 — CI workflow (30 min)
```yaml
# .github/workflows/dvc-ci.yml
on: pull_request
jobs:
  reproduce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: iterative/setup-dvc@v1
      - uses: iterative/setup-cml@v2
      - run: pip install -r requirements.txt
      - run: dvc pull
      - run: dvc repro
      - run: |
          dvc metrics diff main --md >> report.md
          dvc plots diff main --target plots/roc.json --md --out roc-diff.png >> report.md
          cml comment create report.md
```

PR now sees: "+0.012 accuracy" inline.

## Deliverables

1. Working DVC project with all 4 stages.
2. At least 3 experiments showing different hyperparameter combinations.
3. CI workflow posting metric diffs on PR.
4. `EXPERIMENTS.md` documenting your findings from the sweep.

## Validation

- [ ] `dvc repro` is idempotent (rerun does nothing).
- [ ] Editing a single source file re-runs only the affected downstream stages.
- [ ] `dvc exp show` displays at least 3 experiments side-by-side.
- [ ] PR check posts a metrics diff.

## Stretch goals

- Use **MLflow** alongside DVC; compare ergonomics.
- Set up a **shared S3 remote** for team collaboration.
- Implement **data quality gates** between stages (using exercise 07's GE pipeline).

## Common pitfalls

- **`dvc repro` always re-runs every stage** — Output not declared in `outs:`. Without it, DVC can't hash the output to detect changes.
- **Committing data instead of `.dvc` pointers** — `data/raw.parquet` should be in `.gitignore`; the `.dvc` file gets committed.
- **`dvc push` writes nothing** — Remote not set. Check `dvc remote default`.
- **Parameter changes don't trigger reruns** — Stage's `params:` list doesn't include the parameter you changed.
