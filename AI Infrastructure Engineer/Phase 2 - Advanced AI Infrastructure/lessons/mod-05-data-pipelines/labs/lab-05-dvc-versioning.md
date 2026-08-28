# Lab 05: Version Data and Models with DVC

**Duration:** 60 min  **Prerequisites:** Git, Python 3.11+

## Objective
Track large data files and model artifacts under DVC with Git for metadata. Pin a specific dataset version, train against it, and reproduce results months later.

## Steps

### 1. Initialize
```bash
mkdir dvc-lab && cd dvc-lab
git init
python -m venv venv && source venv/bin/activate
pip install 'dvc[s3]' scikit-learn joblib
dvc init
git add .dvc .dvcignore && git commit -m "init dvc"
```

### 2. Set up a local remote (real projects use S3/GCS/Azure)
```bash
mkdir -p ~/dvc-remote
dvc remote add -d localstore ~/dvc-remote
git add .dvc/config && git commit -m "dvc: local remote"
```

### 3. Track a dataset
```bash
mkdir data
python -c "from sklearn.datasets import load_iris; import pandas as pd; \
  df = load_iris(as_frame=True).frame; df.to_parquet('data/iris.parquet')"
dvc add data/iris.parquet
git add data/.gitignore data/iris.parquet.dvc && git commit -m "dvc: track iris v1"
dvc push
```

### 4. Modify and re-track
```bash
python -c "from sklearn.datasets import load_iris; import pandas as pd; \
  df = load_iris(as_frame=True).frame; df = df[df['sepal length (cm)'] > 5]; \
  df.to_parquet('data/iris.parquet')"
dvc add data/iris.parquet
git add data/iris.parquet.dvc && git commit -m "dvc: iris v2 (filtered)"
dvc push
```

### 5. Roll back
```bash
git checkout HEAD~1 data/iris.parquet.dvc
dvc checkout
wc -l data/iris.parquet   # back to v1 size
```

### 6. Pipeline with stages
```bash
cat > dvc.yaml <<'EOF'
stages:
  train:
    cmd: python train.py
    deps: [data/iris.parquet, train.py]
    outs: [model.joblib]
    metrics: [metrics.json]
EOF

cat > train.py <<'EOF'
import json, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/iris.parquet")
X = df.drop(columns=["target"]); y = df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xtr, ytr)
joblib.dump(model, "model.joblib")
json.dump({"accuracy": float(accuracy_score(yte, model.predict(Xte)))}, open("metrics.json","w"))
EOF

dvc repro
git add dvc.yaml dvc.lock train.py && git commit -m "dvc: training pipeline"
dvc push
```

### 7. Reproduce later
```bash
git checkout <commit-hash>
dvc checkout      # pulls matching data + model
cat metrics.json
```

## Validation
- [ ] `data/iris.parquet` is in `.gitignore` and tracked by DVC.
- [ ] `dvc.lock` is committed (it pins the input/output hashes).
- [ ] `dvc repro` re-runs only the stages whose inputs changed.
- [ ] Rolling back the git commit restores the exact data + model.

## Cleanup
```bash
cd .. && rm -rf dvc-lab ~/dvc-remote
```

## Troubleshooting
- **`dvc push` writes nothing** — Remote not configured; check `dvc remote default`.
- **`dvc checkout` says "missing data"** — Your remote doesn't have the version; check `dvc fetch`.
- **Pipeline always re-runs** — Output not declared in `outs` so DVC can't hash it.
