# Lab 04: CI/CD for Models with GitHub Actions

**Duration:** 75 min  **Prerequisites:** GitHub account, repo with `iris-api` from earlier labs

## Objective
On every push to `main` that touches model code: train, evaluate against a held-out gate (≥95% accuracy), build the image, push to GHCR, and deploy to a `staging` environment automatically.

## Steps

### 1. Repo structure
```
.
├── src/train.py
├── src/api.py
├── Dockerfile
├── tests/
└── .github/workflows/ml-cicd.yml
```

### 2. Training script with quality gate
```python
# src/train.py
import json, sys, joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
acc = accuracy_score(yte, model.predict(Xte))
print(f"accuracy={acc:.4f}")
joblib.dump(model, "model.joblib")
json.dump({"accuracy": acc}, open("metrics.json", "w"))
if acc < 0.95:
    print("FAIL: accuracy below 0.95 threshold")
    sys.exit(1)
```

### 3. Workflow
```yaml
# .github/workflows/ml-cicd.yml
name: ml-cicd

on:
  push:
    branches: [main]
    paths: ['src/**','Dockerfile','requirements.txt']

permissions:
  contents: read
  packages: write
  id-token: write

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Train
        run: python src/train.py
      - name: Upload metrics
        uses: actions/upload-artifact@v4
        with: { name: metrics, path: metrics.json }
      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}/iris-api:${{ github.sha }}
      - name: Deploy to staging
        run: |
          kubectl --context=staging set image deployment/iris-api \
            api=ghcr.io/${{ github.repository }}/iris-api:${{ github.sha }}
          kubectl --context=staging rollout status deployment/iris-api --timeout=3m
        env:
          KUBECONFIG: ${{ secrets.STAGING_KUBECONFIG }}
```

### 4. Add `secrets.STAGING_KUBECONFIG` in repo settings.

### 5. Push a change
Edit `src/train.py` (e.g., bump `n_estimators`). Commit and push. Watch the workflow:
- Training gate (if accuracy drops, build halts)
- Image build + push
- Staging rollout

### 6. Verify in staging
`curl https://<staging-domain>/health`

## Validation
- [ ] Workflow runs end-to-end on push.
- [ ] Failing the accuracy gate halts the build (forced by editing train.py to use bad params).
- [ ] Image visible in GHCR.
- [ ] Staging cluster pods running the new image tag (SHA).

## Cleanup
```bash
# Delete the workflow file or just don't merge to main.
```

## Troubleshooting
- **`Permission denied` pushing to GHCR** — Add `packages: write` to `permissions`.
- **`kubectl: command not found`** — Add `azure/setup-kubectl@v4` action.
- **Staging deploy hangs** — Rollout timeout; investigate via SSH/kubectl on staging cluster.
