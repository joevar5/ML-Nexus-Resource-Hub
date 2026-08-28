# Lesson 04: Data Versioning with DVC

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand why data versioning is critical for ML
- Install and configure DVC (Data Version Control)
- Track large datasets with DVC and Git
- Set up remote storage backends (S3, GCS, Azure)
- Create reproducible data pipelines with DVC
- Share datasets and models across teams
- Integrate DVC with CI/CD workflows

## Prerequisites
- Git version control basics
- Understanding of ML workflows
- Familiarity with cloud storage (S3/GCS)
- Python programming

## Introduction

**DVC (Data Version Control)** brings Git-like versioning to large datasets and ML models.

**Why version data?**
- **Reproducibility**: Recreate exact training environment
- **Collaboration**: Share datasets without copying TBs of data
- **Experimentation**: Track which data produced which model
- **Rollback**: Revert to previous dataset versions
- **Lineage**: Understand data → model → metrics relationships

**Real-world usage:**
- **Iterative.ai**: Created DVC, open-source tool with 11k+ GitHub stars
- **Hugging Face**: Uses DVC for dataset versioning
- **Many startups**: Replace manual dataset management with DVC

## 1. Installing and Configuring DVC

### 1.1 Installation

```bash
# Install DVC
pip install dvc

# Install cloud storage support
pip install 'dvc[s3]'      # AWS S3
pip install 'dvc[gs]'      # Google Cloud Storage
pip install 'dvc[azure]'   # Azure Blob Storage
pip install 'dvc[all]'     # All providers

# Verify installation
dvc version
# DVC version: 3.28.0
```

### 1.2 Initialize DVC in Git Repository

```bash
# Initialize Git repo (if not already done)
git init

# Initialize DVC
dvc init

# DVC creates:
# .dvc/            # DVC configuration and cache
# .dvc/.gitignore  # Ignore DVC cache in Git
# .dvc/config      # DVC configuration
# .dvcignore       # Files to ignore

# Commit DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 1.3 Configure Remote Storage

**AWS S3:**

```bash
# Add S3 as remote storage
dvc remote add -d myremote s3://my-ml-data-bucket/dvc-cache

# Configure AWS credentials (use environment variables or AWS config)
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY
dvc remote modify myremote region us-west-2

# Or use IAM role (recommended in production)
dvc remote modify myremote use_ssl true
dvc remote modify myremote credentialpath ~/.aws/credentials

# Commit remote configuration
git add .dvc/config
git commit -m "Configure S3 remote storage"
```

**Google Cloud Storage:**

```bash
dvc remote add -d myremote gs://my-ml-data-bucket/dvc-cache
dvc remote modify myremote projectname my-gcp-project

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

**Azure Blob Storage:**

```bash
dvc remote add -d myremote azure://mycontainer/dvc-cache
dvc remote modify myremote account_name mystorageaccount
dvc remote modify myremote account_key $AZURE_STORAGE_KEY
```

## 2. Tracking Datasets with DVC

### 2.1 Add Large Files to DVC

```bash
# Download dataset (example)
wget https://example.com/imagenet.tar.gz -O data/imagenet.tar.gz

# Add to DVC tracking
dvc add data/imagenet.tar.gz

# DVC creates:
# data/imagenet.tar.gz.dvc  # Metadata file (tracked in Git)
# data/.gitignore           # Ignore actual file in Git

# Commit metadata to Git
git add data/imagenet.tar.gz.dvc data/.gitignore
git commit -m "Track ImageNet dataset with DVC"

# Push data to remote storage
dvc push
```

**What happens:**
1. DVC calculates MD5 hash of file
2. Stores file in `.dvc/cache/`
3. Creates `.dvc` metadata file with hash
4. `.gitignore` prevents tracking large file in Git
5. `dvc push` uploads file to S3/GCS

**File structure:**

```
data/imagenet.tar.gz.dvc (tracked in Git):
  outs:
  - md5: a3e4f5... (hash of file)
    size: 150000000000  # 150 GB
    path: imagenet.tar.gz

.dvc/cache/a3/e4f5... (local cache, not in Git)
  → Actual 150 GB file

s3://bucket/dvc-cache/a3/e4f5... (remote storage)
  → Uploaded copy
```

### 2.2 Retrieving Data

```bash
# Clone repository (gets .dvc metadata files)
git clone https://github.com/company/ml-project.git
cd ml-project

# Download data from remote storage
dvc pull

# DVC downloads files based on .dvc metadata
# Files appear in data/ directory
```

### 2.3 Adding Directories

```bash
# Track entire directory
dvc add data/raw/

# Creates data/raw.dvc with all files
git add data/raw.dvc data/.gitignore
git commit -m "Track raw data directory"
dvc push
```

### 2.4 Updating Datasets

```bash
# Modify dataset
python scripts/add_new_samples.py

# DVC detects changes
dvc status
# Output: data/imagenet.tar.gz.dvc:
#   changed outs:
#     modified:    data/imagenet.tar.gz

# Update DVC tracking
dvc add data/imagenet.tar.gz

# Commit new version
git add data/imagenet.tar.gz.dvc
git commit -m "Update ImageNet with 10k new samples"
dvc push

# Now have versioned history:
# - v1.0: 140 GB, 1.2M images (commit abc123)
# - v1.1: 150 GB, 1.21M images (commit def456)
```

### 2.5 Switching Between Versions

```bash
# Checkout specific version
git checkout v1.0  # Git tag or commit hash
dvc checkout       # Downloads v1.0 data

# Data now matches v1.0 state!

# Return to latest
git checkout main
dvc checkout
```

## 3. DVC Pipelines

### 3.1 Define Reproducible Pipelines

**DVC pipelines track data transformations:**

```yaml
# dvc.yaml
stages:
  download:
    cmd: python scripts/download_data.py
    outs:
      - data/raw/

  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/
      - scripts/preprocess.py
    params:
      - preprocess.min_samples
      - preprocess.max_samples
    outs:
      - data/processed/

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/
      - scripts/train.py
    params:
      - train.epochs
      - train.learning_rate
    metrics:
      - metrics.json:
          cache: false
    outs:
      - models/model.pth
```

```yaml
# params.yaml (hyperparameters)
preprocess:
  min_samples: 1000
  max_samples: 100000

train:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

**Run pipeline:**

```bash
# Run all stages
dvc repro

# DVC:
# 1. Checks if inputs changed
# 2. Skips stages where inputs unchanged (caching!)
# 3. Runs only changed stages
# 4. Tracks outputs and metrics

# Commit pipeline and results
git add dvc.yaml dvc.lock metrics.json
git commit -m "Train model v1.0"
dvc push
```

### 3.2 Pipeline Benefits

**Reproducibility:**

```bash
# Reproduce exact results from 6 months ago
git checkout 6-months-ago-commit
dvc repro

# DVC downloads exact data and reruns pipeline
# Produces identical results!
```

**Caching:**

```bash
# Change only hyperparameters
vim params.yaml  # Change learning_rate: 0.001 → 0.01

dvc repro
# DVC output:
# Stage 'download' didn't change, skipping
# Stage 'preprocess' didn't change, skipping
# Stage 'train' changed, running...

# Only trains model, skips preprocessing!
```

### 3.3 Complex Pipeline Example

```yaml
# dvc.yaml (ML pipeline)
stages:
  fetch_data:
    cmd: python scripts/fetch_from_db.py --date ${date}
    params:
      - fetch.start_date
      - fetch.end_date
    outs:
      - data/raw/transactions.parquet

  validate_data:
    cmd: python scripts/validate.py
    deps:
      - data/raw/transactions.parquet
      - scripts/validate.py
    outs:
      - data/validated/transactions.parquet

  feature_engineering:
    cmd: python scripts/features.py
    deps:
      - data/validated/transactions.parquet
      - scripts/features.py
    params:
      - features.window_days
      - features.aggregations
    outs:
      - data/features/user_features.parquet

  train_model:
    cmd: python scripts/train.py
    deps:
      - data/features/user_features.parquet
      - scripts/train.py
    params:
      - model.type
      - model.hyperparameters
    outs:
      - models/fraud_detector.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.png:
          cache: false

  evaluate_model:
    cmd: python scripts/evaluate.py
    deps:
      - models/fraud_detector.pkl
      - data/features/user_features.parquet
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

```bash
# Run pipeline
dvc repro

# View metrics
dvc metrics show

# Compare experiments
dvc exp show --only-changed
```

## 4. Experiment Tracking with DVC

### 4.1 Running Experiments

```bash
# Run experiment with different parameters
dvc exp run -S train.learning_rate=0.01 -S train.batch_size=64

# Run multiple experiments in queue
dvc exp run -S train.learning_rate=0.001 --queue
dvc exp run -S train.learning_rate=0.01 --queue
dvc exp run -S train.learning_rate=0.1 --queue

# Execute queued experiments
dvc exp run --run-all --jobs 3  # Run 3 in parallel
```

### 4.2 Comparing Experiments

```bash
# List experiments
dvc exp show

# Output:
# ┌────────────────────┬──────────┬────────┬──────────┬──────────┐
# │ Experiment         │ accuracy │ f1     │ train.lr │ train.bs │
# ├────────────────────┼──────────┼────────┼──────────┼──────────┤
# │ main               │ 0.92     │ 0.89   │ 0.001    │ 32       │
# │ exp-lr-0.01        │ 0.94     │ 0.91   │ 0.01     │ 32       │
# │ exp-lr-0.1         │ 0.88     │ 0.85   │ 0.1      │ 32       │
# │ exp-bs-64          │ 0.93     │ 0.90   │ 0.001    │ 64       │
# └────────────────────┴──────────┴────────┴──────────┴──────────┘

# Compare specific experiments
dvc exp diff exp-lr-0.01 exp-lr-0.1

# Apply best experiment to workspace
dvc exp apply exp-lr-0.01

# Commit best experiment
git add dvc.lock params.yaml metrics.json
git commit -m "Use learning rate 0.01 (best accuracy: 0.94)"
```

### 4.3 Experiment Plots

```bash
# Generate plots comparing experiments
dvc plots show

# Compare metric across experiments
dvc plots diff --targets metrics/train_metrics.json -x epoch -y loss

# Generates interactive HTML plot
```

## 5. Collaboration Workflows

### 5.1 Sharing Data with Team

**Team member A:**

```bash
# Add new dataset
dvc add data/new_dataset.parquet
git add data/new_dataset.parquet.dvc .gitignore
git commit -m "Add new dataset"
git push origin main
dvc push  # Upload to S3
```

**Team member B:**

```bash
# Pull latest code
git pull origin main

# Download new data
dvc pull

# Data now available locally!
```

### 5.2 Working with Large Teams

**Best practices:**

1. **Use shared remote storage:**
   ```bash
   # Everyone uses same S3 bucket
   dvc remote add -d shared s3://team-ml-data/
   ```

2. **Don't commit cache to Git:**
   ```
   .dvc/cache/ is in .gitignore (default)
   ```

3. **Pull before training:**
   ```bash
   git pull
   dvc pull
   dvc repro
   ```

4. **Push after training:**
   ```bash
   dvc add models/new_model.pkl
   git add models/new_model.pkl.dvc
   git commit -m "Train new model"
   git push
   dvc push
   ```

## 6. Integrating DVC with CI/CD

### 6.1 GitHub Actions Example

```yaml
# .github/workflows/train-model.yml
name: Train ML Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install dvc[s3]

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Pull data from DVC
      run: dvc pull

    - name: Run DVC pipeline
      run: dvc repro

    - name: Push results to DVC
      run: dvc push

    - name: Commit metrics
      run: |
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add dvc.lock metrics.json
        git commit -m "Update model metrics [skip ci]" || echo "No changes"
        git push
```

### 6.2 Airflow Integration

```python
# dags/dvc_training_pipeline.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG('dvc_training_pipeline', start_date=datetime(2023, 1, 1), schedule_interval='@daily')

pull_data = BashOperator(
    task_id='pull_data',
    bash_command='cd /opt/ml-project && dvc pull',
    dag=dag,
)

run_pipeline = BashOperator(
    task_id='run_pipeline',
    bash_command='cd /opt/ml-project && dvc repro',
    dag=dag,
)

push_results = BashOperator(
    task_id='push_results',
    bash_command='cd /opt/ml-project && dvc push',
    dag=dag,
)

commit_results = BashOperator(
    task_id='commit_results',
    bash_command='''
        cd /opt/ml-project
        git add dvc.lock metrics.json
        git commit -m "Training run {{ ds }}"
        git push
    ''',
    dag=dag,
)

pull_data >> run_pipeline >> push_results >> commit_results
```

## 7. Best Practices

### 7.1 What to Track with DVC

**✅ Track with DVC:**
- Large datasets (> 10 MB)
- Model files (.pkl, .pth, .h5)
- Preprocessed/intermediate data
- Model evaluation results (plots, reports)

**❌ Don't track with DVC:**
- Source code (use Git)
- Small config files (use Git)
- Secrets/credentials (use secrets management)

### 7.2 Directory Structure

```
project/
├── data/
│   ├── raw/              # DVC tracked
│   ├── processed/        # DVC tracked
│   └── features/         # DVC tracked
├── models/               # DVC tracked
├── metrics/              # DVC tracked (if large)
├── plots/                # DVC tracked
├── scripts/              # Git tracked
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── dvc.yaml              # Git tracked
├── dvc.lock              # Git tracked
├── params.yaml           # Git tracked
├── requirements.txt      # Git tracked
└── README.md             # Git tracked
```

### 7.3 Performance Tips

**Use cache for faster downloads:**

```bash
# DVC cache shared across all your projects
# ~/.dvc/cache/

# Link files instead of copying (saves disk space)
dvc config cache.type symlink  # Or 'hardlink' or 'reflink'
```

**Partial data pull:**

```bash
# Pull only specific files
dvc pull data/processed/

# Pull without downloading data (just update .dvc files)
dvc fetch --run-cache
```

## 8. Summary

### Key Takeaways

✅ **DVC enables Git-like versioning for data and models**
- Track large files without storing them in Git
- Store data in S3/GCS/Azure, metadata in Git

✅ **Core workflows:**
- `dvc add` → `git add *.dvc` → `git commit` → `dvc push`
- `git pull` → `dvc pull` to get latest data

✅ **DVC pipelines provide reproducibility:**
- Define stages in `dvc.yaml`
- Track dependencies, parameters, outputs, metrics
- `dvc repro` runs only changed stages (caching!)

✅ **Experiment tracking:**
- `dvc exp run` for parameter sweeps
- `dvc exp show` to compare results
- Apply best experiment and commit

✅ **Collaboration:**
- Shared remote storage for teams
- Avoid duplicate data downloads with cache
- Integrate with CI/CD (GitHub Actions, Airflow)

### Real-World Impact

Companies versioning data with DVC:
- **Iterative.ai**: DVC creators, manage datasets for ML
- **Hugging Face**: Dataset versioning and sharing
- **Many ML teams**: Replace manual "data_v1", "data_v2" folders

## Self-Check Questions

1. What command tracks a large dataset with DVC?
2. How do you configure S3 as a DVC remote?
3. What files does `dvc add data.csv` create?
4. How do you download data after cloning a DVC repository?
5. What's the benefit of DVC pipelines over manual scripts?
6. How do you run multiple experiments with different hyperparameters?
7. Why not track source code with DVC?
8. How does DVC caching save time and disk space?

## Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Get Started Guide](https://dvc.org/doc/start)
- [DVC vs MLflow](https://dvc.org/doc/user-guide/related-technologies)
- [DVC Studio](https://studio.iterative.ai/) (Web UI for DVC)

---

**Next lesson:** Data Processing with Apache Spark for large-scale ML datasets!
