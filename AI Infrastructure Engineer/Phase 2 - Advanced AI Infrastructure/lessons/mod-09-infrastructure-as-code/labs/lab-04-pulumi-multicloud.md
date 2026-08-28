# Lab 04: Pulumi Multi-Cloud Deployment

**Duration:** 75 min  **Prerequisites:** Pulumi CLI, AWS + GCP CLIs configured

## Objective
Use Pulumi (Python flavor) to provision a small workload on both AWS and GCP from one project, demonstrating cross-cloud abstraction in real programming language.

## Steps

### 1. Install
```bash
brew install pulumi
pulumi version
pip install pulumi pulumi-aws pulumi-gcp
```

### 2. New project
```bash
mkdir multicloud && cd multicloud
pulumi new python -y --name multicloud
```

### 3. __main__.py
```python
import pulumi
import pulumi_aws as aws
import pulumi_gcp as gcp

# AWS S3 bucket
aws_bucket = aws.s3.BucketV2("artifacts-aws", force_destroy=True)
aws.s3.BucketObject("model",
    bucket=aws_bucket.id,
    key="iris/model.joblib",
    source=pulumi.FileAsset("../model.joblib"))

# GCP storage bucket
gcp_bucket = gcp.storage.Bucket("artifacts-gcp",
    location="US",
    force_destroy=True)
gcp.storage.BucketObject("model",
    bucket=gcp_bucket.name,
    name="iris/model.joblib",
    source=pulumi.FileAsset("../model.joblib"))

pulumi.export("aws_bucket", aws_bucket.id)
pulumi.export("gcp_bucket", gcp_bucket.name)
```

### 4. Configure providers
```bash
pulumi config set aws:region us-west-2
pulumi config set gcp:project <PROJECT_ID>
pulumi config set gcp:region us-central1
```

### 5. Apply
```bash
pulumi up
```
Review the plan; type `yes`.

### 6. Stacks for environments
```bash
pulumi stack init prod
pulumi config set aws:region us-east-1 --stack prod
pulumi up --stack prod
```

### 7. Inspect outputs
```bash
pulumi stack output
pulumi stack output aws_bucket
```

### 8. Destroy
```bash
pulumi destroy
pulumi stack rm dev --yes
```

## Validation
- [ ] Pulumi plan correctly shows resources from both providers.
- [ ] Apply succeeds.
- [ ] Both buckets exist (verify in AWS + GCP consoles).
- [ ] Switching stack changes region without code changes.

## Cleanup
```bash
pulumi destroy --yes
```

## Troubleshooting
- **`Pulumi.yaml not found`** — Run `pulumi new` first.
- **AWS creds picked up from wrong profile** — `export AWS_PROFILE=...` or set in `Pulumi.yaml`.
- **GCP "Permission denied"** — `gcloud auth application-default login` to set ADC.
