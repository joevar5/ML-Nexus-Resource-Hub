# Lesson 06: Container Registries

**Duration:** 4 hours
**Objectives:** Master container registries for storing and distributing ML Docker images

## Learning Objectives

By the end of this lesson, you will be able to:

1. Push and pull images from Docker Hub
2. Work with cloud registries (ECR, GCR, ACR)
3. Implement semantic versioning for ML images
4. Set up private registries for enterprise use
5. Integrate registries with CI/CD pipelines
6. Scan images for security vulnerabilities
7. Manage image lifecycle and cleanup policies

## What are Container Registries?

**Container registries** are centralized repositories for storing and distributing Docker images.

**Analogy:**
- **GitHub** is to code what **Docker Registry** is to container images
- Version control for deployable artifacts
- Share images across teams and environments

### Why Use Registries for ML?

**Without a registry:**
```bash
# Must build image on every machine
git clone repo
docker build -t ml-model .  # 20 minutes!
```

**With a registry:**
```bash
# Pull pre-built image
docker pull myregistry.com/ml-model:v1.0  # 2 minutes!
docker run myregistry.com/ml-model:v1.0
```

**Benefits:**
1. **Fast deployment** - No rebuild needed
2. **Consistency** - Same image everywhere
3. **Version control** - Track image versions
4. **Collaboration** - Share images with team
5. **CI/CD integration** - Automated builds and deployments
6. **Security scanning** - Automated vulnerability detection

## Docker Hub

### Overview

**Docker Hub** is the default public registry:
- Free public repositories
- 1 free private repository
- Official images (python, ubuntu, nginx, etc.)
- Automated builds from GitHub

### Creating Docker Hub Account

1. Visit https://hub.docker.com
2. Sign up for free account
3. Verify email
4. Create access token (Settings → Security)

### Pushing to Docker Hub

```bash
# 1. Login
docker login
# Enter username and password (or access token)

# 2. Tag image with your Docker Hub username
docker tag ml-model:v1.0 yourusername/ml-model:v1.0

# 3. Push image
docker push yourusername/ml-model:v1.0

# 4. Pull from anywhere
docker pull yourusername/ml-model:v1.0
```

### Image Naming Convention

```
[registry]/[username]/[repository]:[tag]

docker.io/johndoe/ml-model:v1.0
└─registry─┘└username┘└repository┘└tag┘
```

**Examples:**
```bash
# Docker Hub (default registry)
docker push johndoe/ml-model:v1.0
docker pull johndoe/ml-model:v1.0

# Docker Hub with explicit registry
docker push docker.io/johndoe/ml-model:v1.0

# Private registry
docker push myregistry.com/ml-model:v1.0
```

### Tags and Versioning

```bash
# Multiple tags for same image
docker tag ml-model:v1.0 johndoe/ml-model:v1.0
docker tag ml-model:v1.0 johndoe/ml-model:latest
docker tag ml-model:v1.0 johndoe/ml-model:prod

# Push all tags
docker push johndoe/ml-model:v1.0
docker push johndoe/ml-model:latest
docker push johndoe/ml-model:prod
```

### Private Repositories

```bash
# Create private repo on Docker Hub (via web interface)
# Then push as usual
docker push johndoe/private-ml-model:v1.0

# Pull requires authentication
docker login
docker pull johndoe/private-ml-model:v1.0
```

## AWS Elastic Container Registry (ECR)

### Setup ECR

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create ECR repository
aws ecr create-repository --repository-name ml-model --region us-east-1

# Output:
# {
#     "repository": {
#         "repositoryUri": "123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model"
#     }
# }
```

### Authenticate to ECR

```bash
# Get login password and authenticate
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com
```

### Push to ECR

```bash
# Tag image
docker tag ml-model:v1.0 \
    123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0

# Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0

# Pull
docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0
```

### ECR Lifecycle Policies

**Automatically clean up old images:**

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 production images",
      "selection": {
        "tagStatus": "tagged",
        "tagPrefixList": ["prod"],
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    },
    {
      "rulePriority": 2,
      "description": "Delete untagged images after 7 days",
      "selection": {
        "tagStatus": "untagged",
        "countType": "sinceImagePushed",
        "countUnit": "days",
        "countNumber": 7
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
```

Apply policy:
```bash
aws ecr put-lifecycle-policy \
    --repository-name ml-model \
    --lifecycle-policy-text file://policy.json
```

## Google Container Registry (GCR) / Artifact Registry

### Setup GCR

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth configure-docker

# Or for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Push to GCR

```bash
# GCR format: gcr.io/[PROJECT-ID]/[IMAGE]
docker tag ml-model:v1.0 gcr.io/my-project/ml-model:v1.0

# Push
docker push gcr.io/my-project/ml-model:v1.0

# Pull
docker pull gcr.io/my-project/ml-model:v1.0
```

### Artifact Registry (Recommended)

```bash
# Create repository
gcloud artifacts repositories create ml-models \
    --repository-format=docker \
    --location=us-central1

# Tag and push
docker tag ml-model:v1.0 \
    us-central1-docker.pkg.dev/my-project/ml-models/ml-model:v1.0

docker push us-central1-docker.pkg.dev/my-project/ml-models/ml-model:v1.0
```

## Azure Container Registry (ACR)

### Setup ACR

```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create registry
az acr create --resource-group myResourceGroup \
    --name mymlregistry --sku Basic

# Login to registry
az acr login --name mymlregistry
```

### Push to ACR

```bash
# Get login server
az acr show --name mymlregistry --query loginServer

# Output: mymlregistry.azurecr.io

# Tag and push
docker tag ml-model:v1.0 mymlregistry.azurecr.io/ml-model:v1.0
docker push mymlregistry.azurecr.io/ml-model:v1.0

# Pull
docker pull mymlregistry.azurecr.io/ml-model:v1.0
```

## Image Tagging Strategies

### Semantic Versioning

**Format:** `MAJOR.MINOR.PATCH`

```bash
# Version 1.0.0 - initial release
docker tag ml-model myregistry/ml-model:1.0.0

# Version 1.0.1 - bug fix
docker tag ml-model myregistry/ml-model:1.0.1

# Version 1.1.0 - new feature
docker tag ml-model myregistry/ml-model:1.1.0

# Version 2.0.0 - breaking change
docker tag ml-model myregistry/ml-model:2.0.0
```

### Git-Based Tagging

```bash
# Use git commit SHA
GIT_SHA=$(git rev-parse --short HEAD)
docker tag ml-model myregistry/ml-model:${GIT_SHA}

# Use git branch
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
docker tag ml-model myregistry/ml-model:${GIT_BRANCH}

# Use git tag
GIT_TAG=$(git describe --tags --abbrev=0)
docker tag ml-model myregistry/ml-model:${GIT_TAG}
```

### Environment Tags

```bash
# Development
docker tag ml-model myregistry/ml-model:dev

# Staging
docker tag ml-model myregistry/ml-model:staging

# Production
docker tag ml-model myregistry/ml-model:prod
docker tag ml-model myregistry/ml-model:latest  # Also tag as latest
```

### Combined Strategy

```bash
# Multiple tags for production deployment
VERSION="1.2.0"
GIT_SHA=$(git rev-parse --short HEAD)

docker tag ml-model myregistry/ml-model:${VERSION}
docker tag ml-model myregistry/ml-model:${VERSION}-${GIT_SHA}
docker tag ml-model myregistry/ml-model:latest
docker tag ml-model myregistry/ml-model:prod

# Push all
docker push myregistry/ml-model --all-tags
```

## CI/CD Integration

### GitHub Actions Example

**.github/workflows/build-push.yml:**
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: yourusername/ml-model
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### GitLab CI Example

**.gitlab-ci.yml:**
```yaml
variables:
  DOCKER_DRIVER: overlay2
  IMAGE_NAME: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

stages:
  - build
  - push

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $IMAGE_NAME .
    - docker push $IMAGE_NAME

push-latest:
  stage: push
  image: docker:latest
  only:
    - main
  script:
    - docker pull $IMAGE_NAME
    - docker tag $IMAGE_NAME $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest
```

## Security Scanning

### Docker Scan (Snyk)

```bash
# Scan local image
docker scan ml-model:v1.0

# Scan with specific severity
docker scan --severity=high ml-model:v1.0

# View detailed report
docker scan --json ml-model:v1.0 > scan-report.json
```

### Trivy Scanner

```bash
# Install Trivy
sudo apt-get install trivy

# Scan image
trivy image ml-model:v1.0

# Scan for HIGH and CRITICAL only
trivy image --severity HIGH,CRITICAL ml-model:v1.0

# Output to JSON
trivy image -f json -o results.json ml-model:v1.0
```

### ECR Image Scanning

```bash
# Enable scanning on push
aws ecr put-image-scanning-configuration \
    --repository-name ml-model \
    --image-scanning-configuration scanOnPush=true

# Manual scan
aws ecr start-image-scan \
    --repository-name ml-model \
    --image-id imageTag=v1.0

# Get scan results
aws ecr describe-image-scan-findings \
    --repository-name ml-model \
    --image-id imageTag=v1.0
```

## Private Registry Setup

### Docker Registry (Self-Hosted)

```bash
# Run registry container
docker run -d -p 5000:5000 \
    --name registry \
    -v registry-data:/var/lib/registry \
    registry:2

# Push to local registry
docker tag ml-model localhost:5000/ml-model:v1.0
docker push localhost:5000/ml-model:v1.0

# Pull from local registry
docker pull localhost:5000/ml-model:v1.0
```

### Secure Private Registry with TLS

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  registry:
    image: registry:2
    ports:
      - "5000:5000"
    environment:
      REGISTRY_HTTP_TLS_CERTIFICATE: /certs/domain.crt
      REGISTRY_HTTP_TLS_KEY: /certs/domain.key
      REGISTRY_AUTH: htpasswd
      REGISTRY_AUTH_HTPASSWD_PATH: /auth/htpasswd
      REGISTRY_AUTH_HTPASSWD_REALM: Registry Realm
    volumes:
      - ./certs:/certs
      - ./auth:/auth
      - registry-data:/var/lib/registry

volumes:
  registry-data:
```

## Hands-On Exercise: Multi-Registry Workflow

### Objective

Build and push an ML model image to multiple registries with proper tagging.

### Steps

**1. Build image:**
```bash
docker build -t ml-model:local .
```

**2. Tag for multiple registries:**
```bash
VERSION="1.0.0"
GIT_SHA=$(git rev-parse --short HEAD)

# Docker Hub
docker tag ml-model:local \
    yourusername/ml-model:${VERSION}
docker tag ml-model:local \
    yourusername/ml-model:latest

# AWS ECR
docker tag ml-model:local \
    123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:${VERSION}

# GCP Artifact Registry
docker tag ml-model:local \
    us-central1-docker.pkg.dev/my-project/ml-models/ml-model:${VERSION}
```

**3. Push to registries:**
```bash
# Docker Hub
docker push yourusername/ml-model:${VERSION}
docker push yourusername/ml-model:latest

# ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:${VERSION}

# GCR
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/my-project/ml-models/ml-model:${VERSION}
```

**4. Scan for vulnerabilities:**
```bash
trivy image yourusername/ml-model:${VERSION}
```

## Best Practices

### 1. Never Use `:latest` in Production

❌ **Bad:**
```bash
docker pull ml-model:latest  # Which version?
```

✅ **Good:**
```bash
docker pull ml-model:1.2.3  # Explicit version
```

### 2. Tag Images with Multiple Identifiers

```bash
# Semantic version + git SHA + environment
docker tag ml-model myregistry/ml-model:1.2.3
docker tag ml-model myregistry/ml-model:1.2.3-abc123f
docker tag ml-model myregistry/ml-model:prod
```

### 3. Implement Image Cleanup

```bash
# Delete old images from Docker Hub manually
# Use lifecycle policies in ECR/GCR/ACR

# Clean local images
docker image prune -a --filter "until=720h"  # Older than 30 days
```

### 4. Use Multi-Arch Images

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 \
    -t myregistry/ml-model:1.0.0 --push .
```

## Summary

In this lesson, you learned:

1. **Docker Hub** - Public and private repositories
2. **Cloud Registries** - ECR, GCR, ACR setup and usage
3. **Tagging Strategies** - Semantic versioning, git-based tags
4. **CI/CD Integration** - Automated builds and pushes
5. **Security Scanning** - Vulnerability detection with Trivy
6. **Private Registries** - Self-hosted registry setup

**Key Takeaways:**
- Use registries for fast, consistent deployments
- Tag images with explicit versions
- Scan images for security vulnerabilities
- Implement cleanup policies
- Integrate with CI/CD for automation

## What's Next?

In the next lesson, **07-gpu-docker.md**, you'll learn:
- NVIDIA Container Toolkit installation
- Running GPU workloads in containers
- GPU-enabled base images
- Monitoring GPU usage in containers
- Multi-GPU container deployment

---

## Self-Check Questions

1. What's the difference between Docker Hub and private registries?
2. How do you tag an image for AWS ECR?
3. What's wrong with using `:latest` tag in production?
4. How do you scan an image for vulnerabilities?
5. What's a lifecycle policy and why is it useful?

## Additional Resources

- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [AWS ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Trivy Scanner](https://github.com/aquasecurity/trivy)

---

**Next:** [07-gpu-docker.md](./07-gpu-docker.md)
