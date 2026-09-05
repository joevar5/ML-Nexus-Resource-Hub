# Lesson 05: Container Registries

A container registry is to images what GitHub is to code — a place to push a build once and pull it from anywhere, without rebuilding on every machine. This lesson covers Docker Hub and the three major cloud registries, tagging strategy, CI/CD integration, and vulnerability scanning.

**Prerequisites:** ability to build and tag Docker images (Lessons 01-02); an account with at least one cloud provider is useful but not required to follow along.

### Contents

1. [Why Registries Matter](#why-registries-matter)
2. [Docker Hub](#docker-hub)
3. [Cloud Registries: ECR, GCR, ACR](#cloud-registries-ecr-gcr-acr)
4. [Image Tagging Strategy](#image-tagging-strategy)
5. [CI/CD Integration](#cicd-integration)
6. [Security Scanning](#security-scanning)
7. [Self-Hosted Private Registry](#self-hosted-private-registry)
8. [Practical Exercise](#practical-exercise)
9. [Key Takeaways](#key-takeaways)
10. [Additional Resources](#additional-resources)

---

## Why Registries Matter

Without one, every deployment target has to `git clone` and rebuild — for a real ML image with CUDA and framework dependencies that can easily be 15-20 minutes. With a registry, deployment is a `docker pull` of an already-built layer set, usually under a couple of minutes.

| Registry | Type | Best For |
|---|---|---|
| Docker Hub | Public/free | Open-source images, small teams, official base images |
| AWS ECR | Cloud, per-provider | Teams already on AWS/EKS/ECS |
| GCP Artifact Registry | Cloud, per-provider | Teams on GCP/GKE (successor to GCR) |
| Azure ACR | Cloud, per-provider | Teams on Azure/AKS |
| Self-hosted (`registry:2`) | Private, on-prem | Air-gapped environments, full control |

Image naming follows `[registry]/[namespace]/[repository]:[tag]`; the registry segment defaults to `docker.io` when omitted.

---

## Docker Hub

```bash
docker login                              # username + access token (not your password)
docker tag ml-model:v1.0 johndoe/ml-model:v1.0
docker push johndoe/ml-model:v1.0
docker pull johndoe/ml-model:v1.0
```

Free accounts get unlimited public repositories and one private repository. Create an access token under Settings → Security rather than logging in with your account password — tokens can be scoped and revoked independently.

---

## Cloud Registries: ECR, GCR, ACR

All three follow the same shape: authenticate, create a repository, tag with the registry's full path, push.

### AWS ECR

```bash
aws ecr create-repository --repository-name ml-model --region us-east-1
# repositoryUri: 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model

aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag ml-model:v1.0 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:v1.0
```

ECR lifecycle policies clean up old images automatically — useful since untagged layers from repeated CI builds accumulate fast:

```json
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 production images",
      "selection": { "tagStatus": "tagged", "tagPrefixList": ["prod"], "countType": "imageCountMoreThan", "countNumber": 10 },
      "action": { "type": "expire" }
    },
    {
      "rulePriority": 2,
      "description": "Delete untagged images after 7 days",
      "selection": { "tagStatus": "untagged", "countType": "sinceImagePushed", "countUnit": "days", "countNumber": 7 },
      "action": { "type": "expire" }
    }
  ]
}
```

```bash
aws ecr put-lifecycle-policy --repository-name ml-model --lifecycle-policy-text file://policy.json
```

### GCP Artifact Registry (successor to GCR)

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

gcloud artifacts repositories create ml-models \
    --repository-format=docker --location=us-central1

docker tag ml-model:v1.0 us-central1-docker.pkg.dev/my-project/ml-models/ml-model:v1.0
docker push us-central1-docker.pkg.dev/my-project/ml-models/ml-model:v1.0
```

Google recommends Artifact Registry over the older `gcr.io` registry for new projects — it supports fine-grained IAM and non-container artifacts too.

### Azure ACR

```bash
az acr create --resource-group myResourceGroup --name mymlregistry --sku Basic
az acr login --name mymlregistry
# login server: mymlregistry.azurecr.io

docker tag ml-model:v1.0 mymlregistry.azurecr.io/ml-model:v1.0
docker push mymlregistry.azurecr.io/ml-model:v1.0
```

---

## Image Tagging Strategy

`:latest` is a moving target, not a version — pulling it in production means you don't know what you're actually running. Use one of these instead, or combine them:

| Scheme | Example | When |
|---|---|---|
| Semantic version | `1.2.0` | Public releases, clear compatibility contract |
| Git SHA | `abc123f` | Every CI build, exact traceability to a commit |
| Environment | `staging`, `prod` | Deployment target, always repointed to a specific build |
| Combined | `1.2.0-abc123f` | Production images — human-readable and traceable |

```bash
VERSION="1.2.0"
GIT_SHA=$(git rev-parse --short HEAD)

docker tag ml-model myregistry/ml-model:${VERSION}
docker tag ml-model myregistry/ml-model:${VERSION}-${GIT_SHA}
docker tag ml-model myregistry/ml-model:prod

docker push myregistry/ml-model --all-tags
```

Deploy by pinning the exact tag (`ml-model:1.2.3`), never `ml-model:latest`.

---

## CI/CD Integration

GitHub Actions with `docker/build-push-action` and `docker/metadata-action` to derive tags automatically from the git ref:

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
      - uses: docker/setup-buildx-action@v2
      - uses: docker/login-action@v2
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
            type=semver,pattern={{version}}
            type=sha
      - uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

GitLab CI equivalent, using the commit SHA as the tag and promoting to `latest` only on `main`:

```yaml
variables:
  IMAGE_NAME: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

build:
  stage: build
  image: docker:latest
  services: [docker:dind]
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $IMAGE_NAME .
    - docker push $IMAGE_NAME

push-latest:
  stage: push
  image: docker:latest
  only: [main]
  script:
    - docker pull $IMAGE_NAME
    - docker tag $IMAGE_NAME $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest
```

---

## Security Scanning

**Trivy** (open-source, works against any registry):

```bash
trivy image ml-model:v1.0
trivy image --severity HIGH,CRITICAL --exit-code 1 ml-model:v1.0   # fail CI on findings
```

**ECR native scanning:**

```bash
aws ecr put-image-scanning-configuration --repository-name ml-model \
    --image-scanning-configuration scanOnPush=true

aws ecr describe-image-scan-findings --repository-name ml-model --image-id imageTag=v1.0
```

GCR/Artifact Registry and ACR both offer equivalent built-in scan-on-push options in their consoles. Wire the exit-code failure into CI so a HIGH/CRITICAL finding blocks the push rather than getting noticed after deploy.

---

## Self-Hosted Private Registry

For air-gapped environments or full control, run the official `registry:2` image:

```bash
docker run -d -p 5000:5000 --name registry -v registry-data:/var/lib/registry registry:2

docker tag ml-model localhost:5000/ml-model:v1.0
docker push localhost:5000/ml-model:v1.0
```

For anything beyond a local scratch registry, add TLS and basic auth:

```yaml
services:
  registry:
    image: registry:2
    ports: ["5000:5000"]
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

---

## Practical Exercise

Push one image to Docker Hub, ECR, and GCP Artifact Registry with a consistent multi-tag scheme, then scan it.

**Requirements:** tag with semantic version + git SHA + `latest` for Docker Hub; tag with just the version for ECR and Artifact Registry; run Trivy against the final image and record any HIGH/CRITICAL findings.

<details>
<summary><strong>Sample Solution</strong></summary>

```bash
docker build -t ml-model:local .

VERSION="1.0.0"
GIT_SHA=$(git rev-parse --short HEAD)

# Docker Hub
docker tag ml-model:local yourusername/ml-model:${VERSION}
docker tag ml-model:local yourusername/ml-model:${VERSION}-${GIT_SHA}
docker tag ml-model:local yourusername/ml-model:latest
docker push yourusername/ml-model:${VERSION}
docker push yourusername/ml-model:${VERSION}-${GIT_SHA}
docker push yourusername/ml-model:latest

# AWS ECR
docker tag ml-model:local 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:${VERSION}
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-model:${VERSION}

# GCP Artifact Registry
docker tag ml-model:local us-central1-docker.pkg.dev/my-project/ml-models/ml-model:${VERSION}
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/my-project/ml-models/ml-model:${VERSION}

# Scan
trivy image --severity HIGH,CRITICAL yourusername/ml-model:${VERSION}
```

The Hub image carries three tags for three purposes: `latest` for quick local pulls during development, the plain version for anyone reading a deploy manifest, and the version+SHA combo for tracing a running container back to the exact commit that built it.

</details>

---

## Key Takeaways

1. A registry turns a 15-20 minute rebuild into a sub-minute `docker pull`.
2. Docker Hub, ECR, Artifact Registry, and ACR share the same tag-and-push workflow — only the auth step and the registry path differ.
3. Never deploy `:latest` — pin a semantic version, a git SHA, or both.
4. Scan on every push (Trivy or the registry's native scanner) and fail CI on HIGH/CRITICAL findings.
5. Set lifecycle/cleanup policies — untagged CI images accumulate storage cost fast without one.

---

## Additional Resources

- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [AWS ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Trivy Scanner](https://github.com/aquasecurity/trivy)

---

**Next Lesson:** [06-gpu-docker.md](./06-gpu-docker.md) — running GPU-accelerated ML workloads in containers
