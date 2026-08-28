# Exercise 13: IaC for AI/ML Workloads

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** All prior mod-09 exercises

## Objective

Build a complete IaC project for an ML workload that combines everything: VPC, EKS with GPU nodes, S3 buckets for artifacts, IAM via IRSA, Vault for secrets, ArgoCD bootstrap. End state: `terraform apply` + `argocd app sync` produces a usable ML serving cluster.

## Requirements

1. Terraform creates: VPC, EKS (CPU + GPU node pools), S3 buckets, IAM roles, IRSA bindings, ECR repositories.
2. ArgoCD bootstrap installs: ingress-nginx, cert-manager, external-secrets, NVIDIA Device Plugin, kube-prometheus-stack.
3. App manifests deploy: iris-api, vLLM service, monitoring dashboards.
4. Documented `ramp_up.md` from zero to running model API.

## Step-by-step

### Step 1 — Plan the topology (15 min)

```
ml-platform/
├── terraform/
│   ├── modules/        (vpc, eks, s3, irsa from earlier exercises)
│   └── envs/{dev,staging,prod}/
└── manifests/
    ├── bootstrap/      (Apps for ingress, cert-manager, ESO, device plugin, monitoring)
    ├── services/
    │   ├── iris-api/
    │   └── vllm/
    └── monitoring/
```

### Step 2 — Terraform composition (60 min)
```hcl
# envs/dev/main.tf
module "vpc" {
  source = "../../modules/vpc"
  name   = "ml-dev"
  cidr_block = "10.20.0.0/16"
  azs        = ["us-west-2a", "us-west-2b"]
}

module "eks" {
  source = "../../modules/eks"
  cluster_name = "ml-dev"
  kubernetes_version = "1.30"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  
  node_groups = {
    general = { instance_types = ["m6i.large"], desired_size = 2, min_size = 2, max_size = 6 }
    gpu = {
      instance_types = ["g5.xlarge"]
      desired_size = 0     # scaled by Karpenter
      min_size = 0
      max_size = 4
      labels = { "nvidia.com/gpu.present" = "true" }
      taints = [{ key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" }]
    }
  }
}

module "s3_artifacts" {
  source = "../../modules/s3"
  bucket_name = "ml-dev-artifacts"
  versioning  = true
  encryption  = true
}

module "iris_irsa" {
  source = "../../modules/irsa"
  role_name = "iris-api-${terraform.workspace}"
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_issuer_url   = module.eks.oidc_issuer_url
  namespace            = "iris"
  service_account_name = "iris-api"
  managed_policy_arns = []
  inline_policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject"]
      Resource = "${module.s3_artifacts.arn}/*"
    }]
  })
}
```

### Step 3 — ArgoCD bootstrap (45 min)
After `terraform apply`:
```bash
aws eks update-kubeconfig --name ml-dev
kubectl create ns argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Apply root app — points at manifests/bootstrap which contains Applications for everything else
kubectl apply -f manifests/argocd/root-bootstrap.yaml
```

`manifests/bootstrap/` contains Application manifests for:
- ingress-nginx
- cert-manager
- external-secrets-operator
- nvidia-device-plugin
- kube-prometheus-stack
- karpenter

ArgoCD installs all of them.

### Step 4 — Service manifests (30 min)
`manifests/services/iris-api/`:
- Deployment with `nodeSelector: kubernetes.io/arch=amd64` (CPU)
- Service
- Ingress (cert-manager auto-issues TLS)
- ExternalSecret pulling DB creds from Vault

`manifests/services/vllm/`:
- Deployment with `nodeSelector: nvidia.com/gpu.present=true`, toleration for the taint
- Service
- ServiceMonitor for Prometheus

### Step 5 — Monitoring (15 min)
`manifests/monitoring/grafana-dashboards/` — ConfigMaps loaded into Grafana via auto-discovery sidecar.

### Step 6 — End-to-end ramp (15 min)
`ramp_up.md`:
1. `cd terraform/envs/dev && terraform init && terraform apply`
2. `aws eks update-kubeconfig --name ml-dev`
3. `kubectl apply -f manifests/argocd/root-bootstrap.yaml`
4. Wait for all Applications green: `argocd app list` (~10 min)
5. `curl https://iris-dev.example.com/health` → 200

Total: ~30 minutes from zero to live.

## Deliverables

1. Working Terraform + manifests repo.
2. Successful ramp-up in dev.
3. `ramp_up.md` step-by-step.
4. `ARCHITECTURE.md` diagram and explanation.

## Validation

- [ ] `terraform apply` succeeds.
- [ ] All ArgoCD applications reach Synced + Healthy.
- [ ] iris-api endpoint returns predictions.
- [ ] vLLM endpoint serves an LLM completion.
- [ ] Grafana shows metrics from both.

## Stretch goals

- Add **multi-env promotion** per exercise 10.
- Add **policy as code** per exercise 08 gating both Terraform plans and Kubernetes manifests.
- Add **cost dashboard** per exercise 11.

## Common pitfalls

- **GPU nodes scheduled too late** — Karpenter has to provision; first GPU pod takes ~5 min.
- **TLS cert pending** — DNS A record for the domain must exist before cert-manager can validate.
- **IRSA OIDC trust missing** — One-time setup per cluster; module above handles it but verify.
- **Lots of moving parts** — Test bootstrap on a fresh kind cluster first; iterate before doing it on cloud.
