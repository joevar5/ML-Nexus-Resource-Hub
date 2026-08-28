# Lab 04: Multi-Cloud Deployment with a Single Tool

**Duration:** 90 min  **Prerequisites:** Any 2 of AWS/GCP/Azure accounts; Terraform 1.6+

## Objective
Use Terraform to define a portable model-serving deployment, then apply the same root module against two clouds (your choice). By the end you'll have iris-api running on two clouds, accessible via two different DNS names but the same API contract.

## Steps

### 1. Project layout
```
ml-multicloud/
├── modules/
│   ├── aws/ main.tf variables.tf outputs.tf
│   ├── gcp/ main.tf variables.tf outputs.tf
│   └── azure/ main.tf variables.tf outputs.tf
└── envs/
    ├── aws/  main.tf backend.tf
    └── gcp/  main.tf backend.tf
```

Each cloud module provisions: a network, a managed container runtime (ECS/Cloud Run/Container Apps), and a public load balancer.

### 2. Define the AWS module (skeleton)
```hcl
# modules/aws/main.tf
resource "aws_ecs_cluster" "this" { name = var.name }
resource "aws_ecs_service" "iris" {
  name            = "iris-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.iris.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  ...
}
output "endpoint" { value = aws_lb.this.dns_name }
```

### 3. Define the GCP module (skeleton)
```hcl
# modules/gcp/main.tf
resource "google_cloud_run_v2_service" "iris" {
  name     = "iris-api"
  location = var.region
  template {
    containers {
      image = var.image
      ports { container_port = 8000 }
    }
  }
}
output "endpoint" { value = google_cloud_run_v2_service.iris.uri }
```

### 4. Define the root modules
```hcl
# envs/aws/main.tf
module "iris" {
  source = "../../modules/aws"
  name   = "iris-multicloud"
  image  = var.image
  region = "us-west-2"
}
output "endpoint" { value = module.iris.endpoint }
```

Similar for `envs/gcp/main.tf`.

### 5. Apply
```bash
cd envs/aws && terraform init && terraform apply -var image=$AWS_ECR_IMAGE
cd ../gcp  && terraform init && terraform apply -var image=$GCP_ARTIFACT_IMAGE
```

### 6. Verify both endpoints
```bash
AWS_URL=$(cd envs/aws && terraform output -raw endpoint)
GCP_URL=$(cd envs/gcp && terraform output -raw endpoint)

for url in "$AWS_URL" "$GCP_URL"; do
  echo "== $url =="
  curl -s "$url/health"
  curl -s -X POST "$url/predict" -H 'content-type: application/json' \
    -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
done
```

## Validation
- [ ] Both endpoints return 200 on `/health`.
- [ ] Both endpoints return the same prediction for the same input.
- [ ] `terraform destroy` in each env cleanly tears down all resources (no orphans in the provider console).

## Cleanup
```bash
cd envs/aws && terraform destroy -auto-approve
cd ../gcp  && terraform destroy -auto-approve
```

## Troubleshooting
- **Same Terraform state file used for both clouds** — Backend must be per-env (separate S3 prefix or GCS bucket per env).
- **Prediction outputs differ** — Same model artifact must be deployed to both. Sanity-check `model.joblib` SHA in both build pipelines.
- **GCP Cloud Run requires public images** — Configure Cloud Run to use a private Artifact Registry image via the service account.
