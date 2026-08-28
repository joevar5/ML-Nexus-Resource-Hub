# Lab 02: Terraform EKS Cluster

**Duration:** 90 min  **Prerequisites:** Lab 01 VPC up; Terraform 1.6+; AWS CLI

## Objective
Provision an EKS cluster with managed node groups using the official AWS EKS Terraform module. Get kubectl working against it.

## Steps

### 1. Add to your project
```hcl
# eks.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "lab-eks"
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id           # from your VPC module (or output of lab-01)
  subnet_ids = module.vpc.private_subnet_ids
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    general = {
      desired_size = 2
      min_size     = 2
      max_size     = 5
      instance_types = ["m6i.large"]
    }
  }
}

output "cluster_name"     { value = module.eks.cluster_name }
output "cluster_endpoint" { value = module.eks.cluster_endpoint }
```

### 2. Apply
```bash
terraform init -upgrade
terraform plan -out=plan.tfplan
terraform apply plan.tfplan          # ~15-20 minutes
```

### 3. Configure kubectl
```bash
aws eks update-kubeconfig --name lab-eks --region us-west-2
kubectl get nodes
kubectl get pods -A
```

### 4. Deploy a workload
```bash
kubectl create deploy nginx --image=nginx:1.27 --replicas=2
kubectl expose deploy nginx --port=80 --type=LoadBalancer
kubectl get svc nginx -w                # wait for EXTERNAL-IP
curl http://<external-ip>
```

### 5. Add a GPU node group (optional)
```hcl
gpu = {
  desired_size = 0
  min_size     = 0
  max_size     = 3
  instance_types = ["g4dn.xlarge"]
  labels = { "node.kubernetes.io/instance-type" = "g4dn.xlarge" }
  taints = [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]
}
```

### 6. Install NVIDIA device plugin
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/nvidia-device-plugin.yml
```

### 7. Destroy
```bash
kubectl delete svc nginx                     # remove ELB FIRST
terraform destroy -auto-approve
```
Cluster takes another ~10 min to delete.

## Validation
- [ ] `kubectl get nodes` shows 2 Ready nodes.
- [ ] Service gets an external ELB DNS name.
- [ ] `curl` returns the nginx welcome page.

## Cleanup
**Always** `kubectl delete svc <type=LoadBalancer>` BEFORE `terraform destroy`. Otherwise the ELB orphans.

## Troubleshooting
- **`Could not connect to cluster`** — `aws-auth` ConfigMap not updated; module handles this in v20+, older versions need `kubectl_provider`.
- **Nodes not joining** — Check security group rules (eks-cluster-sg → eks-node-sg, ports 1025-65535).
- **Destroy hangs on subnet** — ELB still using it; manually delete LB in console.
