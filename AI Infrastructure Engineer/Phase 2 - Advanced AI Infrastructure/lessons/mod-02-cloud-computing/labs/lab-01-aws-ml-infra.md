# Lab 01: Provision an ML-Ready AWS Environment

**Duration:** 90 min  **Prerequisites:** AWS account, IAM admin on a sandbox account

## Objective
Provision a minimal ML workload on AWS: a VPC with public/private subnets, an S3 bucket for model artifacts, an EC2 instance running your iris-api container, and an Application Load Balancer in front of it. By the end you'll have a public URL serving predictions.

## Steps

### 1. Configure CLI
```bash
aws configure                  # access key, region us-west-2, output json
aws sts get-caller-identity
```

### 2. Create the VPC + subnets (CloudShell or local)
```bash
aws cloudformation deploy --template-file vpc.yaml --stack-name ml-lab-vpc --capabilities CAPABILITY_NAMED_IAM
```
(or use the AWS console: VPC → "Create VPC with subnets" → 2 AZ, 1 public + 1 private each)

### 3. Create S3 bucket
```bash
aws s3 mb s3://ml-lab-$(aws sts get-caller-identity --query Account --output text)-artifacts
aws s3 cp model.joblib s3://ml-lab-$(...)-artifacts/iris/model.joblib
```

### 4. Launch an EC2 instance with the iris-api image
- Instance type: `t3.small`
- AMI: Amazon Linux 2023
- Security group: inbound 22 from your IP, 80 from 0.0.0.0/0
- User data:
  ```bash
  #!/bin/bash
  yum install -y docker
  systemctl enable --now docker
  docker run -d -p 80:8000 --restart=always your-account.dkr.ecr.us-west-2.amazonaws.com/iris-api:0.2
  ```

### 5. Create an ALB and target group
- ALB in public subnets, listener on 80
- Target group: instance protocol HTTP port 80, health check `/health`
- Register the EC2 instance

### 6. Test
```bash
curl http://<alb-dns-name>/health
curl -X POST http://<alb-dns-name>/predict -H 'content-type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Validation
- [ ] `/health` returns 200 from the ALB DNS name.
- [ ] `/predict` returns a JSON prediction.
- [ ] S3 bucket contains `iris/model.joblib`.
- [ ] EC2 security group does NOT allow 0.0.0.0/0 on port 22.

## Cleanup
```bash
aws elbv2 delete-load-balancer --load-balancer-arn <arn>
aws ec2 terminate-instances --instance-ids <id>
aws s3 rb s3://ml-lab-$(...)-artifacts --force
aws cloudformation delete-stack --stack-name ml-lab-vpc
```

## Troubleshooting
- **EC2 user-data didn't run** — check `/var/log/cloud-init-output.log` via SSM Session Manager.
- **ALB health check fails** — ensure the security group on EC2 allows the ALB's security group, not just your IP.
- **ECR pull denied** — attach `AmazonEC2ContainerRegistryReadOnly` to the instance role.
