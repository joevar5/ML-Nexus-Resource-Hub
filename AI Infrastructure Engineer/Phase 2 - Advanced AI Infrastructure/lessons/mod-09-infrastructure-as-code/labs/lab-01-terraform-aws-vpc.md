# Lab 01: Terraform AWS VPC

**Duration:** 60 min  **Prerequisites:** AWS account, Terraform 1.6+

## Objective
Author a Terraform module that provisions a multi-AZ VPC with public + private subnets, route tables, NAT gateway, and Internet Gateway. Apply, modify, and destroy.

## Steps

### 1. Project layout
```
vpc-lab/
├── main.tf
├── variables.tf
├── outputs.tf
└── versions.tf
```

### 2. versions.tf
```hcl
terraform {
  required_version = ">= 1.6"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}
provider "aws" { region = var.region }
```

### 3. variables.tf
```hcl
variable "region"     { type = string  default = "us-west-2" }
variable "name"       { type = string  default = "lab-vpc"  }
variable "cidr_block" { type = string  default = "10.10.0.0/16" }
variable "azs"        { type = list(string)  default = ["us-west-2a","us-west-2b"] }
```

### 4. main.tf (compressed)
```hcl
resource "aws_vpc" "this" {
  cidr_block = var.cidr_block
  enable_dns_support = true; enable_dns_hostnames = true
  tags = { Name = var.name }
}

resource "aws_subnet" "public" {
  count = length(var.azs)
  vpc_id = aws_vpc.this.id
  cidr_block = cidrsubnet(var.cidr_block, 4, count.index)
  availability_zone = var.azs[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.name}-public-${var.azs[count.index]}" }
}

resource "aws_subnet" "private" {
  count = length(var.azs)
  vpc_id = aws_vpc.this.id
  cidr_block = cidrsubnet(var.cidr_block, 4, count.index + 8)
  availability_zone = var.azs[count.index]
  tags = { Name = "${var.name}-private-${var.azs[count.index]}" }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags = { Name = var.name }
}

resource "aws_eip" "nat" { domain = "vpc" }

resource "aws_nat_gateway" "this" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
}
resource "aws_route_table_association" "public" {
  count = length(var.azs)
  subnet_id = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this.id
  }
}
resource "aws_route_table_association" "private" {
  count = length(var.azs)
  subnet_id = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
```

### 5. outputs.tf
```hcl
output "vpc_id"            { value = aws_vpc.this.id }
output "public_subnet_ids" { value = aws_subnet.public[*].id }
output "private_subnet_ids"{ value = aws_subnet.private[*].id }
```

### 6. Apply
```bash
terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan
terraform output
```

### 7. Modify and re-apply
Change `azs` to include a third AZ. Run `terraform plan` — should show only additions, no destructions.

### 8. Destroy
```bash
terraform destroy -auto-approve
```

## Validation
- [ ] `terraform plan` shows expected resource count for 2 AZs.
- [ ] Apply succeeds without errors.
- [ ] Outputs include 2 public + 2 private subnet IDs.
- [ ] Resources visible in AWS console.

## Cleanup
Always end with `terraform destroy`. Forgetting NAT gateways costs ~$1/day.

## Troubleshooting
- **`Error: ... InvalidVpcID.NotFound`** — Race condition; add `depends_on` to enforce order.
- **NAT taking 5+ min** — Normal; NAT provisioning is slow.
- **`UnauthorizedOperation`** — IAM user lacks EC2/VPC permissions.
