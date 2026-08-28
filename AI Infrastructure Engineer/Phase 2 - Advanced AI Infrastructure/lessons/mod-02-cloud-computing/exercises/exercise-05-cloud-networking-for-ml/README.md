# Exercise 05: Cloud Networking for ML Workloads

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 01 (AWS VPC); IaC familiarity from labs 01-04 of mod-09

## Objective

Design and provision an ML-aware VPC with three subnet tiers (public, private app, private data) across 3 AZs, an isolated GPU node pool, restricted egress through a NAT gateway with per-namespace bandwidth limits, and VPC Endpoints for S3/ECR/CloudWatch to keep data on the private backbone. Validate connectivity with a small test matrix.

## Why this matters

ML workloads are network-asymmetric: training reads huge datasets but does little outbound; inference is request/response with strict latency SLOs; experiment tracking and metrics emit constant low-volume traffic. A flat VPC with NAT for everything works until it doesn't — usually around the time GPU instances start saturating NAT or you discover S3 reads are routed through the public internet at $0.09/GB.

## Requirements

Terraform module producing:

1. **VPC** `10.20.0.0/16` across 3 AZs.
2. **Subnet tiers** per AZ:
   - Public `10.20.{0,1,2}.0/24` — ALBs + NAT Gateways
   - Private app `10.20.{10,11,12}.0/24` — inference pods, control plane
   - Private data `10.20.{20,21,22}.0/24` — training nodes, GPU nodes
3. **NAT Gateway** per AZ (high-availability) with egress charges tagged for cost allocation.
4. **VPC Endpoints** (Gateway type for S3 and DynamoDB; Interface type for ECR, ECR Docker, CloudWatch Logs, CloudWatch Monitoring, SSM, Secrets Manager).
5. **Security Groups** that enforce least-privilege:
   - `sg-inference` allows ingress from ALB SG on 8000; egress to S3 VPC Endpoint only.
   - `sg-training` allows egress to S3 + Hugging Face IP ranges (via prefix list); no inbound.
6. **NetworkACL** on private-data subnets denying outbound to 0.0.0.0/0 except via NAT.
7. **Bandwidth caps** on training-job egress via tc qdisc (in a per-pod sidecar).
8. **Flow Logs** to CloudWatch for the entire VPC.

## Step-by-step

### Step 1 — Plan the address space (15 min)
Document subnets in a `NETWORK.md`. Visualize the topology with a diagram (ASCII or Excalidraw).

### Step 2 — Terraform module skeleton (30 min)
```
modules/ml-vpc/
├── main.tf         # VPC + subnets + IGW + NAT + RTs
├── endpoints.tf    # VPC Endpoints
├── security_groups.tf
├── variables.tf
└── outputs.tf
```

### Step 3 — Subnet tiers + routing (45 min)
Use `aws_subnet`, `aws_route_table`, `aws_route_table_association`. Each tier gets its own route table:
- Public RT → `0.0.0.0/0` via IGW
- Private app RT → `0.0.0.0/0` via NAT (own AZ)
- Private data RT → `0.0.0.0/0` via NAT (own AZ), and route to S3 Gateway Endpoint

### Step 4 — VPC Endpoints (30 min)
S3 + DynamoDB are Gateway endpoints (free). The rest are Interface endpoints (cost ~$7/month each).
Attach Interface endpoints only to private subnets and use AZ-local DNS so traffic stays local.

### Step 5 — Security groups (30 min)
Write inline policies. Use `cidr_blocks` references to the VPC's own CIDR for east-west; never `0.0.0.0/0` for ingress.

### Step 6 — Outputs (15 min)
Expose subnet IDs by tier, all security group IDs, and the NAT Gateway IDs. Downstream modules consume these.

### Step 7 — Test connectivity (30 min)
Spin up tiny EC2 instances in each subnet tier. Verify:
- Public can reach the internet.
- Private app can reach the internet via NAT but only on egress.
- Private data can reach S3 via Gateway Endpoint (no NAT charge).
- Inference instance can reach inference-tier S3 prefix but not training-tier.

Script as `scripts/test-connectivity.sh`.

### Step 8 — Cost projection (15 min)
Write `COSTS.md` projecting monthly charges for: NAT, Interface Endpoints, Flow Logs, data egress on synthetic load. Compare against the alternative (no VPC Endpoints).

## Deliverables

1. Terraform module + a `terraform/environments/prod` root applying it.
2. Connectivity test script.
3. `NETWORK.md` with topology diagram.
4. `COSTS.md` with projection + sensitivity table.

## Validation

- [ ] `terraform apply` succeeds; all VPC resources visible in AWS console.
- [ ] EC2 in private data subnet can `aws s3 cp` from S3 with no public IP and no NAT route used (verify via VPC Flow Logs).
- [ ] EC2 in private app subnet can `pip install` (via NAT).
- [ ] Inference SG cannot reach training SG (verify via failed connection).
- [ ] Cost projection within ~10% of AWS Pricing Calculator for the same shape.

## Stretch goals

- Add **Transit Gateway** wiring to a second VPC (simulate prod ↔ staging).
- Configure **AWS Network Firewall** for outbound URL filtering (deny known crypto-mining domains, etc.).
- Implement **Hub-Spoke** topology with shared services VPC.
- Add **VPC peering** to a Cloud HSM VPC for key operations.

## Common pitfalls

- **One NAT for the whole VPC** — Saves money, kills HA. Pay for per-AZ NAT for prod; one shared NAT only for dev.
- **Forgetting S3 Gateway Endpoint** — Then S3 traffic goes via NAT and you pay egress per byte.
- **Interface Endpoints in all AZs by default** — Charges per-AZ per-endpoint. Constrain to the AZs that actually use them.
- **`0.0.0.0/0` in private-data SG egress** — Even if the route table says NAT, you've left a foot-gun for a future contributor.
- **Flow Logs to S3 vs CloudWatch** — CloudWatch is more expensive but query-friendly. Pick deliberately.

## Solutions

Reference Terraform in the engineer-solutions repo.
