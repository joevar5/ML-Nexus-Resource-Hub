# Lab 07: Policy as Code with OPA Conftest

**Duration:** 60 min  **Prerequisites:** Conftest installed (`brew install conftest`)

## Objective
Write Rego policies that catch dangerous Terraform plans (e.g., public S3 buckets, missing tags, oversized instances) before they reach apply. Wire into CI.

## Steps

### 1. Generate a plan
```bash
terraform plan -out=plan.tfplan
terraform show -json plan.tfplan > plan.json
```

### 2. Conftest policy directory
```bash
mkdir -p policies
```

### 3. Policy 1: no public S3 buckets
```rego
# policies/s3.rego
package main

deny[msg] {
  resource := input.planned_values.root_module.resources[_]
  resource.type == "aws_s3_bucket"
  resource.values.acl == "public-read"
  msg := sprintf("Bucket %v has acl=public-read", [resource.address])
}
```

### 4. Policy 2: required tags
```rego
# policies/tags.rego
package main

required_tags := ["Environment", "Owner", "CostCenter"]

deny[msg] {
  resource := input.planned_values.root_module.resources[_]
  resource.type == "aws_instance"
  missing := [t | t := required_tags[_]; not resource.values.tags[t]]
  count(missing) > 0
  msg := sprintf("Instance %v missing tags: %v", [resource.address, missing])
}
```

### 5. Policy 3: no oversized instance types
```rego
# policies/instance_size.rego
package main

forbidden := {"x1.32xlarge", "x1e.32xlarge", "u-24tb1.metal"}

deny[msg] {
  resource := input.planned_values.root_module.resources[_]
  resource.type == "aws_instance"
  forbidden[resource.values.instance_type]
  msg := sprintf("Instance %v uses forbidden type %v", [resource.address, resource.values.instance_type])
}
```

### 6. Run conftest against the plan
```bash
conftest test --policy policies plan.json
```
A passing plan exits 0; a failing plan exits non-zero with the deny messages.

### 7. Test the policies (TDD for policies)
```rego
# policies/s3_test.rego
package main

test_public_bucket_denied {
  deny["Bucket aws_s3_bucket.x has acl=public-read"] with input as {
    "planned_values": {"root_module": {"resources": [{
      "address": "aws_s3_bucket.x", "type": "aws_s3_bucket",
      "values": {"acl": "public-read"}}]}}
  }
}
```
```bash
conftest verify --policy policies
```

### 8. CI integration
Append to your terraform workflow (lab 05):
```yaml
      - run: terraform show -json plan.tfplan > plan.json
      - uses: instrumenta/conftest-action@master
        with: { files: plan.json, policy: policies }
```

## Validation
- [ ] `conftest test` blocks a Terraform plan that has a `public-read` bucket.
- [ ] `conftest test` blocks a plan that has an instance missing required tags.
- [ ] `conftest verify` runs unit tests against the policies themselves.

## Cleanup
```bash
rm -rf policies plan.tfplan plan.json
```

## Troubleshooting
- **All resources pass** — Plan JSON structure changes between TF versions; inspect with `jq '.planned_values.root_module.resources[0]' plan.json`.
- **`undefined function deny`** — Rego module name must match `package main` and rules must be `deny[msg] { ... }`.
- **Policy too strict for legacy** — Use `warn` instead of `deny` for transitional rules.
