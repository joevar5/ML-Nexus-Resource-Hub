# Exercise 08: Policy as Code (OPA + Sentinel)

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 07 (OPA basics); exercise 03 (modules)

## Objective

Implement policy-as-code for both Terraform (using Conftest/Open Policy Agent) and Kubernetes (using Kyverno or Gatekeeper). Cover 4 policy categories: security, cost, naming, compliance.

## Why this matters

Without policies, every team learns the same security mistakes individually. With policies, the mistakes are caught at PR-time before merge. Policy as code scales infosec from "ask Bob" to automated enforcement.

## Requirements

For each category, at least 2 policies in each layer (Terraform + Kubernetes).

### Security
- TF: no public S3 buckets, no `0.0.0.0/0` security group ingress on port 22.
- K8s: no privileged containers, all images signed.

### Cost
- TF: no instance types > `m6i.4xlarge` without justification tag.
- K8s: no `ResourceQuota`-exceeding deployments.

### Naming
- TF: resources must have `team`, `cost_center`, `environment` tags.
- K8s: all Deployments must have `app`, `version`, `team` labels.

### Compliance
- TF: RDS must have encryption + backups.
- K8s: PersistentVolumes must have backup annotation.

## Step-by-step

### Step 1 — Conftest for Terraform (90 min)
Per lab 07. Write Rego policies covering all TF items above. Test with `conftest verify`. Wire into CI per exercise 05 of lab series.

### Step 2 — Kyverno install (15 min)
```bash
helm install kyverno kyverno/kyverno -n kyverno --create-namespace
```

### Step 3 — Kyverno policy 1: no privileged containers (15 min)
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: disallow-privileged }
spec:
  validationFailureAction: Enforce
  rules:
    - name: privileged-not-allowed
      match: { any: [{ resources: { kinds: [Pod] } }] }
      validate:
        message: "Privileged containers are not allowed."
        pattern:
          spec:
            =(securityContext): { =(privileged): "false" }
            containers:
              - =(securityContext): { =(privileged): "false" }
```
Test: create a privileged pod → blocked.

### Step 4 — Kyverno policy 2: require signed images (30 min)
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: verify-signatures }
spec:
  validationFailureAction: Enforce
  rules:
    - name: require-signed
      match: { any: [{ resources: { kinds: [Pod] } }] }
      verifyImages:
        - imageReferences: ["ghcr.io/me/*"]
          attestors:
            - entries:
                - keyless:
                    subject: "me@example.com"
                    issuer: "https://github.com/login/oauth"
```

### Step 5 — Kyverno policy 3: require labels (15 min)
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: require-labels }
spec:
  validationFailureAction: Enforce
  rules:
    - name: required-labels
      match: { any: [{ resources: { kinds: [Deployment, StatefulSet] } }] }
      validate:
        message: "Required labels: app, version, team."
        pattern:
          metadata:
            labels:
              app: "?*"
              version: "?*"
              team: "?*"
```

### Step 6 — Mutating policy 4: auto-inject labels (30 min)
Reduce friction: if a Deployment is in namespace `team-a`, auto-inject `team: team-a`.
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: auto-inject-team }
spec:
  rules:
    - name: from-namespace
      match: { any: [{ resources: { kinds: [Deployment] } }] }
      mutate:
        patchStrategicMerge:
          metadata:
            labels:
              team: "{{request.object.metadata.namespace}}"
```

### Step 7 — Test in CI before applying (30 min)
Generate manifests locally; run `kyverno apply --policies ./policies --resource ./manifests`. Block PR if policy fails.

## Deliverables

1. 4+ Conftest Rego policies.
2. 4+ Kyverno policies (mix of validation + mutation).
3. CI integration for both.
4. `POLICY_CATALOG.md` documenting each policy with rationale.
5. Demo: deliberate violation blocked by each policy.

## Validation

- [ ] Conftest blocks an Terraform plan violating each policy.
- [ ] Kyverno blocks Pod/Deployment violating each validation policy.
- [ ] Mutating policy auto-corrects benign omissions.
- [ ] CI fails PRs with policy violations.

## Stretch goals

- Migrate Kyverno policies to **OPA Gatekeeper** for comparison; pick a winner for your team.
- Add **Falco** for runtime policy enforcement (detect at runtime, not just at admission).
- Add **policy exemptions**: a CRD allowing time-bound exceptions with required justification.

## Common pitfalls

- **`enforcement: Audit` left in prod** — Logs violations but doesn't block. Switch to Enforce after a quiet week.
- **Image signature verification offline** — Requires cluster access to transparency log. Test in air-gapped scenarios.
- **Mutating policy clashes with admission webhooks** — Order matters; mutate before validate.
- **Policy that breaks legitimate workflows** — Always have a stakeholder pre-review.
