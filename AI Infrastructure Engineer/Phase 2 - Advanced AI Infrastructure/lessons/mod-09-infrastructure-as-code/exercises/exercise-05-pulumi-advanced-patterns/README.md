# Exercise 05: Pulumi Advanced Patterns

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 04 (Pulumi intro)

## Objective

Apply Pulumi's programming-language advantages: dynamic resource generation from data, ComponentResources for reusable abstractions, cross-stack references, and policy-as-code with CrossGuard.

## Why this matters

Pulumi's value over Terraform is real Python/TypeScript: loops, conditionals, classes, packages. Engineers who use it idiomatically build infrastructure code 30-50% smaller than equivalent Terraform.

## Requirements

1. **Dynamic generation**: read a YAML config of N tenants, create N namespaces + quotas + ResourceQuotas without manual repetition.
2. **ComponentResource**: package "an ML serving stack" (Deployment + Service + Ingress + HPA + ServiceMonitor) as a single reusable type.
3. **Cross-stack reference**: networking stack outputs VPC ID; app stack consumes it.
4. **Policy as code**: `aws:rds-must-encrypt`, `k8s:no-privileged-containers` enforced at preview.
5. **Stack outputs as the "API"** between layers.

## Step-by-step

### Step 1 — Project setup (15 min)
```bash
mkdir pulumi-advanced && cd pulumi-advanced
pulumi new aws-python -y --name pulumi-advanced
```

### Step 2 — Dynamic tenants (30 min)
```python
import yaml, pulumi
from pulumi_kubernetes.core.v1 import Namespace, ResourceQuota

tenants = yaml.safe_load(open("tenants.yaml"))

for t in tenants:
    ns = Namespace(t["name"], metadata={"name": t["name"], "labels": {"tier": t["tier"]}})
    ResourceQuota(f"{t['name']}-quota",
        metadata={"namespace": ns.metadata["name"]},
        spec={"hard": t["quota"]},
        opts=pulumi.ResourceOptions(depends_on=[ns]),
    )
```

```yaml
# tenants.yaml
- { name: team-a, tier: gold,   quota: { requests.cpu: "10",  requests.memory: 20Gi, pods: "30" }}
- { name: team-b, tier: silver, quota: { requests.cpu: "5",   requests.memory: 10Gi, pods: "15" }}
- { name: team-c, tier: bronze, quota: { requests.cpu: "2",   requests.memory: 4Gi,  pods: "5"  }}
```

Add a tenant by editing YAML. No new code.

### Step 3 — ComponentResource (45 min)
```python
class MLService(pulumi.ComponentResource):
    def __init__(self, name, *, image, replicas=2, hpa_max=10, opts=None):
        super().__init__("custom:ml:Service", name, None, opts)
        child_opts = pulumi.ResourceOptions(parent=self)
        
        self.deployment = Deployment(name, ..., opts=child_opts)
        self.service    = Service(name, ..., opts=child_opts)
        self.ingress    = Ingress(name, ..., opts=child_opts)
        self.hpa        = HorizontalPodAutoscaler(name, ..., opts=child_opts)
        # ServiceMonitor if CRD installed
        
        self.register_outputs({
            "url": self.ingress.spec["rules"][0]["host"],
            "service_dns": self.service.metadata["name"].apply(lambda n: f"{n}.default.svc.cluster.local"),
        })

# Use:
recs = MLService("recs", image="iris-api:v3.2")
fraud = MLService("fraud", image="fraud-detector:v1.5", replicas=4, hpa_max=20)
```

### Step 4 — Stack references (30 min)
Networking stack:
```python
# pulumi-network/__main__.py
vpc = awsx.ec2.Vpc("vpc", ...)
pulumi.export("vpc_id", vpc.vpc_id)
pulumi.export("private_subnets", vpc.private_subnet_ids)
```

App stack:
```python
# pulumi-app/__main__.py
net = pulumi.StackReference("org/pulumi-network/prod")
vpc_id = net.get_output("vpc_id")
private_subnets = net.get_output("private_subnets")

cluster = aws.eks.Cluster("ml", vpc_config={
    "subnet_ids": private_subnets,
})
```

### Step 5 — CrossGuard policies (30 min)
```ts
// policy-pack/index.ts
import { PolicyPack, validateResourceOfType } from "@pulumi/policy";

new PolicyPack("ml-policies", {
    policies: [{
        name: "rds-must-encrypt",
        description: "RDS instances must have storage encryption.",
        enforcementLevel: "mandatory",
        validateResource: validateResourceOfType(aws.rds.Instance, (i, args, report) => {
            if (!i.storageEncrypted) report("RDS instance must have storage_encrypted=true.");
        }),
    }, {
        name: "k8s-no-privileged",
        enforcementLevel: "mandatory",
        validateResource: validateResourceOfType(k8s.core.v1.Pod, (p, _, report) => {
            for (const c of p.spec.containers ?? []) {
                if (c.securityContext?.privileged === true) report(`Container ${c.name} is privileged.`);
            }
        }),
    }],
});
```

Apply to project: `pulumi policy enable org/ml-policies`. Preview blocks non-compliant resources.

### Step 6 — Tests (15 min)
```python
# tests/test_components.py
import pulumi
class MyMocks(pulumi.runtime.Mocks):
    def new_resource(self, args):
        return [args.name + "_id", args.inputs]
    def call(self, args):
        return {}

pulumi.runtime.set_mocks(MyMocks())

# Test that MLService creates the expected children
def test_ml_service_creates_4_resources():
    s = MLService("test", image="my-img:1.0")
    # ... assertions on s.deployment, s.service, s.ingress, s.hpa
```

## Deliverables

1. Working multi-tenant deployment from YAML.
2. ComponentResource demonstrated.
3. Two stacks with reference.
4. Policy pack with at least 3 policies.
5. `PULUMI_PATTERNS.md` documenting when to reach for each technique.

## Validation

- [ ] Adding a tenant to YAML adds matching K8s resources after `pulumi up`.
- [ ] ComponentResource appears as a single node in `pulumi stack graph`.
- [ ] Policy violation blocks preview with clear message.
- [ ] Stack reference works without circular dependencies.

## Stretch goals

- Build a **provider** (custom resource type) to integrate with an internal service.
- Use **Automation API** to drive Pulumi programmatically from your own app.
- Compare project to equivalent Terraform; document lines of code delta.

## Common pitfalls

- **`apply` vs raw value** — Pulumi outputs are async. Always `.apply(lambda x: ...)` for value access.
- **ComponentResource without parent option on children** — Children appear as top-level in graph; lifecycle wrong.
- **CrossGuard policy too strict for legacy** — Add a "warning" enforcement level for migration.
