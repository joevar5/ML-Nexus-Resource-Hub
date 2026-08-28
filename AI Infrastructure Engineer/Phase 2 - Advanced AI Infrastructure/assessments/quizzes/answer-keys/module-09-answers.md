# Module 09: Infrastructure as Code — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-09-infrastructure-as-code/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the primary benefit of using Infrastructure as Code over manual infrastructure management?

**Answer:** C) Infrastructure can be version controlled and reproduced consistently

**Explanation:**
The defining advantage of IaC is that infrastructure definitions live in source control alongside application code, enabling code review, diff history, rollbacks, and deterministic recreation of environments. Manual click-ops cannot guarantee consistent reproduction across regions, accounts, or environments, and there is no audit trail of who changed what.

**Common Mistakes:**
- Choosing A — initial deploys are often *slower* with IaC because of the upfront authoring effort; the win comes from repeatability over time.
- Choosing D — IaC does not replace cloud providers; it codifies how you provision resources on them.

**Related Material:** `lessons/mod-09-infrastructure-as-code/01-introduction-to-iac.md`

---

## Question 2
**Q:** Which statement best describes a declarative approach to IaC?

**Answer:** B) You describe the desired end state and the tool figures out how to achieve it

**Explanation:**
Declarative tools like Terraform and Pulumi take a desired-state specification and reconcile reality to match it, computing the necessary create/update/delete operations. This contrasts with imperative scripts where you write the exact steps. Declarative approaches give you idempotency for free — running the same config twice converges to the same result.

**Common Mistakes:**
- Choosing A — step-by-step instructions describe an *imperative* approach, not declarative.
- Choosing D — shell scripts that run commands sequentially are imperative procedural automation.

**Related Material:** `lessons/mod-09-infrastructure-as-code/01-introduction-to-iac.md`

---

## Question 3
**Q:** What is the purpose of Terraform state?

**Answer:** B) To track the current state of managed infrastructure

**Explanation:**
The state file is Terraform's source of truth mapping configuration blocks to real resource IDs and attributes. It allows Terraform to detect drift, compute minimal diffs in `terraform plan`, and build a dependency graph for updates. Without state, Terraform would have no way to know which resources it previously created.

**Common Mistakes:**
- Choosing A — state is not a credentials store; secrets there are a leakage risk, not a feature.
- Choosing C — the `.tf` configuration files (not state) define what should exist.

**Related Material:** `lessons/mod-09-infrastructure-as-code/03-terraform-state-management.md`

---

## Question 4
**Q:** Why should you use a remote state backend (like S3) instead of local state?

**Answer:** B) It enables team collaboration and prevents concurrent modifications

**Explanation:**
A remote backend lets every team member and CI pipeline work against a single canonical state file rather than each engineer carrying a divergent local copy. Paired with a lock mechanism (e.g., DynamoDB for the S3 backend) it serializes concurrent `apply` operations, preventing the corruption that occurs when two writers race against the same state.

**Common Mistakes:**
- Choosing A — remote backends usually add latency vs. local disk, not reduce it.
- Choosing D — remote backends store state, but they do not automatically back up your underlying cloud resources.

**Related Material:** `lessons/mod-09-infrastructure-as-code/03-terraform-state-management.md`

---

## Question 5
**Q:** How does Terraform determine the order in which to create resources?

**Answer:** C) By analyzing implicit and explicit dependencies

**Explanation:**
Terraform builds a directed acyclic graph from references between resources (implicit dependencies, e.g. `aws_subnet.main.id` referenced in another resource) and from `depends_on` declarations (explicit dependencies). It then walks the graph in topological order, parallelizing independent branches. File order and resource names do not influence the ordering.

**Common Mistakes:**
- Choosing A or B — Terraform ignores file ordering and resource names when choosing execution order.
- Choosing D — Terraform never relies on random ordering with retries; it computes the correct order ahead of time.

**Related Material:** `lessons/mod-09-infrastructure-as-code/02-terraform-fundamentals.md`

---

## Question 6
**Q:** What is the correct way to reference a variable named `instance_type` in a Terraform resource?

**Answer:** B) `var.instance_type`

**Explanation:**
HCL exposes input variables through the `var.` namespace. You can use the reference bare (e.g., `instance_type = var.instance_type`) or inside a template string as `"${var.instance_type}"`. There is no `variable.` namespace, and bare `$instance_type` is not valid HCL.

**Common Mistakes:**
- Choosing A — `${instance_type}` is missing the `var.` prefix; interpolation syntax is `${var.instance_type}`.
- Choosing C — `variable` is the *block* keyword used to declare variables, not the access namespace.

**Related Material:** `lessons/mod-09-infrastructure-as-code/02-terraform-fundamentals.md`

---

## Question 7
**Q:** When should you use `for_each` instead of `count` in Terraform?

**Answer:** B) When you want to create resources from a map or set and need stable identifiers

**Explanation:**
`for_each` keys resources by string identifiers from a map or set, so reordering or removing one entry does not destroy and recreate unrelated resources. `count`-indexed resources are addressed by integer position (`[0]`, `[1]`, ...), which causes destructive churn when the underlying list changes. Stable identifiers matter especially for stateful infrastructure like databases or persistent volumes.

**Common Mistakes:**
- Choosing A — for exactly one resource you usually omit both meta-arguments entirely.
- Choosing D — `for_each` works on resources, data sources, and modules; it is not module-specific.

**Related Material:** `lessons/mod-09-infrastructure-as-code/06-advanced-iac-patterns.md`

---

## Question 8
**Q:** Which Terraform resource meta-argument would you use to provision expensive GPU instances only in production?

**Answer:** B) `count = var.environment == "prod" ? 3 : 0`

**Explanation:**
Combining `count` with a conditional expression lets you create N copies of a resource only when a condition is true, and zero copies otherwise. Setting `count = 0` is the idiomatic Terraform pattern for "skip this resource in this environment." This avoids forking the configuration per environment.

**Common Mistakes:**
- Choosing A — `prevent_destroy` only blocks deletion of already-created resources; it does not gate creation.
- Choosing D — `for_each` on `var.gpu_instances` does not, on its own, restrict creation to production unless the input map is also environment-gated.

**Related Material:** `lessons/mod-09-infrastructure-as-code/04-building-ai-infrastructure-terraform.md`

---

## Question 9
**Q:** What does `terraform plan` do?

**Answer:** B) Shows a preview of changes without making them

**Explanation:**
`terraform plan` refreshes state, compares it against your configuration, and prints the create/update/destroy actions that would occur. It does not mutate cloud resources, making it the safe primitive used in code reviews and CI pre-checks. Only `terraform apply` (or `apply` of a saved plan file) actually performs changes.

**Common Mistakes:**
- Choosing A — that is the job of `terraform apply`.
- Choosing C or D — formatting is `terraform fmt`; initialization is `terraform init`.

**Related Material:** `lessons/mod-09-infrastructure-as-code/02-terraform-fundamentals.md`

---

## Question 10
**Q:** What is the purpose of state locking with DynamoDB in an S3 backend?

**Answer:** B) To prevent concurrent terraform operations that could corrupt state

**Explanation:**
DynamoDB acts as a distributed lock manager: before Terraform reads or writes the S3-backed state, it writes a lock item to a DynamoDB table and releases it when finished. If another operation tries to acquire the lock, it fails fast rather than racing. This is critical for teams and CI pipelines.

**Common Mistakes:**
- Choosing A — encryption of state is configured separately via S3 server-side encryption or KMS, not via DynamoDB.
- Choosing D — state file versioning is handled by S3 versioning, not DynamoDB.

**Related Material:** `lessons/mod-09-infrastructure-as-code/03-terraform-state-management.md`

---

## Question 11
**Q:** What is a Terraform data source used for?

**Answer:** B) To fetch information about existing resources

**Explanation:**
Data sources (`data` blocks) perform read-only queries against providers — for example, looking up the latest Amazon Linux AMI, an existing VPC by tag, or the current account ID. The results can be referenced by other resources, allowing your configuration to react to infrastructure it does not own. Terraform never modifies data sources.

**Common Mistakes:**
- Choosing A — `resource` blocks (not `data` blocks) create new infrastructure.
- Choosing C — sensitive data should be stored in Secrets Manager, SSM Parameter Store, or Vault, not in data sources.

**Related Material:** `lessons/mod-09-infrastructure-as-code/02-terraform-fundamentals.md`

---

## Question 12
**Q:** What is the primary purpose of Terraform modules?

**Answer:** B) To organize and reuse infrastructure code

**Explanation:**
Modules are reusable bundles of related resources exposed through inputs (variables) and outputs. They let you publish opinionated building blocks (e.g., a "GPU training cluster" module) and consume them across environments and teams without copy-paste. They are the Terraform analogue of functions or packages in a programming language.

**Common Mistakes:**
- Choosing A — modules do not measurably speed up Terraform execution; they change code organization.
- Choosing C or D — encryption and state management are orthogonal backend concerns.

**Related Material:** `lessons/mod-09-infrastructure-as-code/06-advanced-iac-patterns.md`

---

## Question 13
**Q:** What is the main difference between Pulumi and Terraform?

**Answer:** B) Pulumi uses general-purpose programming languages, Terraform uses HCL

**Explanation:**
Pulumi expresses infrastructure in real programming languages (Python, TypeScript, Go, C#, Java), so you can use loops, classes, abstractions, and unit-testing frameworks natively. Terraform uses HCL, a purpose-built configuration language. Both are declarative under the hood and both support multi-cloud deployments and state management.

**Common Mistakes:**
- Choosing A — Terraform is cloud-agnostic and supports hundreds of providers, not just AWS.
- Choosing C — Pulumi *does* support state management (via Pulumi Cloud or self-managed backends).
- Choosing D — both tools present a declarative model to the user, regardless of host language.

**Related Material:** `lessons/mod-09-infrastructure-as-code/05-pulumi-infrastructure-as-software.md`

---

## Question 14
**Q:** Why are spot instances beneficial for ML training workloads in Terraform?

**Answer:** B) They're up to 90% cheaper than on-demand instances

**Explanation:**
Spot instances utilize spare cloud capacity at deeply discounted prices, often 70–90% off on-demand. ML training jobs tolerate interruption well when they checkpoint regularly, so spot pricing is a natural fit. Hardware specs and network performance are identical to on-demand counterparts — only the interruption guarantee differs.

**Common Mistakes:**
- Choosing A or D — spot and on-demand share identical hardware and networking; performance is not better.
- Choosing C — spot capacity can be reclaimed by the provider with as little as two minutes' notice; interruption is the trade-off.

**Related Material:** `lessons/mod-09-infrastructure-as-code/04-building-ai-infrastructure-terraform.md`

---

## Question 15
**Q:** How do you mark a Terraform output as sensitive to prevent it from being displayed in the CLI?

**Answer:** B) `output "password" { value = var.password, sensitive = true }`

**Explanation:**
Setting `sensitive = true` on an output block causes Terraform to redact the value in CLI output and plan summaries, printing `<sensitive>` instead. The value still ends up in state (so encrypt your remote state), but it will not be displayed during normal runs. There are no `hidden` or `encrypted` arguments on outputs.

**Common Mistakes:**
- Choosing A — without `sensitive = true`, the password is printed in plain text.
- Choosing C or D — `hidden` and `encrypted` are not valid Terraform output arguments.

**Related Material:** `lessons/mod-09-infrastructure-as-code/08-security-best-practices.md`

---

## Question 16
**Q:** What is the purpose of Terraform workspaces?

**Answer:** B) To manage multiple environments (dev, staging, prod) with the same configuration

**Explanation:**
Workspaces let a single configuration root maintain multiple isolated state files, one per workspace. Switching workspace (`terraform workspace select staging`) changes which state Terraform reads and writes. Note: workspaces are one of several environment strategies; for strong blast-radius isolation between prod and non-prod, separate backends/directories are often preferred.

**Common Mistakes:**
- Choosing A — multi-cloud deployment is handled via providers, not workspaces.
- Choosing C — workspaces do not parallelize operations; they isolate state.

**Related Material:** `lessons/mod-09-infrastructure-as-code/06-advanced-iac-patterns.md`

---

## Question 17
**Q:** What is a key principle of GitOps for infrastructure?

**Answer:** B) Infrastructure changes go through Git pull requests and automated pipelines

**Explanation:**
GitOps treats the Git repository as the single source of truth for desired infrastructure state. Every change is a pull request: reviewed, approved, merged, and then reconciled to the cloud by an automated pipeline or controller. This gives you auditability, peer review, and easy rollback by reverting commits.

**Common Mistakes:**
- Choosing A — direct console changes are precisely what GitOps prohibits; they cause configuration drift.
- Choosing D — GitOps gates *who can merge*, not who can be on the team; anyone can propose changes via PR.

**Related Material:** `lessons/mod-09-infrastructure-as-code/07-gitops-iac-cicd.md`

---

## Question 18
**Q:** What does the `terraform fmt` command do?

**Answer:** B) Formats code to canonical style

**Explanation:**
`terraform fmt` rewrites `.tf` files to Terraform's canonical style — consistent indentation, alignment of `=` signs in argument blocks, and standardized spacing. It is the IaC equivalent of `gofmt` or `prettier`. Running it in CI (with `terraform fmt -check`) prevents formatting churn in code reviews.

**Common Mistakes:**
- Choosing A — syntax validation is `terraform validate`.
- Choosing C or D — plans come from `terraform plan`; initialization from `terraform init`.

**Related Material:** `lessons/mod-09-infrastructure-as-code/02-terraform-fundamentals.md`

---

## Question 19
**Q:** Which lifecycle meta-argument prevents accidental destruction of critical resources?

**Answer:** B) `prevent_destroy = true`

**Explanation:**
Inside a `lifecycle` block, `prevent_destroy = true` causes Terraform to error out at plan time if the change set includes destroying that resource. It is a guardrail for stateful infrastructure (databases, KMS keys, S3 buckets with data). To intentionally delete a protected resource, you must remove the flag and re-plan.

**Common Mistakes:**
- Choosing A — `create_before_destroy` changes *ordering* during replacement; it does not block deletion.
- Choosing C — `ignore_changes` suppresses drift detection on specific attributes, unrelated to deletion.
- Choosing D — `destroy = false` is not a valid lifecycle argument.

**Related Material:** `lessons/mod-09-infrastructure-as-code/08-security-best-practices.md`

---

## Question 20
**Q:** Why should ML training EC2 instances use IAM roles instead of hardcoded credentials?

**Answer:** B) IAM roles provide temporary, automatically rotated credentials

**Explanation:**
Instance profiles deliver short-lived STS credentials to the EC2 instance metadata service, and the SDKs refresh them automatically before expiry. There are no long-lived keys on disk that can leak via logs, snapshots, or compromised containers. This is the AWS-recommended pattern for any workload running on EC2, ECS, EKS, or Lambda.

**Common Mistakes:**
- Choosing A or D — IAM roles do not change runtime performance or pricing.
- Choosing C — AWS allows access keys on EC2; it just strongly recommends against them.

**Related Material:** `lessons/mod-09-infrastructure-as-code/08-security-best-practices.md`

---

## Question 21
**Q:** Terraform state files can contain sensitive data. What's the best practice?

**Answer:** B) Use remote state with encryption (e.g., S3 with KMS)

**Explanation:**
Remote state on S3 with server-side encryption (SSE-KMS), bucket versioning, and access control via IAM policies gives you confidentiality, recoverability, and team collaboration in one configuration. The Terraform Cloud / Enterprise backend offers equivalent guarantees. State should never live in Git or unencrypted shared storage.

**Common Mistakes:**
- Choosing A — committing state to Git exposes plaintext secrets in your repo history.
- Choosing C — local-only state blocks collaboration and is easily lost.
- Choosing D — emailing state circulates secrets uncontrollably and provides no locking.

**Related Material:** `lessons/mod-09-infrastructure-as-code/03-terraform-state-management.md`

---

## Question 22
**Q:** What do tfsec and Checkov do in an IaC workflow?

**Answer:** B) Scan infrastructure code for security issues and misconfigurations

**Explanation:**
tfsec and Checkov are static analysis tools that parse Terraform (and other IaC) and flag insecure patterns such as public S3 buckets, unencrypted volumes, permissive security groups, or missing logging. They run in seconds during CI, catching issues before they become production incidents. Both ship with hundreds of rules mapped to standards like CIS and PCI.

**Common Mistakes:**
- Choosing A — formatting is the job of `terraform fmt`.
- Choosing C — documentation is generated by tools like terraform-docs.
- Choosing D — neither tool deploys infrastructure; they only analyze it.

**Related Material:** `lessons/mod-09-infrastructure-as-code/08-security-best-practices.md`

---

## Question 23
**Q:** Which Terraform resource would you use to auto-scale ML inference servers based on load?

**Answer:** B) `aws_autoscaling_group`

**Explanation:**
An Auto Scaling Group manages a fleet of EC2 instances behind scaling policies that react to CloudWatch metrics (CPU, request count, custom GPU utilization, SQS depth, etc.). It maintains min/max/desired capacity automatically. For containerized inference you might wrap this with ECS or EKS, but the underlying compute auto-scaling primitive on EC2 is the ASG.

**Common Mistakes:**
- Choosing A — a bare `aws_instance` is a single VM with no scaling logic.
- Choosing C — Lambda auto-scales but is not appropriate for long-running GPU inference servers.
- Choosing D — `aws_ecs_task` defines a single task; ECS scaling needs a service plus a scaling policy.

**Related Material:** `lessons/mod-09-infrastructure-as-code/04-building-ai-infrastructure-terraform.md`

---

## Question 24
**Q:** What does `terraform import` do?

**Answer:** B) Brings existing infrastructure under Terraform management

**Explanation:**
`terraform import` (or, in newer Terraform, `import` blocks) records a real cloud resource in state under a Terraform address you supply. You still have to write the matching `resource` block by hand so subsequent plans show no diff. This is how you onboard click-ops-created resources into IaC.

**Common Mistakes:**
- Choosing A — provider plugins are installed by `terraform init`.
- Choosing C — modules are pulled from the registry by `terraform init`/`get`.
- Choosing D — variable files are loaded automatically (`terraform.tfvars`) or via `-var-file`.

**Related Material:** `lessons/mod-09-infrastructure-as-code/06-advanced-iac-patterns.md`

---

## Question 25
**Q:** Which tool can estimate the cost of infrastructure before applying Terraform changes?

**Answer:** B) Infracost

**Explanation:**
Infracost parses Terraform plans and queries cloud pricing APIs to produce a dollar estimate for the proposed change. It integrates into pull requests so reviewers see, for example, "+$1,240/month for two p3.8xlarge instances" before merging. This makes cost a first-class signal in code review.

**Common Mistakes:**
- Choosing A — `terraform plan` shows resource changes, not pricing.
- Choosing C — `terraform validate` checks syntax and internal consistency only.
- Choosing D — tfsec scans for security misconfigurations, not cost.

**Related Material:** `lessons/mod-09-infrastructure-as-code/07-gitops-iac-cicd.md`

---

## Question 26
**Q:** Scenario 1 — Multi-Environment ML Infrastructure. You need separate dev (1× t3.large), staging (2× p3.2xlarge), and prod (4× p3.8xlarge) environments sharing the same codebase. What's the best approach to manage these environments with Terraform?

**Answer:** B) Use Terraform workspaces with variable files for each environment

**Explanation:**
A single configuration parameterized by an `environment` variable, combined with workspace-specific `.tfvars` (or separate backend directories per environment), keeps the code DRY while allowing per-environment instance counts and types. A map variable keyed by environment (`{dev = {...}, staging = {...}, prod = {...}}`) lets a single `aws_instance` block expand correctly per workspace. For stronger prod isolation, many teams use separate state backends per environment rather than workspaces alone — both are valid expressions of the same parameterization principle.

**Common Mistakes:**
- Choosing A — three copy-pasted projects guarantee drift; bug fixes have to be applied three times and inevitably diverge.
- Choosing C — hardcoded values make per-environment differences impossible without editing source for every deploy.
- Choosing D — mixing manual provisioning with Terraform causes state-vs-reality drift and defeats the point of IaC.

**Related Material:** `lessons/mod-09-infrastructure-as-code/06-advanced-iac-patterns.md`

---

## Question 27
**Q:** Scenario 2 — State File Corruption. Two engineers ran `terraform apply` simultaneously and the production GPU cluster state is now inconsistent with reality. What steps should you take to recover and prevent this in the future?

**Answer:** Halt all Terraform operations, restore state from a prior S3 version, verify and refresh, then enforce state locking and pipeline-only applies going forward.

**Explanation:**
The immediate priority is to stop further writes, then recover state from S3 object versioning (`aws s3api list-object-versions` followed by `get-object --version-id`). Run `terraform plan` to confirm the restored state matches reality, and use `terraform refresh` (or targeted imports) to reconcile any genuine drift. The structural fix is to enable a DynamoDB lock table on the S3 backend, turn on S3 versioning and KMS encryption, and route all applies through a CI/CD pipeline (Atlantis, GitHub Actions) that requires PR approval — so two humans cannot race against the same state again.

**Common Mistakes:**
- Skipping the "stop everything" step and continuing to apply, which can compound the corruption.
- Forgetting to enable DynamoDB locking after recovery — without it, the same incident recurs.
- Not enabling S3 versioning before the incident, leaving you with no prior state to restore.
- Relying solely on team discipline instead of pipeline gates and automated locking.

**Related Material:** `lessons/mod-09-infrastructure-as-code/03-terraform-state-management.md`

---

## Question 28
**Q:** Scenario 3 — Secret Management. An ML inference service on EC2 needs a database password, an external ML-API key, and S3 access. How should you securely manage these secrets in your Terraform configuration?

**Answer:** Store the database password and API key in AWS Secrets Manager, use an IAM instance profile (not access keys) for S3 access, retrieve secrets at runtime via the instance role, and keep tfvars out of Git with encrypted remote state.

**Explanation:**
Secrets like passwords and API keys belong in a managed secret store (AWS Secrets Manager or SSM Parameter Store) so they can be rotated and audited centrally; Terraform creates the secret resources but the values are injected via `TF_VAR_*` from secure CI, never committed. For AWS API access (S3), use an IAM role attached via instance profile — that gives short-lived rotated credentials with no static keys on disk. User-data on the instance fetches the runtime secrets with `aws secretsmanager get-secret-value`. Combine this with `.gitignore` entries for `*.tfvars` and KMS-encrypted S3 remote state to close the loop.

**Common Mistakes:**
- Hardcoding secrets directly in `.tf` files or committing `terraform.tfvars` with secret values to Git.
- Embedding long-lived AWS access keys in user-data or env vars instead of using an instance profile.
- Storing secrets only in plaintext environment variables on the instance with no rotation strategy.
- Leaving the Terraform state file unencrypted — even if secrets are external, their values land in state when referenced.

**Related Material:** `lessons/mod-09-infrastructure-as-code/08-security-best-practices.md`

---
