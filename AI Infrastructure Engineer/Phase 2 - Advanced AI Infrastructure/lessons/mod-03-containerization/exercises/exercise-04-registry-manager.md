# Exercise 04: Container Registry Manager

**Duration:** 9-11 hours
**Difficulty:** Advanced
**Prerequisites:** Docker, Python 3.9+, cloud CLI access (AWS/GCP/Azure), Docker Registry HTTP API v2

## Objective

Build `registry-manager`, a Python CLI that synchronizes container images across multiple registries (ECR, GCR, ACR), runs a dev → staging → prod promotion pipeline with approval gates, enforces image retention policies, and logs every operation to an audit trail. Demonstrate it syncing a real image across two regions and promoting it through all three environments.

## Why this matters

Multi-region ML platforms need images available where they're deployed — pulling a 2.5 GB image cross-region can add minutes to pod startup and block autoscaling during traffic spikes. At the same time, unmanaged registries accumulate thousands of untagged images that cost real money and complicate audits ("which image is actually running in prod, and who put it there?"). A registry manager that automates sync, promotion, and cleanup turns this from a manual, error-prone chore into a repeatable pipeline.

## Requirements

1. **Registry abstraction**: a common interface (`list_repositories`, `list_tags`, `get_manifest`, `get_metadata`, `tag_image`, `delete_image`) implemented for ECR at minimum, with GCR/ACR as stretch.
2. **Sync**: copy an image (or all tags matching a filter) from one registry to another, skipping the copy when the target already has the same digest, and verifying digests match after transfer.
3. **Promotion pipeline**: move an image from `dev` → `staging` → `prod`, tagging it with an environment-scoped tag, optionally requiring approval and a passing security scan before a prod promotion.
4. **Retention policy**: delete images older than N days, keep only the last N versions, and always preserve tags matching patterns like `prod-*` or `latest`.
5. **Audit log**: record every sync/promote/delete operation (who, what, when) to a queryable log, and calculate storage cost per registry.

## Step-by-step

### Step 1 — Registry interface + ECR (2 hours)
Define `RegistryInterface` (ABC) and implement `ECRRegistry` with `boto3`:
```python
self.client = boto3.client("ecr", region_name=region)
self.client.describe_repositories()
self.client.list_images(repositoryName=repo)
self.client.batch_get_image(repositoryName=repo, imageIds=[{"imageTag": tag}])
self.client.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])
```
`tag_image` fetches the source manifest via `batch_get_image` and re-pushes it under the new tag with `put_image` — ECR treats tagging as "push this manifest under a different tag," there's no native rename.

### Step 2 — Sync (2 hours)
```python
def sync_image(job: SyncJob) -> SyncResult:
    if target.image_exists(repo, tag) and target.get_metadata(repo, tag).digest == source_digest:
        return SyncResult(success=True, size_bytes=0)  # already synced
    docker_pull(source_url); docker_tag(source_url, target_url); docker_push(target_url)
    assert target.get_metadata(repo, tag).digest == source_digest
```
For real cross-cloud syncs, prefer `skopeo copy docker://src docker://dst` over pull/tag/push — it copies layers directly without a local Docker daemon round-trip. Authenticate each side first (`aws ecr get-login-password`, `gcloud auth print-access-token`).

### Step 3 — Promotion pipeline (2-2.5 hours)
```python
new_tag = f"{to_env.value}-{tag}-{timestamp}"
if policy.require_approval and not request.approved_by:
    raise ValueError("approval required")
if policy.require_security_scan:
    assert security_scan_fn(repo, tag).passed
source_registry.tag_image(repo, tag, new_tag)
for target in registries_for_env(to_env):
    synchronizer.sync_image(SyncJob(source, target, repo, new_tag))
audit_logger.log_event(...)
```
Wire `security_scan_fn` to the `containersec` CLI from Exercise 05 if you built it — this is exactly where a policy gate belongs.

### Step 4 — Retention policy (90 min)
```python
def should_delete(metadata, now) -> bool:
    if any(fnmatch(metadata.tag, p) for p in preserve_tags): return False
    if max_age_days and (now - metadata.created_at).days > max_age_days: return True
    if min_pull_count and metadata.pull_count < min_pull_count: return True
    return False
```
Run with `dry_run=True` first and print what *would* be deleted before enabling real deletion.

### Step 5 — Audit + cost (90 min)
Append one JSON object per line to an audit log (`event_type`, `user`, `registry`, `repository`, `tag`, `correlation_id`). Sum image sizes per registry and multiply by a per-GB storage rate to estimate monthly cost; flag repositories with duplicate digests under different tags as a cleanup opportunity.

### Step 6 — CLI (60 min)
```bash
registry-manager list-repos --config config/registries.yaml
registry-manager sync --source us-east-1-ecr --target eu-west-1-ecr --repository my-model --tag v1.2.3
registry-manager promote --repository my-model --tag v1.2.3 --from-env staging --to-env prod --approved-by alice
registry-manager cleanup --registry us-east-1-ecr --max-age-days 90 --dry-run
registry-manager costs --registry us-east-1-ecr
```

## Deliverables

1. `registry_manager` package with `registry/` (base + ECR at minimum), `sync.py`, `promotion.py`, `retention.py`, `audit.py`, `cost_analyzer.py`, `cli.py`.
2. `config/registries.yaml` defining at least two registries and a dev→staging→prod promotion policy.
3. A demonstrated end-to-end run: sync an image cross-region, promote it through all three environments, and a dry-run cleanup report.

## Validation

- [ ] `sync` skips the transfer when source and target digests already match.
- [ ] `promote` to `prod` fails without `--approved-by` when the policy requires approval, and succeeds with it.
- [ ] `cleanup --dry-run` reports deletions without actually deleting; `--no-dry-run` deletes and frees the reported space.
- [ ] Retention correctly preserves tags matching `prod-*` even when they're older than `max_age_days`.
- [ ] Every sync/promote/delete produces one audit log entry with a timestamp and actor.

## Stretch goals

- Add GCR and ACR implementations behind the same interface, and test cross-cloud sync (ECR → GCR).
- Sign promoted images with cosign and verify the signature before allowing a prod promotion.
- Build a Kubernetes operator that watches an `ImagePolicy` CRD and triggers promotions automatically.
- Add a Grafana dashboard sourcing the audit log and cost analyzer for registry-wide visibility.

## Common pitfalls

- **Treating ECR tagging as a rename** — `put_image` creates a new tag pointing at the same manifest; the old tag still exists unless you explicitly delete it. Forgetting this leaves both `dev-v1` and `staging-v1` pointing at identical images with no clear "latest."
- **Skipping the digest verification after sync** — a registry-side rate limit or truncated push can silently produce a partial image. Always compare source and target digests before marking a sync successful.
- **Retention deleting a digest still tagged elsewhere** — deleting by tag in some registries removes only that tag; deleting by digest removes every tag pointing to it. Know which one your policy needs, or you'll take down an image another tag still references.
- **No pagination on `list_images`/`describe_images`** — ECR paginates at 100 results by default; a naive single call under-reports repository contents on any registry with real usage.
