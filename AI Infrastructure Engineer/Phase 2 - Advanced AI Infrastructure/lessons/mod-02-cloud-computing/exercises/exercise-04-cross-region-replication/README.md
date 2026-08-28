# Exercise 04: Cross-Region Replication for ML Artifacts

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01-03; AWS or GCP account with admin

## Objective

Build an `artifact-replicator` tool that keeps model artifacts and training datasets synchronized across two cloud regions for disaster recovery and lower-latency inference. By the end you'll have an idempotent replicator with conflict detection, bandwidth control, and per-artifact integrity verification.

## Why this matters

Models trained in one region often must serve in another (latency, regulatory, DR). Hand-rolled `aws s3 sync` cron jobs lose track of half-replicated objects, eat bandwidth quotas, and silently swallow integrity failures. A proper replicator is one of the first "real" tools an ML platform team ships.

## Requirements

Build `artifact-replicator` as a Python tool with these commands:

```text
artifact-replicator status   # show source vs dest manifest diff
artifact-replicator sync     # one-shot replication
artifact-replicator watch    # continuous replication with rate limit
artifact-replicator verify   # validate all replicated objects via SHA256
```

### Functional requirements

1. Source and destination buckets are configurable (any combination of S3, GCS, Azure Blob).
2. Replicate only **new or changed** objects (manifest-based, not full re-scan each run).
3. **Atomic** per-object: a partial write must not be exposed; use a temp-then-rename pattern.
4. **Bandwidth limit** (configurable MB/s, default 50).
5. **Concurrency limit** (configurable, default 4 parallel transfers).
6. **Integrity**: compute and verify SHA256 on both ends; fail loudly on mismatch.
7. **Resume**: interrupted runs continue where they left off.

### Non-functional

8. **Structured JSON logs** with object key, size, duration, throughput per transfer.
9. **Prometheus metrics**: `bytes_transferred_total`, `objects_failed_total`, `replication_lag_seconds`.
10. **TLS in transit** (handled by provider SDKs).

## Suggested structure

```
artifact-replicator/
├── pyproject.toml
├── src/replicator/
│   ├── cli.py
│   ├── backends/        # s3.py, gcs.py, azure.py implementing the same interface
│   ├── manifest.py      # local DB tracking what's been replicated
│   ├── transfer.py      # the per-object replication logic
│   └── limits.py        # bandwidth + concurrency throttling
└── tests/
```

### Backend interface

```python
class StorageBackend(Protocol):
    def list(self, prefix: str = "") -> Iterator[ObjectMeta]: ...
    def get_stream(self, key: str) -> Iterator[bytes]: ...
    def put_stream(self, key: str, stream: Iterator[bytes], size: int) -> str: ...
    def head(self, key: str) -> ObjectMeta | None: ...
    def delete(self, key: str) -> None: ...
```

## Step-by-step

### Step 1 — Project + backend interface (30 min)
Write the Protocol and S3 implementation using boto3. List + head + get_stream + put_stream.

### Step 2 — Manifest (30 min)
SQLite database tracking `(key, src_etag, src_size, src_mtime, dst_etag, dst_size, dst_mtime, last_synced_at)`. Use SQLAlchemy or raw `sqlite3`.

### Step 3 — Diff logic (30 min)
`status` command compares source listing vs manifest and prints:
- New objects (in source, not in manifest)
- Updated objects (etag changed since last sync)
- Missing objects (in manifest, not in source — deleted upstream)
- In sync (no action needed)

### Step 4 — Atomic transfer (45 min)
```python
def replicate_one(src: Backend, dst: Backend, key: str, rate_limit: int):
    src_meta = src.head(key)
    if not src_meta: return
    temp_key = f".tmp-replication/{uuid.uuid4()}/{key}"
    sha = hashlib.sha256()
    def streamed():
        for chunk in src.get_stream(key):
            sha.update(chunk)
            yield chunk
    dst_etag = dst.put_stream(temp_key, streamed(), src_meta.size)
    dst.rename(temp_key, key)         # atomic per-object
    return ReplicationResult(sha=sha.hexdigest(), etag=dst_etag)
```

### Step 5 — Concurrency + rate limit (30 min)
`concurrent.futures.ThreadPoolExecutor` for parallelism. Token-bucket limiter (use `aiolimiter` or hand-rolled) for bandwidth.

### Step 6 — Verify command (30 min)
Re-download each replicated object, compute SHA256, compare to manifest. Failures: log and exit non-zero.

### Step 7 — Watch mode (30 min)
`watch` polls source every N seconds (default 60) and replicates deltas. Use S3 event notifications for true push-based replication in a stretch goal.

### Step 8 — Tests (30 min)
- Unit: mock backends, verify manifest math.
- Integration: moto for AWS, or LocalStack.
- Property: random object sizes, verify checksums match.

## Deliverables

1. CLI satisfying all 10 requirements.
2. Test suite with moto/LocalStack.
3. A `BENCHMARKS.md` documenting throughput on a 1000-object workload.
4. A short `OPERATIONS.md` describing how to run this as a cron or systemd timer.

## Validation

- [ ] `artifact-replicator status` shows accurate diff between buckets.
- [ ] `sync` replicates 1000 random files; rerun reports "0 new, 0 updated".
- [ ] Killing the process mid-replication and restarting completes correctly.
- [ ] `verify` catches a deliberately-corrupted destination object.
- [ ] Bandwidth limiter caps at the configured MB/s (validate with `iftop`).
- [ ] All metrics visible in Prometheus.

## Stretch goals

- Add **two-way replication** with conflict resolution (last-write-wins, or per-key user-supplied resolver).
- Replace polling with **S3 event notifications** (via SQS) for push-based replication.
- Add **encryption-at-rest verification**: check that destination objects use the configured KMS key.

## Common pitfalls

- **`aws s3 sync` semantics ≠ correct** — It compares mtimes, which are timezone-flaky and not authoritative. Use etags or content hashes.
- **Listing very large buckets is O(N) per run** — Use the manifest to avoid re-listing fully every cycle.
- **Cross-cloud transfer charges** — Replication between regions/clouds incurs egress; alert on monthly spend, not just sometimes.
- **Memory blowup on streaming** — Don't accumulate chunks in memory; iterate true streams.
