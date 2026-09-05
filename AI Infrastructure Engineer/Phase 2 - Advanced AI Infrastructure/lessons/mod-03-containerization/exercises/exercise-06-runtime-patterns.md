# Exercise 06: Runtime Debugging & ML Container Patterns

**Duration:** 5-5.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01 (Dockerfiles); curl/grep/strace familiarity

## Objective

**Part A** — diagnose three deliberately-broken running containers using only container-native debugging tools (`docker logs`, `docker exec`, `nsenter`, ephemeral debug containers).

**Part B** — implement four ML-specific container patterns not covered by generic Docker tutorials: model warmup at startup, init-container model preloading, sidecar-based dynamic batching, and read-only models with a configurable hot swap. Each pattern has a measurable impact on a real metric.

## Why this matters

Both halves of this exercise are about the same thing: keeping an already-running ML container healthy and fast. The skill gap between engineers who fix a production crashloop in 10 minutes and those who page senior staff is mostly methodology, and that methodology only sticks once you've also built the patterns that prevent the common failures in the first place — slow cold starts, blocked readiness probes, wasted GPU cycles, and risky restarts for a model update. Part A gives you the diagnostic muscle memory; Part B gives you the patterns worth diagnosing around.

---

## Part A — Container runtime debugging

### Requirements

You'll be given (and you can write your own) three broken containers. For each, identify the root cause and fix it.

**Bug 1: Application appears to start but receives no traffic**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app.py .
RUN pip install fastapi uvicorn
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
```

**Bug 2: Container starts then OOM-killed after ~10 seconds** — a PyTorch container that loads a 13B model into a 2GB memory limit.

**Bug 3: Container runs but predictions are wildly inconsistent** — a container that runs as root, mounts a host volume read-write, and another container on the same host periodically truncates the model file.

### Step-by-step debugging methodology

**Step 1 — Container is alive but unresponsive**
```bash
docker ps                                    # is it Running?
docker logs --tail 50 <id>                   # what did it say?
docker exec -it <id> sh                      # poke around
docker stats <id> --no-stream                # CPU/mem usage
```
For bug 1, the app listens on 127.0.0.1, so the port-mapping forwards to nothing. Fix: `--host 0.0.0.0`.

**Step 2 — Container restarts repeatedly**
```bash
docker inspect <id> --format '{{.State.ExitCode}} {{.State.Error}}'
docker inspect <id> --format '{{.State.OOMKilled}}'
dmesg | grep -i 'killed process'             # host-level OOM event
docker logs --previous <id>                  # logs from the prior failed instance
```
For bug 2, `OOMKilled=true` means the container was killed by cgroup limits. Fix: raise memory limit or use a quantized model.

**Step 3 — Behavior inconsistent over time** (the hardest case: predictions correct, then wrong, then correct again)
```bash
docker exec <id> sha256sum /models/current.bin   # compare hashes between calls
docker exec <id> stat /models/current.bin        # check mtime
```
Then check who else can write — inspect volume mounts, look for other containers sharing the mount. For bug 3, another container is modifying the shared host mount. Fix: mount read-only, or copy the model in at startup instead of sharing live.

**Step 4 — When `docker exec` isn't possible** (e.g. a distroless image with no shell):
```bash
kubectl debug -it <pod> --image=busybox --target=<container>
# Or with raw docker:
docker run -it --pid container:<id> --net container:<id> --rm busybox sh
```

**Step 5 — When the container crashes too fast to exec:**
```bash
docker run --rm -it --entrypoint sh <image>     # override the bad CMD
```

**Step 6 — Last resort: strace inside the container**
```bash
docker exec -it <id> sh
strace -p $(pidof python) -f -e trace=network 2>&1 | head -50
```

### Validation (Part A)

- [ ] You fixed all 3 bugs without consulting the answer key.
- [ ] Diagnosis includes the specific command and output that pointed to the root cause, not just the fix.
- [ ] A `DEBUGGING_CHEATSHEET.md` covers the 6-step methodology above plus your own additions.

### Common pitfalls (Part A)

- **Trusting `docker logs` exclusively** — apps that log to a file inside the container won't show up there; check the app's log config.
- **Skipping `docker inspect`** — most "weird" container problems show up in inspect output (mounts, env vars, restart policy, exit codes).
- **Restarting before reading logs** — `--previous` is your friend; without it, a restart wipes the evidence.
- **Confusing OOMKilled with a normal exit** — OOMKilled has its own inspect field; don't guess from exit code alone.
- **Running strace in production without thought** — it can pause a running process for a long time. Practice on staging first.

---

## Part B — ML-specific container patterns

### Pattern 1 — Model Warmup at Startup

**Problem:** First request after deploy takes 10× longer because Python/PyTorch JIT, allocator setup, and CUDA graphs compile lazily.
**Solution:** During lifespan startup, run synthetic predictions through the inference path to warm everything.
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    dummy = np.zeros((1, FEATURE_COUNT), dtype=np.float32)
    for _ in range(5):
        app.state.model.predict(dummy)
    log.info("model warmed up")
    yield
```
**Metric:** first-request latency on a freshly-started pod should land within 2× of steady-state, not 10×.

### Pattern 2 — Init Container for Model Download

**Problem:** Pulling a 7GB model into a container at startup blocks the readiness probe for minutes.
**Solution:** Use a Kubernetes initContainer to download the model to a shared volume; the main container loads from local disk.
```yaml
spec:
  initContainers:
    - name: model-download
      image: curlimages/curl:latest
      command: ["sh", "-c", "curl -fL $MODEL_URL -o /models/current.bin"]
      env: [{ name: MODEL_URL, value: "s3://models/iris/current.bin" }]
      volumeMounts: [{ name: models, mountPath: /models }]
  containers:
    - name: api
      image: iris-api:0.2
      volumeMounts: [{ name: models, mountPath: /models, readOnly: true }]
  volumes:
    - name: models
      emptyDir: { sizeLimit: 10Gi }
```
**Metric:** pod becomes Ready within seconds of the initContainer finishing.

### Pattern 3 — Dynamic Batching Sidecar

**Problem:** Per-request inference wastes GPU; small batches are inefficient.
**Solution:** A sidecar aggregates requests for ~50ms (or up to 32 items, whichever first) then forwards as a batch.
```yaml
spec:
  containers:
    - name: api            # accepts client requests, forwards to batcher
      image: iris-api-frontend:0.2
      ports: [{ containerPort: 8000 }]
    - name: batcher        # aggregates and calls model
      image: iris-api-batcher:0.2
      ports: [{ containerPort: 9000 }]
    - name: model          # actual inference, GPU-bound
      image: iris-api-model:0.2
      resources: { limits: { nvidia.com/gpu: 1 } }
```
**Metric:** GPU utilization rises from ~10% per-request to ~70% batched; throughput 5-10× higher with p99 latency at most 100ms worse.

### Pattern 4 — Hot Model Swap

**Problem:** Updating the model requires a rolling restart — wasteful and risky.
**Solution:** A loader thread watches a versioned symlink; on change, it loads the new model side-by-side and atomically switches. No pod restart.
```python
_lock = threading.RLock()
_model = None
_loaded_version = None

def loader_thread():
    global _model, _loaded_version
    while True:
        try:
            actual = os.readlink("/models/active.symlink")
            if actual != _loaded_version:
                new = joblib.load(actual)
                with _lock:
                    _model, _loaded_version = new, actual
                log.info(f"swapped to model {actual}")
        except Exception as e:
            log.error(f"loader: {e}")
        time.sleep(10)

threading.Thread(target=loader_thread, daemon=True).start()

@app.post("/predict")
def predict(req):
    with _lock:
        m = _model
    return {"prediction": m.predict(req.features), "version": _loaded_version}
```
**Metric:** updating the symlink shows the new version in responses within ~10s, with zero pod restarts.

### Step-by-step (Part B)

1. **Implement each pattern (15-30 min each)** — pick the iris-api or a slightly heavier model; implement and test each independently.
2. **Measure the impact (30 min)** — capture the relevant metric before/after for each pattern (first-request latency; pod time-to-Ready; GPU utilization + throughput; model-swap downtime).
3. **Production refinement (30 min)** — pick one pattern and add structured logging, metrics for the pattern itself, and error handling for failure cases.
4. **Write up (30 min)** — `PATTERNS.md` documenting each pattern: problem, when to use, when NOT to use, implementation summary, measured benefit.

### Validation (Part B)

- [ ] Pattern 1: first-request latency within 2× steady-state.
- [ ] Pattern 2: initContainer-based pod Ready in < 30s vs. original time.
- [ ] Pattern 3: GPU util ≥ 50% under sustained load (vs. single-digit per-request).
- [ ] Pattern 4: swap demonstrated without pod restart.

### Common pitfalls (Part B)

- **Warmup runs on the wrong event loop** — async lifespan + sync model means warmup runs once on the wrong thread; use `asyncio.to_thread`.
- **initContainer doesn't share memory** — use an `emptyDir` volume, not container layers, for the shared model.
- **Sidecar batcher is a single point of failure** — when it crashes, all requests fail; use a replica plus a circuit-breaker fallback to per-request mode.
- **Hot swap isn't atomic** — a reader can see partial state if the lock isn't held during reads too; always lock both write and read paths.

---

## Deliverables

1. Part A: written diagnosis for each of the 3 bugs (root cause, evidence, fix), plus a `DEBUGGING_CHEATSHEET.md`.
2. Part B: working implementation of each of the 4 patterns, a measurement table for each pattern's impact, a `PATTERNS.md` reference doc, and a k8s manifest set for each pattern.

## Stretch goals

- Add a `containers-down.sh` chaos script that randomly applies one of the Part A bugs on schedule; time how long it takes you to detect, diagnose, and fix.
- Practice with **ephemeral debug containers** in Kubernetes (`kubectl debug`), and set up continuous container security scanning with falco.
- Add **graceful drain** to Pattern 3's batcher: on SIGTERM, complete all in-flight batches before exit.
- Combine Patterns 2 and 4: initContainer downloads N models, hot swap chooses between them via config.
- Add **GPU memory pinning** with `torch.cuda.set_per_process_memory_fraction(0.9)` so the model gets predictable VRAM.
