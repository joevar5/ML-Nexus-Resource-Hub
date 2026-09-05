# Exercise 03: Build & Cache Optimization

**Duration:** 13-15 hours (splits cleanly into three sessions: Part A ~8-10h, Part B ~3h, Part C ~2.5h)
**Difficulty:** Intermediate → Advanced
**Prerequisites:** Docker, Docker BuildKit, Python 3.9+, Exercise 01 (Dockerfiles + multi-stage)

## Objective

**Part A** — Build `image-optimizer`, a CLI that parses a Dockerfile, analyzes a built image's layers, and automatically rewrites it into a multi-stage build. Run it against a real ML service Dockerfile and cut image size by 50%+.

**Part B** — Apply advanced BuildKit features (cache mounts, secrets, registry-backed remote cache) to that image. Measure cold vs warm vs CI-cache build times and reduce CI build time by ≥70%.

**Part C** — Extend the build to produce a multi-arch (`linux/amd64` + `linux/arm64`) manifest-list image, verify it serves each platform correctly, and benchmark emulated vs native arm64 build and inference performance.

## Why this matters

The same production build has three independent cost centers, and each has a standard fix. **Size**: a 2.5 GB single-stage ML image (base + gcc/make/cargo + PyTorch + unpurged apt cache) is slow to pull and expensive to store — multi-stage builds fix this by compiling in a `builder` stage and copying only runtime artifacts into a slim final stage. **Speed**: a 12-minute build × 50 PRs/day × N engineers waiting on CI is a real bottleneck — most teams never enable BuildKit's cache mounts, secret mounts, and registry-backed remote cache, all of which are one-line additions. **Portability**: AWS Graviton, GCP Tau T2A, and Apple Silicon all run arm64, and ML inference there can be 20-40% cheaper for CPU-bound models — but only if you ship a multi-arch manifest list instead of amd64-only. An engineer who can systematize all three (not just eyeball one Dockerfile) is the one whose optimizations survive across every service the team ships.

---

## Part A — Multi-stage build optimizer

### Requirements

1. **Parse Dockerfiles** into stages and instructions (handle line continuations, comments, `ARG` before `FROM`, multiple `FROM` stages).
2. **Analyze built image layers** via `docker history` — size per layer, which instruction created it, whether it exceeds a size threshold.
3. **Detect optimization opportunities**: single-stage builds that should be multi-stage, build tools present in the final image, unpurged caches (`/var/cache/apt`, `~/.cache/pip`), suboptimal layer ordering (code copied before dependencies).
4. **Generate an optimized Dockerfile**: builder stage with all build deps, slim/distroless runtime stage copying only required artifacts, combined `RUN` commands, cache cleanup in the same layer that created the cache.
5. **Measure and report** before/after image size and layer count; verify the optimized image still runs correctly.

### Step-by-step

**1. Dockerfile parser (90 min)** — Read the Dockerfile, join backslash-continued lines, and split into `Stage` objects on each `FROM` (capture `FROM ... AS name`). Classify each instruction as layer-creating (`RUN`, `COPY`, `ADD`) or metadata (`ENV`, `LABEL`, `WORKDIR`, etc.). Extract package lists from `RUN apt-get install`, `pip install`, `npm install` lines via regex.

**2. Layer analyzer (90 min)**
```bash
docker history --no-trunc --format '{{json .}}' myapp:latest
docker inspect myapp:latest --format '{{.Size}}'
docker run --rm myapp:latest du -sh /var/cache/apt /root/.cache/pip /tmp 2>/dev/null
```
Build a `Layer` list with size, creating instruction, and a `size_mb > 100` flag.

**3. Multi-stage rewrite (2-3 hours)** — Given parsed stages and runtime dependencies, generate:
```dockerfile
FROM python:3.11 AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
COPY . .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY --from=builder /build/src ./src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
CMD ["python", "src/main.py"]
```
Also implement instruction reordering: system packages → dependency manifests → dependency install → app code → build commands, so app-code changes don't bust the dependency cache layer.

**4. Build cache analysis (60 min)**
```bash
docker buildx build --progress=plain -t myapp:test . 2>&1 | grep -E 'CACHED|DONE'
```
Compute cache hit rate and flag the first cache miss — usually `COPY . .` placed before dependency installation, or an unpinned `apt-get update`.

**5. Verify and measure (60 min)** — Build both original and optimized images, compare `docker images --format "{{.Size}}"`, and run a smoke test in both (e.g. `python -c "import torch"`) to confirm the optimized image is still functional.

**6. CLI (60 min)**
```bash
image-optimizer analyze Dockerfile
image-optimizer optimize Dockerfile -o Dockerfile.optimized -r libgomp1 --report report.html
image-optimizer inspect myapp:latest
```

### Validation

- [ ] Parser correctly splits a multi-stage Dockerfile into named stages and extracts package lists.
- [ ] `analyze` flags at least 3 concrete optimization opportunities on a real Dockerfile.
- [ ] `optimize` produces a Dockerfile that builds successfully.
- [ ] Optimized image is at least 40% smaller than the original.
- [ ] Smoke test passes identically in original and optimized images.

### Common pitfalls

- **Optimizing away a needed runtime lib** — stripping build tools is safe, but some libraries (e.g. `libgomp1` for PyTorch, `libpq5` for psycopg2) are runtime dependencies of compiled wheels. Always smoke-test after optimizing, don't just check size.
- **`COPY . .` before dependency install** — the single most common cache-buster; any source change invalidates the dependency layer and forces a full reinstall.
- **Cleaning cache in a separate `RUN`** — `rm -rf /var/lib/apt/lists/*` in its own layer doesn't shrink the image; Docker layers are additive. Cleanup must be in the *same* `RUN` as the install.
- **Assuming smaller is always better** — Alpine's musl libc breaks some compiled Python wheels (notably ones needing glibc); `-slim` (Debian-based) is often the safer minimal base for ML workloads.

---

## Part B — BuildKit cache optimization

### Requirements

Build a Dockerfile + CI workflow that:

1. Uses **`# syntax=docker/dockerfile:1.6`** to enable modern BuildKit features.
2. Uses **`--mount=type=cache`** for `/root/.cache/pip` and `~/.cache/torch`.
3. Uses **`--mount=type=secret`** to inject a HuggingFace token at build time without baking it into a layer.
4. Pushes build cache to a **registry-backed cache** (GHCR or ECR cache).
5. Achieves **≥70% CI build time reduction** vs a no-cache build.

### Step-by-step

**1. Baseline (15 min)**
```bash
docker buildx build --no-cache -t base:latest .
# ~8-12 minutes on typical CI for PyTorch + transformers + ~500MB of weights
```

**2. Add cache mounts (30 min)**
```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt
COPY . .
```
Rebuild with `--cache-from`/`--cache-to local` and time the second build — should be near-instant for unchanged deps.

**3. Secret mount for HF token (30 min)**
```dockerfile
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
```
```bash
docker buildx build --secret id=hf_token,src=$HOME/.cache/hf-token -t img:latest .
```
Verify the token doesn't appear in any image layer (`docker history img:latest`).

**4. Registry-backed cache (45 min)**
```bash
docker buildx build \
  --cache-from type=registry,ref=ghcr.io/me/iris-api:cache \
  --cache-to type=registry,ref=ghcr.io/me/iris-api:cache,mode=max \
  -t ghcr.io/me/iris-api:latest \
  --push .
```
Mode `max` caches intermediate layers too (more storage, faster subsequent builds).

**5. CI workflow (30 min)**
```yaml
- uses: docker/setup-buildx-action@v3
- uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
- uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ghcr.io/${{ github.repository }}/iris-api:${{ github.sha }}
    cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/iris-api:cache
    cache-to: type=registry,ref=ghcr.io/${{ github.repository }}/iris-api:cache,mode=max
    secrets: |
      hf_token=${{ secrets.HF_TOKEN }}
```

**6. Measure (30 min)** — Run the CI workflow three times: (1) after deleting the cache image (cold), (2) with cache, no source changes (full hit), (3) with cache, only application source changes (partial hit — should reuse deps but rebuild app layers). Record numbers in `BENCHMARKS.md`.

### Validation

- [ ] Cold build (no cache image) succeeds.
- [ ] Warm build (cache + no source changes) takes < 30% of cold time.
- [ ] Partial change build (only app source modified) takes < 30% of cold time.
- [ ] `docker history` does NOT contain the HF token.

### Common pitfalls

- **Cache mount path inside container** — must match the actual cache directory the tool uses (`/root/.cache/pip` for pip, NOT `/app/.cache/pip`).
- **Cache mode `max` consumes lots of registry storage** — set a retention policy or use `mode=min` for less aggressive caching.
- **`--push` overwrites tags concurrently** — two PRs racing can overwrite each other's cache. Use a per-branch cache tag.
- **Secrets leak via `RUN echo $HF_TOKEN`** — secrets are only safe when read inside `RUN`; printing them puts them in build logs.

---

## Part C — Multi-architecture builds (amd64 + arm64)

### Requirements

1. Build the same image for `linux/amd64` AND `linux/arm64`, and push as a **manifest list** (single tag, two platform-specific images).
2. Verify with `docker manifest inspect` that both platforms are advertised.
3. Benchmark inference latency on each platform, and emulated vs native build time.
4. Extend the CI workflow to produce both per PR with a single command.

### Step-by-step

**1. Buildx setup (15 min)**
```bash
docker buildx create --name multi --driver docker-container --bootstrap --use
docker buildx ls   # confirm both platforms available
```

**2. Cross-build via emulation (30 min)**
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --cache-from type=registry,ref=ghcr.io/me/iris-api:cache \
  -t ghcr.io/me/iris-api:multi \
  --push .
```
Note the arm64 build is slow under qemu emulation (often 5-10x slower than native).

**3. Native arm64 build (30 min)** — Provision a Graviton (AWS) or Tau T2A (GCP) instance, install Docker + buildx, then:
```bash
docker buildx build --platform linux/arm64 -t arm64-only:latest --load .
```
Compare time vs emulated.

**4. Verify manifest list (15 min)**
```bash
docker manifest inspect ghcr.io/me/iris-api:multi
# Expect manifests for both linux/amd64 and linux/arm64

# On an arm64 host:
docker run --rm ghcr.io/me/iris-api:multi uname -m   # → aarch64
# On amd64:
docker run --rm ghcr.io/me/iris-api:multi uname -m   # → x86_64
```

**5. Benchmark inference (30 min)** — On each platform, run the iris-api container and a Locust load test (mod-101 lab 04 pattern). Record throughput and p95 latency; compare against equivalent x86 instance (e.g., M7g vs M7i in AWS).

**6. CI workflow (30 min)** — GitHub Actions has a `linux/arm64` runner available (paid for private repos):
```yaml
strategy:
  matrix:
    platform: [linux/amd64, linux/arm64]
runs-on: ${{ matrix.platform == 'linux/arm64' && 'ubuntu-22.04-arm' || 'ubuntu-22.04' }}
```
This avoids emulation entirely.

### Validation

- [ ] `docker manifest inspect` shows both platforms.
- [ ] `docker run --platform linux/arm64` and `linux/amd64` both succeed.
- [ ] On an arm64 host: `docker run <image> uname -m` returns aarch64.
- [ ] Native arm64 build > 3x faster than emulated.
- [ ] arm64 inference benchmark recorded for at least one CPU model.

### Common pitfalls

- **`--load` doesn't work for multi-arch** — `--platform ...` with multiple platforms produces a manifest list, which can't be loaded into the single-platform docker daemon. Use `--push`.
- **Building deps from source on arm64** — `pip install` may compile from source on the missing platform, dramatically slowing the build. Use wheels available for both archs, or compile in the builder stage with `--platform`.
- **arm64 images smaller but slower for pure PyTorch ML** — CUDA isn't available on arm64 (yet). Use arm64 only for CPU inference; keep amd64 for GPU.
- **Manifest list mismatch with image tag** — don't push to the same tag for single-arch and multi-arch builds; the latest one wins and breaks the manifest list.

---

## Deliverables

1. `image_optimizer` package (`dockerfile_parser.py`, `layer_analyzer.py`, `multistage_builder.py`, `cache_analyzer.py`, `optimizer.py`, `cli.py`) plus an optimized Dockerfile + HTML report for a real multi-hundred-MB ML Dockerfile.
2. BuildKit-optimized Dockerfile using cache mounts and secret mounts, with cache pushed to a registry-backed remote cache.
3. Multi-arch manifest-list image in your registry (`linux/amd64` + `linux/arm64`).
4. `BENCHMARKS.md` covering: before/after image size and layer count; cold/warm/partial-hit CI build times; emulated vs native arm64 build time; arm64 vs amd64 inference latency.
5. A single GitHub Actions workflow that builds, caches, and pushes the multi-arch image per PR, with `docker history` verified free of leaked secrets.

## Stretch goals

- Add Google distroless base image support for the runtime stage; open a GitHub PR automatically with the optimized Dockerfile and a size-savings comment.
- Use `COPY --link` for source files to enable cross-stage cache reuse; add a `docker scout cves` step that gates the build on critical CVEs.
- Add a dedicated arm64 CI runner to skip emulation entirely, and attach SLSA provenance via `--provenance=true`.
- Detect on container startup whether NEON/SVE instructions are available and select the matching BLAS implementation.
