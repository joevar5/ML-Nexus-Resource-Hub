# Rubric — High-Performance LLM Inference System

Reviewers score each dimension 1-5. **Pass** requires all hard
performance gates from `requirements.md` Section 3 AND >= 3/5 in
every dimension. **Distinction** requires >= 4/5 in at least 6 of 8.

Hard gates fail the project regardless of rubric score:

- PR-1..PR-14 missed.
- `make verify` fails to reproduce within 5%.
- KV cache leak that grows monotonically over a 30-minute run.
- vLLM baseline not run on identical hardware with the same workload.
- Any unhandled exception in the request hot path under load.

## 1. Scoring scale

| Level | Meaning |
|-------|---------|
| 1 | Missing or fundamentally broken. |
| 2 | Partial; works at low load; collapses under stress. |
| 3 | Meets requirements; defensible. (Pass bar.) |
| 4 | Exceeds requirements with measured ablations. |
| 5 | Reference-quality; reviewer would deploy this to production. |

## 2. Dimensions

### D1. Goodput capacity (weight: highest)

Highest Poisson rate with >= 95% of requests in SLO, as a fraction of vLLM's
on the same GPU and workload (Llama-3 8B BF16).

| Level | Goodput capacity vs vLLM |
|-------|---------------------------------------------|
| 1 | < 20% |
| 2 | 20–50% |
| 3 | >= 50% (PR-1 met) |
| 4 | >= 70% |
| 5 | >= 85%, with a profile-backed explanation of the remaining gap |

### D2. Latency (TTFT + TPOT)

P50 and P99 at 70% of your own goodput capacity.

| Level | Evidence |
|-------|----------|
| 1 | TTFT P99 > 500 ms or TPOT P99 > 200 ms. |
| 2 | TTFT P99 200-500 ms. |
| 3 | TTFT P50 <= 100 ms, P99 <= 500 ms; TPOT P50 <= 25 ms, P99 <= 60 ms (PR-2..PR-5). |
| 4 | TTFT P99 <= 250 ms; TPOT P99 <= 45 ms. |
| 5 | TTFT P99 <= 150 ms; CUDA Graph + speculative both contributing measurably with documented numbers. |

### D3. Goodput / SLO compliance

% of requests within SLO at 70% of your goodput capacity.

| Level | Goodput |
|-------|---------|
| 1 | < 70% |
| 2 | 70-90% |
| 3 | >= 95% (PR-10) |
| 4 | >= 98% with admission policy ablation documented. |
| 5 | >= 99% with cost-aware admission and per-tenant SLO classes. |

### D4. KV cache management

Fragmentation, prefix cache hit rate, no leaks under sustained load.

| Level | Evidence |
|-------|----------|
| 1 | OOM under sustained load at your goodput capacity. |
| 2 | Fragmentation > 10%; prefix cache absent. |
| 3 | Fragmentation <= 5% (PR-9); prefix cache hit rate >= 40% (PR-7). |
| 4 | Block leak test passes (30-min run, free count returns to baseline at idle). COW correctness test passes. |
| 5 | Fragmentation <= 2%; prefix cache hit rate >= 60% on shared-prompt workload; documented per-workload tuning of `block_size`. |

### D5. Code quality

Engine is line-by-line defensible.

| Level | Evidence |
|-------|----------|
| 1 | Thin wrapper around vLLM. |
| 2 | Scheduler is a 500-line god-method. |
| 3 | Module layout matches `architecture.md`; `mypy --strict`, `ruff`, `black` pass. |
| 4 | New scheduler can be plugged in via protocol with < 100 LOC of changes. ADRs document non-obvious choices. |
| 5 | Pull request reviewer in 30 minutes can walk every box of the request lifecycle and explain it. |

### D6. Observability

Metrics, dashboard, alerts.

| Level | Evidence |
|-------|----------|
| 1 | No Prometheus metrics. |
| 2 | Metrics exist but cardinality blow-up risk; dashboard incomplete. |
| 3 | All FR-11 metrics exported; Grafana dashboard with all required panels; alerts for SLO violations. |
| 4 | OTLP tracing; per-tenant counters; alert runbook in `docs/runbook.md`. |
| 5 | Recording rules for cost-per-request; per-tenant SLO dashboards; cost-attribution panel. |

### D7. Engineering judgment

Scheduler design, admission policy, baseline comparison.

| Level | Evidence |
|-------|----------|
| 1 | Defaults uncritically used; vLLM baseline missing. |
| 2 | One scheduler config; no sweep; vLLM comparison incomplete. |
| 3 | Scheduler tuning knobs documented; baseline comparison written. |
| 4 | Chunked prefill ablation showing TTFT improvement; speculative decoding accept rate sweep; "where we win, where we lose vs vLLM" documented. |
| 5 | Disaggregated prefill/decode or tensor-parallel ablation with numbers; per-workload tuning guide. |

### D8. Production hardening

Container, cancellation, graceful shutdown, security.

| Level | Evidence |
|-------|----------|
| 1 | Runs in a `python -m` from a notebook. |
| 2 | Dockerfile exists but image > 8 GB; runs as root. |
| 3 | Multi-stage Dockerfile, < 4 GB, non-root, SIGTERM drain working, healthcheck. |
| 4 | Read-only root FS in K8s manifest; PodDisruptionBudget; resource limits; pre-stop drain. |
| 5 | Chaos test: kill -9 to client mid-stream, kill -SIGTERM during peak load, NVML returns "disconnected" — all handled cleanly. |

## 3. Bonus rubric (distinction only)

- **B1**: Tensor parallelism across 2 GPUs with NCCL; throughput
  number reported.
- **B2**: FP8 weights via TransformerEngine with measured speedup and
  accuracy table.
- **B3**: Multi-LoRA serving: hot-swap LoRA per request without
  re-prefilling.
- **B4**: Disaggregated prefill/decode (DistServe / Splitwise style).
- **B5**: Medusa heads or lookahead decoding without a draft model.
- **B6**: OpenTelemetry traces shipped to Tempo/Jaeger.

## 4. Anti-patterns (auto-deduction)

- **Wrapping vLLM and calling it our engine**: auto-fail. The point
  is to build it.
- **Reporting numbers without locking GPU clocks**: -2 on D1 and D2.
- **`request.is_disconnected()` awaited in the hot loop**: -1 on D2
  (TPOT suffers).
- **Request-id used as a Prometheus label**: -2 on D6 (cardinality
  bomb).
- **No vLLM comparison run on the same hardware**: -2 on D7.
- **No chunked prefill (long prompts block decodes)**: -1 on D2 and
  D7.
- **Synchronous I/O on the hot path (blocking file reads, etc.)**:
  -1 on D5.
- **Container runs as root**: -1 on D8.
- **No SIGTERM drain (kills in-flight requests)**: -1 on D8.

## 5. Reviewer checklist

```
[ ] Hard gates PR-1..PR-14 met?      (else fail)
[ ] make verify reproduces in Docker? (else fail)
[ ] D1 Throughput:           _/5
[ ] D2 Latency:              _/5
[ ] D3 Goodput / SLO:        _/5
[ ] D4 KV cache management:  _/5
[ ] D5 Code quality:         _/5
[ ] D6 Observability:        _/5
[ ] D7 Engineering judgment: _/5
[ ] D8 Production hardening: _/5

Total: __/40
Pass:        all >= 3 and hard gates met
Distinction: >= 6 dimensions at 4+
```

## 6. How to defend your score

For each dimension, the reviewer expects ONE artifact + ONE sentence:

| Dimension | Artifact | Sentence |
|-----------|----------|----------|
| D1 | `reports/goodput_vs_offered.png` | "Goodput capacity <X> req/s vs vLLM's <Y> (<X/Y>%); knee at <Z> req/s." |
| D2 | `reports/latency_cdf.png` | "TTFT P99 <n> ms at 70% load; speculation cut TPOT by <n> ms at concurrency 1; CUDA Graphs cut <n> ms." |
| D3 | `reports/goodput_chart.png` | "Goodput holds at <n>% up to <rate>; admission control sheds cleanly above." |
| D4 | `reports/kv_cache_timeseries.png` | "Fragmentation steady at <n>%; prefix hit rate <n>% on shared-system workload." |
| D5 | `src/engine/scheduler.py` (line count + complexity) | "Scheduler is 180 LOC; protocol allows alternate impl." |
| D6 | Grafana dashboard screenshot + `infra/alerts.yaml` | "Dashboard captures TTFT, TPOT, KV, GPU; 4 SLO alerts wired." |
| D7 | `reports/baseline_comparison.md` | "<n>% of vLLM's goodput; we lose on <phase> because <profiled reason>." |
| D8 | `infra/helm/values.yaml` + kill-test log | "Drains 84 in-flight requests in 12 s on SIGTERM; PodDisruptionBudget set." |
