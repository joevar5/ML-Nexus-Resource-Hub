# Deliverables — High-Performance LLM Inference System

A reviewer should unzip a submission into this directory, run
`make verify` from the project root, and reproduce every published
number within 5%.

## 1. Required submission inventory

```
deliverables/
  src/
    engine/                       # Scheduler, batcher, executor, sampler
    cache/                        # BlockManager, PrefixCache, COW
    kernels/                      # Paged attention wrapper
    spec/                         # Speculative decoding
    api/                          # FastAPI app + admission + rate limit
    obs/                          # Metrics + NVML sampler + logger
    main.py
  bench/
    workloads/
      sharegpt_mixed.yaml
      synthetic_short.yaml
      synthetic_long.yaml
    poisson_client.py
    locust_runner.py
    compare_baselines.py
    plot_sweep.py
  infra/
    prometheus.yaml
    alerts.yaml
    grafana/
      dashboard.json
      datasources.yaml
    helm/
      Chart.yaml
      values.yaml
      templates/
        deployment.yaml
        service.yaml
        servicemonitor.yaml
        pdb.yaml
  tests/
    unit/
    integration/
    load/
  docs/
    api.md
    scheduler.md
    runbook.md
    adr/
  reports/
    load_summary.md               # Reviewer's "30-second" summary
    load_<rate>.json              # Per-rate JSON, one per offered load
    latency_cdf.png
    throughput_vs_offered.png
    goodput_chart.png
    kv_cache_timeseries.png
    baseline_comparison.md
    sols_under_load.md
    grafana_screenshot.png
  profiles/
    nsys_e2e.nsys-rep
    ncu_hot_kernel.ncu-rep
  Dockerfile
  docker-compose.yaml
  Makefile
  pyproject.toml
  README_submission.md            # Submitter notes (hardware, deviations)
```

## 2. `reports/load_summary.md` schema

A single table the SRE reads in 30 seconds:

One row per (workload, offered rate). Fill in your own measurements; the
`<…>` cells are placeholders, not expected values.

```
| Workload         | Offered (req/s) | Completed (req/s) | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | E2E P99 | Goodput | KV Frag | Prefix Hit | SM Busy | Gate |
| sharegpt_mixed   | <0.7 × capacity> | <…>             | <…>      | <…>      | <…>      | <…>      | <…>     | <…>     | <…>     | <…>        | <…>     | <PASS/FAIL> |
| sharegpt_mixed   | <1.0 × capacity> | <…>             | <…>      | <…>      | <…>      | <…>      | <…>     | <…>     | <…>     | <…>        | <…>     | <…> |
| sharegpt_mixed   | <1.5 × capacity> | <…>             | <…>      | <…>      | <…>      | <…>      | <…>     | <…>     | <…>     | <…>        | <…>     | <DEGRADED: 429s> |
```

## 3. `reports/baseline_comparison.md` schema

Side-by-side on the canonical workload (sharegpt_mixed, 300 s per rate,
same GPU, same model, same client). Template — fill with your measurements:

```
| Stack              | Goodput capacity | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | E2E P99 | Notes |
| Ours               | <req/s>          | <…>      | <…>      | <…>      | <…>      | <…>     | |
| vLLM <version>     | <req/s>          | <…>      | <…>      | <…>      | <…>      | <…>     | |
| SGLang / TRT-LLM   | <optional>       |          |          |          |          |         | |

Ours / vLLM goodput capacity: <n>%  (gate >= 50%)

Where we lose to vLLM (each with profile evidence):
- e.g. long-prompt prefill: <n>% slower — <kernel or scheduler reason from nsys/ncu>
- e.g. small-batch decode: <n>% slower — <CUDA-graph coverage, sampler, Python overhead>

Where we tie or win (each with evidence):
- e.g. cancellation latency: <n> ms vs <n> ms
```

## 4. `reports/latency_cdf.png`

Three CDFs on the same axes (TTFT, TPOT, E2E), with vertical lines at
the SLO targets. Annotate which methods (chunked prefill, speculative,
CUDA Graphs) contributed which chunk of the latency reduction.

## 5. `reports/throughput_vs_offered.png`

X-axis: offered req/sec. Y-axis: goodput (req/sec completed within SLO).
Lines: ours, vLLM (and SGLang/TRT-LLM if run). Y=X diagonal as reference.
Annotation: the "knee" point of each.

## 6. `reports/kv_cache_timeseries.png`

A 30-minute timeseries during the load test showing:

- KV blocks free (line)
- KV blocks allocated (line)
- KV fragmentation ratio (line)
- Prefix cache hit rate (rolling, line)

The KV-free line MUST return to the baseline at idle (no leaks).

## 7. Profiles — capture command reference

```bash
# End-to-end timeline under sustained load
nsys profile -o profiles/nsys_e2e \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python -m src.main &
SERVER=$!
sleep 30
python -m bench.poisson_client --rate 500 --duration 60
kill -SIGUSR1 $SERVER

# Hottest kernel under load
ncu -o profiles/ncu_hot_kernel \
    --launch-skip 5000 --launch-count 1 --set full \
    python -m src.main bench-mode --workload sharegpt_mixed
```

## 8. Tests

```
tests/
  unit/
    test_block_manager.py
    test_prefix_cache.py
    test_cow_refcount.py
    test_scheduler_fairness.py
    test_admission.py
    test_request_state_machine.py
    test_speculative_reject_sample.py
    test_chunked_prefill.py
  integration/
    test_e2e_toy_model.py        # CI-friendly, < 60 s
    test_streaming_and_cancel.py
    test_oom_returns_429.py
    test_sigterm_drain.py
  load/
    test_load_smoke_100rps_60s.py
```

Coverage >= 80% on `src/`.

## 9. Reproducibility

```bash
# Boot the full stack (engine + prometheus + grafana + redis)
docker compose up -d

# Run the canonical load test
make load workload=sharegpt_mixed rate=30 duration=300

# Run vLLM and TRT-LLM baselines
make bench-vllm workload=sharegpt_mixed rate=30 duration=300
make bench-trt-llm workload=sharegpt_mixed rate=30 duration=300

# Generate the report bundle
make report

# Verify (re-runs canonical workload, compares to load_summary.md)
make verify
```

`make verify` MUST reproduce the canonical numbers within 5% on the
same host. On a different SKU (e.g. A100 vs H100), `make verify
SKU=a100` widens tolerance to 15% and skips H100-only gates.

## 10. Helm chart contract

```
infra/helm/
  Chart.yaml
  values.yaml          # Tunable: replicas, resources, GPU type, model path
  templates/
    deployment.yaml    # Non-root, read-only FS, resource limits
    service.yaml
    servicemonitor.yaml  # Prometheus operator scrape
    pdb.yaml             # Min available 1
    configmap.yaml       # Scheduler tunables
```

## 11. What NOT to submit

- Model weights (link in `README_submission.md`).
- Locust web UI HTML reports (the JSONL is enough).
- `__pycache__`, `.pytest_cache`, `*.egg-info`, `.venv/`.
- Raw `.nsys-rep` larger than 500 MB (filter capture range first).
- Per-request log files from the load test (only aggregated reports).
- Anything > 1 GB without justification in `README_submission.md`.

## 12. Submission checklist

- [ ] All files in Section 1 present.
- [ ] `make verify` reproduces numbers within 5% on same host.
- [ ] All PR-* hard gates met (see `requirements.md` Section 3).
- [ ] vLLM and TRT-LLM baselines run on identical hardware.
- [ ] `pytest` exit 0; coverage >= 80% on `src/`.
- [ ] Container builds; size <= 4 GB compressed; runs as non-root.
- [ ] SIGTERM drain verified (chaos test in `tests/integration/`).
- [ ] No secrets committed.
- [ ] No files > 1 GB without justification.
- [ ] `README_submission.md` declares hardware, deviations, and any
      known-not-met soft gates.
