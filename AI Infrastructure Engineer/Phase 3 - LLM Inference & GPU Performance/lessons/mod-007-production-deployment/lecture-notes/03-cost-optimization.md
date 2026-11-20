# Lecture 03: Cost Optimization

## Cost levers

| Lever | Savings | Risk |
|---|---|---|
| Spot for batch | 60-70% | spot interruption |
| Reserved (1-3y) | 40-50% | locked into capacity |
| Right-sizing | 20-40% | requires telemetry |
| Quantization (mod-005) | 50-75% | small accuracy hit |
| Spec decoding (mod-004) | 2–3× lower latency at low concurrency; little or negative at high batch | matched draft model; tune or disable under load |
| Continuous batching (mod-004) | 5–20× throughput vs static batching | none (all modern engines do it) |
| Autoscaling to load vs provisioning for peak | peak ÷ average (often 2–3×) | cold starts (Mod 6) |
| Prefix caching (mod-004) | prefill cost × hit rate | needs cache-aware routing at scale |
| Per-model routing (cheap → expensive) | 50-70% | requires good classifier |

## Compounding

Cost scales with 1 / (useful tokens per GPU-hour), so levers combine as a product of their
throughput gains, as long as they target different waste. A worked example that reaches a
70% cut (question 24) is in [Lecture 06](06-capacity-planning.md).

## Companion

- engineer-solutions/mod-110 ex-08 (cost routing)
- engineer-solutions/mod-104 ex-15 (cluster cost optimization)
- engineer-solutions/mod-106 ex-12 (per-model cost attribution)

---

## Deep dive: how it works (core logic)

### The cost equation
```
cost per 1M tokens = (GPU $/hour) ÷ (tokens per hour ÷ 1,000,000)
```
So cost falls when tokens per GPU-hour rises *at a fixed latency target*. That last part is **goodput**, not peak throughput ([Etalon](https://arxiv.org/html/2407.07000)): a configuration that doubles throughput by blowing the TTFT SLO is not cheaper, it is broken.

### Splitting phases to raise goodput ([DistServe](https://arxiv.org/html/2401.09670))
Prefill (compute-bound, drives TTFT) and decode (memory-bound, drives TPOT) interfere on one GPU, forcing you to over-provision to meet both SLOs. DistServe runs them on **separate GPU groups** and picks the parallelism and replica counts for each independently. It optimizes *goodput per GPU under both SLOs*, showing the same hardware serving several times more SLO-compliant requests.

### Benchmark with traffic that looks like yours
- **[ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf):** generates workloads that preserve properties of real production traces (arrival patterns, length distributions, per-client behaviour), instead of fixed-length synthetic requests.
- **[BurstGPT](https://github.com/HPMLL/BurstGPT):** a public trace showing real **burstiness** (arrival gaps far more variable than Poisson). Bursts drive queueing and tail latency.
- **[MLPerf Inference](https://arxiv.org/abs/1911.02549)** and **[MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/):** standardized scenarios (e.g. a *server* scenario with Poisson arrivals and a latency bound) so results are comparable across systems.

### Levers, in the order to try them
Raise batch efficiency (continuous batching, tune max tokens per step) → prefix caching → quantize weights/KV → right-size the model per request (route easy traffic to a smaller one) → better $/bandwidth hardware → spot/reserved mix and aggressive scale-down. Re-run quality evals after every step.

### Read the originals

- [Etalon](https://arxiv.org/html/2407.07000)
- [DistServe](https://arxiv.org/html/2401.09670)
- [ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf)
- [BurstGPT](https://github.com/HPMLL/BurstGPT)
- [MLPerf Inference](https://arxiv.org/abs/1911.02549)
- [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
