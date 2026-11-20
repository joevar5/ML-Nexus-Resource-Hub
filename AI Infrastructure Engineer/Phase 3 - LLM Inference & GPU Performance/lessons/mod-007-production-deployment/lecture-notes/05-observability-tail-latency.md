# Lecture 05: Observability and Tail Latency

Answers questions **16** (what to measure: TTFT, TPOT, queue depth, KV
utilization, prefix-cache hit rate, cost per request) and **17** (p99 is 10×
p50: diagnose it) of [the 25 questions](../../../25-questions.md).

## 1. The metrics that matter

Measure per request, export as **histograms** (not averages), and label by model
and replica.

| Metric | Definition | Why it matters | vLLM metric (names vary by version — check `/metrics`) |
|---|---|---|---|
| **TTFT** | request received → first token sent | what users feel as "it's thinking" | `vllm:time_to_first_token_seconds` |
| **TPOT / ITL** | time between output tokens | streaming smoothness | `vllm:time_per_output_token_seconds` / `vllm:inter_token_latency_seconds` |
| E2E latency | request → last token | batch/API users | `vllm:e2e_request_latency_seconds` |
| **Queue time** | received → scheduled | separates "overloaded" from "slow" | `vllm:request_queue_time_seconds` |
| Running / waiting requests | current batch size and backlog | the autoscaling signal (Mod 6) | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| **KV cache utilization** | fraction of KV blocks in use | near 100% → preemptions and queueing | `vllm:gpu_cache_usage_perc` / `vllm:kv_cache_usage_perc` |
| Preemptions | sequences evicted for lack of KV and recomputed later | a hidden tail-latency source | `vllm:num_preemptions_total` |
| **Prefix-cache hit rate** | cached prompt tokens ÷ prompt tokens | prefill saved; router quality | `vllm:gpu_prefix_cache_hit_rate` or the `prefix_cache_hits/queries` counters |
| Tokens in / out per second | throughput | capacity and cost | `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` |
| **Goodput** | requests/s meeting *both* TTFT and TPOT SLOs | the number to optimize (Lecture 02) | compute from the histograms |
| **Cost per request** | GPU-seconds used × $/GPU-second | unit economics | compute (below) |

### Cost per request, simply

A replica costs `$/hour × GPUs`. Split it across requests by the work they caused.
A good first approximation weights output tokens more than input tokens, because
decode tokens hold the GPU longer per token than prefill tokens do:

```
cost_per_request ≈ replica_$_per_s × (prompt_tokens / prefill_tok_per_s + output_tokens / decode_tok_per_s)
```

with `prefill_tok_per_s` and `decode_tok_per_s` measured on that replica at
typical load. Sum per tenant for chargeback; divide by tokens for $/1M tokens.

## 2. Diagnosing "p99 is 10× p50"

Don't guess. **Decompose, then slice.**

**Step 1 — Decompose E2E latency** for the slow requests into
queue + prefill (TTFT − queue) + decode (output_tokens × TPOT). Whichever term
is big for p99 requests but small for p50 ones is your lead.

**Step 2 — Slice** the slow requests by prompt length, output length, replica,
tenant, time of day. Use exemplars or traces (OpenTelemetry) to pull concrete
slow requests.

| What the decomposition shows | Likely cause | Confirm with | Fix |
|---|---|---|---|
| E2E tail but per-token speed normal | slow requests simply generate more tokens | E2E/output_tokens is flat | report TTFT and TPOT separately; it's not a bug |
| **Queue time** dominates | bursts above capacity; scale-up too slow | waiting requests spike with the tail; BurstGPT-like arrivals | headroom, faster cold start, admission control / early rejection (Mod 6) |
| **TTFT** tail tracks prompt length | long prompts prefill slowly | TTFT vs prompt-length scatter | chunked prefill, prefix caching, separate pool for long contexts, disaggregation (Mod 6 digest §3) |
| **TPOT spikes** for everyone at once | a long prefill stalls the decode batch (head-of-line blocking) | ITL spikes line up with big prompts arriving | chunked prefill with a token budget (Sarathi-Serve, Lecture 01) |
| Tail rises with **KV utilization** near 100% | preemption: sequences evicted and recomputed | `num_preemptions` rising | fewer max sequences, FP8 KV cache, more GPUs, shorter max context |
| Tail concentrated on **one replica** | routing imbalance, a sick GPU, a noisy neighbour | per-replica histograms | load-aware routing, drain and replace the node |
| Tail at **regular intervals** | Python GC, log flushes, metric scrapes, CPU contention | tail timestamps vs host CPU | pin CPU cores, raise GC thresholds, move logging off the hot path |
| Tail right after **scale-up / deploy** | cold replicas (graph capture, empty prefix cache) | tail by replica age | warmup before readiness, slow-start routing |
| Only some **clients** | slow readers, mobile networks | server-side TTFT fine, client-side bad | measure at the server *and* client; bounded buffers |

**Step 3 — Change one thing, re-measure the same percentile** under the same
load (replay a recorded trace rather than a synthetic uniform load).

## 3. Dashboards and alerts

- One row per SLO: TTFT p50/p99, TPOT p50/p99, goodput, error rate.
- One row per resource: running/waiting, KV utilization, preemptions, prefix hit rate, GPU util.
- One row for money: $/1M tokens, $/request by tenant, idle GPU-hours.
- Alert on **SLO burn rate** (how fast the error budget is spent), not on raw
  GPU utilization. High GPU utilization is the goal, not an incident.

## Read the originals

- [Etalon](https://arxiv.org/html/2407.07000) — why averages and TPOT hide stalls; per-token deadlines.
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) — prefill/decode interference and chunked prefill.
- [vLLM documentation](https://docs.vllm.ai/) (search "metrics") — the exact metric names for your version; or read your server's `/metrics` endpoint.
