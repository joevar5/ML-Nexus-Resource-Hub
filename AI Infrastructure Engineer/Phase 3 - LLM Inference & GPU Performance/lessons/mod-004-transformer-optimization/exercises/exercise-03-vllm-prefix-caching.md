# Ex 03: vLLM Prefix Caching — TTFT Under a Shared System Prompt

**Goal:** measure what prefix caching does to TTFT and throughput when many
requests share a long prefix. Question 8.
**Time:** ~3 h. **Needs:** NVIDIA GPU with ≥ 16 GB, `pip install vllm`.
Phase 2 project 101 already got vLLM running; this exercise isolates one feature.

## Steps

1. Serve a small model twice, once with and once without prefix caching
   (recent vLLM enables it by default):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching      # run A
vllm serve Qwen/Qwen2.5-1.5B-Instruct --no-enable-prefix-caching   # run B
```

2. Workload: 1,000 requests, each = the **same 2,000-token system prompt** +
   a different 50-token question, `max_tokens=64`. Send at a fixed rate
   (e.g. 8 req/s) with an async client, streaming, and record TTFT per request
   (time to first streamed chunk) and total output tokens/s.
   `vllm bench serve` with a custom dataset also works.
3. Repeat with a **unique** prefix per request (cache cannot help) as a control.
4. Read `vllm:gpu_prefix_cache_hit_rate` (or the hit-rate log line) from `/metrics`.

## Report

| run | prefix | hit rate | TTFT p50 | TTFT p99 | output tok/s |
|---|---|---|---|---|---|

## Pass criteria

- Shared-prefix TTFT p50 improves by ≥ 3× with caching; unique-prefix control
  shows no gain (proves you measured the cache, not noise).
- One paragraph: why does TTFT gain more than throughput? Relate it to prefill
  being compute-bound (Mod 1 Lecture 1.5) and how many prefill tokens were skipped.
