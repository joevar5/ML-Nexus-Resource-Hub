# Ex 04: Prefix-Aware Routing (CPU simulation, auto-graded)

**Goal:** design a router that keeps prefix-cache hits high *without*
overloading one replica — the trade-off behind question 8.
**Time:** ~2 h. **Needs:** Python 3 only. Read Lecture 03's deep dive.

## Files

- `sim.py` — replicas with LRU prefix caches and a Zipf workload of shared
  system prompts (200 prefixes, 8 replicas, room for 30 prefixes each). Read it; don't edit it.
- `policies.py` — the four routing policies you write.
- `check.py` — runs all four and grades them.

## What to do

1. Implement `round_robin`, `least_loaded`, `prefix_hash`, `prefix_aware`.
2. `python check.py` until it prints `PASS`. The bars:
   round-robin and least-loaded must balance load; prefix-hash must reach
   ≥ 95% hit rate; prefix-aware must keep ≥ 90% hits **and** imbalance ≤ 1.25
   **and** do ≤ half of round-robin's total work.

## Then answer (in `ANSWERS.md`)

1. Your four results (hit rate, imbalance, total work). Why does pure hashing
   overload one replica? (Look at how popular prefix 0 is under Zipf.)
2. How does your `prefix_aware` score trade hits against load? Show what
   happens to both numbers when you halve and double your load weight.
3. Change `zipf_s` to 0.6 (flatter popularity) and to 1.5 (hotter): which
   policy's ranking changes, and why?
4. What does this simulator leave out that a production router (llm-d,
   Dynamo, Gateway API Inference Extension) must handle? Name three things.

## Optional: the real thing

Run two vLLM replicas with prefix caching on one or two GPUs, put a ~60-line
FastAPI router in front that implements your `prefix_aware` policy (hash the
first 1,000 characters of the prompt as the prefix id; read load from each
replica's `vllm:num_requests_running`), and compare TTFT p50/p99 and
`vllm:gpu_prefix_cache_hit_rate` against round-robin.
