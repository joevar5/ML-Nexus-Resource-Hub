# Ex 05: End-to-End — Double the Throughput of a Generation Script

**Goal:** run the full loop (define metric → locate → classify → change one
thing → re-measure) on a realistic serving workload.
**Time:** ~5 h. **Needs:** NVIDIA GPU; your tools from Ex 01–04.

## Starting point (deliberately slow)

Batch of 32 prompts of mixed length (50–1,000 tokens), `max_new_tokens=256`,
HF `generate` in FP32, eager attention, prompts processed **one at a time**.
Metric: **output tokens/s for the whole batch** (also record p95 per-request latency).

## Rules

- Change **one thing per iteration**; re-profile; log it.
- Allowed levers: dtype (bf16), SDPA/FlashAttention-2 attention, batching with
  left padding, static cache + `torch.compile`, sorting prompts by length,
  then (stretch) the same workload on vLLM for comparison.
- Output must stay identical under greedy decoding for the dtype you settle on
  (spot-check 5 prompts).

## Deliverable: `OPTIMIZATION_LOG.md`

| iter | change | tokens/s | p95 latency | bottleneck after (nsys/ncu evidence) |
|---|---|---|---|---|

## Pass criteria

- **≥ 2× tokens/s** over the starting point without vLLM (most people get 5–20×).
- Every row names the evidence that motivated it.
- Final paragraph: what is the bottleneck *now*, and what would you try next?
  (Compare with the vLLM number if you ran it — the gap is continuous batching
  and PagedAttention, Module 7.)
