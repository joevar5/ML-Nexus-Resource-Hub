# Lecture 06: Capacity Planning and Cost — a Worked Design

Answers questions **25** (design ChatGPT-style inference for 1M daily users:
peak QPS, GPU capacity, cost) and **24** (cut cost per million tokens by 70%
without hurting quality) of [the 25 questions](../../../25-questions.md).
Exercise 06 makes you rebuild every number here in code.

The method matters more than the numbers: **write the assumptions down, derive
peak load, estimate per-replica capacity from first principles, then replace
that estimate with a benchmark.**

## 1. Assumptions (state them first; every one is a lever)

| Assumption | Value | Where it comes from |
|---|---|---|
| Daily active users | 1,000,000 | the question |
| Requests per user per day | 5 | product analytics |
| Avg prompt tokens (system prompt + history + message) | 1,500 | logs |
| Avg output tokens | 400 | logs |
| Peak-to-average ratio | 3× | diurnal traffic curve |
| SLO | TTFT p99 ≤ 1 s, TPOT p99 ≤ 50 ms | product |
| Model | 70B dense, FP8 weights + FP8 KV | quality evals (Mod 5) |
| Hardware | H100 80 GB SXM, $2.50 per GPU-hour | quote |
| Replica | TP = 2 (70 GB of FP8 weights → 35 GB per GPU, ~40 GB per GPU left for KV) | Mod 6 |

## 2. Load

```
requests/day   = 1,000,000 × 5                 = 5,000,000
average rps    = 5,000,000 / 86,400            ≈ 57.9
peak rps       = 57.9 × 3                      ≈ 173.6
peak output    = 173.6 × 400                   ≈ 69,400 tokens/s
peak input     = 173.6 × 1,500                 ≈ 260,000 tokens/s
```

## 3. Per-replica capacity, from first principles

**Decode.** Each step reads every weight once: 35 GB per GPU ÷ 3.35 TB/s ≈ **10.4 ms**.
KV per token for a Llama-3-70B-shaped model in FP8 is 2 × 80 layers × 8 KV
heads × 128 × 1 B = 160 KiB; at batch 64 and ~1,700 tokens of context that is
~17.8 GB, split over 2 GPUs → +2.7 ms. Floor ≈ 13 ms; real kernels reach maybe
65% of bandwidth → **≈ 20 ms per step** (TPOT 20 ms, inside the 50 ms SLO).
At batch 64: 64 / 0.020 = **3,200 output tokens/s** per replica.

**Prefill.** `2 × 70e9 × 1,500` = 2.1e14 FLOPs per request. A replica has
2 × 1,979 TFLOP/s FP8 dense; at 40% MFU that is 1.58e15 FLOP/s → **0.133 s** of
replica time per request.

**Replica-seconds per request** = prefill + share of decode
= 0.133 + 400 × 0.020 / 64 = 0.133 + 0.125 = **0.258 s** → **≈ 3.9 requests/s**
per replica at full use. Plan at **80% utilization** to protect the SLO:
**≈ 3.1 requests/s per replica.**

> This estimate is how you size a pilot. Before buying, replace it with a measured
> goodput number: `vllm bench serve` replaying your real prompt/output length
> distribution, reporting the highest request rate that still meets both SLOs.

## 4. GPUs and cost

```
peak replicas      = ceil(173.6 / 3.1)             = 56   → 112 GPUs
+ cold-start headroom (Mod 6 Ex 03): ceil(56 × 1.1) = 62   → 124 GPUs at peak
average replicas   = 57.9 / 3.1                    ≈ 18.7
autoscaled, paying ~1.25× average                  ≈ 23.3 replicas ≈ 47 GPUs on average
GPU-hours per day  = 47 × 24                       ≈ 1,120
cost per day       = 1,120 × $2.50                 ≈ $2,800   (≈ $84K / month)
cost per request   = $2,800 / 5,000,000            ≈ $0.00056
cost per 1M output tokens = $2,800 / 2,000         ≈ $1.40   (input processing included)
```

Static provisioning for peak instead: 124 GPUs × 24 h × $2.50 = **$7,440/day**, 2.7×
more for the same users. That gap is the first lever below.

## 5. Cutting cost by 70% (question 24)

Cost per token ∝ 1 / (useful tokens per GPU-hour). Levers that remove *different*
waste multiply. Start from a naive baseline: BF16, static peak provisioning, no
prefix caching, every request on the 70B model.

| Lever | Cost factor | Reasoning |
|---|---|---|
| Autoscale to load instead of provisioning for peak | 1.25 / (3 × 1.1) ≈ **0.38** | pay ~1.25× average instead of 1.1× peak |
| FP8 weights + FP8 KV instead of BF16 | ≈ **0.63** (≈1.6× throughput per GPU, measure it) | half the bytes per decode step, 2× the KV capacity, FP8 tensor cores for prefill |
| Prefix caching for multi-turn history (50% of prompt tokens hit) | (0.067 + 0.125) / 0.258 ≈ **0.74** | prefill time halves; decode unchanged |
| Route the easiest 50% of requests to an 8B model (~1/6 the cost per request) | 0.5 + 0.5/6 ≈ **0.58** | needs a router and quality evals (Ex 04) |

Any two of the first three already pass 70% (0.38 × 0.63 = 0.24 → **76% cheaper**);
all four give ≈ 0.10, a 90% cut. "Without hurting quality" is the constraint
that decides which you may use:

- FP8 and routing need a **quality gate**: task evals plus a canary judged on
  goodput *and* quality (Ex 02).
- Autoscaling and prefix caching don't change outputs at all; do them first.
- Speculative decoding mostly cuts *latency* at low load; it rarely cuts cost at
  high load (Mod 4 Lecture 03).

## 6. What the design still has to say

- **Regions and failure domains:** N+1 per region; how traffic shifts if one fails.
- **Long-context tail:** a separate pool (or disaggregated prefill, Mod 6) for prompts over e.g. 16K tokens so they don't stall chat.
- **Admission control:** what happens above the planned peak — queue, degrade to the small model, or reject early with a retry hint.
- **Observability:** the Lecture 05 dashboard, with $/1M tokens as a first-class SLO.
- **Re-plan triggers:** prompt length, output length or DAU drifting 20% from the assumptions.
