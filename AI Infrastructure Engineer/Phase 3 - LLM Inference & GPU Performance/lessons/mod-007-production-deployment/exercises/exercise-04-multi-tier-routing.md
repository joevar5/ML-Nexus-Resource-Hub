# Ex 04: Route Easy Requests to a Small Model

**Goal:** the routing lever from Lecture 06 (question 24): send easy traffic to
a cheap model, escalate the rest, and measure cost *and* quality.
**Time:** ~4 h. **Needs:** one GPU with ≥ 24 GB (or two vLLM servers on two GPUs);
any OpenAI-compatible API also works for the large tier.

## Setup

- Small tier: `Qwen2.5-1.5B-Instruct`; large tier: `Qwen2.5-7B-Instruct`
  (or a hosted large model). Serve both with vLLM.
- Eval set: 300 prompts with checkable answers, mixed difficulty (e.g. 150
  GSM8K, 150 short factual questions), each with a reference answer.

## Steps

1. Baselines: all-small and all-large accuracy, and cost per request for each
   tier (from your measured tokens/s and $/GPU-hour, Lecture 05 formula).
2. Router v1: a heuristic (prompt length, keywords like "prove", "step by
   step", code fences) picks the tier.
3. Router v2: try small first; escalate to large when the small model's
   **confidence** is low (mean token log-prob from `logprobs=True` below a
   threshold) — tune the threshold on half the set, report on the other half.
4. Plot accuracy vs cost for: all-small, all-large, v1, v2 at three thresholds.

## Pass criteria

- The accuracy–cost plot with all points.
- A router that keeps accuracy within 2 points of all-large at ≤ 60% of its cost
  (or an explanation of why your data doesn't allow it).
- Latency impact of escalation (v2 pays for two calls on hard prompts) stated.
