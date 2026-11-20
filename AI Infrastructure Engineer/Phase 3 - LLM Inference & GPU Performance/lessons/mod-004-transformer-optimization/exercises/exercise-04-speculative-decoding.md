# Ex 04: Speculative Decoding — When It Helps, When It Hurts

**Goal:** measure speculation at low and high concurrency and check the result
against the formula you implemented in Ex 06. Question 10.
**Time:** ~3 h. **Needs:** one GPU with ≥ 24 GB (L4/A10G/A100), vLLM.

## Setup

Target `Qwen/Qwen2.5-7B-Instruct`, draft `Qwen/Qwen2.5-0.5B-Instruct`
(same tokenizer). Run three servers:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct                                   # baseline
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-config '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4}'
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'
```

(Flag names change between vLLM versions — check `vllm serve --help`.)

## Measure

Two prompt sets: code completion (high acceptance) and open-ended stories
(lower acceptance), 256 output tokens, temperature 0. For each server and set,
run concurrency 1 and 32. Record per-request latency, output tokens/s and the
acceptance rate from vLLM's spec-decode metrics.

## Pass criteria

- A 3 servers × 2 sets × 2 concurrencies table.
- Greedy outputs are identical to the baseline for ≥ 20 spot-checked prompts
  (speculation must not change the output).
- Your measured speedup at concurrency 1 is within ~30% of
  `spec_speedup(α, 4, c)` from Ex 06, using the measured α and an estimated c
  (draft step time ÷ target step time). Explain any gap.
- One paragraph on why the gain shrinks or turns negative at concurrency 32.
