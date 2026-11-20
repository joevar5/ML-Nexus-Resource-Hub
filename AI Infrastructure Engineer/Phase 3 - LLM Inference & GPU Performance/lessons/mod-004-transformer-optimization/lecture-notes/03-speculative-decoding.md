# Lecture 03: Speculative Decoding + Other Inference Tricks

## Speculative decoding

A small draft model proposes N tokens. The large target model verifies them
in one forward pass. If all N are correct, you got N tokens for the cost of
~1. If only K are correct, you got K tokens; throw out the rest.

Typical speedup: 2-3× on long completions for a well-matched draft model.

vLLM supports it out of the box:
```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 \
  --speculative-config '{"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```
(Older vLLM releases used `--speculative-model` / `--num-speculative-tokens`; check
`vllm serve --help` for your version. EAGLE and n-gram drafters use the same flag
with `"method": "eagle"` or `"method": "ngram"`.)

## Continuous batching

Naive serving: pad all requests in a batch to the longest; small requests waste compute.

Continuous batching: each step adds new requests + drops finished ones, regardless of length.

Speedup: often 5–20× throughput for workloads with highly variable request lengths
(Anyscale measured up to 23× vs static batching in 2023).

vLLM, TGI, TensorRT-LLM all do this.

## Prefix caching

Multiple requests share a system prompt. Cache its KV state; reuse across
requests. Speedup: 2-4× for system-prompt-heavy workloads (chat, agents, RAG).

## Compounding

The gains overlap rather than multiply: prefix caching and chunked prefill both
cut prefill work; speculative decoding helps at low batch but continuous batching
pushes you to high batch, where speculation helps less (Mod 4 digest §7). Measure
each step on *your* workload. For scale: vLLM's launch post reported up to 24×
throughput over HuggingFace Transformers on the same model.

## Companion

[engineer-solutions/mod-110 (all 14 LLM exercises)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure).

---

## Deep dive: how it works (core logic)

### The algorithm ([Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html))
Let `q` be the cheap draft model's distribution and `p` the target's.
1. Draft proposes γ tokens: x₁…x_γ, sampled from q.
2. Target runs **one forward pass** over all γ positions, giving p at each (cost ≈ one normal decode step because decode is memory-bound).
3. For each token in order: **accept with probability min(1, p(x)/q(x))**.
4. At the first rejection, **resample from the residual** distribution `norm(max(0, p − q))` and stop. If all γ accepted, sample one bonus token from p.

Why it's *exact*: the accept/resample rule is rejection sampling whose output distribution is mathematically equal to sampling from p alone. You change speed, not quality. ([Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) gives the same scheme with its analysis.)

### How much faster? The formula
With per-token acceptance rate α (assume independent) and draft length γ, expected tokens produced per target pass:
```
E[tokens] = (1 − α^(γ+1)) / (1 − α)
```
Example α=0.8, γ=4: (1 − 0.8⁵)/0.2 = **3.36 tokens per step** instead of 1.
Include draft cost, with c = draft step cost ÷ target step cost (say 0.05):
```
speedup ≈ E[tokens] / (γ·c + 1) = 3.36 / 1.2 ≈ 2.8×
```
Raise γ and you get diminishing returns while draft cost grows linearly. If α is low (0.4): E ≈ 1.65 and the gain mostly vanishes.

### When it flips to a loss
It converts spare compute into speed. At large batch sizes the GPU is already compute-bound, so verifying γ tokens per sequence *costs* real compute and rejected tokens are wasted. Rule: great at low concurrency/interactive, reduce γ or disable at high batch.

### Drafters without a separate model
- **[Medusa](https://arxiv.org/html/2401.10774):** extra prediction heads on the target predict tokens t+1, t+2... in parallel; candidates are arranged as a **tree** and verified together using a tree attention mask.
- **[EAGLE](https://proceedings.mlr.press/v235/li24bt.html):** drafts at the **feature** (hidden state) level, which is more predictable than tokens, then maps to tokens. Higher acceptance.
- **[EAGLE-3](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html):** drafts from fused features of several target layers and trains on its own multi-step outputs, pushing acceptance and speedup higher.

### Read the originals

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/html/2401.10774)
- [EAGLE](https://proceedings.mlr.press/v235/li24bt.html)
- [EAGLE-3](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
