# Exercise 03: vLLM Deep Dive

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Lab 01 (vLLM basics)

## Objective

Master vLLM's production features: tensor parallelism, prefix caching, structured output (guided decoding), speculative decoding, LoRA hot-swap, request priorities. Benchmark each.

## Why this matters

vLLM defaults are excellent; tuning unlocks another 2-3× throughput. Engineers who know the knobs get more from each GPU dollar.

## Requirements

For each feature, demonstrate working config + benchmark before/after.

## Step-by-step

### Step 1 — Tensor parallelism across 2 GPUs (30 min)
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3-8B-Instruct \
  --tensor-parallel-size 2
```
Compare throughput vs single-GPU.

### Step 2 — Prefix caching (30 min)
```bash
--enable-prefix-caching
```
For workloads with shared prompt prefixes (system prompts, RAG context), measure latency improvement. Often 30-60% TTFT reduction.

### Step 3 — Guided decoding (45 min)
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# JSON-constrained output
r = client.chat.completions.create(
    model="...",
    messages=[{"role":"user","content":"Extract person from: Alice is 30 years old."}],
    extra_body={"guided_json": {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}},
)
```
Compare correctness + latency vs free-form generation + JSON parsing.

### Step 4 — Speculative decoding (45 min)
```bash
--speculative-model meta-llama/Llama-3.2-1B-Instruct --num-speculative-tokens 5
```
A small draft model proposes tokens; large model verifies. Often 1.5-2× speedup on text generation.

### Step 5 — LoRA hot-swap (30 min)
```bash
--enable-lora --lora-modules sql-lora=/path/to/sql-adapter chat-lora=/path/to/chat-adapter
```
```python
r = client.chat.completions.create(model="sql-lora", messages=[...])
r = client.chat.completions.create(model="chat-lora", messages=[...])
```
Multiple LoRA adapters served on one base model; no per-adapter VRAM duplication.

### Step 6 — Request priorities (15 min)
```python
extra_body={"priority": 1}    # higher = served first
```
Premium users / interactive UIs get priority over background batch.

### Step 7 — Benchmark matrix (30 min)
For each feature, record:
- Throughput (tokens/s at concurrency 32)
- TTFT (time-to-first-token p50/p95)
- Cost per 1M tokens
- Memory usage

## Deliverables

1. Working configs for each feature.
2. `BENCHMARKS.md` with comparison matrix.
3. `RECOMMENDATIONS.md`: which features to use for which workload.

## Validation

- [ ] Each feature produces measurable improvement.
- [ ] Tensor parallel scales near-linearly to 2 GPUs.
- [ ] Prefix caching reduces TTFT for shared prefixes.
- [ ] Guided JSON output is always valid.

## Common pitfalls

- **TP without high-bandwidth interconnect** — PCIe-only 2x A100 = 30% TP overhead. NVLink is required for good TP.
- **Speculative decoding with bad draft model** — Acceptance rate < 50% = no speedup.
- **LoRA loaded per request** — Hot-swap is fast but startup load is slow; pre-load common adapters.
