# Exercise 03: vLLM Deep Dive

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** [Lab 01](../../labs/lab-01-vllm-deployment.md) and [Lesson 02: vLLM Deployment](../../02-vllm-deployment.md)
**Hardware:** 1 GPU for most steps, 2 GPUs for Step 1

## Goal

vLLM works well out of the box, but its settings decide how much traffic one GPU can carry. In this exercise you turn on one feature at a time, measure it, and write down when each one is worth using.

You will try tensor parallelism, prefix caching, structured output, speculative decoding, LoRA adapters, and request priorities.

## Why it matters

A GPU is your biggest bill. An engineer who knows these settings can often serve the same traffic on fewer GPUs, or the same GPUs with lower latency. The catch is that every feature has a cost, so you have to measure before you trust it.

## Before you start

Pick one model for the whole exercise, such as `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`, and one benchmark prompt set. Changing either halfway makes your numbers useless.

Check your vLLM version with `vllm --version`. Flags change between releases, so if a flag below is rejected, run `vllm serve --help` and use the current name.

Measure a plain baseline first. Every later step is compared against it.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
vllm bench serve --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --num-prompts 200 --max-concurrency 32
```

Write down throughput (tokens/s), time to first token (TTFT, p50 and p95), and GPU memory.

## Steps

### Step 1: Tensor parallelism across 2 GPUs (30 min)

Tensor parallelism splits each layer across GPUs. It exists so that a model too big for one GPU can run at all. It is not a free speed-up.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
```

Compare against your baseline. Then check how your GPUs are connected with `nvidia-smi topo -m`. If you see `NV#` between them, they share a fast link. If you see `PHB` or `SYS`, traffic goes over PCIe and the overhead will be much higher.

Write down: did it get faster, and why or why not?

### Step 2: Prefix caching (30 min)

If many requests start with the same text (a system prompt, or the same retrieved documents), vLLM can reuse the work it already did for that text. Recent vLLM versions turn this on by default. Check the startup log, and use `--enable-prefix-caching` or `--no-enable-prefix-caching` to force either way.

Build two test sets of 200 requests:
- **Shared:** the same 1,500-token system prompt, a different question each time.
- **Unique:** a different 1,500-token prompt each time.

Run both with caching on and off. You should see TTFT drop only on the shared set. If it drops on both, your "unique" prompts are not unique.

### Step 3: Structured output (45 min)

Instead of asking the model for JSON and hoping, you give the server a schema and it only allows tokens that keep the output valid.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Extract the person: Alice is 30 years old."}],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "person", "schema": schema}},
)
print(r.choices[0].message.content)
```

`response_format` is the standard OpenAI-style way, and it works across vLLM versions. vLLM also has its own structured-output options that have been renamed over time, so check the docs for your version if you need regex or grammar constraints.

Run 200 requests two ways: free-form with `json.loads()` on the reply, and schema-constrained. Count how many parse and how many match your schema.

### Step 4: Speculative decoding (45 min)

A small "draft" model guesses several tokens ahead and the big model checks them in one pass. When the guesses are right you get several tokens for the price of one.

The flags for this have changed across vLLM releases. In current versions you pass one JSON setting:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 5}'
```

If that is rejected, check `vllm serve --help` and the speculative decoding page in the vLLM docs for your version. Methods that need no separate draft model, such as n-gram and EAGLE, may also be available.

Measure at low concurrency (1–4) and high concurrency (32+). Speculative decoding usually helps most when the GPU is not already busy.

### Step 5: LoRA adapters on one base model (30 min)

One base model in memory can serve many fine-tuned adapters. This is the serving side of the training you did in [Lesson 05](../../05-llm-fine-tuning-infrastructure.md).

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules sql=/adapters/sql chat=/adapters/chat --max-loras 2
```

Call each adapter by its name:

```python
client.chat.completions.create(model="sql", messages=[...])
client.chat.completions.create(model="chat", messages=[...])
```

Check that the two adapters give clearly different answers to the same prompt, and note how GPU memory changes compared with running two separate servers.

### Step 6: Request priorities (15 min)

You can serve interactive users before background jobs. Priority scheduling must be turned on at startup:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --scheduling-policy priority
```

```python
client.chat.completions.create(..., extra_body={"priority": 0})   # lower number = served first
```

In vLLM a lower number means higher priority, which surprises many people. Test it: flood the server with priority-10 requests, then send a few priority-0 requests and compare their latency.

### Step 7: Benchmark matrix (30 min)

Fill in one row per setup, using the same prompts and the same concurrency every time.

| Setup | Tokens/s | TTFT p50 | TTFT p95 | GPU memory | Notes |
|---|---|---|---|---|---|
| Baseline | | | | | |
| 2 GPUs (TP=2) | | | | | |
| Prefix caching (shared prompts) | | | | | |
| Structured output | | | | | |
| Speculative decoding | | | | | |
| LoRA (2 adapters) | | | | | |

For cost per million tokens, use: `GPU price per hour ÷ (tokens per second × 3600) × 1,000,000`.

## Deliverables

1. The exact command and config for each setup.
2. `BENCHMARKS.md` with the filled-in matrix.
3. `RECOMMENDATIONS.md`: for each feature, which workloads should use it and which should not.

## Check yourself

- [ ] Every feature has a before-and-after number, not just "it worked".
- [ ] Prefix caching helps on shared prompts and makes no difference on unique ones.
- [ ] Schema-constrained output parses 100% of the time. Say how many free-form replies failed.
- [ ] You can explain why tensor parallelism did or did not help on your hardware.

## Common pitfalls

- **Tensor parallelism over PCIe.** Without NVLink, the GPUs spend a lot of time waiting on each other. Use it to fit a bigger model, not to speed up one that already fits.
- **A weak draft model.** If the big model rejects most guesses, speculative decoding adds work and saves nothing. Look at the acceptance rate in the server metrics.
- **Unfair comparisons.** Changing the prompts, concurrency, or output length between runs makes the comparison meaningless. Change one thing at a time.
- **Forgetting warm-up.** The first requests are slow. Send a few before you start timing.
- **Copying flags from old blog posts.** vLLM moves fast. When a flag fails, trust `--help` over the internet.
