# Exercise 06: LLM Inference Optimization

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** [Exercise 03: vLLM Deep Dive](../exercise-03-vllm-deep-dive/README.md) and [Lesson 06: LLM Serving Optimization](../../06-llm-serving-optimization.md)
**Hardware:** 1 GPU with 24 GB or more. Step 6 needs an NVIDIA GPU.

## Goal

Start from the slowest reasonable setup and add one optimization at a time: a serving engine with continuous batching, prefix caching, weight quantization, and a smaller KV cache. Measure each step and record how much quality you give up for the speed.

## Why it matters

There is no single "make it fast" switch. Each optimization trades something: memory, quality, flexibility, or engineering time. Measuring them one by one tells you which are worth it for your workload, and in what order to apply them.

## Before you start

Pick a 7B–8B model and keep it the same throughout. Build two small test sets and use them in every step:

- **Speed set:** 200 prompts of realistic length, some sharing a long system prompt.
- **Quality set:** 100 questions with known answers (or a rubric), plus the scoring script. Run it on every optimized setup, not just at the end.

Measure these every time: requests per second and tokens per second at concurrency 32, TTFT (p50 and p95), inter-token latency, and peak GPU memory.

Do not copy the example numbers from any guide, including old versions of this one. Your hardware and prompts will give different results.

## Steps

### Step 1: Baseline with plain Transformers (15 min)

Process one request at a time with `model.generate`. This is deliberately naive. It shows what you are improving on.

```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

t0 = time.perf_counter()
for p in prompts[:32]:
    inputs = tok(p, return_tensors="pt").to(model.device)
    model.generate(**inputs, max_new_tokens=100)
print(f"{32 / (time.perf_counter() - t0):.2f} req/s")
```

### Step 2: A serving engine with continuous batching (15 min)

Serve the same model with vLLM (or SGLang, which is a good alternative to try). Continuous batching lets new requests join a running batch instead of waiting for it to finish, which keeps the GPU busy. Drive it with the same prompts at concurrency 32.

```bash
vllm serve MODEL
vllm bench serve --model MODEL --num-prompts 200 --max-concurrency 32
```

### Step 3: Prefix caching (10 min)

Prefix caching is on by default in recent vLLM, so the useful experiment is to turn it off and on and compare. Use the speed set where prompts share a long prefix. If your prompts share nothing, expect no gain.

### Step 4: Weight quantization (45 min)

Quantizing weights shrinks the model, which saves memory and often speeds up decoding, because decoding is limited by memory bandwidth.

Try at least two formats:

- **FP8** on GPUs that support it (H100, L40S, and newer). It is often the easiest option, with small quality loss. In vLLM you can try `--quantization fp8` to quantize on load.
- **4-bit AWQ or GPTQ** on any GPU. Use a pre-quantized checkpoint from Hugging Face, or quantize one yourself with [llm-compressor](https://github.com/vllm-project/llm-compressor). The old `AutoAWQ` project is no longer maintained, so prefer llm-compressor.

```bash
vllm serve ./my-model-awq          # vLLM detects the quantization from the model config
```

After each one, run your quality set. 4-bit is more likely than 8-bit to hurt on tasks like math, code, and long reasoning.

### Step 5: KV cache compression (15 min)

The KV cache holds the attention state for every running request. Storing it in 8-bit leaves room for more concurrent requests.

```bash
vllm serve MODEL --kv-cache-dtype fp8
```

Check the effect on the number of requests that fit, and on quality, particularly with long prompts.

### Step 6: TensorRT-LLM, optional (60 min)

TensorRT-LLM can be faster on NVIDIA hardware, at the price of a build step and less flexibility. It has changed a lot, so follow the current NVIDIA quick-start rather than older guides. The newer versions can serve a model with a single `trtllm-serve` command. Engines are built for a specific GPU type, so do not move them between GPU models.

If you skip this step, note that in your report and say why.

### Step 7: Load test (30 min)

Use `vllm bench serve` or a load tool such as `oha` or `k6`. Test at concurrency 1, 8, 32, and 128. Plot throughput against p95 latency. The point where latency climbs steeply is your capacity limit, and that matters more than a single average number.

### Step 8: Report (15 min)

| Setup | Req/s @ c32 | Tokens/s | TTFT p95 | GPU memory | Quality score | Speed-up |
|---|---|---|---|---|---|---|
| HF Transformers, batch 1 | | | | | | 1x |
| vLLM | | | | | | |
| + prefix caching | | | | | | |
| + FP8 weights | | | | | | |
| + 4-bit AWQ (alternative) | | | | | | |
| + FP8 KV cache | | | | | | |
| TensorRT-LLM (optional) | | | | | | |

## Deliverables

1. The commands and configs for each setup.
2. `RESULTS.md` with your measured table, including the quality column.
3. `RECOMMENDATIONS.md`: which optimizations you would ship for a chat product, and which for a batch-processing job. They will not be the same.

## Check yourself

- [ ] Each step shows a measured change, even if the change is zero.
- [ ] The cumulative speed-up is clearly above the baseline. If it is below 5x, explain why.
- [ ] Quality was scored at every step, not once at the end.
- [ ] You found the concurrency level where p95 latency starts to climb.

## Common pitfalls

- **Skipping the quality check.** A faster model that gives worse answers is a regression. Score every setup.
- **Chasing throughput and ignoring TTFT.** Bigger batches raise throughput and also raise latency. Tune to what your users feel.
- **Unrealistic prompts.** Prompt and output lengths change the results a lot. Use lengths that match your real traffic.
- **Forgetting the engine is GPU-specific.** TensorRT-LLM engines and some quantization kernels only work on the hardware they were built for.
- **Comparing a cold run with a warm one.** Warm up first, and run each test more than once.
