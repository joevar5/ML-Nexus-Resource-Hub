# Lecture 03: LoRA + Adapter Patterns

## What LoRA does

Instead of fine-tuning all 7B params of a model, train a low-rank update:

```
W_new = W_base + (A · B)
```

where A is `[d × r]` and B is `[r × d]`, and r << d (typically 16-64).

For a 7B model with r=16 you train ~8M params (LoRA on q and v projections only)
to ~40M (all linear layers) instead of 7B: 0.1–0.6% of the model.

## Why it matters operationally

1. **Cheap fine-tuning**: 1 GPU instead of 8; hours instead of days
2. **Multi-tenant serving**: one base model + many small adapters; hot-swap at inference time
3. **Cheap experimentation**: try 50 fine-tunes for the cost of 1 full fine-tune

## QLoRA

QLoRA = quantize the base model to 4-bit + LoRA on top. Pushes 13B fine-
tuning to a single 24GB GPU.

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                              quantization_config=bnb, device_map="auto")
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=16))
```

## Serving with hot-swap

vLLM serves base model + N LoRA adapters; clients pick by name:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-lora \
  --lora-modules sql=adapters/sql legal=adapters/legal medical=adapters/medical \
  --max-loras 4 --max-lora-rank 64
```

Requests with `"model": "sql"` use the SQL adapter; with `"model": "legal"`
use the legal adapter. Single base model fits in memory.

## Companion

[engineer-solutions/mod-110 ex-05 (finetuning-at-scale)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure/exercise-05-finetuning-at-scale) — LoRA + QLoRA + DDP examples.

---

## Deep dive: how it works (core logic)

The Wafer list does not teach LoRA training; what it adds is how adapters are *served*. Core logic:

### LoRA in one formula
Freeze the pretrained weight `W` (d×k). Learn a low-rank update: `W' = W + (α/r)·B·A` with `A` (r×k, random init) and `B` (d×r, **zero init**, so training starts exactly at the base model). Trainable parameters drop from d·k to r·(d+k): for d=k=4096, r=16 that is ~131K instead of 16.8M per matrix (~0.8%). QLoRA adds a 4-bit quantized frozen base.

### Two ways to serve
1. **Merge:** compute W' once and serve as a normal model. Zero overhead, but one model per adapter.
2. **Unmerged, many adapters on one base:** for each request, `y = xW + (α/r)(xA)B`. The base term is shared; the tiny low-rank term differs per request. A batch can contain requests for **different adapters**, handled by batched gather-style kernels over adapter weights, so one GPU serves hundreds of fine-tunes. Engines such as [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://arxiv.org/html/2312.07104) implement this.

### Where it gets hard
- **Memory:** adapters are small but numerous; they compete with the KV cache for HBM, so engines keep a hot set on the GPU and page others from CPU. [Jenga](https://arxiv.org/abs/2503.18292) shows how KV allocation gets more complicated with mixed layer types.
- **Fairness among tenants:** one busy tenant can starve others. [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588) proposes serving the client with the **lowest weighted-token counter** (a virtual token counter that charges input and output tokens), giving max-min fairness even though request sizes are unknown in advance.
- **Routing:** send requests to replicas that already have the adapter loaded (see Mod 6 load balancing).

### Read the originals

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://arxiv.org/html/2312.07104)
- [Jenga](https://arxiv.org/abs/2503.18292)
- [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
