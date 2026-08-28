# Exercise 05: Fine-Tuning at Scale (LoRA + QLoRA + Distributed)

**Duration:** 4 hours
**Difficulty:** Advanced
**Prerequisites:** Lab 04 (LoRA basics); multi-GPU setup

## Objective

Scale a fine-tuning workflow from single-GPU LoRA to multi-GPU QLoRA on a 13B model. Track experiments, save adapters, serve via vLLM with LoRA hot-swap.

## Requirements

1. Train LoRA on 1 GPU (baseline).
2. Train QLoRA (4-bit) on 1 GPU for a larger model.
3. Multi-GPU DDP for an even larger model.
4. Adapter checkpoints managed via MLflow.
5. Serve trained adapter via vLLM `--enable-lora`.

## Step-by-step

### Step 1 — Single-GPU LoRA baseline (45 min)
Per lab 04. Pick a 3B-7B model (Phi-3-mini or Llama-3.2-3B). Use ~500 instruction examples.

### Step 2 — QLoRA for 13B (60 min)
```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-13B-Instruct",
                                              quantization_config=bnb, device_map="auto")
```
13B model in 4-bit fits in ~10 GB VRAM (vs ~26 GB fp16). Now trainable on a single 24GB consumer GPU.

### Step 3 — DDP across 2-4 GPUs (60 min)
```bash
accelerate config    # configure for multi-GPU
accelerate launch finetune.py
```
Use HuggingFace `accelerate`:
```python
from accelerate import Accelerator
accelerator = Accelerator()
model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
for batch in train_loader:
    with accelerator.accumulate(model):
        loss = model(**batch).loss
        accelerator.backward(loss)
        opt.step(); opt.zero_grad()
```
Scales near-linearly across GPUs.

### Step 4 — Track in MLflow (15 min)
```python
mlflow.log_params({"base_model": "Llama-3-13B", "lora_r": 16, "lora_alpha": 32, "batch_size": 2})
mlflow.log_metric("train_loss", loss, step=global_step)
mlflow.log_artifact("./adapter/")    # at end of training
```

### Step 5 — Adapter promotion (15 min)
Register the adapter directory in MLflow Model Registry. Promote to Staging based on eval set perplexity.

### Step 6 — Serve via vLLM (30 min)
```bash
vllm serve meta-llama/Llama-3-13B-Instruct \
  --enable-lora \
  --lora-modules my-adapter=/path/to/adapter \
  --max-loras 4
```
```python
client.chat.completions.create(model="my-adapter", messages=[...])
```

## Deliverables

1. 3 training runs (LoRA, QLoRA, DDP).
2. MLflow tracking for all.
3. Adapter served + producing responses different from base.

## Validation

- [ ] QLoRA fits a 13B model in single GPU.
- [ ] Multi-GPU achieves ~Nx speedup on N GPUs.
- [ ] Served adapter has measurable behavior shift from base.

## Common pitfalls

- **Adapter target_modules wrong** — Different architectures use different module names; print model to discover.
- **GradScaler with bf16** — bf16 doesn't need GradScaler; fp16 does.
- **Saving full model instead of adapter** — `model.save_pretrained()` saves merged weights; use `peft_model.save_pretrained()` for just adapter.
