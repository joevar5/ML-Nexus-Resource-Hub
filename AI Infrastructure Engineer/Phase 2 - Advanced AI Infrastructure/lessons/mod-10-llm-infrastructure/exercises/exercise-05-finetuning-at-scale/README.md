# Exercise 05: Fine-Tuning at Scale (LoRA, QLoRA, Multi-GPU)

**Duration:** 4 hours
**Difficulty:** Advanced
**Prerequisites:** [Lab 04](../../labs/lab-04-lora-finetune.md) and [Lesson 05: LLM Fine-Tuning Infrastructure](../../05-llm-fine-tuning-infrastructure.md)
**Hardware:** 1 GPU with 24 GB for Steps 1–2, 2 or more GPUs for Step 3

**Start here:** do [Project 104: LLM Fine-Tuning](../../../../projects/project-102-text-to-sql-finetuning/) first. This exercise is its Part 2.

## Goal

Take one fine-tuning job from a small model on one GPU to a bigger model on one GPU (QLoRA), then to several GPUs. Track every run, and serve the finished adapter with vLLM.

## Why it matters

Most fine-tuning starts small and then hits a wall: the model you want does not fit, or training takes too long. Each fix (4-bit loading, sharding across GPUs) has its own failure modes. Doing all three once, on the same task, shows you which one to reach for.

## Before you start

Keep the task, data, and eval set the same in every step. Use the ones from Project 104, or about 500–2,000 instruction examples of your own. Hold out an eval set and score the base model first. Without that baseline you cannot tell whether any run helped.

Pick models by size, not by name, because model names change. Look for a current open model near each size on Hugging Face:

| Step | Model size | Example families |
|---|---|---|
| 1 | 3B–8B | Llama 3.x, Qwen 2.5/3, Phi |
| 2 | 13B–14B | Qwen 2.5 14B, or the closest current model |
| 3 | 30B+ | Whatever fits your GPU count with sharding |

## Steps

### Step 1: LoRA on one GPU, the baseline (45 min)

Train a LoRA adapter on the 3B–8B model, following Lab 04. Log training time, peak GPU memory (`torch.cuda.max_memory_allocated()`), tokens per second, and the eval score.

### Step 2: QLoRA on a bigger model (60 min)

Load the 13B–14B model in 4-bit and train a LoRA adapter on top of it.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
```

4-bit weights take roughly a quarter of the memory of 16-bit ones, so a 14B model's weights drop from about 28 GB to about 8–10 GB. Training memory is higher than that, because of activations and optimizer state. Turn on gradient checkpointing and measure your real peak.

Compare with Step 1: how much slower is a step, and how much better is the eval score?

### Step 3: Train across several GPUs (60 min)

Plain data parallelism (DDP) gives every GPU a full copy of the model. It makes training faster, but it does not make a bigger model fit. To fit a bigger model, you shard the model and optimizer state across GPUs with FSDP or DeepSpeed ZeRO (see Step 6 of the lesson).

Use the Hugging Face `Trainer` and let `accelerate` handle the launch. You do not need to write the training loop by hand.

```bash
accelerate config            # choose multi-GPU, then FSDP or DeepSpeed
accelerate launch train.py
```

Run it twice and compare:

1. **DDP** on the Step 1 model. Check that throughput grows roughly with the number of GPUs.
2. **FSDP or DeepSpeed ZeRO-3** on a model that did not fit in Step 2's setup. Check per-GPU memory.

### Step 4: Track every run in MLflow (15 min)

```python
import mlflow

mlflow.set_experiment("finetune-at-scale")
with mlflow.start_run(run_name="qlora-14b"):
    mlflow.log_params({"base_model": MODEL, "method": "qlora", "lora_r": 16, "lora_alpha": 32,
                       "effective_batch": 16})
    # the Trainer can also log loss for you with report_to="mlflow"
    mlflow.log_metric("eval_exact_match", score)
    mlflow.log_artifacts("./adapter", artifact_path="adapter")
```

Add the training time, peak memory, and GPU type to each run, so you can compare cost, not just quality.

### Step 5: Register and promote an adapter (15 min)

Register the best adapter in the MLflow Model Registry. Promote it based on your eval score on the held-out set. Use aliases such as `staging` and `production`, because MLflow's older fixed "stages" are being phased out in favor of aliases.

Eval loss is a weak signal for this. Use your task metric (exact match, rubric, or judge score) and compare it with the base model.

### Step 6: Serve the adapter with vLLM (30 min)

```bash
vllm serve BASE_MODEL --enable-lora \
  --lora-modules my-adapter=/path/to/adapter --max-loras 4
```

```python
client.chat.completions.create(model="my-adapter", messages=[...])   # your adapter
client.chat.completions.create(model="BASE_MODEL", messages=[...])   # the base model
```

One server now answers as both. Send the same 50 eval prompts to each and compare. Note that the adapter must have been trained on the same base model that vLLM is serving. If you trained against a 4-bit model, check the vLLM docs on how quantized bases and LoRA work together.

## Deliverables

1. Three or more tracked runs: LoRA, QLoRA, and a multi-GPU run.
2. A results table with model size, method, GPUs, peak memory per GPU, time, cost estimate, and eval score.
3. An adapter registered in MLflow with a `staging` alias.
4. A served adapter whose answers differ from the base model in the way you trained for.
5. A short note: for a new project, which setup would you start with and why?

## Check yourself

- [ ] You have a baseline score for the untrained model.
- [ ] The QLoRA run fits on one GPU, and you know its real peak memory.
- [ ] You can explain why DDP did not help with memory but FSDP or ZeRO did.
- [ ] The served adapter beats the base model on your eval set.

## Common pitfalls

- **Wrong `target_modules`.** Module names differ between architectures. Use `target_modules="all-linear"`, or print the model to see the layer names.
- **Mixed precision mismatch.** bf16 does not need a gradient scaler, but fp16 does. On a GPU with no bf16 support (such as a T4), use fp16 everywhere.
- **Saving the wrong thing.** Saving a `PeftModel` writes only the small adapter. Merging into a full model is a separate step, and you should merge into a full-precision base, not a 4-bit one.
- **Expecting DDP to fit a bigger model.** It will not. Each GPU still holds the whole model.
- **Judging by loss.** A falling loss does not mean the model does the task better. Score the held-out set.
- **Forgetting the serving bill.** Training can cost a few dollars, but an always-on GPU to serve it costs far more. Multi-LoRA serving shares one GPU between many adapters.
