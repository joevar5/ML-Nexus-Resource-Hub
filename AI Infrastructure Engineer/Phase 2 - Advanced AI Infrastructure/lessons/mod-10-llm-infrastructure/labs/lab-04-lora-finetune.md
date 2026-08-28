# Lab 04: LoRA Fine-Tune a Small Model

**Duration:** 90 min  **Prerequisites:** GPU with ≥16GB VRAM; pytorch + transformers + peft

## Objective
Fine-tune a 1-3B parameter LLM with LoRA on a small instruction dataset. Compare base vs fine-tuned outputs.

## Steps

### 1. Install
```bash
pip install 'torch>=2.3' 'transformers>=4.42' 'peft>=0.11' 'datasets>=2.20' 'accelerate>=0.31' 'bitsandbytes>=0.43'
```

### 2. Choose base + dataset
- Base: `microsoft/Phi-3-mini-4k-instruct` (3.8B; fits on 16GB with QLoRA)
- Dataset: `HuggingFaceH4/no_robots` or small custom JSONL

### 3. Training script
```python
# finetune.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer       # pip install trl

BASE = "microsoft/Phi-3-mini-4k-instruct"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["qkv_proj","o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()        # << 1% of total

ds = load_dataset("HuggingFaceH4/no_robots", split="train[:500]")

def fmt(ex):
    return {"text": f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['messages'][1]['content']}<|end|>"}

ds = ds.map(fmt)

args = TrainingArguments(
    output_dir="./lora-phi3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False, bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)
trainer = SFTTrainer(model=model, train_dataset=ds, tokenizer=tok, args=args,
                     dataset_text_field="text", max_seq_length=512)
trainer.train()
model.save_pretrained("./lora-phi3-adapter")
```

### 4. Run
```bash
python finetune.py
```
~30-45 min on a single A10/A40/L4/3090.

### 5. Compare base vs fine-tuned
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

BASE = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
ft   = PeftModel.from_pretrained(base, "./lora-phi3-adapter")

def gen(model, prompt):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=200)
    return tok.decode(out[0], skip_special_tokens=True)

prompt = "Explain why distributed systems are hard, in 3 paragraphs."
print("== base ==\n", gen(base, prompt))
print("\n== fine-tuned ==\n", gen(ft, prompt))
```

### 6. Merge adapter for serving
```python
merged = ft.merge_and_unload()
merged.save_pretrained("./phi3-merged")
```
Now serve `./phi3-merged` via vLLM (lab 01) — no PEFT runtime needed.

## Validation
- [ ] Training reports < 1% trainable parameters.
- [ ] Loss decreases over training steps.
- [ ] Fine-tuned outputs differ qualitatively from base (style shift toward training data).
- [ ] Merged model loads and serves successfully.

## Cleanup
```bash
rm -rf lora-phi3 lora-phi3-adapter phi3-merged
```

## Troubleshooting
- **`Cannot import bitsandbytes`** — Linux + CUDA only; macOS users use full-precision LoRA (no 4-bit).
- **OOM during training** — Lower `per_device_train_batch_size`; raise `gradient_accumulation_steps` for same effective batch.
- **`target_modules` not found** — Different architectures use different module names; print `model` to discover them.
