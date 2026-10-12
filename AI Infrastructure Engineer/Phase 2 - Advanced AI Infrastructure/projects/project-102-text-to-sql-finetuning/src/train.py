"""Phase 3: LoRA fine-tune on the train split, logged to W&B. Needs torch, transformers, peft and a GPU (Colab)."""
import argparse
import json
import logging
import os
import random
from pathlib import Path

from src.prompt import build_prompt

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.jsonl"


def build_example(tok, row: dict, max_len: int) -> dict | None:
    """Prompt + SQL + stop token. Only the SQL part is learned (prompt labels are -100). None if too long to fit whole."""
    prompt = tok(build_prompt(row["context"], row["question"]), add_special_tokens=False)["input_ids"]
    answer = tok(row["answer"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(prompt) + len(answer) > max_len:  # a cut-off answer would teach the model to stop mid-query
        return None
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def load_examples(tok, n_train: int, n_val: int, max_len: int, seed: int, path: Path = TRAIN):
    """Sample from the train split only. Validation rows come from it too, so the 200 test rows stay untouched."""
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    random.Random(seed).shuffle(rows)
    exs = [e for e in (build_example(tok, r, max_len) for r in rows[: n_train + n_val]) if e]
    log.info("kept %d of %d rows (dropped %d over %d tokens)", len(exs), n_train + n_val, n_train + n_val - len(exs), max_len)
    return exs[n_val:], exs[:n_val]


def main(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, val = load_examples(tok, a.n_train, a.n_val, a.max_len, a.seed)

    if a.qlora:  # frozen base stored in 4 bits (NF4): about a quarter of the memory, slightly slower
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)  # fp32 weights + fp16 autocast: T4 has no bf16
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    os.environ.setdefault("WANDB_PROJECT", "text-to-sql-finetuning")
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum, per_device_eval_batch_size=a.batch,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, fp16=True, seed=a.seed,
        logging_steps=10, eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="no",
        report_to="wandb" if a.wandb else "none", run_name=a.run_name,
    )
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out)  # adapter only (a few MB), not the base model
    tok.save_pretrained(a.out)
    print("adapter saved to", a.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "adapters" / "run1"))
    ap.add_argument("--n-train", type=int, default=5000, help="rows sampled from the train split (all 76k will not fit a free-tier session)")
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps; use 50 for a smoke run")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (needs bitsandbytes); use for bigger models or if memory runs out")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    main(ap.parse_args())
