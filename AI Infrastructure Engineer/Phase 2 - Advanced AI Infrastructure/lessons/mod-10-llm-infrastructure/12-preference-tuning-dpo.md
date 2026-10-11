# Lesson 12: Preference Tuning With DPO

*Optional.* [Lesson 05](./05-llm-fine-tuning-infrastructure.md) taught supervised fine-tuning (SFT): show the model correct answers and it learns to copy them. SFT can't directly teach the model what *not* to do. After SFT, a model still makes mistakes, and you often have examples of both a good and a bad answer for the same input. **Direct Preference Optimization (DPO)** trains on those pairs so the model prefers the good answer.

It is the improvement stage in [Project 102](../../projects/project-102-text-to-sql-finetuning/README.md): after you find the model's failures ([Lesson 10](./10-text-to-sql-and-execution-eval.md)), train on them.

**Prerequisites:** [Lesson 05](./05-llm-fine-tuning-infrastructure.md), [Lesson 10](./10-text-to-sql-and-execution-eval.md).

### Contents

1. [What DPO Is All About](#what-dpo-is-all-about)
2. [Core Logic](#core-logic)
3. [Step-by-Step: Run DPO After SFT](#step-by-step-run-dpo-after-sft)
4. [Practical Exercise](#practical-exercise)
5. [Key Takeaways](#key-takeaways)
6. [Additional Resources](#additional-resources)

---

## What DPO Is All About

SFT answers "what should the output look like?" DPO answers "which of these two outputs is better?" That fits problems where mistakes have a clear pattern and you can label them automatically.

Text-to-SQL is a good case, because execution gives you the label for free: a query that returns the right result is **chosen**, and a query that errors or returns the wrong rows is **rejected**.

| | SFT | DPO |
|---|---|---|
| Data | Input and one correct output | Input, a better output, and a worse one |
| Teaches | What to produce | What to prefer over what |
| Run | First | After SFT, never instead of it |
| Cost | One model in memory | Needs a reference model too (with LoRA, no second copy) |

DPO won't fix a model that hasn't learned the task. Do SFT first, measure, then use DPO on what is still wrong.

---

## Core Logic

### The Idea

For each input, DPO compares how likely the trained model makes the chosen answer versus the rejected one, relative to a frozen **reference model** (the SFT model before DPO). It adjusts the weights to widen the gap in favor of the chosen answer, without drifting far from the reference.

### The Knob: Beta

`beta` controls how strongly the model is held near the reference. A common starting value is `0.1`. A larger beta keeps the model closer to the reference and learns more cautiously. A smaller beta lets it move further and risks degrading outputs. Treat the value as something to tune against your eval set.

### Where the Pairs Come From

| Source | How |
|---|---|
| **Execution (SQL)** | Run the model on training questions. A correct result is chosen, a wrong or erroring result is rejected. |
| **Reference answers** | Chosen is the dataset's correct output, rejected is the model's own wrong output |
| **Human labels** | Slower, and used where no automatic check exists |

Pairs built from the model's **own** mistakes are the most useful, because they target exactly what it gets wrong.

### Risks

| Risk | What happens | Guard |
|---|---|---|
| **Overfitting** | Perfect on the pairs, worse on new questions | Always evaluate on the same held-out set as before |
| **Length bias** | The model learns "longer is better" | Check output lengths before and after |
| **Leakage** | Pairs built from test questions inflate your score | Build pairs from the training split only |
| **Regression** | Gains on one failure type, losses elsewhere | Compare accuracy by failure type, not just the total |

---

## Step-by-Step: Run DPO After SFT

### Step 1: Start From the SFT Model

Have the SFT adapter from Lesson 05 and its held-out score. That score is the number DPO has to beat.

### Step 2: Build Preference Pairs From Failures

```python
def build_pairs(model_generate, rows, is_correct):
    """rows: dicts with prompt, schema, reference. Returns TRL-style preference rows."""
    pairs = []
    for r in rows:
        pred = model_generate(r["prompt"])
        ok, _ = is_correct(r["schema"], pred, r["reference"])
        if not ok:
            pairs.append({"prompt": r["prompt"], "chosen": r["reference"], "rejected": pred})
    return pairs
```

Use the training split only. Expect a few hundred to a few thousand pairs. If you have very few failures, SFT is already doing well and DPO may add little.

### Step 3: Train With TRL

The `trl` library has a `DPOTrainer`. Argument names have changed between versions (for example `tokenizer` became `processing_class`), so check the docs for your installed version:

```python
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

args = DPOConfig(
    output_dir="dpo-out",
    beta=0.1,
    learning_rate=5e-6,          # much lower than SFT
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    report_to="wandb",
)

trainer = DPOTrainer(
    model=sft_model,             # SFT model with its LoRA adapter, loaded for training
    args=args,
    train_dataset=Dataset.from_list(pairs),
    processing_class=tokenizer,
)
trainer.train()
```

When the model uses a LoRA adapter, TRL can compute the reference log-probabilities by switching the adapter off, so you don't hold two full models in memory. Check that this holds for your library version. Use a **low learning rate** and about one epoch, since DPO moves the model quickly.

### Step 4: Evaluate the Same Way

Score the DPO model on the same held-out set with the same prompt and decoding as before:

| | Baseline | After SFT | After DPO |
|---|---|---|---|
| Execution accuracy | | | |
| Invalid SQL rate | | | |
| Average output length | | | |

Keep DPO only if it beats SFT on held-out accuracy without a length or regression problem. Often it won't, and the honest result is "not worth the extra complexity".

### Step 5: Save and Record

Save the adapter, log the run with the beta and learning rate, and record the comparison in your README.

---

## Practical Exercise

1. Run your SFT model on 500 training questions and build pairs from the failures
2. Train DPO with `beta=0.1` for one epoch
3. Score on the held-out set and fill in the three-column table
4. Try `beta=0.3` and compare
5. Write two sentences: did DPO help, and why do you think so?

*No GPU? Do steps 1 and 2 and inspect the pairs. Look at ten of them and decide whether they are real mistakes the model makes or noise.*

---

## Key Takeaways

1. SFT teaches what to produce, and DPO teaches what to prefer
2. Always run SFT first: DPO refines a model that already knows the task
3. Execution results give free, automatic labels for text-to-SQL pairs
4. Build pairs from the model's own failures on the training split only
5. Use a low learning rate and one epoch, and tune `beta` against held-out accuracy
6. Check for overfitting, length bias, and regressions on specific failure types
7. Keep DPO only if the held-out numbers justify it

---

## Additional Resources

- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [TRL DPO Trainer documentation](https://huggingface.co/docs/trl/dpo_trainer)
- [Lesson 05: LLM Fine-Tuning Infrastructure](./05-llm-fine-tuning-infrastructure.md)
- [Lesson 10: Text-to-SQL and Evaluation by Execution](./10-text-to-sql-and-execution-eval.md)

---

**Next Lesson:** none. Continue with [Project 102](../../projects/project-102-text-to-sql-finetuning/README.md).
