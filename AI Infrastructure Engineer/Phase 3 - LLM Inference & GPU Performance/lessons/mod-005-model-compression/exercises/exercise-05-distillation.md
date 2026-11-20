# Ex 05: Distill a Small Model for One Task

**Goal:** see what distillation buys for a narrow task: a model 3× smaller and
faster that keeps most of the teacher's task quality.
**Time:** ~5 h. **Needs:** one GPU with ≥ 24 GB.

## Setup

- Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `Qwen/Qwen2.5-0.5B-Instruct`
  (same tokenizer, so token-level logit distillation works).
- Task: one narrow format, e.g. the text-to-SQL data from Phase 2 project 102,
  or GSM8K short answers. Hold out 500 examples for evaluation.

## Steps

1. Baselines: teacher and student task accuracy on the held-out set.
2. Generate teacher outputs (and keep the teacher's top-k logits per token,
   k = 20) on 5–20K training prompts.
3. Train the student with
   `loss = α · CE(student, teacher_text) + (1 − α) · T² · KL(softmax(t/T) ‖ softmax(s/T))`,
   T = 2, α = 0.5, 1–2 epochs, LoRA or full fine-tune.
4. Ablation: same run with α = 1 (plain fine-tuning on teacher text, no logits).
5. Measure tokens/s of teacher vs student in vLLM at concurrency 16.

## Report (`DISTILLATION_REPORT.md`)

Dataset, loss, hyperparameters, and:

| model | params | task accuracy | tok/s |
|---|---|---|---|
| teacher | 1.5B | | |
| student (before) | 0.5B | | |
| student (CE only) | 0.5B | | |
| student (CE + KL) | 0.5B | | |

## Pass criteria

- The distilled student closes ≥ 50% of the gap between student-before and teacher.
- The CE-only vs CE+KL comparison is reported (either outcome is fine — explain it).
