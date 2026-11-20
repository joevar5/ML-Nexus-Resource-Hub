# Module 05: Model Compression

**Duration**: ~35h
**Prerequisites**: mod-002, mod-003

Quantization (AWQ, GPTQ, int8), pruning, distillation, LoRA. Make models
smaller without losing too much accuracy.

| Exercise | Answers question | GPU? |
|---|---|---|
| [02 Quantization from scratch](exercises/exercise-02-quantization-from-scratch/) | 11 | **no — auto-graded, do it first** |
| [01 Quantize a real LLM three ways](exercises/exercise-01-awq-quantize.md) | 11, 24 | yes (24 GB) |
| [03 2:4 sparsity](exercises/exercise-03-2-4-sparsity.md) | — | Ampere+ |
| [04 Multi-LoRA serving](exercises/exercise-04-lora-finetune.md) | 21 | yes (24 GB) |
| [05 Distillation](exercises/exercise-05-distillation.md) | 24 | yes (24 GB) |

LoRA *training* is covered in Phase 2 project 102; here the focus is serving.

```
mod-005/
├── README.md
├── lecture-notes/ (3)
├── exercises/     (5)
├── quiz.md + quiz-answers.md
└── resources.md
```

---

## Advanced layer: source digest

[source-digest.md](source-digest.md) summarizes what the linked sources (papers, docs, worklogs) actually say, as steps and numbers, with anything not read marked. Read it after the lecture notes.
