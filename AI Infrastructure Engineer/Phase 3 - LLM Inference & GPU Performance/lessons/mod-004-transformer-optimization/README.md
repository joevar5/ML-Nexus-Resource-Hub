# Module 04: Transformer Optimization

**Duration**: ~40h
**Prerequisites**: mod-002, mod-003

FlashAttention, KV-cache optimization, fused kernels, paged attention, speculative decoding, with the focus on **inference**.

| Exercise | Answers question | GPU? |
|---|---|---|
| [01 FlashAttention](exercises/exercise-01-flash-attention.md) | 5 | yes |
| [02 torch.compile](exercises/exercise-02-torch-compile.md) | 9 | yes |
| [03 vLLM prefix caching](exercises/exercise-03-vllm-prefix-caching.md) | 8 | yes (16 GB) |
| [04 Speculative decoding](exercises/exercise-04-speculative-decoding.md) | 10 | yes (24 GB) |
| [05 Triton RMSNorm](exercises/exercise-05-write-triton-kernel.md) | 15 | yes |
| [06 Inference arithmetic](exercises/exercise-06-inference-arithmetic/) | 3, 10, 12, 25 | **no — auto-graded, do it first** |

```
mod-004/
├── README.md
├── lecture-notes/ (3)
├── exercises/     (6; ex 06 is CPU-only with check.py)
├── quiz.md + quiz-answers.md
└── resources.md
```

---

## Advanced layer: source digest

[source-digest.md](source-digest.md) summarizes what the linked sources (papers, docs, worklogs) actually say, as steps and numbers, with anything not read marked. Read it after the lecture notes.
