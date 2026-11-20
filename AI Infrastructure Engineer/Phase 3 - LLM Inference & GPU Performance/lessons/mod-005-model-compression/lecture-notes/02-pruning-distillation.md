# Lecture 02: Pruning + Distillation

## Pruning

Set unimportant weights to zero. Sparse computation skips them.

| Type | Effect | Hardware acceleration |
|---|---|---|
| Unstructured | individual weights zero | minimal speedup on GPU |
| Structured (2:4) | every 4 weights have at most 2 nonzero | up to 2× tensor-core math on A100+; ~1.1–1.5× end to end in practice |
| Block sparse | larger contiguous blocks zero | requires custom kernels |

In practice, 2:4 sparsity is the only widely-deployable variant: NVIDIA
Sparse Tensor Cores natively support it.

```python
from torch.sparse import to_sparse_semi_structured
# W must already follow the 2:4 pattern (prune, then fine-tune to recover accuracy)
linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight.half().cuda()))
```

Sparsity stacks with quantization: vs FP16, INT8 halves the weights and 2:4
roughly halves them again (minus ~12% index metadata), so ≈ 3.5× smaller.

## Distillation

A small "student" model is trained to mimic a large "teacher" model. Student
inherits the teacher's behavior with smaller params + faster inference.

Loss = α × CE(student_logits, ground_truth) + β × KL(student_softmax, teacher_softmax)

Known examples:
- BERT-base → DistilBERT: 40% smaller, 60% faster, keeps ~97% of GLUE performance
- Llama 3.2 1B/3B: pruned from Llama 3.1 8B, then trained with 8B/70B logits as targets
- Task-specific distillation (one domain, one format) retains far more than general-purpose distillation

The catch: distillation requires the teacher to actually have learned the
task well. Don't distill from a teacher that itself is bad.

## When pruning vs quantization vs distillation

| Goal | Method |
|---|---|
| Smaller inference memory | quantization (AWQ for large models) |
| Faster compute | pruning (only on sparse-tensor-core HW) |
| Smaller weights AND different architecture | distillation |
| All of the above | combine |

---

## Deep dive: how it works (core logic)

The Wafer list has no pruning/distillation section. The closest relevant idea is **sparsity inside attention**, which is where long-context cost lives. Core logic of each:

### Why sparse attention
Full attention costs O(N²). At 100K+ tokens, prefill attention dominates. But for any query, most keys get near-zero weight. If you can find the important ones cheaply, you only compute those.

### MInference ([MInference 1.0](https://arxiv.org/abs/2407.02490)): sparsity without retraining
Heads in long-context models fall into a few stable **patterns**: *A-shape* (initial tokens + local window), *vertical-slash* (a few key columns plus diagonals), and *block-sparse*. Offline, each head is assigned its pattern. Online, a cheap estimate (using the last queries against the keys) finds the actual vertical/slash/block indices for that prompt, then a **sparse FlashAttention kernel** computes only those. Works on existing models; speeds up prefill, not decode.

### Native Sparse Attention ([Native Sparse Attention](https://arxiv.org/abs/2502.11089)): train it in
Sparsity is part of the architecture, trained end to end, with three branches whose outputs are combined by learned gates:
1. **Compressed** tokens (coarse summaries of blocks) for global context,
2. **Selected** blocks (top-k blocks chosen using the compressed scores) for detail,
3. **Sliding window** for local context.
Block-wise selection is *hardware-aligned*: it reads contiguous blocks so GPUs stay efficient, and it works in both training and decoding.

### DeepSeek sparse attention ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556))
A small, fast **indexer** scores every previous token for each query; attention then runs over only the **top-k** highest-scoring KV entries (a fixed k; see the source digest for what was verified). Cost per token becomes ~O(k) instead of O(N), so long contexts get cheaper.

### Takeaway
All three trade exactness for speed, so the same rule applies as with quantization: report quality on long-context tasks next to every speed-up.

### Read the originals

- [MInference 1.0](https://arxiv.org/abs/2407.02490)
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
