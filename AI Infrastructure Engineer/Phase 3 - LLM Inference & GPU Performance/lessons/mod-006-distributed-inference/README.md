# Module 06: Distributed Inference

**Duration**: ~30h
**Prerequisites**: mod-001-005

Tensor parallel, pipeline parallel, expert parallel, multi-replica auto-
scaling. Make models that don't fit on one GPU serve fast across many.

| Exercise | Answers question | Needs |
|---|---|---|
| [02 Parallelism math](exercises/exercise-02-parallelism-math/) | 22, 23 | **CPU — auto-graded, do it first** |
| [04 Prefix-aware routing](exercises/exercise-04-prefix-aware-routing/) | 8 | **CPU — auto-graded** (optional real-vLLM part) |
| [01 Tensor parallelism](exercises/exercise-01-tensor-parallel.md) | 23 | 2 GPUs |
| [03 Autoscaling headroom](exercises/exercise-03-custom-hpa-metric.md) | 20 | Python; cluster optional |
| [05 Cold-start breakdown](exercises/exercise-05-cold-start-mitigation.md) | 20 | 1 GPU |

Autoscaling *mechanics* (KEDA/HPA on queue depth, spot nodes) were built in
Phase 2 project 105; this module covers the performance side.

```
mod-006/
├── README.md, lecture-notes/ (3), exercises/ (5), quiz.md, quiz-answers.md, resources.md
```

---

## Advanced layer: source digest

[source-digest.md](source-digest.md) summarizes what the linked sources (papers, docs, worklogs) actually say, as steps and numbers, with anything not read marked. Read it after the lecture notes.
