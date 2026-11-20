# Module 07: Production Deployment

**Duration**: ~25h
**Prerequisites**: mod-001-006

Containerization + serving frameworks + deployment patterns specific to ML.
Bridges performance work to production reality.

## Lectures

| # | Lecture | Answers question |
|---|---|---|
| 01 | [Serving frameworks](lecture-notes/01-serving-frameworks.md) (Orca, PagedAttention, Sarathi, SGLang) | 4, 6, 7, 18 |
| 02 | [Deployment patterns](lecture-notes/02-deployment-patterns.md) | 19 |
| 03 | [Cost optimization](lecture-notes/03-cost-optimization.md) | 24 |
| 04 | [Request lifecycle, streaming, cancellation](lecture-notes/04-request-lifecycle-streaming.md) | **1, 2** |
| 05 | [Observability and tail latency](lecture-notes/05-observability-tail-latency.md) | **16, 17** |
| 06 | [Capacity planning: a worked design for 1M users](lecture-notes/06-capacity-planning.md) | **24, 25** |

## Exercises

| Exercise | Answers question | Needs |
|---|---|---|
| [06 Capacity and cost model](exercises/exercise-06-capacity-planning/) | 24, 25 | **CPU — auto-graded** |
| [07 SSE relay with cancellation](exercises/exercise-07-sse-streaming-cancellation/) | 2 | **CPU — auto-graded** |
| [01 Pick a framework](exercises/exercise-01-pick-a-framework.md) | 18 | GPU |
| [02 Canary for a quantized model](exercises/exercise-02-canary-deployment.md) | 19 | GPU + k8s |
| [03 Batch inference on spot](exercises/exercise-03-spot-with-resilience.md) | 24 | spot GPU |
| [04 Multi-tier routing](exercises/exercise-04-multi-tier-routing.md) | 24 | GPU |
| [05 End-to-end service](exercises/exercise-05-end-to-end-deploy.md) | 16, 17, 19, 20 | GPU + k8s |

```
mod-007/
├── README.md, lecture-notes/ (6), exercises/ (7), quiz.md, quiz-answers.md, resources.md
```

---

## Advanced layer: source digest

[source-digest.md](source-digest.md) summarizes what the linked sources (papers, docs, worklogs) actually say, as steps and numbers, with anything not read marked. Read it after the lecture notes.
