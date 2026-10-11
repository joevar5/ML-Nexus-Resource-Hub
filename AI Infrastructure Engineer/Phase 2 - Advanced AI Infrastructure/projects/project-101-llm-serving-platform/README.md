# Project 101: LLM Serving Platform

Serve an open LLM with vLLM, then make it cheaper and faster, and prove it with numbers.

**What's new vs Phase 1:** vLLM, quantization, continuous batching, goodput, cost per million tokens
**Prep:** Lessons 02 and 06 of mod-10


## Lessons that prepare you

- [Lesson 02: vLLM Deployment](../../lessons/mod-10-llm-infrastructure/02-vllm-deployment.md)
- [Lesson 06: LLM Serving Optimization](../../lessons/mod-10-llm-infrastructure/06-llm-serving-optimization.md)
- [GPU Memory Management](../../lessons/mod-07-gpu-computing/07-gpu-memory-management.md)
- [Prometheus Metrics](../../lessons/mod-08-monitoring-observability/01-prometheus-metrics.md)

## Steps

- [ ] Serve a small model with vLLM behind an API
- [ ] Benchmark a baseline (TTFT, TPOT, throughput, goodput)
- [ ] Quantize and re-measure
- [ ] Tune the scheduler and add prefix caching
- [ ] Compute cost per million tokens before and after

## Result

Before/after table: TTFT, TPOT, goodput, cost per 1M tokens. *(fill in when done)*

## Links

| What | Link |
| :--- | :--- |
| Kaggle notebook | *(add)* |
| Hugging Face model or dataset | *(add)* |
| W&B project | *(add)* |

Put Kaggle notebooks in `notebooks/` and keep real code in `src/`, with tests in `tests/`, as the earlier projects do.
