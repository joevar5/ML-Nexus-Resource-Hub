# Project 105: GPU Cluster and Autoscaling

Run LLM serving on Kubernetes with GPU scheduling and autoscaling that follows real load.

**What's new vs Phase 1:** Kubernetes GPU scheduling, KEDA/HPA on queue depth, spot instances
**Prep:** Mods 04, 07, and 09

Needs a cloud GPU or a local cluster. Plan on a few dollars of rented GPU time.

## Lessons that prepare you

- [Kubernetes GPU Scheduling](../../lessons/mod-04-kubernetes/02-gpu-scheduling.md)
- [GPU Memory Management](../../lessons/mod-07-gpu-computing/07-gpu-memory-management.md)
- [Building AI Infrastructure with Terraform](../../lessons/mod-09-infrastructure-as-code/03-building-ai-infrastructure-terraform.md)
- [Lesson 02: vLLM Deployment](../../lessons/mod-10-llm-infrastructure/02-vllm-deployment.md)
- [Lesson 11: Autoscaling on Queue Depth (KEDA)](../../lessons/mod-10-llm-infrastructure/11-autoscaling-on-queue-depth-keda.md)

## Steps

- [ ] Deploy vLLM on Kubernetes with GPU requests
- [ ] Expose queue depth and latency metrics
- [ ] Autoscale on queue depth with KEDA or HPA
- [ ] Test spot-instance interruption handling
- [ ] Report cost at idle vs peak

## Result

Scaling timeline under a load test, with cost numbers. *(fill in when done)*

## Links

| What | Link |
| :--- | :--- |
| Kaggle notebook | *(add)* |
| Hugging Face model or dataset | *(add)* |
| W&B project | *(add)* |

Put Kaggle notebooks in `notebooks/` and keep real code in `src/`, with tests in `tests/`, as the earlier projects do.
