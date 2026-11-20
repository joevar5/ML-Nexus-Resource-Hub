# Lecture 02: Multi-Replica Autoscaling

## Three axes

| Axis | Mechanism |
|---|---|
| HPA on CPU | not useful for GPU inference (CPU is rarely the bottleneck) |
| HPA on custom metric | scale on `gpu_queue_depth` or `tokens_per_second` |
| KEDA on Kafka lag | scale on backlogged requests in queue |

## Per-replica concurrency

A single vLLM replica handles N concurrent requests (continuous batching).
"Replicas to add" is not "requests / 1" — it's "requests / capacity_per_replica".

Measure capacity_per_replica empirically; vary based on model + GPU + sequence length.

## Cold start

Adding a replica for a 70B model (keep weights **out** of the container image):
- Provision a GPU node, if none is free: 2–10 minutes
- Pull the serving image (CUDA + vLLM, ~10 GB): 1–5 minutes on a cold node
- Download 140 GB of weights: ~2–4 minutes at 1 GB/s from object storage; seconds from a local NVMe cache
- Load weights into GPU memory: 30–90 seconds
- Warmup (CUDA graph capture, `torch.compile`): 10–60 seconds

Strategies:
- **Pre-pull image** on every node (DaemonSet)
- **Pre-warm pods** kept idle below traffic floor
- **Predictive scaling**: anticipate traffic from time-of-day patterns

## Cost of over-scaling

GPUs are expensive. Don't over-scale.
- Size headroom from your **scale-up time**: keep enough spare capacity to
  absorb the traffic growth that can arrive while a new replica starts
  (e.g. traffic can rise 20% in the 6 minutes a replica takes → keep ≥ 20%
  spare). Faster starts let you run with less headroom.
- Use spot/preemptible GPUs for batch and offline work (Phase 2 project 105)
- Scale down slowly (e.g. 30–60 min stabilization window) so you don't pay
  cold starts twice; scale down to a warm minimum during predictable troughs

---

## Deep dive: how it works (core logic)

### Why LLM autoscaling is hard
New replica = provision node (minutes) + pull a multi-GB image + load 10s–100s of GB of weights + warm up. Scaling on "GPU busy now" arrives 5–15 minutes late. The two levers are **faster starts** and **earlier signals**.

### Faster starts: ServerlessLLM ([ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu))
- **Loading-optimized checkpoint format:** laid out so weights stream sequentially into GPU memory with parallel reads, instead of generic deserialization.
- **Multi-tier locality:** keep checkpoints in the fastest tier available (GPU → host DRAM → local SSD → remote), and schedule new instances on servers that already hold the checkpoint locally.
- **Live migration:** to free a GPU for another model without dropping requests, move the running inference by sending the *tokens so far* to the new server and recomputing KV there. This is far smaller than moving the KV cache.

### Predictable latency: Clockwork ([Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati))
Insight: DNN inference on a GPU is **deterministic** in time. So make scheduling centralized and predictive: the controller knows each model's exact execution time per batch size, assigns each request to a worker/time slot, and **rejects early** requests that can't meet their deadline instead of letting everything slip. Result: tail latency is controlled by design.

### Earlier, better signals
- Scale on **queue depth / waiting requests / KV utilization**, not CPU or GPU %. Kubernetes HPA rule: `desired = ceil(current × metric / target)`.
- Keep a **warm minimum** and lower the trigger; add scheduled or predictive scaling for known peaks; scale down slowly.
- Kubernetes-native tooling: [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) (model, accelerator and KV-aware routing, so scaling and routing share signals) and [llm-d](https://github.com/llm-d/llm-d) (distributed scheduling, disaggregated serving on Kubernetes). A production reference is [Dynamo](https://github.com/ai-dynamo/dynamo).

### Read the originals

- [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu)
- [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [llm-d](https://github.com/llm-d/llm-d)
- [Dynamo](https://github.com/ai-dynamo/dynamo)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
