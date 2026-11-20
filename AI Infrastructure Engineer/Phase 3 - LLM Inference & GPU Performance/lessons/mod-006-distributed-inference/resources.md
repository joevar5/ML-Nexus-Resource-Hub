# Resources — Distributed Inference

- [vLLM docs: distributed serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [Megatron-LM (TP/PP reference impl)](https://github.com/NVIDIA/Megatron-LM)
- [GPT-NeoX (multi-node PP)](https://github.com/EleutherAI/gpt-neox)
- [NVIDIA NCCL](https://developer.nvidia.com/nccl) — interconnect library
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [KEDA](https://keda.sh/)
- engineer-solutions/mod-110 ex-03 (vLLM deep dive) + ex-07 (multi-tenant)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Distributed inference


*Parallelism, collectives, and topology*

- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Intra-layer tensor parallelism for transformer models.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA's collective communication implementation.
- [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) - Kernel-issued communication over NVLink load/store, NVLink multicast, and GPU-initiated RDMA.
- [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) - NVLink and InfiniBand topology in GB200 NVL systems.
- [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf) - The scale-out transport specification.

*Mixture-of-experts serving*

- [GShard](https://arxiv.org/abs/2006.16668) - Expert parallelism through all-to-all token dispatch and combine under a per-expert capacity.
- [DeepSeek-V3](https://arxiv.org/html/2412.19437) - Routed experts, shared experts, and the model-system design.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Expert dispatch and combine kernels.
- [EPLB](https://github.com/deepseek-ai/EPLB) - Expert placement and replication from measured load.
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263) - Large-scale MoE inference and communication overlap.

*Prefill and decode disaggregation*

- [DistServe](https://arxiv.org/html/2401.09670) - Separate prefill and decode workers optimized for goodput under latency constraints.
- [Splitwise](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/) - Phase-specific allocation and scheduling.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) - KV-centric disaggregated inference built on a distributed KV cache pool.
- [NIXL](https://github.com/ai-dynamo/nixl) - A transport layer for moving inference state across memory and network backends.
- [Dynamo](https://github.com/ai-dynamo/dynamo) - A production implementation of disaggregated prefill and decode with KV-aware routing.
