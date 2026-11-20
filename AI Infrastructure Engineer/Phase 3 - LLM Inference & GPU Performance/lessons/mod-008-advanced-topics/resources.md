# Resources — Advanced Topics

- [CUDA Graphs documentation](https://developer.nvidia.com/blog/cuda-graphs/)
- [NVIDIA NCCL](https://github.com/NVIDIA/nccl) + [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [InfiniBand basics](https://www.nvidia.com/en-us/networking/infiniband/)
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [MIG user guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Transformer Engine (fp8)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- engineer-solutions/mod-107 ex-10 (GPU sharing strategies)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Distributed inference (interconnect)

- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Intra-layer tensor parallelism for transformer models.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA's collective communication implementation.
- [NCCL device-initiated communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) - Kernel-issued communication over NVLink load/store, NVLink multicast, and GPU-initiated RDMA.
- [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) - NVLink and InfiniBand topology in GB200 NVL systems.
- [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf) - The scale-out transport specification.

### Frontier (changes quickly)


*AI-generated kernels*

- [KernelBench](https://proceedings.mlr.press/v267/ouyang25a.html) - The original benchmark for converting PyTorch operators into faster GPU kernels.
- [KernelBench-Verified](https://arxiv.org/html/2607.16241) - Stronger correctness tests and baseline parity.
- [FlashInfer-Bench](https://proceedings.mlsys.org/paper_files/paper/2026/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html) - Kernel tasks drawn from real serving traces, with correctness-checked kernels substituted into SGLang and vLLM.
- [SOL-ExecBench](https://github.com/nvidia/sol-execbench) - Correctness and performance measured against a hardware speed-of-light model.
- [KernelGuard](https://icml.cc/virtual/2026/67922) - Layered defenses against reward-hacked kernel submissions: static rules, physics-floor checks, and an LLM judge.

*Watchlist*

- [NVIDIA Rubin](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/), pending a tuning guide or architecture whitepaper and reproducible measurements on generally available systems.
- AMD MI455X and Helios, with the [CDNA 5 whitepaper](https://www.amd.com/content/dam/amd/en/documents/products/technologies/cdna/amd-cdna5-whitepaper.pdf) and ISA guide published, pending reproducible measurements on shipped systems.
- [Google TPU 8t and TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), pending Cloud TPU architecture documentation and reproducible measurements on generally available systems.
- [UALink 2.0](https://ualinkconsortium.org/specification/), pending shipped silicon and reproducible measurements.
- [Gluon](https://triton-lang.org/main/gluon/index.html), Triton's lower-level GPU programming model, pending a stable API across releases.
- Session-aware and agentic scheduling against public production traces.
- Real-time voice and video serving with complete quality and latency metrics.
- Inference ASICs, processing in memory, analog compute, and photonic compute with reproducible deployments.
- Individual AI kernel agents that have not been rerun on a hardened evaluator.
