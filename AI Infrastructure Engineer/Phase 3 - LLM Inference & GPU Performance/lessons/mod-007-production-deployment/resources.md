# Resources — Production Deployment

- [vLLM](https://docs.vllm.ai/), [TGI](https://huggingface.co/docs/text-generation-inference/index)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [TorchServe](https://pytorch.org/serve/)
- [BentoML](https://docs.bentoml.org/)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [Argo Rollouts](https://argo-rollouts.readthedocs.io/)
- [Karpenter](https://karpenter.sh/)
- engineer-solutions/mod-104 (Kubernetes operational), mod-106 ex-08 (deployment strategies), mod-110 (LLM infra)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Inference engines


*Scheduling and continuous batching*

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) - Iteration-level scheduling for autoregressive serving.
- [PagedAttention and vLLM](https://arxiv.org/html/2309.06180) - Paged KV allocation and continuous batching.
- [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) - Chunked prefills that reduce interference with decode.
- [SGLang](https://arxiv.org/html/2312.07104) - Prefix reuse, structured programs, and a serving runtime.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - The main production engine implementations.

*Structured decoding and fairness*

- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) - Regex constraints compiled into a finite-state index over the vocabulary, extended to grammars with an incremental parser.
- [Guiding LLMs the Right Way](https://proceedings.mlr.press/v235/beurer-kellner24a.html) - Constrained decoding without changing the intended token distribution.
- [XGrammar](https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf) - A fast grammar engine for structured generation.
- [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588) - Fair scheduling when request sizes are different and unknown.

### Distributed inference


*Production systems*

- [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati) - Predictable model serving through centralized scheduling.
- [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu) - Faster model startup and live migration.
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) - Model, accelerator, and KV-aware request routing.
- [llm-d](https://github.com/llm-d/llm-d) - Distributed routing, scheduling, and disaggregated serving on Kubernetes.

*Serving benchmarks*

- [MLPerf Inference](https://arxiv.org/abs/1911.02549) - Reproducible benchmark scenarios and load generation.
- [Etalon](https://arxiv.org/html/2407.07000) - Goodput under per-request latency SLOs.
- [ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf) - Workload generation that preserves important production-trace properties.
- [BurstGPT](https://github.com/HPMLL/BurstGPT) - A public trace for bursty LLM workloads.
- [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/) - An endpoint-level benchmark for interactive generative AI.
