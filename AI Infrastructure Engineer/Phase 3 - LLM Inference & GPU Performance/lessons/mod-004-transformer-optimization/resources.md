# Resources — Transformer Optimization

- [FlashAttention paper](https://arxiv.org/abs/2205.14135) + [FA2](https://arxiv.org/abs/2307.08691)
- [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180)
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192)
- [vLLM docs](https://docs.vllm.ai/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [HuggingFace text-generation-inference](https://huggingface.co/docs/text-generation-inference)
- engineer-solutions/mod-110 (14 LLM infrastructure exercises)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Attention

- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware exact attention.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Better work partitioning and parallelism.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) - Asynchronous movement and tensor-core overlap on Hopper.
- [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf) - The Blackwell attention schedule.
- [FlashInfer](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html) - Block-sparse KV formats, JIT-compiled attention variants, and CUDA Graph-compatible load balancing for serving.

### Programming models


*Triton*

- [Triton paper](https://eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) - The original blocked-program language and compiler design.
- [Triton programming guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) - The official programming model.
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) - Official worked kernels for fused softmax, matmul with grouped tile ordering, fused attention, persistent matmul, and block-scaled matmul.
- [Linear Layouts](https://arxiv.org/abs/2505.23819) - Triton's layout system, which represents layouts and layout conversions as binary matrices over F2.
- [Triton repository](https://github.com/triton-lang/triton) - Compiler, examples, tests, and backend implementation.

*CUTLASS, CuTe, and CUDA Tile*

- [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html) - Layouts and layout composition.
- [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html) - A GEMM expressed through CuTe layouts and atoms.
- [CUTLASS pipeline documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html) - Producer-consumer pipelines and asynchronous stages.
- [Writing tile kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html) - Tile-level kernels in cuTile Python and CUDA Tile C++, with the compiler mapping tiles to threads, tensor cores, and TMA.
- [CUDA Tile IR programming model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html) - NVIDIA's compiler-owned tile abstraction.

*Other hardware stacks*

- [ROCm Composable Kernel](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/) - AMD tiling, layout, and operator primitives.
- [ROCm AITER](https://github.com/ROCm/aiter) - AMD inference and transformer operator implementations.
- [HipKittens](https://github.com/HazyResearch/HipKittens) - A tile abstraction for AMD GPUs.
- [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) - The JAX kernel model for GPU and TPU backends.
- [NKI language guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/nki-language-guide.html) - The tile-level kernel language for AWS NeuronCore hardware.

### Inference engines


*KV cache systems*

- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) - Fewer key-value heads and a smaller KV cache.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - Multi-head latent attention and compressed KV state.
- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html) - KV quantization with separate treatment for keys and values.
- [CacheGen](https://arxiv.org/abs/2310.07240) - KV compression for transfer.
- [Jenga](https://arxiv.org/abs/2503.18292) - KV memory allocation and per-layer caching for models that mix attention types, state-space layers, and vision embeddings.

*Speculative decoding*

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html) - Exact sampling with a draft model.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - The parallel formulation and analysis.
- [Medusa](https://arxiv.org/html/2401.10774) - Multiple prediction heads on the target model.
- [EAGLE](https://proceedings.mlr.press/v235/li24bt.html) - Feature-level drafting.
- [EAGLE-3](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html) - Drafters that predict tokens directly from fused multi-layer target features, trained on their own multi-step outputs.

*Long context and multimodal inference*

- [Ring Attention](https://arxiv.org/abs/2310.01889) - Exact distributed attention by circulating KV blocks around a device ring.
- [MInference 1.0](https://arxiv.org/abs/2407.02490) - Dynamic sparse patterns for long-context prefill on existing models.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) - A model trained with a hardware-aligned sparse attention hierarchy.
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) - DeepSeek Sparse Attention, where a lightweight learned indexer selects the top-k key-value entries for each query.
- [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) - Current engine support for text, image, audio, and video inputs.
