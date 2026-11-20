# Resources — CUDA Programming

## Books
- *Programming Massively Parallel Processors* — Hwu & Kirk (4th ed)
- *CUDA by Example* — Sanders & Kandrot (older but accessible)

## Docs + papers
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Volta architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

## PyTorch + extension docs
- [PyTorch C++ extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Triton language](https://triton-lang.org/) — Python-style kernel authoring

## Companion
- engineer-solutions/mod-107 ex-02 (vector-add kernel + PyTorch extension)
- engineer-solutions/mod-107 ex-08 (mixed precision)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Kernel optimization


*Foundational kernel exercises*

- [Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) - Coalescing, shared-memory tiling, and bank conflicts.
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) - Synchronization, divergence, occupancy, and instruction cost.
- [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) - A work-efficient scan with one pass over memory.
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) - Numerically stable online softmax without materialized intermediates.
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Floating-point non-associativity and batch-dependent reduction order, with batch-invariant RMSNorm, matmul, and attention kernels.

*Matrix multiplication*

- [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf) - The canonical case for reasoning from measured hardware behavior instead of occupancy alone.
- [CUTLASS 3.x design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html) - The collective and kernel structure used by modern CUTLASS.
- [Stream-K](https://arxiv.org/abs/2301.03598) - Work-centric GEMM decomposition that splits k-loop iterations evenly across persistent CTAs to avoid wave quantization.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - A compact production FP8 GEMM implementation for Hopper.

*Direct implementation work*

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - A matrix multiplication built from naive CUDA through shared-memory and register tiling.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - Layouts, tiling, PTX, machine code, and roofline analysis.
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - A direct Hopper optimization worklog using tensor cores and asynchronous movement.
- [CUTLASS Tutorial: Mastering TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) - Working kernels built around the Tensor Memory Accelerator.
- [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) - A Blackwell GEMM built step by step from tcgen05 MMA and tensor memory to 2-SM MMA and persistent scheduling.
