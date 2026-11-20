# Module 1 — Resources

> Primary sources only. The rule is: if a number appears in
> `lecture-notes.md` or in an exercise's check, it traces back to one
> of these documents.

## NVIDIA primary sources

**CUDA C Programming Guide** — the authoritative reference for the
execution model, the memory model, and per-compute-capability limits.
The "Compute Capabilities" section (Appendix H in older revisions)
is what you reach for when you need the resource limits for a specific
GPU generation.
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

**NVIDIA H100 Tensor Core GPU Datasheet** — H100 SXM and PCIe peak
throughputs (FP64, FP32, TF32, BF16, FP16, FP8, INT8), memory
bandwidth (3.35 TB/s SXM, 2 TB/s PCIe), TDP, NVLink bandwidth.
*The "with sparsity" asterisk is important*: published tensor-core
TFLOPS are the 2× sparsity number; divide by 2 for dense.
<https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306>

**NVIDIA A100 Tensor Core GPU Datasheet** — A100 SXM4 and PCIe specs
including the 80GB and 40GB variants. The 80GB SXM4 has 2039 GB/s
HBM2e bandwidth (vs 1555 GB/s for the 40GB variant) — they are
not interchangeable.
<https://www.nvidia.com/en-us/data-center/a100/>

**NVIDIA Hopper Architecture Whitepaper** — the deepest source for
the SM micro-architecture: fourth-gen Tensor Cores, the Transformer
Engine, distributed shared memory, tensor memory accelerator.
<https://resources.nvidia.com/en-us-tensor-core>

**NVIDIA Ampere Architecture Whitepaper** — same for A100. Third-gen
Tensor Cores, TF32, sparsity acceleration, MIG.
<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf>

**NVIDIA Nsight Compute documentation** — kernel-level profiler. The
metrics dictionary (`smsp__*`, `dram__*`, `sm__*`) is dense; the
"Profiling Guide" is the right entry point.
<https://docs.nvidia.com/nsight-compute/>

**NVIDIA Nsight Systems documentation** — system-level / multi-process
timeline profiler. Use when the question is "is the GPU even busy?"
rather than "why is this kernel slow?"
<https://docs.nvidia.com/nsight-systems/>

## Roofline model

**S. Williams, A. Waterman, D. Patterson.** "Roofline: An Insightful
Visual Performance Model for Multicore Architectures."
*Communications of the ACM*, April 2009. The original paper that
defined the model used in `exercises/exercise-04-roofline-analysis/`.
<https://dl.acm.org/doi/10.1145/1498765.1498785>

**LBNL Roofline tutorial** — the practical companion to the Williams
paper. Shows how to actually measure the points the model predicts.
<https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/>

## Occupancy beyond the basics

**V. Volkov.** "Better Performance at Lower Occupancy."
*GTC 2010.* The canonical counterexample to "always maximize
occupancy." Volkov shows kernels that beat their high-occupancy
versions by exploiting instruction-level parallelism. Required
reading once you have done Exercise 3.
<https://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf>

## Optional but high-value

**NVIDIA CUDA C++ Best Practices Guide** — the "what to do" companion
to the Programming Guide's "how it works." Coalescing rules, shared
memory patterns, kernel launch overhead.
<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

**Mark Harris — "Optimizing Parallel Reduction in CUDA"** (NVIDIA
slides, 2007). A worked example of taking one kernel through seven
stages of optimization. Still the cleanest demonstration of the
methodology in Module 1.
<https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>

---

*Last revised: 2026-05-22 — initial build.*


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Start here: the minimum mental model

Read these in order if you are new to the field.
1. [How to Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) - One request from prefill through decode, with batching, KV memory, and parallelism.
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The transformer computation that the rest of the list optimizes.
3. [CUDA C++ basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) - The shortest official introduction to the CUDA execution model.
4. [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1) - The main textbook for GPU programming, memory, and kernel design.
5. [Roofline: An Insightful Visual Performance Model](https://users.cs.duke.edu/~lkw34/papers/roofline-cacm2008.pdf) - The compute, memory-bandwidth, and arithmetic-intensity model.
6. [Transformer Inference Arithmetic](https://kipply.github.io/blog/transformer-inference-arithmetic/) - FLOPs, parameter bytes, KV bytes, and communication for transformer inference.
7. [Efficiently Scaling Transformer Inference](https://proceedings.mlsys.org/paper_files/paper/2023/file/c4be71ab8d24cdfb45e3d06dbfca2780-Paper-mlsys2023.pdf) - Latency, memory, and parallelism costs for large-model inference.
8. [Etalon](https://arxiv.org/html/2407.07000) - TTFT, TPOT, goodput, and latency SLOs for generative-model serving.
For a practical companion, use the [GPU Mode lectures](https://github.com/gpu-mode/lectures).

### GPU fundamentals


*Programming model*

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - The normative CUDA reference.
- [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) - Threads, warps, blocks, grids, and the memory hierarchy.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Coalescing, shared memory, occupancy, synchronization, and optimization workflow.
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) - TMA, thread-block clusters, asynchronous execution, and Hopper-specific limits.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) - Tensor memory, Blackwell execution features, and architecture limits.

*Compilation and machine code*

- [NVCC Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) - The CUDA compilation trajectory and artifact controls.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - NVIDIA's virtual instruction set and memory model.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) - `cuobjdump` and `nvdisasm` for inspecting GPU binaries.
- [Understanding PTX](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) - NVIDIA's introduction to the role of PTX between CUDA and machine code.

### Current hardware (read with the ISA or tuning guide; vendor peaks are not measurements)


*NVIDIA*

- [Blackwell architecture brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief) - Blackwell and Blackwell Ultra system architecture.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) - Programming and optimization guidance.
- [CUTLASS Blackwell documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html) - Blackwell matrix multiply and data-movement support.

*AMD*

- [HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) - Compute units, SIMDs, the LDS, matrix cores, and caches that HIP kernels map onto.
- [CDNA 4 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) - MI350 compute, memory, and chiplet architecture.
- [CDNA 4 instruction set](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) - The native machine instruction reference.
- [MI350 performance counters](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350-performance-counters.html) - Counter definitions and measurement guidance.

*Google TPU*

- [TPU v1 analysis](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/) - The original datacenter TPU paper.
- [Ten Lessons From Three Generations Shaped Google's TPUv4i](https://doi.org/10.1109/ISCA52012.2021.00010) - Inference-accelerator design driven by P99 latency limits, multi-tenancy, backward ML compatibility, and cost of ownership.
- [TPU v4](https://arxiv.org/abs/2304.01433) - The TPU v4 chip, interconnect, and system.
- [Ironwood documentation](https://docs.cloud.google.com/tpu/docs/tpu7x) - Current TPU v7 architecture and configuration.
- [Pallas TPU hardware model](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) - The TPU execution and memory model for kernel authors.

*AWS Trainium*

- [Trainium and Inferentia2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html) - NeuronCore v2 compute and memory architecture.
- [Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html) - NeuronCore v3 compute and memory architecture.
- [Trainium3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html) - The current NeuronCore architecture.
- [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html) - Kernel optimization for Trainium and Inferentia.
