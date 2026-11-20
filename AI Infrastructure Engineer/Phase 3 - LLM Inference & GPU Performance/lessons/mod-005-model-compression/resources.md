# Resources — Model Compression

- [AWQ paper](https://arxiv.org/abs/2306.00978) + [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [LoRA paper](https://arxiv.org/abs/2106.09685) + [PEFT library](https://github.com/huggingface/peft)
- [QLoRA paper](https://arxiv.org/abs/2305.14314) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [2:4 sparsity (NVIDIA)](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- [Distillation: DistilBERT paper](https://arxiv.org/abs/1910.01108)
- engineer-solutions/mod-110 ex-05 (LoRA + QLoRA + DDP)
- engineer-solutions/mod-110 ex-06 (inference optimization chain incl. AWQ)


---

## Further reading: Wafer GPU performance list

Curated links from [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). Read these after the lecture notes, not before.

### Kernel optimization

- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf) - E4M3 and E5M2 formats.
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) - Shared-scale MX formats.
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - FP8 and FP4 transformer execution with scaling controls.
- [Blackwell matrix multiply instructions](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html) - `tcgen05`, tensor memory, and Blackwell MMA programming.

### Inference engines

- [GPTQ](https://arxiv.org/abs/2210.17323) - One-shot second-order weight quantization.
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) - W8A8 execution by moving quantization difficulty from activations into weights.
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) - Low-bit weight-only inference with salient-weight protection.
- [QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract.html) - Output-preserving rotations that remove outliers so weights, activations, and the KV cache can run in 4 bits.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) - Hardware-native 4-bit micro-block scaling for Blackwell inference.
