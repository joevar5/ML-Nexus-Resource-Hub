# Lecture 03: MIG + FP8

## MIG (Multi-Instance GPU)

A100/H100 GPUs can partition into 7 isolated instances. Each:
- Has its own memory + compute
- Is hardware-isolated from other instances
- Looks like an independent GPU to the OS

Use for:
- Multi-tenant dev/notebook clusters
- Mixing serving (need isolation) + batch on the same node
- Better utilization for many small models

Trade-offs: smaller per-instance memory; can't use full GPU for one workload.

## FP8

Ada (L4, L40S), Hopper (H100/H200) and Blackwell have FP8 tensor cores. Two formats:
- E4M3: range [-448, 448], finer precision
- E5M2: range [-57344, 57344], wider range for gradients

Use cases:
- Training: often 30–50% faster than bf16 with minimal accuracy loss (use Transformer Engine)
- Inference: up to ~2× on compute-bound GEMMs, plus half the weight and KV bytes
  for memory-bound decode (Mod 5 Lecture 01, question 11)

The catch: fp8 dynamic range is narrow; per-tensor scaling factors are required.
Transformer Engine + Megatron-LM handle this; rolling your own is risky.

## When FP8

- Ada/Hopper/Blackwell GPUs (A100 and older have no FP8 tensor cores)
- Large matmuls (transformer linears); tiny layers gain little after scaling overhead
- Inference serving: FP8 weights + FP8 KV is the usual first quantization step on H100

When NOT:
- Layers that are sensitive to range (embeddings, final `lm_head`, often kept in BF16)
- Tiny models (won't benefit much)
- Sensitive accuracy domains (medical, financial) without thorough testing

---

## Deep dive: how it works (core logic)

MIG (Multi-Instance GPU, which splits one GPU into isolated slices with their own memory and SMs) is **not covered by the Wafer list**. Use NVIDIA's MIG documentation for it. The FP8 side is covered well:

### FP8 formats ([OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf))
| Format | Bits (sign/exp/mantissa) | Max | Use |
| :--- | :--- | :--- | :--- |
| E4M3 | 1/4/3 | ±448 | weights, activations (more precision) |
| E5M2 | 1/5/2 | ±57,344 | gradients (more range) |
Only 8 bits, so values must be **scaled** into range: `scale = amax / 448` for E4M3, `x_fp8 = cast(x / scale)`.

### How FP8 execution actually works ([NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine))
- Keep **master weights** in higher precision; cast to FP8 for the GEMM.
- **Scaling recipes:** *delayed scaling* uses a history of recent `amax` values to pick the scale (cheap, no extra pass); *current scaling* computes `amax` of the tensor right now (more accurate, extra pass).
- The tensor core multiplies FP8 inputs and **accumulates in higher precision**; output is rescaled and written in BF16/FP16 (or FP8 for the next layer).

### Accumulation precision matters ([DeepGEMM](https://github.com/deepseek-ai/DeepGEMM))
FP8 tensor-core accumulation has limited effective mantissa bits; summing thousands of products can lose accuracy. DeepGEMM (a compact FP8 GEMM for Hopper, ~one core file) uses **fine-grained scaling** (per-tile / per-block scales instead of per-tensor) and **promotes partial sums to FP32 on CUDA cores** at intervals, restoring accuracy at small cost.

### Microscaling and FP4 ([OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf))
Instead of one scale per tensor, give each small **block** its own scale (MX: 32 elements, shared power-of-two scale). Narrow element formats (FP4/FP6/FP8) then keep good dynamic range. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) is NVIDIA's 4-bit variant: blocks of 16 with an FP8 scale plus a tensor-level scale, run natively on Blackwell.

### Read the originals

- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
