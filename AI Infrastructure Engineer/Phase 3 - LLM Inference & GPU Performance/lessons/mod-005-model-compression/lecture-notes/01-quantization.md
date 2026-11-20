# Lecture 01: Quantization

## Why quantize

A 7B fp16 model = ~14 GB. At int8 = 7 GB. At int4 (AWQ/GPTQ) = ~4 GB. Smaller
weights → fits in cheaper GPUs + faster memory transfer.

## The numeric trade-off

- fp32: training default; full precision
- bf16: training + inference; same range as fp32, half precision
- fp16: inference; smaller range than bf16 (gradient underflow risk in training)
- int8: weight + activation quantization; modest accuracy hit
- int4 (AWQ, GPTQ): weights only; near-zero accuracy hit on large models
- int2 / int3: experimental; substantial accuracy loss

## Methods

Typical accuracy hits below are rough, model- and task-dependent; always run your own eval.

| Method | Type | Hardware | Accuracy hit |
|---|---|---|---|
| Dynamic quantization | post-train, int8 | CPU | ~1pp |
| Static quantization | post-train w/ calibration | CPU/GPU | ~0.5pp |
| QAT (quant-aware training) | train-time | any | < 0.2pp |
| AWQ | activation-aware weight-only | GPU | < 0.2pp on large models |
| GPTQ | one-shot weight quantization | GPU | < 0.3pp on large models |

## Which format to pick (question 11)

| Your hardware / bottleneck | First choice | Why |
|---|---|---|
| H100/H200, L4/L40S (FP8 tensor cores) | **FP8 W8A8** (E4M3, per-channel weights, per-token dynamic activations) | ~2× smaller than BF16, faster GEMMs, usually within noise of BF16 |
| A100 / older (no FP8) | **INT8 W8A8** (SmoothQuant) or INT8 weight-only | INT8 tensor cores; SmoothQuant handles activation outliers |
| Memory-bound: small batch, big model, one GPU | **INT4 weight-only** (AWQ / GPTQ, group 128) | 4× fewer weight bytes per decode step; prefill gains little (still dequant + BF16 math) |
| Blackwell | **NVFP4** (W4A4) | native FP4 tensor cores with fine-grained (16-element) scales |
| Long contexts, many users | add **FP8 KV cache** | halves the KV cache (question 3) |

Rule: measure quality loss (perplexity + your task eval) and speed for *your*
batch size; INT4 weight-only can be slower than FP8 at large batch.

## Tooling

```bash
pip install llmcompressor   # successor to AutoAWQ (archived in 2025) for vLLM-ready checkpoints
```

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
# FP8 dynamic: no calibration data needed
oneshot(model="Qwen/Qwen2.5-7B-Instruct",
        recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
        output_dir="qwen2.5-7b-fp8")
# INT4 weight-only: scheme="W4A16" (GPTQ-style, needs a calibration dataset=...)
```

vLLM reads the quantization config from the checkpoint (`vllm serve qwen2.5-7b-fp8`);
for quick tests `vllm serve <model> --quantization fp8` quantizes on load.

## When NOT to quantize

- Small models (< 1B params): accuracy hit can be significant
- Specialty tasks (medical, legal, code): test thoroughly
- Anything with extreme precision needs (e.g., scientific computing)

---

## Deep dive: how it works (core logic)

### The basic mechanics
Affine quantization maps floats to b-bit integers: `q = round(x / s) + z`, dequantize `x ≈ s·(q − z)`, with scale `s = (max − min) / (2^b − 1)`. One scale for a whole tensor is crude, so use **groups** (e.g. 128 weights share one scale/zero). Smaller groups = better accuracy, more scale overhead (a 4-bit weight with 128-groups costs about 4.25 bits).

**Memory arithmetic (70B model):** FP16 140 GB → INT8/FP8 70 GB → 4-bit ≈ 35–40 GB. And because decode is memory-bound, fewer bytes ≈ proportionally faster steps.

### The hard part: outliers
A few channels in transformer **activations** are 10–100× larger than the rest. One scale for the tensor then crushes all normal values to zero. Weights are well-behaved; activations are not. Each method below attacks this differently.

### Weight-only methods
- **[GPTQ](https://arxiv.org/abs/2210.17323):** quantize layer by layer to minimize `‖WX − ŴX‖²` on calibration data. Go column by column; after rounding a column, push its error onto the *not-yet-quantized* columns using the inverse Hessian (`H = 2XXᵀ`) so later columns compensate. One-shot, no retraining.
- **[AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf):** observation: ~1% of weight channels matter most, and you find them by **activation magnitude**, not weight magnitude. Instead of keeping them in FP16 (messy), scale those channels up before quantizing (and scale the activations down to compensate), which shrinks their relative rounding error.

### Weights *and* activations (W8A8)
- **[SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html):** since `Y = X·W`, you can move scale between them: `Y = (X·diag(s)⁻¹)(diag(s)·W)`. Choose `sⱼ = max|Xⱼ|^α / max|Wⱼ|^(1−α)` so activation outliers get divided down and absorbed (harmlessly) into the weights. Now both are easy to quantize to INT8.
- **[QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract.html):** multiply by an orthogonal (Hadamard) rotation Q: `(XQ)(QᵀW) = XW`. The rotation spreads an outlier's energy across all channels, so activations, weights and even the KV cache can go to 4 bits. The rotations are folded into weights, so little runtime cost.

### Floating-point formats instead of integers
- **FP8** ([OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)): E4M3 (max ±448, more precision, for weights/activations) and E5M2 (max ±57,344, more range, for gradients). Use a per-tensor scale = `amax / 448`.
- **Microscaling** ([OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)): a block of values (e.g. 32) shares one scale; elements are tiny floats. **NVFP4** ([Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)): FP4 (E2M1) elements, blocks of **16** sharing an FP8 scale, plus a per-tensor FP32 scale. Small blocks = good dynamic range at 4 bits; Blackwell runs it natively.
- **Execution:** [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) handles the scaling bookkeeping and casting for FP8/FP4.

### KV cache quantization
[KIVI](https://proceedings.mlr.press/v235/liu24bz.html) finds that **keys** have outlier *channels* and **values** don't, so it quantizes keys **per-channel** and values **per-token** (2-bit), keeping a few recent tokens in full precision. Frees large memory for longer contexts or more users.

### Always measure
Quality loss depends on your model and tasks. Evaluate on your own eval set plus long-context and non-English slices, then compare speed and cost.

### Read the originals

- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)
- [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html)
- [QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract.html)
- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [KIVI](https://proceedings.mlr.press/v235/liu24bz.html)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
