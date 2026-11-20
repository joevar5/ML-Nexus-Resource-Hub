# Source Digest — Model Compression

An **advanced layer** on top of the lecture notes. What the papers and docs actually say, as steps and numbers. All read from the source text unless marked *(not read)*.

---

## 1. The outlier problem, and four ways around it

Quantization to 4–8 bits fails mostly because of **outliers**: a few huge values stretch the scale and crush everything else. Each method below attacks that differently.

| Method | What it quantizes | Core idea | Headline result |
| :--- | :--- | :--- | :--- |
| **GPTQ** | weights only | fix rounding error column by column using second-order info | 175B model in ~4 GPU-hours at 3–4 bits |
| **AWQ** | weights only | protect the ~1% salient channels (found via activations) by scaling | big perplexity drop at INT3 |
| **SmoothQuant** | weights + activations (W8A8) | move scale from activations into weights | ~1.56× speedup, 2× memory |
| **QuaRot** | weights + activations + KV (all 4-bit) | rotate to remove outliers, output unchanged | LLaMA-2 70B: ≤0.47 perplexity loss |

---

## 2. GPTQ: correct the error as you quantize
**Source:** [GPTQ](https://arxiv.org/abs/2210.17323)

1. **One-shot, post-training:** no retraining, uses approximate **second-order (Hessian)** information per layer.
2. **Procedure:** quantize a weight column, then **update the remaining unquantized weights** to compensate for the error just introduced, using the inverse Hessian.
3. **Makes it fast:** (a) arbitrary-order quantization (order matters little); (b) **lazy batch updates**: process **B = 128 columns at a time**, keeping updates inside that block and applying global updates only when the block is done. The compute is the same but it fixes the *memory-throughput* bottleneck (an "order of magnitude" speedup); (c) a **Cholesky reformulation** to stay numerically stable.
4. **Results:** 175B-parameter models quantized in ~**4 GPU hours** to 3–4 bits with negligible accuracy loss; the first time a 175B model ran on a **single GPU**. End-to-end generation speedups of **~3.25× on A100** and **~4.5×** on cheaper GPUs. Still reasonable at 2-bit/ternary.

---

## 3. AWQ: only ~1% of weights matter, and activations tell you which
**Source:** [AWQ](https://arxiv.org/abs/2306.00978)

1. **Finding:** protecting just **0.1%–1% salient weights** greatly cuts quantization error.
2. **How to find them:** look at the **activation** distribution, *not* the weight magnitudes.
3. **Problem:** keeping 1% in FP16 makes a hardware-unfriendly mixed-precision kernel.
4. **Solution:** mathematically, **scaling up the salient channels** (and scaling the matching activations down, an equivalent transformation) reduces their relative rounding error, with everything staying in low bit.
5. **Scales come from offline activation statistics.** No backprop or reconstruction, so it does not overfit the calibration set and generalizes to instruction-tuned and multi-modal models.
6. **Evidence (OPT-6.7B, INT3, group 128):** plain round-to-nearest perplexity **43.2**; keeping 1% salient in FP16 → **13.0**; AWQ's scaling gets close to that without mixed precision.
7. **System side:** TinyChat framework (fused kernels, platform-aware weight packing) gives **>3×** over Hugging Face FP16.

---

## 4. SmoothQuant: W8A8 by migrating difficulty
**Source:** [SmoothQuant](https://arxiv.org/abs/2211.10438) *(abstract page only)*

1. Training-free, post-training; shifts quantization difficulty **from activations to weights** with a mathematically equivalent transformation, making INT8 weights **and** activations possible for all matmuls.
2. Reported: **up to 1.56× speedup, 2× memory reduction**, negligible accuracy loss; lets a **530B** model serve on a single node. Covers OPT, BLOOM, GLM, MT-NLG, Llama-1/2, Falcon, Mistral, Mixtral.

---

## 5. QuaRot: rotate the outliers away
**Source:** [QuaRot](https://arxiv.org/abs/2404.00456)

1. Uses **randomized Hadamard transformations** on the model's inputs (and applies the same invariance to the residual stream, FFN activations, parts of attention, and the KV cache).
2. **Computational invariance:** rotating by an orthogonal matrix and folding the inverse into the next weights leaves the output identical but removes outlier features.
3. Result: **all matmuls in 4 bits** with no channels kept in higher precision, covering **weights, activations and KV cache**.
4. **LLaMA-2 70B at 4-bit: ≤ 0.47 WikiText-2 perplexity loss and 99% of zero-shot performance retained.** Also lossless 6- and 8-bit with plain round-to-nearest and no calibration data.

---

## 6. KIVI: how to quantize the KV cache
**Source:** [KIVI](https://arxiv.org/abs/2402.02750)

1. Studied the element distribution of KV caches.
2. **Finding:** quantize the **key cache per-channel** (group along the channel dimension) and the **value cache per-token**.
3. Tuning-free **2-bit** algorithm with a hardware-friendly implementation.
4. Result: Llama, Falcon, Mistral keep near-identical quality with **2.6× lower peak memory** (including weights), enabling **up to 4× larger batch** and **2.35–3.47× throughput**.

---

## 7. NVFP4 vs MX formats: the numbers
**Source:** [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

| | MXFP4 (OCP spec) | NVFP4 |
| :--- | :--- | :--- |
| Element | FP4 (E2M1) | FP4 (E2M1): magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| Block size | 32 | **16** |
| Block scale | E8M0 (power of two) | **FP8 E4M3** (fractional) |
| Extra scale | none | one **FP32 per tensor** |
| Bits per value | 4.25 | **~4.5** |

1. **Why smaller blocks and fractional scales:** twice as many chances to match the local range; in the article's toy example **E4M3 scaling gave MSE 0.08 vs 0.72 for E8M0**.
2. **Footprint:** ~**3.5× smaller than FP16**, ~**1.8× smaller than FP8**.
3. **Accuracy, DeepSeek-R1-0528, FP8 vs NVFP4 (post-training):** MMLU-PRO 85→84, GPQA Diamond 81→80, LiveCodeBench 77→76, HLE 15→14, SciCode 40→40, MATH-500 98→98, AIME 2024 89→91. Claim: ≤1% degradation on key tasks.
4. **Hardware:** Blackwell 5th-gen tensor cores handle microscaled FP4 natively; plain FP4 has no accelerated scaling.
5. **Efficiency claim:** up to 25× (Blackwell) and 50× (Blackwell Ultra) energy per token vs an H100 baseline when running NVFP4 workloads (vendor figure).

**Tooling:** [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) supports FP8 training on Hopper/Ada/Blackwell and **MXFP8 and NVFP4 on Blackwell**, tracking scaling factors and amax histories; PyTorch and JAX. *(The repo page names no usage example or "delayed/current scaling" recipes; read its low-precision guide for those.)*

---

## 8. Sparsity inside attention (long contexts)
**Sources:** [MInference](https://arxiv.org/abs/2407.02490), [Native Sparse Attention](https://arxiv.org/abs/2502.11089), [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)

**MInference (works on existing models):**
1. Problem: prefilling a **1M-token** prompt takes **~30 minutes** on one A100 for an 8B model.
2. Three attention patterns: **A-shape**, **Vertical-Slash**, **Block-Sparse**.
3. **Offline:** assign each head its best pattern. **Online:** build sparse indices cheaply per prompt (for Vertical-Slash, use the last few queries' attention to locate vertical and slash lines; for Block-Sparse, mean-pool then select blocks), then run custom sparse kernels.
4. No pre-training changes or fine-tuning. Up to **10× faster prefill** at 1M tokens on A100 with comparable accuracy (InfiniteBench, RULER, PG-19, Needle-in-a-Haystack; LLaMA-3-1M, GLM-4-1M, Yi-200K, Phi-3-128K, Qwen2-128K).

**NSA (trained in from scratch):**
1. Three attention paths: **compressed coarse-grained tokens**, **selectively retained fine-grained tokens**, and **sliding windows** for local context.
2. "Arithmetic-intensity-balanced" design, hardware-aligned; trainable end to end.
3. Matches or exceeds full attention on benchmarks, with large speedups on **64K** sequences in decoding, forward and backward. *(Exact speedup figures not extracted.)*

**DeepSeek Sparse Attention (DSA) in V3.2:**
1. A **lightning indexer** computes a score for each earlier token: `I(t,s) = Σ_j w(t,j) · ReLU(q(t,j) · k(s))` over a small number of indexer heads, implementable in **FP8**.
2. **Fine-grained token selection:** attend only to the KV entries with the **top-k** index scores.
3. Built on MLA (served as MQA at kernel level so entries are shared across queries). *(The value of k was not extracted; an earlier note's "2,048" is unverified.)*

---

## 9. KV memory: MLA, Jenga and fairness
**Sources:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [Jenga](https://arxiv.org/abs/2503.18292), [Fairness in Serving LLMs](https://arxiv.org/abs/2401.00588)

**MLA (read from DeepSeek-V2):**
1. Cache a **compressed latent** `c_KV ∈ R^{d_c}` (with d_c ≪ d_h·n_h) instead of full keys/values; up-project with `W_UK`, `W_UV` at use.
2. Cache per layer = **d_c** elements, plus a decoupled RoPE key: total **(d_c + d_h^R) × layers**, about **4.5 d_h per layer**.
3. The paper says this **equals GQA with only 2.25 groups** in cache size, while performing stronger than MHA. Overall: **KV cache −93.3%, throughput 5.76×**.

**Jenga (memory management for heterogeneous layers):**
1. Problem: modern models mix embedding sizes and attention types, so one fixed block size wastes memory, and one eviction policy doesn't fit all layers.
2. **Two-level allocator** using the **least common multiple** of embedding sizes to cut fragmentation; APIs for **layer-specific caching/eviction**.
3. Implemented on vLLM: **up to 79.6% better GPU memory utilization**, throughput **up to 4.92× (1.80× average)**.

**Fairness (Virtual Token Counter):**
1. Rate limits are a crude fairness notion and waste capacity when idle.
2. Define fairness by a **cost function over input and output tokens** processed (output tokens cost more than input).
3. **VTC:** keep a counter per client; always schedule the client with the **smallest counter**; handle newly arriving clients carefully so they cannot bank credit. Works inside continuous batching.
4. **Proof:** a tight **2× bound** on the service difference between two backlogged clients, while staying work-conserving.

---

## Not read
[LoRA / QLoRA / 2:4 sparsity / distillation] are not in the Wafer list (the lecture notes cover them). [KIVI's 2-bit per-group details], [QuaRot's exact kernel design], and NSA's speedup table were not extracted.
