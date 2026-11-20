# Module 05 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **d** | AWQ (Activation-aware Weight Quantization) is engineered around LLM weight distributions — weight-only int4 with per-channel scaling. |
| 2 | **a** | Sparse Tensor Cores (Ampere+) natively skip zeros in 2:4 patterns; on other hardware, "sparse" multiplies still touch the zero values. |
| 3 | **a** | QLoRA = quantize the frozen base model to 4-bit + train a LoRA adapter on top. Puts 13B fine-tuning on a single 24GB GPU. |
| 4 | **d** | Distillation transfers the teacher's behavior to the student via soft labels. A weak teacher produces a weak student. |
| 5 | **a** | AWQ's published results consistently show < 0.5pp drop on MMLU/HellaSwag/etc. for 7B+ models; smaller models suffer more. |
| 6 | **d** | Per adapted 4096×4096 matrix: 16·(4096+4096) = 131K params. Llama-7B (32 layers): q,v only → 8.4M params ≈ 17 MB in FP16; all 7 linear layers → ~40M ≈ 80 MB. Either way ~0.1–0.6% of the 13 GB base. |
| 7 | **d** | vLLM `--enable-lora` keeps N adapters next to one base model; the `model` field in the request picks the adapter, and requests for different adapters share a batch. |
| 8 | **a** | Quantization changes the bit-width of weights but leaves architecture + structure unchanged. |
| 9 | **d** | Distillation is the only method that lets you change the architecture (different layer count, hidden size, etc.) while preserving behavior. |
| 10 | **d** | vs FP16: INT8 = 2×; 2:4 keeps half the values plus ~12% index metadata ≈ 1.8×; together ≈ 3.5×. (vs FP32 it would be ≈ 7×.) |
