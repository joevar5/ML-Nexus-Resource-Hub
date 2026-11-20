# Module 05 — Quiz

10 questions. 70% pass.

### 1. AWQ is most useful for:
- a) CPU-only inference acceleration
- b) Training-time acceleration
- c) Embedding model compression
- d) Weight-only int4 quantization of large language models

### 2. The hardware feature that natively accelerates 2:4 structured sparsity is:
- a) NVIDIA Sparse Tensor Cores (A100 generation onward)
- b) Any GPU with CUDA support
- c) CPU AVX-512 instructions
- d) TPU-only

### 3. QLoRA combines:
- a) 4-bit base model quantization + LoRA fine-tuning
- b) Quantization-aware training + pruning
- c) Knowledge distillation + structured sparsity
- d) Pruning + post-training quantization

### 4. Distillation requires:
- a) A smaller training dataset
- b) Fewer GPUs than full training
- c) A different optimizer
- d) A teacher model that already performs well on the target task

### 5. Typical AWQ int4 accuracy hit on a 7B+ LLM:
- a) < 0.5pp drop on most academic benchmarks
- b) 5-10pp drop (substantial regression)
- c) Unmeasurable — no tooling exists
- d) > 20pp drop (model effectively broken)

### 6. A LoRA adapter for a 7B model at rank 16 is roughly:
- a) The same size as the base model
- b) Larger than the base model
- c) Zero bytes (LoRA stores nothing on disk)
- d) Tens of MB (≈ 16–80 MB in FP16, depending on which layers get adapters)

### 7. vLLM's LoRA hot-swap supports:
- a) Replacing the vLLM server with a custom one per adapter
- b) Requiring full retraining per request
- c) Latency that scales linearly with the number of adapters loaded
- d) One base model + N adapters in memory; clients pick an adapter by name in the request

### 8. To make weights smaller while keeping architecture identical:
- a) Quantization
- b) Distillation (changes the model)
- c) Pruning (changes weights to zeros — architecture varies)
- d) Fine-tuning (orthogonal to size)

### 9. To produce a smaller model with a different architecture:
- a) Quantization
- b) Pruning
- c) LoRA adaptation
- d) Distillation

### 10. Stacking INT8 quantization + 2:4 sparsity on an FP16 model:
- a) Cannot be combined
- b) Halves accuracy
- c) Doubles inference latency
- d) Combines memory savings: ≈ 3.5× smaller weights overall

Answers: [quiz-answers.md](quiz-answers.md)
