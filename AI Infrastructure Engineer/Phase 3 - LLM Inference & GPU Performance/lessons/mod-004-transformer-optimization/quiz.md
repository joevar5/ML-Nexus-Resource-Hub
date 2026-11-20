# Module 04 — Quiz

10 questions. 70% pass.

### 1. FlashAttention reduces attention memory from:
- a) O(N²) to O(N)
- b) O(N) to O(log N)
- c) O(N²) to O(N²) — only speed improved
- d) O(1) regardless of sequence length

### 2. PagedAttention's primary value is:
- a) Smaller model size
- b) Faster training
- c) Better generation accuracy
- d) Eliminates KV-cache fragmentation → more concurrent requests per GPU

### 3. Speculative decoding requires:
- a) Two equal-size models for verification
- b) A pre-trained vocab swap step
- c) That it always be slower than greedy decoding
- d) A small draft model that shares a tokenizer with the target model

### 4. Continuous batching speeds LLM serving by:
- a) Padding all requests in a batch to the longest sequence length
- b) Batching by time-of-day buckets
- c) Disabling batching entirely
- d) Dropping finished requests + adding new ones at every step (no fixed batches)

### 5. Prefix caching benefits workloads where:
- a) Many requests share a common system prompt or context
- b) Every prompt is uniquely random
- c) Only during training
- d) Only for embedding generation

### 6. `torch.compile(model, mode="reduce-overhead")`:
- a) Disables most optimizations
- b) Reduces memory only, not latency
- c) Is required for any inference workload
- d) Captures CUDA graphs (on top of Inductor's usual fusion) to cut per-launch overhead

### 7. KV cache size scales as:
- a) batch × seq_len × num_layers × num_kv_heads × head_dim × 2 (K + V) × precision_bytes
- b) Just batch size
- c) Only sequence length
- d) A constant regardless of inputs

### 8. Triton is:
- a) An LLM developed by Anthropic
- b) A vendored inference server (NVIDIA)
- c) A profiler for GPU kernels
- d) A Python-style language + JIT compiler for authoring GPU kernels

### 9. The combined speedup from FlashAttn + continuous batching + prefix cache + spec decode is typically:
- a) Strictly additive; rarely exceeds 2×
- b) Cannot be combined (mutually exclusive)
- c) Always exactly 100×
- d) Large but not cleanly multiplicative (gains overlap); up to ~24× throughput over a HuggingFace baseline was reported at vLLM's launch

### 10. The right place to start optimizing transformer inference is:
- a) Use vLLM defaults; profile + tune from there
- b) Write hand-rolled CUDA kernels from scratch
- c) Add more GPUs before tuning
- d) Quantize to int4 immediately, no profiling

Answers: [quiz-answers.md](quiz-answers.md)
