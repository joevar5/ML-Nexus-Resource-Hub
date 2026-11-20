# Module 04 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **a** | FlashAttention tiles the attention computation so the intermediate N×N matrix never lives in HBM. Memory drops from O(N²) to O(N). |
| 2 | **d** | Vanilla KV-cache allocation fragments memory; PagedAttention's fixed-page approach (analogous to virtual memory) eliminates that, increasing concurrent requests/GPU by 2-4×. |
| 3 | **d** | Verification compares draft + target token distributions; that requires sharing a tokenizer. Same-family fine-tunes are the typical pairing. |
| 4 | **d** | Continuous batching's distinguishing move is removing head-of-line blocking — finished requests drop, new ones join at every decoding step. |
| 5 | **a** | The prefix that's worth caching is one that repeats across requests — system prompts in chat/RAG/agent workloads are the canonical fit. |
| 6 | **d** | `mode="reduce-overhead"` adds CUDA graph capture so launch overhead amortizes across replays. Fusion comes from Inductor in every mode; `max-autotune` adds kernel autotuning. |
| 7 | **a** | Memorize: batch × seq × layers × **KV** heads × head_dim × 2 (K + V) × precision_bytes. With GQA, KV heads < query heads (e.g. 8 vs 64), which cuts the cache 8×. |
| 8 | **d** | Triton (OpenAI) is a Python-style DSL that compiles to PTX; it is the code generator behind `torch.compile` (Inductor) and many vLLM/SGLang kernels. The official FlashAttention kernels are CUDA/CUTLASS. |
| 9 | **d** | The gains overlap (e.g. prefix caching and chunked prefill both cut prefill work), so they do not multiply cleanly. vLLM's launch blog reported up to 24× throughput over HF Transformers. |
| 10 | **a** | vLLM defaults capture ~80% of the wins. Hand-rolling kernels before measuring is premature optimization. |
