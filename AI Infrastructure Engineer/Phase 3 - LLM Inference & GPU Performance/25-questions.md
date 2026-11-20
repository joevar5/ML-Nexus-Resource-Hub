# The 25 Questions: Where Each Is Answered

**Covered** = a lecture note or digest teaches it. **Partial** = the ingredients are there but no single place answers the question end to end. **Gap** = not written yet.

| # | Question | Where | Status |
| :-: | :--- | :--- | :--- |
| 1 | What happens between a request arriving and the first token leaving? | 001 (roofline, scaling book), 007 (Orca, Sarathi) | Partial |
| 2 | Stream over SSE, handle disconnects, cancel in-flight generation | none | **Gap** |
| 3 | Size the KV cache: 70B, 32K context, 100 users (GQA, precision) | 001 digest §5–6, 004 digest §5 (GQA, MLA) | Covered (formulas); worked answer is in the lecture-note math |
| 4 | PagedAttention: the problem it solves | 007 digest §1.2, 003 note 03 | Covered |
| 5 | Why FlashAttention is faster with the same math | 004 digest §1–4, 004 note 01 | Covered |
| 6 | Continuous batching under deadlines vs static batching | 007 digest §1.1, §3 | Covered |
| 7 | 100K-token prompt stalls short chats: chunked prefill | 007 digest §1.3 | Covered |
| 8 | Prefix caching and cache-aware routing | 006 digest §3 and §5, 007 digest §1.4 (RadixAttention) | Covered |
| 9 | Cut inter-token latency without tanking throughput | 004 (speculation), 008 (CUDA graphs), 007 (chunked prefill) | Partial |
| 10 | Speculative decoding: when it helps and hurts | 004 digest §6–7 | Covered |
| 11 | FP8 vs INT8 vs INT4 vs NVFP4: pick one and measure the loss | 005 digest §1–7 | Covered |
| 12 | Pick GPUs: H100 vs H200 vs B200 vs L40S | 001 note 06 and digest §8, §12 | Partial (L40S not covered; some specs unread) |
| 13 | Profile with Nsight: compute, bandwidth, or launch overhead? | 003 digest §1–2 | Covered |
| 14 | GPU at 30% utilization: debug it | 003 digest §4 (checklist), 008 | Partial |
| 15 | Fused CUDA/Triton RMSNorm kernel, benchmarked vs PyTorch | 004 note 02, 002 digest §6 (benchmarking), Project 02 | Partial (kernel code is in the note; exercise 004-05 is the practice) |
| 16 | Observability: TTFT, TPOT, queue depth, KV use, hit rate, cost | 001 digest §7 (Etalon), 007 digest §3 | Partial |
| 17 | p99 is 10× p50: diagnose it | 003 (method), 001 digest §7 | Partial |
| 18 | vLLM vs TensorRT-LLM vs SGLang: benchmark and pick | 007 digest §1, §5 (MLPerf), Project 03 | Partial (TensorRT-LLM not read) |
| 19 | Safe rollout of a model or quantization, with rollback | 007 note 02, 007 digest §4 | Partial |
| 20 | Autoscaling GPUs with slow cold starts | 006 digest §4 (ServerlessLLM, Clockwork), §5 | Covered |
| 21 | Multi-tenant serving: fairness, admission control, priorities | 005 digest §9 (VTC), 006 digest §3 (early rejection), §5 (priority) | Covered |
| 22 | Disaggregate prefill and decode: why, KV transfer, cost | 006 digest §3 (DistServe, Splitwise, Mooncake), §5 (NIXL) | Covered |
| 23 | Serve a large MoE: TP vs PP vs EP, all-to-all | 006 digest §1–2 (Megatron, DeepSeek-V3, DeepEP, EPLB) | Covered |
| 24 | Cut cost per million tokens by 70% | 007 note 03, 006 digest §3 | Partial |
| 25 | Design ChatGPT-style inference for 1M daily users | none end to end | **Gap** (the building blocks are in 006 and 007) |

**Read this as:** 10 questions are covered, 13 are partial, and 2 are open gaps (2 and 25).

<details>
<summary>The 25 questions in plain list form</summary>

1. What happens between a request arriving and the first token leaving?
2. Stream over SSE, handle disconnects, cancel in-flight generation.
3. Size the KV cache for a 70B model at 32K context and 100 users (GQA, precision).
4. PagedAttention: what problem does it solve?
5. Why is FlashAttention faster when the math is the same?
6. Continuous batching under latency deadlines, and why it beats static batching.
7. A 100K-token prompt stalls short chats. Fix TTFT with chunked prefill and scheduling.
8. Prefix caching and cache-aware routing across GPU replicas.
9. Cut inter-token latency without tanking throughput.
10. Speculative decoding: when it helps and when it hurts.
11. Quantization: FP8 vs INT8 vs INT4 vs NVFP4. Pick one and measure the loss.
12. Pick GPUs for a workload: H100 vs H200 vs B200 vs L40S.
13. Profile a slow model with Nsight: compute, bandwidth, or launch overhead?
14. GPU at 30% utilization under load. Debug it.
15. Write a fused CUDA/Triton RMSNorm kernel and benchmark it against PyTorch.
16. Observability: TTFT, TPOT, queue depth, KV utilization, prefix-cache hit rate, cost per request.
17. p99 is 10× p50. Diagnose it.
18. vLLM vs TensorRT-LLM vs SGLang: benchmark on real traffic and pick one.
19. Roll out a new model or quantization safely, with rollback.
20. Autoscaling for GPUs with slow cold starts.
21. Multi-tenant serving: fair scheduling, admission control, priority queues.
22. Disaggregate prefill and decode: why, how to transfer KV cache, what it costs.
23. Serve a large MoE model: TP vs PP vs EP, and all-to-all overhead.
24. Cut cost per million tokens by 70% without hurting quality.
25. Design ChatGPT-style inference for 1M daily users: peak QPS, GPU capacity, cost.

</details>

Back to the [Phase 3 learning plan](phase-3-learning-plan.md).
