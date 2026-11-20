# Module 07 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **a** | LLM engines bake in continuous batching + paged KV. Generic servers would need months of custom work to match. |
| 2 | **b** | Triton + TorchServe are tuned for stateless request/response with dynamic batching. vLLM's KV-cache machinery is wasted on embedding workloads. |
| 3 | **d** | Canary's value is gating on real metrics. Trivial bugfixes don't need it; schema-breaking changes need blue-green's atomic flip. |
| 4 | **d** | Shadow's defining property: real traffic mirrored, but responses ignored — zero risk to users. |
| 5 | **d** | Spot pricing comes with eviction risk. Batch training that can checkpoint + resume is the canonical workload type that tolerates it. |
| 6 | **d** | Auto-gate metrics must reflect user impact: error rate, latency p95, accuracy delta. Image size + replica count tell you nothing about correctness. |
| 7 | **c** | Tier routing exploits the cost gap between model sizes. Savings depend on the share of traffic the small model can handle at acceptable quality (Ex 04). |
| 8 | **d** | Static batches wait for their longest member; iteration-level scheduling doesn't. Anyscale measured up to 23× in 2023; 5–20× is typical. |
| 9 | **a** | AWQ frees memory (more KV cache → more concurrent requests); speculation helps per-request latency mostly at low batch (Mod 4 digest §7). |
| 10 | **b** | Framework choice is the most consequential decision; all subsequent layers (k8s, observability, cost) depend on it. |
