# Module 06 — Quiz Answers

| # | Answer | Rationale |
|---|---|---|
| 1 | **a** | TP requires high-bandwidth interconnect (NVLink) on every layer. Only worth the comm overhead when the model can't fit on one GPU. |
| 2 | **b** | PP saves activation-transfer bandwidth but creates "bubbles" at pipeline boundaries where some stages idle. |
| 3 | **c** | 3D = TP (within node) + PP (across nodes) + DP (across replicas). Required for 100B+ models on multi-node clusters. |
| 4 | **d** | GPU jobs typically run at <10% CPU; CPU-based HPA never triggers. Use queue-based metrics instead. |
| 5 | **a** | `vllm:num_requests_waiting` is the canonical metric; tokens/s also works. Both reflect actual saturation. |
| 6 | **b** | Node provisioning + image pull + weight download + weight load + warmup add up to several minutes on a cold node; local weight caches and warm pools cut it to under a minute. |
| 7 | **c** | RR distributes request *count* evenly. LLM requests vary 10-100× in cost. Least-loaded or token-aware routing is needed. |
| 8 | **d** | Routing a shared prefix to the replica that already holds it skips that prefill and cuts TTFT; pure hashing can overload one replica, so real routers also weigh load. |
| 9 | **a** | Headroom only has to cover the growth during one cold start. Faster starts (Lecture 02) mean less idle capacity to pay for. |
| 10 | **b** | TP costs comm overhead. A 7B model that fits on one GPU should be replicated horizontally; you get redundancy + throughput without the comm tax. |
