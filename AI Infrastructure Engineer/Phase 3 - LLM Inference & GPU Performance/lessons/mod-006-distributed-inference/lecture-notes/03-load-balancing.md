# Lecture 03: Load Balancing for LLM Inference

## Why naive LB hurts

Round-robin LB distributes requests evenly by count. For LLM serving, a
request that generates 10 tokens is much cheaper than one generating 2000
tokens. Even-by-count → uneven-by-load → some replicas saturate while others
idle.

## Better strategies

- **Least-loaded**: route to the replica with the lowest current queue depth (vLLM exposes this)
- **Power-of-two-choices**: sample 2 random replicas, pick the less loaded
- **Token-aware**: send long-prompt requests to replicas with more KV-cache headroom

A single vLLM instance does not route between replicas; that is the job of a
router in front of them: the vLLM production-stack router, llm-d, NVIDIA
Dynamo, or an Envoy-based gateway with the Gateway API Inference Extension.

## Routing for prefix caching

If you have prefix caching (mod-004), routing matters more — a request sent to
a replica that already holds its system prompt skips that prefill, so TTFT can
drop several-fold for long shared prefixes.

Use consistent hashing on the prefix: same system prompt → same replica → cache hit.

## Failure modes

- Sticky session to a dead replica: TCP timeouts retry; LB should detect + remove
- Slow-loris: one client tying up a slot for hours; per-tenant concurrency limits
- Thundering herd on cold start: new replica gets 100% of traffic before warmup
  → use slow-start in the LB

---

## Deep dive: how it works (core logic)

### Why not round-robin
Each replica has its own KV/prefix cache, and requests differ hugely in cost (1K vs 100K tokens). Round-robin scatters shared prefixes (cache misses) and ignores load.

### Prefix-aware routing ([SGLang](https://arxiv.org/html/2312.07104))
SGLang's **RadixAttention** stores cached KV in a **radix tree keyed by token sequences**. A new request walks the tree to find the longest matching prefix; only the unmatched tail is prefilled. Leaves are evicted LRU when memory is short. A router can mirror this: keep an approximate per-replica prefix index and score replicas by
```
score = matched_prefix_tokens  −  λ × current_load
```
so you gain cache hits but don't pile onto one hot replica. Production routers ([Dynamo](https://github.com/ai-dynamo/dynamo), [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)) combine prefix match with queue length, KV utilization and adapter presence.

### KV as a cluster-wide resource ([Mooncake](https://www.usenix.org/conference/fast25/presentation/qin))
Mooncake treats KV cache as the central thing to manage. A global scheduler chooses the prefill instance by *cache-match length and queue time*; KV blocks live in a **pooled store across CPU DRAM and SSD** of the whole cluster and move over RDMA, so a cache hit on any node helps. Under overload it also does **early rejection**, dropping requests it predicts will miss deadlines.

### Fair and load-aware
Two simple, effective rules: **power-of-two-choices** (sample two replicas, send to the less loaded) avoids herding without global state; and [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588) (serve the tenant with the lowest weighted-token counter) keeps one tenant from starving the rest.

### Test with realistic arrivals
Uniform traffic hides routing problems. Use bursty traces such as [BurstGPT](https://github.com/HPMLL/BurstGPT) and watch p99 TTFT, cache hit rate and per-replica load imbalance.

### Read the originals

- [SGLang](https://arxiv.org/html/2312.07104)
- [Dynamo](https://github.com/ai-dynamo/dynamo)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)
- [Fairness in Serving Large Language Models](https://arxiv.org/html/2401.00588)
- [BurstGPT](https://github.com/HPMLL/BurstGPT)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
