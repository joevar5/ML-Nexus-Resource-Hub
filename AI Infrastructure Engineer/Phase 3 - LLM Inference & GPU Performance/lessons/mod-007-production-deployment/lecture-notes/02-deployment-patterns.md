# Lecture 02: Deployment Patterns

## Rolling, blue-green, canary, shadow

(See engineer-solutions/mod-106 ex-08 for fully-implemented YAML manifests.)

Pick by risk profile:
- Low risk (re-trained version of same architecture): **rolling**
- Schema change or major version bump: **blue-green**
- Risky model change with measurable success metrics: **canary** (Argo Rollouts)
- New model architecture: **shadow** first, then canary

## Auto-revert gates

Argo Rollouts can auto-revert based on:
- 5xx rate during canary
- Latency p95
- Custom metric (e.g., model accuracy delta vs A/B control)

## Companion

- engineer-solutions/mod-106 ex-08 for all 4 strategies with manifests
- engineer-solutions/mod-110 ex-07 for multi-tenant LLM platform

---

## Deep dive: how it works (core logic)

### Define success the way users feel it ([Etalon](https://arxiv.org/html/2407.07000))
Average latency lies. Define an SLO per request, such as TTFT ≤ X ms **and** TPOT ≤ Y ms. **Goodput** = requests per second that meet *all* their SLOs. Etalon adds a token-level view (a *fluidity* measure): each token has a deadline, so a mid-stream stall counts as a violation even if the mean looks fine. Use goodput as the canary pass/fail metric and the autoscaling objective.

### Predictable serving ([Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati))
Because model execution times are stable, a centralized scheduler can *plan* and reject requests that would miss deadlines. Pattern to copy: **admission control based on predicted wait**, not on hope.

### Startup and migration ([ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu))
Fast loading (optimized checkpoint format, local multi-tier storage) and moving live requests by re-sending tokens make blue/green and scale-up less painful.

### Kubernetes routing
- **[Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/):** adds an **endpoint picker** to the gateway. For each request it scores candidate pods using live signals (queue length, KV-cache utilization, loaded LoRA adapters, prefix cache) and picks the best, instead of blind load balancing. It also supports model-name routing and traffic splitting, which you use for **canary/weighted rollouts** between model versions.
- **[llm-d](https://github.com/llm-d/llm-d):** Kubernetes-native distributed serving with this routing plus prefill/decode disaggregation.

### A safe rollout, concretely
1. Offline eval gate (quality + latency). 2. Shadow traffic. 3. Weighted canary (1→5→25→100%) judged on **goodput and quality**. 4. Automatic rollback when a guardrail trips; keep the previous version **warm** so rollback is instant.

### Read the originals

- [Etalon](https://arxiv.org/html/2407.07000)
- [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)
- [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [llm-d](https://github.com/llm-d/llm-d)

Source list: [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) (MIT). The explanations above are a study summary; check the papers for exact figures.
