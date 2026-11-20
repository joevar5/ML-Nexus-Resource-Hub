# Module 07 — Quiz

10 questions. 70% pass.

### 1. The best serving framework for an autoregressive LLM:
- a) An LLM engine such as vLLM, SGLang or TensorRT-LLM — continuous batching + paged KV cache
- b) TorchServe — model-based serving for PyTorch
- c) FastAPI alone (with a generation loop)
- d) TensorFlow Serving

### 2. The best serving framework for an embedding model (high-QPS request/response):
- a) vLLM
- b) Triton Inference Server or TorchServe
- c) A bash curl wrapper
- d) Argo Workflows

### 3. Canary deployment fits best for:
- a) Trivial bugfixes with no metric impact
- b) Schema-breaking changes (blue-green is better)
- c) Cron jobs
- d) Risky model changes with measurable success metrics that gate the rollout

### 4. Shadow deployment serves:
- a) 100% real traffic to the new version
- b) 50/50 split between old + new
- c) Internal users only
- d) 0% real traffic; mirror traffic to the candidate; ignore its responses

### 5. Spot instances are appropriate for:
- a) Critical real-time inference
- b) etcd / control plane
- c) Ingress load balancers
- d) Batch training that can checkpoint + resume

### 6. The auto-revert metric on a canary should be:
- a) Image size delta
- b) Replica count
- c) Time of day
- d) Success rate or 5xx rate or a custom signal (model accuracy delta)

### 7. Per-tier routing saves cost by:
- a) Always using the cheapest model
- b) Always using the most expensive model
- c) Sending simple requests to cheap models; escalating to expensive only when needed
- d) Routing randomly across tiers

### 8. Continuous batching's typical speedup over naive padded batching:
- a) Marginal (<10%)
- b) Strictly slower
- c) Required for accuracy (changes outputs)
- d) Often 5–20× higher throughput on variable-length traffic

### 9. Stacking AWQ int4 + speculative decoding:
- a) They combine: smaller weights leave more memory for KV cache, and speculation cuts latency at low concurrency (less at high load)
- b) Cannot be combined (mutually exclusive)
- c) Halves accuracy
- d) Locks the model into fp32

### 10. The right place to start production deployment work is:
- a) Start with HPA configuration
- b) Pick the right serving framework first; then containerize, add observability, attribute cost
- c) Custom FastAPI always
- d) vLLM regardless of workload

Answers: [quiz-answers.md](quiz-answers.md)
