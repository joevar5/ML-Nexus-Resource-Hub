# Module 06 — Quiz

10 questions. 70% pass.

### 1. Tensor parallelism is the right tool when:
- a) The model doesn't fit on one GPU AND fast interconnect (NVLink) is available
- b) The model fits comfortably on one GPU but you want redundancy
- c) Single-GPU latency is the only concern
- d) Storage I/O is the bottleneck

### 2. Pipeline parallelism vs tensor parallelism:
- a) Always faster than TP
- b) Lower communication cost than TP, but pipeline bubbles add latency
- c) Functionally identical
- d) Never used in production

### 3. 3D parallelism is:
- a) Three independent TP groups
- b) Three different model architectures stacked
- c) Tensor-parallel within a node + pipeline-parallel across nodes + data-parallel across replicas
- d) Random per-layer sharding

### 4. Configuring HPA on CPU utilization for GPU inference is:
- a) The default best practice
- b) Required by the Kubernetes spec
- c) Mandatory for any GPU workload
- d) Poor — CPU is rarely the bottleneck on GPU-saturated serving

### 5. The right autoscaling metric for a vLLM deployment is:
- a) `vllm:num_requests_waiting` (queue depth) or generated tokens/s
- b) CPU utilization
- c) Container memory utilization
- d) Disk I/O latency

### 6. Cold start time for a 70B LLM deployment is typically:
- a) Sub-second
- b) Minutes: node provisioning, image pull, weight download, 30–90 s weight load, then warmup
- c) Several hours
- d) Not measurable in practice

### 7. Round-robin load balancing for LLM serving is:
- a) The recommended best practice
- b) Required by HTTP/2
- c) Poor — request cost varies wildly with generation length
- d) Identical to least-loaded routing

### 8. Prefix-aware routing helps most when:
- a) Always; it's a universal optimization
- b) Only for embedding workloads
- c) Required by Kubernetes service mesh
- d) Many requests share a system prompt — route to the replica with the cached prefix

### 9. How should you size spare GPU capacity (headroom) above current load?
- a) From scale-up time: enough to absorb the traffic growth that can arrive while a new replica starts
- b) Always 200% above peak (run at one-third utilization)
- c) Zero — run at peak to minimize cost
- d) A fixed 10% regardless of workload

### 10. When NOT to tensor-parallel a 7B model:
- a) Always TP regardless of size
- b) When it fits on one GPU; replicate to N pods instead of TP-sharding across N GPUs
- c) Only on H100 hardware
- d) Required for accuracy

Answers: [quiz-answers.md](quiz-answers.md)
