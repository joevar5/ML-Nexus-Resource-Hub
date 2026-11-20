# Source Digest — Distributed Inference

An **advanced layer** on top of the lecture notes. What the papers and docs actually say, as steps and numbers. Read from the source text unless marked *(not read)*.

---

## 1. Tensor parallelism: how the matrix is really split
**Source:** [Megatron-LM](https://arxiv.org/abs/1909.08053)

1. **Approach:** simple intra-layer model parallelism, "a few communication operations in native PyTorch", no new compiler; orthogonal to pipeline parallelism.
2. **MLP block:** split the first weight matrix **A by columns** (A = [A₁, A₂]) so each GPU computes its own slice with no communication (the nonlinearity is applied independently per slice); split the second matrix **by rows**. Then **one all-reduce in the forward pass** (operator *g*) and one in the backward (operator *f*).
3. **Attention:** the Q, K, V projections are split **column-parallel** (so each GPU owns a subset of heads), followed by a row-parallel output projection and an all-reduce. So a full transformer layer needs **two forward all-reduces** (attention + MLP).
4. **Scale reported:** 8.3B parameters on **512 GPUs**, **15.1 PFLOPs** sustained, **76% scaling efficiency** vs a 39-TFLOP single-GPU baseline (30% of peak).
5. **Takeaway:** TP cost = all-reduces on the critical path every layer. That is why it stays inside a fast NVLink domain.

---

## 2. Expert parallelism and real MoE deployments
**Sources:** [GShard](https://arxiv.org/abs/2006.16668), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [MegaScale-Infer](https://arxiv.org/abs/2504.02263), [DeepEP](https://github.com/deepseek-ai/DeepEP), [EPLB](https://github.com/deepseek-ai/EPLB)

**GShard:**
1. A set of **annotation APIs plus an XLA compiler extension**: you annotate how tensors are sharded and the compiler partitions the program.
2. MoE Transformer with **top-2 gating**, auxiliary balancing loss, and per-expert capacity.
3. Scaled a multilingual translation MoE to **600B+ parameters**, trained on **2,048 TPU v3 in 4 days**.

**DeepSeek-V3 deployment (the clearest real-world recipe, read from §3.4):**
- Model: **671B total, 37B active per token**; auxiliary-loss-free load balancing.
- **Prefill unit:** **4 nodes / 32 GPUs**. Attention: **TP4 + sequence parallelism, with DP8**. MoE: **EP32** so each expert sees a large enough batch. All-to-all: tokens go across nodes by **InfiniBand**, then forward among GPUs inside the node over **NVLink**. **32 redundant experts** (each GPU hosts its original 8 plus 1 extra). Two micro-batches with similar workloads are processed together so one's compute **hides the other's all-to-all**.
- **Decode unit:** **40 nodes / 320 GPUs**. Attention: TP4 + SP with **DP80**. MoE: **EP320, one expert per GPU**, with 64 GPUs hosting redundant and shared experts. The shared expert is treated as a routed one, so **each token selects 9 experts**. All-to-all uses **direct IB point-to-point** and **IBGDA** for low latency.
- Redundant-expert sets are **recomputed periodically** from observed load.

**EPLB (the load-balancing algorithm):**
1. Problem: expert load varies, so some GPUs overload.
2. **Duplicate heavily loaded experts**, then heuristically pack replicas onto GPUs; keep experts of one group on the same node to cut inter-node traffic.
3. **Hierarchical policy** (when nodes divide expert groups): pack groups onto nodes, replicate within node, pack onto GPUs. Suits **prefill** (smaller EP). **Global policy** otherwise: replicate globally. Suits **decode** (large EP).
4. Input: estimated per-layer expert loads (e.g. moving average). Output: `phy2log`, `log2phy`, `logcnt` placement tensors.

**DeepEP (the communication kernels):**
1. GPU all-to-all kernels for MoE **dispatch** (send tokens to experts) and **combine** (return and reduce).
2. Intra-node needs **NVLink**; inter-node needs **RDMA** (via NCCL's GIN backend); InfiniBand virtual lanes recommended for traffic isolation.
3. Dispatch accepts **FP8** input with scale factors; combine takes **BF16**.
4. Prefill uses CPU sync for exact output sizes; **decode uses `do_cpu_sync=False`** with pre-allocated capacity and GPU-side counts (so CUDA graphs work).
5. Needs Hopper or newer, CUDA 13.1+. *(No bandwidth numbers on the README.)*

**MegaScale-Infer:**
1. In MoE inference the FFNs become **memory-intensive**, lowering GPU utilization.
2. It **disaggregates attention and FFN modules** within each layer so each scales and is parallelized independently, even on different hardware.
3. **Ping-pong pipeline parallelism:** split the batch into micro-batches and shuttle them between attention and FFN nodes to hide communication.
4. Custom **M2N communication library** (no GPU→CPU copies, no group-init overhead, no extra syncs).
5. Result: **up to 1.90× higher per-GPU throughput** than state of the art.

---

## 3. Prefill/decode disaggregation: three designs
**Sources:** [DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677), [Mooncake](https://arxiv.org/abs/2407.00079)

**DistServe (optimize goodput):**
1. Colocating prefill and decode causes **interference** and **couples** their resource and parallelism choices.
2. Assign prefill and decode to **different GPUs**; **co-optimize** each phase's allocation and parallelism for the application's TTFT and TPOT targets; place phases according to cluster **bandwidth** to minimize KV transfer.
3. Metric: **per-GPU goodput** = max request rate served within *both* SLOs (lower cost per query).
4. Result: **7.4× more requests or 12.6× tighter SLO**, staying within latency constraints for >90% of requests.

**Splitwise (optimize cost and power):**
1. Prompt phase is **compute-intensive**, token phase is **memory-intensive** with different power and cost profiles. Token generation does not need the newest GPUs.
2. Split the phases onto **separate machines**, possibly **different hardware** (homogeneous and heterogeneous clusters). Transfer request state over fast back-end interconnect (InfiniBand) with optimized libraries.
3. Result: up to **1.4× higher throughput at 20% lower cost**, or **2.35× more throughput** at the same power and cost.

**Mooncake (KV-centric; serves Kimi):**
1. Separate prefill and decode clusters **plus a disaggregated KV cache pool** built from the cluster's idle **CPU, DRAM and SSD**.
2. The **Conductor** scheduler picks a prefill/decode pair per request, considering KV reuse and waiting time, and decides **replication and swapping** of KV blocks: hot blocks are copied to several nodes to avoid fetch congestion; cold ones are evicted.
3. **Prediction-based early rejection** under overload (don't start requests you will fail).
4. Result: up to **525% throughput** in some simulated long-context scenarios; **75% more requests** handled on real Kimi workloads, within SLOs.

---

## 4. Fast starts and predictable latency
**Sources:** [ServerlessLLM](https://arxiv.org/abs/2401.14351), [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)

**ServerlessLLM:**
1. Use the near-GPU **storage and memory** of inference servers for local checkpoints.
2. Three contributions: (i) **loading-optimized checkpoint format + multi-tier loading** to use the bandwidth of the storage hierarchy; (ii) **live migration**: the source server migrates **only the tokens**, not the large KV cache, and the destination **recomputes the KV cache** (cheap on network); (iii) **startup-time-optimized scheduling** that evaluates each server's checkpoint locality.
3. Result: latency reduced **10–200×** vs state-of-the-art serverless systems.

**Clockwork:**
1. Insight: DNN inference execution times are **deterministic**, not fundamentally unpredictable.
2. Build the system **bottom-up** on predictable per-inference times to get predictable end-to-end performance (central, fully distributed design).
3. Result: thousands of models while meeting **100 ms** targets for **99.9999%** of requests, with request-level isolation and tight SLOs.

---

## 5. Routing and Kubernetes stacks
**Sources:** [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/), [llm-d](https://github.com/llm-d/llm-d), [Dynamo](https://github.com/ai-dynamo/dynamo), [NIXL](https://github.com/ai-dynamo/nixl)

**Gateway API Inference Extension:**
1. A proxy/load balancer plus an **Endpoint Picker (EPP)** that picks the best endpoint per request using metrics from model servers.
2. Routes by **model name** (including LoRA fine-tunes), supports **traffic splitting** for gradual model rollouts, and **priority** (latency-sensitive chat over latency-tolerant summarization).
3. Request flow: gateway picks an **InferencePool** → forwards request info to the pool's EPP → EPP gathers endpoint metrics → tells gateway which endpoint.
4. *The page names prefix-cache status and LoRA-adapter availability as example signals; it does **not** explicitly list queue length or KV utilization* (so an earlier lecture note's wording about those is a reasonable inference, not a quote).

**llm-d (CNCF sandbox; Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA):**
1. Prefill/decode disaggregation and **wide expert parallelism** (DeepSeek-R1, GPT-OSS).
2. **KV-cache-aware routing**: prefix-cache and load-aware (experimental predicted-latency scheduling).
3. **Tiered KV cache** offloaded to CPU or disk with a global index.
4. **SLO-aware autoscaling** (scale-to-zero in v0.5); multi-tenant flow control; batch API.
5. Builds on **vLLM and SGLang** and Kubernetes.

**Dynamo (NVIDIA):**
1. Orchestration layer *above* engines (SGLang, TensorRT-LLM, vLLM); not an engine.
2. **Router** (load + KV overlap), **KV Block Manager** (offload GPU→CPU→SSD→remote), **Planner** (SLA-driven autoscaler), disaggregated prefill/decode pools.
3. Vendor-reported claims (configurations not shown): 7× per-GPU throughput (DeepSeek-R1, GB200 NVL72 vs B200); 2× faster TTFT with KV-aware routing (Qwen3-Coder 480B); 80% fewer SLA breaches with Planner.

**NIXL:** point-to-point transfer library with plug-in backends (UCX primary; GDS, GPUNETIO, LIBFABRIC, POSIX, object stores, MOONCAKE, and others) over CPU/GPU memory and storage; Linux only. *(API details not on the README.)*

---

## 6. Load test with real bursts
**Source:** [BurstGPT](https://github.com/HPMLL/BurstGPT)

1. Real traces of **ChatGPT (GPT-3.5) and GPT-4** usage on Azure, ~**5.3M lines per file** across **110–121 days**.
2. Fields: timestamp, session ID (conversation logs), elapsed time, model, request tokens, response tokens, total tokens, log type (conversation vs API). Versions with and without failed (zero-response) requests.
3. *The README shows periodicity figures but does not state burstiness statistics or fitted distributions; those are in the linked KDD '25 paper (not read).*

---

## Not read
[NCCL repo](https://github.com/NVIDIA/nccl), [Multi-node NVLink Systems Tuning Guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) (the landing page only lists sections: system, networking, UCX, NCCL, power/thermals, benchmarking), [Ultra Ethernet spec](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf), [ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf), [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/).
