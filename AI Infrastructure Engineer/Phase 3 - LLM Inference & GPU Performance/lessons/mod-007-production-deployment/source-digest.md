# Source Digest — Production Deployment

An **advanced layer** on top of the lecture notes. What the papers actually say, as steps and numbers. Read from the source text unless marked *(not read)*.

---

## 1. How serving engines evolved: four papers in order

### 1.1 Orca: schedule per iteration
**Source:** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI '22)
1. **Problem:** existing systems batch *requests*; a request that finishes early can't return, and a new request must wait for the whole batch.
2. **Iteration-level scheduling:** the scheduler calls the engine for **one iteration** at a time, so finished requests leave and new ones join every step.
3. **Selective batching:** apply batching only to a chosen set of operations (the ones that work with ragged lengths), so batching and iteration-level scheduling can coexist in a Transformer; attention is handled per request.
4. **Result:** **36.9× throughput** over NVIDIA FasterTransformer at the same latency (the abstract's headline figure; the model and setup were not extracted).

### 1.2 vLLM / PagedAttention: fix KV memory waste
**Source:** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
1. **Measured waste:** in existing systems only **20.4%–38.2%** of KV-cache memory stores actual token states (the rest is reservation and fragmentation).
2. **Idea:** borrow virtual memory and paging from operating systems. KV is stored in fixed-size **blocks** mapped through a block table; allocate on demand.
3. **Sharing:** blocks can be shared within and across requests; **copy-on-write** when a shared block must change (parallel sampling, beam search).
4. **Block size trade-off:** a larger block size also increases fragmentation (studied in §7.2).
5. **Result:** near-zero waste; **2–4× throughput** at the same latency vs FasterTransformer and Orca, more with longer sequences, bigger models and complex decoding (on ShareGPT it sustains 1.7×–2.7× higher request rates than Orca variants).

### 1.3 Sarathi-Serve: end prefill stalls
**Source:** [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
1. **Tension:** prefill iterations are slow but saturate compute; decode iterations are fast but under-use compute, so batching helps decode. Interleaving them hurts either throughput or latency.
2. **Chunked prefill:** split a prefill into **near-equal chunks**.
3. **Stall-free scheduling:** add new work into a batch **without pausing ongoing decodes**; uniform batches also reduce **pipeline bubbles**.
4. **Result vs vLLM:** **2.6×** higher serving capacity (Mistral-7B, one A100), **3.7×** (Yi-34B, two A100), up to **5.6×** (Falcon-180B with pipeline parallelism), under tail-latency constraints.

### 1.4 SGLang: reuse and programs
**Source:** [SGLang](https://arxiv.org/abs/2312.07104)
1. Frontend language for multi-call LLM programs + runtime.
2. **RadixAttention:** keep an **LRU cache of KV for all requests in a radix tree**: manage KV like a traditional cache with efficient matching, insertion and eviction, so shared prefixes are reused automatically.
3. **Compressed finite-state machines** for faster structured (JSON) decoding.
4. **Result:** up to **6.4× higher throughput** on agents, reasoning, few-shot, JSON decoding, RAG and multi-turn chat.

---

## 2. Structured output without slowing decoding
**Source:** [XGrammar](https://arxiv.org/abs/2411.15100)
1. Problem: running a context-free grammar means checking stack states against **every token in the vocabulary** at each step.
2. **Split the vocabulary:** **context-independent** tokens can be **pre-checked** and cached; only **context-dependent** tokens are interpreted at runtime.
3. Extras: grammar transformations to expand context and shrink the runtime set; a **persistent stack** for fast checks; **overlap grammar computation with GPU execution**.
4. **Result:** up to **100× speedup** over existing solutions; near-zero overhead end-to-end.
5. Mechanism: a **pushdown automaton** parses the prior output to define valid next tokens.

---

## 3. Define success: goodput and deadlines
**Sources:** [Etalon](https://arxiv.org/abs/2407.07000), [DistServe](https://arxiv.org/abs/2401.09670)
- **Goodput:** max request rate served within **both** TTFT and TPOT SLOs (DistServe), giving lower cost per query.
- **Fluidity-index:** each token has a deadline `D_i = D_p + i·D_d`; score = fraction met (Etalon). Full details in the [Module 1 digest §7](../mod-001-gpu-fundamentals/source-digest.md).
- Etalon finding worth remembering: on Llama3-8B/H100, vLLM and Sarathi-Serve had **equal capacity at 0.6 QPS** but Sarathi-Serve had **fewer deadline misses** thanks to chunked prefill. Averages hid it.

---

## 4. Deployment patterns, from the Kubernetes tools
**Sources:** [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/), [llm-d](https://github.com/llm-d/llm-d), [Dynamo](https://github.com/ai-dynamo/dynamo)
- **Model-name routing + traffic splitting** → incremental rollouts of new model versions.
- **Priority** → latency-sensitive vs latency-tolerant models share a pool.
- **SLO-aware autoscaling** (llm-d; Dynamo's Planner) and **scale-to-zero** (llm-d v0.5).
- **Predictable serving:** Clockwork's approach is to *reject* requests it predicts will miss their deadline: [Module 6 digest §4](../mod-006-distributed-inference/source-digest.md).

---

## 5. Benchmarking
**Source:** [MLPerf Inference](https://arxiv.org/abs/1911.02549)
1. Goal: **architecture-neutral, representative, reproducible** comparison across systems spanning orders of magnitude in power and performance.
2. Driven by **30+ organizations and 200+ engineers**; the first round produced **600+ reproducible measurements from 14 organizations** on 30+ systems.
3. Defines rules and best practices so results are comparable; defines load-generation *scenarios* (scenario names and details not extracted; from general knowledge they include single-stream, multi-stream, server and offline).
4. For real LLM traffic realism see BurstGPT in the [Module 6 digest §6](../mod-006-distributed-inference/source-digest.md).

---

## Not read
[ServeGen](https://www.usenix.org/system/files/nsdi26-xiang-servegen.pdf), [MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) repos. Cost-per-token examples in the lecture notes are my arithmetic, not from a source.
