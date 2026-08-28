# Module 10: LLM Infrastructure — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-10-llm-infrastructure/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the primary difference between LLM infrastructure and traditional ML infrastructure?

**Answer:** B) LLMs require massive GPU memory and specialized serving optimizations

**Explanation:**
LLMs typically range from 7B to 70B+ parameters, which forces specialized GPU infrastructure and memory optimizations such as PagedAttention, KV cache management, and continuous batching. Traditional ML models (random forests, small neural nets) rarely have these constraints, so the serving stack looks fundamentally different.

**Common Mistakes:**
- Choosing A: LLMs almost never run on CPUs for production inference because of memory-bandwidth requirements.
- Choosing D: Cloud-hosted LLMs are extremely common; on-prem is just one option, not a defining trait.

**Related Material:** lessons/mod-10-llm-infrastructure/01-introduction-llm-infrastructure.md

---

## Question 2
**Q:** Which deployment pattern is most cost-effective for low-traffic LLM applications?

**Answer:** B) Serverless inference with auto-scaling to zero

**Explanation:**
Low-traffic workloads spend most of their time idle, so paying per-second only when requests arrive is the cheapest model. Scale-to-zero patterns (e.g., SageMaker Serverless, Modal, RunPod Serverless) avoid the wasted spend of an always-on GPU.

**Common Mistakes:**
- Choosing A: A dedicated 24/7 GPU cluster maximizes idle cost, the opposite of cost-effective for sporadic traffic.
- Choosing D: CPU-only inference is too slow and produces poor unit economics even when "cheap" per hour.

**Related Material:** lessons/mod-10-llm-infrastructure/01-introduction-llm-infrastructure.md

---

## Question 3
**Q:** What is the typical GPU memory requirement for serving a 7B parameter LLM in FP16 precision?

**Answer:** C) 14-16 GB

**Explanation:**
FP16 uses 2 bytes per parameter, so 7B × 2 bytes ≈ 14 GB just for weights. Add KV cache, activations, and framework overhead, and you land in the 14–16 GB range — which is why T4 (16 GB) is marginal and A10G (24 GB) is comfortable.

**Common Mistakes:**
- Choosing B (7-10 GB): This confuses parameter count (7B) with byte count; FP16 doubles the storage.
- Choosing D (40+ GB): That number applies to much larger models like 30B+ in FP16 or unquantized 70B.

**Related Material:** lessons/mod-10-llm-infrastructure/01-introduction-llm-infrastructure.md

---

## Question 4
**Q:** Which factor has the MOST impact on LLM inference latency?

**Answer:** C) GPU memory bandwidth and KV cache management

**Explanation:**
Autoregressive decoding is memory-bandwidth-bound: every generated token requires streaming the full model weights and the KV cache through the GPU's memory hierarchy. CPU, storage IOPS, and network are rarely the bottleneck once the model is loaded.

**Common Mistakes:**
- Choosing A: CPU speed barely matters once tensors are on the GPU.
- Choosing D: Storage IOPS only matters during model load, not per-token latency.

**Related Material:** lessons/mod-10-llm-infrastructure/01-introduction-llm-infrastructure.md

---

## Question 5
**Q:** What is PagedAttention in vLLM?

**Answer:** B) A technique for managing KV cache memory using virtual memory paging

**Explanation:**
PagedAttention borrows the OS virtual-memory paging concept and applies it to KV cache blocks. This eliminates fragmentation, lets sequences share blocks (e.g., for prefix caching), and dramatically increases batch sizes — the core reason vLLM achieves high throughput.

**Common Mistakes:**
- Choosing A: PagedAttention does not change the model's size or weights.
- Choosing D: It is a memory management technique, not a GPU scheduler.

**Related Material:** lessons/mod-10-llm-infrastructure/02-vllm-deployment.md

---

## Question 6
**Q:** Which API compatibility does vLLM provide out of the box?

**Answer:** C) OpenAI API compatibility

**Explanation:**
vLLM ships an OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, streaming) so existing OpenAI SDK clients can point at a self-hosted vLLM endpoint with only a base URL change.

**Common Mistakes:**
- Choosing A: vLLM does not expose a Hugging Face Inference API surface.
- Choosing D: AWS Bedrock has its own SigV4-signed API, which vLLM does not emulate.

**Related Material:** lessons/mod-10-llm-infrastructure/02-vllm-deployment.md

---

## Question 7
**Q:** What does the `--max-model-len` parameter in vLLM control?

**Answer:** B) Maximum context length for inputs and outputs

**Explanation:**
`--max-model-len` caps the total token length per sequence (prompt + generation). It is a key memory-budgeting knob that interacts with `--max-num-seqs` to determine batch size and KV cache footprint.

**Common Mistakes:**
- Choosing A: Model parameter count is fixed by the checkpoint, not a CLI flag.
- Choosing C: That role is played by `--max-num-seqs`, not `--max-model-len`.

**Related Material:** lessons/mod-10-llm-infrastructure/02-vllm-deployment.md

---

## Question 8
**Q:** When deploying vLLM on Kubernetes, what is the recommended resource request pattern?

**Answer:** B) Request and limit should be equal for GPUs

**Explanation:**
GPUs in Kubernetes are exclusive, non-overcommittable resources. The device plugin only honors integer counts at both request and limit, so they must be equal (e.g., `nvidia.com/gpu: 1` on both). Mismatched values either fail validation or behave the same as equal values.

**Common Mistakes:**
- Choosing A: GPU overcommit is not supported by the standard NVIDIA device plugin.
- Choosing C: Omitting limits causes scheduling and quota problems and is not a valid GPU pattern.

**Related Material:** lessons/mod-10-llm-infrastructure/02-vllm-deployment.md

---

## Question 9
**Q:** What is the primary purpose of Retrieval-Augmented Generation (RAG)?

**Answer:** C) Augment LLM responses with relevant context from external knowledge bases

**Explanation:**
RAG injects retrieved, up-to-date, or proprietary content into the prompt at inference time, addressing the training-cutoff and missing-private-data limitations of base LLMs without retraining.

**Common Mistakes:**
- Choosing A: RAG is an inference-time pattern; it has no effect on training speed.
- Choosing B: RAG does not shrink the model; it adds a retrieval system around it.

**Related Material:** lessons/mod-10-llm-infrastructure/03-rag-systems.md

---

## Question 10
**Q:** In a RAG pipeline, what is "chunking"?

**Answer:** B) Dividing documents into smaller, semantically meaningful segments

**Explanation:**
Chunking splits source documents into retrieval-sized units (typically 200–1000 tokens) so they fit embedding-model windows and provide focused context. Strategy choices — fixed-size, sentence-aware, recursive — directly affect retrieval quality.

**Common Mistakes:**
- Choosing A: Compressing weights is quantization, not chunking.
- Choosing D: Splitting GPU workload is sharding / tensor parallelism, a completely different concern.

**Related Material:** lessons/mod-10-llm-infrastructure/03-rag-systems.md

---

## Question 11
**Q:** What is the purpose of the reranking step in advanced RAG systems?

**Answer:** B) Improve retrieval precision by reordering candidates with a more sophisticated model

**Explanation:**
A bi-encoder retrieves a broad candidate set quickly, then a cross-encoder reranker scores each (query, candidate) pair jointly for far more accurate relevance ranking. The result is much higher precision at the top of the context window.

**Common Mistakes:**
- Choosing A: Alphabetical sorting has nothing to do with relevance.
- Choosing D: Dedup is a separate post-processing step, not reranking.

**Related Material:** lessons/mod-10-llm-infrastructure/03-rag-systems.md

---

## Question 12
**Q:** Which embedding model characteristic is MOST important for RAG retrieval quality?

**Answer:** C) Semantic similarity between query and document embeddings

**Explanation:**
Retrieval quality is ultimately defined by how well the embedding space clusters semantically related queries and documents. Benchmarks like MTEB/NDCG measure exactly this. Size and speed are secondary trade-offs once quality is acceptable.

**Common Mistakes:**
- Choosing A: A large model is not automatically better; many small models top retrieval benchmarks.
- Choosing B: Inference speed matters for cost/latency, not retrieval quality.

**Related Material:** lessons/mod-10-llm-infrastructure/03-rag-systems.md

---

## Question 13
**Q:** What is the primary data structure used by vector databases for similarity search?

**Answer:** C) Approximate Nearest Neighbor (ANN) indices like HNSW or IVF

**Explanation:**
Brute-force search over millions of high-dimensional vectors is too slow, so vector databases rely on ANN structures — HNSW (graph-based), IVF (inverted-list), PQ (product quantization), and combinations — to achieve sub-linear search with tunable recall.

**Common Mistakes:**
- Choosing A: B-trees are designed for 1-D ordered data, not high-dimensional similarity.
- Choosing B: Hash tables provide exact lookup, not approximate nearest neighbors.

**Related Material:** lessons/mod-10-llm-infrastructure/04-vector-databases.md

---

## Question 14
**Q:** In Qdrant, what is a "collection"?

**Answer:** B) A named set of vectors with the same dimensionality and configuration

**Explanation:**
A Qdrant collection is the top-level container — analogous to a SQL table — that fixes the vector dimensionality, distance metric, and HNSW configuration for all points it holds, along with their payload metadata.

**Common Mistakes:**
- Choosing A: Users are managed at the cluster/auth layer, not as collections.
- Choosing C: Backups are snapshots, not collections.

**Related Material:** lessons/mod-10-llm-infrastructure/04-vector-databases.md

---

## Question 15
**Q:** What is the trade-off when increasing the HNSW `ef_construct` parameter?

**Answer:** A) Higher accuracy but slower indexing

**Explanation:**
`ef_construct` is the size of the candidate list explored when inserting each point into the HNSW graph. Larger values build a higher-quality graph (better recall at search time) but increase indexing CPU cost. It does not directly change query-time `ef`.

**Common Mistakes:**
- Choosing B: Reverses the actual trade-off direction.
- Choosing D: "Always increase it" ignores the very real indexing-time cost.

**Related Material:** lessons/mod-10-llm-infrastructure/04-vector-databases.md

---

## Question 16
**Q:** Which vector database feature is critical for production RAG systems?

**Answer:** B) Filtering on metadata while maintaining vector search performance

**Explanation:**
Production RAG needs multi-tenant isolation, permission scoping, freshness windows, document-type filters, etc. Without efficient filtered ANN search (pre-filter or filterable HNSW), accuracy and tenant separation collapse.

**Common Mistakes:**
- Choosing A: Vector DBs are not LLM hosts; that is the inference layer's job.
- Choosing C: Vector DBs do not train models.

**Related Material:** lessons/mod-10-llm-infrastructure/04-vector-databases.md

---

## Question 17
**Q:** What does LoRA (Low-Rank Adaptation) do?

**Answer:** B) Adds trainable low-rank matrices to frozen pre-trained weights

**Explanation:**
LoRA freezes the base weights `W` and trains small rank-`r` matrices `A` and `B` such that the effective weight is `W + BA`. This shrinks trainable parameters by ~1000x, making fine-tuning feasible on a single GPU while preserving base capabilities.

**Common Mistakes:**
- Choosing A: Compression for inference is quantization/distillation, not LoRA.
- Choosing D: LoRA enables efficient fine-tuning but does not automatically improve accuracy.

**Related Material:** lessons/mod-10-llm-infrastructure/05-llm-fine-tuning-infrastructure.md

---

## Question 18
**Q:** What is the main advantage of QLoRA over LoRA?

**Answer:** C) Quantizes base model to 4-bit while training LoRA adapters in higher precision

**Explanation:**
QLoRA stores the frozen base model in NF4 (4-bit), dequantizes on the fly for the forward pass, and keeps LoRA adapters in BF16. The result is ~4x lower base-model memory, enabling 65B fine-tuning on 24–48 GB GPUs.

**Common Mistakes:**
- Choosing A: QLoRA is typically slower than plain LoRA per step due to dequantization.
- Choosing B: QLoRA does not inherently improve accuracy versus LoRA.

**Related Material:** lessons/mod-10-llm-infrastructure/05-llm-fine-tuning-infrastructure.md

---

## Question 19
**Q:** When fine-tuning an LLM, what is "catastrophic forgetting"?

**Answer:** B) Model forgetting its pre-trained knowledge when overfitting to fine-tuning data

**Explanation:**
Aggressive full fine-tuning shifts weights so far toward the new distribution that general pre-training capabilities degrade. Mitigations include LoRA, lower learning rates, replay buffers of general data, and shorter training runs.

**Common Mistakes:**
- Choosing A: GPU OOM is an infrastructure error, not a learning phenomenon.
- Choosing C: Lost checkpoints are an ops issue, unrelated to model behavior.

**Related Material:** lessons/mod-10-llm-infrastructure/05-llm-fine-tuning-infrastructure.md

---

## Question 20
**Q:** What is the effect of quantizing an LLM from FP16 to INT8?

**Answer:** A) 2x memory reduction with minimal quality loss

**Explanation:**
FP16 uses 16 bits per value, INT8 uses 8, giving a clean 2x reduction. Modern schemes (LLM.int8(), SmoothQuant, GPTQ-8bit) keep quality regression in the low single digits and unlock INT8 tensor cores on supported GPUs.

**Common Mistakes:**
- Choosing B (4x): That figure corresponds to FP16 → INT4, not INT8.
- Choosing D: INT8 quantization is widely deployed and very much usable.

**Related Material:** lessons/mod-10-llm-infrastructure/06-llm-serving-optimization.md

---

## Question 21
**Q:** What is "continuous batching" in LLM serving?

**Answer:** C) Dynamically adding new requests to a batch as others complete

**Explanation:**
Also called iteration-level scheduling, continuous batching swaps finished sequences out of the batch each decode step and slots new ones in. This eliminates head-of-line blocking from variable-length generations and is the largest throughput win in modern LLM servers.

**Common Mistakes:**
- Choosing A: That describes streaming training data, not serving.
- Choosing B: Batching identical prompts is prefix sharing, not continuous batching.

**Related Material:** lessons/mod-10-llm-infrastructure/06-llm-serving-optimization.md

---

## Question 22
**Q:** What is Flash Attention designed to optimize?

**Answer:** B) Memory access patterns during attention computation

**Explanation:**
Flash Attention is an IO-aware exact attention algorithm that tiles QKV in SRAM and avoids materializing the full N×N attention matrix in HBM. The result is the same math with far less memory traffic and significantly faster wall-clock time.

**Common Mistakes:**
- Choosing A: Flash Attention does not compress the model.
- Choosing D: It is a single-GPU kernel optimization, not a networking technique.

**Related Material:** lessons/mod-10-llm-infrastructure/06-llm-serving-optimization.md

---

## Question 23
**Q:** In a multi-model LLM platform, what is the purpose of a model router?

**Answer:** B) Route requests to appropriate models based on task, cost, or latency requirements

**Explanation:**
A model router classifies each request and dispatches it to the right model — small/fast for trivial chat, large/slow for hard reasoning, code-specialized for code tasks, etc. This balances quality, latency, and cost across a heterogeneous model fleet.

**Common Mistakes:**
- Choosing A: GPU load balancing is the job of the service mesh / replica scheduler, not the model router.
- Choosing C: Network-layer routing is a layer-3/4 concern, not a model platform concern.

**Related Material:** lessons/mod-10-llm-infrastructure/07-llm-platform-architecture.md

---

## Question 24
**Q:** What is the primary benefit of implementing semantic caching in LLM systems?

**Answer:** B) Reduced infrastructure costs by serving cached responses for similar queries

**Explanation:**
Semantic caching keys on embedding similarity rather than exact strings, so paraphrased queries hit the cache. For workloads with repeating intent (FAQs, support, common prompts), this typically eliminates 30–70% of LLM calls — direct cost and latency wins.

**Common Mistakes:**
- Choosing A: The cache helps inference latency, not model load time.
- Choosing C: Cached responses cannot improve model accuracy; they reuse prior outputs.

**Related Material:** lessons/mod-10-llm-infrastructure/07-llm-platform-architecture.md

---

## Question 25
**Q:** What should be monitored FIRST when an LLM service shows degraded performance?

**Answer:** B) GPU utilization, memory, and queue depth

**Explanation:**
LLM serving is GPU-bound, so the first triage step is the GPU resource triangle: utilization (compute saturation), memory (OOM and KV cache pressure), and queue depth (admission backlog). These pinpoint capacity vs. workload issues within seconds.

**Common Mistakes:**
- Choosing A: Code quality metrics have no runtime signal for live degradation.
- Choosing D: Databases are rarely the bottleneck for raw LLM inference (RAG aside).

**Related Material:** lessons/mod-10-llm-infrastructure/08-production-llm-best-practices.md

---
