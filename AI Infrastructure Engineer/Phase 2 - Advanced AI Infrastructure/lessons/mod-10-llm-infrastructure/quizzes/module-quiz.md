# Module 10: LLM Infrastructure - Quiz

## Instructions
- This quiz covers all 8 lessons in Module 10: LLM Infrastructure
- 25 multiple choice questions + 3 practical scenarios
- Passing score: 80% (20/25 correct)
- Estimated time: 60 minutes
- Answer key and explanations provided at the end

---

## Section 1: LLM Infrastructure Fundamentals (Questions 1-4)

### Question 1
What is the primary difference between LLM infrastructure and traditional ML infrastructure?

A) LLMs only run on CPUs
B) LLMs require massive GPU memory and specialized serving optimizations
C) LLMs don't need monitoring
D) LLMs are always deployed on-premise

**Answer: B**

**Explanation:** LLMs typically range from 7B to 70B+ parameters, requiring specialized GPU infrastructure, memory optimization techniques (like PagedAttention), and serving frameworks designed specifically for their massive compute requirements.

### Question 2
Which deployment pattern is most cost-effective for low-traffic LLM applications?

A) Dedicated GPU cluster running 24/7
B) Serverless inference with auto-scaling to zero
C) Multi-GPU distributed serving
D) CPU-only inference

**Answer: B**

**Explanation:** For low-traffic applications, serverless deployments that can scale to zero when not in use prevent wasted GPU costs. High-traffic applications justify dedicated resources, but low traffic benefits from pay-per-use models.

### Question 3
What is the typical GPU memory requirement for serving a 7B parameter LLM in FP16 precision?

A) 2-4 GB
B) 7-10 GB
C) 14-16 GB
D) 40+ GB

**Answer: C**

**Explanation:** A 7B parameter model at FP16 (2 bytes per parameter) requires ~14GB just for weights, plus additional memory for KV cache, activations, and serving overhead. Total requirements are typically 14-16GB, making it suitable for T4 (16GB) or A10G (24GB) GPUs.

### Question 4
Which factor has the MOST impact on LLM inference latency?

A) CPU speed
B) Network bandwidth
C) GPU memory bandwidth and KV cache management
D) Storage IOPS

**Answer: C**

**Explanation:** For autoregressive LLMs, each token generation is memory-bound rather than compute-bound. GPU memory bandwidth determines how quickly the model can load weights and manage the KV cache, which is the primary bottleneck for inference latency.

---

## Section 2: vLLM Deployment (Questions 5-8)

### Question 5
What is PagedAttention in vLLM?

A) A method to reduce model size
B) A technique for managing KV cache memory using virtual memory paging
C) A way to speed up training
D) A GPU scheduling algorithm

**Answer: B**

**Explanation:** PagedAttention is vLLM's key innovation that manages KV cache memory efficiently using virtual memory-style paging. This reduces memory waste and enables higher throughput by allowing more requests to be batched together.

### Question 6
Which API compatibility does vLLM provide out of the box?

A) Hugging Face API only
B) Custom vLLM API only
C) OpenAI API compatibility
D) AWS Bedrock API

**Answer: C**

**Explanation:** vLLM provides OpenAI-compatible API endpoints, making it a drop-in replacement for applications built with the OpenAI SDK. This includes chat completions, completions, and streaming endpoints.

### Question 7
What does the `--max-model-len` parameter in vLLM control?

A) Maximum model size in parameters
B) Maximum context length for inputs and outputs
C) Maximum number of concurrent requests
D) Maximum GPU memory allocation

**Answer: B**

**Explanation:** `--max-model-len` sets the maximum total token length (input + output) that the model will process. This parameter is crucial for memory management and should be balanced with `max-num-seqs` for optimal throughput.

### Question 8
When deploying vLLM on Kubernetes, what is the recommended resource request pattern?

A) Request fewer resources than limits to allow overcommitment
B) Request and limit should be equal for GPUs
C) No resource limits needed
D) Only set requests, never limits

**Answer: B**

**Explanation:** For GPUs in Kubernetes, requests and limits should be identical (e.g., `nvidia.com/gpu: 1` for both). GPUs are exclusive resources that cannot be shared or overcommitted, unlike CPU and memory.

---

## Section 3: RAG Systems (Questions 9-12)

### Question 9
What is the primary purpose of Retrieval-Augmented Generation (RAG)?

A) Speed up model training
B) Reduce model size
C) Augment LLM responses with relevant context from external knowledge bases
D) Improve GPU utilization

**Answer: C**

**Explanation:** RAG addresses the knowledge limitation of LLMs by retrieving relevant information from external sources (documents, databases) and including it in the prompt. This enables LLMs to answer questions about proprietary data or information not in their training set.

### Question 10
In a RAG pipeline, what is "chunking"?

A) Compressing the model weights
B) Dividing documents into smaller, semantically meaningful segments
C) Batching multiple queries together
D) Splitting GPU workload across multiple devices

**Answer: B**

**Explanation:** Chunking divides large documents into smaller segments (typically 200-1000 tokens) that can fit within embedding model limits and provide focused, relevant context. Good chunking strategies maintain semantic coherence while optimizing retrieval quality.

### Question 11
What is the purpose of the reranking step in advanced RAG systems?

A) Sort documents alphabetically
B) Improve retrieval precision by reordering candidates with a more sophisticated model
C) Compress retrieved documents
D) Remove duplicate results

**Answer: B**

**Explanation:** Reranking uses a cross-encoder or specialized model to score the relevance of retrieved candidates more accurately than vector similarity alone. This improves precision by placing the most relevant documents at the top of the context.

### Question 12
Which embedding model characteristic is MOST important for RAG retrieval quality?

A) Model size in parameters
B) Inference speed
C) Semantic similarity between query and document embeddings
D) GPU memory usage

**Answer: C**

**Explanation:** The quality of embeddings directly impacts retrieval accuracy. Models that produce embeddings with high semantic similarity between related queries and documents (measured by metrics like NDCG) will retrieve more relevant context, leading to better RAG responses.

---

## Section 4: Vector Databases (Questions 13-16)

### Question 13
What is the primary data structure used by vector databases for similarity search?

A) B-tree
B) Hash table
C) Approximate Nearest Neighbor (ANN) indices like HNSW or IVF
D) Linked list

**Answer: C**

**Explanation:** Vector databases use ANN algorithms like HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), or FAISS to enable fast similarity search over high-dimensional embeddings. These provide sub-linear search time with controllable accuracy trade-offs.

### Question 14
In Qdrant, what is a "collection"?

A) A group of database users
B) A named set of vectors with the same dimensionality and configuration
C) A backup of vectors
D) A query result set

**Answer: B**

**Explanation:** A collection in Qdrant is analogous to a table in relational databases. It contains vectors of the same dimensionality, along with metadata (payload), and has specific index and configuration settings.

### Question 15
What is the trade-off when increasing the HNSW `ef_construct` parameter?

A) Higher accuracy but slower indexing
B) Lower accuracy but faster indexing
C) More memory usage only
D) No trade-off, always increase it

**Answer: A**

**Explanation:** `ef_construct` controls the size of the dynamic candidate list during graph construction in HNSW. Higher values create better-quality graphs with more accurate search results, but require more computation during index building.

### Question 16
Which vector database feature is critical for production RAG systems?

A) Built-in LLM hosting
B) Filtering on metadata while maintaining vector search performance
C) Automatic model training
D) Video storage

**Answer: B**

**Explanation:** Production RAG systems need to filter by metadata (e.g., document type, date, user permissions) while searching by vector similarity. Efficient filtered search prevents retrieving irrelevant but similar documents and enables multi-tenant RAG systems.

---

## Section 5: LLM Fine-Tuning Infrastructure (Questions 17-19)

### Question 17
What does LoRA (Low-Rank Adaptation) do?

A) Compresses the model for faster inference
B) Adds trainable low-rank matrices to frozen pre-trained weights
C) Reduces dataset size
D) Increases model accuracy automatically

**Answer: B**

**Explanation:** LoRA freezes the pre-trained model weights and adds small trainable rank-decomposition matrices alongside them. This drastically reduces the number of trainable parameters (often by 1000x), enabling fine-tuning on consumer GPUs while maintaining quality.

### Question 18
What is the main advantage of QLoRA over LoRA?

A) Faster training
B) Better accuracy
C) Quantizes base model to 4-bit while training LoRA adapters in higher precision
D) Simpler implementation

**Answer: C**

**Explanation:** QLoRA combines 4-bit quantization of the base model with LoRA fine-tuning. This reduces memory requirements by ~4x compared to LoRA alone, enabling fine-tuning of 65B models on consumer GPUs with 24-48GB VRAM.

### Question 19
When fine-tuning an LLM, what is "catastrophic forgetting"?

A) GPU memory overflow
B) Model forgetting its pre-trained knowledge when overfitting to fine-tuning data
C) Loss of saved checkpoints
D) Optimizer state corruption

**Answer: B**

**Explanation:** Catastrophic forgetting occurs when aggressive fine-tuning causes the model to lose its general capabilities in favor of the narrow fine-tuning task. It's mitigated through techniques like LoRA, lower learning rates, and maintaining diverse training data.

---

## Section 6: LLM Serving Optimization (Questions 20-22)

### Question 20
What is the effect of quantizing an LLM from FP16 to INT8?

A) 2x memory reduction with minimal quality loss
B) 4x memory reduction
C) No memory savings
D) Model becomes unusable

**Answer: A**

**Explanation:** INT8 quantization (8 bits vs FP16's 16 bits) provides ~2x memory reduction and can accelerate inference on hardware with INT8 support. Modern quantization techniques (like LLM.int8() or GPTQ) maintain minimal quality degradation.

### Question 21
What is "continuous batching" in LLM serving?

A) Training with continuous data streams
B) Batching requests with the same prompt
C) Dynamically adding new requests to a batch as others complete
D) Preprocessing batches before serving

**Answer: C**

**Explanation:** Continuous batching (iteration-level batching) allows new requests to join the batch as soon as earlier requests complete their generation, rather than waiting for the entire batch to finish. This significantly improves GPU utilization and throughput.

### Question 22
What is Flash Attention designed to optimize?

A) Model compression
B) Memory access patterns during attention computation
C) Dataset loading
D) Network communication

**Answer: B**

**Explanation:** Flash Attention optimizes the memory hierarchy by reorganizing attention computation to minimize GPU memory reads/writes. It achieves the same results as standard attention but with significantly reduced memory usage and faster execution.

---

## Section 7: LLM Platform Architecture (Questions 23-24)

### Question 23
In a multi-model LLM platform, what is the purpose of a model router?

A) Load balance across GPU servers
B) Route requests to appropriate models based on task, cost, or latency requirements
C) Route network traffic
D) Store routing tables

**Answer: B**

**Explanation:** A model router intelligently directs requests to different models based on criteria like task type (coding vs chat), required quality, latency constraints, or cost. For example, simple tasks might route to smaller/faster models, while complex tasks use larger models.

### Question 24
What is the primary benefit of implementing semantic caching in LLM systems?

A) Faster model loading
B) Reduced infrastructure costs by serving cached responses for similar queries
C) Better model accuracy
D) Simplified deployment

**Answer: B**

**Explanation:** Semantic caching stores responses for queries and reuses them for semantically similar requests (e.g., "What is AI?" and "Define artificial intelligence"). This can reduce LLM API calls by 30-70% for common query patterns, significantly cutting costs.

---

## Section 8: Production LLM Best Practices (Question 25)

### Question 25
What should be monitored FIRST when an LLM service shows degraded performance?

A) Code quality metrics
B) GPU utilization, memory, and queue depth
C) User login patterns
D) Database query performance

**Answer: B**

**Explanation:** LLM performance is typically bottlenecked by GPU resources. Checking GPU utilization (should be high for optimal throughput), memory usage (watch for OOM), and request queue depth (indicates capacity issues) provides immediate insight into resource constraints.

---

## Section 9: Practical Scenarios (Extended Answer)

### Scenario 1: Cost Optimization Challenge

**Situation:** Your startup is serving Llama 2 70B for a chatbot application. Current costs are $10,000/month on AWS (4x A100 GPUs running 24/7). You have 500 daily active users with sporadic usage patterns (mostly business hours, 9 AM - 5 PM PST). Average response time is 800ms, which is acceptable.

**Questions:**
1. Identify at least 5 specific cost optimization strategies
2. Estimate potential savings for each
3. Explain trade-offs
4. Recommend an implementation plan

**Sample Answer:**

**Cost Optimization Strategies:**

1. **Switch to Smaller Model (Llama 2 13B or Mistral 7B)**
   - Potential savings: 50-70% ($5,000-7,000/month)
   - Trade-off: Slightly reduced response quality
   - Implementation: A/B test to validate quality is acceptable
   - Justification: For chatbot use cases, smaller models often provide 80-90% of the quality

2. **Implement Auto-Scaling with Scheduled Scaling**
   - Potential savings: 40-50% ($4,000-5,000/month)
   - Trade-off: Cold start delays (2-5 min) during scale-up
   - Implementation: Scale to 0-1 GPUs during off-hours (6 PM - 8 AM), pre-warm at 8:30 AM
   - Justification: Usage concentrated in business hours

3. **Quantization (4-bit with GPTQ or AWQ)**
   - Potential savings: 30-40% ($3,000-4,000/month)
   - Trade-off: Minimal quality impact (<5%), but lower memory enables fewer GPUs
   - Implementation: Quantize model, test quality, redeploy
   - Justification: 4-bit 70B fits on 2x A100 instead of 4x

4. **Semantic Caching**
   - Potential savings: 20-30% ($2,000-3,000/month)
   - Trade-off: Stale responses for cached queries, cache storage costs
   - Implementation: Deploy Redis with embedding-based cache, 1-hour TTL
   - Justification: User queries likely have common patterns (FAQs, etc.)

5. **Spot Instances (AWS EC2 Spot)**
   - Potential savings: 60-70% ($6,000-7,000/month)
   - Trade-off: Potential interruptions (2% chance per hour)
   - Implementation: Run on spot with on-demand fallback, save checkpoints
   - Justification: Can handle interruptions with proper retry logic

6. **Move to More Cost-Effective Cloud**
   - Potential savings: 30-50% ($3,000-5,000/month)
   - Trade-off: Different tooling, potential migration effort
   - Implementation: Evaluate Lambda Labs, RunPod, or CoreWeave pricing
   - Justification: Specialized GPU clouds often 2-3x cheaper than AWS

**Recommended Implementation Plan (Phased Approach):**

**Phase 1 (Week 1-2): Quick Wins - $5,500 savings**
- Implement semantic caching (-$2,500/month)
- Enable scheduled auto-scaling for off-hours (-$3,000/month)
- Total savings: ~55%

**Phase 2 (Week 3-4): Model Optimization - Additional $3,000 savings**
- Test quantized 70B model (4-bit)
- If quality acceptable, deploy to reduce GPU count
- Additional savings: ~30%
- Running total: ~85% savings

**Phase 3 (Month 2): Infrastructure Migration - Additional $1,500 savings**
- Evaluate and migrate to cost-effective GPU cloud
- Use spot instances where appropriate
- Additional savings: ~15%
- **Total savings: ~$10,000/month → ~$1,500-2,000/month (80-85% reduction)**

**Risk Mitigation:**
- Always A/B test quality before deploying optimizations
- Maintain on-demand fallback for spot instances
- Set up alerting for cache hit rates and response times
- Monitor user satisfaction scores throughout

---

### Scenario 2: RAG System Quality Issues

**Situation:** You've deployed a RAG system for internal company documentation (10,000 documents, ~50M tokens). Users report that the system frequently returns irrelevant information or misses obvious answers. You're using:
- Embeddings: `all-MiniLM-L6-v2` (384 dimensions)
- Vector DB: Qdrant with default settings
- Chunk size: 1500 tokens with no overlap
- Retrieval: Top-5 chunks
- LLM: Llama 2 13B

**Questions:**
1. Diagnose likely root causes
2. Propose specific improvements
3. Explain how to measure success
4. Describe testing methodology

**Sample Answer:**

**Root Cause Analysis:**

1. **Poor Embedding Model:**
   - `all-MiniLM-L6-v2` is a lightweight general-purpose model
   - 384 dimensions may not capture sufficient semantic information
   - Not optimized for domain-specific technical documentation

2. **Suboptimal Chunking:**
   - 1500 tokens is very large (near LLM context limits)
   - No overlap means context boundaries might split related information
   - Fixed size doesn't respect semantic boundaries

3. **Limited Retrieval:**
   - Top-5 chunks may miss relevant context
   - No reranking to improve precision
   - No diversity in retrieval results

4. **Default Vector DB Settings:**
   - Qdrant HNSW parameters not tuned
   - Possible accuracy/speed trade-off issues

**Proposed Improvements:**

**1. Upgrade Embedding Model**
```python
# Before: all-MiniLM-L6-v2 (384 dim)
# After: e5-large-v2 or instructor-xl (768-1024 dim)

from sentence_transformers import SentenceTransformer

# Better embedding model
model = SentenceTransformer('intfloat/e5-large-v2')

# Add instruction prefix for queries
query_embedding = model.encode("query: " + user_query)
```
- Expected improvement: 20-30% better retrieval accuracy
- Trade-off: 2-3x slower embedding, 2x storage

**2. Optimize Chunking Strategy**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Improved chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # Smaller chunks
    chunk_overlap=50,         # Overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""],  # Respect semantic boundaries
    length_function=len,
)

chunks = splitter.split_documents(docs)
```
- Expected improvement: 15-25% better retrieval
- Trade-off: 2-3x more chunks, more storage

**3. Implement Two-Stage Retrieval + Reranking**
```python
# Stage 1: Retrieve top-20 candidates (recall)
candidates = vector_db.search(query_embedding, top_k=20)

# Stage 2: Rerank with cross-encoder (precision)
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

scores = reranker.predict([(query, doc.text) for doc in candidates])
top_5 = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```
- Expected improvement: 25-35% better precision
- Trade-off: +100-200ms latency per query

**4. Tune Vector Database Settings**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name="docs",
    hnsw_config=HnswConfigDiff(
        m=32,              # More connections per node (default: 16)
        ef_construct=256,  # Better graph quality (default: 100)
    )
)

# Increase search accuracy
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    limit=20,
    search_params={"hnsw_ef": 128, "exact": False}  # Higher ef for better recall
)
```
- Expected improvement: 10-15% better recall
- Trade-off: 2x indexing time, +20% memory

**5. Add Hybrid Search (Dense + Sparse)**
```python
from qdrant_client.models import SparseVector

# Combine dense embeddings with BM25 sparse vectors
results = client.search_batch(
    collection_name="docs",
    requests=[
        # Dense search
        SearchRequest(vector=dense_embedding, limit=10),
        # Sparse search (BM25)
        SearchRequest(sparse_vector=sparse_embedding, limit=10)
    ]
)

# Reciprocal Rank Fusion to combine results
final_results = reciprocal_rank_fusion(results[0], results[1])
```
- Expected improvement: 15-20% better retrieval
- Trade-off: More complex indexing pipeline

**Measuring Success:**

1. **Offline Metrics (Before Production):**
   - Create eval set of 100-200 query-document pairs
   - Measure: Recall@5, Recall@10, MRR, NDCG@10
   - Set targets: Recall@10 > 85%, MRR > 0.7

2. **Online Metrics (Production):**
   - User feedback: thumbs up/down on responses
   - Answer relevance: LLM-as-judge to score answers
   - Click-through rate on retrieved documents
   - Task completion rate

3. **Quality Assurance:**
   ```python
   # Create eval dataset
   eval_queries = [
       {
           "query": "How do I reset my password?",
           "relevant_docs": ["doc_123", "doc_456"],
           "irrelevant_docs": ["doc_789"]
       },
       # ... 100+ more
   ]

   # Calculate metrics
   def evaluate_retrieval(eval_set, retrieval_fn):
       recalls = []
       for item in eval_set:
           retrieved = retrieval_fn(item["query"])
           relevant_found = len(set(retrieved) & set(item["relevant_docs"]))
           recalls.append(relevant_found / len(item["relevant_docs"]))
       return sum(recalls) / len(recalls)
   ```

**Testing Methodology:**

1. **A/B Testing Framework:**
   - Deploy improvements incrementally
   - Route 10% traffic to new version
   - Compare metrics over 1 week
   - Gradually increase to 50%, then 100%

2. **Specific Test Sequence:**
   - Week 1: Test embedding model upgrade
   - Week 2: Test new chunking strategy
   - Week 3: Add reranking
   - Week 4: Tune vector DB settings
   - Week 5: Full deployment with monitoring

3. **Rollback Plan:**
   - Keep old embeddings/indices for 2 weeks
   - Automated rollback if error rate >2%
   - Manual review if quality score drops >10%

**Expected Results:**
- Retrieval accuracy: 60% → 85%+ (Recall@10)
- User satisfaction: 65% → 90%+
- Latency: 400ms → 600ms (acceptable trade-off)
- Infrastructure cost: +30% (better embeddings, more chunks)

---

### Scenario 3: Production LLM Incident Response

**Situation:** It's 2 AM and you receive a PagerDuty alert: "LLM Service - High P99 Latency." Your production metrics show:
- P50 latency: 800ms (normal)
- P99 latency: 45 seconds (normal: 2s)
- Request rate: 120 req/s (normal: 100 req/s)
- GPU utilization: 98% (normal: 85%)
- GPU memory: 78GB/80GB (normal: 60GB/80GB)
- Request queue depth: 450 (normal: 20)
- Error rate: 0.5% (normal)

**Questions:**
1. What is the likely root cause?
2. What immediate actions would you take?
3. What data would you collect for RCA?
4. What preventive measures should be implemented?

**Sample Answer:**

**Immediate Diagnosis (First 5 Minutes):**

The issue appears to be **GPU saturation causing queue buildup**, not a service outage. Key indicators:
- GPU utilization at 98% (maxed out)
- Queue depth 450 vs normal 20 (22.5x increase)
- P99 latency spike but P50 normal (suggests some requests still fast)
- Small increase in request rate (20%)

**Root Cause Hypothesis:**
1. **Primary:** A few very long generation requests (high max_tokens) are blocking the batch
2. **Secondary:** Slight traffic increase pushed system past optimal capacity
3. **Contributing:** No request timeouts or max token limits enforced

**Immediate Actions (0-15 Minutes):**

```bash
# 1. Check current requests and identify outliers
kubectl exec -it vllm-pod-0 -- curl localhost:8000/metrics | grep request_duration

# 2. Check for stuck requests
kubectl logs vllm-deployment --tail=100 | grep -i "error\|timeout\|oom"

# 3. Check request parameters distribution
# Look for abnormally high max_tokens in recent requests
curl http://prometheus:9090/api/v1/query?query='histogram_quantile(0.99, request_max_tokens)'

# 4. Immediate mitigation: Scale up pods (if auto-scaling not enabled)
kubectl scale deployment vllm-deployment --replicas=6  # From 4 to 6

# 5. Implement request timeout if not present
kubectl set env deployment/vllm-deployment VLLM_REQUEST_TIMEOUT=30s

# 6. Monitor queue decrease
watch -n 5 'kubectl get pods -l app=vllm -o wide && echo "Queue depth:" && curl -s prometheus:9090/api/v1/query?query=vllm_queue_depth'
```

**Data Collection for RCA (15-30 Minutes):**

1. **Request Traces:**
```python
# Query distributed traces for slow requests
import jaeger_client

# Find P99 requests in last hour
slow_requests = jaeger.query(
    service="vllm-service",
    start_time=now - 1h,
    duration_min=10000ms,  # >10s requests
    limit=100
)

# Analyze their parameters
for trace in slow_requests:
    print(f"Request ID: {trace.id}")
    print(f"Max tokens: {trace.tags['max_tokens']}")
    print(f"Input length: {trace.tags['input_length']}")
    print(f"Output length: {trace.tags['output_length']}")
```

2. **GPU Metrics:**
```bash
# Export GPU metrics for analysis
kubectl exec vllm-pod-0 -- nvidia-smi dmon -s u -c 60 > gpu_utilization.log

# Check for memory fragmentation
kubectl exec vllm-pod-0 -- nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

3. **Request Distribution:**
```promql
# PromQL queries to run
histogram_quantile(0.99, rate(vllm_request_tokens_bucket[5m]))
histogram_quantile(0.99, rate(vllm_generation_tokens_bucket[5m]))
rate(vllm_requests_total[5m])
```

4. **Application Logs:**
```bash
# Extract all requests from incident window
kubectl logs vllm-deployment --since=2h --timestamps > incident_logs.txt

# Parse for patterns
cat incident_logs.txt | jq 'select(.duration_ms > 10000)' | jq -s 'group_by(.endpoint) | map({endpoint: .[0].endpoint, count: length})'
```

**Root Cause Analysis Findings (Example):**

After investigation, discovered:
- 15 requests with `max_tokens=4096` (vs normal 512)
- These came from a new integration that didn't set token limits
- At 98% GPU utilization, continuous batching couldn't efficiently handle mixed request sizes
- Queue built up as long requests blocked batch throughput

**Preventive Measures:**

**1. Implement Request Validation:**
```python
# Add to vLLM frontend
from fastapi import HTTPException

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Enforce limits
    if request.max_tokens > 1024:
        raise HTTPException(
            status_code=400,
            detail="max_tokens cannot exceed 1024. Use streaming for longer outputs."
        )

    if len(request.prompt) > 4096:
        raise HTTPException(
            status_code=400,
            detail="Prompt length cannot exceed 4096 tokens"
        )

    # Add timeout
    try:
        result = await asyncio.wait_for(
            generate(request),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

    return result
```

**2. Set Up Proactive Alerts:**
```yaml
# prometheus_rules.yml
groups:
  - name: vllm_capacity
    interval: 30s
    rules:
      # Alert before queue builds up
      - alert: VLLMQueueDepthHigh
        expr: vllm_queue_depth > 50
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "vLLM queue depth high (current: {{ $value }})"

      # Alert on GPU memory pressure
      - alert: GPUMemoryHigh
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.85
        for: 5m
        labels:
          severity: warning

      # Alert on P99 latency degradation
      - alert: HighP99Latency
        expr: histogram_quantile(0.99, rate(vllm_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: critical
```

**3. Implement Auto-Scaling:**
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 4
  maxReplicas: 12
  metrics:
    # Scale on queue depth
    - type: Pods
      pods:
        metric:
          name: vllm_queue_depth
        target:
          type: AverageValue
          averageValue: "30"
    # Scale on GPU utilization
    - type: Pods
      pods:
        metric:
          name: gpu_utilization_percent
        target:
          type: AverageValue
          averageValue: "80"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

**4. Implement Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/completions")
@limiter.limit("100/minute")  # Per-IP limit
async def completions(request: Request):
    # Also implement token-bucket for total system throughput
    if not await acquire_token():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    ...
```

**5. Better Capacity Planning:**
```python
# Capacity model
def calculate_capacity(gpu_count, model_size, avg_request_params):
    """
    GPU count: 4x A100 (80GB each)
    Model: Llama 2 70B (FP16)
    Avg request: 512 input + 256 output tokens
    """

    # Memory per request (KV cache)
    memory_per_token = 0.5  # MB per token for 70B model
    tokens_per_request = 512 + 256
    memory_per_request = tokens_per_token * memory_per_token

    # Available memory (after model weights)
    total_memory = gpu_count * 80 * 1024  # MB
    model_memory = 140 * 1024  # 140GB for 70B FP16
    available_memory = total_memory - model_memory

    # Max concurrent requests
    max_concurrent = available_memory / memory_per_request

    # Throughput (tokens/sec per GPU ~= 30 for 70B)
    throughput_tokens_per_sec = gpu_count * 30

    # Requests per second
    requests_per_sec = throughput_tokens_per_sec / tokens_per_request

    print(f"Max concurrent requests: {max_concurrent:.0f}")
    print(f"Max throughput: {requests_per_sec:.1f} req/s")
    print(f"Safe operating point (80% capacity): {requests_per_sec * 0.8:.1f} req/s")

    return max_concurrent, requests_per_sec

# Run capacity planning
calculate_capacity(gpu_count=4, model_size="70B", avg_request_params={})
```

**6. Implement Request Prioritization:**
```python
import heapq
from enum import IntEnum

class Priority(IntEnum):
    CRITICAL = 0  # Production API
    HIGH = 1      # Internal tools
    MEDIUM = 2    # Batch processing
    LOW = 3       # Experimental

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def add_request(self, request, priority: Priority):
        heapq.heappush(self.queue, (priority.value, time.time(), request))

    def get_next_batch(self, batch_size):
        batch = []
        for _ in range(min(batch_size, len(self.queue))):
            if self.queue:
                _, _, request = heapq.heappop(self.queue)
                batch.append(request)
        return batch
```

**Post-Incident Report:**

**Impact:**
- Duration: 45 minutes
- Affected requests: ~5,400 (10% experienced high latency)
- No data loss or errors
- User-facing impact: Minimal (90% requests unaffected)

**Timeline:**
- 02:00 AM: Alert triggered
- 02:05 AM: Engineer paged, investigation started
- 02:15 AM: Root cause identified, mitigation deployed
- 02:30 AM: Queue cleared, latency normalized
- 02:45 AM: Incident resolved

**Action Items:**
1. ✅ Implement request parameter validation (Week 1)
2. ✅ Add proactive queue depth alerts (Week 1)
3. ✅ Deploy HPA with queue-based scaling (Week 2)
4. ✅ Add rate limiting per client (Week 2)
5. ✅ Improve capacity planning documentation (Week 3)
6. ✅ Implement request prioritization (Week 4)

---

## Scoring Guide

### Multiple Choice (Questions 1-25)
- 1 point each = 25 points total
- Passing score: 20/25 (80%)

### Scenario Questions
- Graded on completeness and depth
- Each scenario worth bonus points for exceptional answers
- Key evaluation criteria:
  - Root cause analysis accuracy
  - Solution comprehensiveness
  - Understanding of trade-offs
  - Production readiness of recommendations

### Grading Rubric for Scenarios

**Excellent (90-100%):**
- Identifies root causes accurately
- Proposes comprehensive solutions with code examples
- Explains trade-offs clearly
- Includes measurement and validation strategies
- Shows production-grade thinking

**Good (80-89%):**
- Identifies main root causes
- Proposes practical solutions
- Mentions key trade-offs
- Includes basic metrics

**Satisfactory (70-79%):**
- Identifies some root causes
- Proposes some solutions
- Acknowledges trade-offs exist
- Missing some details

**Needs Improvement (<70%):**
- Misses key root causes
- Solutions incomplete or impractical
- Doesn't consider trade-offs
- Lacks measurement strategy

---

## Answer Key Summary

**Section-by-Section Breakdown:**

1-4:   B, B, C, C (Fundamentals)
5-8:   B, C, B, B (vLLM)
9-12:  C, B, B, C (RAG)
13-16: C, B, A, B (Vector Databases)
17-19: B, C, B (Fine-Tuning)
20-22: A, C, B (Optimization)
23-24: B, B (Platform Architecture)
25:    B (Production)

**Complete Answer Key:**
1. B  | 2. B  | 3. C  | 4. C  | 5. B
6. C  | 7. B  | 8. B  | 9. C  | 10. B
11. B | 12. C | 13. C | 14. B | 15. A
16. B | 17. B | 18. C | 19. B | 20. A
21. C | 22. B | 23. B | 24. B | 25. B

---

## Next Steps Based on Your Score

### 90-100% (23-25 correct)
Excellent! You have a strong grasp of LLM infrastructure.
- **Next steps:**
  - Complete advanced exercises (Exercise 06-08)
  - Build a portfolio project combining multiple concepts
  - Contribute to open-source LLM tools
  - Start preparing for Senior AI Infrastructure Engineer role

### 80-89% (20-22 correct)
Good work! You understand the core concepts.
- **Review areas:**
  - Any sections with incorrect answers
  - Hands-on exercises for weaker topics
- **Next steps:**
  - Complete all 8 exercises
  - Build a production-grade LLM project
  - Focus on optimization and scaling topics

### 70-79% (18-19 correct)
You're on the right track but need more practice.
- **Review thoroughly:**
  - Re-read lessons for missed questions
  - Focus on hands-on exercises
  - Review optimization and production topics
- **Next steps:**
  - Spend extra time on Lessons 05-08
  - Redo exercises with solutions
  - Build small projects for each major topic

### Below 70% (<18 correct)
Additional study needed before moving forward.
- **Action plan:**
  1. Re-read all lessons carefully
  2. Take notes on key concepts
  3. Complete beginner exercises first (01-03)
  4. Review prerequisite modules (Kubernetes, Monitoring)
  5. Retake quiz after one week of focused study
- **Resources:**
  - See resources.md for tutorials and courses
  - Join community forums for questions
  - Consider pairing study with hands-on practice

---

## Additional Study Resources

### If You Struggled With Specific Topics:

**vLLM (Questions 5-8):**
- Re-read Lesson 02
- Complete Exercise 01: Deploy Llama 2 with vLLM
- Watch: vLLM YouTube tutorials
- Read: vLLM documentation

**RAG Systems (Questions 9-12):**
- Re-read Lesson 03
- Complete Exercise 02: Build Basic RAG System
- Read: LangChain RAG tutorials
- Practice: Build RAG with different chunking strategies

**Vector Databases (Questions 13-16):**
- Re-read Lesson 04
- Complete Exercise 03: Set Up Vector Database
- Read: Qdrant/Weaviate documentation
- Practice: Benchmark different ANN algorithms

**Fine-Tuning (Questions 17-19):**
- Re-read Lesson 05
- Complete Exercise 04: Fine-Tune with LoRA
- Read: Hugging Face PEFT documentation
- Practice: Fine-tune models on different datasets

**Optimization (Questions 20-22):**
- Re-read Lesson 06
- Complete Exercise 05: Optimize LLM Inference
- Read: Quantization guides (GPTQ, AWQ)
- Practice: Benchmark quantized vs full precision

**Platform Architecture (Questions 23-24):**
- Re-read Lesson 07
- Complete Exercise 08: Build Multi-Model API
- Study: Production LLM platform architectures
- Practice: Design your own platform

**Production Practices (Question 25):**
- Re-read Lesson 08
- Complete Exercise 06: Deploy on Kubernetes
- Complete Exercise 07: Implement Monitoring
- Practice: Incident response scenarios

---

## Certification Recommendation

Based on your quiz performance, consider pursuing these certifications:

**Score 90%+:**
- Ready for role-specific certifications
- Consider Kubernetes certifications (CKA/CKAD)
- Explore cloud certifications (AWS ML Specialty, GCP ML Engineer)

**Score 80-89%:**
- Complete remaining exercises first
- Build 1-2 portfolio projects
- Then pursue certifications

**Score <80%:**
- Focus on fundamentals
- Complete all exercises
- Gain hands-on experience
- Retake quiz before pursuing certifications

---

## Feedback and Improvements

This quiz is designed to test practical LLM infrastructure knowledge. If you have suggestions for improvements:
- Additional scenario topics
- Clarifications needed
- Real-world examples to add

Please submit feedback to help improve this curriculum!

---

**Good luck, and happy learning!**
