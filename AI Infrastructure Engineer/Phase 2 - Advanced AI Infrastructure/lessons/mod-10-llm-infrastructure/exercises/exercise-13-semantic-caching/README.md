# Exercise 13: Semantic Caching for LLMs

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Lab 03 (vector DB), exercise 04

## Objective

Build a semantic cache that returns cached responses for semantically-equivalent queries, not just byte-exact matches. Measure hit rate, cost savings, latency reduction.

## Why this matters

Standard caches keyed by query string have ~5% hit rate for LLM workloads (users phrase the same intent differently). Semantic caching pushes that to 30-60% by matching by embedding similarity, halving the bill.

## Requirements

1. Cache keyed by query embedding.
2. Hit threshold: cosine similarity > 0.92.
3. TTL: 24 hours.
4. Per-tenant isolation (no cross-tenant leakage).
5. Hit-rate, savings, latency metrics.
6. Bypass query parameter for testing.

## Step-by-step

### Step 1 — Embedding + vector store (30 min)
Use sentence-transformers + Qdrant (per lab 03).
```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.recreate_collection(
    collection_name="llm_cache",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

### Step 2 — Cache check (30 min)
```python
import hashlib, time
THRESHOLD = 0.92
TTL_SEC = 24 * 3600

def cache_lookup(query: str, tenant: str) -> dict | None:
    q = emb_model.encode([query])[0].tolist()
    hits = qdrant.search(
        collection_name="llm_cache",
        query_vector=q,
        limit=1,
        query_filter={"must": [{"key": "tenant", "match": {"value": tenant}}]},
    )
    if hits and hits[0].score >= THRESHOLD:
        # Check TTL
        if time.time() - hits[0].payload["ts"] < TTL_SEC:
            return hits[0].payload
    return None
```

### Step 3 — Cache write (15 min)
```python
def cache_write(query: str, response: dict, tenant: str, usage: dict):
    q = emb_model.encode([query])[0].tolist()
    qdrant.upsert(
        collection_name="llm_cache",
        points=[PointStruct(
            id=hashlib.md5(f"{tenant}:{query}:{time.time()}".encode()).hexdigest()[:32],
            vector=q,
            payload={
                "tenant": tenant,
                "query": query,
                "response": response,
                "ts": time.time(),
                "usage": usage,
            },
        )],
    )
```

### Step 4 — Integration (15 min)
```python
async def chat(req, tenant):
    if not req.get("bypass_cache"):
        cached = cache_lookup(req["query"], tenant)
        if cached:
            CACHE_HIT.labels(tenant=tenant).inc()
            return cached["response"]
    
    CACHE_MISS.labels(tenant=tenant).inc()
    resp = await llm.chat.completions.create(...)
    cache_write(req["query"], resp, tenant, resp.usage)
    return resp
```

### Step 5 — Metrics (15 min)
```python
CACHE_HIT  = Counter("llm_cache_hits_total",   "Cache hits",   ["tenant"])
CACHE_MISS = Counter("llm_cache_misses_total", "Cache misses", ["tenant"])
CACHE_SAVINGS = Counter("llm_cache_tokens_saved_total", "Tokens saved by cache", ["tenant"])
```

### Step 6 — Hit rate test (30 min)
Generate 1000 query variations of 100 semantic topics. After warmup:
- Expected: ~30-60% hit rate.
- Verify: cache hits return the cached response, not a fresh LLM call.
- Compare cost vs no-cache.

### Step 7 — TTL + eviction (15 min)
Schedule a job that deletes Qdrant points older than TTL_SEC. Or use Qdrant's `payload_indexing` with a TTL filter at query time.

## Deliverables

1. Cache implementation + Qdrant config.
2. Cache hit/miss metrics.
3. Test demonstrating semantic equivalence handling.
4. `RESULTS.md`: hit rate, cost savings, latency improvement.

## Validation

- [ ] Hit rate ≥ 30% on representative workload.
- [ ] Cached response identical to original.
- [ ] Tenant A's queries don't return tenant B's cache.
- [ ] TTL expires entries.

## Common pitfalls

- **Threshold too low** — Returns wrong answers for slightly different questions.
- **No per-tenant filter** — Data leakage between tenants.
- **Caching dynamic content** — "What's the weather?" should not cache. Tag prompts as cacheable/not.
- **Cache stampede** — N concurrent identical misses all hit the LLM. Use singleflight pattern.
