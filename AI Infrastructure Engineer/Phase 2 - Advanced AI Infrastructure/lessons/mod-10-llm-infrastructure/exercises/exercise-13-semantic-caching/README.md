# Exercise 13: Semantic Caching for LLMs

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** [Lab 03](../../labs/lab-03-vector-db-qdrant.md) and [Exercise 04](../exercise-04-vector-db-benchmarking/README.md)
**Hardware:** Docker for Qdrant, plus a model endpoint

## Goal

Build a cache that recognizes when a new question means the same as one you have already answered, and returns the saved answer without calling the model. Measure how often it hits, how much it saves, and how often it gets it wrong.

## Why it matters

People ask the same thing in different words. An exact-match cache misses all of those, so it helps little for chat traffic. A semantic cache can help a lot for repetitive workloads like FAQs and support, and not at all for open-ended chat. It also has a risk exact caches do not: a near-match can return the wrong answer.

So the real goal is not a high hit rate. It is a high hit rate **with almost no wrong answers**.

## Before you start

Know where this cache sits among the others:

| Cache | What it saves | Risk |
|---|---|---|
| **Exact response cache** | The whole model call, for identical requests | None, if the request is the same |
| **Prefix / KV cache** (built into vLLM and hosted APIs) | Re-processing a shared prompt start | None, the output is still freshly generated |
| **Semantic cache** (this exercise) | The whole model call, for similar requests | Wrong answers for near-misses |

Try the first two before you add the third, since they are free of that risk. Tools such as GPTCache, Redis' semantic cache, and gateway-level caches (for example in LiteLLM) already implement this idea, so look at one when you design yours.

## Steps

### Step 1: Embeddings and storage (30 min)

Embed with a small model and store in Qdrant, as in Lab 03. Store the answer in the payload together with the tenant, a timestamp, and everything that shapes the answer: model name, prompt version, and settings.

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

emb = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.create_collection("llm_cache", vectors_config=models.VectorParams(
    size=384, distance=models.Distance.COSINE))
qdrant.create_payload_index("llm_cache", "tenant", models.PayloadSchemaType.KEYWORD)
qdrant.create_payload_index("llm_cache", "cache_key", models.PayloadSchemaType.KEYWORD)
```

### Step 2: Lookup (30 min)

Filter by tenant and by the "cache key" (model plus prompt version), so a cached answer is only reused in exactly the situation it was produced in. Use the current Qdrant `query_points` call, since older `search` calls are deprecated.

```python
def lookup(query: str, tenant: str, cache_key: str, threshold: float = 0.92):
    vec = emb.encode(query, normalize_embeddings=True).tolist()
    res = qdrant.query_points(
        "llm_cache", query=vec, limit=1, with_payload=True,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
            models.FieldCondition(key="cache_key", match=models.MatchValue(value=cache_key)),
            models.FieldCondition(key="expires_at", range=models.Range(gt=time.time())),
        ]),
    ).points
    return res[0].payload if res and res[0].score >= threshold else None
```

The 0.92 threshold is a starting point, not a rule. The right value depends on your embedding model and your data, and you will tune it in Step 5.

### Step 3: Write and expire (15 min)

Store an `expires_at` timestamp instead of a creation time, so expiry is a simple filter. Pick the lifetime per kind of content. FAQ answers may last days. Anything about prices, availability, or "today" should not be cached at all.

Delete expired points on a schedule, so the collection does not grow forever. Use a deterministic ID such as a SHA-256 of `tenant + cache_key + normalized query`, so the same question overwrites itself instead of piling up. (Avoid MD5 for new code.)

### Step 4: Put it in front of the model (30 min)

```python
async def chat(req, tenant, cache_key):
    if cacheable(req) and not req.bypass_cache:
        hit = lookup(req.query, tenant, cache_key)
        if hit:
            CACHE_HIT.labels(tenant).inc()
            return hit["response"]
    CACHE_MISS.labels(tenant).inc()
    resp = await llm.chat.completions.create(...)
    if cacheable(req):
        write(req.query, resp, tenant, cache_key)
    return resp
```

Decide what `cacheable` means: no tool calls, no personal data in the answer, no time-sensitive content, and a low temperature. Write the rule down.

Also handle many identical misses arriving at once. Without protection, 50 users asking the same new question will all call the model. Make the first one do the work and let the others wait for its result (often called "single flight").

### Step 5: Measure the thing that matters (45 min)

1. Build 100 base questions, then write 5–10 paraphrases of each. Also write **look-alike questions with different answers** ("How do I cancel my subscription?" versus "How do I cancel my order?").
2. Run them through the cache. Record the **hit rate**, the **cost and latency saved**, and the number of **wrong hits**: cases where the cache returned an answer that does not fit the question.
3. Repeat at thresholds from 0.85 to 0.98. Plot hit rate and wrong-hit rate against the threshold, and choose the point where wrong hits are close to zero.

For borderline matches, an extra check helps: a cheap model, or a reranker, confirms that the two questions really mean the same thing.

### Step 6: Metrics (15 min)

Export hits, misses, tokens saved, lookup latency, and the similarity score of each hit. Watch the distribution of scores over time. A drift toward the threshold means more risky hits.

### Step 7: Test isolation (15 min)

Ask the same question as tenant A and tenant B. Confirm B never receives A's cached answer. Then change the prompt version and confirm the old entries are no longer used.

## Deliverables

1. The cache service and its Qdrant setup.
2. Metrics for hits, misses, savings, and latency.
3. A test set with paraphrases and look-alikes, and the threshold plot.
4. `RESULTS.md`: hit rate, wrong-hit rate, savings, and latency at your chosen threshold.
5. `CACHE_POLICY.md`: what is cacheable, lifetime per content type, and how entries are invalidated.

## Check yourself

- [ ] Hit rate is meaningful on repetitive traffic, and you can explain when it would be near zero.
- [ ] Wrong hits are near zero at your threshold, and you can show the evidence.
- [ ] Tenant A never gets tenant B's answers.
- [ ] Changing the model or prompt version stops old entries from being used.
- [ ] Expired entries are no longer returned.

## Common pitfalls

- **A threshold copied from a blog post.** It depends on your embedding model and data. Measure it.
- **No tenant filter.** A cache without one leaks one customer's answer to another.
- **Ignoring the model and prompt version.** After a prompt change you would serve answers from the old prompt.
- **Caching personal or time-sensitive answers.** Do not cache "my balance" or "today's weather".
- **Stampedes.** Many identical misses at once defeat the cache. Deduplicate in-flight requests.
- **Counting hits as success.** A hit that returns the wrong answer is worse than a miss.
