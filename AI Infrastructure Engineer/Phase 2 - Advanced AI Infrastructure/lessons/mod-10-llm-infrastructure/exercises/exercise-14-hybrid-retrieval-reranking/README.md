# Exercise 14: Hybrid Retrieval + Reranking Pipeline

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Lab 02 (RAG), exercises 04 + 13

## Objective

Upgrade a basic RAG pipeline (dense vectors only) to a production pattern: hybrid search (BM25 + dense), reranking with a cross-encoder, citation grounding, retrieval evaluation. Measure quality improvement.

## Why this matters

Single-vector dense retrieval misses ~20% of true positives. Hybrid + reranker is the industry-standard "good RAG" pattern; teams that ship it answer accuracy questions with confidence intervals, not vibes.

## Requirements

1. BM25 + dense retrieval as separate paths.
2. Reciprocal Rank Fusion (RRF) combining them.
3. Cross-encoder reranker on top-k from RRF.
4. Citation grounding: response includes [source] markers.
5. Retrieval evaluation on a held-out set (recall@10, MRR).

## Step-by-step

### Step 1 — Corpus + indexing (30 min)
Use a real-ish corpus (e.g., 10K Wikipedia abstracts).

Dense (per lab 02):
```python
qdrant.recreate_collection("docs", VectorParams(size=384, distance=Distance.COSINE))
qdrant.upsert(collection_name="docs", points=[PointStruct(id=i, vector=emb, payload={"text": text}) for ...])
```

BM25 (sparse):
```python
from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
```

### Step 2 — Dense + BM25 retrieval (30 min)
```python
def dense_retrieve(query, k=20):
    q = emb_model.encode([query])[0].tolist()
    hits = qdrant.search("docs", query_vector=q, limit=k)
    return [(h.id, h.score) for h in hits]

def bm25_retrieve(query, k=20):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in top_k]
```

### Step 3 — Reciprocal Rank Fusion (15 min)
```python
def rrf(rankings, k=60):
    """Each ranking is [(doc_id, score), ...] ordered by relevance."""
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, k=20):
    dense = dense_retrieve(query, k=k)
    sparse = bm25_retrieve(query, k=k)
    fused = rrf([dense, sparse])[:k]
    return fused
```

### Step 4 — Cross-encoder reranker (45 min)
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, k=4):
    pairs = [[query, corpus[doc_id]] for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for (doc_id, _), score in ranked[:k]]
```

### Step 5 — Compose with citations (30 min)
```python
def answer(question):
    candidates = hybrid_retrieve(question, k=20)
    top_docs = rerank(question, candidates, k=4)
    context = "\n\n".join([f"[doc-{i}] {corpus[doc_id]}" for i, (doc_id, _) in enumerate(top_docs)])
    
    prompt = f"""Answer the question using ONLY the context provided.
Cite sources by their bracketed id (e.g., [doc-1]). If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    r = llm.chat.completions.create(model="...", messages=[{"role":"user","content":prompt}])
    return {"answer": r.choices[0].message.content, "sources": [doc_id for doc_id, _ in top_docs]}
```

### Step 6 — Evaluation (30 min)
Use a held-out set of (question, gold_answer, gold_source_ids):
```python
def evaluate():
    metrics = []
    for q in eval_set:
        result = answer(q["question"])
        retrieved = set(result["sources"])
        relevant = set(q["gold_source_ids"])
        recall = len(retrieved & relevant) / max(len(relevant), 1)
        first_relevant_rank = next((i for i, s in enumerate(result["sources"]) if s in relevant), -1)
        mrr = 1 / (first_relevant_rank + 1) if first_relevant_rank >= 0 else 0
        metrics.append({"recall@4": recall, "mrr": mrr})
    return {
        "recall@4": mean(m["recall@4"] for m in metrics),
        "mrr": mean(m["mrr"] for m in metrics),
    }
```

Compare:
- Dense only
- BM25 only
- RRF (hybrid)
- RRF + reranker

Expect: RRF improves recall by ~10pp over either alone; reranker improves MRR by ~15pp.

## Deliverables

1. Working hybrid + rerank pipeline.
2. Evaluation matrix (4 strategies × 2 metrics).
3. Sample queries showing citation grounding.
4. `RAG_DESIGN.md`: design decisions + when to reach for what.

## Validation

- [ ] All 4 strategies measurable.
- [ ] Hybrid > either alone on the eval set.
- [ ] Reranker improves MRR but adds latency.
- [ ] Responses cite sources.

## Common pitfalls

- **Reranker on full candidate set** — Reranker is slow (cross-attention); retrieve top-k first.
- **Citation hallucinations** — LLM cites doc-99 when only doc-1..4 provided. Validate citations against the actual passed context.
- **Dense embedding from a different model than the index** — All embeddings must come from the same model.
- **Eval set bias** — Easy questions inflate scores; mine real prod questions too.
