# Exercise 14: Hybrid Retrieval and Reranking

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** [Lab 02](../../labs/lab-02-rag-pipeline.md), [Exercise 04](../exercise-04-vector-db-benchmarking/README.md), [Exercise 13](../exercise-13-semantic-caching/README.md), and [Lesson 03: RAG Systems](../../03-rag-systems.md)
**Hardware:** A laptop is enough. A GPU makes the reranker faster.

## Goal

Upgrade a basic RAG pipeline step by step: add keyword search next to vector search, merge the results, rerank them, and make the answer cite its sources. Measure each step so you know which ones actually helped on your data.

## Why it matters

Vector search is good at meaning and weak at exact terms: product codes, error messages, names, and rare words. Keyword search is the opposite. Combining them and reranking the shortlist is now the standard way to get dependable retrieval, and it is usually the cheapest way to improve a RAG system before touching the model.

But "standard" is not "always better." Your job in this exercise is to prove it on your data.

## Before you start

You need a corpus of a few thousand to tens of thousands of passages, and an **evaluation set**: 50–100 questions, each with the passages that should be retrieved. Build the evaluation set first. Without it you cannot tell whether any of this helped.

To build one, take a sample of passages, ask a model to write a question each one answers, and have a person check a sample of them. Add some real questions if you have them. Include questions that depend on exact terms and questions that depend on meaning.

## Steps

### Step 1: Index the corpus twice (30 min)

**Dense:** embed each passage and store it in Qdrant (or any vector store you used in Exercise 04). Use the same embedding model for passages and queries. Strong current open embedding models are listed on the MTEB leaderboard, and many of them handle long text and several languages. Pick one that fits your corpus.

**Keyword (BM25):** use a proper search index, not a Python loop. Options include Elasticsearch/OpenSearch, Postgres full-text search, Qdrant's sparse vectors, or the `bm25s` library for experiments. The older `rank_bm25` package works for a quick start but is slow. Tokenize in a way that suits your data: lowercase, remove punctuation, and be careful with code and identifiers.

### Step 2: Retrieve with each method (30 min)

```python
def dense_search(query, k=50):
    vec = embed(query)
    pts = qdrant.query_points("docs", query=vec, limit=k).points
    return [p.id for p in pts]

def keyword_search(query, k=50):
    return bm25_index.search(query, k)    # list of doc ids, best first
```

Retrieve more than you need (50 is common), because the next steps will narrow it.

### Step 3: Fuse the rankings (15 min)

Reciprocal Rank Fusion combines two ranked lists using only the ranks, so you do not need to compare the scores of different systems.

```python
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            score[doc_id] = score.get(doc_id, 0) + 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)
```

Many vector databases now do hybrid search with fusion for you in one call. Try that too and compare, since it removes code you would otherwise maintain.

### Step 4: Rerank the shortlist (45 min)

A reranker reads the question and a passage **together** and scores how well they match. It is slower than the first-stage search but more accurate, so you only run it on the top 30–50 candidates.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")     # or another current reranker

def rerank(query, doc_ids, k=5):
    pairs = [(query, corpus[i]) for i in doc_ids]
    scores = reranker.predict(pairs)
    return [i for i, _ in sorted(zip(doc_ids, scores), key=lambda x: -x[1])][:k]
```

Rerankers also come from hosted providers (Cohere, Voyage, Jina, and others) as an API. Compare one hosted and one open model on quality, latency, and cost. Also look at late-interaction models like ColBERT as an alternative to think about, though you do not need to build one here.

### Step 5: Build the answer with citations (30 min)

```python
def answer(question):
    top = rerank(question, rrf([dense_search(question), keyword_search(question)])[:50], k=5)
    context = "\n\n".join(f"[{n}] {corpus[i]}" for n, i in enumerate(top, start=1))
    msg = ("Answer using only the context. Cite sources like [1]. "
           "If the context does not contain the answer, say you don't know.\n\n"
           f"Context:\n{context}\n\nQuestion: {question}")
    reply = llm.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": msg}])
    return reply.choices[0].message.content, top
```

Then **check the citations in code.** Find every `[n]` in the reply and confirm that `n` was in the context you gave. Flag or strip any that were not. Also check that a cited passage really supports the claim, using a judge model on a sample.

### Step 6: Evaluate retrieval (30 min)

Measure the **retrieval step on its own**, before the model sees anything. Use the standard metrics:

- **Recall@k:** the share of the correct passages that appear in the top k.
- **MRR:** how high up the first correct passage appears.
- **nDCG@k,** if your questions have more than one correct passage.

Compare these four setups:

| Setup | Recall@5 | Recall@50 | MRR | p95 latency |
|---|---|---|---|---|
| Dense only | | | | |
| Keyword only | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + reranker | | | | |

Break the results down by question type (exact-term vs meaning-based). That is where hybrid usually shows its value.

### Step 7: Evaluate the answers (15 min)

Retrieval metrics are not enough. Sample 30–50 questions and check the final answers for **faithfulness** (does the answer stick to the context?) and **correctness**. Tools like Ragas can score this, or you can use the judge from [Exercise 12](../exercise-12-llm-evaluation-harness/README.md).

## Deliverables

1. The working pipeline: dense, keyword, fusion, rerank, and answer with citations.
2. The evaluation set and the results table.
3. A sample of answers showing the citations, plus your citation checker.
4. `RAG_DESIGN.md`: the design choices, what each step contributed, and what you would do differently with ten times more documents.

## Check yourself

- [ ] All four setups are measured on the same evaluation set.
- [ ] You can say which question types got better and which did not.
- [ ] The reranker's gain is compared with the latency it adds.
- [ ] Every citation in the output refers to a passage you actually supplied.

## Common pitfalls

- **Skipping the evaluation set.** Without it, "it feels better" is all you have.
- **Reranking too many candidates.** Reranking cost grows with the number of candidates. Keep the list to a few dozen.
- **Mismatched embeddings.** Queries and passages must use the same model, and re-embed everything if you change it.
- **Fake citations.** Models sometimes cite sources they were not given. Verify in code.
- **Judging only the final answer.** A bad answer can come from bad retrieval or from bad generation, and you need to know which.
- **Chunking ignored.** The way you split documents often matters more than the search method. Try two chunk sizes.
