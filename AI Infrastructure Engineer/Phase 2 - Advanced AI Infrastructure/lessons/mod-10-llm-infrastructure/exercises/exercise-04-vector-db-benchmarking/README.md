# Exercise 04: Vector Database Benchmarking

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** [Lab 03](../../labs/lab-03-vector-db-qdrant.md) and [Lesson 04: Vector Databases](../../04-vector-databases.md)
**Hardware:** A laptop with Docker and about 16 GB of RAM is enough for 1M vectors

## Goal

Load the same vectors into Qdrant, Weaviate, and pgvector, run the same queries, and compare speed, accuracy, and memory. Then write a recommendation that is based on your numbers.

## Why it matters

A vector database choice sticks with you. It shapes your operations, your costs, and your team's habits for years. Marketing pages tell you what each one is good at in the best case. A benchmark on your own data tells you what it does for you.

Many teams also find that the "boring" option wins: if you already run Postgres and have a few million vectors, pgvector may be all you need. This exercise gives you the evidence to decide.

## Before you start

The three databases must be compared fairly. Decide these up front and use them everywhere:

- **Distance metric:** cosine for all three. Cosine and L2 give different rankings.
- **Index type:** HNSW for all three, with the same `m` (for example 16) and `ef_construction` (for example 128).
- **Search setting:** the same `ef` at query time, so you compare speed at the same accuracy.
- **Same machine,** with the same CPU and memory limits on every container.

## Steps

### Step 1: Build the dataset (30 min)

You need about 1M vectors with 384 dimensions, plus a "category" field for the filter test. The `wikipedia` dataset script on Hugging Face may no longer load directly, so use any text corpus you can load, such as `wikimedia/wikipedia` (`20231101.simple`), or generate documents from a dataset of your choice.

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
texts = [t[:200] for t in ds["text"][:1_000_000]]
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
# 1M x 384 float32 is about 1.5 GB. Save it with numpy so you only compute it once.
```

If the corpus has fewer than 1M rows, use what you have, or start with 100K to get your harness working first and scale up after.

Assign each vector a category (for example 20 values chosen at random) so you can test filtered search.

### Step 2: Create ground truth (15 min)

To measure accuracy you need the true nearest neighbors. Take 1,000 query vectors, compute exact top-10 results by brute force with numpy or FAISS `IndexFlatIP`, and save them. Recall@10 is the share of those true neighbors that each database returns.

### Step 3: Qdrant (45 min)

Run Qdrant in Docker, create a collection with cosine distance and your HNSW settings, upload the vectors in batches, and wait for indexing to finish. Record the time and memory (`docker stats`).

### Step 4: Weaviate (45 min)

Run Weaviate in Docker with a current release tag (check the Weaviate docs for the latest). Set HNSW to the same `m` and `ef_construction`, and upload the same vectors with the same IDs.

### Step 5: pgvector (45 min)

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=app pgvector/pgvector:pg17
```

```sql
CREATE EXTENSION vector;
CREATE TABLE docs (id int, category text, embedding vector(384));
-- load data with COPY, then build the index AFTER loading
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
SET hnsw.ef_search = 100;   -- per session, at query time
```

Build the index after the bulk load. Building it first makes loading much slower.

### Step 6: Benchmark harness (45 min)

Write one script that works with all three through a small adapter function (`search(vector, k, filter)`). For each database:

1. Warm up with 100 queries you throw away.
2. Run the 1,000 queries at concurrency 1, then at concurrency 32 with a thread pool.
3. Record p50 and p95 latency, queries per second, and recall@10 against your ground truth.

### Step 7: Filtered search (15 min)

Repeat with a filter such as `category = "science"`. Test two cases: a filter that matches about 50% of the data, and one that matches under 1%. The narrow filter is where databases differ most, because the index has to work around the filter without returning too few results. Check that you still get 10 results back.

### Step 8: Decision matrix (15 min)

| Metric | Qdrant | Weaviate | pgvector |
|---|---|---|---|
| Index build time | | | |
| Memory after indexing | | | |
| Query p50 / p95, concurrency 1 | | | |
| Queries per second, concurrency 32 | | | |
| Recall@10 | | | |
| Filtered query, 50% match | | | |
| Filtered query, under 1% match | | | |
| What you have to run and maintain | | | |

## Deliverables

1. The loading and benchmark scripts for each database.
2. `RESULTS.md` with the matrix above.
3. `RECOMMENDATION.md`: when to pick each one, based on your numbers. Include a rule such as "if you already run Postgres and have under N million vectors, use pgvector".

## Check yourself

- [ ] All three use the same metric, index settings, and data.
- [ ] Recall@10 is at least 0.95 for each. If one is far below, compare its search setting with the others.
- [ ] You ran each benchmark at least twice and the numbers agree.
- [ ] The recommendation names a case where each database wins.

## Common pitfalls

- **Comparing speed at different accuracy.** A faster query that returns worse results is not faster. Always read latency and recall together.
- **Different index settings.** HNSW `m` and `ef` change both speed and recall. Match them.
- **No warm-up.** The first queries are slow while caches fill.
- **Benchmarking on a tiny dataset.** HNSW behaves differently at 10K and at 10M. Use enough data to matter for your case.
- **Ignoring the filter.** Filtering with a rare value is a common source of slow queries and missing results in production.
