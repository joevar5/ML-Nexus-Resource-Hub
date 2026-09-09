# Day 13: RAG Basics

## Concept

Retrieval-Augmented Generation solves a specific problem: an LLM's knowledge is frozen at training time and doesn't include your private data. RAG fixes this by retrieving relevant text from an external source at query time and putting it in the prompt, rather than relying on what the model memorized.

The core pipeline, four stages:

```
1. Index (offline, done once/periodically):
   documents -> split into chunks -> embed each chunk -> store vectors in a vector DB

2. Query (per request):
   user question -> embed the question -> search vector DB for nearest chunks
   -> insert retrieved chunks into the prompt -> generate answer
```

```python
def rag_answer(question: str, vector_db, llm, k=5) -> str:
    query_embedding = embed(question)
    chunks = vector_db.search(query_embedding, top_k=k)
    context = "\n\n".join(c.text for c in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context above."
    return llm.call(prompt)
```

Why retrieval instead of just fine-tuning on your data: fine-tuning bakes facts into weights, which is expensive to update, doesn't tell you *where* an answer came from, and doesn't reliably prevent the model from also drawing on unrelated training data. RAG keeps the source data external and swappable — update the index, the next query sees new data immediately, no retraining. The tradeoff is that RAG's quality is bottlenecked entirely by retrieval quality: if the wrong chunks come back, the model confidently answers from wrong information, and there's no mechanism inside the generation step to notice.

This is why RAG is not "vector search + LLM call" as an afterthought — chunking strategy (Day 14), whether the model actually cites what it used (Day 15), and evaluating whether retrieval is working (Days 18–19) are the real engineering surface area, not the LLM call itself.

## Coding Problem

Write `rag_answer(question, vector_db, llm, k)` matching the pattern above, but add a guard: if `vector_db.search()` returns zero chunks (empty index, or no match above a similarity threshold the DB enforces), skip the LLM call entirely and return a fixed string `"No relevant information found."` rather than sending an empty-context prompt to the model — an empty-context RAG call is a common source of confident hallucination, since the model will still try to answer from general knowledge without saying so.

## Quiz

### Question 1: Why RAG Instead of Fine-Tuning

**What is the main practical advantage of RAG over fine-tuning for keeping an LLM up to date with private/changing data?**

A) RAG produces smaller model files
B) The source data stays external and swappable — updating the index makes new data available immediately, with no retraining required
C) Fine-tuning is not possible for any LLM
D) RAG always produces more accurate answers regardless of retrieval quality

**Answer**: B

**Explanation**: RAG decouples data freshness from model training — you update the index, not the weights. Fine-tuning requires a retraining cycle for every data update and doesn't provide a clear mechanism for citing sources.

### Question 2: The RAG Bottleneck

**Why is retrieval quality described as the bottleneck for the entire RAG pipeline?**

A) Retrieval is the most computationally expensive step
B) If the wrong chunks are retrieved, the model has no way to know they're wrong and will confidently generate an answer from incorrect context
C) Retrieval always returns perfect results, so this isn't actually a concern
D) The LLM ignores retrieved context entirely

**Answer**: B

**Explanation**: The generation step trusts whatever context it's given — it has no independent way to verify retrieved chunks are relevant or correct. A retrieval failure becomes a generation failure that looks just as confident as a correct answer.

### Question 3: Empty Retrieval Results

**Why does the coding problem special-case zero retrieved chunks instead of just sending an empty context to the LLM?**

A) The LLM API rejects empty context strings
B) With empty (or irrelevant) context, the model will often still answer from general training knowledge without signaling that it lacks grounded information — a common source of confident hallucination in RAG systems
C) Empty context always produces an API error
D) This case never actually happens in practice

**Answer**: B

**Explanation**: An LLM prompted to "answer using the context" with no real context provided doesn't reliably refuse — it often answers anyway from its general knowledge, which defeats the purpose of RAG (grounded, sourced answers) without any visible signal that it did so.

## Interview Practice

**1.** A stakeholder asks "why not just fine-tune the model on our documents instead of building a RAG pipeline?" Give a substantive answer covering at least two concrete tradeoffs, not just "RAG is standard practice."

**2.** Walk through what happens end-to-end when a user asks a question that's outside your knowledge base's coverage — from the vector search through to what's shown to the user. Identify the point where a bad implementation would let this fail silently.

**3.** Describe a case where you'd deliberately choose not to use RAG for a question-answering feature, even though the questions are about your own data. What property of that scenario makes RAG the wrong tool?
