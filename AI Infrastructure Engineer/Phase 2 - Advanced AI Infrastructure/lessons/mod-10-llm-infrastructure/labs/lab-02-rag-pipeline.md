# Lab 02: Build a RAG Pipeline End-to-End

**Duration:** 90 min  **Prerequisites:** Lab 01 vLLM running; Qdrant (lab 03) or in-memory FAISS

## Objective
Ingest a small document corpus, embed it, store in a vector DB, retrieve top-k for a user query, and pass to the LLM via prompt template. Measure retrieval precision and end-to-end latency.

## Steps

### 1. Sample corpus
Use the Python docs ToC or any small set of markdown files.
```bash
mkdir corpus && cd corpus
curl -L https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/introduction.rst -o intro.rst
curl -L https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/controlflow.rst -o controlflow.rst
curl -L https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/datastructures.rst -o datastructures.rst
```

### 2. Chunking
```python
# chunk.py
from pathlib import Path
def chunk(text, size=500, overlap=50):
    chunks=[]; i=0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

docs = []
for p in Path("corpus").glob("*.rst"):
    for j, c in enumerate(chunk(p.read_text())):
        docs.append({"id": f"{p.stem}-{j}", "source": p.name, "text": c})
print(f"{len(docs)} chunks")
```

### 3. Embed
```python
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = emb.encode([d["text"] for d in docs], normalize_embeddings=True)
```

### 4. Index in FAISS (in-memory; for prod use Qdrant per lab 03)
```python
import faiss, numpy as np
index = faiss.IndexFlatIP(vectors.shape[1])    # cosine via normalized inner product
index.add(np.asarray(vectors, dtype="float32"))
```

### 5. Retrieve
```python
def retrieve(query, k=4):
    q = emb.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    return [(docs[i], float(D[0][j])) for j, i in enumerate(I[0])]

print(retrieve("how do for loops work in python"))
```

### 6. Compose prompt and call LLM
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def answer(question):
    hits = retrieve(question, k=4)
    context = "\n\n".join(f"[{h[0]['source']}#{h[0]['id']}]\n{h[0]['text']}" for h in hits)
    prompt = f"""You are answering questions strictly using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer (cite sources by id):"""
    r = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300, temperature=0.2,
    )
    return r.choices[0].message.content

print(answer("How do I write a for loop iterating over a list?"))
```

### 7. Evaluation harness
```python
test_set = [
    ("How do for loops work in python", ["controlflow"]),
    ("What is a dictionary in python", ["datastructures"]),
]
for q, expected_sources in test_set:
    hits = retrieve(q, k=4)
    sources = {h[0]["source"].replace(".rst","") for h in hits}
    print(q, "->", "PASS" if any(e in sources for e in expected_sources) else "FAIL", sources)
```

## Validation
- [ ] Retrieval returns relevant chunks for known queries.
- [ ] Answer cites source ids when context is sufficient.
- [ ] Answer says "I don't know" when context doesn't contain the answer (test by asking about Rust).

## Cleanup
```bash
cd .. && rm -rf corpus
```

## Troubleshooting
- **Hallucinated answers** — Lower temperature; tighten the "don't know" instruction; verify retrieved chunks actually contain the answer.
- **Wrong source retrieved** — Chunk size too small or too large. 300-800 chars works for prose; smaller for code.
- **Slow first embedding** — Model download on first run; cache lives in `~/.cache/torch/sentence_transformers/`.
