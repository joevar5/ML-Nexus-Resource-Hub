# Lab 03: Stand Up a Vector Database (Qdrant)

**Duration:** 60 min  **Prerequisites:** Docker

## Objective
Run Qdrant locally, create a collection, upsert vectors with metadata, and run filtered nearest-neighbor queries.

## Steps

### 1. Run Qdrant
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.10.0
```
HTTP API on 6333; gRPC on 6334. Dashboard at http://localhost:6333/dashboard.

### 2. Create a collection
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

### 3. Upsert with payload
```python
from qdrant_client.http.models import PointStruct
import numpy as np

points = [
    PointStruct(
        id=i,
        vector=np.random.rand(384).tolist(),
        payload={"category": np.random.choice(["a","b","c"]), "ts": 1717000000 + i*60},
    )
    for i in range(1000)
]
client.upsert(collection_name="docs", points=points)
print(client.count("docs"))
```

### 4. Plain nearest-neighbor
```python
q = np.random.rand(384).tolist()
hits = client.search(collection_name="docs", query_vector=q, limit=5)
for h in hits:
    print(h.id, h.score, h.payload)
```

### 5. Filtered search
```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
hits = client.search(
    collection_name="docs",
    query_vector=q,
    query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="a"))]),
    limit=5,
)
```

### 6. Update payload (no need to re-embed)
```python
client.set_payload(collection_name="docs", payload={"tag": "important"}, points=[1, 2, 3])
```

### 7. Snapshot for backup
```python
client.create_snapshot(collection_name="docs")
client.list_snapshots(collection_name="docs")
```

### 8. Pair with the RAG pipeline (lab 02)
Replace the FAISS in-memory index with Qdrant. Production gains: persistence, filtering, multi-collection.

## Validation
- [ ] Collection visible in Qdrant dashboard with 1000 points.
- [ ] Nearest-neighbor returns 5 results ordered by score.
- [ ] Filtered search returns only points with `category=a`.

## Cleanup
```bash
docker stop qdrant && docker rm qdrant
```

## Troubleshooting
- **Snapshot fails** — Qdrant needs a writable volume; the docker run above uses the default ephemeral volume.
- **Filter returns 0 results** — Check field name spelling; payload keys are case-sensitive.
- **`Wrong vector size`** — Collection size must match vector dim exactly.
