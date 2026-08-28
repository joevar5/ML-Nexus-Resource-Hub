# Lesson 04: Vector Databases

## Table of Contents
1. [Introduction](#introduction)
2. [Vector Database Fundamentals](#vector-database-fundamentals)
3. [Detailed Comparison](#detailed-comparison)
4. [Qdrant Deep Dive](#qdrant-deep-dive)
5. [Weaviate Deep Dive](#weaviate-deep-dive)
6. [Chroma Deep Dive](#chroma-deep-dive)
7. [Pinecone Deep Dive](#pinecone-deep-dive)
8. [Milvus and FAISS](#milvus-and-faiss)
9. [Deployment Strategies](#deployment-strategies)
10. [Vector Indexing](#vector-indexing)
11. [Scaling Strategies](#scaling-strategies)
12. [Kubernetes Deployments](#kubernetes-deployments)
13. [Monitoring and Performance](#monitoring-and-performance)
14. [Cost Optimization](#cost-optimization)
15. [Summary](#summary)

## Introduction

Vector databases are specialized databases optimized for storing, indexing, and searching high-dimensional vectors (embeddings). They are essential infrastructure for RAG systems, semantic search, recommendation engines, and other AI applications.

### Learning Objectives

- Understand vector database architecture and indexing
- Compare major vector database options
- Deploy vector databases in production
- Optimize performance and costs
- Implement proper monitoring and scaling
- Choose the right vector database for your use case

## Vector Database Fundamentals

### What is a Vector Database?

A vector database is purpose-built for:
- Storing high-dimensional vectors (embeddings)
- Efficient similarity search (ANN - Approximate Nearest Neighbor)
- Metadata filtering alongside vector search
- Horizontal scaling for billions of vectors
- Real-time updates and queries

### Key Concepts

**Embeddings**: Dense vector representations of data
```python
text = "Machine learning is amazing"
embedding = [0.12, -0.45, 0.78, ..., 0.23]  # 768 dimensions
```

**Similarity Metrics**:
- **Cosine Similarity**: Measures angle between vectors
- **Euclidean Distance**: Straight-line distance
- **Dot Product**: Inner product of vectors

**Indexing Algorithms**:
- **HNSW** (Hierarchical Navigable Small World): Fast, memory-intensive
- **IVF** (Inverted File Index): Good balance
- **LSH** (Locality Sensitive Hashing): Fast approximate search
- **ANNOY**: Tree-based, memory-efficient

## Detailed Comparison

### Feature Comparison Matrix

| Feature | Qdrant | Weaviate | Chroma | Pinecone | Milvus |
|---------|--------|----------|--------|----------|--------|
| **Open Source** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Managed Cloud** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Language** | Rust | Go | Python | Proprietary | C++/Python |
| **Filtering** | Excellent | Excellent | Basic | Good | Excellent |
| **Performance** | Excellent | Very Good | Good | Excellent | Excellent |
| **Ease of Use** | Very Good | Good | Excellent | Excellent | Moderate |
| **Hybrid Search** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Multi-tenancy** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **GraphQL** | ❌ | ✅ | ❌ | ❌ | ❌ |

### When to Use Each

**Qdrant**:
```python
# High performance, complex filtering, self-hosted
use_cases = [
    "Production RAG systems",
    "High-throughput applications",
    "Complex metadata filtering",
    "Cost-conscious deployments"
]
```

**Weaviate**:
```python
# Hybrid search, GraphQL, knowledge graphs
use_cases = [
    "Hybrid vector + keyword search",
    "GraphQL API requirements",
    "Multi-modal search (text, images)",
    "Knowledge graph applications"
]
```

**Chroma**:
```python
# Development, prototyping, embedded
use_cases = [
    "Rapid prototyping",
    "Small to medium datasets",
    "Embedded in applications",
    "Development and testing"
]
```

**Pinecone**:
```python
# Fully managed, enterprise, scale
use_cases = [
    "Fully managed solution needed",
    "Enterprise support required",
    "Massive scale (billions of vectors)",
    "Don't want to manage infrastructure"
]
```

## Qdrant Deep Dive

### Architecture

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# Initialize client
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=768,  # Embedding dimension
        distance=Distance.COSINE
    )
)

# Insert vectors
points = [
    PointStruct(
        id=1,
        vector=np.random.rand(768).tolist(),
        payload={"text": "Sample document", "category": "tech"}
    )
    for i in range(1000)
]

client.upsert(
    collection_name="my_collection",
    points=points
)

# Search
results = client.search(
    collection_name="my_collection",
    query_vector=np.random.rand(768).tolist(),
    limit=10
)
```

### Advanced Features

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Filtered search
results = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="tech")
            )
        ]
    ),
    limit=10
)

# Scroll through all vectors
records, next_offset = client.scroll(
    collection_name="my_collection",
    limit=100,
    with_payload=True,
    with_vectors=False
)

# Batch operations
client.upsert(
    collection_name="my_collection",
    points=points,
    wait=True  # Wait for operation to complete
)
```

### Qdrant Production Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"  # gRPC port
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: always
```

```python
# Production configuration
client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,  # Use gRPC for better performance
    timeout=60
)

# Optimize for performance
client.create_collection(
    collection_name="production",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    ),
    optimizers_config={
        "memmap_threshold": 20000,  # Use memory-mapped files
        "indexing_threshold": 10000
    },
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    }
)
```

## Weaviate Deep Dive

### Setup and Basic Usage

```python
import weaviate

# Initialize client
client = weaviate.Client("http://localhost:8080")

# Create schema
schema = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "properties": [
        {
            "name": "content",
            "dataType": ["text"]
        },
        {
            "name": "category",
            "dataType": ["string"]
        }
    ]
}

client.schema.create_class(schema)

# Insert data (automatic vectorization)
client.data_object.create(
    {
        "content": "Machine learning is a subset of AI",
        "category": "technology"
    },
    "Document"
)

# Search
result = client.query.get(
    "Document",
    ["content", "category"]
).with_near_text({
    "concepts": ["artificial intelligence"]
}).with_limit(10).do()
```

### Hybrid Search

```python
# Combine vector and keyword search
result = client.query.get(
    "Document",
    ["content", "category"]
).with_hybrid(
    query="machine learning",
    alpha=0.5  # 0=keyword only, 1=vector only
).with_limit(10).do()
```

### Weaviate Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
    volumes:
      - ./weaviate_data:/var/lib/weaviate

  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      ENABLE_CUDA: '0'
```

## Chroma Deep Dive

### Setup and Usage

```python
import chromadb
from chromadb.config import Settings

# Persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)

# Add documents (automatic embedding with default model)
collection.add(
    documents=[
        "This is a document about machine learning",
        "This document is about natural language processing"
    ],
    metadatas=[
        {"category": "ML"},
        {"category": "NLP"}
    ],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=10
)

print(results)
```

### Custom Embedding Function

```python
from chromadb.utils import embedding_functions

# Use custom embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

collection = client.create_collection(
    name="custom_embeddings",
    embedding_function=sentence_transformer_ef
)

# Or use OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-ada-002"
)
```

## Pinecone Deep Dive

### Setup and Usage

```python
import pinecone

# Initialize
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# Create index
pinecone.create_index(
    "my-index",
    dimension=768,
    metric="cosine",
    pods=1,
    replicas=1,
    pod_type="p1.x1"
)

# Connect to index
index = pinecone.Index("my-index")

# Upsert vectors
vectors = [
    ("id1", [0.1] * 768, {"text": "Document 1"}),
    ("id2", [0.2] * 768, {"text": "Document 2"})
]

index.upsert(vectors=vectors)

# Query
results = index.query(
    vector=[0.15] * 768,
    top_k=10,
    include_metadata=True
)
```

### Pinecone Namespaces

```python
# Use namespaces for multi-tenancy
index.upsert(
    vectors=vectors,
    namespace="user_123"
)

# Query specific namespace
results = index.query(
    vector=query_vector,
    top_k=10,
    namespace="user_123"
)
```

## Deployment Strategies

### Self-Hosted vs. Managed

```python
deployment_decision_matrix = {
    "self_hosted": {
        "pros": [
            "Full control",
            "Lower long-term costs",
            "Data privacy",
            "Custom optimizations"
        ],
        "cons": [
            "Operations overhead",
            "Scaling complexity",
            "Maintenance burden"
        ],
        "recommended_for": [
            "Qdrant",
            "Weaviate",
            "Milvus",
            "Chroma"
        ]
    },
    "managed": {
        "pros": [
            "No operations overhead",
            "Automatic scaling",
            "Enterprise support",
            "Managed backups"
        ],
        "cons": [
            "Higher costs",
            "Less control",
            "Vendor lock-in"
        ],
        "recommended_for": [
            "Pinecone",
            "Weaviate Cloud",
            "Qdrant Cloud"
        ]
    }
}
```

### Kubernetes Deployment (Qdrant)

```yaml
# qdrant-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
spec:
  clusterIP: None
  selector:
    app: qdrant
  ports:
  - port: 6333
    name: http
  - port: 6334
    name: grpc
```

## Vector Indexing

### HNSW (Hierarchical Navigable Small World)

```python
# HNSW is the most popular indexing algorithm

# Qdrant HNSW configuration
client.create_collection(
    collection_name="hnsw_collection",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    ),
    hnsw_config={
        "m": 16,  # Number of connections per layer
        "ef_construct": 100,  # Size of dynamic candidate list
        "full_scan_threshold": 10000
    }
)

# Tuning parameters:
# m: Higher = better recall, more memory
# ef_construct: Higher = better index quality, slower indexing
# ef (search-time): Higher = better recall, slower search
```

### IVF (Inverted File Index)

```python
# Common in FAISS and Milvus
import faiss

# Create IVF index
dimension = 768
nlist = 100  # Number of clusters

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train index
index.train(training_vectors)

# Add vectors
index.add(vectors)

# Search
index.nprobe = 10  # Number of clusters to search
distances, indices = index.search(query_vectors, k=10)
```

## Scaling Strategies

### Horizontal Scaling

```python
class VectorDBScaling:
    """Scaling strategies for vector databases"""

    @staticmethod
    def sharding_strategy():
        """
        Split data across multiple nodes
        """
        strategies = {
            "by_tenant": "Shard by user/tenant ID",
            "by_category": "Shard by metadata category",
            "by_hash": "Hash-based sharding",
            "by_time": "Time-based partitioning"
        }
        return strategies

    @staticmethod
    def replication_strategy():
        """
        Replicate data for high availability
        """
        configs = {
            "qdrant": {
                "replication_factor": 2,
                "write_consistency_factor": 1
            },
            "weaviate": {
                "replicationFactor": 2
            }
        }
        return configs
```

### Performance Optimization

```python
optimization_techniques = {
    "indexing": {
        "hnsw_tuning": "Adjust m and ef_construct",
        "quantization": "Use scalar or product quantization",
        "filtering": "Use payload indexes for metadata"
    },
    "search": {
        "batch_queries": "Process multiple queries together",
        "ef_tuning": "Adjust search-time ef parameter",
        "result_caching": "Cache frequent queries"
    },
    "storage": {
        "quantization": "Reduce memory footprint",
        "mmap": "Use memory-mapped files",
        "compression": "Compress stored data"
    }
}
```

## Monitoring and Performance

### Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Vector database metrics
VECTOR_DB_QUERIES = Counter(
    'vector_db_queries_total',
    'Total vector DB queries',
    ['collection', 'status']
)

VECTOR_DB_LATENCY = Histogram(
    'vector_db_query_duration_seconds',
    'Query duration',
    ['collection']
)

VECTOR_DB_SIZE = Gauge(
    'vector_db_collection_size',
    'Number of vectors in collection',
    ['collection']
)

INDEX_QUALITY = Gauge(
    'vector_db_index_quality',
    'Index quality metrics',
    ['collection', 'metric']
)
```

### Qdrant Monitoring

```python
# Get collection info
info = client.get_collection("my_collection")

metrics = {
    "vectors_count": info.vectors_count,
    "indexed_vectors_count": info.indexed_vectors_count,
    "points_count": info.points_count,
    "segments_count": len(info.segments),
    "status": info.status
}

# Monitor index quality
for segment in info.segments:
    print(f"Segment: {segment.segment_id}")
    print(f"  Vectors: {segment.num_vectors}")
    print(f"  Deleted: {segment.num_deleted_vectors}")
```

## Cost Optimization

### Strategies

```python
class CostOptimization:
    """Cost optimization for vector databases"""

    @staticmethod
    def quantization():
        """
        Reduce memory and storage costs
        """
        return {
            "scalar_quantization": "4-8x reduction, minimal quality loss",
            "product_quantization": "16-64x reduction, moderate quality loss",
            "binary_quantization": "32x reduction, significant quality loss"
        }

    @staticmethod
    def rightsizing():
        """
        Choose appropriate instance sizes
        """
        guidelines = {
            "development": "Small instances, Chroma or single Qdrant",
            "production_small": "1-2 nodes, moderate resources",
            "production_large": "3+ nodes, high resources, replication"
        }
        return guidelines

    @staticmethod
    def caching():
        """
        Reduce query costs
        """
        return {
            "query_caching": "Cache frequent queries",
            "result_caching": "Cache search results",
            "embedding_caching": "Cache embeddings"
        }
```

### Cost Comparison

```python
monthly_cost_estimates = {
    "qdrant_cloud": {
        "starter": 25,  # 1GB
        "standard": 100,  # 4GB
        "business": 500   # 32GB
    },
    "pinecone": {
        "starter": 70,   # 1 pod
        "standard": 140,  # 2 pods
        "enterprise": 500  # Custom
    },
    "self_hosted_qdrant": {
        "small": 30,  # 4GB RAM instance
        "medium": 100,  # 16GB RAM instance
        "large": 300   # 64GB RAM instance
    },
    "weaviate_cloud": {
        "sandbox": 0,    # Free tier
        "standard": 100,  # Production
        "enterprise": 500  # Custom
    }
}
```

## Summary

Vector databases are critical infrastructure for LLM applications. Key takeaways:

1. **Choose based on needs**: Qdrant for performance, Weaviate for hybrid search, Chroma for development, Pinecone for managed
2. **Indexing matters**: HNSW for most cases, tune parameters for your workload
3. **Scale appropriately**: Use sharding and replication for large deployments
4. **Monitor actively**: Track query latency, index quality, and costs
5. **Optimize costs**: Quantization, caching, and rightsizing save money

---

**Next Lesson**: [05-llm-fine-tuning-infrastructure.md](./05-llm-fine-tuning-infrastructure.md)
