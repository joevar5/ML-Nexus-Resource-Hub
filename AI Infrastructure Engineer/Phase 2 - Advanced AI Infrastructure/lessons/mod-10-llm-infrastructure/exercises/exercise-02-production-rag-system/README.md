# Exercise 02: Production RAG System (Retrieval-Augmented Generation)

**Estimated Time**: 34-42 hours

## Business Context

Your company has 10 years of proprietary documentation (product manuals, internal wikis, customer support tickets) and wants to build an AI assistant that can answer questions using this knowledge base.

**Current Situation**:
- Using GPT-4 API for customer support chatbot
- **Problem**: GPT-4 doesn't know about company-specific products/processes
- **Result**: 60% of answers are generic or incorrect
- Customer satisfaction: 3.2/5 stars
- Support agents manually intervene in 70% of conversations

**The Solution**: Build a **Retrieval-Augmented Generation (RAG) system** that:
1. Embeds all company documentation into vector database
2. Retrieves relevant context for user questions
3. Generates accurate answers using company knowledge
4. Reduces hallucinations from 40% → <5%

**Business Goals**:
- **Improve accuracy**: 60% → 90% correct answers
- **Reduce support costs**: $500K/year (fewer manual interventions)
- **Improve customer satisfaction**: 3.2/5 → 4.5/5 stars
- **Data privacy**: All data stays in-house
- **Scalability**: Handle 10,000 queries/day

## Learning Objectives

After completing this exercise, you will be able to:

1. Design and implement production RAG architecture
2. Deploy and configure vector databases (Qdrant)
3. Implement document chunking strategies for optimal retrieval
4. Build embedding pipelines with sentence transformers
5. Optimize retrieval quality through re-ranking and hybrid search
6. Implement RAG evaluation metrics
7. Handle RAG challenges: context length, relevance, citations
8. Build production-grade RAG API with caching and monitoring

## Prerequisites

- Module 10 Exercise 01 (LLM Serving) - vLLM deployment
- Understanding of vector embeddings and similarity search
- Python programming (advanced)
- Kubernetes fundamentals
- Basic NLP concepts

## Problem Statement

Build a **Production RAG System** that:

1. **Document Ingestion Pipeline**:
   - Load documents from multiple sources (PDF, HTML, Markdown)
   - Chunk documents intelligently
   - Generate embeddings
   - Store in vector database

2. **Retrieval Engine**:
   - Semantic search using embeddings
   - Hybrid search (dense + sparse)
   - Re-ranking for relevance
   - Return top-k most relevant chunks

3. **Generation Layer**:
   - LLM integration (Llama 2, Mistral)
   - Context-aware prompting
   - Citation extraction
   - Hallucination detection

4. **Production Features**:
   - Query caching for common questions
   - Real-time document updates
   - Multi-user isolation
   - Comprehensive monitoring

5. **Evaluation Framework**:
   - Retrieval quality metrics
   - Answer accuracy assessment
   - A/B testing infrastructure

### Success Metrics

- Retrieval accuracy: 90% (correct documents in top-5)
- Answer accuracy: 90% (correct, grounded answers)
- Query latency: <2 seconds end-to-end
- Handle 10,000 queries/day
- Support 1M+ documents
- Hallucination rate: <5% (down from 40%)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Production RAG System Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Document Ingestion Pipeline                  │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │  │   Load      │─▶│   Chunk     │─▶│   Embed     │     │  │
│  │  │  Documents  │  │  Splitter   │  │ SentenceT5  │     │  │
│  │  │  (PDF/HTML) │  │  - Overlap  │  │  (384-dim)  │     │  │
│  │  └─────────────┘  │  - Metadata │  └─────────────┘     │  │
│  │                    └─────────────┘          │            │  │
│  │                                              ▼            │  │
│  │                                    ┌─────────────────┐   │  │
│  │                                    │  Qdrant         │   │  │
│  │                                    │  Vector DB      │   │  │
│  │                                    │  - 1M vectors   │   │  │
│  │                                    │  - HNSW index   │   │  │
│  │                                    └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  RAG Query Pipeline                       │  │
│  │                                                           │  │
│  │  User Query: "How do I reset password?"                 │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  ┌─────────────┐                                         │  │
│  │  │  Embed      │  (Convert query to 384-dim vector)     │  │
│  │  │  Query      │                                         │  │
│  │  └─────────────┘                                         │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  ┌─────────────────────────────────────────┐            │  │
│  │  │  Retrieval (Hybrid Search)              │            │  │
│  │  │                                          │            │  │
│  │  │  1. Dense Search (Vector Similarity)    │            │  │
│  │  │     - Cosine similarity                 │            │  │
│  │  │     - Retrieve top 20 chunks            │            │  │
│  │  │                                          │            │  │
│  │  │  2. Sparse Search (BM25 Keywords)       │            │  │
│  │  │     - Exact keyword matching            │            │  │
│  │  │     - Retrieve top 20 chunks            │            │  │
│  │  │                                          │            │  │
│  │  │  3. Fusion (Combine Results)            │            │  │
│  │  │     - Reciprocal Rank Fusion            │            │  │
│  │  │     - Top 10 chunks                     │            │  │
│  │  └─────────────────────────────────────────┘            │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  ┌─────────────┐                                         │  │
│  │  │  Re-Rank    │  (Cross-encoder for relevance)         │  │
│  │  │  Top 5      │  → [Doc1: 0.95, Doc2: 0.89, ...]      │  │
│  │  └─────────────┘                                         │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  ┌─────────────────────────────────────────┐            │  │
│  │  │  LLM Generation                          │            │  │
│  │  │                                          │            │  │
│  │  │  Prompt:                                 │            │  │
│  │  │  "Based on the following context:       │            │  │
│  │  │   [Retrieved chunks]                     │            │  │
│  │  │   Answer: How do I reset password?"     │            │  │
│  │  │                                          │            │  │
│  │  │  Response:                               │            │  │
│  │  │  "To reset your password:               │            │  │
│  │  │   1. Go to Settings > Account           │            │  │
│  │  │   2. Click 'Reset Password'             │            │  │
│  │  │   [Citation: user-guide.pdf, p.12]"     │            │  │
│  │  └─────────────────────────────────────────┘            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Evaluation & Monitoring                      │  │
│  │                                                           │  │
│  │  - Retrieval Precision/Recall                            │  │
│  │  - Answer correctness (vs ground truth)                  │  │
│  │  - Citation accuracy                                      │  │
│  │  - Hallucination detection                               │  │
│  │  - Query latency breakdown                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Document Ingestion Pipeline (8-10 hours)

Build pipeline to load, chunk, and embed documents.

#### 1.1 Document Loader

Create `src/ingestion/document_loader.py`:

```python
"""
TODO: Load documents from various sources.

Supported formats:
- PDF (PyPDF2, pdfplumber)
- HTML (BeautifulSoup)
- Markdown
- Plain text
- DOCX
"""

from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
import PyPDF2
from bs4 import BeautifulSoup
import markdown

@dataclass
class Document:
    """Document with content and metadata."""
    content: str
    metadata: Dict  # source, title, page, url, etc.
    doc_id: str

class PDFLoader:
    """Load PDF documents."""

    def load(self, file_path: str) -> List[Document]:
        """
        TODO: Extract text from PDF.

        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)

            documents = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                # Create document per page (for better granularity)
                doc = Document(
                    content=text,
                    metadata={
                        'source': file_path,
                        'page': page_num + 1,
                        'total_pages': len(pdf.pages),
                        'type': 'pdf'
                    },
                    doc_id=f"{Path(file_path).stem}_page_{page_num + 1}"
                )
                documents.append(doc)

        return documents
        """
        pass

class HTMLLoader:
    """Load HTML documents."""

    def load(self, file_path: str) -> Document:
        """
        TODO: Parse HTML and extract text.

        with open(file_path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

            # Remove script and style tags
            for script in soup(['script', 'style']):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Extract title
            title = soup.find('title')
            title_text = title.string if title else Path(file_path).stem

            return Document(
                content=text,
                metadata={
                    'source': file_path,
                    'title': title_text,
                    'type': 'html'
                },
                doc_id=Path(file_path).stem
            )
        """
        pass

class DocumentLoader:
    """Main document loader - auto-detects format."""

    def __init__(self):
        self.loaders = {
            '.pdf': PDFLoader(),
            '.html': HTMLLoader(),
            '.htm': HTMLLoader(),
            # TODO: Add more loaders
        }

    def load(self, file_path: str) -> List[Document]:
        """
        TODO: Load document based on file extension.

        suffix = Path(file_path).suffix.lower()
        loader = self.loaders.get(suffix)

        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader.load(file_path)
        """
        pass

    def load_directory(self, directory: str) -> List[Document]:
        """
        TODO: Recursively load all documents in directory.

        documents = []
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix in self.loaders:
                try:
                    docs = self.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents
        """
        pass
```

#### 1.2 Intelligent Chunking

Create `src/ingestion/chunker.py`:

```python
"""
TODO: Chunk documents for optimal retrieval.

Chunking strategies:
1. Fixed-size chunks (simple, works well)
2. Sentence-based chunks (preserves meaning)
3. Semantic chunks (split by topic)
4. Recursive chunks (hierarchical)

Key parameters:
- chunk_size: 512 tokens (fits in most embedding models)
- chunk_overlap: 50 tokens (preserves context across chunks)
"""

from typing import List
from dataclasses import dataclass
import tiktoken

@dataclass
class Chunk:
    """Document chunk with metadata."""
    content: str
    metadata: Dict
    chunk_id: str
    embedding: Optional[List[float]] = None

class RecursiveCharacterSplitter:
    """
    Split text recursively by separators.

    Hierarchy:
    1. Try splitting by paragraphs (\n\n)
    2. If too large, split by sentences (. ! ?)
    3. If still too large, split by words
    4. Last resort: split by characters

    This preserves semantic boundaries better than fixed-size splits.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    def split(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        TODO: Recursively split text into chunks.

        Algorithm:
        1. Try splitting by first separator (e.g., \n\n)
        2. For each piece:
           a. If piece < chunk_size: Keep as chunk
           b. If piece > chunk_size: Try next separator
           c. If at last separator: Force split at chunk_size
        3. Add overlap between chunks (last N tokens of chunk[i] become
           first N tokens of chunk[i+1])

        Return list of Chunk objects with metadata preserved
        """
        chunks = []

        # TODO: Implement recursive splitting
        # ...

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

class SemanticChunker:
    """
    Split text based on semantic similarity.

    Uses embeddings to detect topic changes:
    1. Split text into sentences
    2. Embed each sentence
    3. Calculate similarity between adjacent sentences
    4. Split where similarity drops (topic change)

    More sophisticated but slower than recursive splitter.
    """

    def __init__(self, embedding_model, similarity_threshold: float = 0.7):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def split(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        TODO: Split based on semantic boundaries.

        1. Split into sentences
        2. Embed all sentences
        3. Calculate cosine similarity between adjacent pairs
        4. Where similarity < threshold: Insert split point
        5. Group sentences into chunks
        """
        pass
```

#### 1.3 Embedding Generation

Create `src/ingestion/embedder.py`:

```python
"""
TODO: Generate embeddings for chunks.

Embedding models:
- sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)
- sentence-transformers/all-mpnet-base-v2 (768-dim, quality)
- BAAI/bge-large-en-v1.5 (1024-dim, SOTA)

Choice: all-MiniLM-L6-v2 for speed/quality balance
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch

class EmbeddingModel:
    """Generate embeddings for text chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        TODO: Generate embeddings for list of texts.

        # Batch encode for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        return embeddings  # Shape: (len(texts), dimension)
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """
        TODO: Embed single query.

        # Some models have instruction prefixes for queries
        # Example: "Represent this sentence for searching relevant passages:"
        # For all-MiniLM, no prefix needed

        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        """
        pass

class EmbeddingPipeline:
    """Complete embedding pipeline."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.model = embedding_model

    def process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        TODO: Add embeddings to chunks.

        # Extract text from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        embeddings = self.model.embed(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        return chunks
        """
        pass
```

### Part 2: Vector Database Setup (7-9 hours)

Deploy and configure Qdrant vector database.

#### 2.1 Qdrant Deployment

Create `kubernetes/qdrant/deployment.yaml`:

```yaml
# TODO: Deploy Qdrant vector database

apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: rag-system
spec:
  serviceName: qdrant
  replicas: 3  # HA with replication
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
          image: qdrant/qdrant:v1.7.4
          ports:
            - containerPort: 6333
              name: http
            - containerPort: 6334
              name: grpc
          env:
            # Cluster configuration
            - name: QDRANT__CLUSTER__ENABLED
              value: "true"
            - name: QDRANT__CLUSTER__P2P__PORT
              value: "6335"
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
          livenessProbe:
            httpGet:
              path: /healthz
              port: 6333
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /readyz
              port: 6333
            initialDelaySeconds: 10
            periodSeconds: 5

  volumeClaimTemplates:
    - metadata:
        name: qdrant-storage
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: rag-system
spec:
  selector:
    app: qdrant
  ports:
    - port: 6333
      targetPort: 6333
      name: http
    - port: 6334
      targetPort: 6334
      name: grpc
  type: ClusterIP
```

#### 2.2 Qdrant Client

Create `src/vectordb/qdrant_client.py`:

```python
"""
TODO: Qdrant client for vector operations.

Operations:
- Create collection
- Upsert vectors
- Search similar vectors
- Delete vectors
- Manage indexes (HNSW)
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest
)
from typing import List, Dict, Optional
import uuid

class VectorDatabase:
    """Qdrant vector database client."""

    def __init__(self, host: str = "qdrant", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: str = "Cosine"
    ):
        """
        TODO: Create vector collection.

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE  # Cosine similarity
            ),
            # HNSW index configuration
            hnsw_config={
                "m": 16,  # Number of connections per layer
                "ef_construct": 100  # Search quality during build
            }
        )

        # Create payload index for metadata filtering
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema="keyword"
        )
        """
        pass

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Chunk]
    ):
        """
        TODO: Insert/update chunks in vector database.

        points = []
        for chunk in chunks:
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID
                vector=chunk.embedding,
                payload={
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'chunk_id': chunk.chunk_id
                }
            )
            points.append(point)

        # Batch upsert (efficient for large datasets)
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        """
        pass

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        TODO: Search for similar vectors.

        # Build filter if provided
        query_filter = None
        if filter_conditions:
            # Example: filter by source
            # filter_conditions = {'source': 'user-guide.pdf'}
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Execute search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False  # Don't return vectors (save bandwidth)
        )

        # Format results
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'content': hit.payload['content'],
                'metadata': hit.payload['metadata']
            }
            for hit in results
        ]
        """
        pass
```

### Part 3: Retrieval Engine with Re-Ranking (7-9 hours)

Implement hybrid search and re-ranking for better retrieval quality.

#### 3.1 Hybrid Search

Create `src/retrieval/hybrid_search.py`:

```python
"""
TODO: Hybrid search combining dense + sparse retrieval.

Dense (Vector): Semantic similarity
Sparse (BM25): Keyword matching
Fusion: Reciprocal Rank Fusion

Hybrid often outperforms either method alone.
"""

from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """Combine dense and sparse retrieval."""

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_model: EmbeddingModel,
        collection_name: str
    ):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # TODO: Build BM25 index
        self.bm25_index = None
        self.documents = []  # For BM25
        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        TODO: Build BM25 index from all documents.

        # Fetch all documents from vector DB
        # (In production, maintain separate BM25 index)
        self.documents = self._fetch_all_documents()

        # Tokenize documents
        tokenized_docs = [
            doc['content'].lower().split()
            for doc in self.documents
        ]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        """
        pass

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict]:
        """
        TODO: Hybrid retrieval with RRF fusion.

        1. Dense retrieval (vector search):
           query_embedding = self.embedding_model.embed_query(query)
           dense_results = self.vector_db.search(
               collection_name=self.collection_name,
               query_vector=query_embedding,
               top_k=top_k * 2  # Retrieve more for fusion
           )

        2. Sparse retrieval (BM25):
           tokenized_query = query.lower().split()
           bm25_scores = self.bm25_index.get_scores(tokenized_query)
           # Get top_k * 2 documents
           sparse_results = ...

        3. Reciprocal Rank Fusion:
           For each document, calculate fusion score:
           score = dense_weight * (1 / (rank_dense + 60)) +
                   sparse_weight * (1 / (rank_sparse + 60))

           Constant 60 is from RRF paper - prevents division by zero
           and balances contribution of lower-ranked results

        4. Re-rank and return top_k

        Return fused results
        """
        pass

class ReRanker:
    """
    Re-rank retrieved results using cross-encoder.

    Cross-encoder: Jointly encodes query and document
    More accurate than bi-encoder (separate embeddings) but slower

    Use: Retrieve 20 docs with fast bi-encoder,
         Re-rank to top 5 with slow cross-encoder
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        TODO: Re-rank documents using cross-encoder.

        # Create query-document pairs
        pairs = [[query, doc['content']] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # Sort by score and return top_k
        ranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return ranked[:top_k]
        """
        pass
```

### Part 4: RAG Generation Pipeline (6-8 hours)

Build generation pipeline with LLM integration.

#### 4.1 RAG Pipeline

Create `src/rag/rag_pipeline.py`:

```python
"""
TODO: Complete RAG pipeline: Retrieve + Generate.

Flow:
1. Receive query
2. Retrieve relevant context
3. Build prompt with context
4. Generate answer using LLM
5. Extract citations
6. Detect hallucinations
"""

from typing import List, Dict, Optional
import httpx

class RAGPipeline:
    """End-to-end RAG system."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: ReRanker,
        llm_endpoint: str = "http://llama2-7b-service:8000"
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_endpoint = llm_endpoint

    def query(
        self,
        query: str,
        top_k: int = 5,
        include_citations: bool = True
    ) -> Dict:
        """
        TODO: Execute RAG query.

        1. Retrieve relevant chunks
        retrieved_docs = self.retriever.retrieve(query, top_k=20)

        2. Re-rank to top_k
        ranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)

        3. Build prompt
        prompt = self._build_prompt(query, ranked_docs)

        4. Generate answer
        answer = self._generate(prompt)

        5. Add citations
        if include_citations:
            answer = self._add_citations(answer, ranked_docs)

        6. Check for hallucinations
        hallucination_score = self._detect_hallucination(answer, ranked_docs)

        Return:
        {
            'query': query,
            'answer': answer,
            'citations': [...],
            'context_used': ranked_docs,
            'hallucination_score': hallucination_score
        }
        """
        pass

    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        TODO: Build RAG prompt.

        Prompt template:
        ```
        You are a helpful AI assistant. Answer the question based ONLY on the
        provided context. If the answer is not in the context, say
        "I don't have enough information to answer that."

        Context:
        {context}

        Question: {query}

        Answer:
        ```

        Format context:
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}\nSource: {doc['metadata']['source']}"
            for i, doc in enumerate(context_docs)
        ])
        """
        pass

    async def _generate(self, prompt: str) -> str:
        """
        TODO: Call LLM to generate answer.

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json={
                    "model": "llama2",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for factual answers
                    "max_tokens": 512
                },
                timeout=60.0
            )

            result = response.json()
            return result['choices'][0]['message']['content']
        """
        pass

    def _add_citations(self, answer: str, context_docs: List[Dict]) -> str:
        """
        TODO: Add citations to answer.

        Look for sentences in answer that match content from context_docs.
        Append citation: [Source: doc_name, Page: X]

        Example:
        "To reset your password, go to Settings. [1]"
        Where [1] refers to context document 1
        """
        pass

    def _detect_hallucination(
        self,
        answer: str,
        context_docs: List[Dict]
    ) -> float:
        """
        TODO: Detect if answer contains information not in context.

        Simple heuristic:
        1. Extract key facts from answer (using NER or sentence splitting)
        2. Check if each fact appears in context
        3. Hallucination score = (facts_not_in_context / total_facts)

        Advanced: Use NLI (Natural Language Inference) model
        to check entailment between answer and context
        """
        pass
```

### Part 5: Evaluation and Monitoring (6-8 hours)

Implement RAG evaluation metrics and monitoring.

#### 5.1 Evaluation Framework

Create `src/evaluation/rag_eval.py`:

```python
"""
TODO: Evaluate RAG system performance.

Metrics:
1. Retrieval Quality:
   - Precision@K: How many retrieved docs are relevant?
   - Recall@K: What % of relevant docs were retrieved?
   - MRR (Mean Reciprocal Rank): Position of first relevant doc
   - NDCG: Normalized Discounted Cumulative Gain

2. Generation Quality:
   - Correctness: Is answer factually correct?
   - Completeness: Does answer address all aspects of question?
   - Groundedness: Is answer supported by context?
   - Citation accuracy: Are citations correct?

3. End-to-End:
   - Latency: Time from query to answer
   - User satisfaction: Thumbs up/down
"""

from typing import List, Dict
import numpy as np

class RAGEvaluator:
    """Evaluate RAG system performance."""

    def __init__(self, ground_truth_file: str):
        # TODO: Load ground truth Q&A pairs
        # Format: [{"query": "...", "answer": "...", "relevant_docs": [...]}]
        self.ground_truth = self._load_ground_truth(ground_truth_file)

    def evaluate_retrieval(
        self,
        rag_pipeline: RAGPipeline,
        k: int = 5
    ) -> Dict[str, float]:
        """
        TODO: Evaluate retrieval quality.

        For each ground truth query:
        1. Retrieve top-k documents
        2. Compare with ground truth relevant docs
        3. Calculate metrics

        Metrics:
        - Precision@K = (relevant docs in top-k) / k
        - Recall@K = (relevant docs in top-k) / total relevant
        - MRR = 1 / rank_of_first_relevant_doc
        - NDCG@K = normalized discounted cumulative gain

        Return average metrics across all queries
        """
        precisions = []
        recalls = []
        mrrs = []

        for item in self.ground_truth:
            query = item['query']
            relevant_doc_ids = set(item['relevant_docs'])

            # Retrieve docs
            retrieved = rag_pipeline.retriever.retrieve(query, top_k=k)
            retrieved_ids = [doc['id'] for doc in retrieved]

            # Calculate metrics
            relevant_retrieved = set(retrieved_ids) & relevant_doc_ids

            precision = len(relevant_retrieved) / k
            recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0

            precisions.append(precision)
            recalls.append(recall)

            # MRR: Find position of first relevant doc
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_doc_ids:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)  # No relevant doc found

        return {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'mrr': np.mean(mrrs)
        }

    def evaluate_generation(
        self,
        rag_pipeline: RAGPipeline,
        use_llm_judge: bool = True
    ) -> Dict[str, float]:
        """
        TODO: Evaluate answer quality.

        For each ground truth Q&A:
        1. Generate answer using RAG
        2. Compare with ground truth answer
        3. Score on: correctness, completeness, groundedness

        Scoring methods:
        - Exact match (too strict)
        - Token overlap (F1, ROUGE)
        - Semantic similarity (embedding cosine similarity)
        - LLM-as-judge: Use GPT-4 to evaluate (most accurate)

        Example LLM-as-judge prompt:
        "Rate the following answer on a scale of 1-5 for correctness and completeness.
         Question: {question}
         Reference Answer: {ground_truth}
         Generated Answer: {generated}
         Provide scores and brief explanation."
        """
        pass
```

#### 5.2 Monitoring Dashboard

Create `dashboards/rag-dashboard.json`:

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Query Rate",
        "targets": [{"expr": "rate(rag_queries_total[5m])"}]
      },
      {
        "title": "End-to-End Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(rag_query_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Retrieval Precision@5",
        "targets": [{"expr": "rag_retrieval_precision_at_5"}]
      },
      {
        "title": "Hallucination Rate",
        "targets": [{
          "expr": "rate(rag_hallucinations_total[1h]) / rate(rag_queries_total[1h])"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(rag_cache_hits_total[5m]) / rate(rag_queries_total[5m])"
        }]
      }
    ]
  }
}
```

## Acceptance Criteria

### Functional Requirements

- [ ] Document ingestion pipeline processes PDF, HTML, Markdown
- [ ] Intelligent chunking with overlap
- [ ] Qdrant vector database deployed with HA
- [ ] Hybrid search (dense + sparse) implemented
- [ ] Re-ranking with cross-encoder
- [ ] RAG generation with LLM integration
- [ ] Citation extraction
- [ ] Hallucination detection

### Performance Requirements

- [ ] Retrieval precision@5: >90%
- [ ] Answer accuracy: >90%
- [ ] End-to-end latency: <2 seconds
- [ ] Handle 10,000 queries/day
- [ ] Support 1M+ documents
- [ ] Hallucination rate: <5%

### Code Quality

- [ ] Comprehensive unit tests (>80% coverage)
- [ ] Integration tests for full RAG pipeline
- [ ] Evaluation framework with metrics
- [ ] Monitoring dashboards

## Testing Strategy

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/ -v

# Evaluation
python src/evaluation/run_eval.py --ground-truth data/eval_set.json

# Load test
python tests/load_test_rag.py --queries-per-second 100
```

## Deliverables

1. **Source Code** (ingestion, retrieval, generation pipelines)
2. **Kubernetes Manifests** (Qdrant deployment)
3. **Evaluation Framework** (metrics, ground truth dataset)
4. **Documentation** (architecture, API docs, evaluation results)
5. **Dashboards** (Grafana monitoring)

## Bonus Challenges

1. **Multi-Modal RAG** (+8 hours): Support images, tables in documents
2. **Conversational RAG** (+6 hours): Handle follow-up questions with context
3. **Active Learning** (+6 hours): Learn from user feedback to improve retrieval

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Papers](https://arxiv.org/abs/2005.11401)

## Submission

```bash
git add .
git commit -m "Complete Exercise 02: Production RAG System"
git push origin exercise-02-production-rag-system
```

---

**Estimated Time Breakdown**:
- Part 1 (Document Ingestion): 8-10 hours
- Part 2 (Vector Database): 7-9 hours
- Part 3 (Retrieval Engine): 7-9 hours
- Part 4 (RAG Generation): 6-8 hours
- Part 5 (Evaluation): 6-8 hours
- **Total**: 34-42 hours
