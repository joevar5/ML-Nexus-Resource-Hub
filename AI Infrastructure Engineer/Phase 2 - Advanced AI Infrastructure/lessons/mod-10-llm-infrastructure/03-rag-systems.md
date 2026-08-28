# Lesson 03: RAG Systems (Retrieval-Augmented Generation)

## Table of Contents
1. [Introduction to RAG](#introduction-to-rag)
2. [RAG Architecture and Components](#rag-architecture-and-components)
3. [Vector Databases Overview](#vector-databases-overview)
4. [Embedding Models](#embedding-models)
5. [Document Processing and Chunking](#document-processing-and-chunking)
6. [Building a Complete RAG System](#building-a-complete-rag-system)
7. [Retrieval Techniques](#retrieval-techniques)
8. [Re-ranking Strategies](#re-ranking-strategies)
9. [RAG Evaluation](#rag-evaluation)
10. [Production RAG Pipelines](#production-rag-pipelines)
11. [Advanced RAG Patterns](#advanced-rag-patterns)
12. [Summary](#summary)

## Introduction to RAG

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances LLMs by providing them with relevant context retrieved from external knowledge bases. RAG addresses one of the fundamental limitations of LLMs: hallucination and lack of up-to-date or domain-specific knowledge.

### The RAG Paradigm

**Traditional LLM Approach:**
```
User Query → LLM → Response (based only on training data)
```

Problems:
- May hallucinate facts
- No knowledge of recent events
- Cannot access private/domain-specific data
- "Knows" only what was in training data

**RAG Approach:**
```
User Query → Retrieve Relevant Docs → LLM (with context) → Grounded Response
```

Benefits:
- Responses grounded in actual documents
- Can work with private/proprietary data
- Up-to-date information (if documents are current)
- Citable sources for fact-checking
- Reduced hallucination

### Learning Objectives

By the end of this lesson, you will be able to:
- Understand RAG architecture and when to use it
- Choose appropriate vector databases for different use cases
- Select and deploy embedding models
- Implement document chunking strategies
- Build a complete RAG pipeline from scratch
- Optimize retrieval quality and performance
- Evaluate RAG system effectiveness
- Deploy production-ready RAG systems
- Implement advanced RAG patterns (HyDE, multi-hop, etc.)

### When to Use RAG

**RAG is ideal for:**
- Question answering over documents (legal, medical, technical)
- Customer support with knowledge bases
- Code search and documentation
- Research and analysis tools
- Chatbots requiring factual accuracy
- Applications needing audit trails

**Consider fine-tuning instead when:**
- You need to change model behavior/style
- Domain-specific language is needed
- Limited context is acceptable
- Retrieval overhead is too high

## RAG Architecture and Components

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RAG System Architecture                │
└──────────────────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────────────────────────┐
│   Document  │────▶│  Indexing Pipeline                   │
│   Corpus    │     │  1. Parse documents                  │
└─────────────┘     │  2. Chunk text                       │
                    │  3. Generate embeddings              │
                    │  4. Store in vector DB               │
                    └──────────────────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────┐
                    │      Vector Database                 │
                    │  (Qdrant, Weaviate, Pinecone, etc.) │
                    └──────────────────────────────────────┘
                                     │
┌─────────────┐                      │
│ User Query  │                      │
└──────┬──────┘                      │
       │                             │
       ▼                             │
┌─────────────────┐                  │
│  Embed Query    │                  │
└────────┬────────┘                  │
         │                           │
         ▼                           │
┌─────────────────┐                  │
│ Search Vector DB│◀─────────────────┘
│ (Similarity)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieved Docs  │
│ (Top-K)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Optional:       │
│ Re-rank Results │
└────────┬────────┘
         │
         ▼
┌─────────────────┐         ┌──────────────┐
│ Build Prompt    │────────▶│     LLM      │
│ (Query + Docs)  │         │   (vLLM)     │
└─────────────────┘         └──────┬───────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Response   │
                            └─────────────┘
```

### Core Components

#### 1. Document Ingestion Pipeline
```python
class DocumentIngestionPipeline:
    """
    Load → Parse → Chunk → Embed → Index
    """
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def ingest(self, documents):
        """Process and index documents"""
        for doc in documents:
            # Parse
            text = self.parse(doc)

            # Chunk
            chunks = self.chunk(text)

            # Embed
            embeddings = self.embed(chunks)

            # Index
            self.vector_db.add(chunks, embeddings)
```

#### 2. Query Processing
```python
class QueryProcessor:
    """
    Query → Embed → Search → Retrieve
    """
    def process_query(self, query):
        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Search vector DB
        results = self.vector_db.search(query_embedding, top_k=5)

        return results
```

#### 3. Context Assembly
```python
class ContextAssembler:
    """
    Retrieved Docs → Format → Context String
    """
    def assemble_context(self, retrieved_docs, max_tokens=2000):
        context = []
        token_count = 0

        for doc in retrieved_docs:
            doc_tokens = count_tokens(doc.text)
            if token_count + doc_tokens > max_tokens:
                break
            context.append(doc.text)
            token_count += doc_tokens

        return "\n\n".join(context)
```

#### 4. Generation
```python
class ResponseGenerator:
    """
    Query + Context → LLM → Response
    """
    def generate(self, query, context):
        prompt = self.build_prompt(query, context)
        response = self.llm.generate(prompt)
        return response
```

## Vector Databases Overview

Vector databases are specialized databases optimized for storing and searching high-dimensional vectors (embeddings).

### Popular Vector Databases

| Database | Type | Best For | Strengths |
|----------|------|----------|-----------|
| **Pinecone** | Managed | Production, scale | Fully managed, easy setup, fast |
| **Weaviate** | Open-source/Managed | Hybrid search | GraphQL, hybrid search, flexible |
| **Qdrant** | Open-source/Managed | Performance | Rust-based, fast, filtering |
| **Chroma** | Open-source | Development | Lightweight, embeddable, simple |
| **Milvus** | Open-source | Large scale | Distributed, enterprise features |
| **FAISS** | Library | Research | Facebook AI, very fast, in-memory |

### Quick Comparison

```python
# Qdrant - High performance, production-ready
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)

# Chroma - Simple, embeddable
import chromadb
client = chromadb.Client()

# Pinecone - Managed, scalable
import pinecone
pinecone.init(api_key="your-key", environment="us-west1-gcp")

# Weaviate - Hybrid search capable
import weaviate
client = weaviate.Client("http://localhost:8080")
```

### Vector Database Selection Criteria

**Choose Qdrant if:**
- Need high performance
- Want self-hosted option
- Require complex filtering
- Production deployment

**Choose Chroma if:**
- Rapid prototyping
- Small to medium datasets
- Want embedded database
- Development/testing

**Choose Pinecone if:**
- Want fully managed solution
- Don't want to manage infrastructure
- Need enterprise support
- Budget allows ($70+/month)

**Choose Weaviate if:**
- Need hybrid search (vector + keyword)
- Want GraphQL API
- Require multi-modal search
- Complex schema needed

## Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning.

### Popular Embedding Models

```python
# embedding_models.py

class EmbeddingModelComparison:
    """Comparison of popular embedding models"""

    models = {
        "openai_ada_002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k": 0.0001,
            "performance": "excellent",
            "use_case": "General purpose, high quality"
        },
        "sentence_transformers_mini": {
            "dimensions": 384,
            "max_tokens": 256,
            "cost_per_1k": 0,  # Free, self-hosted
            "performance": "good",
            "use_case": "Fast, resource-constrained"
        },
        "sentence_transformers_mpnet": {
            "dimensions": 768,
            "max_tokens": 384,
            "cost_per_1k": 0,
            "performance": "excellent",
            "use_case": "Best open-source option"
        },
        "instructor_xl": {
            "dimensions": 768,
            "max_tokens": 512,
            "cost_per_1k": 0,
            "performance": "excellent",
            "use_case": "Task-specific instructions"
        },
        "bge_large": {
            "dimensions": 1024,
            "max_tokens": 512,
            "cost_per_1k": 0,
            "performance": "excellent",
            "use_case": "SOTA open-source"
        }
    }
```

### Using Different Embedding Models

#### OpenAI Embeddings

```python
# openai_embeddings.py
import openai
import os

class OpenAIEmbeddings:
    """OpenAI embedding model wrapper"""

    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def encode(self, texts):
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]

        response = openai.Embedding.create(
            model=self.model,
            input=texts
        )

        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings if len(embeddings) > 1 else embeddings[0]

    def encode_batch(self, texts, batch_size=100):
        """Encode large batches efficiently"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encode(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

# Usage
embedder = OpenAIEmbeddings()
embedding = embedder.encode("What is machine learning?")
print(f"Embedding dimension: {len(embedding)}")
```

#### Sentence Transformers (Open Source)

```python
# sentence_transformers_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbeddings:
    """Open-source embedding models"""

    def __init__(self, model_name="all-mpnet-base-v2"):
        """
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dimensions
        - all-mpnet-base-v2: Balanced, 768 dimensions
        - all-MiniLM-L12-v2: Good performance, 384 dimensions
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize=True):
        """Generate embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )

        return embeddings if len(embeddings) > 1 else embeddings[0]

    def encode_batch(self, texts, batch_size=32):
        """Efficient batch encoding"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

# Usage
embedder = SentenceTransformerEmbeddings("all-mpnet-base-v2")

# Single text
embedding = embedder.encode("What is machine learning?")

# Batch
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = embedder.encode_batch(texts)

print(f"Shape: {embeddings.shape}")  # (3, 768)
```

#### BGE Embeddings (Current SOTA)

```python
# bge_embeddings.py
from sentence_transformers import SentenceTransformer

class BGEEmbeddings:
    """BGE (BAAI General Embedding) models - current SOTA"""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        """
        Available models:
        - BAAI/bge-small-en-v1.5: 384 dimensions, fast
        - BAAI/bge-base-en-v1.5: 768 dimensions, balanced
        - BAAI/bge-large-en-v1.5: 1024 dimensions, best quality
        """
        self.model = SentenceTransformer(model_name)

        # BGE instruction for queries (important!)
        self.query_instruction = "Represent this sentence for searching relevant passages:"

    def encode_queries(self, queries):
        """Encode queries (with instruction prefix)"""
        if isinstance(queries, str):
            queries = [queries]

        # Add instruction to queries
        instructed_queries = [
            f"{self.query_instruction} {q}" for q in queries
        ]

        embeddings = self.model.encode(
            instructed_queries,
            normalize_embeddings=True
        )

        return embeddings

    def encode_documents(self, documents):
        """Encode documents (no instruction needed)"""
        if isinstance(documents, str):
            documents = [documents]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True
        )

        return embeddings

# Usage
embedder = BGEEmbeddings()

# Encode query
query_embedding = embedder.encode_queries("What is Python?")

# Encode documents
doc_embeddings = embedder.encode_documents([
    "Python is a programming language.",
    "Java is also a programming language."
])
```

## Document Processing and Chunking

Effective chunking is crucial for RAG system performance.

### Why Chunking Matters

**Problem**: Documents are often too long to:
- Fit in LLM context window
- Retrieve precisely relevant information
- Generate meaningful embeddings

**Solution**: Split documents into chunks that:
- Capture coherent concepts
- Fit in context window
- Have good embedding representation

### Chunking Strategies

```python
# chunking_strategies.py
from typing import List
import re

class DocumentChunker:
    """Various document chunking strategies"""

    @staticmethod
    def fixed_size_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Simple fixed-size chunking with overlap

        Pros: Simple, predictable size
        Cons: May split sentences/paragraphs awkwardly
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    @staticmethod
    def sentence_chunks(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """
        Chunk by sentences

        Pros: Preserves sentence boundaries
        Cons: Variable chunk sizes
        """
        # Simple sentence splitting (use spacy/nltk for better results)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ". ".join(sentences[i:i + sentences_per_chunk]) + "."
            chunks.append(chunk)

        return chunks

    @staticmethod
    def paragraph_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Chunk by paragraphs, combining small ones

        Pros: Preserves logical structure
        Cons: Variable sizes, may need combining
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def semantic_chunks(text: str, model, similarity_threshold: float = 0.7) -> List[str]:
        """
        Semantic chunking based on embedding similarity

        Pros: Semantically coherent chunks
        Cons: More computationally expensive
        """
        from sentence_transformers import SentenceTransformer

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Embed sentences
        embeddings = model.encode(sentences)

        # Group by similarity
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            if similarity > similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

# Example usage
text = """
Machine learning is a subset of artificial intelligence. It focuses on
building systems that learn from data. Deep learning is a subset of machine
learning that uses neural networks.

Python is widely used for machine learning. Libraries like TensorFlow and
PyTorch make it easy to build models. These frameworks provide high-level
APIs for common tasks.
"""

chunker = DocumentChunker()

print("Fixed Size Chunks:")
print(chunker.fixed_size_chunks(text, chunk_size=20, overlap=5))

print("\nSentence Chunks:")
print(chunker.sentence_chunks(text, sentences_per_chunk=2))

print("\nParagraph Chunks:")
print(chunker.paragraph_chunks(text))
```

### Advanced Chunking with LangChain

```python
# langchain_chunking.py
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)

class AdvancedChunker:
    """Advanced chunking using LangChain"""

    @staticmethod
    def recursive_character_split(text: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Intelligent recursive splitting

        Tries to split on: \n\n, then \n, then spaces
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_text(text)
        return chunks

    @staticmethod
    def token_based_split(text: str, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Split by tokens (more accurate for LLM context)
        """
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_text(text)
        return chunks

    @staticmethod
    def markdown_split(markdown_text: str):
        """
        Split markdown preserving structure
        """
        splitter = MarkdownTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

        chunks = splitter.split_text(markdown_text)
        return chunks

# Usage
chunker = AdvancedChunker()

text = "Your long document here..."

chunks = chunker.recursive_character_split(
    text,
    chunk_size=512,
    chunk_overlap=50
)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}:")
    print(chunk[:200] + "...")
```

### Chunking Best Practices

```python
class ChunkingBestPractices:
    """
    Guidelines for effective chunking
    """

    recommendations = {
        "chunk_size": {
            "small_models": "256-512 tokens",
            "large_models": "512-1024 tokens",
            "rationale": "Balance between context and precision"
        },
        "overlap": {
            "recommended": "10-20% of chunk size",
            "rationale": "Prevents loss of context at boundaries"
        },
        "method": {
            "structured_docs": "Paragraph or section-based",
            "unstructured_docs": "Sentence or semantic-based",
            "code": "Function or class-based",
            "rationale": "Match structure to content type"
        },
        "metadata": {
            "include": [
                "source document",
                "chunk index",
                "section/heading",
                "timestamp"
            ],
            "rationale": "Enables filtering and attribution"
        }
    }
```

## Building a Complete RAG System

Let's build a production-ready RAG system from scratch.

### Complete RAG Implementation

```python
# rag_system.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vllm import LLM, SamplingParams
from typing import List, Dict
import uuid
from dataclasses import dataclass

@dataclass
class Document:
    """Document structure"""
    id: str
    text: str
    metadata: Dict

class RAGSystem:
    """Complete RAG system implementation"""

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        vector_db_url: str = "http://localhost:6333",
        collection_name: str = "documents"
    ):
        # Initialize components
        print("Initializing RAG system...")

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Vector database
        self.vector_client = QdrantClient(url=vector_db_url)
        self.collection_name = collection_name
        self._init_collection()

        # LLM
        self.llm = LLM(model=llm_model_name)

        print("RAG system initialized!")

    def _init_collection(self):
        """Initialize vector database collection"""
        # Create collection if it doesn't exist
        collections = self.vector_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.vector_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")

    def chunk_document(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Chunk document into smaller pieces"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def ingest_document(self, document_text: str, metadata: Dict = None):
        """Ingest a document into the RAG system"""
        # Chunk document
        chunks = self.chunk_document(document_text)
        print(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)

        # Create points for vector DB
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
            )
            points.append(point)

        # Upload to vector DB
        self.vector_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Ingested {len(chunks)} chunks into vector database")

    def ingest_documents(self, documents: List[Dict]):
        """Ingest multiple documents"""
        for doc in documents:
            self.ingest_document(
                document_text=doc["text"],
                metadata=doc.get("metadata", {})
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Search vector DB
        results = self.vector_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        # Format results
        documents = []
        for result in results:
            documents.append({
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })

        return documents

    def build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt with retrieved context"""
        # Format context
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])

        # Build prompt (Llama 2 format)
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain enough information to answer the question, say so.
Always cite which document you're referencing.
<</SYS>>

Context:
{context}

Question: {query}
[/INST]"""

        return prompt

    def generate_response(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "query": query,
                "response": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }

        # Build prompt with context
        prompt = self.build_prompt(query, retrieved_docs)

        # Generate response
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>"]
        )

        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Return response with sources
        return {
            "query": query,
            "response": response,
            "sources": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }

    def query(self, question: str) -> str:
        """Simple query interface"""
        result = self.generate_response(question)
        return result["response"]

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()

    # Example documents
    documents = [
        {
            "text": """
            Machine learning is a subset of artificial intelligence that focuses on
            building systems that learn from data. The main types of machine learning
            are supervised learning, unsupervised learning, and reinforcement learning.
            Supervised learning uses labeled data to train models, unsupervised learning
            finds patterns in unlabeled data, and reinforcement learning learns through
            interaction with an environment.
            """,
            "metadata": {"source": "ml_intro.txt", "topic": "machine_learning"}
        },
        {
            "text": """
            Deep learning is a subset of machine learning that uses neural networks
            with multiple layers. Deep learning has achieved remarkable results in
            computer vision, natural language processing, and speech recognition.
            Popular deep learning frameworks include TensorFlow, PyTorch, and JAX.
            """,
            "metadata": {"source": "deep_learning.txt", "topic": "deep_learning"}
        },
        {
            "text": """
            Natural language processing (NLP) is a field that focuses on enabling
            computers to understand and generate human language. Key NLP tasks include
            text classification, named entity recognition, machine translation, and
            question answering. Transformers have revolutionized NLP since their
            introduction in 2017.
            """,
            "metadata": {"source": "nlp_overview.txt", "topic": "nlp"}
        }
    ]

    # Ingest documents
    rag.ingest_documents(documents)

    # Query the system
    questions = [
        "What is machine learning?",
        "What are deep learning frameworks?",
        "What is NLP used for?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.generate_response(question)
        print(f"Response: {result['response']}")
        print(f"Sources: {result['num_sources']} documents")
```

### Running the RAG System

```python
# run_rag.py

# 1. Start Qdrant (Docker)
"""
docker run -p 6333:6333 qdrant/qdrant
"""

# 2. Initialize and use RAG system
from rag_system import RAGSystem

rag = RAGSystem(
    embedding_model_name="BAAI/bge-base-en-v1.5",
    llm_model_name="meta-llama/Llama-2-7b-chat-hf",
    vector_db_url="http://localhost:6333"
)

# Ingest your documents
rag.ingest_document(
    document_text=open("your_document.txt").read(),
    metadata={"source": "your_document.txt"}
)

# Query
response = rag.query("What is the main topic of the document?")
print(response)
```

## Retrieval Techniques

### Basic Similarity Search

```python
def basic_similarity_search(query_embedding, top_k=5):
    """Simple cosine similarity search"""
    results = vector_client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=top_k
    )
    return results
```

### Filtered Search

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def filtered_search(query_embedding, filters, top_k=5):
    """Search with metadata filters"""

    # Example: Only search documents from specific source
    search_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value="specific_source.pdf")
            )
        ]
    )

    results = vector_client.search(
        collection_name="documents",
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k
    )

    return results
```

### Hybrid Search (Vector + Keyword)

```python
class HybridSearch:
    """Combine vector and keyword search"""

    def __init__(self, vector_client, embedding_model):
        self.vector_client = vector_client
        self.embedding_model = embedding_model

    def search(self, query, top_k=10, alpha=0.7):
        """
        Hybrid search: alpha * vector_score + (1-alpha) * keyword_score
        """
        # Vector search
        query_embedding = self.embedding_model.encode(query)
        vector_results = self.vector_client.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k
        )

        # Keyword search (BM25 or simple matching)
        keyword_results = self.keyword_search(query, top_k)

        # Combine scores
        combined = self._combine_results(
            vector_results,
            keyword_results,
            alpha
        )

        return combined

    def keyword_search(self, query, top_k):
        """Simple keyword search implementation"""
        # Implementation depends on your text search index
        pass

    def _combine_results(self, vector_results, keyword_results, alpha):
        """Combine and re-rank results"""
        # Normalize and combine scores
        pass
```

## Re-ranking Strategies

Re-ranking improves retrieval quality by reordering initial results.

```python
# reranking.py
from sentence_transformers import CrossEncoder

class Reranker:
    """Re-rank retrieved documents"""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Cross-encoder models for re-ranking:
        - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality
        - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5):
        """Re-rank documents for query"""

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score pairs
        scores = self.model.predict(pairs)

        # Sort by score
        ranked_indices = scores.argsort()[::-1][:top_k]

        # Return ranked documents
        ranked_docs = [(documents[i], scores[i]) for i in ranked_indices]

        return ranked_docs

# Usage with RAG
class RAGWithReranking(RAGSystem):
    """RAG system with re-ranking"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5, rerank_top_k: int = 20):
        """Retrieve with re-ranking"""

        # Initial retrieval (get more candidates)
        initial_docs = super().retrieve(query, top_k=rerank_top_k)

        # Extract texts
        texts = [doc["text"] for doc in initial_docs]

        # Re-rank
        reranked = self.reranker.rerank(query, texts, top_k=top_k)

        # Format results
        results = []
        for text, score in reranked:
            # Find original document for metadata
            orig_doc = next(d for d in initial_docs if d["text"] == text)
            results.append({
                "text": text,
                "score": float(score),
                "metadata": orig_doc["metadata"]
            })

        return results
```

## RAG Evaluation

Evaluating RAG systems is crucial for production deployment.

```python
# rag_evaluation.py
from typing import List, Dict
import numpy as np

class RAGEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate_retrieval(
        self,
        test_queries: List[Dict]  # {"query": str, "relevant_docs": List[str]}
    ):
        """
        Evaluate retrieval quality

        Metrics:
        - Precision@K
        - Recall@K
        - MRR (Mean Reciprocal Rank)
        - NDCG (Normalized Discounted Cumulative Gain)
        """
        precisions = []
        recalls = []
        mrrs = []

        for item in test_queries:
            query = item["query"]
            relevant_docs = set(item["relevant_docs"])

            # Retrieve
            retrieved = self.rag.retrieve(query, top_k=10)
            retrieved_ids = [doc["metadata"].get("id") for doc in retrieved]

            # Calculate metrics
            relevant_retrieved = set(retrieved_ids) & relevant_docs

            # Precision@K
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            precisions.append(precision)

            # Recall@K
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            recalls.append(recall)

            # MRR
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_docs:
                    mrrs.append(1 / (i + 1))
                    break
            else:
                mrrs.append(0)

        return {
            "precision@10": np.mean(precisions),
            "recall@10": np.mean(recalls),
            "mrr": np.mean(mrrs)
        }

    def evaluate_generation(
        self,
        test_queries: List[Dict]  # {"query": str, "expected_answer": str}
    ):
        """
        Evaluate generation quality

        Note: This requires human evaluation or LLM-as-judge
        """
        results = []

        for item in test_queries:
            query = item["query"]
            expected = item["expected_answer"]

            # Generate
            response = self.rag.generate_response(query)
            generated = response["response"]

            # Store for evaluation
            results.append({
                "query": query,
                "expected": expected,
                "generated": generated
            })

        return results

    def evaluate_faithfulness(
        self,
        test_queries: List[str]
    ):
        """
        Check if responses are grounded in retrieved documents

        Requires LLM-as-judge or NLI model
        """
        pass

# Usage
evaluator = RAGEvaluator(rag_system)

test_set = [
    {
        "query": "What is machine learning?",
        "relevant_docs": ["doc_1", "doc_5"],
        "expected_answer": "Machine learning is..."
    }
]

retrieval_metrics = evaluator.evaluate_retrieval(test_set)
print(f"Retrieval Metrics: {retrieval_metrics}")
```

## Production RAG Pipelines

### Production Architecture

```python
# production_rag.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="Production RAG API")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 512

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict]
    processing_time_ms: float

class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = None

# Initialize RAG system
rag = RAGSystem()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint"""
    import time

    start_time = time.time()

    try:
        result = rag.generate_response(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            query=result["query"],
            response=result["response"],
            sources=result["sources"],
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest document endpoint"""
    try:
        rag.ingest_document(
            document_text=request.text,
            metadata=request.metadata
        )
        return {"status": "success", "message": "Document ingested"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Deployment Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VECTOR_DB_URL=http://qdrant:6333
      - LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
    depends_on:
      - qdrant
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  qdrant_data:
```

## Advanced RAG Patterns

### HyDE (Hypothetical Document Embeddings)

```python
class HyDERAG(RAGSystem):
    """RAG with Hypothetical Document Embeddings"""

    def retrieve_with_hyde(self, query: str, top_k: int = 5):
        """
        Generate hypothetical answer, embed it, use for retrieval
        """
        # Generate hypothetical answer
        hyde_prompt = f"Generate a detailed answer to: {query}"
        sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        outputs = self.llm.generate([hyde_prompt], sampling_params)
        hypothetical_answer = outputs[0].outputs[0].text

        # Embed hypothetical answer
        hyde_embedding = self.embedding_model.encode(hypothetical_answer)

        # Search with hypothetical embedding
        results = self.vector_client.search(
            collection_name=self.collection_name,
            query_vector=hyde_embedding.tolist(),
            limit=top_k
        )

        return results
```

### Multi-Hop Retrieval

```python
class MultiHopRAG(RAGSystem):
    """Multi-hop retrieval for complex questions"""

    def multi_hop_retrieve(self, query: str, max_hops: int = 3):
        """
        Iteratively retrieve and refine
        """
        all_docs = []
        current_query = query

        for hop in range(max_hops):
            # Retrieve for current query
            docs = self.retrieve(current_query, top_k=3)
            all_docs.extend(docs)

            # Generate follow-up query
            follow_up = self.generate_follow_up_query(query, docs)
            if not follow_up:
                break

            current_query = follow_up

        # Deduplicate and return
        return self.deduplicate_docs(all_docs)

    def generate_follow_up_query(self, original_query, retrieved_docs):
        """Generate follow-up query based on retrieved docs"""
        # Use LLM to generate follow-up
        pass
```

## Summary

This lesson covered comprehensive RAG systems:

### Key Takeaways

1. **RAG enhances LLMs** with external knowledge, reducing hallucination
2. **Core components**: Embeddings, vector databases, chunking, retrieval, generation
3. **Vector databases** provide efficient semantic search
4. **Chunking strategy** significantly impacts RAG quality
5. **Re-ranking** improves retrieval precision
6. **Evaluation** is crucial for production systems
7. **Advanced patterns** (HyDE, multi-hop) handle complex scenarios

### Next Steps

In the next lesson, we'll dive deep into vector databases, comparing options and implementing production deployments.

---

**Next Lesson**: [04-vector-databases.md](./04-vector-databases.md)
