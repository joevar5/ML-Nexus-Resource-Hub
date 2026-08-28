# Module 10: LLM Infrastructure - Hands-On Exercises

## Overview

This comprehensive exercise set provides hands-on experience building production-grade LLM infrastructure. You'll progress from deploying your first LLM to building complete multi-model platforms with monitoring and optimization.

**Total Exercises:** 8 progressive exercises
**Estimated Total Time:** 25-35 hours
**Difficulty Range:** Beginner to Advanced

---

## Exercise Prerequisites

### Required Software
```bash
# Core tools
- Python 3.11+
- Docker 24.0+
- kubectl 1.28+
- git
- curl, jq

# Python packages
pip install vllm transformers torch langchain qdrant-client sentence-transformers
pip install prometheus-client opentelemetry-api fastapi uvicorn
```

### Hardware Requirements

**Minimum (Exercises 1-3, 7):**
- CPU: 8 cores
- RAM: 32GB
- Storage: 100GB

**Recommended (All exercises):**
- GPU: NVIDIA T4 (16GB) or better
- CPU: 16 cores
- RAM: 64GB
- Storage: 200GB

**Cloud GPU Options:**
- AWS: `g4dn.xlarge` (T4), `g5.xlarge` (A10G)
- GCP: `n1-standard-8` with T4 GPU
- Lambda Labs: GPU cloud instances
- RunPod: Spot instances for cost savings

### Cost Warnings

**💰 IMPORTANT: These exercises involve GPU costs**

- **Exercise 01:** ~$0.50/hour (T4 GPU)
- **Exercise 02-03:** ~$0.25/hour (CPU only)
- **Exercise 04:** ~$1.00/hour (A10G recommended)
- **Exercise 05-06:** ~$0.75/hour (T4 or A10G)
- **Exercise 07-08:** ~$1.50/hour (multi-GPU)

**Cost-saving tips:**
- Use spot instances (60-70% cheaper)
- Terminate instances when not in use
- Start with smaller models (7B instead of 70B)
- Use free tier credits (GCP $300, AWS free tier)

---

## Exercise Roadmap

```
Exercise 01: Deploy Llama 2 with vLLM (Foundation)
     ↓
Exercise 02: Build Basic RAG System (Application)
     ↓
Exercise 03: Set Up Vector Database (Infrastructure)
     ↓ (Combine 02 + 03)
Exercise 04: Fine-Tune with LoRA (Customization)
     ↓
Exercise 05: Optimize LLM Inference (Performance)
     ↓
Exercise 06: Production LLM on Kubernetes (Scale)
     ↓
Exercise 07: Implement Monitoring (Observability)
     ↓
Exercise 08: Build Multi-Model API (Platform)
```

---

## Exercise 01: Deploy Llama 2 with vLLM

**Difficulty:** Beginner
**Estimated Time:** 2-3 hours
**GPU Required:** Yes (T4 or better)
**Cost:** ~$0.50-1.00 for exercise

### Learning Objectives

- Deploy an LLM using vLLM
- Understand vLLM configuration parameters
- Create OpenAI-compatible API
- Test inference performance
- Monitor GPU utilization

### Prerequisites

- Basic Python knowledge
- Docker fundamentals
- Access to GPU instance

### Step-by-Step Instructions

#### Part 1: Environment Setup (30 minutes)

**1.1 Launch GPU Instance**

```bash
# AWS example (adjust for your cloud provider)
# Instance type: g4dn.xlarge (T4 GPU, 16GB VRAM)

# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Verify GPU
nvidia-smi
```

**1.2 Install Dependencies**

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install NVIDIA drivers (if not present)
sudo apt-get install -y nvidia-driver-535

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU works with Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**1.3 Set Up Python Environment**

```bash
# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
python3.11 -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM
pip install vllm
pip install ray  # For distributed serving

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

#### Part 2: Download and Deploy Model (45 minutes)

**2.1 Download Llama 2 7B**

```bash
# Create directory for models
mkdir -p ~/models
cd ~/models

# Option 1: Download from Hugging Face (requires authentication)
pip install huggingface-hub

# Login to Hugging Face (get token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Download Llama 2 7B Chat
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir llama-2-7b-chat \
  --local-dir-use-symlinks False

# Option 2: Use pre-downloaded model or alternative
# Alternative models that don't require auth:
# - mistralai/Mistral-7B-Instruct-v0.2
# - HuggingFaceH4/zephyr-7b-beta
```

**2.2 Basic vLLM Deployment**

```python
# File: ~/deploy_vllm_basic.py

from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="/home/ubuntu/models/llama-2-7b-chat",
    # GPU memory utilization (0.9 = 90%)
    gpu_memory_utilization=0.9,
    # Maximum model context length
    max_model_len=4096,
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# Test prompts
prompts = [
    "Explain what a Large Language Model is in simple terms:",
    "Write a Python function to calculate factorial:",
]

# Generate responses
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 80)
```

```bash
# Run basic deployment
python ~/deploy_vllm_basic.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

**2.3 Deploy vLLM Server with OpenAI-Compatible API**

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/llama-2-7b-chat \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096

# In another terminal, test the API
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/ubuntu/models/llama-2-7b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

#### Part 3: Performance Testing (30 minutes)

**3.1 Create Benchmark Script**

```python
# File: ~/benchmark_vllm.py

import time
import asyncio
import aiohttp
import statistics
from typing import List

API_URL = "http://localhost:8000/v1/chat/completions"

async def send_request(session, prompt: str, request_id: int):
    """Send single request and measure latency"""
    payload = {
        "model": "/home/ubuntu/models/llama-2-7b-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    start_time = time.time()
    async with session.post(API_URL, json=payload) as response:
        result = await response.json()
        latency = time.time() - start_time

        tokens_generated = len(result['choices'][0]['message']['content'].split())

        return {
            'request_id': request_id,
            'latency': latency,
            'tokens': tokens_generated,
            'tokens_per_second': tokens_generated / latency
        }

async def run_benchmark(num_requests: int, concurrent: int):
    """Run benchmark with specified concurrency"""
    print(f"\n🔬 Running benchmark: {num_requests} requests, {concurrent} concurrent")

    # Create test prompts
    prompts = [
        f"Explain concept {i} in machine learning in detail."
        for i in range(num_requests)
    ]

    # Create session
    async with aiohttp.ClientSession() as session:
        latencies = []
        throughputs = []

        # Send requests in batches
        for i in range(0, num_requests, concurrent):
            batch = prompts[i:i+concurrent]
            tasks = [
                send_request(session, prompt, i+j)
                for j, prompt in enumerate(batch)
            ]

            results = await asyncio.gather(*tasks)

            for result in results:
                latencies.append(result['latency'])
                throughputs.append(result['tokens_per_second'])
                print(f"Request {result['request_id']}: "
                      f"{result['latency']:.2f}s, "
                      f"{result['tokens_per_second']:.1f} tok/s")

        # Calculate statistics
        print(f"\n📊 Results:")
        print(f"Average latency: {statistics.mean(latencies):.2f}s")
        print(f"P50 latency: {statistics.median(latencies):.2f}s")
        print(f"P95 latency: {sorted(latencies)[int(0.95*len(latencies))]:.2f}s")
        print(f"P99 latency: {sorted(latencies)[int(0.99*len(latencies))]:.2f}s")
        print(f"Avg throughput: {statistics.mean(throughputs):.1f} tokens/sec")

if __name__ == "__main__":
    # Run different concurrency levels
    asyncio.run(run_benchmark(num_requests=20, concurrent=1))
    asyncio.run(run_benchmark(num_requests=20, concurrent=5))
    asyncio.run(run_benchmark(num_requests=20, concurrent=10))
```

```bash
# Install dependencies
pip install aiohttp

# Run benchmark
python ~/benchmark_vllm.py
```

#### Part 4: Configuration Tuning (30 minutes)

**4.1 Experiment with vLLM Parameters**

```bash
# Test 1: Higher GPU utilization
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/llama-2-7b-chat \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048

# Test 2: Enable tensor parallelism (if multiple GPUs)
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/llama-2-7b-chat \
  --tensor-parallel-size 2  # Requires 2 GPUs

# Test 3: Adjust max number of sequences
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/llama-2-7b-chat \
  --max-num-seqs 64  # Default is 256

# Compare performance after each change
```

**4.2 Create Systemd Service**

```bash
# File: /etc/systemd/system/vllm.service
sudo tee /etc/systemd/system/vllm.service > /dev/null <<EOF
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/vllm-env/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/models/llama-2-7b-chat \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm

# View logs
sudo journalctl -u vllm -f
```

### Challenge Tasks

**Challenge 1: Multi-GPU Deployment**
- If you have access to multiple GPUs, deploy with tensor parallelism
- Measure speedup compared to single GPU

**Challenge 2: Custom Prompt Template**
- Implement custom chat template for Llama 2 format
- Add system prompts and conversation history

**Challenge 3: Streaming Responses**
- Implement streaming API endpoint
- Display tokens as they're generated

### Expected Results

- ✅ vLLM server running and accessible
- ✅ Average latency: 0.5-2 seconds for 100 tokens
- ✅ Throughput: 20-50 tokens/second (depending on GPU)
- ✅ GPU utilization: 80-95%
- ✅ Successful OpenAI-compatible API calls

### Troubleshooting

**Issue: Out of Memory (OOM)**
```bash
# Solution: Reduce memory utilization or context length
--gpu-memory-utilization 0.8
--max-model-len 2048
```

**Issue: Slow startup**
```bash
# Solution: First run downloads/compiles. Subsequent runs faster.
# Check logs for actual loading time
```

**Issue: Import errors**
```bash
# Solution: Verify installation
pip install --upgrade vllm torch
```

### Deliverables

- [ ] vLLM server running successfully
- [ ] Benchmark results documented
- [ ] Screenshots of nvidia-smi during inference
- [ ] API test examples saved

---

## Exercise 02: Build a Basic RAG System

**Difficulty:** Intermediate
**Estimated Time:** 3-4 hours
**GPU Required:** Optional (CPU works for embeddings)
**Cost:** ~$0.25/hour (CPU) or ~$0.50/hour (with GPU)

### Learning Objectives

- Implement document chunking strategies
- Generate and store embeddings
- Build retrieval pipeline
- Integrate with LLM for generation
- Evaluate RAG quality

### Prerequisites

- Completed Exercise 01
- Understanding of embeddings
- Basic NLP concepts

### Step-by-Step Instructions

#### Part 1: Document Preparation (45 minutes)

**1.1 Create Sample Document Collection**

```python
# File: ~/rag/prepare_documents.py

import os
from pathlib import Path

# Create documents directory
docs_dir = Path("~/rag/documents").expanduser()
docs_dir.mkdir(parents=True, exist_ok=True)

# Sample documents (AI Infrastructure topics)
documents = {
    "kubernetes_basics.txt": """
Kubernetes is an open-source container orchestration platform that automates
the deployment, scaling, and management of containerized applications. It was
originally developed by Google and is now maintained by the Cloud Native
Computing Foundation (CNCF).

Key concepts in Kubernetes include:

1. Pods: The smallest deployable units in Kubernetes, consisting of one or
more containers that share storage and network resources.

2. Services: An abstract way to expose applications running on a set of Pods
as a network service.

3. Deployments: Provide declarative updates for Pods and ReplicaSets.

4. Namespaces: Virtual clusters backed by the same physical cluster,
providing scope for resource names.

5. ConfigMaps and Secrets: Mechanisms to inject configuration data into Pods.

For AI/ML workloads, Kubernetes offers several advantages:
- GPU scheduling and isolation
- Auto-scaling based on metrics
- Resource management and quota
- Rolling updates for model deployments
- Service mesh integration for model serving
""",

    "llm_deployment.txt": """
Deploying Large Language Models (LLMs) in production requires careful
consideration of several factors:

Infrastructure Requirements:
- GPU memory: 7B models need ~14-16GB, 13B need ~26GB, 70B need ~140GB
- VRAM calculation: parameters * 2 bytes (FP16) or 1 byte (INT8)
- Batch processing capabilities for throughput

Serving Frameworks:
1. vLLM: High-throughput serving with PagedAttention
2. TGI (Text Generation Inference): Hugging Face's production server
3. TensorRT-LLM: NVIDIA's optimized inference engine
4. Ray Serve: Distributed model serving

Key optimization techniques:
- Quantization (INT8, INT4) to reduce memory
- Flash Attention for faster inference
- Continuous batching for higher throughput
- KV cache management for memory efficiency

Production considerations:
- Load balancing across multiple GPUs
- Monitoring latency and throughput
- Cost optimization strategies
- Auto-scaling based on queue depth
""",

    "vector_databases.txt": """
Vector databases are specialized database systems designed to store and
query high-dimensional vector embeddings efficiently.

Popular vector databases:

1. Qdrant:
   - Written in Rust for performance
   - Rich filtering capabilities
   - Easy Docker deployment
   - Good for production RAG systems

2. Weaviate:
   - GraphQL API
   - Built-in vectorization
   - Multi-modal support

3. Pinecone:
   - Fully managed service
   - Easy to use
   - Can be expensive at scale

4. Chroma:
   - Lightweight and embeddable
   - Great for development
   - Python-native

Key features to consider:
- ANN algorithm (HNSW, IVF, etc.)
- Filtering performance
- Scalability and replication
- Query performance
- Metadata support

For RAG systems, you want:
- Fast similarity search (<100ms)
- Metadata filtering
- High recall rate
- Easy integration with embedding models
""",

    "gpu_optimization.txt": """
GPU Optimization for AI Infrastructure

Memory Management:
- Unified Memory: Allows oversubscription
- Memory pools: Reduce allocation overhead
- Gradient checkpointing: Trade compute for memory

Performance Optimization:
1. Mixed Precision Training:
   - FP16/BF16 for faster computation
   - Automatic loss scaling
   - 2-3x speedup typical

2. Kernel Fusion:
   - Combine operations
   - Reduce memory transfers
   - PyTorch JIT, XLA compilation

3. Data Loading:
   - Pin memory for faster transfers
   - Prefetch next batch
   - Multiple workers

4. Batch Size Tuning:
   - Larger batches = better GPU utilization
   - Find maximum stable batch size
   - Consider gradient accumulation

Monitoring:
- nvidia-smi for basic stats
- DCGM for detailed metrics
- GPU utilization should be >80% for training
"""
}

# Write documents
for filename, content in documents.items():
    filepath = docs_dir / filename
    filepath.write_text(content)
    print(f"Created: {filepath}")

print(f"\n✅ Created {len(documents)} documents in {docs_dir}")
```

```bash
# Run document preparation
python ~/rag/prepare_documents.py
```

**1.2 Implement Chunking Strategy**

```python
# File: ~/rag/chunking.py

from typing import List, Dict
from pathlib import Path
import re

class DocumentChunker:
    """Chunk documents with overlap for better context"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text with metadata"""
        # Split by separator first
        splits = text.split(self.separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            # If adding this split exceeds chunk size, save current chunk
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': metadata or {}
                    })
                current_chunk = split
            else:
                current_chunk += (self.separator if current_chunk else "") + split

        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': metadata or {}
            })

        # Add overlap between chunks
        chunks_with_overlap = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']

            # Add overlap from previous chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_text = chunks[i-1]['text']
                overlap = prev_text[-self.chunk_overlap:]
                chunk_text = overlap + " " + chunk_text

            chunks_with_overlap.append({
                'text': chunk_text,
                'metadata': {
                    **chunk['metadata'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })

        return chunks_with_overlap

    def chunk_documents(self, docs_dir: str) -> List[Dict]:
        """Chunk all documents in directory"""
        docs_path = Path(docs_dir).expanduser()
        all_chunks = []

        for filepath in docs_path.glob("*.txt"):
            text = filepath.read_text()
            metadata = {
                'source': filepath.name,
                'filepath': str(filepath)
            }

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

            print(f"📄 {filepath.name}: {len(chunks)} chunks")

        return all_chunks

if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_documents("~/rag/documents")

    print(f"\n✅ Total chunks: {len(chunks)}")
    print(f"\n📝 Sample chunk:")
    print(chunks[0])
```

```bash
pip install pathlib
python ~/rag/chunking.py
```

#### Part 2: Embedding Generation (45 minutes)

**2.1 Set Up Embedding Model**

```python
# File: ~/rag/embeddings.py

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import json
from pathlib import Path

class EmbeddingGenerator:
    """Generate embeddings for documents"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        Options:
        - all-MiniLM-L6-v2: Fast, 384 dim
        - all-mpnet-base-v2: Better quality, 768 dim
        - BAAI/bge-base-en-v1.5: Good quality, 768 dim
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to chunks"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()

        return chunks

    def save_embeddings(self, chunks: List[Dict], output_path: str):
        """Save chunks with embeddings to JSON"""
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(chunks, f, indent=2)

        print(f"💾 Saved {len(chunks)} chunks to {output_file}")

if __name__ == "__main__":
    from chunking import DocumentChunker

    # Chunk documents
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_documents("~/rag/documents")

    # Generate embeddings
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # Save
    embedder.save_embeddings(
        chunks_with_embeddings,
        "~/rag/embeddings.json"
    )
```

```bash
pip install sentence-transformers
python ~/rag/embeddings.py
```

#### Part 3: Build Retrieval System (60 minutes)

**3.1 Simple Vector Search**

```python
# File: ~/rag/retrieval.py

import numpy as np
import json
from typing import List, Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer

class SimpleRetriever:
    """Simple in-memory vector retrieval"""

    def __init__(self, embeddings_file: str, model_name: str = "all-MiniLM-L6-v2"):
        # Load chunks with embeddings
        with open(Path(embeddings_file).expanduser()) as f:
            self.chunks = json.load(f)

        # Extract embeddings as numpy array
        self.embeddings = np.array([
            chunk['embedding'] for chunk in self.chunks
        ])

        # Load embedding model for queries
        self.model = SentenceTransformer(model_name)

        print(f"✅ Loaded {len(self.chunks)} chunks")
        print(f"📊 Embedding shape: {self.embeddings.shape}")

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Dict]:
        """Search for similar chunks"""
        # Embed query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append({
                    'text': self.chunks[idx]['text'],
                    'metadata': self.chunks[idx]['metadata'],
                    'score': score
                })

        return results

if __name__ == "__main__":
    # Initialize retriever
    retriever = SimpleRetriever("~/rag/embeddings.json")

    # Test queries
    queries = [
        "How do I deploy LLMs on Kubernetes?",
        "What is a vector database?",
        "GPU optimization techniques"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']:.3f}):")
            print(f"  Source: {result['metadata']['source']}")
            print(f"  Text: {result['text'][:200]}...")
```

```bash
python ~/rag/retrieval.py
```

#### Part 4: Build Complete RAG System (60 minutes)

**4.1 Integrate with LLM**

```python
# File: ~/rag/rag_system.py

import requests
from retrieval import SimpleRetriever
from typing import List, Dict

class RAGSystem:
    """Complete RAG system with retrieval and generation"""

    def __init__(
        self,
        embeddings_file: str,
        llm_api_url: str = "http://localhost:8000/v1/chat/completions",
        model_name: str = "llama-2-7b-chat"
    ):
        self.retriever = SimpleRetriever(embeddings_file)
        self.llm_api_url = llm_api_url
        self.model_name = model_name

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context"""
        results = self.retriever.search(query, top_k=top_k)

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata']['source']
            text = result['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")

        return "\n\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> Dict:
        """Generate response using LLM with context"""
        # Build prompt
        system_prompt = """You are a helpful AI infrastructure assistant.
Answer questions based on the provided context. If the context doesn't
contain relevant information, say so clearly."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

        # Call LLM API
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.llm_api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            return {
                'answer': result['choices'][0]['message']['content'],
                'context': context,
                'success': True
            }
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'context': context,
                'success': False
            }

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """Complete RAG query"""
        # Retrieve
        context = self.retrieve_context(question, top_k=top_k)

        if verbose:
            print(f"\n🔍 Retrieved Context:")
            print("=" * 80)
            print(context)
            print("=" * 80)

        # Generate
        result = self.generate_response(question, context)

        if verbose:
            print(f"\n💬 Answer:")
            print("=" * 80)
            print(result['answer'])
            print("=" * 80)

        return result

if __name__ == "__main__":
    # Initialize RAG system (requires vLLM server running from Exercise 01)
    rag = RAGSystem(
        embeddings_file="~/rag/embeddings.json",
        llm_api_url="http://localhost:8000/v1/chat/completions"
    )

    # Test queries
    questions = [
        "What GPU memory do I need for a 70B parameter model?",
        "Explain vector databases for RAG systems",
        "How can I optimize GPU performance?",
    ]

    for question in questions:
        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        result = rag.query(question, top_k=2)
```

```bash
# Make sure vLLM server is running from Exercise 01
# Then run RAG system
pip install requests
python ~/rag/rag_system.py
```

### Challenge Tasks

**Challenge 1: Advanced Chunking**
- Implement semantic chunking (split at sentence boundaries)
- Try different chunk sizes and overlaps
- Measure impact on retrieval quality

**Challenge 2: Better Embeddings**
- Experiment with different embedding models
- Try `BAAI/bge-large-en-v1.5` or `intfloat/e5-large-v2`
- Compare retrieval quality

**Challenge 3: Reranking**
- Add a reranking step with cross-encoder
- Compare results with and without reranking

### Expected Results

- ✅ Documents chunked and embedded
- ✅ Retrieval returns relevant chunks (score > 0.5)
- ✅ LLM generates answers based on retrieved context
- ✅ System answers domain-specific questions accurately

### Deliverables

- [ ] Chunked documents with embeddings
- [ ] Working retrieval system
- [ ] Complete RAG system with LLM integration
- [ ] Test queries and results documented

---

## Exercise 03: Set Up Vector Database (Qdrant)

**Difficulty:** Intermediate
**Estimated Time:** 2-3 hours
**GPU Required:** No
**Cost:** Free (runs locally)

### Learning Objectives

- Deploy Qdrant vector database
- Create collections and index vectors
- Implement filtered search
- Optimize for production
- Integrate with RAG system

### Prerequisites

- Completed Exercise 02
- Docker installed
- Basic understanding of databases

### Step-by-Step Instructions

#### Part 1: Deploy Qdrant (30 minutes)

**1.1 Deploy with Docker**

```bash
# Pull Qdrant image
docker pull qdrant/qdrant

# Run Qdrant
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333/

# Check dashboard (optional)
# Open browser: http://localhost:6333/dashboard
```

**1.2 Install Python Client**

```bash
pip install qdrant-client
```

#### Part 2: Create Collection and Index Vectors (45 minutes)

**2.1 Set Up Qdrant Collection**

```python
# File: ~/rag/qdrant_setup.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import json
from pathlib import Path
from typing import List, Dict

class QdrantVectorDB:
    """Qdrant vector database manager"""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        print(f"✅ Connected to Qdrant at {host}:{port}")

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ):
        """Create collection for vectors"""
        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"⚠️  Collection '{collection_name}' already exists")
            return

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        print(f"✅ Created collection '{collection_name}'")

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Dict]
    ):
        """Insert chunks with embeddings"""
        points = []

        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=i,
                vector=chunk['embedding'],
                payload={
                    'text': chunk['text'],
                    'source': chunk['metadata']['source'],
                    'chunk_id': chunk['metadata']['chunk_id'],
                    'filepath': chunk['metadata'].get('filepath', '')
                }
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"📤 Uploaded batch {i//batch_size + 1}")

        print(f"✅ Inserted {len(points)} points")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        filter_source: str = None,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """Search vectors with optional filtering"""
        # Build filter if source specified
        query_filter = None
        if filter_source:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filter_source)
                    )
                ]
            )

        # Search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.payload['text'],
                'source': result.payload['source'],
                'score': result.score,
                'id': result.id
            })

        return formatted_results

if __name__ == "__main__":
    # Initialize Qdrant
    db = QdrantVectorDB()

    # Load chunks with embeddings
    with open(Path("~/rag/embeddings.json").expanduser()) as f:
        chunks = json.load(f)

    # Get embedding dimension
    vector_size = len(chunks[0]['embedding'])

    # Create collection
    collection_name = "ai_infrastructure_docs"
    db.create_collection(collection_name, vector_size)

    # Upsert chunks
    db.upsert_chunks(collection_name, chunks)

    # Test search
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "vector database features"
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    results = db.search(collection_name, query_vector, top_k=3)

    print(f"\n🔍 Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Text: {result['text'][:150]}...")
```

```bash
python ~/rag/qdrant_setup.py
```

#### Part 3: Integrate with RAG System (45 minutes)

**3.1 Update RAG System to Use Qdrant**

```python
# File: ~/rag/rag_qdrant.py

import requests
from sentence_transformers import SentenceTransformer
from qdrant_setup import QdrantVectorDB
from typing import List, Dict

class QdrantRAG:
    """RAG system using Qdrant vector database"""

    def __init__(
        self,
        collection_name: str = "ai_infrastructure_docs",
        llm_api_url: str = "http://localhost:8000/v1/chat/completions",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.db = QdrantVectorDB()
        self.collection_name = collection_name
        self.llm_api_url = llm_api_url
        self.embedding_model = SentenceTransformer(embedding_model)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_source: str = None
    ) -> List[Dict]:
        """Retrieve relevant chunks"""
        # Embed query
        query_vector = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )[0].tolist()

        # Search
        results = self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_source=filter_source
        )

        return results

    def generate(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        system_prompt = """You are an AI infrastructure expert. Answer questions
based on the provided context. Be specific and technical."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        payload = {
            "model": "llama-2-7b-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.llm_api_url, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    def query(
        self,
        question: str,
        top_k: int = 3,
        filter_source: str = None,
        verbose: bool = True
    ) -> Dict:
        """Complete RAG query"""
        # Retrieve
        results = self.retrieve(question, top_k, filter_source)

        # Build context
        context = "\n\n".join([
            f"[{r['source']}]\n{r['text']}"
            for r in results
        ])

        if verbose:
            print(f"\n🔍 Retrieved {len(results)} chunks")
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['source']} (score: {r['score']:.3f})")

        # Generate
        answer = self.generate(question, context)

        if verbose:
            print(f"\n💬 Answer:\n{answer}")

        return {
            'question': question,
            'answer': answer,
            'sources': results
        }

if __name__ == "__main__":
    # Initialize RAG with Qdrant
    rag = QdrantRAG()

    # Test queries
    questions = [
        "What are the key features of Qdrant?",
        "How much GPU memory does a 70B model need?",
        "What is continuous batching in vLLM?"
    ]

    for question in questions:
        print(f"\n{'='*80}")
        print(f"❓ {question}")
        result = rag.query(question, top_k=2)
```

```bash
python ~/rag/rag_qdrant.py
```

#### Part 4: Production Optimizations (30 minutes)

**4.1 Tune HNSW Parameters**

```python
# File: ~/rag/qdrant_optimize.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    HnswConfigDiff,
    OptimizersConfigDiff,
    QuantizationConfig,
    ScalarQuantization,
    ScalarType
)

client = QdrantClient("localhost", 6333)
collection_name = "ai_infrastructure_docs"

# Update HNSW configuration
client.update_collection(
    collection_name=collection_name,
    hnsw_config=HnswConfigDiff(
        m=32,  # Number of edges per node (higher = better quality, more memory)
        ef_construct=256,  # Construction time parameter
        full_scan_threshold=10000,  # When to use full scan
    )
)

# Update optimizer configuration
client.update_collection(
    collection_name=collection_name,
    optimizer_config=OptimizersConfigDiff(
        indexing_threshold=20000,  # When to trigger indexing
        memmap_threshold=50000,  # When to use memory mapping
    )
)

# Optional: Enable scalar quantization to save memory
client.update_collection(
    collection_name=collection_name,
    quantization_config=ScalarQuantization(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True
    )
)

print("✅ Updated Qdrant configuration")
```

**4.2 Benchmark Performance**

```python
# File: ~/rag/qdrant_benchmark.py

import time
import statistics
from qdrant_setup import QdrantVectorDB
from sentence_transformers import SentenceTransformer

# Initialize
db = QdrantVectorDB()
model = SentenceTransformer('all-MiniLM-L6-v2')
collection_name = "ai_infrastructure_docs"

# Test queries
queries = [
    "vector database optimization",
    "GPU memory requirements for LLMs",
    "Kubernetes deployment strategies",
    "model serving frameworks",
    "quantization techniques"
] * 20  # 100 queries

# Benchmark
latencies = []
for query in queries:
    query_vector = model.encode([query], normalize_embeddings=True)[0].tolist()

    start = time.time()
    results = db.search(collection_name, query_vector, top_k=5)
    latency = time.time() - start

    latencies.append(latency * 1000)  # Convert to ms

# Results
print(f"\n📊 Benchmark Results ({len(queries)} queries):")
print(f"Mean latency: {statistics.mean(latencies):.2f}ms")
print(f"Median latency: {statistics.median(latencies):.2f}ms")
print(f"P95 latency: {sorted(latencies)[int(0.95*len(latencies))]:.2f}ms")
print(f"P99 latency: {sorted(latencies)[int(0.99*len(latencies))]:.2f}ms")
```

### Challenge Tasks

**Challenge 1: Filtered Search**
- Implement search with multiple filters
- Filter by source, date range, or custom metadata

**Challenge 2: Hybrid Search**
- Combine dense and sparse vectors
- Implement BM25 + vector search

**Challenge 3: Production Deployment**
- Deploy Qdrant cluster (3 nodes)
- Set up replication and backups

### Expected Results

- ✅ Qdrant deployed and accessible
- ✅ Collection created with vectors indexed
- ✅ Search latency < 50ms for most queries
- ✅ RAG system using Qdrant successfully

### Deliverables

- [ ] Qdrant running with data indexed
- [ ] Benchmark results documented
- [ ] Updated RAG system using Qdrant
- [ ] Test queries showing filtered search

---

## Exercise 04: Fine-Tune LLM with LoRA

**Difficulty:** Advanced
**Estimated Time:** 4-5 hours
**GPU Required:** Yes (A10G or A100 recommended)
**Cost:** ~$2-4 for full exercise

### Learning Objectives

- Prepare dataset for fine-tuning
- Implement LoRA fine-tuning
- Evaluate fine-tuned model
- Deploy fine-tuned model with vLLM

### Prerequisites

- Completed Exercise 01
- Understanding of transformer architecture
- Familiarity with PyTorch

### (Continue with detailed steps for Exercise 04...)

### Expected Results

- ✅ Model fine-tuned on custom dataset
- ✅ Improved performance on target task
- ✅ Fine-tuned model deployed with vLLM

---

## Exercises 05-08: Summary

Due to length constraints, here are the key components:

**Exercise 05: Optimize LLM Inference**
- Implement quantization (GPTQ, AWQ)
- Compare FP16 vs INT8 vs INT4
- Benchmark throughput improvements
- Expected: 2-4x speedup with minimal quality loss

**Exercise 06: Deploy Production LLM on Kubernetes**
- Create Kubernetes manifests
- Deploy vLLM with GPU scheduling
- Implement HPA based on queue depth
- Add health checks and monitoring

**Exercise 07: Implement Monitoring for LLM Services**
- Set up Prometheus metrics
- Create Grafana dashboards
- Add distributed tracing
- Configure alerts for latency/errors

**Exercise 08: Build Multi-Model LLM API**
- Deploy multiple models (7B, 13B)
- Implement intelligent routing
- Add semantic caching
- Monitor costs and performance

---

## Completion Checklist

- [ ] Exercise 01: Deploy Llama 2 with vLLM
- [ ] Exercise 02: Build Basic RAG System
- [ ] Exercise 03: Set Up Vector Database
- [ ] Exercise 04: Fine-Tune with LoRA
- [ ] Exercise 05: Optimize LLM Inference
- [ ] Exercise 06: Production LLM on Kubernetes
- [ ] Exercise 07: Implement Monitoring
- [ ] Exercise 08: Build Multi-Model API

---

## Additional Resources

- See `resources.md` for documentation links
- Review lesson materials for theoretical background
- Join communities for help (Discord, forums)

**Good luck with the exercises!**
