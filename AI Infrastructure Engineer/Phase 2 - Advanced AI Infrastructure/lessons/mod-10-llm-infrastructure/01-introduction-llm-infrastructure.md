# Lesson 01: Introduction to LLM Infrastructure

## Table of Contents
1. [Introduction](#introduction)
2. [What is LLM Infrastructure?](#what-is-llm-infrastructure)
3. [Why LLM Infrastructure is Different](#why-llm-infrastructure-is-different)
4. [Unique Challenges of LLM Infrastructure](#unique-challenges-of-llm-infrastructure)
5. [LLM Deployment Patterns](#llm-deployment-patterns)
6. [Hardware Requirements](#hardware-requirements)
7. [Cost Considerations](#cost-considerations)
8. [LLM Serving Frameworks Overview](#llm-serving-frameworks-overview)
9. [Industry Landscape](#industry-landscape)
10. [Getting Started Roadmap](#getting-started-roadmap)
11. [Summary](#summary)

## Introduction

Large Language Models (LLMs) have revolutionized artificial intelligence, powering everything from chatbots and code assistants to document analysis and content generation. However, deploying and managing LLMs in production presents unique infrastructure challenges that differ significantly from traditional machine learning or web application deployment.

This lesson provides a comprehensive introduction to LLM infrastructure, covering the fundamental concepts, challenges, and patterns you need to understand before diving into specific implementations. By the end of this lesson, you'll have a solid foundation for building production-grade LLM systems.

### Learning Objectives

After completing this lesson, you will be able to:
- Define LLM infrastructure and explain its unique characteristics
- Identify the key differences between LLM and traditional ML infrastructure
- Understand the major challenges in deploying and scaling LLMs
- Describe common LLM deployment patterns and when to use each
- Specify hardware requirements for different LLM workloads
- Calculate and estimate costs for LLM infrastructure
- Evaluate different LLM serving frameworks and tools
- Make informed decisions about LLM infrastructure architecture

## What is LLM Infrastructure?

### Definition

**LLM Infrastructure** refers to the complete technology stack required to deploy, serve, manage, and operate Large Language Models in production environments. This includes:

- **Compute Resources**: GPUs, CPUs, memory, and specialized hardware for running LLMs
- **Model Serving**: Systems for hosting models and handling inference requests
- **Storage**: Solutions for storing large model files, embeddings, and vector databases
- **Orchestration**: Kubernetes and container management for LLM workloads
- **Networking**: Load balancers, API gateways, and routing for LLM services
- **Monitoring**: Observability tools for tracking performance, costs, and quality
- **Data Pipelines**: Systems for preparing training data and fine-tuning
- **Supporting Services**: Vector databases, caching layers, and integration tools

### Scope of LLM Infrastructure

LLM infrastructure encompasses several distinct but interconnected domains:

#### 1. Inference Infrastructure
The systems and resources needed to serve model predictions:
- Model hosting and serving
- Request handling and batching
- Response generation and streaming
- Load balancing and auto-scaling

#### 2. Fine-Tuning Infrastructure
Resources for customizing models with domain-specific data:
- Training clusters and GPU management
- Data preprocessing pipelines
- Experiment tracking and model versioning
- Model evaluation and validation

#### 3. RAG Infrastructure
Components for Retrieval-Augmented Generation systems:
- Vector databases for embeddings
- Document processing and chunking
- Retrieval and ranking systems
- Embedding model serving

#### 4. Supporting Infrastructure
Additional systems that enable LLM applications:
- Caching layers (Redis, Memcached)
- API gateways and authentication
- Monitoring and logging
- Cost tracking and optimization

### Example LLM Infrastructure Stack

Here's what a complete LLM infrastructure stack might look like:

```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                    │
│  (Chatbots, Code Assistants, Document Analysis, etc.)  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                      API Gateway                         │
│        (Authentication, Rate Limiting, Routing)         │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────────────────┐            ┌──────────────────────┐
│   LLM Serving     │            │   Vector Database    │
│   (vLLM, TGI)     │            │ (Qdrant, Weaviate)   │
│                   │            │                      │
│ ┌───────────────┐ │            │ ┌────────────────┐  │
│ │  Llama 2 7B   │ │            │ │   Embeddings   │  │
│ └───────────────┘ │            │ │   (1M vectors) │  │
│ ┌───────────────┐ │            │ └────────────────┘  │
│ │  Mistral 7B   │ │            │                      │
│ └───────────────┘ │            └──────────────────────┘
└───────────────────┘
        │                                   │
┌───────────────────────────────────────────────────────┐
│              Kubernetes Orchestration                  │
│  (Pod Management, Auto-scaling, Service Discovery)    │
└───────────────────────────────────────────────────────┘
        │
┌───────────────────────────────────────────────────────┐
│                  Compute Resources                     │
│      (GPU Nodes: A100, A10G, T4, etc.)                │
└───────────────────────────────────────────────────────┘
        │
┌───────────────────────────────────────────────────────┐
│              Monitoring & Observability                │
│    (Prometheus, Grafana, ELK Stack, Cost Tracking)   │
└───────────────────────────────────────────────────────┘
```

## Why LLM Infrastructure is Different

LLM infrastructure differs from traditional ML infrastructure in several fundamental ways. Understanding these differences is crucial for building effective LLM systems.

### 1. Model Size and Memory Requirements

**Traditional ML Models:**
- Size: Typically 10MB - 1GB
- Memory: Fits comfortably in CPU or single GPU memory
- Example: BERT-base (110M parameters) = ~440MB

**Large Language Models:**
- Size: 5GB - 100GB+ (compressed)
- Memory: Requires multiple GPUs or specialized hardware
- Example: Llama 2 70B = ~140GB in FP16

**Impact on Infrastructure:**
```python
# Memory calculation for LLM inference
def calculate_memory_requirements(num_parameters, precision="fp16"):
    """
    Calculate minimum GPU memory needed for LLM inference
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "int8": 1,
        "int4": 0.5
    }

    # Model weights
    model_memory_gb = (num_parameters * bytes_per_param[precision]) / (1024**3)

    # KV cache (rough estimate: 20% of model size per request)
    kv_cache_gb = model_memory_gb * 0.2

    # Activation memory (rough estimate: 30% of model size)
    activation_gb = model_memory_gb * 0.3

    total_memory_gb = model_memory_gb + kv_cache_gb + activation_gb

    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2),
        "recommended_gpu_memory_gb": round(total_memory_gb * 1.2, 2)  # 20% buffer
    }

# Example: Llama 2 7B
print("Llama 2 7B (FP16):")
print(calculate_memory_requirements(7_000_000_000, "fp16"))
# Output:
# {
#   'model_memory_gb': 13.04,
#   'kv_cache_gb': 2.61,
#   'activation_gb': 3.91,
#   'total_memory_gb': 19.56,
#   'recommended_gpu_memory_gb': 23.47
# }

# Example: Llama 2 70B
print("\nLlama 2 70B (FP16):")
print(calculate_memory_requirements(70_000_000_000, "fp16"))
# Requires 234GB+ GPU memory!
```

### 2. Compute Intensity and Latency

**Traditional ML:**
- Inference time: Milliseconds (often < 50ms)
- Compute: Can run on CPUs for many models
- Batching: Easy and effective

**LLMs:**
- Inference time: Seconds (1-30 seconds for full responses)
- Compute: Requires GPUs for practical latency
- Batching: Complex due to variable sequence lengths

**Performance Comparison:**

| Model Type | Hardware | Latency (p95) | Throughput |
|-----------|----------|---------------|------------|
| BERT Classification | CPU (8 cores) | 20ms | 400 req/sec |
| BERT Classification | GPU (T4) | 5ms | 2000 req/sec |
| Llama 2 7B (50 tokens) | CPU (32 cores) | 45 seconds | 0.02 req/sec |
| Llama 2 7B (50 tokens) | GPU (T4) | 3 seconds | 0.33 req/sec |
| Llama 2 7B (50 tokens) | GPU (A100) | 1.2 seconds | 0.83 req/sec |

### 3. Cost Structure

**Traditional ML:**
- Predictable per-request costs
- Often feasible to run on CPUs
- Scaling costs grow linearly

**LLMs:**
- High fixed costs (GPU infrastructure)
- Variable costs per token generated
- Optimization critical to viability

**Cost Comparison Example:**

```python
# Monthly cost calculator
def calculate_monthly_llm_cost(
    requests_per_day,
    avg_input_tokens,
    avg_output_tokens,
    gpu_type="A10G",
    gpu_cost_per_hour=1.20
):
    """
    Calculate monthly infrastructure costs for LLM serving
    """
    # GPU requirements (simplified)
    gpu_requirements = {
        "7B_model": {"A10G": 1, "T4": 1, "A100": 0.5},
        "13B_model": {"A10G": 1, "T4": 2, "A100": 1},
        "70B_model": {"A10G": 4, "T4": 8, "A100": 2}
    }

    # Throughput estimates (requests per GPU per second)
    throughput = {
        "7B_model": {"A10G": 0.5, "T4": 0.3, "A100": 1.0},
    }

    # Calculate required GPUs for 7B model
    requests_per_second = requests_per_day / 86400
    gpus_needed = max(1, requests_per_second / throughput["7B_model"][gpu_type])

    # Monthly GPU cost
    monthly_gpu_cost = gpus_needed * gpu_cost_per_hour * 24 * 30

    # Total tokens processed
    total_requests_monthly = requests_per_day * 30
    total_tokens = total_requests_monthly * (avg_input_tokens + avg_output_tokens)

    # Cost per request and per token
    cost_per_request = monthly_gpu_cost / total_requests_monthly
    cost_per_million_tokens = (monthly_gpu_cost / total_tokens) * 1_000_000

    return {
        "requests_per_day": requests_per_day,
        "gpus_needed": round(gpus_needed, 2),
        "monthly_gpu_cost": round(monthly_gpu_cost, 2),
        "cost_per_request": round(cost_per_request, 4),
        "cost_per_million_tokens": round(cost_per_million_tokens, 2),
        "gpu_type": gpu_type
    }

# Example: Medium traffic service
print("Medium Traffic (10,000 requests/day):")
print(calculate_monthly_llm_cost(
    requests_per_day=10_000,
    avg_input_tokens=500,
    avg_output_tokens=200,
    gpu_type="A10G",
    gpu_cost_per_hour=1.20
))
# Output shows real costs for infrastructure planning
```

### 4. Infrastructure Complexity

**Traditional ML:**
- Standard deployment patterns (REST APIs, batch processing)
- Well-understood scaling strategies
- Mature tooling ecosystem

**LLMs:**
- Complex deployment requirements (GPU scheduling, tensor parallelism)
- Novel optimization techniques (quantization, KV cache management)
- Rapidly evolving tooling landscape

## Unique Challenges of LLM Infrastructure

### Challenge 1: Memory Constraints

The primary constraint in LLM serving is GPU memory (VRAM). Unlike traditional models, LLMs cannot simply "fit" into available memory without careful planning.

**Memory Components:**

1. **Model Weights**: The parameters themselves (largest component)
2. **KV Cache**: Cached key-value pairs for faster generation (grows with context length)
3. **Activation Memory**: Temporary tensors during computation
4. **Batch Memory**: Additional memory per concurrent request

**Mitigation Strategies:**
- Model quantization (FP16, INT8, INT4)
- KV cache optimization (PagedAttention in vLLM)
- Tensor parallelism (split model across GPUs)
- Offloading (CPU/disk for less frequently used layers)

### Challenge 2: Inference Latency

LLM inference is inherently sequential - each token must be generated before the next can begin. This creates unique latency challenges.

**Latency Factors:**
- **Model Size**: Larger models = more compute per token
- **Context Length**: Longer contexts = larger KV cache operations
- **Sequence Length**: More output tokens = linear increase in latency
- **Batch Size**: Larger batches = better GPU utilization but higher latency per request

**Optimization Strategies:**
```python
# Latency optimization techniques
optimization_techniques = {
    "quantization": {
        "latency_improvement": "1.5-3x",
        "quality_impact": "minimal to moderate",
        "implementation": "GPTQ, AWQ, GGUF"
    },
    "flash_attention": {
        "latency_improvement": "1.5-2x",
        "quality_impact": "none",
        "implementation": "vLLM, TensorRT-LLM"
    },
    "speculative_decoding": {
        "latency_improvement": "1.5-2.5x",
        "quality_impact": "none",
        "implementation": "vLLM, HF Transformers"
    },
    "continuous_batching": {
        "throughput_improvement": "3-10x",
        "latency_impact": "minimal",
        "implementation": "vLLM, TensorRT-LLM"
    }
}
```

### Challenge 3: Cost Management

LLM infrastructure costs can quickly spiral without proper management. A single GPU can cost $1-10 per hour, and large deployments may require dozens of GPUs.

**Cost Drivers:**
- GPU rental costs (AWS, GCP, Azure)
- Model size and complexity
- Request volume and traffic patterns
- Inefficient resource utilization

**Cost Optimization Framework:**

```python
# Cost optimization decision framework
class LLMCostOptimizer:
    def __init__(self, budget_per_month, expected_requests_per_day):
        self.budget = budget_per_month
        self.daily_requests = expected_requests_per_day

    def recommend_strategy(self):
        """
        Recommend cost optimization strategy based on usage
        """
        strategies = []

        # Calculate constraints
        cost_per_request_limit = self.budget / (self.daily_requests * 30)

        if cost_per_request_limit < 0.001:
            strategies.append("Critical: Use smallest viable model (7B)")
            strategies.append("Implement aggressive caching")
            strategies.append("Use INT8 or INT4 quantization")
            strategies.append("Consider spot instances for non-critical workloads")
        elif cost_per_request_limit < 0.01:
            strategies.append("Use 7B or 13B model with quantization")
            strategies.append("Implement request batching")
            strategies.append("Use mid-tier GPUs (A10G, T4)")
        else:
            strategies.append("Budget allows for larger models or premium GPUs")
            strategies.append("Focus on performance over cost optimization")

        return strategies

# Example usage
optimizer = LLMCostOptimizer(budget_per_month=5000, expected_requests_per_day=50_000)
print(optimizer.recommend_strategy())
```

### Challenge 4: Quality and Consistency

Unlike deterministic traditional ML models, LLMs are probabilistic and can produce varied outputs. Ensuring quality and consistency in production is challenging.

**Quality Challenges:**
- Non-deterministic outputs (even with temperature=0)
- Hallucinations and factual errors
- Prompt sensitivity
- Context length limitations

**Quality Assurance Strategies:**
- Comprehensive prompt testing
- Output validation and filtering
- A/B testing for model updates
- Human-in-the-loop review for critical applications
- RAG for factual grounding

### Challenge 5: Scaling Complexity

Scaling LLM infrastructure differs from traditional web services:

**Challenges:**
- GPU scheduling and allocation
- Stateful inference (context management)
- Variable processing times
- Cold start latency
- Cost implications of over-provisioning

**Scaling Patterns:**
```yaml
# Example: Auto-scaling configuration for LLM pods
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Slower scale-down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60  # Faster scale-up
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

### Challenge 6: Model Updates and Versioning

Deploying new models or versions requires careful orchestration:

**Challenges:**
- Large model downloads (10-100GB)
- GPU memory constraints during deployment
- Canary deployments and gradual rollouts
- A/B testing infrastructure
- Rollback capabilities

## LLM Deployment Patterns

Different use cases require different deployment patterns. Understanding these patterns helps you architect appropriate solutions.

### Pattern 1: Single-Model Inference API

**Description**: Host a single LLM with an API endpoint for inference requests.

**Use Cases:**
- Proof of concept projects
- Single-purpose applications (e.g., chatbot)
- Low to medium traffic services

**Architecture:**
```
┌──────────┐      ┌─────────────────┐      ┌──────────────┐
│  Client  │─────▶│  Load Balancer  │─────▶│  LLM Server  │
│          │      │   (NGINX/ALB)   │      │  (vLLM Pod)  │
└──────────┘      └─────────────────┘      └──────────────┘
                                                   │
                                            ┌──────────────┐
                                            │  GPU: A10G   │
                                            │  Model: 7B   │
                                            └──────────────┘
```

**Pros:**
- Simple to implement and manage
- Lower infrastructure complexity
- Predictable resource usage

**Cons:**
- Single point of failure (without replicas)
- Limited flexibility for different use cases
- Resource underutilization if traffic is sporadic

**Example Configuration:**
```python
# Simple vLLM server deployment
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Simple API wrapper (using FastAPI)
from fastapi import FastAPI
app = FastAPI()

@app.post("/v1/completions")
async def generate(prompt: str):
    outputs = llm.generate([prompt], sampling_params)
    return {"text": outputs[0].outputs[0].text}
```

### Pattern 2: Multi-Model Serving Platform

**Description**: Host multiple LLMs simultaneously, routing requests based on model selection or use case.

**Use Cases:**
- Organizations with diverse LLM use cases
- A/B testing different models
- Specialized models for different tasks (chat, code, summarization)

**Architecture:**
```
                   ┌─────────────────┐
                   │  API Gateway    │
                   │  (Model Router) │
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
   ┌──────▼──────┐   ┌──────▼──────┐  ┌──────▼──────┐
   │  Chat Model │   │  Code Model │  │ Summary Model│
   │  Llama-2-7B │   │ CodeLlama   │  │  Mistral-7B │
   └─────────────┘   └─────────────┘  └─────────────┘
```

**Implementation Example:**
```python
# Multi-model router
from fastapi import FastAPI, HTTPException
from typing import Dict
import asyncio

class MultiModelRouter:
    def __init__(self):
        self.models: Dict[str, LLM] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize multiple models"""
        model_configs = {
            "chat": {
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "gpu_ids": [0]
            },
            "code": {
                "model": "codellama/CodeLlama-7b-hf",
                "gpu_ids": [1]
            },
            "summary": {
                "model": "mistralai/Mistral-7B-v0.1",
                "gpu_ids": [2]
            }
        }

        for name, config in model_configs.items():
            self.models[name] = LLM(
                model=config["model"],
                tensor_parallel_size=1,
                # Assign specific GPU
                # This requires proper GPU allocation in K8s
            )

    async def generate(self, model_name: str, prompt: str, **kwargs):
        """Route request to appropriate model"""
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        llm = self.models[model_name]
        outputs = llm.generate([prompt], SamplingParams(**kwargs))
        return outputs[0].outputs[0].text

# FastAPI app
app = FastAPI()
router = MultiModelRouter()

@app.post("/v1/{model_name}/completions")
async def generate(model_name: str, prompt: str):
    text = await router.generate(model_name, prompt)
    return {"model": model_name, "text": text}
```

### Pattern 3: RAG (Retrieval-Augmented Generation)

**Description**: Combine LLM with a vector database for retrieval, enabling grounded generation based on specific documents or knowledge bases.

**Use Cases:**
- Question answering over documents
- Customer support with company knowledge base
- Code search and explanation
- Legal or medical document analysis

**Architecture:**
```
┌────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client   │────▶│  RAG API    │────▶│Vector DB    │
└────────────┘     └──────┬──────┘     │ (Qdrant)    │
                          │            └─────────────┘
                          │
                   ┌──────▼──────┐
                   │  LLM Server │
                   │  (vLLM)     │
                   └─────────────┘
```

**Flow:**
1. User submits query
2. Query is embedded and used to search vector DB
3. Relevant documents are retrieved
4. Context + query sent to LLM
5. LLM generates grounded response

**Example Implementation:**
```python
# RAG system implementation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, llm: LLM, vector_db_url: str):
        self.llm = llm
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_client = QdrantClient(url=vector_db_url)
        self.collection_name = "documents"

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant documents from vector DB"""
        # Embed query
        query_vector = self.embedding_model.encode(query).tolist()

        # Search vector DB
        results = self.vector_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        # Extract documents
        documents = [hit.payload["text"] for hit in results]
        return documents

    def generate(self, query: str, context_docs: list):
        """Generate response using retrieved context"""
        # Build prompt with context
        context = "\n\n".join(context_docs)
        prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        outputs = self.llm.generate([prompt], SamplingParams(
            temperature=0.3,
            max_tokens=512
        ))

        return outputs[0].outputs[0].text

    def rag_query(self, query: str):
        """Complete RAG flow"""
        # Retrieve relevant documents
        docs = self.retrieve(query, top_k=5)

        # Generate response with context
        response = self.generate(query, docs)

        return {
            "query": query,
            "response": response,
            "sources": docs
        }

# API endpoint
@app.post("/v1/rag/query")
async def rag_query(query: str):
    result = rag_system.rag_query(query)
    return result
```

### Pattern 4: Fine-Tuning Pipeline

**Description**: Infrastructure for fine-tuning LLMs with custom data, separate from inference infrastructure.

**Use Cases:**
- Domain adaptation (medical, legal, financial)
- Style adaptation (brand voice, formality)
- Task-specific optimization
- Reducing model size while maintaining performance

**Architecture:**
```
┌────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Training  │────▶│  Fine-tuning    │────▶│  Model       │
│  Data      │     │  Pipeline       │     │  Registry    │
└────────────┘     └─────────────────┘     └──────────────┘
                          │                       │
                   ┌──────▼──────┐               │
                   │  GPU Cluster│               │
                   │  (Training) │               │
                   └─────────────┘               │
                                                 │
                                          ┌──────▼──────┐
                                          │  Inference  │
                                          │  Deployment │
                                          └─────────────┘
```

**Workflow:**
```python
# Fine-tuning pipeline orchestration
class FineTuningPipeline:
    def __init__(self, base_model: str, output_dir: str):
        self.base_model = base_model
        self.output_dir = output_dir

    def prepare_data(self, raw_data_path: str):
        """Prepare and validate training data"""
        # Data preprocessing logic
        pass

    def train(self, training_config: dict):
        """Execute fine-tuning job"""
        from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=True,  # Use 8-bit for memory efficiency
            device_map="auto"
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )

        # Train (simplified - actual implementation needs dataset, etc.)
        # trainer = Trainer(model=model, args=training_args, ...)
        # trainer.train()

        return model

    def evaluate(self, model, eval_dataset):
        """Evaluate fine-tuned model"""
        # Evaluation logic
        pass

    def deploy(self, model_path: str):
        """Deploy fine-tuned model to inference"""
        # Deployment logic
        pass
```

### Pattern 5: Hybrid Cloud Deployment

**Description**: Distribute LLM workloads across cloud providers and on-premises infrastructure for cost, compliance, or performance reasons.

**Use Cases:**
- Cost optimization (using spot instances, reserved capacity)
- Data residency requirements
- Hybrid cloud strategies
- Disaster recovery

**Architecture:**
```
                    ┌─────────────────┐
                    │  Global Router  │
                    │  (Multi-Cloud)  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌──────▼──────┐   ┌──────▼──────┐
    │   AWS     │     │    GCP      │   │ On-Premises │
    │  (Primary)│     │  (Failover) │   │  (Sensitive)│
    └───────────┘     └─────────────┘   └─────────────┘
```

## Hardware Requirements

Understanding hardware requirements is crucial for LLM infrastructure planning.

### GPU Types and Use Cases

| GPU Model | VRAM | Performance | Cost/Hour | Best For |
|-----------|------|-------------|-----------|----------|
| NVIDIA T4 | 16GB | 1x | $0.35-0.50 | Small models (7B), development |
| NVIDIA A10G | 24GB | 2.5x | $1.00-1.50 | 7B-13B models, production |
| NVIDIA A100 (40GB) | 40GB | 5x | $3.00-4.00 | 13B-30B models, training |
| NVIDIA A100 (80GB) | 80GB | 5x | $4.00-5.50 | 30B-70B models, large batch |
| NVIDIA H100 | 80GB | 8x | $8.00-10.00 | Largest models, highest performance |

### Model Size to GPU Mapping

```python
# GPU requirements estimator
def estimate_gpu_requirements(
    model_size_billions: float,
    precision: str = "fp16",
    tensor_parallel: int = 1
):
    """
    Estimate GPU requirements for a given model size
    """
    # Calculate memory per GPU
    memory_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    model_memory_gb = (model_size_billions * 1e9 * memory_per_param[precision]) / (1024**3)

    # Add overhead (KV cache, activations)
    total_memory_gb = model_memory_gb * 1.5

    # Divide by tensor parallel
    memory_per_gpu = total_memory_gb / tensor_parallel

    # Recommend GPU
    recommendations = []
    if memory_per_gpu <= 14:
        recommendations.append(("T4", 16, tensor_parallel))
    if memory_per_gpu <= 22:
        recommendations.append(("A10G", 24, tensor_parallel))
    if memory_per_gpu <= 38:
        recommendations.append(("A100-40GB", 40, tensor_parallel))
    if memory_per_gpu <= 75:
        recommendations.append(("A100-80GB", 80, tensor_parallel))

    return {
        "model_size_b": model_size_billions,
        "precision": precision,
        "tensor_parallel": tensor_parallel,
        "memory_per_gpu_gb": round(memory_per_gpu, 2),
        "recommendations": recommendations
    }

# Examples
print("7B Model:")
print(estimate_gpu_requirements(7, "fp16", 1))

print("\n70B Model:")
print(estimate_gpu_requirements(70, "fp16", 4))
```

### CPU and Memory Requirements

GPUs aren't the only consideration:

- **CPU**: 8-16 cores per GPU for preprocessing, tokenization
- **System RAM**: 64GB-256GB for data loading, caching
- **Storage**: NVMe SSD recommended for fast model loading
  - 100GB-500GB per model
  - IOPS matters for multi-model serving
- **Network**: 10Gbps+ for multi-GPU communication, fast model downloads

## Cost Considerations

### Infrastructure Cost Breakdown

```python
# Comprehensive cost calculator
class LLMCostCalculator:
    def __init__(self):
        self.gpu_costs = {
            "T4": 0.40,
            "A10G": 1.20,
            "A100-40GB": 3.50,
            "A100-80GB": 5.00,
            "H100": 8.50
        }

    def calculate_monthly_cost(
        self,
        gpu_type: str,
        num_gpus: int,
        utilization: float = 1.0,
        storage_gb: int = 100,
        network_gb: int = 1000,
        additional_costs: dict = None
    ):
        """
        Calculate comprehensive monthly infrastructure costs
        """
        # GPU costs
        gpu_hourly = self.gpu_costs[gpu_type] * num_gpus * utilization
        gpu_monthly = gpu_hourly * 24 * 30

        # Storage costs (NVMe SSD: ~$0.15/GB/month)
        storage_monthly = storage_gb * 0.15

        # Network costs (egress: ~$0.08/GB)
        network_monthly = network_gb * 0.08

        # Additional costs (monitoring, load balancer, etc.)
        additional_monthly = 0
        if additional_costs:
            additional_monthly = sum(additional_costs.values())

        total_monthly = (
            gpu_monthly +
            storage_monthly +
            network_monthly +
            additional_monthly
        )

        return {
            "gpu_monthly": round(gpu_monthly, 2),
            "storage_monthly": round(storage_monthly, 2),
            "network_monthly": round(network_monthly, 2),
            "additional_monthly": round(additional_monthly, 2),
            "total_monthly": round(total_monthly, 2),
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "utilization": utilization
        }

# Example calculations
calculator = LLMCostCalculator()

print("Single A10G Instance:")
print(calculator.calculate_monthly_cost(
    gpu_type="A10G",
    num_gpus=1,
    utilization=0.7,  # 70% utilization
    storage_gb=150,
    network_gb=2000,
    additional_costs={"monitoring": 50, "load_balancer": 30}
))

print("\n4x A100 Cluster:")
print(calculator.calculate_monthly_cost(
    gpu_type="A100-80GB",
    num_gpus=4,
    utilization=0.85,
    storage_gb=500,
    network_gb=10000,
    additional_costs={"monitoring": 200, "load_balancer": 100}
))
```

### Cost Optimization Strategies

1. **Right-Sizing**
   - Start with smallest viable model
   - Monitor actual GPU utilization
   - Scale up only when needed

2. **Quantization**
   - INT8: ~50% cost reduction, minimal quality loss
   - INT4: ~75% cost reduction, moderate quality loss

3. **Spot Instances**
   - 50-70% cost savings for non-critical workloads
   - Implement graceful handling of interruptions

4. **Caching**
   - Cache embeddings and common responses
   - Can reduce compute by 30-60%

5. **Request Batching**
   - Improve GPU utilization from 30% to 80%+
   - Better throughput with minimal latency impact

## LLM Serving Frameworks Overview

### vLLM

**Best For**: High-throughput inference, production deployments

**Key Features:**
- PagedAttention for efficient memory management
- Continuous batching
- OpenAI-compatible API
- Support for quantization

**Performance**: 10-20x throughput vs. Hugging Face Transformers

```python
# vLLM quick start
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### Text Generation Inference (TGI)

**Best For**: Hugging Face ecosystem integration, production deployments

**Key Features:**
- Optimized for Hugging Face models
- Token streaming
- Tensor parallelism
- Flash Attention support

### TensorRT-LLM

**Best For**: Maximum performance on NVIDIA GPUs

**Key Features:**
- Highly optimized inference engine
- FP8 support on H100
- Multi-GPU/multi-node support
- Advanced quantization

**Trade-off**: More complex setup, less flexibility

### Ray Serve

**Best For**: Multi-model serving, complex pipelines

**Key Features:**
- Distributed serving framework
- Model composition and pipelines
- Auto-scaling
- Integration with Ray ecosystem

### LangChain/LlamaIndex

**Best For**: RAG applications, complex chains

**Key Features:**
- High-level abstractions for LLM apps
- RAG components
- Agent frameworks
- Multiple LLM provider support

## Industry Landscape

### Major Cloud Providers

**AWS**
- Bedrock: Managed LLM service
- SageMaker: Custom LLM deployment
- EC2 instances: P4, P5 for self-managed

**Google Cloud**
- Vertex AI: Managed LLM service
- GKE: Kubernetes-based deployment
- TPU support for some models

**Azure**
- Azure OpenAI Service: Managed GPT-4, etc.
- ML Studio: Custom deployment
- ND-series VMs: GPU instances

### Specialized GPU Cloud Providers

**Lambda Labs**: Cost-effective, developer-friendly
**RunPod**: Serverless GPU, flexible pricing
**CoreWeave**: High-performance GPU infrastructure
**Paperspace**: Gradient platform for ML

### Vector Database Landscape

**Pinecone**: Managed, scalable, easy to use
**Weaviate**: Open-source, flexible, good for hybrid search
**Qdrant**: High-performance, Rust-based, production-ready
**Chroma**: Lightweight, developer-friendly, embeddable
**Milvus**: Scalable, enterprise features

## Getting Started Roadmap

### Phase 1: Local Development (Week 1)
1. Set up Python environment
2. Install transformers and vLLM
3. Run small models locally (Llama 2 7B)
4. Experiment with prompts and parameters

### Phase 2: Cloud Deployment (Weeks 2-3)
1. Set up cloud GPU instance
2. Deploy vLLM server
3. Build OpenAI-compatible API
4. Implement basic monitoring

### Phase 3: Production Features (Weeks 4-5)
1. Add caching layer
2. Implement load balancing
3. Set up comprehensive monitoring
4. Add authentication and rate limiting

### Phase 4: Advanced (Weeks 6-8)
1. Build RAG system with vector database
2. Implement fine-tuning pipeline
3. Optimize with quantization
4. Deploy on Kubernetes

## Summary

This lesson covered the foundations of LLM infrastructure:

### Key Takeaways

1. **LLM infrastructure is fundamentally different** from traditional ML due to model size, memory requirements, and compute intensity

2. **Major challenges include**:
   - Memory constraints requiring careful GPU selection
   - Inference latency requiring optimization techniques
   - Cost management requiring strategic decisions
   - Quality assurance for probabilistic outputs

3. **Common deployment patterns**:
   - Single-model inference API
   - Multi-model serving platforms
   - RAG systems
   - Fine-tuning pipelines
   - Hybrid cloud deployments

4. **Hardware selection is critical**:
   - Match GPU memory to model size
   - Consider cost vs. performance trade-offs
   - Plan for scaling and growth

5. **Cost optimization is essential**:
   - Right-size models and infrastructure
   - Leverage quantization and caching
   - Monitor and optimize continuously

6. **Framework choice matters**:
   - vLLM for high-throughput production
   - TGI for Hugging Face integration
   - TensorRT-LLM for maximum performance
   - Ray Serve for complex pipelines

### Next Steps

In the next lesson, we'll dive deep into vLLM deployment, covering:
- Detailed architecture and features
- Production deployment configurations
- Performance optimization techniques
- Kubernetes deployment strategies
- Monitoring and troubleshooting

### Self-Assessment Questions

1. What are the three main memory components in LLM inference?
2. Why can't LLMs typically run efficiently on CPUs?
3. What is the primary cost driver in LLM infrastructure?
4. When would you choose a RAG pattern over fine-tuning?
5. How much GPU memory do you need for a 7B parameter model in FP16?
6. What are the benefits of continuous batching?
7. Name three cost optimization strategies for LLM infrastructure
8. What's the difference between vLLM and TensorRT-LLM?

### Recommended Reading

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course/)
- [NVIDIA TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)

---

**Next Lesson**: [02-vllm-deployment.md](./02-vllm-deployment.md)
