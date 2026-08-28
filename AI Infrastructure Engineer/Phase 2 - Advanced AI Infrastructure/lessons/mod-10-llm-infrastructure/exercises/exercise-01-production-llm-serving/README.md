# Exercise 01: Production LLM Serving Platform (vLLM + Kubernetes)

**Estimated Time**: 36-44 hours

## Business Context

Your startup is building an AI-powered customer support platform and needs to deploy LLMs at scale. The requirements are demanding:

**Current Situation**:
- Using OpenAI API for LLM inference ($150,000/month)
- 1M requests/day (growing 20% monthly)
- Average response time: 800ms (too slow)
- Zero control over model behavior or data privacy
- Compliance team blocked deployment in EU (data sovereignty)

**Business Goals**:
- **Reduce costs by 70%**: Self-hosted LLMs on GPU infrastructure
- **Improve latency**: <300ms P95 response time
- **Data privacy**: All data stays in company infrastructure
- **Scalability**: Handle 5M requests/day by Q4
- **Multi-model support**: Different models for different use cases (chat, summarization, code)

**Your Mission**: Build a **production-grade LLM serving platform** using open-source tools that:
1. Deploys multiple LLMs (Llama 2, Mistral, CodeLlama) on Kubernetes
2. Serves requests via OpenAI-compatible API for easy migration
3. Auto-scales based on traffic with GPU resource optimization
4. Provides comprehensive monitoring and cost tracking
5. Achieves <$50,000/month operating costs (70% reduction)

## Learning Objectives

After completing this exercise, you will be able to:

1. Deploy production LLMs using vLLM inference engine
2. Design and implement multi-model serving architecture on Kubernetes
3. Configure auto-scaling for GPU workloads with cost optimization
4. Implement request routing, load balancing, and caching
5. Set up comprehensive monitoring for LLM performance and costs
6. Optimize LLM inference through quantization and batching
7. Implement OpenAI-compatible API for seamless migration
8. Handle production challenges: rate limiting, error handling, failover

## Prerequisites

- Module 04 (Kubernetes) - pod deployments, services, HPA
- Module 07 (GPU Computing) - GPU resource management
- Module 08 (Monitoring) - Prometheus, Grafana
- Python programming (advanced level)
- Access to GPU instances (T4, A10G, or A100)
- Basic understanding of transformer models

## Problem Statement

Build a **Production LLM Serving Platform** that:

1. **Multi-Model Deployment**:
   - Llama 2 7B (general chat)
   - Mistral 7B (fast inference)
   - CodeLlama 7B (code generation)
   - Each with optimized configurations

2. **Scalable Infrastructure**:
   - Kubernetes deployment on GKE/EKS
   - GPU node pools with auto-scaling
   - Horizontal Pod Autoscaler based on custom metrics
   - Cost-optimized spot instances

3. **API Layer**:
   - OpenAI-compatible REST API
   - Request routing to appropriate model
   - Rate limiting and authentication
   - Response caching

4. **Observability**:
   - Request latency, throughput, token usage
   - GPU utilization and memory
   - Cost per request tracking
   - Model performance metrics

5. **Production Features**:
   - Health checks and readiness probes
   - Graceful shutdown and rolling updates
   - Circuit breakers and retries
   - Automated failover

### Success Metrics

- Deploy 3 LLM models with <15 minute deployment time
- P95 latency <300ms (vs 800ms baseline)
- Handle 5,000 concurrent requests
- Auto-scaling: Scale from 1 to 10 GPU pods in <3 minutes
- Total cost: <$50,000/month (70% reduction from $150,000)
- Uptime: 99.9% availability
- Cache hit rate: >40% (reduces GPU compute)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│         Production LLM Serving Platform (Kubernetes)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Gateway Layer                      │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │  │   Ingress   │─▶│   FastAPI   │─▶│   Redis     │     │  │
│  │  │   (NGINX)   │  │   Gateway   │  │   Cache     │     │  │
│  │  │  - TLS      │  │  - Auth     │  │  - 40% hits │     │  │
│  │  │  - LB       │  │  - Routing  │  └─────────────┘     │  │
│  │  └─────────────┘  │  - Rate Lim │                       │  │
│  │                    └─────────────┘                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            │ Route based on task               │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Model Serving Layer                     │  │
│  │                                                           │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ │  │
│  │  │  Llama 2 7B     │  │  Mistral 7B     │  │CodeLlama│  │  │
│  │  │  (vLLM)         │  │  (vLLM)         │  │  7B     │  │  │
│  │  │  - Chat use case│  │  - Fast infer   │  │ (vLLM)  │  │  │
│  │  │  - 2 replicas   │  │  - 3 replicas   │  │ - Code  │  │  │
│  │  │  - A10G GPU     │  │  - T4 GPU       │  │ - 1 rep │  │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────┘ │  │
│  │         │                      │                 │        │  │
│  │         └──────────────────────┴─────────────────┘        │  │
│  │                            │                               │  │
│  └────────────────────────────┼───────────────────────────────┘  │
│                               │                                  │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Monitoring & Observability                   │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │  │ Prometheus  │  │   Grafana   │  │    Loki     │     │  │
│  │  │ - Latency   │  │ - Dashboards│  │ - Logs      │     │  │
│  │  │ - GPU util  │  │ - Alerts    │  │ - Traces    │     │  │
│  │  │ - Costs     │  └─────────────┘  └─────────────┘     │  │
│  │  └─────────────┘                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Auto-Scaling Layer                      │  │
│  │                                                           │  │
│  │  - HPA: Scale based on GPU utilization, queue depth     │  │
│  │  - Cluster Autoscaler: Add GPU nodes when needed        │  │
│  │  - Cost optimizer: Use spot instances, scale to zero    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: vLLM Model Deployment (8-10 hours)

Deploy LLMs using vLLM inference engine with optimal configurations.

#### 1.1 vLLM Deployment Configuration

Create `kubernetes/llm-deployments/llama2-deployment.yaml`:

```yaml
# TODO: Deploy Llama 2 7B with vLLM

apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama2-7b-vllm
  namespace: llm-serving
  labels:
    app: llama2-7b
    model: llama2
    version: 7b
spec:
  replicas: 2  # Start with 2 for HA
  selector:
    matchLabels:
      app: llama2-7b
  template:
    metadata:
      labels:
        app: llama2-7b
        model: llama2
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # Node selector: Only schedule on GPU nodes
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a10g
        workload: llm-serving

      # Tolerations for GPU taints
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.2.7
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
            - --model
            - meta-llama/Llama-2-7b-chat-hf
            - --host
            - "0.0.0.0"
            - --port
            - "8000"
            # vLLM optimization flags
            - --tensor-parallel-size
            - "1"  # Single GPU per pod
            - --dtype
            - auto  # Use FP16 automatically
            - --max-model-len
            - "4096"  # Context length
            - --gpu-memory-utilization
            - "0.9"  # Use 90% of GPU memory
            - --max-num-batched-tokens
            - "8192"  # Continuous batching
            - --max-num-seqs
            - "256"  # Max concurrent sequences
            # Performance tuning
            - --disable-log-requests  # Reduce overhead in production
            - --enable-prefix-caching  # Cache common prefixes
            - --trust-remote-code  # Required for some models

          ports:
            - containerPort: 8000
              name: http
              protocol: TCP

          env:
            # Hugging Face token for model downloads
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: huggingface-token
                  key: token
            # CUDA optimizations
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            # vLLM configuration
            - name: VLLM_WORKER_MULTIPROC_METHOD
              value: spawn

          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"  # Request 1 GPU
            limits:
              memory: "24Gi"
              cpu: "8"
              nvidia.com/gpu: "1"

          # Readiness probe: Check if model is loaded and ready
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120  # Model loading takes time
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          # Liveness probe: Check if server is responsive
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3

          # Volume mounts for model caching
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm  # Shared memory for CUDA

      # Volumes
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: huggingface-model-cache
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi  # Shared memory for GPU communication

      # Priority for GPU pods
      priorityClassName: high-priority-llm

---
# Service for Llama 2
apiVersion: v1
kind: Service
metadata:
  name: llama2-7b-service
  namespace: llm-serving
spec:
  selector:
    app: llama2-7b
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  type: ClusterIP  # Internal only, accessed via API gateway
```

Create `kubernetes/llm-deployments/mistral-deployment.yaml`:

```yaml
# TODO: Deploy Mistral 7B (optimized for speed)

apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-7b-vllm
  namespace: llm-serving
spec:
  replicas: 3  # More replicas for high-throughput model
  selector:
    matchLabels:
      app: mistral-7b
  template:
    metadata:
      labels:
        app: mistral-7b
        model: mistral
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4  # Cheaper T4 GPUs

      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.2.7
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
            - --model
            - mistralai/Mistral-7B-Instruct-v0.2
            - --host
            - "0.0.0.0"
            - --port
            - "8000"
            # Mistral optimizations
            - --tensor-parallel-size
            - "1"
            - --dtype
            - bfloat16  # Better for Mistral
            - --max-model-len
            - "8192"  # Mistral supports longer context
            - --gpu-memory-utilization
            - "0.85"
            - --quantization
            - awq  # 4-bit quantization for faster inference
            # Speed optimizations
            - --enable-prefix-caching
            - --disable-log-requests

          # Similar ports, env, resources, probes as Llama 2
          # ... (repeat structure)

---
apiVersion: v1
kind: Service
metadata:
  name: mistral-7b-service
  namespace: llm-serving
spec:
  selector:
    app: mistral-7b
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

#### 1.2 Model Download and Caching

Create `scripts/download-models.sh`:

```bash
#!/bin/bash
# TODO: Pre-download models to persistent volume for faster pod startup

set -e

MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "codellama/CodeLlama-7b-Instruct-hf"
)

HF_TOKEN="${HF_TOKEN:-}"
CACHE_DIR="${CACHE_DIR:-/models}"

echo "Downloading models to ${CACHE_DIR}..."

for MODEL in "${MODELS[@]}"; do
    echo "Downloading ${MODEL}..."

    # TODO: Use huggingface-cli to download models
    huggingface-cli download "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --token "${HF_TOKEN}" \
        --local-dir "${CACHE_DIR}/${MODEL}"

    echo "✓ Downloaded ${MODEL}"
done

echo "All models downloaded successfully!"
```

### Part 2: API Gateway and Request Routing (8-10 hours)

Build API gateway with routing, caching, and authentication.

#### 2.1 FastAPI Gateway

Create `src/api_gateway/main.py`:

```python
"""
TODO: API Gateway for LLM serving platform.

Features:
- OpenAI-compatible API
- Model routing based on task
- Response caching
- Rate limiting
- Authentication
- Request logging and metrics
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import hashlib
import redis
import time
from prometheus_client import Counter, Histogram, Gauge
import asyncio

app = FastAPI(title="LLM Serving Platform API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching
redis_client = redis.Redis(
    host="redis-cache",
    port=6379,
    db=0,
    decode_responses=True
)

# Prometheus metrics
request_count = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'endpoint', 'status']
)

request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model', 'endpoint']
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

cache_hits = Counter(
    'llm_cache_hits_total',
    'Cache hits',
    ['model']
)

# Model routing configuration
MODEL_ENDPOINTS = {
    'llama2': 'http://llama2-7b-service:8000',
    'mistral': 'http://mistral-7b-service:8000',
    'codellama': 'http://codellama-7b-service:8000'
}

# Task-to-model routing
TASK_MODEL_MAPPING = {
    'chat': 'llama2',
    'summarization': 'mistral',
    'code': 'codellama',
    'default': 'mistral'  # Fast default
}


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "gpt-3.5-turbo"  # For compatibility
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    task: Optional[str] = None  # Custom field for routing


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict[str, int]


def get_cache_key(messages: List[Dict], model: str, params: Dict) -> str:
    """
    TODO: Generate cache key from request parameters.

    Cache key includes:
    - Messages (user inputs)
    - Model name
    - Temperature, max_tokens (affects output)
    """
    cache_str = f"{model}:{messages}:{params}"
    return hashlib.sha256(cache_str.encode()).hexdigest()


def select_model(task: Optional[str], requested_model: str) -> str:
    """
    TODO: Select appropriate model based on task.

    Routing logic:
    - If task specified: Use task-to-model mapping
    - If specific model requested: Honor if available
    - Default: Use fastest model (mistral)
    """
    if task and task in TASK_MODEL_MAPPING:
        return TASK_MODEL_MAPPING[task]

    # Map OpenAI model names to our models
    if 'gpt-3.5' in requested_model.lower():
        return 'mistral'  # Fast, similar capability
    elif 'gpt-4' in requested_model.lower():
        return 'llama2'  # Higher quality
    elif 'code' in requested_model.lower():
        return 'codellama'

    return TASK_MODEL_MAPPING['default']


async def check_rate_limit(request: Request) -> bool:
    """
    TODO: Rate limiting using Redis.

    Implement token bucket algorithm:
    - Each user gets 100 requests/minute
    - Store in Redis with TTL

    Key: rate_limit:{user_id}:{minute}
    Value: request count

    If count > limit: raise HTTPException(429)
    """
    pass


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    rate_limit_ok: bool = Depends(check_rate_limit)
):
    """
    TODO: OpenAI-compatible chat completions endpoint.

    Flow:
    1. Select model based on task/request
    2. Check cache for identical request
    3. If cached: Return cached response (fast!)
    4. If not cached: Forward to model service
    5. Cache response for future requests
    6. Return response
    """
    start_time = time.time()

    # Select model
    model = select_model(request.task, request.model)
    model_endpoint = MODEL_ENDPOINTS.get(model)

    if not model_endpoint:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    # Check cache
    cache_key = get_cache_key(
        request.messages,
        model,
        {
            'temperature': request.temperature,
            'max_tokens': request.max_tokens
        }
    )

    cached_response = redis_client.get(f"cache:{cache_key}")
    if cached_response:
        cache_hits.labels(model=model).inc()
        request_count.labels(model=model, endpoint='chat', status='cached').inc()
        return ChatCompletionResponse.parse_raw(cached_response)

    # Forward to model service
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{model_endpoint}/v1/chat/completions",
                json=request.dict(exclude={'task'})
            )
            response.raise_for_status()

            result = response.json()

            # Cache response (TTL: 1 hour)
            redis_client.setex(
                f"cache:{cache_key}",
                3600,
                ChatCompletionResponse(**result).json()
            )

            # Metrics
            request_count.labels(
                model=model,
                endpoint='chat',
                status='success'
            ).inc()

            tokens_generated.labels(model=model).inc(
                result.get('usage', {}).get('total_tokens', 0)
            )

            duration = time.time() - start_time
            request_duration.labels(model=model, endpoint='chat').observe(duration)

            return ChatCompletionResponse(**result)

    except httpx.HTTPError as e:
        request_count.labels(
            model=model,
            endpoint='chat',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=f"Model service error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    TODO: Health check endpoint.

    Check:
    - Redis connectivity
    - At least one model service available
    """
    health = {
        "status": "healthy",
        "redis": False,
        "models": {}
    }

    # Check Redis
    try:
        redis_client.ping()
        health["redis"] = True
    except Exception:
        health["status"] = "unhealthy"

    # Check model services
    async with httpx.AsyncClient(timeout=5.0) as client:
        for model_name, endpoint in MODEL_ENDPOINTS.items():
            try:
                response = await client.get(f"{endpoint}/health")
                health["models"][model_name] = response.status_code == 200
            except Exception:
                health["models"][model_name] = False

    if not any(health["models"].values()):
        health["status"] = "unhealthy"

    return health


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

Create `kubernetes/api-gateway/deployment.yaml`:

```yaml
# TODO: Deploy API Gateway

apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api-gateway
  namespace: llm-serving
spec:
  replicas: 3  # HA
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
        - name: gateway
          image: llm-platform/api-gateway:latest
          ports:
            - containerPort: 8080
          env:
            - name: REDIS_HOST
              value: redis-cache
            - name: REDIS_PORT
              value: "6379"
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20

---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: llm-serving
spec:
  selector:
    app: api-gateway
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer  # Expose externally
```

### Part 3: Auto-Scaling and Cost Optimization (8-10 hours)

Implement GPU-aware auto-scaling with cost controls.

#### 3.1 Horizontal Pod Autoscaler

Create `kubernetes/autoscaling/hpa.yaml`:

```yaml
# TODO: HPA for LLM pods based on custom metrics

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama2-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama2-7b-vllm
  minReplicas: 1
  maxReplicas: 10
  metrics:
    # Scale based on GPU utilization
    - type: Pods
      pods:
        metric:
          name: nvidia_gpu_utilization_percent
        target:
          type: AverageValue
          averageValue: "70"  # Target 70% GPU utilization

    # Scale based on request queue depth
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting
        target:
          type: AverageValue
          averageValue: "5"  # Max 5 requests waiting per pod

    # Scale based on CPU (secondary metric)
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before scaling down
      policies:
        - type: Percent
          value: 50  # Scale down max 50% at a time
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
        - type: Percent
          value: 100  # Double capacity if needed
          periodSeconds: 15
```

#### 3.2 Cluster Autoscaler Configuration

Create `kubernetes/autoscaling/cluster-autoscaler-config.yaml`:

```yaml
# TODO: Configure cluster autoscaler for GPU nodes

apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |-
    # Priority configuration for node groups
    # Higher number = higher priority

    10:  # Lowest priority: Spot instances (cheapest)
      - .*-spot-t4.*
    20:  # Medium priority: On-demand T4 (balanced)
      - .*-ondemand-t4.*
    30:  # High priority: On-demand A10G (premium)
      - .*-ondemand-a10g.*

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cluster-autoscaler
  namespace: kube-system

---
# TODO: ClusterRole for autoscaler permissions
# ... (standard cluster autoscaler RBAC)

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
        - name: cluster-autoscaler
          image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.28.0
          command:
            - ./cluster-autoscaler
            - --v=4
            - --cloud-provider=gce
            - --nodes=0:10:llm-gpu-spot-t4  # Spot T4: 0-10 nodes
            - --nodes=0:5:llm-gpu-ondemand-a10g  # On-demand A10G: 0-5 nodes
            # Cost optimization
            - --scale-down-enabled=true
            - --scale-down-delay-after-add=10m
            - --scale-down-unneeded-time=5m
            - --max-node-provision-time=15m
            # Balance efficiency
            - --balance-similar-node-groups=true
            - --expander=priority  # Use priority expander config
```

#### 3.3 Cost Optimization with Spot Instances

Create `src/cost_optimizer/spot_manager.py`:

```python
"""
TODO: Manage spot instance lifecycle for cost savings.

Spot instances are 70% cheaper but can be preempted.
Strategy:
- Use spot for batch/async workloads
- Use on-demand for critical real-time requests
- Implement graceful handling of spot terminations
"""

import asyncio
import httpx
from kubernetes import client, config

class SpotInstanceManager:
    """
    Manage spot instance lifecycle.

    Handles:
    - Spot termination warnings (2-minute notice)
    - Graceful pod eviction
    - Automatic failover to on-demand
    """

    def __init__(self):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()

    async def watch_spot_termination_notices(self):
        """
        TODO: Watch for spot instance termination notices.

        GCP/AWS provide 2-minute warning before termination.

        AWS: http://169.254.169.254/latest/meta-data/spot/termination-time
        GCP: http://metadata.google.internal/computeMetadata/v1/instance/preempted

        When termination detected:
        1. Mark node as unschedulable (cordon)
        2. Drain pods gracefully
        3. Pods reschedule to healthy nodes
        """
        while True:
            try:
                # Check termination notice from metadata service
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                        headers={"Metadata-Flavor": "Google"},
                        timeout=1.0
                    )

                    if response.text == "TRUE":
                        await self.handle_spot_termination()

            except Exception as e:
                # Metadata service unavailable (not on spot) or other error
                pass

            await asyncio.sleep(5)  # Check every 5 seconds

    async def handle_spot_termination(self):
        """
        TODO: Handle spot instance termination.

        1. Get node name
        2. Cordon node (prevent new pods)
        3. Drain pods (graceful shutdown)
        4. Alert monitoring system
        """
        print("Spot termination notice received! Initiating graceful shutdown...")

        # TODO: Implement graceful shutdown logic
        # self.v1.patch_node(node_name, {"spec": {"unschedulable": True}})
        # ... drain pods ...
```

### Part 4: Monitoring and Observability (6-8 hours)

Implement comprehensive monitoring for LLM platform.

#### 4.1 Prometheus Metrics

Create `kubernetes/monitoring/servicemonitor.yaml`:

```yaml
# TODO: ServiceMonitor for Prometheus to scrape LLM metrics

apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-vllm-metrics
  namespace: llm-serving
spec:
  selector:
    matchLabels:
      app: llama2-7b
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

#### 4.2 Grafana Dashboard

Create `dashboards/llm-platform-dashboard.json`:

```json
{
  "dashboard": {
    "title": "LLM Serving Platform",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [{
          "expr": "rate(llm_requests_total[5m])"
        }]
      },
      {
        "title": "P95 Latency by Model",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "nvidia_gpu_utilization_percent"
        }]
      },
      {
        "title": "Cost per 1M Tokens",
        "targets": [{
          "expr": "TODO: Calculate based on GPU hours and tokens generated"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(llm_cache_hits_total[5m]) / rate(llm_requests_total[5m])"
        }]
      }
    ]
  }
}
```

### Part 5: Production Deployment and Testing (6-8 hours)

Deploy complete platform and validate performance.

#### 5.1 Deployment Script

Create `scripts/deploy-platform.sh`:

```bash
#!/bin/bash
# TODO: Deploy complete LLM serving platform

set -e

echo "Deploying LLM Serving Platform..."

# Create namespace
kubectl create namespace llm-serving --dry-run=client -o yaml | kubectl apply -f -

# Deploy Redis cache
kubectl apply -f kubernetes/redis/

# Deploy model services
kubectl apply -f kubernetes/llm-deployments/

# Wait for models to be ready
echo "Waiting for models to load (this may take 5-10 minutes)..."
kubectl wait --for=condition=ready pod -l app=llama2-7b -n llm-serving --timeout=600s

# Deploy API gateway
kubectl apply -f kubernetes/api-gateway/

# Deploy monitoring
kubectl apply -f kubernetes/monitoring/

# Deploy autoscaling
kubectl apply -f kubernetes/autoscaling/

echo "✓ Deployment complete!"
echo "API Gateway: $(kubectl get svc api-gateway-service -n llm-serving -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
```

#### 5.2 Load Testing

Create `tests/load_test.py`:

```python
"""
TODO: Load test LLM platform.

Test scenarios:
1. Sustained load (1000 req/min for 1 hour)
2. Spike load (0 → 5000 req/min)
3. Auto-scaling validation
4. Cache effectiveness
"""

import asyncio
import httpx
from typing import List
import time

async def send_request(client: httpx.AsyncClient, prompt: str):
    """Send single chat completion request."""
    start = time.time()

    response = await client.post(
        "http://API_GATEWAY_IP/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
    )

    latency = time.time() - start
    return {
        "latency": latency,
        "status": response.status_code,
        "tokens": response.json().get("usage", {}).get("total_tokens", 0)
    }

async def load_test(target_rps: int, duration_seconds: int):
    """
    TODO: Run load test.

    target_rps: Requests per second
    duration_seconds: Test duration
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        interval = 1.0 / target_rps

        for _ in range(target_rps * duration_seconds):
            task = send_request(client, "What is machine learning?")
            tasks.append(task)
            await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)

    # Analyze results
    latencies = [r["latency"] for r in results]
    print(f"P50 latency: {sorted(latencies)[len(latencies)//2]:.2f}s")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    print(f"P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")

if __name__ == "__main__":
    asyncio.run(load_test(target_rps=100, duration_seconds=60))
```

## Acceptance Criteria

### Functional Requirements

- [ ] 3 LLM models deployed (Llama 2, Mistral, CodeLlama)
- [ ] OpenAI-compatible API working
- [ ] Request routing to appropriate models
- [ ] Response caching with Redis
- [ ] Rate limiting implemented
- [ ] Auto-scaling configured (HPA + Cluster Autoscaler)

### Performance Requirements

- [ ] P95 latency <300ms (vs 800ms baseline)
- [ ] Handle 5,000 concurrent requests
- [ ] Cache hit rate >40%
- [ ] Auto-scaling: 1 → 10 pods in <3 minutes
- [ ] GPU utilization >70%

### Cost Requirements

- [ ] Total monthly cost <$50,000 (70% reduction)
- [ ] Spot instance usage >60% of compute
- [ ] Cost per 1M tokens <$5

### Operational Requirements

- [ ] 99.9% uptime (measured over 1 week)
- [ ] Zero downtime deployments
- [ ] Graceful spot instance handling
- [ ] Comprehensive monitoring dashboards

## Testing Strategy

```bash
# Unit tests
pytest tests/

# Integration tests
./scripts/test-api.sh

# Load tests
python tests/load_test.py

# Chaos testing
kubectl delete pod -l app=llama2-7b --random  # Random pod deletion
```

## Deliverables

1. **Kubernetes Manifests** (all deployments, services, HPA)
2. **API Gateway** (FastAPI application)
3. **Monitoring** (Grafana dashboards, Prometheus rules)
4. **Documentation** (architecture, deployment guide, runbook)
5. **Tests** (unit tests, load tests, chaos tests)

## Bonus Challenges

1. **Multi-Region Deployment** (+8 hours): Deploy across regions for <100ms global latency
2. **A/B Testing Framework** (+6 hours): Test new models against production
3. **Cost Attribution** (+4 hours): Track costs per user/team

## Resources

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/)
- [HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

## Submission

```bash
git add .
git commit -m "Complete Exercise 01: Production LLM Serving"
git push origin exercise-01-production-llm-serving
```

---

**Estimated Time Breakdown**:
- Part 1 (vLLM Deployment): 8-10 hours
- Part 2 (API Gateway): 8-10 hours
- Part 3 (Auto-Scaling): 8-10 hours
- Part 4 (Monitoring): 6-8 hours
- Part 5 (Deployment & Testing): 6-8 hours
- **Total**: 36-44 hours
