# Project 01: Basic Model Serving - Architecture Documentation

## Overview

This project implements a production-ready ML model serving system with REST API, containerization, Kubernetes deployment, and comprehensive monitoring.

## Architecture Diagram

```
                                                    ┌─────────────┐
                                                    │   Grafana   │
                                                    │  Dashboard  │
                                                    └──────┬──────┘
                                                           │
                                                           │ queries
┌──────────┐         ┌──────────────┐         ┌──────────▼──────┐
│  Client  │────────▶│ Load Balancer│────────▶│   Prometheus    │
│ (User)   │  HTTP   │  (Service)   │         │   Monitoring    │
└──────────┘  Req    └──────┬───────┘         └─────────────────┘
                             │                          │
                             │                          │ scrapes metrics
                             ▼                          │
                   ┌─────────────────┐                  │
                   │   FastAPI Pod   │◀─────────────────┘
                   │  (ML Serving)   │
                   │                 │
                   │  ┌───────────┐  │
                   │  │  ML Model │  │
                   │  │ (ResNet18)│  │
                   │  └───────────┘  │
                   │                 │
                   │  /predict       │
                   │  /health        │
                   │  /metrics       │
                   └─────────────────┘
                             │
                             │ autoscale
                             ▼
                   ┌─────────────────┐
                   │ Horizontal Pod  │
                   │   Autoscaler    │
                   └─────────────────┘
```

## Components

### 1. FastAPI Application (`src/api.py`)

**Purpose**: REST API for model inference

**Key Features:**
- `/health` - Health check endpoint
- `/metrics` - Prometheus metrics
- `/v1/predict` - Image classification endpoint
- Async request handling
- Input validation
- Error handling

**Technology Stack:**
- FastAPI (async web framework)
- Pydantic (request validation)
- Prometheus Client (metrics)

### 2. Model Wrapper (`src/model.py`)

**Purpose**: Load and manage ML model

**Responsibilities:**
- Model loading at startup
- Inference execution
- Device management (CPU/GPU)
- Preprocessing and postprocessing

**Model**: ResNet18 (pre-trained on ImageNet)

### 3. Configuration (`src/config.py`)

**Purpose**: Centralized configuration management

**Configuration Sources:**
1. Environment variables (.env)
2. Config files
3. Default values

**Key Settings:**
- Model path
- Device (CPU/GPU)
- API settings
- Logging level

### 4. Utilities (`src/utils.py`)

**Purpose**: Helper functions

**Functions:**
- Image download and validation
- Logging setup
- Metrics helpers
- Error handling utilities

## Data Flow

### Inference Request Flow

```
1. Client sends POST /v1/predict with image
   ↓
2. FastAPI receives request
   ↓
3. Pydantic validates input
   ↓
4. Image downloaded/loaded
   ↓
5. Image preprocessed (resize, normalize)
   ↓
6. Model inference (forward pass)
   ↓
7. Postprocessing (softmax, top-k)
   ↓
8. Response formatted and sent
   ↓
9. Metrics recorded (latency, count)
```

### Health Check Flow

```
1. Kubernetes sends GET /health
   ↓
2. FastAPI checks:
   - Model loaded
   - Service healthy
   ↓
3. Returns 200 OK or 503 Unavailable
```

### Metrics Collection Flow

```
1. Prometheus scrapes GET /metrics every 15s
   ↓
2. FastAPI returns metrics:
   - predictions_total (counter)
   - prediction_duration_seconds (histogram)
   - http_requests_total (counter)
   ↓
3. Prometheus stores time-series data
   ↓
4. Grafana visualizes metrics
```

## Deployment Architecture

### Local Development

```
docker-compose up
├── api:8000 (ML serving)
├── prometheus:9090 (metrics)
└── grafana:3000 (dashboards)
```

### Kubernetes Production

```
Kubernetes Cluster
├── Namespace: ml-serving
├── Deployment: ml-api
│   ├── Pod 1 (replica)
│   ├── Pod 2 (replica)
│   └── Pod N (auto-scaled)
├── Service: ml-api-service (LoadBalancer)
├── ConfigMap: ml-api-config
├── HPA: ml-api-hpa (auto-scaling rules)
├── Prometheus: monitoring namespace
└── Grafana: monitoring namespace
```

## Scaling Strategy

### Horizontal Pod Autoscaling (HPA)

**Metrics:**
- Target CPU: 70%
- Target Memory: 80%
- Custom: requests per second

**Configuration:**
```yaml
minReplicas: 1
maxReplicas: 5
scaleUp: add pod when CPU > 70% for 30s
scaleDown: remove pod when CPU < 50% for 5min
```

**Scaling Behavior:**
```
Low traffic (< 10 req/sec):  1 pod
Medium traffic (10-50 req/sec): 2-3 pods
High traffic (> 50 req/sec): 4-5 pods
```

## Performance Characteristics

### Latency

| Metric | Target | Measured |
|--------|--------|----------|
| p50 latency | < 50ms | TODO: Measure |
| p95 latency | < 100ms | TODO: Measure |
| p99 latency | < 200ms | TODO: Measure |

### Throughput

| Metric | Target | Measured |
|--------|--------|----------|
| Requests/sec (single pod) | 20-50 | TODO: Measure |
| Requests/sec (5 pods) | 100-250 | TODO: Measure |

### Resource Usage

| Resource | Per Pod | Notes |
|----------|---------|-------|
| CPU | 500m-1000m | 0.5-1 CPU core |
| Memory | 512Mi-1Gi | Model + runtime |
| Disk | 1Gi | Model weights |

## Monitoring and Observability

### Key Metrics

**Application Metrics:**
- `predictions_total` - Total predictions by status
- `prediction_duration_seconds` - Inference latency histogram
- `http_requests_total` - Total HTTP requests

**System Metrics:**
- CPU utilization
- Memory usage
- Network I/O
- Disk I/O

**Model Metrics:**
- Prediction distribution (class counts)
- Confidence scores (average)
- Error rate

### Logging

**Log Levels:**
- DEBUG: Detailed debugging info
- INFO: General information
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical failures

**Log Format (JSON):**
```json
{
  "timestamp": "2025-10-15T12:00:00Z",
  "level": "INFO",
  "message": "Prediction successful",
  "request_id": "uuid-1234",
  "inference_time_ms": 25,
  "class": "cat",
  "probability": 0.95
}
```

## Security Considerations

### Current Implementation

- ✅ Run as non-root user in container
- ✅ Input validation (file size, format)
- ✅ CORS enabled (configurable)
- ✅ Health checks for availability

### TODO: Production Hardening

- ⬜ API authentication (JWT, API keys)
- ⬜ Rate limiting per user/IP
- ⬜ HTTPS/TLS encryption
- ⬜ Network policies in Kubernetes
- ⬜ Secret management (for model encryption)
- ⬜ Input sanitization (prevent adversarial attacks)

## Disaster Recovery

### Failure Scenarios

**Pod Failure:**
- Kubernetes automatically restarts failed pods
- HPA maintains minimum replicas
- Service routes traffic to healthy pods

**Node Failure:**
- Pods rescheduled to healthy nodes
- PersistentVolumes (if used) reattached

**Model Corruption:**
- Keep multiple model versions
- Implement model validation on load
- Rollback to previous version

### Backup Strategy

**Model Artifacts:**
- Store in S3/GCS with versioning
- Automated backups every model update
- Retention: 30 days

**Configuration:**
- Git repository (version controlled)
- ConfigMaps backed up

## Cost Optimization

### Current Setup (Estimated Monthly Costs)

**Development:**
```
1 pod (24/7): ~$30/month
Prometheus: ~$10/month
Grafana: ~$10/month
Storage: ~$5/month
Total: ~$55/month
```

**Production (with auto-scaling):**
```
Average 3 pods: ~$90/month
Monitoring: ~$30/month
Storage: ~$10/month
Network: ~$20/month
Total: ~$150/month
```

### Optimization Tips

1. **Use spot instances** (60% savings)
2. **Scale to zero** during off-hours (save 50%)
3. **Model compression** (reduce memory, use smaller instances)
4. **Caching** (reduce redundant inference)
5. **Batch processing** (higher throughput per instance)

## Future Enhancements

### Phase 1 (Near-term)
- [ ] Add model versioning and A/B testing
- [ ] Implement request batching
- [ ] Add model warm-up on startup
- [ ] Distributed tracing with Jaeger

### Phase 2 (Mid-term)
- [ ] GPU support for faster inference
- [ ] Model compression (quantization, pruning)
- [ ] Multi-model serving
- [ ] Feature store integration

### Phase 3 (Long-term)
- [ ] Automated model retraining pipeline
- [ ] Drift detection and monitoring
- [ ] Federated learning support
- [ ] Edge deployment (TensorFlow Lite, ONNX)

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [ML Serving Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
