# Troubleshooting Guide - ML Model Serving

## Common Issues and Solutions

### 1. Docker Build Issues

#### Issue: "Docker build fails with 'no space left on device'"

**Symptoms:**
```
ERROR: failed to solve: write /var/lib/docker/...: no space left on device
```

**Solution:**
```bash
# Clean up Docker
docker system prune -a --volumes
docker builder prune

# Check disk space
df -h
```

#### Issue: "Image build is very slow"

**Symptoms:** Build takes 10+ minutes

**Solutions:**
1. Order Dockerfile layers by change frequency
2. Use .dockerignore to exclude unnecessary files
3. Use multi-stage builds
4. Enable BuildKit:
   ```bash
   export DOCKER_BUILDKIT=1
   docker build -t ml-api:v1 .
   ```

---

### 2. Container Runtime Issues

#### Issue: "Container exits immediately"

**Symptoms:**
```bash
docker run ml-api:v1
# Container exits with code 1
```

**Debugging:**
```bash
# Check logs
docker logs <container-id>

# Run interactively
docker run -it ml-api:v1 /bin/bash

# Check if model file exists
docker run ml-api:v1 ls -la /app/models/
```

**Common causes:**
- Model file not found
- Permission issues
- Missing dependencies

#### Issue: "Model not loading"

**Symptoms:**
```
ERROR: Failed to load model from /app/models/resnet18.pth
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
```bash
# Ensure model file exists
ls -la models/

# Check volume mount
docker run -v $(pwd)/models:/app/models ml-api:v1 ls /app/models/

# Download model if missing
# TODO: Add model download script
```

---

### 3. Kubernetes Deployment Issues

#### Issue: "Pod stuck in Pending state"

**Symptoms:**
```bash
kubectl get pods -n ml-serving
NAME                      READY   STATUS    RESTARTS   AGE
ml-api-xxxxxxxxx-xxxxx   0/1     Pending   0          5m
```

**Debugging:**
```bash
kubectl describe pod ml-api-xxxxxxxxx-xxxxx -n ml-serving

# Common reasons:
# 1. Insufficient resources
# 2. Image pull issues
# 3. Volume mount issues
```

**Solutions:**
```bash
# Check node resources
kubectl top nodes

# Check events
kubectl get events -n ml-serving --sort-by='.lastTimestamp'

# Reduce resource requests in deployment.yaml if needed
```

#### Issue: "ImagePullBackOff error"

**Symptoms:**
```
Events:
  Warning  Failed     5m    kubelet  Failed to pull image "ml-api:v1"
  Warning  BackOff    2m    kubelet  Back-off pulling image "ml-api:v1"
```

**Solutions:**
```bash
# For local Kubernetes (minikube/kind):
minikube image load ml-api:v1
# or
kind load docker-image ml-api:v1

# For cloud Kubernetes:
# Push image to container registry (GCR, ECR, ACR)
docker push gcr.io/project/ml-api:v1

# Update deployment.yaml with full image path
image: gcr.io/project/ml-api:v1
```

#### Issue: "CrashLoopBackOff"

**Symptoms:**
```
NAME                      READY   STATUS             RESTARTS   AGE
ml-api-xxxxxxxxx-xxxxx   0/1     CrashLoopBackOff   5          5m
```

**Debugging:**
```bash
# Check logs
kubectl logs ml-api-xxxxxxxxx-xxxxx -n ml-serving

# Check previous container logs
kubectl logs ml-api-xxxxxxxxx-xxxxx -n ml-serving --previous

# Common causes:
# - Application crashes on startup
# - Health check failing
# - Out of memory (OOM)
```

**Solutions:**
```bash
# If OOM, increase memory limit
resources:
  limits:
    memory: "2Gi"  # Increase from 1Gi

# If health check failing, adjust timing
livenessProbe:
  initialDelaySeconds: 60  # Increase if model loading is slow
  periodSeconds: 30
  timeoutSeconds: 10
```

---

### 4. API Issues

#### Issue: "API returns 503 Service Unavailable"

**Symptoms:**
```bash
curl http://localhost:8000/health
{"detail": "Service unavailable"}
```

**Causes:**
- Model not loaded
- Application not started
- Health check failing

**Solutions:**
```bash
# Check logs
docker logs <container-id>
# or
kubectl logs deployment/ml-api -n ml-serving

# Verify model file exists
# Increase startup timeout
# Check resource availability (memory, CPU)
```

#### Issue: "Slow inference (>1 second)"

**Symptoms:**
- Prediction takes multiple seconds
- Timeouts

**Debugging:**
```python
# Add timing in code
import time
start = time.time()
result = model(input)
print(f"Inference time: {time.time() - start:.3f}s")
```

**Solutions:**
1. **Use GPU:** Add GPU support to container
2. **Model optimization:** Use ONNX, TensorRT, or quantization
3. **Batch inference:** Process multiple images together
4. **Model caching:** Ensure model is pre-loaded (not loading per request)
5. **Reduce image size:** Downsample input images

**CPU vs GPU comparison:**
```
CPU inference: 100-500ms
GPU inference: 5-50ms (10-100x faster)
```

#### Issue: "Out of Memory (OOM) during inference"

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 1  # or smaller

# 2. Use model with smaller memory footprint
model = models.resnet18(pretrained=True)  # instead of resnet152

# 3. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 4. Use CPU instead of GPU for small workloads
device = torch.device("cpu")
```

---

### 5. Monitoring Issues

#### Issue: "Prometheus not scraping metrics"

**Symptoms:**
- Metrics not appearing in Prometheus
- Targets show as "Down"

**Debugging:**
```bash
# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Visit: http://localhost:9090/targets

# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics
```

**Solutions:**
```bash
# 1. Verify Prometheus configuration
kubectl get configmap prometheus-config -n monitoring -o yaml

# 2. Check service discovery
kubectl get servicemonitor -n ml-serving

# 3. Verify network connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://ml-api-service.ml-serving.svc.cluster.local:80/metrics
```

#### Issue: "Grafana dashboard shows no data"

**Solutions:**
```bash
# 1. Check Prometheus datasource
# Grafana → Configuration → Data Sources → Prometheus

# 2. Verify metrics exist in Prometheus
# Query: predictions_total

# 3. Check time range (last 5 minutes)

# 4. Verify dashboard queries match metric names
```

---

### 6. Performance Issues

#### Issue: "High latency (p99 > 500ms)"

**Profiling:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your inference code
result = model(input)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)  # Top 10 functions
```

**Common bottlenecks:**
1. **Image preprocessing:** 20-50ms
   - Solution: Optimize resize/normalize operations
2. **Model inference:** 50-200ms (CPU), 5-50ms (GPU)
   - Solution: Use GPU, model optimization
3. **Postprocessing:** 5-10ms
   - Solution: Vectorize operations

#### Issue: "Low throughput (<10 req/sec)"

**Solutions:**
1. **Horizontal scaling:** Add more pods
2. **Async processing:** Use FastAPI async
3. **Request batching:** Process multiple requests together
4. **Connection pooling:** Reuse connections
5. **Caching:** Cache common predictions

---

### 7. Scaling Issues

#### Issue: "HPA not scaling up"

**Symptoms:**
- CPU/Memory above target but no new pods created

**Debugging:**
```bash
# Check HPA status
kubectl get hpa -n ml-serving
kubectl describe hpa ml-api-hpa -n ml-serving

# Check metrics-server
kubectl top pods -n ml-serving
```

**Solutions:**
```bash
# 1. Ensure metrics-server is installed
kubectl get deployment metrics-server -n kube-system

# 2. Verify resource requests are set
# HPA requires resource requests to be defined

# 3. Check HPA configuration
kubectl edit hpa ml-api-hpa -n ml-serving
```

---

## Debugging Commands Cheatsheet

### Docker

```bash
# View container logs
docker logs <container-id>
docker logs -f <container-id>  # Follow logs

# Execute command in running container
docker exec -it <container-id> /bin/bash

# Inspect container
docker inspect <container-id>

# Check resource usage
docker stats
```

### Kubernetes

```bash
# Pod status
kubectl get pods -n ml-serving
kubectl describe pod <pod-name> -n ml-serving

# Logs
kubectl logs <pod-name> -n ml-serving
kubectl logs -f deployment/ml-api -n ml-serving

# Execute command in pod
kubectl exec -it <pod-name> -n ml-serving -- /bin/bash

# Port forward
kubectl port-forward <pod-name> 8000:8000 -n ml-serving

# Resource usage
kubectl top pods -n ml-serving
kubectl top nodes

# Events
kubectl get events -n ml-serving --sort-by='.lastTimestamp'
```

---

## Getting Help

If you encounter issues not covered here:

1. Check application logs
2. Search GitHub Issues
3. Ask in Discussions
4. Create a new Issue with:
   - Error message
   - Steps to reproduce
   - Environment (Docker version, K8s version, etc.)
   - Logs

---

## Additional Resources

- [Docker Troubleshooting](https://docs.docker.com/config/containers/troubleshoot/)
- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug/)
- [FastAPI Debugging](https://fastapi.tiangolo.com/tutorial/debugging/)
