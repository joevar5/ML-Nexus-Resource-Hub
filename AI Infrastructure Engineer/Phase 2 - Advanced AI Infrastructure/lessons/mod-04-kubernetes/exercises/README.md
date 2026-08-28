# Module 04: Kubernetes - Exercises

Master Kubernetes for ML workloads with these hands-on exercises.

## Exercise 1: Deploy Your First ML Model

**Difficulty:** Beginner
**Duration:** 45 minutes

### Objective
Deploy a simple ML model inference service to Kubernetes.

### Tasks
1. Create Deployment YAML for model server
2. Expose via Service
3. Verify pod is running
4. Test inference endpoint
5. Scale deployment to 3 replicas

### Starter YAML
```yaml
# exercises/01-model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  # TODO: Set replicas
  # TODO: Add labels and selectors
  template:
    spec:
      containers:
      - name: model-server
        # TODO: Set image
        # TODO: Add resource requests/limits
        # TODO: Add ports
        # TODO: Add health checks
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  # TODO: Configure service type and ports
  # TODO: Add selector
```

### Commands to Complete
```bash
# Apply deployment
kubectl apply -f exercises/01-model-deployment.yaml

# TODO: Verify deployment
# kubectl get ...

# TODO: Test service
# kubectl port-forward ...
# curl ...

# TODO: Scale deployment
# kubectl scale ...
```

### Success Criteria
- [ ] Deployment created successfully
- [ ] All pods running
- [ ] Service accessible
- [ ] Health checks passing
- [ ] Can scale up and down

---

## Exercise 2: ConfigMaps and Secrets

**Difficulty:** Beginner
**Duration:** 30 minutes

### Objective
Manage configuration and secrets for ML applications.

### Tasks
1. Create ConfigMap for model config
2. Create Secret for API keys
3. Mount ConfigMap as volume
4. Inject Secret as environment variable
5. Verify application uses config

### Starter YAML
```yaml
# exercises/02-config-secrets.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  # TODO: Add model configuration
  model_name: "..."
  batch_size: "..."

---
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
type: Opaque
stringData:
  # TODO: Add API credentials (base64 encoded)
  api_key: ""
```

### Success Criteria
- [ ] ConfigMap created and mounted
- [ ] Secret created and injected
- [ ] Application reads config correctly
- [ ] Secrets not visible in logs
- [ ] Can update config without pod restart (for volume mounts)

---

## Exercise 3: Resource Management

**Difficulty:** Intermediate
**Duration:** 45 minutes

### Objective
Properly configure resource requests and limits for ML workloads.

### Tasks
1. Set CPU and memory requests
2. Set CPU and memory limits
3. Observe pod scheduling with resources
4. Trigger OOMKill with low limits
5. Configure ResourceQuota for namespace

### Resource Configuration
```yaml
# exercises/03-resources.yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:latest
    resources:
      requests:
        # TODO: Set appropriate requests
        memory: ""
        cpu: ""
      limits:
        # TODO: Set appropriate limits
        memory: ""
        cpu: ""
```

### Tasks
```bash
# TODO: Monitor resource usage
# kubectl top pod ml-training-pod

# TODO: Create ResourceQuota
# Limit namespace to 4 CPU and 8Gi memory
```

### Success Criteria
- [ ] Resources correctly requested
- [ ] Pod scheduled successfully
- [ ] Resource limits enforced
- [ ] ResourceQuota prevents over-allocation
- [ ] Can observe resource usage with `kubectl top`

---

## Exercise 4: GPU Scheduling

**Difficulty:** Intermediate
**Duration:** 60 minutes

### Objective
Schedule ML training jobs on GPU nodes.

### Tasks
1. Label GPU nodes
2. Create pod requesting GPU
3. Verify GPU allocated
4. Run sample GPU workload
5. Set up node affinity for GPU pods

### GPU Pod YAML
```yaml
# exercises/04-gpu-scheduling.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-job
spec:
  containers:
  - name: pytorch-trainer
    image: pytorch/pytorch:latest-cuda
    resources:
      limits:
        # TODO: Request GPU
        nvidia.com/gpu: 1
    # TODO: Add node selector for GPU nodes
    # TODO: Add GPU-specific environment variables
```

### Verification
```bash
# TODO: Verify GPU allocation
# kubectl exec gpu-training-job -- nvidia-smi

# TODO: Check node has GPU
# kubectl describe node <node-name> | grep nvidia.com/gpu
```

### Success Criteria
- [ ] GPU node properly labeled
- [ ] Pod scheduled on GPU node
- [ ] GPU visible in container
- [ ] Can run CUDA code
- [ ] Resource limits respected

---

## Exercise 5: Horizontal Pod Autoscaling (HPA)

**Difficulty:** Intermediate
**Duration:** 60 minutes

### Objective
Implement autoscaling for ML inference service.

### Tasks
1. Deploy metrics-server
2. Create HPA targeting CPU usage
3. Generate load to trigger scaling
4. Observe pod scaling behavior
5. Configure custom metrics (optional)

### HPA Configuration
```yaml
# exercises/05-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    # TODO: Reference your deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # TODO: Add custom metrics (requests per second, etc.)
```

### Load Testing
```bash
# TODO: Generate load
# kubectl run -it load-generator --rm --image=busybox -- /bin/sh
# while true; do wget -q -O- http://ml-model-service; done
```

### Success Criteria
- [ ] HPA created and active
- [ ] Pods scale up under load
- [ ] Pods scale down when idle
- [ ] Scaling respects min/max replicas
- [ ] Custom metrics working (if implemented)

---

## Exercise 6: StatefulSet for ML Experiment Tracking

**Difficulty:** Advanced
**Duration:** 90 minutes

### Objective
Deploy MLflow or similar using StatefulSet with persistent storage.

### Tasks
1. Create PersistentVolumeClaim
2. Deploy StatefulSet for MLflow
3. Configure pod affinity/anti-affinity
4. Set up headless service
5. Test data persistence across pod restarts

### StatefulSet YAML
```yaml
# exercises/06-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow-server
spec:
  serviceName: mlflow
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    spec:
      containers:
      - name: mlflow
        # TODO: Configure MLflow container
        volumeMounts:
        - name: data
          mountPath: /mlflow
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Success Criteria
- [ ] StatefulSet running
- [ ] PVC bound and mounted
- [ ] Data persists across pod restarts
- [ ] Stable network identity
- [ ] Can scale StatefulSet

---

## Exercise 7: Jobs and CronJobs for Training

**Difficulty:** Intermediate
**Duration:** 45 minutes

### Objective
Run batch ML training jobs using Kubernetes Jobs and scheduled retraining with CronJobs.

### Tasks
1. Create Job for one-time training
2. Create CronJob for daily retraining
3. Configure parallelism
4. Set up TTL for job cleanup
5. Access job logs

### Job Configuration
```yaml
# exercises/07-jobs.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
spec:
  # TODO: Configure job
  template:
    spec:
      containers:
      - name: trainer
        # TODO: Configure training container
        # TODO: Mount data and output volumes
      restartPolicy: Never
  backoffLimit: 3
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-retraining
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      # TODO: Configure job template
```

### Success Criteria
- [ ] Job completes successfully
- [ ] CronJob scheduled correctly
- [ ] Failed jobs retry as configured
- [ ] Old jobs cleaned up
- [ ] Can access job logs

---

## Exercise 8: Ingress for ML APIs

**Difficulty:** Intermediate
**Duration:** 45 minutes

### Objective
Expose ML API externally using Ingress.

### Tasks
1. Install Ingress controller (nginx)
2. Create Ingress resource
3. Configure path-based routing
4. Add TLS (optional)
5. Test external access

### Ingress Configuration
```yaml
# exercises/08-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  annotations:
    # TODO: Add ingress annotations
spec:
  rules:
  - host: ml-api.example.com
    http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: ml-model-service
            port:
              number: 80
  # TODO: Configure TLS
```

### Success Criteria
- [ ] Ingress controller running
- [ ] Ingress resource created
- [ ] Can access API externally
- [ ] Path routing works
- [ ] TLS configured (if attempted)

---

## Solutions

Solutions are available in `solutions/` directory. Attempt exercises independently first!

## Tips

1. **Use kubectl explain** - `kubectl explain pod.spec.containers`
2. **Dry run** - Test YAML with `--dry-run=client -o yaml`
3. **Watch resources** - Use `kubectl get pods --watch`
4. **Debug pods** - Use `kubectl describe` and `kubectl logs`
5. **Port forwarding** - Test services locally before exposing

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](../../../resources/cheat-sheets/kubernetes-cheat-sheet.md)
- [Kubernetes Patterns](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/)

---

**Need help?** Ask in [GitHub Discussions](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning/discussions)
