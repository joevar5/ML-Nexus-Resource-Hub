# Lesson 05: Kubernetes Networking and Ingress

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand the Kubernetes networking model
- Configure different Service types (ClusterIP, NodePort, LoadBalancer)
- Implement Ingress controllers for HTTP/HTTPS routing
- Configure TLS/SSL termination for ML APIs
- Set up Network Policies for security
- Route traffic to multiple ML models with Ingress rules
- Troubleshoot common networking issues

## Prerequisites
- Completed lessons 01-04 (Kubernetes fundamentals)
- Understanding of basic networking concepts (IP, DNS, ports)
- Familiarity with HTTP/HTTPS protocols
- Understanding of Services and Deployments

## Introduction

Networking is critical for ML infrastructure because:
- **Model serving:** ML APIs must be accessible to clients
- **Microservices:** ML pipelines often consist of multiple services
- **Security:** Models and data must be protected
- **Load balancing:** Distribute traffic across model replicas
- **TLS termination:** Secure communication with HTTPS

**Real-world examples:**
- **Uber:** Routes millions of requests to ML prediction services via K8s Ingress
- **Netflix:** Uses Ingress to serve recommendation APIs to 200M+ users
- **Spotify:** Load balances music recommendation requests across K8s clusters
- **Airbnb:** Exposes pricing and search ML models through Kubernetes services

## 1. Kubernetes Networking Model

### 1.1 Core Networking Principles

Kubernetes follows these networking requirements:

1. **Every Pod gets its own IP address**
   - No NAT between Pods on the same node
   - Pods can communicate directly with each other

2. **All Pods can communicate with all other Pods**
   - Without NAT, across all nodes
   - Flat network namespace

3. **All Nodes can communicate with all Pods**
   - Without NAT
   - Nodes can reach any Pod directly

**Network architecture:**

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                   │
│                                                         │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │   Node 1         │         │   Node 2         │    │
│  │  10.100.1.0/24   │         │  10.100.2.0/24   │    │
│  │                  │         │                  │    │
│  │  ┌────────────┐  │         │  ┌────────────┐  │    │
│  │  │ Pod A      │  │         │  │ Pod C      │  │    │
│  │  │ 10.244.1.2 │◄─┼─────────┼─►│ 10.244.2.3 │  │    │
│  │  └────────────┘  │         │  └────────────┘  │    │
│  │                  │         │                  │    │
│  │  ┌────────────┐  │         │  ┌────────────┐  │    │
│  │  │ Pod B      │  │         │  │ Pod D      │  │    │
│  │  │ 10.244.1.5 │  │         │  │ 10.244.2.8 │  │    │
│  │  └────────────┘  │         │  └────────────┘  │    │
│  │                  │         │                  │    │
│  └──────────────────┘         └──────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘

Pod A (10.244.1.2) can directly reach Pod C (10.244.2.3)
No NAT, no port mapping, direct IP communication
```

### 1.2 Container Network Interface (CNI)

CNI plugins implement the Kubernetes networking model:

**Popular CNI plugins:**

| Plugin     | Network Type | Performance | Features                    | Best For                |
|------------|--------------|-------------|-----------------------------|-------------------------|
| Calico     | Overlay/BGP  | High        | Network policies, encryption| Production, security    |
| Flannel    | Overlay      | Medium      | Simple, easy setup          | Development, small clusters |
| Cilium     | eBPF         | Very High   | Advanced policies, observability | High-performance ML     |
| Weave Net  | Overlay      | Medium      | Easy setup, encryption      | Quick setup             |
| AWS VPC CNI| Native       | Very High   | Native AWS networking       | AWS EKS                 |

**Install Calico (example):**

```bash
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

# Verify installation
kubectl get pods -n kube-system | grep calico
# calico-node-xxxxx                1/1     Running
# calico-kube-controllers-xxxxx    1/1     Running
```

### 1.3 DNS in Kubernetes

Every Service gets a DNS name automatically:

**DNS format:**
```
<service-name>.<namespace>.svc.cluster.local
```

**Example:**

```yaml
# Service in namespace "ml-serving"
apiVersion: v1
kind: Service
metadata:
  name: bert-inference
  namespace: ml-serving
spec:
  selector:
    app: bert
  ports:
  - port: 8080
```

**DNS names created:**
- `bert-inference.ml-serving.svc.cluster.local` (FQDN)
- `bert-inference.ml-serving` (cross-namespace)
- `bert-inference` (same namespace only)

**Test DNS resolution:**

```bash
kubectl run -it --rm debug --image=busybox --restart=Never -- sh

# Inside the pod:
nslookup bert-inference.ml-serving
# Output:
# Server:    10.96.0.10
# Address:   10.96.0.10:53
# Name:      bert-inference.ml-serving.svc.cluster.local
# Address:   10.100.200.50
```

## 2. Kubernetes Services

### 2.1 ClusterIP (Default)

**Internal-only access** - Service accessible only within the cluster.

```yaml
# clusterip-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-inference-internal
  namespace: ml-serving
spec:
  type: ClusterIP  # Default type
  selector:
    app: model-inference
  ports:
  - name: http
    protocol: TCP
    port: 80        # Service port
    targetPort: 8080 # Container port
```

**Use case:** Internal microservices, backend ML models called by other services.

**Access:**

```bash
# From another pod in the cluster
curl http://model-inference-internal.ml-serving/predict

# From outside cluster: NOT accessible (internal only)
```

### 2.2 NodePort

**External access via node IP** - Exposes service on each node's IP at a static port.

```yaml
# nodeport-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-inference-nodeport
  namespace: ml-serving
spec:
  type: NodePort
  selector:
    app: model-inference
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 8080
    nodePort: 30080  # External port on nodes (30000-32767)
```

**Access:**

```bash
# Get node IP
kubectl get nodes -o wide
# NAME      STATUS   INTERNAL-IP   EXTERNAL-IP
# node-1    Ready    10.0.1.5      203.0.113.10

# Access via any node IP + nodePort
curl http://203.0.113.10:30080/predict
curl http://<any-node-ip>:30080/predict
```

**Pros:**
- Simple, no additional components needed
- Works in most environments

**Cons:**
- Non-standard ports (30000-32767)
- Must manage firewall rules manually
- No SSL termination

**Use case:** Development, testing, bare-metal clusters.

### 2.3 LoadBalancer

**Cloud load balancer** - Provisions an external load balancer (AWS ELB, GCP LB, Azure LB).

```yaml
# loadbalancer-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-inference-lb
  namespace: ml-serving
  annotations:
    # AWS-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"  # Network Load Balancer
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:us-west-2:123456789:certificate/abc123"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
spec:
  type: LoadBalancer
  selector:
    app: model-inference
  ports:
  - name: https
    protocol: TCP
    port: 443       # External port (internet-facing)
    targetPort: 8080 # Pod port
```

**Deploy and get external IP:**

```bash
kubectl apply -f loadbalancer-service.yaml

# Wait for external IP (can take 1-3 minutes)
kubectl get service model-inference-lb -n ml-serving --watch

# Output:
# NAME                  TYPE           CLUSTER-IP      EXTERNAL-IP                                     PORT(S)
# model-inference-lb    LoadBalancer   10.100.200.50   a1b2c3-123456.us-west-2.elb.amazonaws.com       443:32567/TCP

# Access via external IP/hostname
curl https://a1b2c3-123456.us-west-2.elb.amazonaws.com/predict
```

**Pros:**
- Production-ready, highly available
- Automatic health checks
- SSL termination (with annotations)
- Standard ports (80, 443)

**Cons:**
- Expensive (one LB per service)
- Cloud-specific (doesn't work on bare-metal without MetalLB)

**Use case:** Production ML APIs, critical services, public-facing models.

### 2.4 Service Comparison for ML Use Cases

| Service Type  | Use Case                                    | Cost    | Complexity | External Access |
|---------------|---------------------------------------------|---------|------------|-----------------|
| ClusterIP     | Internal model-to-model communication       | Free    | Low        | No              |
| NodePort      | Development, testing, demos                 | Free    | Low        | Yes (via nodes) |
| LoadBalancer  | Production inference APIs                   | High    | Medium     | Yes (via LB)    |
| Ingress       | Multiple ML models, routing, SSL            | Medium  | Medium     | Yes (via Ingress) |

## 3. Ingress Controllers

### 3.1 What is Ingress?

**Ingress** provides HTTP/HTTPS routing to services based on URL paths or hostnames.

**Why use Ingress instead of multiple LoadBalancers?**

```
Without Ingress (expensive):
- Model A → LoadBalancer 1 → $$$
- Model B → LoadBalancer 2 → $$$
- Model C → LoadBalancer 3 → $$$

With Ingress (cost-effective):
- Ingress Controller → Single LoadBalancer → $
  - /model-a → Model A Service
  - /model-b → Model B Service
  - /model-c → Model C Service
```

**Ingress architecture:**

```
                        Internet
                           │
                           ▼
                ┌──────────────────┐
                │  LoadBalancer    │
                │  (Single)        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Ingress Controller│
                │  (Nginx/Traefik) │
                └────────┬─────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │Service A│     │Service B│     │Service C│
   │(ClusterIP)│  │(ClusterIP)│  │(ClusterIP)│
   └─────────┘     └─────────┘     └─────────┘
```

### 3.2 Install Ingress-Nginx Controller

**Nginx is the most popular Ingress controller:**

```bash
# Install ingress-nginx via Helm
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer

# Wait for external IP
kubectl get service ingress-nginx-controller -n ingress-nginx --watch

# Output:
# NAME                       TYPE           EXTERNAL-IP                        PORT(S)
# ingress-nginx-controller   LoadBalancer   a1b2c3.us-west-2.elb.amazonaws.com  80:31234/TCP,443:31567/TCP
```

### 3.3 Create Ingress Rules for ML Models

**Scenario:** Expose 3 ML models via a single IP with path-based routing.

```yaml
# ml-models-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2  # Rewrite /bert/predict to /predict
    nginx.ingress.kubernetes.io/ssl-redirect: "true"  # Force HTTPS
spec:
  ingressClassName: nginx
  rules:
  - host: ml-api.example.com
    http:
      paths:
      # Route /bert/* to bert-service
      - path: /bert(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: bert-inference
            port:
              number: 80

      # Route /gpt/* to gpt-service
      - path: /gpt(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: gpt-inference
            port:
              number: 80

      # Route /resnet/* to resnet-service
      - path: /resnet(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: resnet-inference
            port:
              number: 80
```

**Create backend services:**

```yaml
# ml-services.yaml
apiVersion: v1
kind: Service
metadata:
  name: bert-inference
  namespace: ml-serving
spec:
  type: ClusterIP
  selector:
    app: bert
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: gpt-inference
  namespace: ml-serving
spec:
  type: ClusterIP
  selector:
    app: gpt
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: resnet-inference
  namespace: ml-serving
spec:
  type: ClusterIP
  selector:
    app: resnet
  ports:
  - port: 80
    targetPort: 8080
```

**Access models:**

```bash
# Point DNS ml-api.example.com to Ingress LoadBalancer IP

# Call BERT model
curl https://ml-api.example.com/bert/predict -d '{"text": "Hello world"}'

# Call GPT model
curl https://ml-api.example.com/gpt/predict -d '{"prompt": "Once upon a time"}'

# Call ResNet model
curl https://ml-api.example.com/resnet/predict -F "image=@cat.jpg"
```

### 3.4 TLS/SSL Termination

**Create TLS certificate secret:**

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=ml-api.example.com/O=MyOrg"

# Create Kubernetes secret
kubectl create secret tls ml-api-tls \
  --namespace ml-serving \
  --cert=tls.crt \
  --key=tls.key

# For production: use cert-manager for automatic Let's Encrypt certificates
```

**Add TLS to Ingress:**

```yaml
# ml-models-ingress-tls.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-ingress
  namespace: ml-serving
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"  # Auto TLS with cert-manager
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - ml-api.example.com
    secretName: ml-api-tls  # TLS certificate secret

  rules:
  - host: ml-api.example.com
    http:
      paths:
      - path: /bert
        pathType: Prefix
        backend:
          service:
            name: bert-inference
            port:
              number: 80
```

**Install cert-manager for automatic TLS (production):**

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create Let's Encrypt issuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Certificates will be auto-generated and renewed!
```

## 4. Network Policies

### 4.1 What are Network Policies?

**Network Policies** control traffic between Pods (like firewall rules).

**Use cases:**
- Isolate sensitive ML models
- Allow only frontend to call backend
- Block public internet access from training pods

### 4.2 Example: Isolate ML Inference Service

**Scenario:** Allow only the frontend service to call the ML model backend.

```yaml
# network-policy-ml-isolation.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-model-isolation
  namespace: ml-serving
spec:
  podSelector:
    matchLabels:
      app: bert-inference  # Apply to BERT pods

  policyTypes:
  - Ingress
  - Egress

  ingress:
  # Allow traffic ONLY from frontend pods
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

  egress:
  # Allow DNS queries
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53

  # Allow access to model storage (S3)
  - to:
    - podSelector:
        matchLabels:
          app: s3-gateway
    ports:
    - protocol: TCP
      port: 443
```

**Result:**
- BERT pods can ONLY receive traffic from frontend pods
- All other traffic is blocked
- BERT pods can query DNS and access S3

### 4.3 Default Deny All Traffic

**Best practice:** Start with deny-all, then explicitly allow needed traffic.

```yaml
# default-deny-all.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: ml-serving
spec:
  podSelector: {}  # Apply to ALL pods in namespace
  policyTypes:
  - Ingress
  - Egress
  # No ingress/egress rules = deny all
```

**Then whitelist specific traffic:**

```yaml
# allow-frontend-to-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: ml-serving
spec:
  podSelector:
    matchLabels:
      tier: backend
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080
```

## 5. Advanced Ingress Features

### 5.1 Rate Limiting

**Protect ML APIs from abuse:**

```yaml
# rate-limiting-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"  # 100 req/min per IP
    nginx.ingress.kubernetes.io/limit-rps: "10"    # 10 req/sec per IP
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"  # Allow bursts up to 50 req/sec
spec:
  ingressClassName: nginx
  rules:
  - host: ml-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bert-inference
            port:
              number: 80
```

### 5.2 Request/Response Size Limits

**Prevent large uploads from overwhelming ML models:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"  # Max 10MB request body
    nginx.ingress.kubernetes.io/client-body-buffer-size: "1m"
spec:
  # ... (same as above)
```

### 5.3 Custom Headers and CORS

**Enable CORS for browser-based ML APIs:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-models-ingress
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://myapp.com, https://dashboard.myapp.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Content-Type, Authorization"
spec:
  # ... (same as above)
```

### 5.4 Canary Deployments with Ingress

**Route 10% of traffic to new model version:**

```yaml
# canary-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bert-inference-canary
  namespace: ml-serving
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"  # 10% traffic to canary
spec:
  ingressClassName: nginx
  rules:
  - host: ml-api.example.com
    http:
      paths:
      - path: /bert
        pathType: Prefix
        backend:
          service:
            name: bert-inference-v2  # New version
            port:
              number: 80
```

**Main Ingress (90% traffic):**

```yaml
# main-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bert-inference-main
  namespace: ml-serving
spec:
  ingressClassName: nginx
  rules:
  - host: ml-api.example.com
    http:
      paths:
      - path: /bert
        pathType: Prefix
        backend:
          service:
            name: bert-inference-v1  # Stable version
            port:
              number: 80
```

## 6. Troubleshooting Networking Issues

### 6.1 Service Not Reachable

**Symptom:** `curl: (7) Failed to connect to service`

**Diagnosis:**

```bash
# 1. Check service exists
kubectl get service <service-name> -n <namespace>

# 2. Check service endpoints (are pods healthy?)
kubectl get endpoints <service-name> -n <namespace>
# If ENDPOINTS column is <none>, no healthy pods are backing the service

# 3. Verify pod selector matches pod labels
kubectl get service <service-name> -n <namespace> -o yaml | grep selector
kubectl get pods -n <namespace> --show-labels

# 4. Check pod port matches service targetPort
kubectl get service <service-name> -n <namespace> -o yaml | grep targetPort
kubectl describe pod <pod-name> -n <namespace> | grep Port
```

**Solutions:**

```bash
# Fix label mismatch
kubectl label pod <pod-name> app=correct-label

# Restart unhealthy pods
kubectl rollout restart deployment <deployment-name> -n <namespace>

# Fix port mismatch in service
kubectl edit service <service-name> -n <namespace>
```

### 6.2 Ingress Not Working

**Symptom:** `404 Not Found` or `502 Bad Gateway`

**Diagnosis:**

```bash
# 1. Check Ingress resource
kubectl describe ingress <ingress-name> -n <namespace>

# Look for:
# - Address: should have external IP
# - Backend: should show service:port
# - Events: check for errors

# 2. Check Ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# 3. Verify backend service is healthy
kubectl get service <backend-service> -n <namespace>
kubectl get endpoints <backend-service> -n <namespace>

# 4. Test service directly (bypass Ingress)
kubectl port-forward service/<backend-service> 8080:80 -n <namespace>
curl localhost:8080/health
```

**Common issues:**

```yaml
# ❌ WRONG: Missing ingressClassName
spec:
  # No ingressClassName specified (won't work)
  rules: ...

# ✅ CORRECT:
spec:
  ingressClassName: nginx
  rules: ...
```

### 6.3 Network Policy Blocking Traffic

**Symptom:** Connection timeouts after applying NetworkPolicy

**Diagnosis:**

```bash
# 1. List all network policies
kubectl get networkpolicy -n <namespace>

# 2. Check if policy is too restrictive
kubectl describe networkpolicy <policy-name> -n <namespace>

# 3. Temporarily delete policy to test
kubectl delete networkpolicy <policy-name> -n <namespace>
# If traffic works now, policy was the issue

# 4. Check pod labels match policy selectors
kubectl get pods -n <namespace> --show-labels
```

**Debug with ephemeral container:**

```bash
# Start debug pod in same namespace
kubectl run debug-pod --image=nicolaka/netshoot -n <namespace> -- sleep 3600

# Test connectivity
kubectl exec -it debug-pod -n <namespace> -- curl http://<service-name>/health

# Test DNS resolution
kubectl exec -it debug-pod -n <namespace> -- nslookup <service-name>
```

## 7. Hands-On Exercise: Deploy Multi-Model ML API with Ingress

**Objective:** Deploy 2 ML models with Ingress routing, TLS, and rate limiting.

**Step 1: Create deployments for 2 models**

```yaml
# sentiment-model.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-model
  namespace: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment
  template:
    metadata:
      labels:
        app: sentiment
    spec:
      containers:
      - name: sentiment
        image: huggingface/transformers:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          value: "distilbert-base-uncased-finetuned-sst-2-english"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: translation-model
  namespace: ml-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: translation
  template:
    metadata:
      labels:
        app: translation
    spec:
      containers:
      - name: translation
        image: huggingface/transformers:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_NAME
          value: "Helsinki-NLP/opus-mt-en-de"
```

**Step 2: Create ClusterIP services**

```yaml
# ml-services.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentiment-service
  namespace: ml-api
spec:
  type: ClusterIP
  selector:
    app: sentiment
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: translation-service
  namespace: ml-api
spec:
  type: ClusterIP
  selector:
    app: translation
  ports:
  - port: 80
    targetPort: 8080
```

**Step 3: Create Ingress with TLS and rate limiting**

```yaml
# ml-api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  namespace: ml-api
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "5m"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - ml-api.yourcompany.com
    secretName: ml-api-tls

  rules:
  - host: ml-api.yourcompany.com
    http:
      paths:
      - path: /sentiment
        pathType: Prefix
        backend:
          service:
            name: sentiment-service
            port:
              number: 80
      - path: /translate
        pathType: Prefix
        backend:
          service:
            name: translation-service
            port:
              number: 80
```

**Step 4: Deploy and test**

```bash
# Create namespace
kubectl create namespace ml-api

# Deploy all resources
kubectl apply -f sentiment-model.yaml
kubectl apply -f ml-services.yaml
kubectl apply -f ml-api-ingress.yaml

# Wait for deployments
kubectl rollout status deployment/sentiment-model -n ml-api
kubectl rollout status deployment/translation-model -n ml-api

# Check Ingress
kubectl get ingress -n ml-api
kubectl describe ingress ml-api-ingress -n ml-api

# Point DNS ml-api.yourcompany.com to Ingress external IP

# Test sentiment API
curl https://ml-api.yourcompany.com/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Test translation API
curl https://ml-api.yourcompany.com/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_lang": "de"}'
```

## 8. Summary

### Key Takeaways

✅ **Kubernetes networking model:**
- Every Pod gets its own IP
- Pods communicate directly without NAT
- CNI plugins (Calico, Cilium) implement networking

✅ **Service types:**
- **ClusterIP:** Internal-only (default)
- **NodePort:** External via node IP + port (30000-32767)
- **LoadBalancer:** Cloud load balancer (production)

✅ **Ingress for HTTP/HTTPS routing:**
- Single LoadBalancer for multiple services
- Path-based routing (/model-a, /model-b)
- TLS/SSL termination
- Rate limiting, CORS, canary deployments

✅ **Network Policies:**
- Control traffic between Pods (firewall rules)
- Best practice: default deny-all, then whitelist
- Use for multi-tenancy and security

✅ **Production best practices:**
- Use Ingress instead of multiple LoadBalancers (cost savings)
- Implement TLS with cert-manager for auto-renewal
- Apply rate limiting to protect ML APIs
- Use Network Policies for isolation

### Real-World Impact

Companies rely on Kubernetes networking for ML at scale:
- **Uber:** Routes 100M+ prediction requests/day via Ingress
- **Netflix:** Serves 200M users with K8s Services and Ingress
- **Spotify:** Load balances personalization APIs across 1000+ pods
- **Airbnb:** Uses Network Policies for multi-tenant ML platforms

## Self-Check Questions

1. What are the 3 core principles of the Kubernetes networking model?
2. When would you use ClusterIP vs LoadBalancer vs Ingress?
3. How does Ingress save costs compared to multiple LoadBalancer services?
4. What's the DNS name format for a Kubernetes Service?
5. How do you enforce that only frontend pods can call backend pods?
6. What annotations enable rate limiting in nginx-ingress?
7. How would you troubleshoot a 502 error from an Ingress?
8. What's the difference between path types Prefix vs Exact?

## Additional Resources

- [Kubernetes Networking Model](https://kubernetes.io/docs/concepts/cluster-administration/networking/)
- [Ingress-Nginx Documentation](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager for TLS](https://cert-manager.io/docs/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Calico CNI](https://docs.tigera.io/calico/latest/about/)

---

**Next lesson:** Storage and Persistence in Kubernetes
