# Kubernetes Model Serving: Step-by-Step Execution & Hosting Guide

This guide provides a comprehensive, step-by-step recipe to build, host, and verify the model serving API on a local Kubernetes cluster using Minikube. 

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Step 1: Start Minikube & Enable Addons](#step-1-start-minikube--enable-addons)
3. [Step 2: Configure Docker Environment & Build Image](#step-2-configure-docker-environment--build-image)
4. [Step 3: Deploy Kubernetes Resources (Manifests vs. Helm)](#step-3-deploy-kubernetes-resources-manifests-vs-helm)
5. [Step 4: Verify Deployment & Host Status](#step-4-verify-deployment--host-status)
6. [Step 5: Test Model API Endpoints](#step-5-test-model-api-endpoints)
7. [Step 6: Set Up Monitoring (Prometheus & Grafana)](#step-6-set-up-monitoring-prometheus--grafana)
8. [Step 7: Run Advanced Load Testing with Locust](#step-7-run-advanced-load-testing-with-locust)
9. [Step 8: Run Automated Integration Tests](#step-8-run-automated-integration-tests)
10. [Step 9: Perform Zero-Downtime Rolling Updates & Rollbacks](#step-9-perform-zero-downtime-rolling-updates--rollbacks)
11. [Step 10: Clean Up Resources](#step-10-clean-up-resources)

---

## 1. Prerequisites

Before starting, ensure you have the following CLI tools installed:

| Tool | Purpose | Install Command (macOS - Homebrew) |
| :--- | :--- | :--- |
| **Docker** | Container runtimes | `brew install --cask docker` |
| **Minikube** | Local single-node Kubernetes cluster | `brew install minikube` |
| **kubectl** | Kubernetes cluster CLI | `brew install kubernetes-cli` |
| **Helm** | Kubernetes package manager | `brew install helm` |
| **Locust** | Load testing tool | `pip install locust` |
| **pytest** | Python testing framework | `pip install pytest` |
| **curl** | HTTP test client | Pre-installed |

> [!IMPORTANT]
> Make sure Docker Desktop is running on your machine before starting Minikube.

---

## Step 1: Start Minikube & Enable Addons

We start by spawning a local Kubernetes cluster with sufficient resources for our application, metrics server, and ingress controller.

```bash
# Start Minikube with resources tailored for ML serving workload simulation
minikube start --cpus=4 --memory=8192 --driver=docker
```

Once Minikube is running, verify the node status:
```bash
kubectl get nodes
```

Next, enable the required Kubernetes addons:
```bash
# Enable Metrics Server (required by HPA to monitor CPU & memory usage)
minikube addons enable metrics-server

# Enable NGINX Ingress Controller (required for ingress path routing)
minikube addons enable ingress
```

Verify that the metrics server and ingress controller pods are running successfully in the cluster:
```bash
kubectl get pods -n kube-system
kubectl get pods -n ingress-nginx
```

---

## Step 2: Configure Docker Environment & Build Image

Instead of pushing our Docker image to a public registry (Docker Hub/GCR), we configure our local environment to build the image directly inside Minikube's internal Docker registry.

```bash
# Point your shell's Docker CLI to Minikube's built-in Docker daemon
eval $(minikube docker-env)
```

Now, build the model serving Docker image from the root of `project-02-kubernetes-serving`:
```bash
# Build container image tagged as model-api:v1.0
docker build -t model-api:v1.0 .
```

Verify that the image is available in Minikube's Docker registry:
```bash
docker images | grep model-api
```

---

## Step 3: Deploy Kubernetes Resources (Manifests vs. Helm)

You have two choices for deploying the model serving API: **Option A (Raw Manifests)** or **Option B (Helm Chart)**.

### Option A: Deploying with Raw YAML Manifests
Apply the manifests located in the [kubernetes](./kubernetes) directory manually:

```bash
# 1. Create the ml-serving namespace and ConfigMap configuration parameters
kubectl apply -f kubernetes/configmap.yaml

# 2. Create the Deployment (starts 3 replica pods of model-api)
kubectl apply -f kubernetes/deployment.yaml

# 3. Create the Service exposing internal and external load balancing
kubectl apply -f kubernetes/service.yaml

# 4. Apply the Horizontal Pod Autoscaler (HPA)
kubectl apply -f kubernetes/hpa.yaml

# 5. Apply Ingress path-based routing rules
kubectl apply -f kubernetes/ingress.yaml
```

---

### Option B: Deploying with Helm (Recommended for Production)
The [helm](./helm) directory contains a parameterized chart of the manifests. This abstracts the raw configurations and enables promoting the same application release across `dev`, `staging`, and `production` environments by overriding variables.

```bash
# 1. Lint the Helm chart to check for errors
helm lint helm/model-api

# 2. Render templates locally to verify manifest output
helm template model-api ./helm/model-api

# 3. Install/upgrade the app inside the ml-serving namespace
# (Note: For local Minikube development, we set serviceMonitor.enabled=false to bypass CRD checks, 
# modelStorage.type=none to skip network PV mounts, and target the locally built image tag v1.0)
helm upgrade --install model-api ./helm/model-api \
  -n ml-serving \
  --create-namespace \
  --set serviceMonitor.enabled=false \
  --set modelStorage.type=none \
  --set image.tag=v1.0

# 4. View active Helm releases
helm list -n ml-serving
```

> [!NOTE]
> Values like replica count and CPU limits can be customized via `./helm/model-api/values.yaml` or overridden on the fly with `--set` flags (e.g. `helm upgrade --install model-api ./helm/model-api --set replicaCount=5 -n ml-serving`).

> [!WARNING]
> **Resource & Filesystem Guidelines for Local Development (8 GB RAM Mac):**
> * **Memory Limits (OOM):** Loading the real PyTorch ResNet50 model takes ~500 MB of RAM. Do not set Gunicorn workers (`WORKERS`) higher than `1` per container, or it will exceed the pod's 1 GiB memory limit and get OOM-killed.
> * **Read-only Filesystem:** The chart enforces a read-only root filesystem. You must direct Gunicorn writes (`GUNICORN_CMD_ARGS="--worker-tmp-dir /tmp"`) and PyTorch weight downloads (`TORCH_HOME="/tmp/torch"`, `HOME="/tmp"`) to the writable `/tmp` volume to avoid crashes.
> * **Resource Starvation:** If you have too many stale/crashed pods running in the background, they will completely fill up Minikube's 4 GB memory pool. If this happens, perform a clean reset:
>   ```bash
>   helm uninstall model-api -n ml-serving
>   kubectl delete pods --all -n ml-serving --force
>   ```

---

## Step 4: Verify Deployment & Host Status

Verify that all deployed resources are working in the isolated `ml-serving` namespace.

```bash
# List all resources under the ml-serving namespace
kubectl get all -n ml-serving
```

### Checking Pod Readiness Check Process
Our stand-in model API simulates model load latency of 1.5 seconds.
During the startup phase, track how pods transition from `0/1 READY` to `1/1 READY`:

```bash
# Watch pod initialization progress
kubectl get pods -n ml-serving -w
```

If any pod fails to enter the `Running` state, view the event log or container stdout:
```bash
# View deployment events
kubectl describe deployment model-api -n ml-serving

# View standard output logs of a specific pod
kubectl logs -n ml-serving deployment/model-api --tail=100
```

---

## Step 5: Test Model API Endpoints

To interact with the running pods without exposing them publicly, set up a secure port-forward tunnel from your local workstation:

```bash
# Forward traffic from localhost:8080 to service target port 80 (pointing to pod port 5000)
kubectl port-forward -n ml-serving svc/model-api 8080:80
```

> [!TIP]
> Keep this terminal window open. Use a new terminal window to run the tests below.

### 1. Check Root Endpoint
```bash
curl -i http://localhost:8080/
```
*Response:* Returns API metadata and version.

### 2. Check Health Endpoints
```bash
# Combined Liveness & Readiness Check
curl -i http://localhost:8080/health

# Dedicated Liveness Check
curl -i http://localhost:8080/health/live

# Dedicated Readiness Check
curl -i http://localhost:8080/health/ready
```

### 3. Check Metrics Endpoint (Prometheus formatting)
```bash
curl -s http://localhost:8080/metrics | grep model_api
```

### 4. Trigger Predictions (Inference)
Send a POST request containing input instances to the `/predict` route:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"instances": ["hello world", "test input data", 42]}' \
  http://localhost:8080/predict
```

---

## Step 6: Set Up Monitoring (Prometheus & Grafana)

The [monitoring](./monitoring) directory contains resources for automatic metrics collection. The Prometheus Operator will scrape metrics from the model API using the Custom Resource Definition (CRD) defined in `servicemonitor.yaml`.

### 1. Install Prometheus Operator & Grafana
Add the Prometheus community Helm repository and install the monitoring stack:

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack in a dedicated monitoring namespace
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### 2. Apply the ServiceMonitor Manifest
First, update the placeholders in `monitoring/servicemonitor.yaml`:
* Set `spec.selector.matchLabels` to `app: model-api` to target the Service.
* Set `spec.namespaceSelector.matchNames` to `["ml-serving"]`.
* In `endpoints`, configure the port named `http`, the path to `/metrics`, and the scraping interval to `30s`.

Apply the configuration file:
```bash
kubectl apply -f monitoring/servicemonitor.yaml -n ml-serving
```

### 3. Verify Scraping Targets in Prometheus UI
```bash
# Port-forward Prometheus Server
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```
Navigate to `http://localhost:9090/targets` in your browser. Verify that the `model-api` endpoints are successfully discovered and healthy.

### 4. Import the Grafana Dashboard
```bash
# Port-forward Grafana Service
kubectl port-forward -n monitoring svc/prometheus-grafana 8081:80
```
1. Open `http://localhost:8081` in your browser.
2. Log in using default credentials:
   * **Username:** `admin`
   * **Password:** `prom-operator` (Or retrieve it with: `kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode`)
3. Navigate to **Dashboards** -> **Import**.
4. Load the dashboard JSON definition located in the [grafana/model-api-dashboard.json](./grafana/model-api-dashboard.json) file.
5. Select the **Prometheus** data source and click **Import**.
6. The dashboard will visualize pod count, CPU/Memory consumption, requests per second (RPS), request latencies, and error rates.

---

## Step 7: Run Advanced Load Testing with Locust

Instead of a simple curl loop, the [loadtest](./loadtest) directory contains a Python Locust load script (`locustfile.py`) designed to simulate realistic user request spikes.

1. Install Locust on your workstation:
   ```bash
   pip install locust
   ```

2. Make sure your tunnel to the API is active: `kubectl port-forward -n ml-serving svc/model-api 8080:80`

3. Run the load test in headless mode to simulate a traffic spike (500 users, spawning at 15 users/sec):
   ```bash
   mkdir -p loadtest/results
   
   locust -f loadtest/locustfile.py \
     --host http://localhost:8080 \
     --headless --users 500 --spawn-rate 15 --run-time 3m \
     --csv loadtest/results/spike
   ```

4. While the test is running, watch your HPA scale up the replicas from 3 to 10 in response to the high CPU load:
   ```bash
   kubectl get hpa -n ml-serving -w
   ```

---

## Step 8: Run Automated Integration Tests

The project includes an integration test suite located in the [tests](./tests) directory to perform end-to-end configuration and deployment validation. These tests query the live Kubernetes API directly.

1. Make sure you have python testing dependencies installed:
   ```bash
   pip install pytest kubernetes requests
   ```

2. Run quick cluster structure and resource validation checks:
   ```bash
   pytest tests/test_k8s.py
   ```

3. Run the entire verification test suite (including rolling update and HPA auto-scaling simulation tests):
   ```bash
   pytest tests/test_k8s.py -m slow
   ```

---

## Step 9: Perform Zero-Downtime Rolling Updates & Rollbacks

Kubernetes is configured to perform updates with `maxSurge: 1` and `maxUnavailable: 0` to maintain constant uptime.

### Option A: Managing Updates with Helm (Recommended)
Upgrades are managed by amending parameters in the chart and performing a rollout.

```bash
# Upgrade the Helm release (e.g. increase replicas)
helm upgrade model-api ./helm/model-api -n ml-serving --set replicaCount=4

# Roll back to the previous Helm revision instantly if something goes wrong
helm rollback model-api -n ml-serving
```

### Option B: Managing Updates with raw kubectl
```bash
# Update Log Level configuration to DEBUG
kubectl patch configmap model-api-config -n ml-serving -p '{"data":{"log_level":"DEBUG"}}'

# Force a rollout restart of the deployment to apply the ConfigMap changes
kubectl rollout restart deployment/model-api -n ml-serving

# Monitor rollout progress
kubectl rollout status deployment/model-api -n ml-serving

# Review rollout history
kubectl rollout history deployment/model-api -n ml-serving

# Rollback to the previous deployment revision
kubectl rollout undo deployment/model-api -n ml-serving
```

---

## Step 10: Clean Up Resources

Once you have completed verification, tear down all Kubernetes resources to free up resources on your workstation.

```bash
# Delete all resources created in the ml-serving and monitoring namespaces
kubectl delete namespace ml-serving
kubectl delete namespace monitoring

# Stop and delete local Minikube VM/container resources
minikube destroy
```
