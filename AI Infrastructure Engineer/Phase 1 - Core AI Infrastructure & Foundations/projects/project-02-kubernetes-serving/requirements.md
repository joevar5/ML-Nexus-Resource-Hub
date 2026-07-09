# Project 02: Kubernetes Serving Requirements

This document outlines the core functional, non-functional, and Kubernetes-specific requirements for deploying the model serving API.

---

## 1. Functional Requirements (FR)

### FR-1: Kubernetes Deployment Configuration
* **Replication:** Deploy with `3` baseline replicas (`spec.replicas: 3`) for high availability.
* **Health Probes:**
  * **Liveness:** `GET /health` (delay: 30s, period: 10s, timeout: 5s, failure: 3). Restarts hung pods.
  * **Readiness:** `GET /health` (delay: 10s, period: 5s, timeout: 3s, failure: 3). Controls traffic routing.
* **Resource Allocations:**
  | Resource | Request (Guaranteed) | Limit (Hard Cap) |
  | :--- | :--- | :--- |
  | **CPU** | `500m` (0.5 cores) | `1000m` (1.0 core) |
  | **Memory** | `1Gi` | `2Gi` |
* **ConfigMap (`model-config`):** Separate settings from code by injecting env vars:
  - `model_name`: e.g., `"resnet50"`
  - `log_level`: `"INFO"`
  - `max_batch_size`: `32`
  - `timeout`: `30`

### FR-2: Networking & Load Balancing
* **ClusterIP Service:** Internal load balancer matching label selector `app: model-api` on port `80` (targetPort `5000`).
* **LoadBalancer Service:** External-facing entry point on port `80`.
* **NGINX Ingress:** Path-based HTTP routing mapping `/predict`, `/health`, and `/metrics` to the service.

### FR-3: Auto-Scaling (HPA)
* **Scale Limits:** Min replicas: `3`, Max replicas: `10`.
* **Utilization Targets:** CPU: `70%`, Memory: `80%`.
* **Scaling Behavior:**
  - **Scale Up:** Immediate, up to 100% replica increase per 30 seconds.
  - **Scale Down:** Conservative, 5-minute stabilization window to avoid replica flapping.

### FR-4: Rolling Updates & Rollbacks
* **RollingUpdate Policy:** `maxSurge: 1`, `maxUnavailable: 0` to maintain minimum capacity during rollouts.
* **Rollbacks:** Maintain revision history to support quick `kubectl rollout undo` commands.

### FR-5: Observability & Monitoring
* **Prometheus:** Scrapes `/metrics` every 30 seconds using a custom `ServiceMonitor`.
* **Grafana:** Visualize pod count, restarts, CPU/Memory usage, requests per second (RPS), latency, and error rates.
* **Exposed Metrics:**
  | Metric Name | Type | Description / Labels |
  | :--- | :--- | :--- |
  | `model_api_requests_total` | Counter | Total requests (`method`, `endpoint`, `status`) |
  | `model_api_predictions_total` | Counter | Total predictions (`model_name`, `status`) |
  | `model_api_request_duration_seconds` | Histogram | Request latency (`method`, `endpoint`) |
  | `model_api_inference_duration_seconds` | Histogram | Inference processing time (`model_name`) |
  | `model_api_model_loaded` | Gauge | Model loaded state (`model_name`, `version`) |
  | `model_api_active_connections` | Gauge | Number of active requests |

* **Alert Rules:**
  - **HighErrorRate:** HTTP 5xx error rate > 5% for 2 minutes.
  - **PodCrashLooping:** Container restart rate > 0 for 5 minutes.
  - **HighMemoryUsage:** Memory usage exceeds 90% of limit for 5 minutes.

---

## 2. Non-Functional Requirements (NFR)

* **Performance:**
  - **Throughput:** Sustain `1000+` requests per second across the cluster under load.
  - **Latency:** P95 latency `< 300ms`, P99 `< 500ms`.
  - **Scaling Speed:** Auto-scaling triggers additional pods within `2` minutes of threshold breach.
* **Reliability:**
  - **Downtime:** 99.9% availability.
  - **Updates:** Zero downtime during rolling upgrades.
* **Security:**
  - **Secrets:** Store sensitive data (keys/credentials) in Kubernetes `Secrets`.
  - **Access:** Run container as a non-root user; apply Pod Security Standards.
  - **Network Isolation:** Apply Network Policies to restrict ingress traffic to Ingress Controller only.

---

## 3. Kubernetes Standards

* **Namespace:** Isolated deployment within the `ml-serving` namespace.
* **Metadata Labels:** Apply labels consistently:
  ```yaml
  labels:
    app: model-api
    version: v1.0
    component: inference
  ```
* **Metrics Scraping Annotations:**
  ```yaml
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "5000"
    prometheus.io/path: "/metrics"
  ```
