# Midterm Practical Exam: AI Infrastructure Engineer

**Duration:** 3 hours
**Passing Score:** 70% (70/100 points)
**Open Book:** Yes (documentation, notes allowed)
**Open Internet:** Yes (no communication with others)

---

## Exam Overview

This practical exam tests your ability to deploy and manage an end-to-end ML infrastructure system. You will containerize an application, deploy it to Kubernetes, set up monitoring, and troubleshoot issues.

## Setup

You will be provided with:
- A simple ML model (pre-trained sklearn or PyTorch model)
- Basic Python inference code
- Access to a Kubernetes cluster
- Prometheus and Grafana installed

## Part 1: Containerization (25 points)

### Task 1.1: Create Optimized Dockerfile (15 points)

Create a Dockerfile for the provided ML inference application with these requirements:

**Requirements:**
- [ ] Use Python 3.11 as base image (2 pts)
- [ ] Implement multi-stage build (4 pts)
- [ ] Run as non-root user (3 pts)
- [ ] Install only production dependencies (2 pts)
- [ ] Image size < 500MB (2 pts)
- [ ] Properly set working directory and copy files (2 pts)

**Deliverable:** `Dockerfile`

### Task 1.2: Docker Compose for Local Testing (10 points)

Create a `docker-compose.yml` file that runs:
- Your ML inference service
- Redis (for caching)
- Prometheus (for metrics)

**Requirements:**
- [ ] All services defined (3 pts)
- [ ] Proper networking between services (3 pts)
- [ ] Health checks configured (2 pts)
- [ ] Volumes for persistence (2 pts)

**Deliverable:** `docker-compose.yml`

**Test:** Run `docker-compose up` and verify all services start

---

## Part 2: Kubernetes Deployment (30 points)

### Task 2.1: Create Kubernetes Manifests (20 points)

Create Kubernetes manifests for deploying your ML service:

**2.1.1 Deployment (10 points)**
- [ ] Deployment with 3 replicas (2 pts)
- [ ] Resource requests and limits (2 pts)
- [ ] Liveness and readiness probes (3 pts)
- [ ] Environment variables from ConfigMap (2 pts)
- [ ] Proper labels and selectors (1 pt)

**2.1.2 Service (5 points)**
- [ ] ClusterIP service exposing port 80 (3 pts)
- [ ] Proper selector matching deployment (2 pts)

**2.1.3 ConfigMap (3 points)**
- [ ] ConfigMap with model configuration (3 pts)

**2.1.4 HorizontalPodAutoscaler (2 points)**
- [ ] HPA targeting 70% CPU (2 pts)
- [ ] Min 2, max 10 replicas

**Deliverables:**
- `deployment.yaml`
- `service.yaml`
- `configmap.yaml`
- `hpa.yaml`

### Task 2.2: Deploy and Verify (10 points)

Deploy your application to Kubernetes:

**Commands to run:**
```bash
# Apply manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get pods
kubectl get svc
kubectl describe hpa

# Test endpoint
# (Provide command showing successful API call)
```

**Evaluation:**
- [ ] All pods running (3 pts)
- [ ] Service accessible (3 pts)
- [ ] HPA configured correctly (2 pts)
- [ ] Can successfully make prediction (2 pts)

**Deliverable:** Screenshot or command output showing successful deployment

---

## Part 3: Monitoring Setup (25 points)

### Task 3.1: Implement Prometheus Metrics (15 points)

Add Prometheus metrics to your application:

**Required Metrics:**
- [ ] `predictions_total` - Counter for total predictions (3 pts)
- [ ] `prediction_duration_seconds` - Histogram for latency (4 pts)
- [ ] `model_loaded` - Gauge for model status (3 pts)
- [ ] `prediction_errors_total` - Counter for errors (3 pts)
- [ ] Metrics properly labeled (model_name, version) (2 pts)

**Deliverable:** Updated application code with metrics

### Task 3.2: Create Grafana Dashboard (10 points)

Create a Grafana dashboard with:

**Required Panels:**
- [ ] Request rate (QPS) - Graph (2 pts)
- [ ] P95 latency - Graph (3 pts)
- [ ] Error rate - Graph (2 pts)
- [ ] Active pods - Stat panel (1 pt)
- [ ] Proper time range and refresh (1 pt)
- [ ] Dashboard variables for filtering (1 pt)

**Deliverable:** `dashboard.json` (exported dashboard)

---

## Part 4: Troubleshooting & Operations (20 points)

### Task 4.1: Debugging Issues (10 points)

You will be given a broken deployment. Diagnose and fix the following issues:

**Scenario:** A deployment is provided with 3 intentional issues.

**Your task:**
1. Identify all issues (3 points, 1 per issue)
2. Fix each issue (6 points, 2 per fix)
3. Document what was wrong and how you fixed it (1 point)

**Possible issues might include:**
- Incorrect image tag
- Missing environment variable
- Wrong port configuration
- Resource limits too low
- Health check misconfiguration

**Deliverable:**
- Fixed YAML files
- `TROUBLESHOOTING.md` documenting issues and fixes

### Task 4.2: Performance Optimization (10 points)

Given current deployment, identify and implement 3 optimizations:

**Areas to optimize:**
- [ ] Reduce Docker image size (3 pts)
- [ ] Optimize resource allocation (3 pts)
- [ ] Improve startup time (2 pts)
- [ ] Add caching layer (2 pts)

**Deliverable:** `OPTIMIZATIONS.md` documenting:
- What you optimized
- Before/after metrics
- Why it improves performance

---

## Submission Requirements

### Directory Structure
```
exam-submission/
├── Dockerfile
├── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── app/
│   └── main.py (with metrics)
├── grafana/
│   └── dashboard.json
├── TROUBLESHOOTING.md
├── OPTIMIZATIONS.md
└── README.md
```

### README.md Must Include
1. How to build and run locally
2. How to deploy to Kubernetes
3. How to access the service
4. How to view metrics/dashboard
5. Any assumptions or notes

---

## Evaluation Criteria

### Functionality (40%)
- Code runs without errors
- All services deploy successfully
- Endpoints respond correctly
- Monitoring works

### Completeness (30%)
- All required files submitted
- All tasks attempted
- Documentation complete

### Quality (20%)
- Code follows best practices
- Proper use of Kubernetes resources
- Security considerations
- Clean, organized code

### Problem Solving (10%)
- Troubleshooting approach
- Optimization choices
- Documentation quality

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Part 1: Containerization** | 25 | Dockerfile quality, docker-compose setup |
| **Part 2: Kubernetes** | 30 | Manifests correct, deployment successful |
| **Part 3: Monitoring** | 25 | Metrics implemented, dashboard functional |
| **Part 4: Troubleshooting** | 20 | Issues fixed, optimizations implemented |
| **Total** | **100** | |

### Bonus Points (up to +10)
- Additional optimizations (+5)
- Exceptional documentation (+3)
- Creative solutions (+2)

---

## Time Management Suggestions

- **Hour 1:** Parts 1 & 2 (Containerization & Kubernetes)
- **Hour 2:** Part 3 (Monitoring)
- **Hour 3:** Part 4 (Troubleshooting) + Documentation

**Tip:** Submit what you have even if incomplete. Partial credit is awarded!

---

## Resources Allowed

✅ **Allowed:**
- Kubernetes documentation
- Docker documentation
- Prometheus/Grafana documentation
- Your own notes
- Course materials
- Stack Overflow (reading only)

❌ **Not Allowed:**
- Communicating with others
- Posting questions online
- Using ChatGPT or AI assistants
- Copying solutions from others

---

## Submission

1. Create a ZIP file of your submission directory
2. Name it: `LastName_FirstName_MidtermExam.zip`
3. Upload to the exam submission portal
4. Verify your submission was received

**Deadline:** End of exam period (3 hours from start)

**Late submissions:** -10% per 15 minutes

---

## Good Luck! 🚀

Remember:
- Read all instructions carefully
- Test your work before submitting
- Document your decisions
- Manage your time wisely
- Submit even if incomplete

**Questions during exam:** Raise hand or use designated communication channel (for clarifications only, not for help with solutions)
