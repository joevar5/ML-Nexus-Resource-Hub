# Project 01: Basic Model Serving System

**Duration:** 30 hours
**Difficulty:** Medium
**Prerequisites:** Completed Modules 01-04

## Project Overview

Build a complete model serving system that deploys a pre-trained image classification model (ResNet) as a REST API. The system will be containerized with Docker, deployed on Kubernetes, monitored with Prometheus and Grafana, and deployed automatically via CI/CD pipeline.

This project demonstrates fundamental skills in:
- Model serving and inference
- Containerization with Docker
- Kubernetes orchestration
- Monitoring and observability
- CI/CD for ML systems

## Learning Objectives

By completing this project, you will be able to:

1. **Deploy ML Model as REST API** - Serve a PyTorch model via FastAPI
2. **Containerize ML Application** - Create optimized Docker images for ML services
3. **Deploy on Kubernetes** - Manage deployments, services, and resources
4. **Implement Monitoring** - Track custom metrics with Prometheus and visualize with Grafana
5. **Build CI/CD Pipeline** - Automate testing and deployment with GitHub Actions
6. **Document Infrastructure** - Create comprehensive documentation for your system

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User/Client                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  LoadBalancer/Service (Port 8080)                  │    │
│  └─────────────────────┬──────────────────────────────┘    │
│                        │                                     │
│     ┌──────────────────┼──────────────────┐                │
│     │                  │                  │                 │
│     ↓                  ↓                  ↓                 │
│  ┌─────┐           ┌─────┐           ┌─────┐              │
│  │ Pod │           │ Pod │           │ Pod │               │
│  │ #1  │           │ #2  │           │ #3  │               │
│  │     │           │     │           │     │               │
│  │FastAPI          │FastAPI          │FastAPI              │
│  │+Model           │+Model           │+Model               │
│  └─────┘           └─────┘           └─────┘              │
│     │                  │                  │                 │
│     └──────────────────┼──────────────────┘                │
│                        │                                     │
│                        ↓                                     │
│            ┌───────────────────────┐                        │
│            │ ConfigMap (settings)  │                        │
│            └───────────────────────┘                        │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ↓ (metrics scraping)
┌─────────────────────────────────────────────────────────────┐
│              Monitoring Stack                               │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │   Prometheus    │ ──────> │    Grafana      │           │
│  │  (Metrics DB)   │         │  (Dashboards)   │           │
│  └─────────────────┘         └─────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Functional Requirements

### Must Have (Required)

#### FR-1: REST API for Image Classification
- **POST /predict** endpoint accepting image upload or URL
- Returns top-5 predictions with confidence scores
- API response time <100ms for small images (<1MB)
- Proper error handling for invalid inputs
- API documentation available at **/docs** (FastAPI auto-generates)

#### FR-2: Containerization
- Dockerfile with multi-stage builds
- Image size <2GB
- Container runs as non-root user
- Environment variables for configuration
- Health check endpoint at **/health**

#### FR-3: Kubernetes Deployment
- Deployment manifest with resource requests/limits
- Service exposing the API (LoadBalancer or NodePort)
- ConfigMap for application configuration
- Liveness and readiness probes configured
- Zero-downtime updates supported

#### FR-4: Monitoring
- Prometheus scraping application metrics
- Custom metrics: **request_count**, **request_duration**, **prediction_count**
- Grafana dashboard showing 5-7 key metrics
- Alert configured for service downtime
- Metrics retained for 7+ days

#### FR-5: CI/CD Pipeline
- Pipeline runs on every commit to main branch
- Automated linting and testing
- Docker image built and pushed to registry
- Automated deployment to Kubernetes (optional for learning environment)
- Pipeline completes in <10 minutes

### Should Have (Recommended)

#### FR-6: Performance
- API latency p95 <100ms
- Handle 10+ concurrent requests
- Model loaded in memory (no cold start delays)
- CPU utilization <70% under normal load

## Technical Stack

### Required Technologies

- **Python 3.9+** - Application language
- **PyTorch** - ML framework
- **FastAPI** - Web framework for API
- **Uvicorn** - ASGI server
- **Docker** - Containerization
- **Kubernetes** - Orchestration (minikube or cloud)
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **GitHub Actions** - CI/CD

### Optional Technologies

- **TorchServe** - Alternative to custom FastAPI serving
- **Nginx** - Reverse proxy
- **Redis** - Caching layer
- **Locust** - Load testing

## Project Structure

```
project-101-basic-model-serving/
├── README.md                       # This file
├── REQUIREMENTS.md                 # Detailed requirements
├── ARCHITECTURE.md                 # Architecture documentation
├── MILESTONES.md                   # Implementation milestones
│
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI application (CODE STUB)
│   ├── model.py                    # Model loading and inference (CODE STUB)
│   ├── utils.py                    # Utility functions (CODE STUB)
│   └── config.py                   # Configuration management (CODE STUB)
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # API tests (TEST STUBS)
│   └── test_model.py               # Model tests (TEST STUBS)
│
├── kubernetes/
│   ├── deployment.yaml             # K8s deployment (TEMPLATE)
│   ├── service.yaml                # K8s service (TEMPLATE)
│   ├── configmap.yaml              # Configuration (TEMPLATE)
│   └── hpa.yaml                    # Horizontal Pod Autoscaler (optional)
│
├── monitoring/
│   ├── prometheus.yml              # Prometheus config
│   └── grafana-dashboard.json      # Grafana dashboard template
│
├── docs/
│   ├── API.md                      # API documentation template
│   ├── DEPLOYMENT.md               # Deployment guide template
│   └── TROUBLESHOOTING.md          # Troubleshooting guide template
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # CI/CD pipeline (TEMPLATE)
│
├── requirements.txt                # Python dependencies
├── requirements-dev.txt            # Development dependencies
├── Dockerfile                      # Docker build instructions (TEMPLATE)
├── docker-compose.yml              # Local development setup
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── pytest.ini                      # Pytest configuration
```

## Implementation Milestones

### Milestone 1: Working FastAPI Application (6 hours)
**Goal:** Model serving locally with REST API

**Tasks:**
- Implement model loading in `model.py`
- Create FastAPI application in `api.py`
- Implement `/predict`, `/health`, `/metrics` endpoints
- Test locally with sample images
- Validate response format

**Deliverable:** Working API on `localhost:8000`

### Milestone 2: Docker Container (4 hours)
**Goal:** Containerized application

**Tasks:**
- Write Dockerfile with multi-stage build
- Optimize image size (<2GB)
- Configure non-root user
- Add health check
- Test container locally

**Deliverable:** Docker image running successfully

### Milestone 3: Kubernetes Deployment (6 hours)
**Goal:** Service running on Kubernetes

**Tasks:**
- Write deployment.yaml
- Write service.yaml
- Create configmap.yaml
- Deploy to minikube
- Verify accessibility

**Deliverable:** Service accessible via K8s

### Milestone 4: Monitoring (6 hours)
**Goal:** Prometheus and Grafana operational

**Tasks:**
- Install Prometheus and Grafana on K8s
- Configure Prometheus scraping
- Implement custom metrics in application
- Create Grafana dashboard
- Set up basic alert

**Deliverable:** Dashboard showing metrics

### Milestone 5: CI/CD Pipeline (4 hours)
**Goal:** Automated testing and deployment

**Tasks:**
- Write GitHub Actions workflow
- Implement linting and testing
- Configure Docker build and push
- Add deployment step (optional)
- Test pipeline end-to-end

**Deliverable:** Working CI/CD pipeline

### Milestone 6: Documentation (4 hours)
**Goal:** Complete documentation

**Tasks:**
- Write API documentation
- Create deployment guide
- Add troubleshooting guide
- Update README with architecture diagram
- Add code comments

**Deliverable:** Comprehensive documentation

## Getting Started

### Step 1: Set Up Project

```bash
# Create project directory
mkdir -p project-101-basic-model-serving
cd project-101-basic-model-serving

# Copy starter code from this repository
# Or create structure manually

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 2: Understand the Code Stubs

Review the code stubs in `src/` directory. Each file has:
- **Complete function signatures**
- **TODO comments** explaining what to implement
- **Hints and examples** for implementation
- **Type hints** for clarity

### Step 3: Start with Milestone 1

Follow the milestones in order. Each milestone builds on the previous one.

### Step 4: Test As You Go

Run tests after each component:

```bash
# Test API
pytest tests/test_api.py

# Test model
pytest tests/test_model.py

# Run all tests
pytest

# Check coverage
pytest --cov=src
```

### Step 5: Document Your Work

Keep your documentation updated as you progress.

## Assessment Rubric

Your project will be assessed on:

| Category | Weight | Criteria |
|----------|--------|----------|
| **Functionality** | 40% | API works correctly with various inputs<br>Kubernetes deployment stable<br>Monitoring operational<br>CI/CD working<br>All acceptance criteria met |
| **Code Quality** | 25% | Clean, well-organized code<br>PEP 8 compliance<br>Proper error handling<br>Externalized configuration<br>Docker/K8s best practices |
| **Documentation** | 20% | Comprehensive README<br>Architecture diagram<br>Clear deployment instructions<br>API documentation<br>Troubleshooting guide |
| **Testing** | 10% | Unit tests (70%+ coverage)<br>Integration tests<br>All tests passing |
| **Innovation** | 5% | Creative solutions<br>Performance optimizations<br>Additional features |

**Passing Score:** 70/100

## Success Metrics

Your project is successful when:

- ✅ API response time p95 <100ms
- ✅ 99%+ uptime over 7 days of running
- ✅ Monitoring dashboard showing all key metrics
- ✅ Load test passing with 20+ concurrent users
- ✅ Complete documentation allowing others to deploy
- ✅ CI/CD pipeline deploying successfully

## Common Pitfalls

1. **Docker image too large** - Use multi-stage builds, slim base images
2. **Model not loaded in memory** - Leads to slow inference
3. **Missing health checks** - Causes deployment issues
4. **Hardcoded configuration** - Use ConfigMap and environment variables
5. **Not testing container locally** - Test with Docker before K8s
6. **Insufficient resource limits** - Can cause OOM kills

## Optimization Tips

1. Use slim Python base image (`python:3.11-slim`)
2. Cache pip dependencies in Docker layer
3. Pre-load model at application startup
4. Use async FastAPI endpoints for better concurrency
5. Implement request batching if needed
6. Add caching for repeated predictions

## Extensions for Advanced Learners

Once you complete the core requirements, try:

1. **Request Batching** - Batch multiple inference requests for throughput
2. **Redis Caching** - Cache common predictions
3. **A/B Testing** - Deploy two model versions simultaneously
4. **API Authentication** - Add API key validation
5. **Helm Chart** - Package for easier deployment
6. **Horizontal Pod Autoscaling** - Auto-scale based on load
7. **Distributed Tracing** - Add Jaeger for request tracing
8. **Image Optimization** - Reduce Docker image to <1GB

## Real-World Relevance

This project mirrors production ML serving systems at companies like:
- **Airbnb** - Serving recommendation models
- **Netflix** - Personalization APIs
- **Uber** - Pricing and ETA prediction

The architecture pattern (API + K8s + monitoring + CI/CD) is industry standard.

## Interview Talking Points

Be prepared to discuss:
- Design decisions for API framework choice (FastAPI vs Flask vs TorchServe)
- Trade-offs in model serving approaches (in-memory vs external service)
- How you handle scaling and resource management in Kubernetes
- Monitoring strategy and key metrics tracked
- CI/CD pipeline design and testing strategy
- How you would debug a production issue using monitoring tools

## Help and Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Tutorials
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Getting Help
- Post questions in GitHub Discussions
- Check TROUBLESHOOTING.md for common issues
- Review solution repository (after attempting yourself)

## Submission

When complete, submit:

1. **GitHub Repository** - All code and documentation
2. **Demo Video** (5-10 minutes) - Show your system working
3. **Architecture Diagram** - Visual representation of your system
4. **README** - Comprehensive documentation
5. **Reflection** (1-2 pages) - What you learned, challenges faced, how you solved them

## Next Steps

After completing this project:

1. **Review and Refactor** - Improve code quality
2. **Extend** - Try advanced features
3. **Share** - Add to your LinkedIn/resume
4. **Move to Project 02** - Build an end-to-end MLOps pipeline

---

**Ready to start?** Head to `REQUIREMENTS.md` for detailed functional requirements, then review the code stubs in `src/` to begin implementation!

**Good luck!** 🚀
