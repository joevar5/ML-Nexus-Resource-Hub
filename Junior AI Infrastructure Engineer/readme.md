# Junior AI Infrastructure Engineer Learning Repository

Welcome! If you've ever wondered how massive AI models get served to millions of users without crashing, or how teams deploy machine learning models at scale, you're in the right place. 

---

## What is an AI Infrastructure Engineer?

An **AI Infrastructure Engineer** (often closely related to MLOps) is the backbone of any AI team. While data scientists and ML researchers focus on training models, AI Infrastructure Engineers build the robust systems, compute platforms, and automated pipelines that allow those models to run reliably, scale effortlessly, and serve predictions instantly in production.

Simply put: **They bridge the gap between a model running on a researcher's laptop and a model serving millions of users in the real world.**

---

## The Story Behind This Repository

Let’s be honest: finding good resources for learning AI/ML Infrastructure is frustrating. 

Most online guides are either **purely theoretical** or **locked behind incredibly expensive paywalls**. I wanted a better way. I built this learning repository by collaborating with AI assistants—burning my own API tokens to synthesize, structure, and share deep MLOps and LLM infrastructure knowledge. It serves as both my personal learning journey and a practical, hands-on path to help others build true production confidence.

---

## The Learning Journey

Here is the exact step-by-step path we will follow to take you from writing basic Python APIs to managing full-scale production infrastructure:

**Model Serving & APIs** ➔ **Application Containerization** ➔ **Cluster Orchestration** ➔ **MLOps & Data Pipelines** ➔ **CI/CD & Workflows** ➔ **Cloud & Monitoring**

---

## What You'll Learn

*   **Deploy ML Models:** Wrap your models in high-performance REST APIs using **FastAPI** and **Flask**.
*   **Containerize Applications:** Package your apps and dependencies consistently using **Docker**.
*   **Orchestrate Deployments:** Scale, manage, and heal your services using **Kubernetes (K8s)**.
*   **Build ML Pipelines:** Version and manage dataset variations with **DVC**, and track experiments with **MLflow**.
*   **Automate Workflows:** Schedule and manage robust data pipelines using **Airflow** or **Prefect**.
*   **Apply CI/CD & IaC:** Automate testing and deployments with CI/CD pipelines, and provision infrastructure using **Terraform**.
*   **Implement Observability:** Monitor health and performance in real time using **Prometheus** and **Grafana**.
*   **Deploy to the Cloud:** Bring everything to life on major cloud platforms (**AWS / GCP / Azure**).
*   **Production Security:** Protect model endpoints, manage secrets, and implement secure network policies.

---

## What You'll Build

I believe that while theory is valuable, to make a real impact we need meaningful, hands-on practical experience. That is why this repository is completely project-driven, focusing on building end-to-end, production-ready systems.

| Project | Description | Duration | Technologies |
|---------|-------------|----------|--------------|
| **01** | [Simple Model API Deployment](projects/project-01-simple-model-api/) | 60 hours | Flask/FastAPI, Docker, AWS/GCP |
| **02** | [Kubernetes Model Serving](projects/project-02-kubernetes-serving/) | 80 hours | Kubernetes, Helm, Prometheus |
| **03** | [ML Pipeline with Experiment Tracking](projects/project-03-ml-pipeline-tracking/) | 100 hours | MLflow, Airflow, DVC |
| **04** | [Monitoring & Alerting System](projects/project-04-monitoring-alerting/) | 80 hours | Prometheus, Grafana, ELK Stack |
| **05** | [Production-Ready ML System (Capstone)](projects/project-05-production-ml-capstone/) | 120 hours | All above + CI/CD |
