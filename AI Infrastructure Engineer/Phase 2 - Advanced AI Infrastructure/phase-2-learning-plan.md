# Phase 2 Learning Plan: Advanced AI Infrastructure

Welcome to the **Phase 2 Learning Plan**. This document serves as your structured learning path and index. Building on Phase 1's foundations, you will advance from basic deployment skills to architecting production-grade cloud infrastructure, MLOps pipelines, distributed GPU training, and LLM serving platforms.

---

## Curriculum Roadmap

$$
\begin{array}{c}
\boxed{\text{Advanced Foundation Lessons}} \cr
\downarrow \cr
\boxed{\text{Hands-on Advanced Infrastructure Projects}} \cr
\downarrow \cr
\boxed{\text{Self Final Assessment}} \cr
\downarrow \cr
\boxed{\text{Continuous Learning and Resources}}
\end{array}
$$

---

## 1. Advanced Lessons & Modules

Build on your Phase 1 foundation with these advanced module directories covering enterprise-scale AI infrastructure:

| Module | Focus Area | Path |
| :--- | :--- | :--- |
| **Mod 02** | Cloud Computing for ML Infrastructure | [Cloud Computing](lessons/mod-02-cloud-computing/) |
| **Mod 03** | GPU Docker (Docker basics are in Phase 1) | [Containerization](lessons/mod-03-containerization/) |
| **Mod 04** | Kubernetes for GPUs: Helm, GPU scheduling, troubleshooting | [Kubernetes](lessons/mod-04-kubernetes/) |
| **Mod 05** | Data Quality Validation | [Data Pipelines](lessons/mod-05-data-pipelines/) |
| **Mod 06** | MLflow, Model Registry, CI/CD, Deployment Strategies | [MLOps](lessons/mod-06-mlops/) |
| **Mod 07** | GPU Computing & Distributed Training | [GPU Computing](lessons/mod-07-gpu-computing/) |
| **Mod 08** | Prometheus/PromQL, tracing, ML model observability, SLOs | [Monitoring & Observability](lessons/mod-08-monitoring-observability/) |
| **Mod 09** | Infrastructure as Code with Terraform | [Infrastructure as Code](lessons/mod-09-infrastructure-as-code/) |
| **Mod 10** | LLM Infrastructure, Text-to-SQL, Autoscaling, DPO | [LLM Infrastructure](lessons/mod-10-llm-infrastructure/) |

---

## 2. Advanced Projects & Learning Strategy

Putting theory into practice is the core of this curriculum. Instead of trying to build all projects at the end, you should build them **iteratively** as you progress through the modules.

### Project Progression Matrix

Below is the recommended sequence, showing exactly which lesson modules prepare you for each project:

| Project | What's new vs Phase 1 | Prep Modules | Path |
| :--- | :--- | :--- | :--- |
| **Project 101**<br>LLM Serving Platform | vLLM, quantization, continuous batching, goodput, cost per million tokens | **Mod 07** (GPU), **Mod 08** (Monitoring), **Mod 10** (Lessons 02, 06) | [project-101-llm-serving-platform](projects/project-101-llm-serving-platform/) |
| **Project 102**<br>Text-to-SQL Fine-Tuning | LoRA/QLoRA, eval harness, failure analysis, guardrails | **Mod 10** (Lesson 05) | [project-102-text-to-sql-finetuning](projects/project-102-text-to-sql-finetuning/) |
| **Project 103**<br>RAG and Vector Search Service | Embeddings, hybrid search, reranking, retrieval evaluation | **Mod 10** (Lessons 03, 04) | [project-103-rag-vector-search](projects/project-103-rag-vector-search/) |
| **Project 104**<br>LLM Gateway and Multi-Tenant Platform | Routing, rate limits, per-tenant cost tracking, caching | **Mod 10** (Lesson 07, Exercise 07) | [project-104-llm-gateway-multitenant](projects/project-104-llm-gateway-multitenant/) |
| **Project 105**<br>GPU Cluster and Autoscaling | Kubernetes GPU scheduling, KEDA/HPA on queue depth, spot instances | **Mod 04**, **Mod 07**, **Mod 09** | [project-105-gpu-cluster-autoscaling](projects/project-105-gpu-cluster-autoscaling/) |
| **Project 106**<br>Capstone: Text-to-SQL Platform | Fine-tuned model + RAG schema retrieval + gateway + monitoring in one product | All of the above | [project-106-text-to-sql-platform](projects/project-106-text-to-sql-platform/) |

> Serving, Kubernetes, monitoring, and MLOps basics are already covered by the Phase 1 projects, so they are not repeated here.

---

## 3. Assessments & Evaluation

Test your skills and verify your expertise with modular check-ins:

*   **Knowledge Checks**: Review test packages, quizzes, and mock exams under the [Assessments Directory](assessments/).
*   **Coding Challenges**: Sharpen implementation skills with scenario-based tasks in [coding-challenges/](assessments/coding-challenges/).
*   **Practical Exams**: Complete architecture and design tasks in [practical-exams/](assessments/practical-exams/).
*   **Project Evaluation**: Grade your implementations against professional standards using the [Project Rubrics](assessments/rubrics/).

---

## 4. Continuous Learning & Resources

Stay sharp and keep expanding your tooling knowledge as you progress through advanced infrastructure topics:

*   **Module Resources**: Each module includes a `resources.md` with curated docs, tools, and further reading (e.g. [Mod 10 resources](lessons/mod-10-llm-infrastructure/resources.md)).
