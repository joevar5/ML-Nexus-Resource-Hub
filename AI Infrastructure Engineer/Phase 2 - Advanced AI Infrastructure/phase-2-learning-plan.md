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
| **Mod 03** | Containerization with Docker | [Containerization](lessons/mod-03-containerization/) |
| **Mod 04** | Kubernetes Fundamentals | [Kubernetes](lessons/mod-04-kubernetes/) |
| **Mod 05** | Data Pipelines & Orchestration | [Data Pipelines](lessons/mod-05-data-pipelines/) |
| **Mod 06** | MLOps & Experiment Tracking | [MLOps](lessons/mod-06-mlops/) |
| **Mod 07** | GPU Computing & Distributed Training | [GPU Computing](lessons/mod-07-gpu-computing/) |
| **Mod 08** | Monitoring & Observability | [Monitoring & Observability](lessons/mod-08-monitoring-observability/) |
| **Mod 09** | Infrastructure as Code (Terraform/Pulumi) | [Infrastructure as Code](lessons/mod-09-infrastructure-as-code/) |
| **Mod 10** | LLM Infrastructure | [LLM Infrastructure](lessons/mod-10-llm-infrastructure/) |

---

## 2. Advanced Projects & Learning Strategy

Putting theory into practice is the core of this curriculum. Instead of trying to build all projects at the end, you should build them **iteratively** as you progress through the modules.

### Project Progression Matrix

Below is the recommended sequence, showing exactly which lesson modules prepare you for each project:

| Project | Target Technologies | Prep Modules | Path |
| :--- | :--- | :--- | :--- |
| **Project 101**<br>Basic Model Serving System | FastAPI, Docker, Kubernetes | **Mod 02** (Cloud), **Mod 03** (Docker), **Mod 04** (Kubernetes) | [project-101-basic-model-serving](projects/project-101-basic-model-serving/) |
| **Project 102**<br>End-to-End MLOps Pipeline | Airflow, MLflow, DVC, CI/CD | **Mod 05** (Data Pipelines), **Mod 06** (MLOps) | [project-102-mlops-pipeline](projects/project-102-mlops-pipeline/) |
| **Project 103**<br>LLM Deployment Platform | vLLM, RAG, Vector DBs, GPU Serving | **Mod 07** (GPU Computing), **Mod 08** (Monitoring), **Mod 09** (IaC), **Mod 10** (LLM Infrastructure) | [project-103-llm-deployment](projects/project-103-llm-deployment/) |

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
