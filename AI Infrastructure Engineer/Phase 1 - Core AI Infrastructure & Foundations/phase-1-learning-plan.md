# Phase 1 Learning Plan: Core AI Infrastructure & Foundations

Welcome to the **Phase 1 Learning Plan**. This document serves as your structured learning path and index. By following this curriculum, you will transition from writing basic Python scripts to architecting, deploying, and monitoring production-ready machine learning systems.

---

## Curriculum Roadmap

$$
\begin{array}{c}
\boxed{\text{Core Foundation Lessons}} \cr
\downarrow \cr
\boxed{\text{Hands-on AI Infrastructure Projects}} \cr
\downarrow \cr
\boxed{\text{Self Final Assessment}} \cr
\downarrow \cr
\boxed{\text{Continuous Learning and Resources}}
\end{array}
$$

---

## 1. Core Lessons & Modules

Begin by mastering the foundation. Work through these structured module directories to learn the theoretical and practical basics of AI engineering:

| Module | Focus Area | Path |
| :--- | :--- | :--- |
| **Mod 01** | Python Fundamentals for AI Infrastructure | [Python Fundamentals](lessons/mod-001-python-fundamentals/) |
| **Mod 02** | Linux Essentials for DevOps | [Linux Essentials](lessons/mod-002-linux-essentials/) |
| **Mod 03** | Git Version Control & Collaboration | [Git Version Control](lessons/mod-003-git-version-control/) |
| **Mod 04** | Machine Learning Basics | [ML Basics](lessons/mod-004-ml-basics/) |
| **Mod 05** | Containerization with Docker | [Docker Containers](lessons/mod-005-docker-containers/) |
| **Mod 06** | Orchestration with Kubernetes | [Kubernetes Intro](lessons/mod-006-kubernetes-intro/) |
| **Mod 07** | REST APIs & Model Serving | [APIs & Web Services](lessons/mod-007-apis-web-services/) |
| **Mod 08** | Databases & SQL for Data Ops | [Databases & SQL](lessons/mod-008-databases-sql/) |
| **Mod 09** | System Monitoring & Logging | [Monitoring Basics](lessons/mod-009-monitoring-basics/) |
| **Mod 10** | Deploying to Cloud Platforms | [Cloud Platforms](lessons/mod-010-cloud-platforms/) |

---

## 2. Hands-on Projects & Learning Strategy

Putting theory into practice is the core of this curriculum. Instead of trying to build all projects at the end, you should build them **iteratively** as you progress through the modules.

### Project Progression Matrix

Below is the recommended sequence, showing exactly which lesson modules prepare you for each project:

| Project | Target Technologies | Prep Modules | Path |
| :--- | :--- | :--- | :--- |
| **Project 01**<br>Simple Model API | FastAPI, Flask, Docker, AWS/GCP | **Mod 01** (Python), **Mod 05** (Docker), **Mod 07** (REST APIs) | [project-01-simple-model-api](projects/project-01-simple-model-api/) |
| **Project 02**<br>Kubernetes Serving | Kubernetes, Helm, Prometheus | **Mod 02** (Linux), **Mod 06** (Kubernetes) | [project-02-kubernetes-serving](projects/project-02-kubernetes-serving/) |
| **Project 03**<br>ML Pipeline & DVC | MLflow, Airflow, DVC | **Mod 03** (Git), **Mod 04** (ML Basics), **Mod 08** (Databases) | [project-03-ml-pipeline-tracking](projects/project-03-ml-pipeline-tracking/) |
| **Project 04**<br>Observability & Monitoring | Prometheus, Grafana, ELK Stack | **Mod 09** (System Monitoring & Logging) | [project-04-monitoring-alerting](projects/project-04-monitoring-alerting/) |
| **Project 05**<br>Capstone Production ML | All above + GitHub Actions & Terraform | **Mod 10** (Cloud Platforms) + All Prior Modules | [project-05-production-ml-capstone](projects/project-05-production-ml-capstone/) |
---

## 3. Assessments & Evaluation

Test your skills and verify your expertise with modular check-ins:

*   **Knowledge Checks**: Review test packages, quizzes, and mock exams under the [Assessments Directory](assessments/).
*   **Quizzes**: Assess your understanding of each lesson module in [quizzes/](assessments/quizzes/).
*   **Practical Exams**: Complete scenario-based architecture tasks in [practical-exams/](assessments/practical-exams/).
*   **Project Evaluation**: Grade your implementations against professional standards using the [Project Rubrics](assessments/rubrics/).

---

## 4. Continuous Learning & Resources

Stay sharp and keep expanding your tooling knowledge:

*   **Reference Cheat Sheets**: Quick commands for Git, Docker, Kubernetes, and SQL in [cheat-sheets/](resources/cheat-sheets/).
*   **Reading Lists**: Deep-dives, engineering blogs, and book recommendations in [reading-lists/](resources/reading-lists/).
*   **Core Tools**: List of industry-standard tools for infrastructure engineering in [tools.md](resources/tools.md).
