# Module 06: MLOps & Experiment Tracking

**Duration:** 40–50 hours · **Difficulty:** Intermediate to Advanced · **Prerequisites:** Modules 02–05, basic ML/data science knowledge

## Overview

MLOps is DevOps for models: the practices that turn "a notebook that got good accuracy once" into a system you can retrain, redeploy, and trust. This module walks the full lifecycle — tracking experiments so results are reproducible, versioning models with real governance, building a feature store so training and serving don't drift apart, and wiring it all into CI/CD.

## What you'll walk away with

- Every experiment tracked, compared, and reproducible via MLflow
- A model registry with real stages (staging → production → archived), not just a folder of `.pkl` files
- A feature store serving the same features online and offline — the drift-between-training-and-serving bug, solved
- A CI/CD pipeline that tests and deploys models automatically
- Comfort running A/B tests to actually validate that a new model is better, not just different

## Lessons

| # | Lesson |
|---|--------|
| 01 | [Introduction to MLOps](./01-introduction-to-mlops.md) — the ML lifecycle, MLOps vs. DevOps, maturity model |
| 02 | [MLflow Experiment Tracking](./02-mlflow-experiment-tracking.md) — tracking server, runs, artifacts, comparison |
| 03 | [Model Registry & Versioning](./03-model-registry-versioning.md) — lifecycle stages, lineage, approval workflows |
| 04 | [Feature Stores & Engineering](./04-feature-stores-engineering.md) — online/offline serving, Feast, point-in-time correctness |
| 05 | [CI/CD for ML Models](./05-cicd-ml-models.md) — testing ML code, data/model validation, GitOps |
| 06 | [Model Deployment Strategies](./06-model-deployment-strategies.md) — batch/real-time/edge, blue-green, canary |
| 07 | [A/B Testing & Experimentation](./07-ab-testing-experimentation.md) — statistical significance, multi-armed bandits |
| 08 | [Best Practices & Governance](./08-best-practices-governance.md) — drift, retraining, cost, security, compliance |

## Hands-on

Five labs in [`labs/`](./labs/), building from MLflow tracking up to a full end-to-end MLOps system: experiment tracking → feature store → CI/CD pipeline → multi-strategy deployment → capstone system with monitoring and A/B testing wired together.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 25 questions, 80% to pass
- **Capstone** — a complete MLOps system: tracking, feature store, CI/CD, multiple deployment strategies, A/B testing, monitoring, and documentation

## Tooling

```bash
pip install mlflow feast scikit-learn pandas fastapi uvicorn pytest great-expectations
```
Plus Docker/Kubernetes (from earlier modules), a Git remote, and a free-tier cloud account.

## What's next

Once models are tracked, registered, and deploying through CI/CD, move to **Module 07: GPU Computing** to scale up the training side.

More reading in [`resources.md`](./resources.md).
