# Module 06: MLOps & Experiment Tracking

> Trimmed to what Projects 102 and 106 use: MLflow tracking, model registry, CI/CD for models, and deployment strategies.

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
| 01 | [MLflow Experiment Tracking](./01-mlflow-experiment-tracking.md) — tracking server, runs, artifacts, comparison |
| 02 | [Model Registry & Versioning](./02-model-registry-versioning.md) — lifecycle stages, lineage, approval workflows |
| 03 | [CI/CD for ML Models](./03-cicd-ml-models.md) — testing ML code, data/model validation, GitOps |
| 04 | [Model Deployment Strategies](./04-model-deployment-strategies.md) — batch/real-time/edge, blue-green, canary |

## Hands-on

Five labs in [`labs/`](./labs/), building from MLflow tracking up to a full end-to-end MLOps system: experiment tracking → feature store → CI/CD pipeline → multi-strategy deployment → capstone system with monitoring and A/B testing wired together.

## Assessment

- **Quiz** — [`quizzes/module-quiz.md`](./quizzes/module-quiz.md), 11 questions, 80% to pass
- **Capstone** — a complete MLOps system: tracking, feature store, CI/CD, multiple deployment strategies, A/B testing, monitoring, and documentation

## Tooling

```bash
pip install mlflow feast scikit-learn pandas fastapi uvicorn pytest great-expectations
```
Plus Docker/Kubernetes (from earlier modules), a Git remote, and a free-tier cloud account.

## What's next

Once models are tracked, registered, and deploying through CI/CD, move to **Module 07: GPU Computing** to scale up the training side.

More reading in [`resources.md`](./resources.md).
