# Module 06: MLOps & Experiment Tracking - Quiz

## Instructions
- This quiz covers the four lessons in this module: MLflow tracking, model registry, CI/CD for models, and deployment strategies
- 11 questions total
- Mix of multiple choice, true/false, and short answer
- Passing score: 80% (9/11 correct)

---

## Section 1: Experiment Tracking (Questions 1-3)

### Question 1
What information should be logged during ML experiments? (Select all that apply)

A) Hyperparameters
B) Evaluation metrics
C) Training duration
D) Dataset version
E) All of the above

**Answer:** E

---

### Question 2
What is the purpose of experiment tagging in MLflow?

A) To organize and filter experiments
B) To improve model accuracy
C) To reduce storage costs
D) To automatically deploy models

**Answer:** A

---

### Question 3 (Short Answer)
Explain the difference between logging parameters and logging metrics in MLflow.

**Answer:** Parameters are input values that configure the training process (e.g., learning rate, batch size) and are logged once at the start. Metrics are output values that measure model performance (e.g., accuracy, loss) and can be logged multiple times throughout training to track progress.

---

## Section 2: CI/CD for ML (Questions 4-7)

### Question 4
How does CI/CD for ML differ from traditional software CI/CD?

A) It includes data validation steps
B) It tests model performance, not just code correctness
C) It versions data and models in addition to code
D) All of the above

**Answer:** D

---

### Question 5
What should be included in automated ML tests? (Select all that apply)

A) Data schema validation
B) Model performance thresholds
C) Inference latency checks
D) Model fairness tests
E) All of the above

**Answer:** E

---

### Question 6
In a deployment gate for ML models, what criteria might prevent a model from being deployed?

A) Performance regression compared to current production model
B) Exceeds latency thresholds
C) Fails fairness checks
D) All of the above

**Answer:** D

---

### Question 7 (Short Answer)
Explain what a "canary deployment" is and why it's useful for ML models.

**Answer:** A canary deployment gradually routes a small percentage of traffic (e.g., 10%) to a new model version while keeping most traffic on the existing version. This allows monitoring of the new model's performance in production with real traffic before fully rolling it out. If issues are detected, you can quickly rollback with minimal impact.

---

## Section 3: Model Deployment (Questions 8-11)

### Question 8
Which deployment pattern is best for making predictions on large datasets that don't require immediate results?

A) Real-time serving
B) Batch inference
C) Stream processing
D) Edge deployment

**Answer:** B

---

### Question 9
What is the primary benefit of model quantization?

A) Increases model accuracy
B) Reduces model size and improves inference speed
C) Makes models easier to train
D) Improves data quality

**Answer:** B

---

### Question 10 (True/False)
Horizontal Pod Autoscaling in Kubernetes can automatically scale ML model serving pods based on CPU, memory, and custom metrics like request rate.

**Answer:** True

---

### Question 11
Which model serving protocol is typically faster for high-throughput scenarios?

A) REST API
B) gRPC
C) GraphQL
D) WebSockets

**Answer:** B

---

## Scoring Guide

- 11 questions, 1 point each (11 points)
- Passing score: 80% (9/11)
