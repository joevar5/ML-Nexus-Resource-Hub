# Module 06: MLOps & Experiment Tracking - Quiz

## Instructions
- This quiz covers all 8 lessons in the MLOps module
- 25 questions total
- Mix of multiple choice, true/false, and short answer
- Passing score: 80% (20/25 correct)

---

## Section 1: Introduction to MLOps (Lessons 01-02)

### Question 1
What is the primary purpose of MLOps?

A) To replace data scientists
B) To automate and streamline the ML lifecycle from development to production
C) To make models more accurate
D) To reduce data storage costs

**Answer:** B

---

### Question 2
Which of the following is NOT a core component of the MLOps lifecycle?

A) Data versioning
B) Model training
C) Frontend development
D) Model monitoring

**Answer:** C

---

### Question 3 (True/False)
Experiment tracking tools like MLflow only store model training metrics and cannot store artifacts like model files.

**Answer:** False (MLflow stores both metrics and artifacts including model files, datasets, and images)

---

### Question 4
In MLflow, what is the purpose of a "Run"?

A) To execute a deployed model
B) To represent a single execution of model training code
C) To deploy a model to production
D) To version control your code

**Answer:** B

---

## Section 2: Experiment Tracking (Lesson 03)

### Question 5
What information should be logged during ML experiments? (Select all that apply)

A) Hyperparameters
B) Evaluation metrics
C) Training duration
D) Dataset version
E) All of the above

**Answer:** E

---

### Question 6
What is the purpose of experiment tagging in MLflow?

A) To organize and filter experiments
B) To improve model accuracy
C) To reduce storage costs
D) To automatically deploy models

**Answer:** A

---

### Question 7 (Short Answer)
Explain the difference between logging parameters and logging metrics in MLflow.

**Answer:** Parameters are input values that configure the training process (e.g., learning rate, batch size) and are logged once at the start. Metrics are output values that measure model performance (e.g., accuracy, loss) and can be logged multiple times throughout training to track progress.

---

## Section 3: Feature Stores (Lesson 04)

### Question 8
What problem does a feature store solve?

A) Training-serving skew
B) Feature reusability across teams
C) Point-in-time correctness
D) All of the above

**Answer:** D

---

### Question 9
What is the difference between offline and online feature stores?

A) Offline is for training, online is for inference
B) Offline is slower, online is faster
C) Offline stores historical data, online stores recent data
D) All of the above

**Answer:** D

---

### Question 10 (True/False)
Feature stores automatically guarantee that features used in training match features used in production serving.

**Answer:** True (This is a primary benefit - feature stores ensure consistency by using the same feature definitions and computation logic for both training and serving)

---

### Question 11
In Feast, what is a Feature View?

A) A UI for viewing features
B) A definition of a group of related features with a common data source
C) A visualization of feature importance
D) A cache for frequently accessed features

**Answer:** B

---

## Section 4: CI/CD for ML (Lesson 05)

### Question 12
How does CI/CD for ML differ from traditional software CI/CD?

A) It includes data validation steps
B) It tests model performance, not just code correctness
C) It versions data and models in addition to code
D) All of the above

**Answer:** D

---

### Question 13
What should be included in automated ML tests? (Select all that apply)

A) Data schema validation
B) Model performance thresholds
C) Inference latency checks
D) Model fairness tests
E) All of the above

**Answer:** E

---

### Question 14
In a deployment gate for ML models, what criteria might prevent a model from being deployed?

A) Performance regression compared to current production model
B) Exceeds latency thresholds
C) Fails fairness checks
D) All of the above

**Answer:** D

---

### Question 15 (Short Answer)
Explain what a "canary deployment" is and why it's useful for ML models.

**Answer:** A canary deployment gradually routes a small percentage of traffic (e.g., 10%) to a new model version while keeping most traffic on the existing version. This allows monitoring of the new model's performance in production with real traffic before fully rolling it out. If issues are detected, you can quickly rollback with minimal impact.

---

## Section 5: Model Deployment (Lesson 06)

### Question 16
Which deployment pattern is best for making predictions on large datasets that don't require immediate results?

A) Real-time serving
B) Batch inference
C) Stream processing
D) Edge deployment

**Answer:** B

---

### Question 17
What is the primary benefit of model quantization?

A) Increases model accuracy
B) Reduces model size and improves inference speed
C) Makes models easier to train
D) Improves data quality

**Answer:** B

---

### Question 18 (True/False)
Horizontal Pod Autoscaling in Kubernetes can automatically scale ML model serving pods based on CPU, memory, and custom metrics like request rate.

**Answer:** True

---

### Question 19
Which model serving protocol is typically faster for high-throughput scenarios?

A) REST API
B) gRPC
C) GraphQL
D) WebSockets

**Answer:** B

---

## Section 6: A/B Testing (Lesson 07)

### Question 20
What is the purpose of calculating sample size before running an A/B test?

A) To determine how long the test needs to run
B) To ensure statistical power to detect the minimum effect
C) To avoid running tests longer than necessary
D) All of the above

**Answer:** D

---

### Question 21
What is "Sample Ratio Mismatch" (SRM) and why is it concerning?

A) When the observed traffic split doesn't match the expected ratio
B) It can indicate bugs in assignment logic or data collection
C) It can invalidate experiment results
D) All of the above

**Answer:** D

---

### Question 22 (True/False)
In A/B testing, you should always wait until reaching statistical significance before making a deployment decision.

**Answer:** False (Sequential testing approaches allow for early stopping when significance is reached before planned duration, and you should also consider guardrail metrics and practical significance, not just statistical significance)

---

### Question 23
What is a guardrail metric in A/B testing?

A) A metric that must not degrade beyond a threshold
B) A primary success metric
C) A metric used to calculate sample size
D) A metric for traffic allocation

**Answer:** A

---

## Section 7: Best Practices & Governance (Lesson 08)

### Question 24
What is the purpose of a Model Card?

A) To document model details, intended use, and limitations
B) To track model performance over time
C) To deploy models to production
D) To version control models

**Answer:** A

---

### Question 25 (Short Answer)
Name three types of ML-specific technical debt and provide a brief example of each.

**Answer (Examples):**

1. **Data Debt**: Dependency on multiple unstable data sources, lack of data versioning, no automated data validation
2. **Model Debt**: Multiple redundant models solving similar problems, complex model ensembles that are hard to maintain, undocumented model assumptions
3. **Monitoring Debt**: No drift detection, lack of model performance tracking in production, missing alerting on degraded performance

---

## Answer Key Summary

| Question | Answer | Points |
|----------|--------|--------|
| 1 | B | 1 |
| 2 | C | 1 |
| 3 | False | 1 |
| 4 | B | 1 |
| 5 | E | 1 |
| 6 | A | 1 |
| 7 | Short answer | 2 |
| 8 | D | 1 |
| 9 | D | 1 |
| 10 | True | 1 |
| 11 | B | 1 |
| 12 | D | 1 |
| 13 | E | 1 |
| 14 | D | 1 |
| 15 | Short answer | 2 |
| 16 | B | 1 |
| 17 | B | 1 |
| 18 | True | 1 |
| 19 | B | 1 |
| 20 | D | 1 |
| 21 | D | 1 |
| 22 | False | 1 |
| 23 | A | 1 |
| 24 | A | 1 |
| 25 | Short answer | 3 |

**Total Points: 28**
**Passing Score: 23/28 (82%)**

---

## Grading Rubric for Short Answer Questions

### Question 7 (2 points)
- **2 points:** Clearly distinguishes parameters as inputs and metrics as outputs, mentions timing/frequency of logging
- **1 point:** Partially correct but missing key details
- **0 points:** Incorrect or missing

### Question 15 (2 points)
- **2 points:** Explains gradual traffic routing and monitoring benefits
- **1 point:** Mentions canary concept but lacks detail on benefits
- **0 points:** Incorrect or missing

### Question 25 (3 points)
- **3 points:** Provides 3 distinct types with relevant examples
- **2 points:** Provides 2 types with examples
- **1 point:** Provides 1 type with example
- **0 points:** No correct examples

---

## Study Tips

Focus your review on these key areas:
- MLOps lifecycle components
- Experiment tracking best practices
- Feature store architecture and use cases
- CI/CD pipeline stages for ML
- Model deployment patterns
- A/B testing statistical concepts
- Model governance and documentation
- Technical debt identification

Good luck!
