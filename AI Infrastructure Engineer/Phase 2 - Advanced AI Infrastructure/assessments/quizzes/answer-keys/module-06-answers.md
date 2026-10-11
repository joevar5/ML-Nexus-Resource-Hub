# Module 06: MLOps — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-06-mlops/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What information should be logged during ML experiments? (Select all that apply)

**Answer:** E) All of the above (Hyperparameters, Evaluation metrics, Training duration, Dataset version)

**Explanation:**
Reproducibility and meaningful comparison require capturing inputs (hyperparameters, dataset version), outputs (evaluation metrics), and execution context (training duration, hardware, code commit). Skipping any of these makes it hard to explain why a run won or to recreate it later. The cost of logging is tiny compared to the cost of an undebuggable model.

**Common Mistakes:**
- Picking only A and B and skipping dataset version is a frequent mistake; without dataset versioning you can't tell whether a metric change came from code or data.
- Skipping training duration ignores resource and cost signals that matter for production planning.

**Related Material:** lessons/mod-06-mlops/01-mlflow-experiment-tracking.md

---

## Question 2
**Q:** What is the purpose of experiment tagging in MLflow?

**Answer:** A) To organize and filter experiments

**Explanation:**
Tags are arbitrary key-value metadata attached to Runs or Experiments (e.g., `team=ranking`, `dataset=v3`, `purpose=baseline`) so you can search, group, and filter through the UI or API. They are organizational metadata only; they do not influence model behavior or deployment.

**Common Mistakes:**
- Choosing B ("improve model accuracy") confuses metadata labels with training inputs — tags don't enter the model.
- Choosing D ("automatically deploy models") confuses tagging with the Model Registry's stage transitions, which are the actual deployment-related labels.

**Related Material:** lessons/mod-06-mlops/01-mlflow-experiment-tracking.md

---

## Question 3
**Q:** (Short Answer) Explain the difference between logging parameters and logging metrics in MLflow.

**Answer:** Parameters are input configuration values (e.g., learning rate, batch size, model type) logged once at the start of a Run, typically via `mlflow.log_param`. Metrics are measured output values (e.g., accuracy, loss, F1) logged via `mlflow.log_metric`, can be logged repeatedly with a step index, and form time-series you can plot over training epochs.

**Explanation:**
The split matters because parameters describe the experiment's setup while metrics describe its outcome. Parameters are immutable once set on a Run; metrics are append-only series. Mixing them up makes runs hard to compare and breaks tooling that expects scalar params and time-series metrics.

**Common Mistakes:**
- Saying both are "just values" misses the time-series nature of metrics and the configuration role of parameters.
- Forgetting that metrics support a `step` argument leads people to overwrite values instead of building training curves.

**Related Material:** lessons/mod-06-mlops/01-mlflow-experiment-tracking.md

---

## Question 4
**Q:** How does CI/CD for ML differ from traditional software CI/CD?

**Answer:** D) All of the above (Includes data validation, tests model performance not just code correctness, versions data and models in addition to code)

**Explanation:**
ML pipelines fail not only from code defects but also from data quality regressions, so validation has to extend to schema and distribution checks. Tests must verify model behavior on held-out and slice-level data, not only unit-test code paths. And reproducibility requires versioning the trio of code + data + model artifacts together.

**Common Mistakes:**
- Choosing only A overlooks that "the model itself is now part of the build artifact" is the deeper difference.
- Choosing only C misses that the actual test surface (data and model behavior) is what makes ML CI/CD distinctive.

**Related Material:** lessons/mod-06-mlops/03-cicd-ml-models.md

---

## Question 5
**Q:** What should be included in automated ML tests? (Select all that apply)

**Answer:** E) All of the above (Data schema validation, Model performance thresholds, Inference latency checks, Model fairness tests)

**Explanation:**
A complete ML test suite checks inputs (schema and distribution), outputs (accuracy/precision/recall thresholds and slice metrics), runtime characteristics (latency and throughput SLOs), and responsible-AI properties (fairness across demographic slices). Skipping any one creates a class of bugs that won't be caught before production. Each should run automatically in the deployment pipeline.

**Common Mistakes:**
- Selecting only A and B treats ML tests as a beefier unit-test suite and forgets non-functional requirements like latency.
- Omitting fairness tests is a common oversight that surfaces only when production users complain.

**Related Material:** lessons/mod-06-mlops/03-cicd-ml-models.md

---

## Question 6
**Q:** In a deployment gate for ML models, what criteria might prevent a model from being deployed?

**Answer:** D) All of the above (Performance regression vs current prod, Exceeds latency thresholds, Fails fairness checks)

**Explanation:**
Deployment gates compare a candidate model against the current production champion across quality (no metric regression), service (latency/throughput within SLO), and responsibility (fairness, bias) dimensions. Any single failure should block promotion to keep production safe. Gates are the automated equivalent of a release checklist.

**Common Mistakes:**
- Picking only A treats deployment gates as a pure accuracy gate and ignores serving and ethical constraints.
- Picking only B treats them as a performance test, missing the model-quality and fairness dimensions.

**Related Material:** lessons/mod-06-mlops/03-cicd-ml-models.md

---

## Question 7
**Q:** (Short Answer) Explain what a "canary deployment" is and why it's useful for ML models.

**Answer:** A canary deployment routes a small fraction of live traffic (commonly 1–10%) to a new model version while the bulk of traffic continues to hit the existing version. You monitor the canary's business and operational metrics on real production traffic, then ramp up gradually or roll back fast if regressions appear. For ML models this is especially valuable because offline metrics often diverge from real user behavior, so canary exposure is the cheapest way to validate that an improved offline score actually helps in production.

**Explanation:**
Canary deployments blast-radius cap the impact of a bad model: at 5% traffic, even a catastrophic regression affects 5% of users for the rollout window. They also generate live signal — latency, error rate, business KPIs — that you cannot get from a staging environment. Combined with automated rollback on guardrail breach, they make ML rollouts safe to do frequently.

**Common Mistakes:**
- Confusing canary with blue/green (which is 0% then 100%, with no gradual ramp).
- Skipping the rollback half of the answer; canary without monitoring and rollback is just a slow deploy.

**Related Material:** lessons/mod-06-mlops/04-model-deployment-strategies.md

---

## Question 8
**Q:** Which deployment pattern is best for making predictions on large datasets that don't require immediate results?

**Answer:** B) Batch inference

**Explanation:**
Batch inference runs the model over a large dataset on a schedule (e.g., nightly), writes predictions to a store, and is consumed asynchronously. It maximizes throughput and hardware utilization per prediction and is the right choice when latency-to-prediction can be hours, not milliseconds. Examples include nightly recommendation refresh, lead scoring, or churn risk for a customer base.

**Common Mistakes:**
- Picking A (real-time serving) wastes infrastructure when there is no per-request latency requirement.
- Picking C (stream processing) is appropriate for continuous event streams, not for bulk-dataset scoring with no urgency.

**Related Material:** lessons/mod-06-mlops/04-model-deployment-strategies.md

---

## Question 9
**Q:** What is the primary benefit of model quantization?

**Answer:** B) Reduces model size and improves inference speed

**Explanation:**
Quantization converts model weights and/or activations from higher-precision floats (FP32) to lower-precision representations (FP16, INT8, sometimes INT4). The result is a smaller artifact, lower memory bandwidth, and faster matrix multiplies on hardware that supports the lower precision. Accuracy typically drops slightly, which is a tradeoff, not a goal.

**Common Mistakes:**
- Selecting A ("increases accuracy") inverts the tradeoff — quantization usually costs a little accuracy in exchange for size and speed.
- Selecting C ("easier to train") confuses post-training quantization with training optimizations; quantization is primarily an inference-time technique.

**Related Material:** lessons/mod-06-mlops/04-model-deployment-strategies.md

---

## Question 10
**Q:** (True/False) Horizontal Pod Autoscaling in Kubernetes can automatically scale ML model serving pods based on CPU, memory, and custom metrics like request rate.

**Answer:** True

**Explanation:**
HPA scales the number of pod replicas based on observed metrics. Out of the box it supports CPU and memory; through the `external` and `custom` metrics APIs (often backed by Prometheus Adapter or KEDA) it can also scale on RPS, queue depth, GPU utilization, or any custom signal. This makes HPA suitable for bursty inference workloads.

**Common Mistakes:**
- Answering False often comes from thinking HPA only supports CPU; the custom-metrics API has been GA for many releases.

**Related Material:** lessons/mod-06-mlops/04-model-deployment-strategies.md

---

## Question 11
**Q:** Which model serving protocol is typically faster for high-throughput scenarios?

**Answer:** B) gRPC

**Explanation:**
gRPC uses HTTP/2 with binary Protocol Buffers, supports multiplexed streams over a single connection, and avoids JSON serialization overhead. For high-throughput, low-latency model serving — especially internal service-to-service traffic — it generally outperforms REST/JSON. Major model servers (Triton, TF Serving, TorchServe, KServe) expose gRPC endpoints for this reason.

**Common Mistakes:**
- Selecting A (REST) confuses ubiquity and ease of debugging with raw performance; REST/JSON has higher serialization cost.
- Selecting D (WebSockets) suits bidirectional streaming UIs but isn't the standard high-throughput inference protocol.

**Related Material:** lessons/mod-06-mlops/04-model-deployment-strategies.md

---
