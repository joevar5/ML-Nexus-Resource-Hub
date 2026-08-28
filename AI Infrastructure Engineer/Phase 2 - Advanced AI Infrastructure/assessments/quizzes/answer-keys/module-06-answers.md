# Module 06: MLOps — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-06-mlops/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the primary purpose of MLOps?

**Answer:** B) To automate and streamline the ML lifecycle from development to production

**Explanation:**
MLOps applies DevOps principles to machine learning, providing the practices, tooling, and culture that move models reliably from notebook to production and keep them healthy there. It spans data versioning, training, validation, deployment, monitoring, and governance, so teams can iterate faster with fewer regressions. The goal is reliability and velocity of ML systems, not improving any single model's score.

**Common Mistakes:**
- Choosing A ("replace data scientists") misunderstands MLOps — it augments data scientists by removing manual deployment and monitoring toil, not their judgment.
- Choosing C ("make models more accurate") confuses MLOps (the operational discipline) with modeling research; accuracy improvements come from data and modeling work, not from MLOps itself.

**Related Material:** lessons/mod-06-mlops/01-introduction-to-mlops.md

---

## Question 2
**Q:** Which of the following is NOT a core component of the MLOps lifecycle?

**Answer:** C) Frontend development

**Explanation:**
The MLOps lifecycle covers data management (including versioning), model development and training, evaluation, deployment, and ongoing monitoring. Frontend development may consume model predictions but is outside the MLOps discipline itself. Data versioning, model training, and model monitoring are all canonical stages.

**Common Mistakes:**
- Selecting A ("data versioning") overlooks that reproducibility — and therefore data versioning — is a foundational MLOps capability.
- Selecting D ("model monitoring") ignores that production monitoring for drift and performance regressions is one of the most distinctive MLOps stages.

**Related Material:** lessons/mod-06-mlops/01-introduction-to-mlops.md

---

## Question 3
**Q:** (True/False) Experiment tracking tools like MLflow only store model training metrics and cannot store artifacts like model files.

**Answer:** False

**Explanation:**
MLflow Tracking records parameters, metrics, tags, and arbitrary artifacts including serialized models, plots, datasets, and config files. The MLflow Models component additionally standardizes a packaging format so logged models can be reloaded and deployed. Limiting MLflow to "just metrics" misses the artifact store, which is central to reproducibility.

**Common Mistakes:**
- Answering True assumes tracking servers are write-only metric databases; in practice, `mlflow.log_artifact` and flavor-specific `log_model` calls persist files alongside the run.

**Related Material:** lessons/mod-06-mlops/02-mlflow-experiment-tracking.md

---

## Question 4
**Q:** In MLflow, what is the purpose of a "Run"?

**Answer:** B) To represent a single execution of model training code

**Explanation:**
A Run is the unit of execution in MLflow Tracking — one invocation of training (or evaluation) code that logs its parameters, metrics, tags, and artifacts under a parent Experiment. Runs make individual training jobs comparable and reproducible. Multiple Runs under one Experiment let you compare hyperparameter sweeps or code changes side-by-side.

**Common Mistakes:**
- Selecting A ("execute a deployed model") confuses tracking with serving — production inference is handled by MLflow Models / Serving, not by a Tracking Run.
- Selecting D ("version control your code") misattributes a Git/DVC responsibility to MLflow; Runs can record a Git commit, but they don't replace source control.

**Related Material:** lessons/mod-06-mlops/02-mlflow-experiment-tracking.md

---

## Question 5
**Q:** What information should be logged during ML experiments? (Select all that apply)

**Answer:** E) All of the above (Hyperparameters, Evaluation metrics, Training duration, Dataset version)

**Explanation:**
Reproducibility and meaningful comparison require capturing inputs (hyperparameters, dataset version), outputs (evaluation metrics), and execution context (training duration, hardware, code commit). Skipping any of these makes it hard to explain why a run won or to recreate it later. The cost of logging is tiny compared to the cost of an undebuggable model.

**Common Mistakes:**
- Picking only A and B and skipping dataset version is a frequent mistake; without dataset versioning you can't tell whether a metric change came from code or data.
- Skipping training duration ignores resource and cost signals that matter for production planning.

**Related Material:** lessons/mod-06-mlops/02-mlflow-experiment-tracking.md

---

## Question 6
**Q:** What is the purpose of experiment tagging in MLflow?

**Answer:** A) To organize and filter experiments

**Explanation:**
Tags are arbitrary key-value metadata attached to Runs or Experiments (e.g., `team=ranking`, `dataset=v3`, `purpose=baseline`) so you can search, group, and filter through the UI or API. They are organizational metadata only; they do not influence model behavior or deployment.

**Common Mistakes:**
- Choosing B ("improve model accuracy") confuses metadata labels with training inputs — tags don't enter the model.
- Choosing D ("automatically deploy models") confuses tagging with the Model Registry's stage transitions, which are the actual deployment-related labels.

**Related Material:** lessons/mod-06-mlops/02-mlflow-experiment-tracking.md

---

## Question 7
**Q:** (Short Answer) Explain the difference between logging parameters and logging metrics in MLflow.

**Answer:** Parameters are input configuration values (e.g., learning rate, batch size, model type) logged once at the start of a Run, typically via `mlflow.log_param`. Metrics are measured output values (e.g., accuracy, loss, F1) logged via `mlflow.log_metric`, can be logged repeatedly with a step index, and form time-series you can plot over training epochs.

**Explanation:**
The split matters because parameters describe the experiment's setup while metrics describe its outcome. Parameters are immutable once set on a Run; metrics are append-only series. Mixing them up makes runs hard to compare and breaks tooling that expects scalar params and time-series metrics.

**Common Mistakes:**
- Saying both are "just values" misses the time-series nature of metrics and the configuration role of parameters.
- Forgetting that metrics support a `step` argument leads people to overwrite values instead of building training curves.

**Related Material:** lessons/mod-06-mlops/02-mlflow-experiment-tracking.md

---

## Question 8
**Q:** What problem does a feature store solve?

**Answer:** D) All of the above (Training-serving skew, Feature reusability across teams, Point-in-time correctness)

**Explanation:**
Feature stores centralize feature definitions and computation so the same logic produces features for both training and serving, eliminating skew. They expose those features as a shared catalog so multiple teams can discover and reuse them. They also enforce point-in-time joins to prevent label leakage when assembling historical training sets.

**Common Mistakes:**
- Picking only A (training-serving skew) overlooks reuse and point-in-time correctness, which are equally important reasons feature stores exist.
- Picking only C misses that skew prevention is arguably the headline value proposition.

**Related Material:** lessons/mod-06-mlops/04-feature-stores-engineering.md

---

## Question 9
**Q:** What is the difference between offline and online feature stores?

**Answer:** D) All of the above (Offline is for training / online for inference; offline is slower / online is faster; offline stores historical data / online stores recent fresh values)

**Explanation:**
Offline stores (e.g., warehouses like BigQuery, Snowflake, or Parquet on object storage) hold full feature history for training and backfills, optimized for throughput over latency. Online stores (e.g., Redis, DynamoDB, Cassandra) hold the latest feature values keyed by entity ID for low-latency lookups during real-time inference. A feature store synchronizes the two so the same logical feature is available in both shapes.

**Common Mistakes:**
- Picking only A (training vs inference) is technically correct but ignores the latency and data-shape distinctions that drive the storage choice.
- Picking only B (slow vs fast) misses the equally important difference in what data each tier retains.

**Related Material:** lessons/mod-06-mlops/04-feature-stores-engineering.md

---

## Question 10
**Q:** (True/False) Feature stores automatically guarantee that features used in training match features used in production serving.

**Answer:** True

**Explanation:**
Because feature definitions and transformation logic live in one place and are materialized to both the offline and online stores, the same code path produces values for training and serving. This is the structural mechanism by which feature stores eliminate training-serving skew. It's the headline guarantee that justifies adopting a feature store.

**Common Mistakes:**
- Answering False often comes from conflating skew caused by upstream data quality issues (still possible) with definitional skew (which the store eliminates).

**Related Material:** lessons/mod-06-mlops/04-feature-stores-engineering.md

---

## Question 11
**Q:** In Feast, what is a Feature View?

**Answer:** B) A definition of a group of related features with a common data source

**Explanation:**
A `FeatureView` in Feast declares a logical group of features derived from a single data source, tied to one or more entities, with a TTL and schema. It is the unit Feast uses for materialization to the online store and for point-in-time joins for training data. Feature Views are definitions, not UIs or caches.

**Common Mistakes:**
- Selecting A ("UI for viewing features") confuses the configuration object with a visualization surface.
- Selecting D ("cache for frequently accessed features") confuses Feature Views with the online store, which is the actual low-latency cache.

**Related Material:** lessons/mod-06-mlops/04-feature-stores-engineering.md

---

## Question 12
**Q:** How does CI/CD for ML differ from traditional software CI/CD?

**Answer:** D) All of the above (Includes data validation, tests model performance not just code correctness, versions data and models in addition to code)

**Explanation:**
ML pipelines fail not only from code defects but also from data quality regressions, so validation has to extend to schema and distribution checks. Tests must verify model behavior on held-out and slice-level data, not only unit-test code paths. And reproducibility requires versioning the trio of code + data + model artifacts together.

**Common Mistakes:**
- Choosing only A overlooks that "the model itself is now part of the build artifact" is the deeper difference.
- Choosing only C misses that the actual test surface (data and model behavior) is what makes ML CI/CD distinctive.

**Related Material:** lessons/mod-06-mlops/05-cicd-ml-models.md

---

## Question 13
**Q:** What should be included in automated ML tests? (Select all that apply)

**Answer:** E) All of the above (Data schema validation, Model performance thresholds, Inference latency checks, Model fairness tests)

**Explanation:**
A complete ML test suite checks inputs (schema and distribution), outputs (accuracy/precision/recall thresholds and slice metrics), runtime characteristics (latency and throughput SLOs), and responsible-AI properties (fairness across demographic slices). Skipping any one creates a class of bugs that won't be caught before production. Each should run automatically in the deployment pipeline.

**Common Mistakes:**
- Selecting only A and B treats ML tests as a beefier unit-test suite and forgets non-functional requirements like latency.
- Omitting fairness tests is a common oversight that surfaces only when production users complain.

**Related Material:** lessons/mod-06-mlops/05-cicd-ml-models.md

---

## Question 14
**Q:** In a deployment gate for ML models, what criteria might prevent a model from being deployed?

**Answer:** D) All of the above (Performance regression vs current prod, Exceeds latency thresholds, Fails fairness checks)

**Explanation:**
Deployment gates compare a candidate model against the current production champion across quality (no metric regression), service (latency/throughput within SLO), and responsibility (fairness, bias) dimensions. Any single failure should block promotion to keep production safe. Gates are the automated equivalent of a release checklist.

**Common Mistakes:**
- Picking only A treats deployment gates as a pure accuracy gate and ignores serving and ethical constraints.
- Picking only B treats them as a performance test, missing the model-quality and fairness dimensions.

**Related Material:** lessons/mod-06-mlops/05-cicd-ml-models.md

---

## Question 15
**Q:** (Short Answer) Explain what a "canary deployment" is and why it's useful for ML models.

**Answer:** A canary deployment routes a small fraction of live traffic (commonly 1–10%) to a new model version while the bulk of traffic continues to hit the existing version. You monitor the canary's business and operational metrics on real production traffic, then ramp up gradually or roll back fast if regressions appear. For ML models this is especially valuable because offline metrics often diverge from real user behavior, so canary exposure is the cheapest way to validate that an improved offline score actually helps in production.

**Explanation:**
Canary deployments blast-radius cap the impact of a bad model: at 5% traffic, even a catastrophic regression affects 5% of users for the rollout window. They also generate live signal — latency, error rate, business KPIs — that you cannot get from a staging environment. Combined with automated rollback on guardrail breach, they make ML rollouts safe to do frequently.

**Common Mistakes:**
- Confusing canary with blue/green (which is 0% then 100%, with no gradual ramp).
- Skipping the rollback half of the answer; canary without monitoring and rollback is just a slow deploy.

**Related Material:** lessons/mod-06-mlops/06-model-deployment-strategies.md

---

## Question 16
**Q:** Which deployment pattern is best for making predictions on large datasets that don't require immediate results?

**Answer:** B) Batch inference

**Explanation:**
Batch inference runs the model over a large dataset on a schedule (e.g., nightly), writes predictions to a store, and is consumed asynchronously. It maximizes throughput and hardware utilization per prediction and is the right choice when latency-to-prediction can be hours, not milliseconds. Examples include nightly recommendation refresh, lead scoring, or churn risk for a customer base.

**Common Mistakes:**
- Picking A (real-time serving) wastes infrastructure when there is no per-request latency requirement.
- Picking C (stream processing) is appropriate for continuous event streams, not for bulk-dataset scoring with no urgency.

**Related Material:** lessons/mod-06-mlops/06-model-deployment-strategies.md

---

## Question 17
**Q:** What is the primary benefit of model quantization?

**Answer:** B) Reduces model size and improves inference speed

**Explanation:**
Quantization converts model weights and/or activations from higher-precision floats (FP32) to lower-precision representations (FP16, INT8, sometimes INT4). The result is a smaller artifact, lower memory bandwidth, and faster matrix multiplies on hardware that supports the lower precision. Accuracy typically drops slightly, which is a tradeoff, not a goal.

**Common Mistakes:**
- Selecting A ("increases accuracy") inverts the tradeoff — quantization usually costs a little accuracy in exchange for size and speed.
- Selecting C ("easier to train") confuses post-training quantization with training optimizations; quantization is primarily an inference-time technique.

**Related Material:** lessons/mod-06-mlops/06-model-deployment-strategies.md

---

## Question 18
**Q:** (True/False) Horizontal Pod Autoscaling in Kubernetes can automatically scale ML model serving pods based on CPU, memory, and custom metrics like request rate.

**Answer:** True

**Explanation:**
HPA scales the number of pod replicas based on observed metrics. Out of the box it supports CPU and memory; through the `external` and `custom` metrics APIs (often backed by Prometheus Adapter or KEDA) it can also scale on RPS, queue depth, GPU utilization, or any custom signal. This makes HPA suitable for bursty inference workloads.

**Common Mistakes:**
- Answering False often comes from thinking HPA only supports CPU; the custom-metrics API has been GA for many releases.

**Related Material:** lessons/mod-06-mlops/06-model-deployment-strategies.md

---

## Question 19
**Q:** Which model serving protocol is typically faster for high-throughput scenarios?

**Answer:** B) gRPC

**Explanation:**
gRPC uses HTTP/2 with binary Protocol Buffers, supports multiplexed streams over a single connection, and avoids JSON serialization overhead. For high-throughput, low-latency model serving — especially internal service-to-service traffic — it generally outperforms REST/JSON. Major model servers (Triton, TF Serving, TorchServe, KServe) expose gRPC endpoints for this reason.

**Common Mistakes:**
- Selecting A (REST) confuses ubiquity and ease of debugging with raw performance; REST/JSON has higher serialization cost.
- Selecting D (WebSockets) suits bidirectional streaming UIs but isn't the standard high-throughput inference protocol.

**Related Material:** lessons/mod-06-mlops/06-model-deployment-strategies.md

---

## Question 20
**Q:** What is the purpose of calculating sample size before running an A/B test?

**Answer:** D) All of the above (Determine runtime, ensure power to detect minimum effect, avoid running longer than necessary)

**Explanation:**
Sample size calculation, given a baseline metric, minimum detectable effect, significance level, and desired power, tells you how many users (and roughly how many days) you need before results are meaningful. Without it you risk underpowered tests that miss real effects, or overlong tests that waste opportunity cost. It's the precondition for trustworthy experimentation.

**Common Mistakes:**
- Picking only A treats sample sizing as a scheduling exercise and forgets the statistical-power half.
- Picking only C frames it as cost optimization and ignores the false-negative risk of underpowered tests.

**Related Material:** lessons/mod-06-mlops/07-ab-testing-experimentation.md

---

## Question 21
**Q:** What is "Sample Ratio Mismatch" (SRM) and why is it concerning?

**Answer:** D) All of the above (Observed split deviates from expected ratio, can indicate bugs in assignment or logging, can invalidate results)

**Explanation:**
SRM is a chi-squared test (typically with p < 0.001) flagging that the actual user split between variants differs significantly from the intended ratio (e.g., 49.2/50.8 vs the configured 50/50 at large N). The usual root causes are bucketing bugs, biased exposure logging, bot filtering applied unevenly, or redirects that drop users. Because the populations are no longer comparable, any treatment-effect estimate from an SRM-affected experiment is suspect and the test should be halted and investigated before trusting results.

**Common Mistakes:**
- Picking only A describes the symptom but misses why practitioners panic about it.
- Picking only C treats SRM as a results problem rather than an instrumentation problem with a usually fixable root cause.

**Related Material:** lessons/mod-06-mlops/07-ab-testing-experimentation.md

---

## Question 22
**Q:** (True/False) In A/B testing, you should always wait until reaching statistical significance before making a deployment decision.

**Answer:** False

**Explanation:**
"Always wait for p<0.05" is wrong on several fronts. Sequential testing methods (mSPRT, group-sequential, Bayesian posterior) allow valid early stopping. Guardrail metrics may force a halt even if the primary metric isn't yet significant. And a statistically significant effect may still be too small to ship (practical vs statistical significance), while a non-significant effect with tight CIs around zero can justify shipping a neutral but cheaper variant.

**Common Mistakes:**
- Answering True ignores guardrail breaches that should stop a test immediately regardless of primary-metric significance.
- It also confuses statistical significance with decision-worthiness; tiny detected effects can be statistically real but practically meaningless.

**Related Material:** lessons/mod-06-mlops/07-ab-testing-experimentation.md

---

## Question 23
**Q:** What is a guardrail metric in A/B testing?

**Answer:** A) A metric that must not degrade beyond a threshold

**Explanation:**
Guardrail metrics protect against shipping a treatment that wins on the primary metric but hurts something else important — latency, error rate, revenue, retention, fairness. The test should ship only if the primary metric improves and no guardrail crosses its acceptable degradation threshold. They encode the constraints that "winning" must respect.

**Common Mistakes:**
- Selecting B ("primary success metric") confuses guardrails with the optimization target; they are the opposite — constraints, not objectives.
- Selecting D ("metric for traffic allocation") confuses guardrails with bucketing or allocation logic, which is unrelated.

**Related Material:** lessons/mod-06-mlops/07-ab-testing-experimentation.md

---

## Question 24
**Q:** What is the purpose of a Model Card?

**Answer:** A) To document model details, intended use, and limitations

**Explanation:**
A Model Card (Mitchell et al., 2019) is a short structured document describing a model's intended use, training data, evaluation results across slices, ethical considerations, and known limitations. It supports informed downstream use and is increasingly required by governance frameworks and regulators. Model Cards are documentation, not telemetry or tooling.

**Common Mistakes:**
- Selecting B ("track performance over time") confuses Model Cards with production monitoring dashboards.
- Selecting D ("version control models") confuses them with the Model Registry, which handles versioning and stage transitions.

**Related Material:** lessons/mod-06-mlops/08-best-practices-governance.md

---

## Question 25
**Q:** (Short Answer) Name three types of ML-specific technical debt and provide a brief example of each.

**Answer:** Three common categories from Sculley et al., "Hidden Technical Debt in Machine Learning Systems":

1. **Data debt** — unstable or undocumented input pipelines. Example: a feature consumed by a production model is silently rebuilt nightly by another team's ETL with no schema contract, so an upstream schema change breaks inference.
2. **Model debt** — entangled or redundant models. Example: three teams each ship their own churn model from overlapping features, none documented; retraining one shifts the joint distribution downstream (the CACE principle: "Changing Anything Changes Everything").
3. **Monitoring debt** — no observability for model behavior. Example: a recommender's CTR has been silently degrading for six weeks because no drift detector or performance dashboard exists, and the team only learns about it from a product manager.

Other acceptable examples include glue-code/pipeline-jungle debt, configuration debt (sprawling untested hyperparameter configs), feedback-loop debt (model outputs becoming next training inputs), and reproducibility debt (lost training datasets or environments).

**Explanation:**
ML systems accrue debt in ways traditional software doesn't because behavior depends on data and the world both change. Naming the categories helps teams prioritize: data and monitoring debt usually have higher blast radius than model debt because they cause silent failures. The seminal reference is Sculley et al.'s NeurIPS paper.

**Common Mistakes:**
- Listing only generic software debt (e.g., "untested code") without ML-specific framing misses the point of the question.
- Mentioning categories without concrete examples loses points on the rubric, which awards examples explicitly.

**Related Material:** lessons/mod-06-mlops/08-best-practices-governance.md

---
