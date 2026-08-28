# Module 101: Foundations — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-101-foundations/quizzes/module-quiz.md).
>
> **Academic integrity:** This file is for self-study after attempting the quiz. Do not use it as a primary reference while taking the quiz.

---

## Question 1
**Q:** What is the primary role of an ML Infrastructure Engineer?

**Answer:** B) Building and maintaining systems that enable ML model deployment and operation

**Explanation:**
ML Infrastructure Engineers are platform-builders, not model-builders. Their job is to create the systems — training clusters, serving infrastructure, data pipelines, monitoring — that let data scientists and ML engineers put models into production reliably. Achieving high model accuracy (A), inventing new algorithms (C), and extracting business insights (D) are the jobs of ML researchers, applied scientists, and data analysts respectively.

**Common Mistakes:**
- Picking A because "ML" appears in the title and assuming the role centers on training models; in reality, infrastructure engineers rarely train production models themselves.
- Conflating the role with a data scientist (D), which focuses on insight generation rather than system reliability and scalability.

**Related Material:** `lessons/mod-101-foundations/01-introduction.md`

---

## Question 2
**Q:** Which of the following is NOT a stage in the ML lifecycle?

**Answer:** C) Customer Acquisition

**Explanation:**
The ML lifecycle covers Data Collection → Training → Evaluation → Deployment → Monitoring → Retraining. Customer Acquisition is a marketing/business function, not a technical stage of building or operating ML systems. The other three options (A, B, D) are all canonical stages.

**Common Mistakes:**
- Picking D (Model Monitoring) because it happens "after" deployment and learners forget that monitoring is what enables retraining and drift detection.
- Reading "Customer Acquisition" as a stand-in for "user feedback collection" and assuming it belongs in the loop, when in fact feedback is captured during Monitoring.

**Related Material:** `lessons/mod-101-foundations/03-ml-infrastructure-basics.md`

---

## Question 3
**Q:** What is the main difference between MLOps and traditional DevOps?

**Answer:** B) MLOps includes data and model versioning in addition to code versioning

**Explanation:**
DevOps versions code and infrastructure. MLOps adds two more first-class artifacts: datasets and trained model weights. This is necessary because reproducing a model requires the exact code + exact data + exact training configuration, and because models drift even when code does not. CI/CD pipelines (C) are core to MLOps too, and MLOps is cloud-agnostic (D).

**Common Mistakes:**
- Choosing C because learners assume "Ops" practices look different enough that CI/CD must be replaced — in reality MLOps extends CI/CD with CT (continuous training).
- Choosing D because most tutorials happen on cloud platforms, leading to the false belief that MLOps is cloud-only.

**Related Material:** `lessons/mod-101-foundations/03-ml-infrastructure-basics.md`

---

## Question 4
**Q:** [Short Answer] Explain why model retraining is necessary in production ML systems.

**Answer:** Model performance degrades over time due to (1) data drift — the distribution of input features changes (e.g., user behavior shifts, new product categories appear), and (2) concept drift — the relationship between inputs and the target variable changes (e.g., what predicted fraud last year no longer predicts it today). Periodic retraining on fresh, representative data restores accuracy and keeps the model aligned with current real-world conditions.

**Explanation:**
A complete answer must mention both drift types or at least the general concept of "the world changes after training." Simply saying "to improve accuracy" is incomplete — accuracy on the original test set does not degrade; accuracy on *new* production data does, and that is the entire motivation for retraining.

**Common Mistakes:**
- Answering "to make the model more accurate over time," which conflates retraining with additional training epochs on the same dataset.
- Forgetting to distinguish data drift from concept drift, or treating "drift" as a single undifferentiated concept.

**Related Material:** `lessons/mod-101-foundations/03-ml-infrastructure-basics.md`

---

## Question 5
**Q:** Which salary range is typical for an entry-level AI Infrastructure Engineer in the US?

**Answer:** B) $107k - $230k

**Explanation:**
The role sits at the intersection of distributed systems, ML, and cloud — three high-demand skill areas — so even entry-level total compensation in major US tech hubs lands in the $107k–$230k range (base + equity + bonus). $50k–$80k (A) and $40k–$60k (D) reflect generic IT or junior software roles, and $300k–$500k (C) is staff/principal-level compensation.

**Common Mistakes:**
- Picking A or D because the candidate compares to generalist software engineering or sysadmin salaries rather than the AI infrastructure niche.
- Picking C by confusing entry-level with senior or staff-level FAANG compensation.

**Related Material:** `lessons/mod-101-foundations/01-introduction.md`

---

## Question 6
**Q:** What is the main difference between PyTorch and TensorFlow computation graphs?

**Answer:** B) PyTorch uses dynamic graphs, TensorFlow traditionally uses static graphs

**Explanation:**
PyTorch builds the computation graph on-the-fly as Python executes (define-by-run), which makes debugging and dynamic control flow natural. TensorFlow 1.x required you to define a static graph first and then run it inside a session; TF 2.x added eager execution to match PyTorch's ergonomics, but the historical and architectural distinction still drives many design choices.

**Common Mistakes:**
- Reversing the two and picking A — a very common mix-up because "Tensor" sounds more "flowing" and "dynamic."
- Picking C because TF 2.x's eager mode looks like PyTorch, ignoring that the underlying graph philosophies still differ (TF can compile to a static graph with `@tf.function`).

**Related Material:** `lessons/mod-101-foundations/04-ml-frameworks.md`

---

## Question 7
**Q:** Which method is recommended for saving PyTorch models in production?

**Answer:** B) `torch.save(model.state_dict(), 'model.pth')` - save state dictionary

**Explanation:**
Saving only the `state_dict` (the tensor weights) decouples weights from the Python class definition, makes the file roughly half the size, and lets you load the same weights into a refactored or subclassed model. Saving the whole model object (A) pickles the class path, so any rename or refactor breaks loading. Manual file copying (C) and raw pickle (D) are unsafe and non-portable.

**Common Mistakes:**
- Choosing A because it looks simpler — until the model class moves to a different module and `torch.load` fails with an import error.
- Choosing D because "PyTorch uses pickle under the hood" — true, but raw pickle bypasses PyTorch's tensor-aware serialization and security checks.

**Related Material:** `lessons/mod-101-foundations/04-ml-frameworks.md`

---

## Question 8
**Q:** Why is it critical to call `model.eval()` before inference in PyTorch?

**Answer:** B) It changes behavior of layers like dropout and batch normalization

**Explanation:**
The `model.eval()` call flips an internal `training` flag that Dropout and BatchNorm (and a few other layers) check. In evaluation mode, Dropout becomes a no-op and BatchNorm uses the running mean/variance learned during training instead of recomputing statistics from the current batch. Skipping it produces non-deterministic and incorrect predictions, especially with small batch sizes.

**Common Mistakes:**
- Picking A because users notice inference in evaluation mode is sometimes faster (it can be, due to disabled dropout) and incorrectly assume speed is the purpose.
- Picking C because the model "still produces an output" without it — but the output is silently wrong, which is the most dangerous kind of bug.

**Related Material:** `lessons/mod-101-foundations/04-ml-frameworks.md`

---

## Question 9
**Q:** [Multiple Choice] Which of the following techniques can optimize model inference? (Select all that apply)

**Answer:** A, B, C, E (Quantization, TorchScript compilation, Batch inference, Model warm-up)

**Explanation:**
Quantization (A) shrinks weights from FP32 to INT8, giving ~4× smaller models and 2–4× faster inference on supported hardware. TorchScript (B) compiles Python models to a graph IR for 10–30% speedups and Python-free deployment. Batch inference (C) amortizes GPU launch overhead across many requests. Warm-up (E) primes JIT and CUDA caches so the first user request is not 10–100× slower than steady-state. Running inference in training mode (D) is wrong — it keeps Dropout active and recomputes BatchNorm stats, hurting both speed *and* correctness.

**Common Mistakes:**
- Including D because "training mode does more work, therefore it must be more thorough" — the opposite is true for inference.
- Omitting E (warm-up) because it doesn't change steady-state latency, missing that p99 latency on cold starts is what users actually feel.

**Related Material:** `lessons/mod-101-foundations/04-ml-frameworks.md`, `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 10
**Q:** What is ONNX and why is it useful?

**Answer:** B) A universal model format for framework-agnostic deployment

**Explanation:**
ONNX (Open Neural Network Exchange) defines a portable computation-graph format plus a shared set of operators, letting you train in PyTorch and serve in TensorFlow, TensorRT, or ONNX Runtime. The ONNX Runtime is heavily optimized and often outperforms the source framework on the same hardware, which is why ONNX is the de facto interchange format for production serving.

**Common Mistakes:**
- Picking A because ONNX has Python bindings and tutorials often look like training code — but ONNX itself is a *format*, not a trainer.
- Picking D, confusing the format name with an architecture name like ResNet or Transformer.

**Related Material:** `lessons/mod-101-foundations/04-ml-frameworks.md`

---

## Question 11
**Q:** Which cloud service model gives you the most control over the infrastructure?

**Answer:** C) IaaS (Infrastructure as a Service)

**Explanation:**
IaaS exposes raw compute, storage, and networking primitives (VMs, disks, VPCs) and lets you choose the OS, runtime, middleware, and application. As you move up the stack — PaaS, then SaaS — the provider takes over more layers, which trades control for convenience. FaaS (D) sits above PaaS and gives you the least control of all.

**Common Mistakes:**
- Confusing "managed by the provider" (which makes SaaS attractive) with "most control" — these are opposites. Picking A (SaaS) inverts the question.
- Picking B (PaaS) because it sounds like the "platform" gives you platform-level control, when in fact PaaS hides the OS and runtime layers.

**Related Material:** `lessons/mod-101-foundations/05-cloud-intro.md`

---

## Question 12
**Q:** What is a spot/preemptible instance and when should you use it?

**Answer:** B) A discounted instance that can be terminated with short notice, suitable for non-critical workloads

**Explanation:**
Cloud providers sell spare capacity at 60–90% discount in exchange for the right to reclaim it with 30 seconds to 2 minutes notice. They are perfect for fault-tolerant workloads — batch ETL, distributed training with checkpointing, CI runners — but unsafe for stateful, low-latency serving where eviction means downtime.

**Common Mistakes:**
- Picking A because the word "spot" sounds premium or VIP, when in fact it's the opposite: the cheapest and least reliable tier.
- Picking C because some learners confuse spot instances with scheduled scaling rules.

**Related Material:** `lessons/mod-101-foundations/05-cloud-intro.md`

---

## Question 13
**Q:** [Multiple Choice] Which of the following are valid strategies for cloud cost optimization? (Select all that apply)

**Answer:** A, B, D, E (Right-sizing, Spot instances, Reserved instances, Auto-shutdown of dev environments)

**Explanation:**
Right-sizing (A) eliminates the most common waste: instances provisioned for peak load that sit at 5% CPU. Spot (B) saves up to 90% on interruptible work. Reserved instances or Savings Plans (D) give 30–75% off for 1–3 year commitments on steady baseline workloads. Auto-shutdown (E) of dev/staging environments overnight and on weekends often halves their bill. Running everything 24/7 "for consistency" (C) is the textbook anti-pattern that motivates all four correct answers.

**Common Mistakes:**
- Including C because "always-on equals reliable" — reliability comes from redundancy and auto-scaling, not from never turning anything off.
- Omitting D because "commitment" feels risky, missing that reserved capacity is essentially free savings on workloads you'd run anyway.

**Related Material:** `lessons/mod-101-foundations/05-cloud-intro.md`

---

## Question 14
**Q:** What is the typical cost structure for cloud GPU instances?

**Answer:** B) ~$1-5 per hour depending on GPU type

**Explanation:**
On-demand GPU instances are priced per second/hour: entry-tier (T4, K80) around $0.50–$1.50/hr, mid-tier (V100, A10) around $2–$4/hr, and high-end (A100, H100) $4–$8+/hr on most providers. Flat monthly pricing (C) and pay-per-prediction (D) are not how raw GPU compute is sold, and nothing on a major cloud is free at any meaningful scale (A).

**Common Mistakes:**
- Picking C because learners are used to SaaS subscription pricing and project that model onto raw IaaS GPU rental.
- Picking D, confusing inference *endpoints* (which sometimes meter per request) with raw GPU VM rental.

**Related Material:** `lessons/mod-101-foundations/05-cloud-intro.md`

---

## Question 15
**Q:** Which cloud provider is generally considered best for ML/AI workloads?

**Answer:** D) All three are equally suitable depending on requirements

**Explanation:**
There is no single "best." AWS has the deepest service catalog and the most mature ML stack (SageMaker, Bedrock). GCP leads on ML-native primitives (TPUs, Vertex AI, BigQuery + BQML, GKE). Azure wins on enterprise/Microsoft integration (Active Directory, Office, Azure OpenAI). The right choice depends on existing contracts, team expertise, regional availability, and which managed services you'll actually use.

**Common Mistakes:**
- Picking B because GCP markets aggressively to the ML community, ignoring that AWS hosts the majority of production ML workloads.
- Picking A out of "AWS is the biggest, so it must be best" reasoning, which ignores TPU access and enterprise integration trade-offs.

**Related Material:** `lessons/mod-101-foundations/05-cloud-intro.md`

---

## Question 16
**Q:** What is the main difference between training and serving workloads?

**Answer:** B) Training optimizes for accuracy, serving optimizes for latency and throughput

**Explanation:**
Training is a batch job: it runs for hours or days over huge datasets and is judged by the final model's accuracy. Serving is an online system: it answers individual requests in milliseconds and is judged by p99 latency, requests per second, and uptime. The two have different SLAs, different hardware sweet spots, and different scaling strategies.

**Common Mistakes:**
- Picking C because GPUs are heavily associated with training, missing that serving also frequently uses GPUs (or specialized accelerators like Inferentia/TPU) for latency-sensitive workloads.
- Picking A, reversing the directional intuition — training is the *slower* of the two per sample.

**Related Material:** `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 17
**Q:** What is batch inference and what are its benefits?

**Answer:** B) Processing multiple inference requests together; improves GPU utilization and throughput

**Explanation:**
GPUs are massively parallel; running one request at a time leaves most cores idle. Dynamic batching collects N concurrent requests, runs them through the model as a single tensor, and splits the outputs back out — yielding 3–10× higher throughput at the cost of a small queueing delay. This is why nearly every production serving stack (Triton, vLLM, TGI) has a batcher.

**Common Mistakes:**
- Picking A, confusing batch *inference* with batch *training* (training multiple models in parallel).
- Picking C, confusing batching with A/B model version routing, which is a serving-layer traffic-splitting concern.

**Related Material:** `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 18
**Q:** In a blue-green deployment strategy, what happens during the switch?

**Answer:** B) Traffic is instantly switched from old version to new version

**Explanation:**
Blue-green keeps two complete production environments side-by-side. The old (blue) serves all traffic while the new (green) is deployed and warmed. A single router flip sends 100% of traffic to green; if anything misbehaves, flip back to blue. This gives near-zero rollback time at the cost of running double the infrastructure during cutover.

**Common Mistakes:**
- Picking A — gradual traffic shifting is canary deployment, not blue-green. This is the single most common confusion in serving strategy questions.
- Picking D, which describes an in-place restart and defeats the entire point of having two environments.

**Related Material:** `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 19
**Q:** [Short Answer] Explain why model warm-up is important before serving production traffic.

**Answer:** The first inference after a process starts is typically 10–100× slower than steady-state because the framework must JIT-compile kernels, initialize CUDA contexts and cuDNN/cuBLAS handles, allocate and page in GPU memory, and load weights from disk into device memory. Running a few dummy inferences during startup forces all of this lazy initialization to happen before real users hit the endpoint, so production requests see consistent low latency from the very first call.

**Explanation:**
A strong answer mentions both the *cause* (JIT/CUDA/memory initialization) and the *consequence* (cold p99 latency spikes). Without warm-up, every container restart, autoscale event, or deployment causes the first few real users to experience seconds-long delays — which dominates p99 metrics and SLO violations even when the average looks fine.

**Common Mistakes:**
- Saying only "to make it faster" without explaining that steady-state speed is unchanged; warm-up specifically fixes the *first request* problem.
- Forgetting to mention autoscaling/restart implications, missing that warm-up is critical in any elastic deployment, not just at initial launch.

**Related Material:** `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 20
**Q:** Which deployment strategy allows testing a new model with real traffic without impacting users?

**Answer:** C) Shadow deployment

**Explanation:**
In a shadow (or "dark launch") deployment, production traffic is duplicated and sent to both the current model and the new candidate, but only the current model's response is returned to the user. The candidate's predictions are logged and compared offline — perfect for validating latency, resource usage, and prediction quality on real inputs with zero user risk. Canary (B) and blue-green (A) both expose at least some real users to the new model.

**Common Mistakes:**
- Picking B (canary) because it's the most familiar "safe" strategy, missing that canary still routes a small percentage of real users to the new model.
- Picking A (blue-green) because both environments exist simultaneously, forgetting that blue-green flips all traffic at once rather than mirroring it.

**Related Material:** `lessons/mod-101-foundations/06-model-serving-basics.md`

---

## Question 21
**Q:** What is the primary benefit of multi-stage Docker builds?

**Answer:** B) Smaller final image size by excluding build dependencies

**Explanation:**
A multi-stage Dockerfile uses one stage with the full toolchain (compilers, dev headers, build caches) and a second, minimal stage that `COPY --from=builder` only the compiled artifacts. The final image ships without gcc, build-essential, pip caches, or source trees, often shrinking from 2 GB to a few hundred MB. Smaller images mean faster pulls, smaller attack surface, and lower registry costs.

**Common Mistakes:**
- Picking A because "more stages = faster" sounds intuitive — in reality, multi-stage builds are about *layer composition*, not parallelism.
- Picking C, conflating "multiple stages" with "multiple base images for isolation"; security gains come from the smaller surface area, not from base image diversity.

**Related Material:** `lessons/mod-101-foundations/07-docker-basics.md`

---

## Question 22
**Q:** Why should you run containers as non-root users?

**Answer:** B) It's a security best practice to minimize potential damage from container escape

**Explanation:**
Containers share the host kernel; a kernel exploit or misconfigured mount can let a process inside the container reach the host. If that process is root inside the container, by default it maps to UID 0 on the host (unless user-namespace remapping is enabled), so an escape grants root-level damage. Running as a dedicated non-root UID limits the blast radius to whatever that low-privilege user can do.

**Common Mistakes:**
- Picking D because Docker *can* be run rootless, but it does not *require* non-root users inside containers; the requirement is a hardening recommendation, not an enforcement.
- Picking A based on a vague intuition that "non-root is lightweight," which is unrelated to performance.

**Related Material:** `lessons/mod-101-foundations/07-docker-basics.md`

---

## Question 23
**Q:** [Multiple Choice] Which Dockerfile instructions affect layer caching? (Select all that apply)

**Answer:** B, C, E (RUN, COPY, ENV)

**Explanation:**
Each `RUN`, `COPY`, `ADD`, and `ENV` produces a new filesystem layer keyed on the instruction text and the files involved. Change a copied file and every layer after that `COPY` is invalidated; change an `ENV` and every later `RUN` that references it is invalidated. `FROM` selects the base layers but doesn't itself create your cached layers in the same way, and `CMD`/`ENTRYPOINT` only set metadata — they don't build filesystem layers and so don't affect the build cache.

**Common Mistakes:**
- Including A (FROM) because it does pull layers — but those are the base image's layers, not layers your build cache controls; the answer key treats it as not directly part of *your* cache logic.
- Including D (CMD), assuming any instruction that "changes the container" must affect caching — but CMD only updates image metadata.

**Related Material:** `lessons/mod-101-foundations/07-docker-basics.md`

---

## Question 24
**Q:** What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?

**Answer:** B) CMD provides default arguments that can be overridden; ENTRYPOINT sets the main command

**Explanation:**
`ENTRYPOINT` defines the executable that always runs when the container starts. `CMD` provides default arguments (or a default command if no ENTRYPOINT is set) that the user can override with `docker run image <args>`. The idiomatic combination is `ENTRYPOINT ["python"]` + `CMD ["app.py"]`, which runs `python app.py` by default but `python other.py` if you pass `other.py` at the CLI.

**Common Mistakes:**
- Picking A because tutorials always show `CMD`, leading to the assumption that it's mandatory — neither is strictly required, but at least one must exist or the container has nothing to run.
- Picking D, inventing a build-time vs runtime distinction; both directives only take effect at container *runtime*.

**Related Material:** `lessons/mod-101-foundations/07-docker-basics.md`

---

## Question 25
**Q:** How do you enable GPU access in Docker containers?

**Answer:** B) Install NVIDIA Container Toolkit and use --gpus flag

**Explanation:**
The NVIDIA Container Toolkit installs a runtime hook that, when you pass `--gpus all` (or `--gpus '"device=0,1"'`), mounts the host's NVIDIA driver libraries and `/dev/nvidia*` device nodes into the container. The container itself needs only the CUDA *user-space* libraries (commonly via an `nvidia/cuda` base image); the kernel driver always lives on the host.

**Common Mistakes:**
- Picking A because containers feel like "VMs that just work" — by default they have no GPU access at all.
- Picking D, attempting manual `/dev` mounts, which skips the driver-library mounting that the Container Toolkit handles automatically and usually results in CUDA initialization errors.

**Related Material:** `lessons/mod-101-foundations/07-docker-basics.md`

---

## Question 26
**Q:** Why is FastAPI particularly well-suited for ML model serving?

**Answer:** B) It's fast (async), has auto-documentation, and built-in validation

**Explanation:**
FastAPI is built on Starlette + Pydantic and runs on the async ASGI standard, giving it Node-class concurrency for I/O-bound workloads. Type-annotated request and response models become both runtime validators and an automatically generated OpenAPI/Swagger UI — three features that together remove a huge amount of boilerplate around inference endpoints.

**Common Mistakes:**
- Picking A, an absolutist claim ("the only framework"); Flask, Django, BentoML, TorchServe, and Triton can all serve models — FastAPI is preferred, not exclusive.
- Picking D, which is true but trivial (many frameworks are free and open source) and misses the actual technical advantages.

**Related Material:** `lessons/mod-101-foundations/08-api-development.md`

---

## Question 27
**Q:** What does Pydantic provide in FastAPI applications?

**Answer:** B) Automatic request/response validation based on Python types

**Explanation:**
Pydantic turns Python type hints (`int`, `str`, `List[float]`, custom `BaseModel` classes) into runtime validators. FastAPI uses these to parse JSON bodies, coerce types, enforce constraints (min/max, regex, enum), and return structured 422 errors when validation fails — all without you writing manual `if not isinstance(...)` checks.

**Common Mistakes:**
- Picking A, confusing Pydantic with SQLAlchemy or another ORM; Pydantic does not talk to databases.
- Picking C, conflating validation of *inputs to a model* with *hosting the model itself* (which is FastAPI's responsibility, not Pydantic's).

**Related Material:** `lessons/mod-101-foundations/08-api-development.md`

---

## Question 28
**Q:** [Short Answer] What is the difference between a 400 and 500 HTTP status code, and when would you use each?

**Answer:** 4xx codes signal *client* errors — the request itself is invalid — and 400 specifically means the request is malformed or contains invalid data (missing fields, wrong types, failed validation). 5xx codes signal *server* errors — the request was valid but the server failed to process it — and 500 is the generic "unexpected exception" code used for uncaught errors, database failures, or downstream service outages. Practically: use 400 (or 422) when the caller must fix their request before retrying; use 500 when the caller did nothing wrong and a retry might succeed once the server recovers.

**Explanation:**
The strongest answers explicitly name the client-vs-server distinction and give a concrete trigger for each. A weak answer just says "400 is bad request, 500 is server error" without explaining who is responsible for fixing the problem, which is the operational point of the distinction.

**Common Mistakes:**
- Returning 500 for validation errors — this misleads the caller into retrying the same malformed request and pollutes server-error dashboards/alerts.
- Returning 400 for downstream service failures, masking real outages from monitoring systems that key off the 5xx error rate.

**Related Material:** `lessons/mod-101-foundations/08-api-development.md`

---

## Question 29
**Q:** What is the purpose of Prometheus metrics in an ML serving API?

**Answer:** B) To monitor system performance, request rates, latencies, and errors

**Explanation:**
Prometheus scrapes a `/metrics` endpoint exposing counters (requests, errors), histograms (latency distributions for p50/p95/p99), and gauges (in-flight requests, GPU memory). These feed Grafana dashboards and Alertmanager rules, enabling SLO tracking, capacity planning, and on-call alerting. For ML specifically, you also expose model-level metrics like prediction confidence distributions and feature drift indicators.

**Common Mistakes:**
- Picking C, confusing observability (Prometheus) with the control loop that *acts on* observability (HPA, KEDA, custom autoscalers).
- Picking D, confusing metrics with OpenAPI documentation — both are introspection endpoints, but they serve completely different audiences.

**Related Material:** `lessons/mod-101-foundations/08-api-development.md`

---

## Question 30
**Q:** Which of the following is a best practice for production API design?

**Answer:** B) Use generic error messages and log detailed errors server-side

**Explanation:**
Returning stack traces, SQL errors, file paths, or library versions to a caller hands attackers a free reconnaissance tool and often exposes secrets in connection strings or environment dumps. The correct pattern is to return a stable, generic error code/message to the client (e.g., `{"error": "internal_error", "request_id": "abc123"}`) while logging the full exception, stack trace, and context server-side, correlated by request ID.

**Common Mistakes:**
- Picking A because "more information is friendlier to debug" — true for *internal* tools, dangerous for *public* APIs.
- Picking D, treating logging as a performance cost; structured async logging has negligible overhead and is non-negotiable for production debugging.

**Related Material:** `lessons/mod-101-foundations/08-api-development.md`

---
