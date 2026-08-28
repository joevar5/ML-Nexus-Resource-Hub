# Exercise 06: Managed ML Services Comparison

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 01-02

## Objective

Deploy the same model (the iris-api from earlier exercises) using each cloud's managed ML serving option (SageMaker Endpoint, Vertex AI Endpoint, Azure ML Online Endpoint) and produce a structured trade-off matrix. By the end you'll have hands-on evidence for the recurring "should we use managed or roll our own?" question.

## Why this matters

Managed services trade flexibility for operations. The trade is usually right for early-stage products, sometimes right for mature ones, almost never right at extreme scale or with regulated workloads. Engineers who've actually deployed to each can advise their team; engineers who've only read marketing pages cannot.

## Requirements

Deploy iris-api to all three services and produce:

### 1. Working deployments
- AWS **SageMaker Endpoint** with the model packaged via PyTorch container.
- GCP **Vertex AI Endpoint** with a custom container.
- Azure **ML Online Endpoint** with a managed online deployment.

### 2. Comparison matrix
A `COMPARISON.md` scoring each on:
- Cold-start latency (first request after deploy)
- Warm-request p50/p95 latency
- Cost per 1M predictions at 100 RPS sustained
- Cost when idle (instances vs serverless)
- Time to first successful deploy from a clean account
- Logs accessibility (1 = grep-friendly, 5 = enterprise hellscape)
- Metrics availability (which percentile/breakdown surfaces are first-party)
- Maximum payload size
- A/B testing primitive (built-in vs roll-your-own)
- Multi-model serving (multiple models per endpoint)
- BYO custom Docker support (1 = locked-down, 5 = full custom)

### 3. A short essay (~500 words)
`RECOMMENDATION.md` describing, for a hypothetical "fintech startup at 100k req/day, 10 engineers", which one you'd pick and why. Be specific. "It depends" is not an answer; describe the conditions.

## Step-by-step

### Step 1 — Plan (15 min)
Get accounts ready. Estimate cost: each endpoint runs for ~2 hours during testing. Budget $10-30 total. Set up billing alarms.

### Step 2 — SageMaker (45 min)
- Package the model: `tar -czf model.tar.gz model.joblib inference.py`
- Push to S3.
- Create SageMaker Model + EndpointConfig + Endpoint.
- Code can be Terraform (`aws_sagemaker_endpoint`) or boto3 script.
- Test: `boto3.client('sagemaker-runtime').invoke_endpoint(...)`.

### Step 3 — Vertex AI (45 min)
- Package as a custom container (FastAPI service exposing `/health`, `/predict`).
- Push to Artifact Registry.
- Upload Model + create Endpoint + deploy Model to Endpoint.
- Test: `from google.cloud import aiplatform; endpoint.predict(...)`.

### Step 4 — Azure ML Online (45 min)
- Create an MLflow-formatted model artifact (or BYO container).
- `az ml model create`, `az ml online-endpoint create`, `az ml online-deployment create`.
- Test: `az ml online-endpoint invoke --request-file ...`.

### Step 5 — Standardize the test harness (30 min)
Write `bench.py` that runs the same payload against each endpoint and produces the latency + cost numbers for the matrix.

### Step 6 — Fill in the matrix (30 min)
Run each test. Record cold-start (deploy, immediately invoke), warm latency (after 10 warmup calls), and use each cloud's pricing calculator for the cost columns.

### Step 7 — Write the recommendation (30 min)
Pick a position. Defend it.

### Step 8 — Tear everything down (15 min)
Endpoints incur cost while idle. Always destroy after testing.

## Deliverables

1. Three deployed endpoints (taken down after measurement).
2. `bench.py` harness (re-runnable).
3. `COMPARISON.md` populated matrix.
4. `RECOMMENDATION.md` essay with a specific pick + reasoning.
5. Optional: a 5-minute Loom or recorded demo walking through the matrix.

## Validation

- [ ] All three endpoints return valid predictions on the same input.
- [ ] Matrix has measured numbers (not "TBD") in every cell.
- [ ] Tear-down verified: no endpoint left running.
- [ ] Recommendation cites at least 3 concrete numbers from the matrix.

## Stretch goals

- Add **AWS Bedrock**, **Vertex AI Generative AI**, **Azure OpenAI** for a comparison of managed LLM endpoints.
- Add an open-source comparison row: BentoML or KServe self-deployed on the same cluster.
- Compute the **break-even point**: at what request rate does self-hosting beat managed by 30%?

## Common pitfalls

- **Cold-start vs warm-up confusion** — Cold = endpoint never received traffic. Warm = endpoint already provisioned + container loaded. Measure both.
- **Pricing calculators understate cost** — They don't include log storage, traffic egress, sidecar instances. Trust your bill, not the calculator.
- **SageMaker shutdown asymmetry** — Deleting the Endpoint stops the instance bill, but Models and EndpointConfigs remain (free, but cluttering). Clean them too.
- **Different metrics across clouds** — Each cloud exposes a different subset of percentiles by default. Normalize via your own histogram.

## Solutions

Reference deployments + benchmark in the engineer-solutions repo.
