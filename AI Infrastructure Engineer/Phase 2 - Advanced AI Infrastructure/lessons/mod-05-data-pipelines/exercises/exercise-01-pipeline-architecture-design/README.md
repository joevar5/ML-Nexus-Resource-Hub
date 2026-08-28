# Exercise 01: Data Pipeline Architecture Design

**Duration:** 3 hours
**Difficulty:** Beginner+
**Prerequisites:** None

## Objective

Design a data pipeline architecture for a realistic scenario: a recommendation system that needs daily training data (batch) and sub-minute user-event ingestion (streaming). Produce a complete design document covering ingestion, processing, storage, scheduling, observability, and failure handling — without writing any code.

## Why this matters

Most pipeline failures originate in design choices made before the first line of code. Engineers who can produce a written design that survives review reduce failed-launch rate by an order of magnitude.

## The scenario

**Product:** A homepage product-recommendation system for a mid-size e-commerce site.

**Data sources:**
- `events.click`, `events.view`, `events.purchase` — user behavior, ~10M/day
- `catalog.products` — product master data, daily snapshot, ~500K rows
- `users.profile` — user dimension, daily snapshot, ~5M rows
- `inventory.stock` — sub-minute updates needed for "out of stock" filtering

**Outputs:**
- Daily training dataset (`s3://recs/training/<date>/`) for offline model retraining
- Online feature store updates (Redis) for serving features
- Real-time "out of stock" cache updates (Redis pub/sub)

**Constraints:**
- 3-engineer team
- AWS-only
- Budget: $5K/month for data infra (excluding ML compute)
- 99% pipeline reliability SLO

## Requirements

Produce a single `DESIGN.md` (~3000 words + diagrams) covering:

### 1. Architecture diagram
A clear picture (ASCII or mermaid or excalidraw) showing every component and the flow of data between them.

### 2. Ingestion strategy
For each data source: how does it get into your pipeline? (S3 dropoff, Kinesis, CDC from Postgres, etc.) Justify each.

### 3. Processing layer
Batch (Spark/EMR? Glue? Airflow + Pandas?), streaming (Kinesis Analytics? Flink on EKS? Kafka Streams?). Justify.

### 4. Storage layers
Raw → Refined → Curated zones with technology + retention + access patterns for each.

### 5. Orchestration
Airflow / Prefect / Argo Workflows / dagster / etc. — pick + justify.

### 6. Observability
Metrics (what), logs (what), traces (where), alerts (which).

### 7. Failure handling
What happens when ingestion lags? When a transform fails? When the schema changes upstream? When a Spark job OOMs?

### 8. Cost model
Estimated monthly cost per component, totaling to a budget recommendation.

### 9. Migration path
Day 1 minimal viable pipeline → Day 30 full pipeline → Day 90 polished. What gets built first, what gets deferred.

### 10. Risk register
Top 5 things that could derail the rollout, with mitigation per risk.

## Step-by-step

### Step 1 — Inventory the requirements (30 min)
Re-read the scenario. List every functional + non-functional requirement explicitly.

### Step 2 — Sketch components (45 min)
Draw the high-level diagram. Don't optimize yet — just identify the boxes.

### Step 3 — Justify each technology choice (60 min)
For each component, write a 2-sentence justification. "We picked Glue because the team has it and the data volume is < 1TB/day where its serverless model works well; if volume grows to 10TB/day we'd switch to EMR on Spot."

### Step 4 — Failure mode analysis (45 min)
For each component, ask "what happens when this fails?" Document the answer. Include partial-failure modes (slow but not down).

### Step 5 — Cost spreadsheet (30 min)
Use AWS Pricing Calculator. Add 20% buffer.

### Step 6 — Migration plan (30 min)
Walk through the day-1 → day-90 progression. What's the minimum thing that produces value?

### Step 7 — Risk register (15 min)
Brainstorm: schema drift, vendor outages, team turnover, cost overrun, security incident.

## Deliverables

1. `DESIGN.md` (one file with all 10 sections + diagrams).
2. `COST.csv` cost spreadsheet.
3. `RISK_REGISTER.md` (can be inside the main doc).

## Validation

Have a colleague review and answer:
- [ ] Can a new engineer build it from this doc without asking design questions?
- [ ] Are the technology choices justified, not just listed?
- [ ] Are the failure modes specific (not "we'd handle errors")?
- [ ] Are the costs realistic vs the budget?
- [ ] Is there a clear day-1 minimum viable version?

## Stretch goals

- Add a **multi-region** version of the design.
- Add a **vendor-portability** analysis: how locked-in are you? What would migrating off cost?
- Run a **design review** with someone outside your team; document the feedback.

## Common pitfalls

- **Over-engineering** — Day-90 design pretending to be day-1. The "minimum first" discipline matters.
- **Technology shopping** — Picking 5 distinct stream processors. Pick one, use it everywhere it works.
- **Ignoring data quality at design time** — Adding GE later is 3× the work of designing for it.
- **Estimating cost from on-demand prices only** — Most production pipelines benefit from reserved/spot/committed-use. Model both.
- **No SLOs** — A 99% reliability SLO is your North Star. Without it, every design decision is shaped by "what if?" instead of "what for?".
