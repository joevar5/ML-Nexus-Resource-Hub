# Exercise 01: MLOps Maturity Assessment

**Duration:** 3 hours
**Difficulty:** Beginner+
**Prerequisites:** Familiarity with ML workflows

## Objective

Conduct a structured MLOps maturity assessment for a real or hypothetical team using a published framework (Google's MLOps maturity levels), identify the team's current level, prioritize 5 next investments, and write a 6-month roadmap.

## Why this matters

Most "we should improve MLOps" conversations are vague. A maturity assessment converts them to concrete: "we're at Level 1; investing in CI/CD for models moves us to Level 2; here's the work and order." Engineers and leaders both want this clarity.

## Background — the levels

Google's framework (used widely):

- **Level 0**: Manual everything. Notebook-driven training, copy-paste deployments, no automation.
- **Level 1**: Automated ML pipeline (training pipelined, but model deployment still manual).
- **Level 2**: CI/CD for the pipeline (the *pipeline itself* is deployed via CI/CD; new models flow continuously).

Each level has ~12 sub-criteria across people, process, technology, monitoring.

## Requirements

1. **Score** a team across 4 categories: Process Maturity, Tooling, Reliability, Governance.
2. **Identify the level** (0, 1, or 2) with justification.
3. **Prioritize 5 next investments** with cost/benefit per item.
4. **Write a 6-month roadmap** with milestones.
5. **Stakeholder summary** (1-page) tailored to non-technical leadership.

## Step-by-step

### Step 1 — Choose a subject (15 min)
Real: your team or a team you've worked with. Hypothetical: "fintech with 5 engineers, 3 models in prod, ad-hoc training in notebooks."

### Step 2 — Score across the rubric (90 min)

Use this scoresheet:

**Process Maturity** (1=ad-hoc, 5=fully automated)
- Code versioning: [ ]
- Data versioning: [ ]
- Experiment tracking: [ ]
- Reproducibility: [ ]
- Code review process: [ ]
- Test coverage: [ ]

**Tooling** (1=none, 5=mature)
- Pipeline orchestrator: [ ]
- Feature store: [ ]
- Model registry: [ ]
- Monitoring stack: [ ]
- Deployment automation: [ ]
- Documentation tools: [ ]

**Reliability**
- SLOs defined: [ ]
- Incident response runbooks: [ ]
- Automated rollback: [ ]
- A/B test infrastructure: [ ]
- Drift detection: [ ]

**Governance**
- Audit trail: [ ]
- Model cards: [ ]
- Data privacy controls: [ ]
- Bias / fairness review: [ ]
- Compliance documentation: [ ]

Score each 1-5 with a written justification ("3 — we track experiments in MLflow but not data versions").

### Step 3 — Compute level (15 min)
- Average < 2 → Level 0
- 2-3.5 → Level 1
- > 3.5 → Level 2

### Step 4 — Top 5 investment list (45 min)
For each: what gap it closes, estimated effort (engineer-weeks), expected ROI. Order by ROI/effort.

Sample:
```
1. Add experiment tracking (MLflow) — 2 weeks — eliminates "what produced this number" debates
2. Codify a training pipeline (Airflow + dbt) — 6 weeks — moves Level 0 → Level 1
3. Add model registry + stage promotion — 3 weeks — replaces ad-hoc deploys
4. Implement drift monitoring — 4 weeks — catches degradation before it becomes incidents
5. Define SLOs for top 3 models — 1 week — anchor for production conversations
```

### Step 5 — 6-month roadmap (30 min)
Months 1-2: investments 1+2 (foundation)
Months 3-4: investment 3 (deploy pipeline)
Months 5-6: investments 4+5 (production-grade)

Include risks per month.

### Step 6 — Stakeholder 1-pager (15 min)
"Where we are now, what we'd be after 6 months, what it costs, what the upside is." Written for an exec, not for an ML engineer.

## Deliverables

1. Filled `MATURITY_SCORECARD.md`.
2. `ROADMAP.md` 6-month plan.
3. `STAKEHOLDER_SUMMARY.md` 1-pager.

## Validation

- [ ] Every criterion has a numeric score + 1-sentence justification.
- [ ] Top-5 investments cited specific gaps from the rubric.
- [ ] Roadmap milestones are SMART (Specific, Measurable, Achievable, Relevant, Timeboxed).
- [ ] Stakeholder summary readable in < 3 minutes by a non-technical person.

## Stretch goals

- Repeat the assessment 6 months later (real or simulated) and document the delta.
- Run the assessment with 3 different teammates separately, then reconcile differences.
- Build a **quarterly review template** that other teams can adopt.

## Common pitfalls

- **Scoring high to look good** — Sandbag a little; the goal is to find investments, not look complete.
- **Generic recommendations** — "Implement MLOps" isn't a recommendation. "Add MLflow to track experiments for the recs model by 2026-07-15" is.
- **Ignoring people + process** — Tooling investments without process changes (review, on-call) underperform.
- **No buy-in milestone** — A roadmap without leadership sign-off is a wish list.
