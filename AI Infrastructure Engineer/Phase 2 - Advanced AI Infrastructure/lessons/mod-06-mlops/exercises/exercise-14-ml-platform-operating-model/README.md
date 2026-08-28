# Exercise 14: ML Platform Team Operating Model

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 01 + organizational context

## Objective

Design (or critique) the operating model of an ML platform team: scope, services offered, intake process, support model, on-call rotation, success metrics. Produce a working "team charter" that other teams can use.

## Why this matters

Technical excellence doesn't matter if no one uses your platform. The ML platform teams that succeed have a deliberate operating model — what they own, what they don't, how teams self-serve, how exceptions are handled. The teams that fail are technically strong but operationally vague.

## Requirements

Produce a `CHARTER.md` (~2000 words) covering:

1. **Mission**: one paragraph.
2. **Scope**: what the platform owns vs. what's the user's responsibility (a table).
3. **Services offered**: each service with SLO + how to consume.
4. **Intake process**: how users request capabilities; turnaround targets.
5. **Support model**: support tiers, response times, escalation.
6. **On-call rotation**: schedule, handoff, paging policy.
7. **Self-serve vs ticketed**: what teams can do without us, what requires our involvement.
8. **Success metrics**: how the team measures its own value.
9. **Annual roadmap process**: how priorities get set.

## Step-by-step

### Step 1 — Pick a real or hypothetical org (15 min)
Real: your current team. Hypothetical: "200-engineer fintech, 12 ML engineers across 4 product teams, 1 platform team of 4".

### Step 2 — Mission statement (15 min)
A good mission says what you do AND what you don't:
> "We provide self-serve infrastructure for ML training, serving, and observability. We do not build models; that's product teams' job. We do not own data quality; that's data engineering's job. We do enable both."

### Step 3 — Scope table (30 min)

| Capability | Platform owns | Product team owns |
|---|---|---|
| Training pipeline orchestration | The orchestrator (Airflow) + standard images | DAG content + model code |
| Model serving infra | K8s clusters, serving framework | Model image build + deployment manifests |
| Feature store | The infrastructure | Feature definitions |
| Monitoring | Prometheus, dashboards templates, alerting infra | Per-model SLOs, alert thresholds |
| Cost attribution | The pipeline | Tagging + reading their own bill |
| Model registry | MLflow + access control | Registration + tags |

### Step 4 — Services + SLOs (30 min)

| Service | SLO | How to consume |
|---|---|---|
| Training cluster | 99.5% availability | submit Airflow DAG |
| Model serving | 99.9% availability, p95 < 200ms | helm install with our chart |
| Feature store | 99.95% availability, p99 < 10ms read | feast SDK |
| Monitoring | 99% scrape uptime | Add `prometheus.io/scrape=true` label |
| Cost reports | Daily, fresh by 09:00 UTC | Athena dashboard |

### Step 5 — Intake (15 min)
- Standard request (within scope): Self-serve documentation; expected 0 platform-team hours.
- Non-standard (new capability): Slack + form → triaged in 24h → roadmapped or rejected with reason.
- Urgent (production breaking): Pager.

### Step 6 — Support model (30 min)
- **Tier 1**: Self-serve docs + Slack channel monitored 9-6 by rotating team member.
- **Tier 2**: Filed ticket; response within 24h business hours.
- **Tier 3**: Paged on-call; response within 30 min (production-impacting).

### Step 7 — On-call (15 min)
- 4 engineers × 1 week rotations.
- Pages from: monitoring SLO breaches, Tier-3 user-filed issues.
- Handoff: weekly 30-min meeting at the rotation boundary.

### Step 8 — Self-serve vs ticketed (30 min)
Make a list of common requests and assign each:

| Request | Mode | Why |
|---|---|---|
| Deploy a new model to staging | Self-serve | Standard pattern, documented |
| Add a feature to feature store | Self-serve | Once approved in PR, automatic |
| New training cluster GPU type | Ticketed | Capacity planning + cost |
| One-off compute for a research project | Ticketed | We need to know it's coming |
| Spin up an LLM serving endpoint | Self-serve via Helm | Pattern is established |

### Step 9 — Success metrics (30 min)
- **Adoption**: # of product teams using us / total teams.
- **Velocity**: time from "team starts" to "first model in prod".
- **Reliability**: SLO attainment.
- **Toil**: hours/week of repetitive manual work (target: < 20% of platform team's time).
- **Satisfaction**: quarterly NPS from product teams.

### Step 10 — Roadmap process (30 min)
- Quarterly: gather requests, prioritize via cost/benefit.
- Half of capacity: planned roadmap work.
- 25%: keeping the lights on (toil, on-call).
- 25%: opportunistic (team picks).

## Deliverables

1. `CHARTER.md` covering all 9 sections.
2. `INTAKE_FORM.md` template for new requests.
3. `ONCALL_RUNBOOK.md`: shift expectations + first responses for common pages.
4. `METRICS_DASHBOARD.md`: how each success metric is measured.

## Validation

Have a colleague outside your team read the charter and confirm:
- [ ] Could they file a request correctly?
- [ ] Could they self-serve a model deployment?
- [ ] Do they know who's on-call?
- [ ] Do they understand what the platform won't do for them?

## Stretch goals

- Add a **request portal** (small web app) implementing the intake form.
- Run a **stakeholder interview series** (5-10 product team members) and refine charter.
- Publish the charter externally + collect feedback from peer ML platform teams.

## Common pitfalls

- **Scope creep** — Every PM wants a custom dashboard. The charter is what helps you say no without being a jerk.
- **No SLOs** — Means every issue is an emergency. Define + publish SLOs.
- **Self-serve docs that lie** — Keep them current; broken docs are worse than no docs.
- **On-call without rotation** — Burns out individuals. Even a 4-person team should rotate weekly.
- **Success metrics no one tracks** — If nothing measures the metric, it doesn't exist. Wire dashboards from day 1.

## Solutions

There's no single right answer. Reference charters from companies (Netflix, Spotify, Shopify have published versions) for inspiration, but adapt to your org.
