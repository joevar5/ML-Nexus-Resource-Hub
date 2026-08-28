# Exercise 10: Model Governance and Audit

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercise 03 (registry); compliance familiarity helpful

## Objective

Implement a model governance framework for an ML platform: model cards, decision logs, bias/fairness reviews, approval workflows, audit trails, and a quarterly review cadence. Make the framework practical (lightweight) so engineers will actually use it.

## Why this matters

Regulation (EU AI Act, US executive orders, sector-specific rules like SR 11-7 for banking) and customer trust both demand model governance. Heavy-handed governance kills velocity; lightweight governance gets ignored. The sweet spot is "engineers help themselves with these templates."

## Requirements

1. **Model card template** auto-generated from MLflow metadata + manual fields.
2. **Decision log** capturing rationale for every architectural choice (model, framework, data sources).
3. **Bias/fairness checklist** with at least 3 dimensions (demographic, geographic, temporal).
4. **Approval workflow** distinct from technical promotion (governance approval is separate from "good metrics").
5. **Audit trail** queryable by model, by date, by approver.
6. **Quarterly review template** for a model in production.

## Step-by-step

### Step 1 — Model card template (30 min)
```markdown
# Model Card: recs-ranker v3.2

## Intended use
**Primary purpose:** Rank product candidates for homepage display.
**Out of scope:** Search ranking, ad ranking, fraud detection.

## Training data
- **Source:** Production click + purchase events 2024-2026.
- **Volume:** 100M rows, 5M users.
- **Time range:** 2024-01-01 to 2026-05-01.
- **Known biases:** US-skewed (60% of users), recent-skewed (more weight on 2025 events).

## Performance
| Slice | AUC | Precision@10 | Recall@10 |
|---|---|---|---|
| Overall | 0.872 | 0.34 | 0.51 |
| US users | 0.881 | 0.36 | 0.53 |
| Non-US users | 0.842 | 0.28 | 0.47 |
| New users (< 7d) | 0.798 | 0.22 | 0.41 |

## Ethical considerations
- Disparate accuracy across regions; mitigation: per-region threshold tuning.
- Cold-start performance lower; fallback to popularity-based ranker.

## Limitations
- Performance degrades after 30 days without retraining.
- Cannot handle products with no historical interactions.
- Trained without privacy guarantees (no DP).

## Maintenance
- Retraining: weekly, automated.
- Quarterly review: next 2026-08-01.
- Owner: alice@example.com / @ml-recs slack channel.

## Audit references
- Decision log: link
- Bias review: link
- Approval: 2026-05-15 by jane@ (compliance@)
```

### Step 2 — Auto-generate from MLflow (45 min)
```python
# generate_card.py
import mlflow
import jinja2

def generate(model_name, version):
    client = mlflow.MlflowClient()
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    
    template = jinja2.Template(open("templates/model_card.md.j2").read())
    return template.render(
        model_name=model_name,
        version=version,
        metrics=run.data.metrics,
        params=run.data.params,
        tags=run.data.tags,
        run_id=mv.run_id,
        created_at=mv.creation_timestamp,
    )

print(generate("recs-ranker", "12"))
```

### Step 3 — Decision log (30 min)
Markdown ADRs (Architecture Decision Records):
```
docs/decisions/
├── 001-choose-mlflow-over-w&b.md
├── 002-use-redis-for-feature-store.md
├── 003-no-personalization-for-new-users.md
└── ...
```

Template:
```markdown
# ADR-007: Drop demographic features from recs model

**Date:** 2026-05-12  **Status:** Accepted

## Context
Including demographic features (inferred age, gender) improved overall AUC by 0.4pp but introduced disparate performance across age groups and exposed PII risk.

## Decision
Drop demographic features. Use behavioral signals only.

## Consequences
- AUC down 0.4pp.
- Disparate-performance metric down 60%.
- PII exposure surface reduced.
- Simplifies compliance review.

## Alternatives considered
1. Keep features with mitigation (k-anonymity) — rejected, complex + ongoing burden.
2. Selective drop (gender only) — rejected, halfway compromise.
```

### Step 4 — Bias/fairness checklist (30 min)
```markdown
# Pre-launch bias review: recs-ranker v3.2

## Demographic
- [x] Per-country accuracy difference < 5pp? (computed: 3.9pp, US vs non-US)
- [x] Per-age-bracket accuracy difference < 5pp? (computed: 4.2pp, 18-25 vs 55+)

## Geographic
- [x] Performance equitable across 5 largest markets? (yes)

## Temporal
- [x] Performance stable across day-of-week, weekday vs weekend?
- [x] Recency bias: model doesn't only recommend last-7-days items?

## Coverage
- [x] Min items recommended per user > 5? (yes, 8 by design)
- [x] Cold-start strategy documented? (yes, fallback to popularity ranker)

## Sign-offs
- ML lead: alice@ (date)
- Compliance: jane@ (date)
- Product: bob@ (date)
```

### Step 5 — Approval workflow (30 min)
Two parallel approvals before Production:
1. **Technical approval** (Exercise 03's quality gate) — automatic.
2. **Governance approval** — manual, requires:
   - Model card published.
   - Bias review checklist.
   - Decision log entries for major choices.
   - Owner of record.

Slack bot prompts for both; promotion to Production blocked until both approvals.

### Step 6 — Audit trail (30 min)
SQLite or DB:
```sql
CREATE TABLE governance_events (
    ts TIMESTAMP, model_name TEXT, version TEXT,
    event TEXT, actor TEXT, justification TEXT, attachments TEXT
);
```
Every event (registration, promotion, retirement, bias-review-completed) appended.

```python
def audit_query(model_name=None, since=None):
    # Return chronological events; support filtering by model, date range
    ...
```

### Step 7 — Quarterly review template (15 min)
```markdown
## Q2 2026 Review: recs-ranker

### Production performance
- Availability: 99.6% (target 99.5%) ✅
- Latency p95: 142ms (target 200ms) ✅
- Click-through: 8.4% (vs Q1 8.1%) ✅

### Incidents
- 2026-04-22: 18-min outage (auth dep). MTTR met.
- 2026-05-15: predictions for users in DE wildly off — root cause: training set excluded EU registrations after GDPR scrubbing. Fixed in v3.3.

### Drift
- Feature `amount_30d`: PSI 0.08 — within bounds.
- Feature `country`: distribution stable.

### Retraining cadence
- Currently weekly; consider biweekly to reduce cost.

### Action items for Q3
- [ ] Add per-country accuracy alerts (driven by incident above).
- [ ] Re-validate bias checklist.
- [ ] Refresh model card based on Q2 incidents.
```

## Deliverables

1. Model card template + at least 2 generated cards.
2. Decision log directory with ≥ 5 ADRs.
3. Bias review checklist + 1 completed example.
4. Approval workflow (script or webhook).
5. Audit trail with queryable history.
6. Quarterly review template + 1 example.

## Validation

- [ ] Auto-generation from MLflow produces a useful card.
- [ ] ADRs follow the template; team can locate any past decision quickly.
- [ ] Bias checklist would have caught the Q2 EU bug if completed for v3.2.
- [ ] Audit query returns the right events for a given model.

## Stretch goals

- Add a **regulatory mapping**: which fields satisfy which clause of which framework (EU AI Act, NIST AI RMF, ISO 42001).
- Implement **explainability outputs** (SHAP feature attributions) as a standard artifact alongside the model.
- Build a **governance dashboard**: status across all production models at a glance.

## Common pitfalls

- **Heavy approval = ignored** — Keep approvals lightweight. The audit trail is what matters.
- **Model cards that lie** — Generated cards reflect training-time metrics; production performance diverges. Set quarterly refresh.
- **Bias review as one-time** — Drift happens. Re-run quarterly.
- **Audit log not queryable in incident** — Test access during a Game Day. If only one person knows how to query it, it's not really an audit.
