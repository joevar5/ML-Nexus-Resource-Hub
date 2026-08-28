# Exercise 08: Cost-Aware LLM Routing

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 07; lab 08

## Objective

Build a routing layer that selects the cheapest capable model per request. Implement a classifier (heuristic + small LLM) that decides tier, with confidence-based escalation. Measure cost savings vs always-frontier baseline.

## Requirements

1. Three tiers: small (Mistral-7B local), medium (gpt-4o-mini), large (gpt-4o).
2. Classifier choosing tier per request.
3. Confidence-based escalation (small first; if uncertain, retry on larger).
4. Cost tracking + comparison to baseline.
5. Quality eval on held-out test set.

## Step-by-step

### Step 1 — Heuristic classifier (30 min)
```python
def classify_heuristic(query: str) -> int:
    if len(query) < 200 and not any(k in query.lower() for k in ["analyze","reason","derive","compare"]):
        return 0   # small
    if len(query) < 4000:
        return 1   # medium
    return 2       # large
```

### Step 2 — LLM-based classifier (60 min)
A small LLM call (cheap; ~$0.001 per classification) to pick tier:
```python
def classify_llm(query: str) -> int:
    r = client_small.chat.completions.create(
        model="mistral-7b",
        messages=[
            {"role":"system","content":"Classify the user query difficulty as 0 (trivial), 1 (moderate), or 2 (complex). Reply with only the digit."},
            {"role":"user","content":query},
        ],
        max_tokens=2, temperature=0,
    )
    return int(r.choices[0].message.content.strip())
```

### Step 3 — Confidence escalation (45 min)
```python
def route_with_escalation(query, max_tier=2):
    tier = classify_heuristic(query)
    for t in range(tier, max_tier+1):
        ans = clients[t].chat.completions.create(model=models[t], messages=[{"role":"user","content":query}], max_tokens=400).choices[0].message.content
        if confident(ans):
            return ans, t
    return ans, max_tier

def confident(text):
    return len(text) > 100 and not any(p in text.lower() for p in ["i don't know", "unable to", "not sure", "cannot determine"])
```

### Step 4 — Test set (30 min)
Construct 100 queries spanning difficulty. Run through:
1. Always-small baseline
2. Always-large baseline
3. Heuristic routing
4. LLM-based routing
5. With escalation

Record cost + quality (have a held-out gold answer; use exact-match or LLM-as-judge per Exercise 12).

### Step 5 — Results matrix (15 min)

| Strategy | Cost ($) | Acceptable answer rate | Avg latency |
|---|---|---|---|
| Always small | 0.10 | 60% | 200ms |
| Always large | 5.00 | 95% | 2000ms |
| Heuristic | 1.20 | 88% | 500ms |
| LLM-based | 1.50 | 92% | 700ms |
| With escalation | 2.10 | 96% | 600ms |

### Step 6 — Production refinement (30 min)
- Cache classifier decisions (similar queries → same tier).
- Track per-tier acceptance rate; retrain heuristic if it drifts.
- A/B test routing changes weekly.

## Deliverables

1. Routing implementation with all 3 strategies.
2. Test set + eval harness.
3. `RESULTS.md` comparison.
4. `ROUTING_PLAYBOOK.md`: how to add new tiers, when to retrain classifier.

## Validation

- [ ] Cost reduction ≥ 50% vs always-large at 90%+ quality.
- [ ] Escalation correctly catches misclassified easy-but-actually-hard queries.

## Common pitfalls

- **Classifier hallucination** — LLM classifier sometimes returns "the query is..." instead of just a digit. Strict parsing + fallback.
- **Latency overhead from classifier** — Adds ~100ms; worth it if it routes away from a 3000ms call.
- **Drift** — Heuristic stops working as query mix shifts; monitor.
