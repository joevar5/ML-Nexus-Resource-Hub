# Lab 08: Multi-Model Serving with Intelligent Routing

**Duration:** 75 min  **Prerequisites:** vLLM or OpenAI API; one strong model + one cheap model

## Objective
Build a routing layer that sends each query to the smallest model capable of handling it. Measure cost savings while maintaining quality.

## Steps

### 1. Mental model
- Tier 0: cheap small model (Mistral-7B, Phi-3-mini, gpt-4o-mini) for short factual / formulaic queries.
- Tier 1: medium model (Llama-3-70B, gpt-4o) for reasoning or long context.
- Tier 2: strong frontier model (gpt-4o, claude-3.5-sonnet) for hardest queries only.

### 2. Router
```python
import re
from openai import OpenAI

tier_clients = {
    0: OpenAI(base_url="http://localhost:8000/v1", api_key="dummy"),
    1: OpenAI(api_key="$OPENAI_KEY"),
    2: OpenAI(api_key="$OPENAI_KEY"),
}
tier_models = {
    0: "mistralai/Mistral-7B-Instruct-v0.3",
    1: "gpt-4o-mini",
    2: "gpt-4o",
}

def classify(query: str) -> int:
    """Cheap heuristic classifier. Real systems use a tiny LLM classifier."""
    if len(query) < 200 and not any(k in query.lower() for k in ["analyze","reason","compare","derive"]):
        return 0
    if len(query) < 4000:
        return 1
    return 2

def route(query: str) -> str:
    tier = classify(query)
    client = tier_clients[tier]
    model = tier_models[tier]
    r = client.chat.completions.create(model=model, messages=[{"role":"user","content":query}], max_tokens=400)
    return r.choices[0].message.content, tier
```

### 3. Confidence-based escalation
If the small model's answer is short or uncertain, retry on a larger model:
```python
def confident(text):
    return len(text) > 100 and not any(p in text.lower() for p in ["i don't know", "unable to", "i'm not sure"])

def route_with_escalation(query, max_tier=2):
    for t in range(max_tier+1):
        ans = tier_clients[t].chat.completions.create(
            model=tier_models[t], messages=[{"role":"user","content":query}], max_tokens=400
        ).choices[0].message.content
        if confident(ans) or t == max_tier:
            return ans, t
```

### 4. Measure
```python
queries = [
    "What is 2+2?",                                         # → tier 0
    "Summarize the plot of Hamlet in one sentence.",        # → tier 0
    "Compare the design tradeoffs of Raft and Paxos in 200 words.",  # → tier 1
    "<5000-char legal document>... Identify all clauses related to termination.",  # → tier 2
]
import time
results = []
for q in queries:
    t0 = time.perf_counter()
    ans, tier = route(q)
    results.append({"q":q[:50], "tier":tier, "ms":int((time.perf_counter()-t0)*1000), "len":len(ans)})
for r in results: print(r)
```

### 5. Cost report
Using the cost lookup from lab 06:
```python
# pseudo: total = sum(price(tier) * tokens(tier))
```
Compare against always-tier-2 baseline; typical 50-70% cost reduction with minimal quality loss for well-tuned routing.

### 6. Production refinement
- Replace heuristic classifier with a trained classifier (logistic regression on TF-IDF, or a tiny BERT).
- Track per-tier acceptance rate; if tier-0 is escalating > 50% of the time, raise its tier.
- A/B test routing changes against a held-out evaluation set.

## Validation
- [ ] Simple queries route to tier 0.
- [ ] Long/complex queries route to tier 2.
- [ ] Total cost on a representative workload < always-frontier baseline.

## Cleanup
None.

## Troubleshooting
- **Classifier sends everything to tier 0** — Heuristics too lenient; tighten length/keyword thresholds.
- **Quality drop noticed by users** — Build an offline eval set with known-good answers; alert when tier-0 accuracy on it drops.
- **Latency increases due to escalation** — Cap escalation depth; some queries are fundamentally hard for small models.
