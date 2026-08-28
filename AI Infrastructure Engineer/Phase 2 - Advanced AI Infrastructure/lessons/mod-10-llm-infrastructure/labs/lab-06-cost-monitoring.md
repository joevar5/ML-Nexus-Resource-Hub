# Lab 06: Cost Tracking and Budget Alerts

**Duration:** 60 min  **Prerequisites:** vLLM with `/metrics` endpoint or commercial LLM API

## Objective
Track per-request and per-tenant token usage, compute estimated cost, surface in Grafana, alert when daily/monthly budgets are exceeded.

## Steps

### 1. Capture token counts
Every OpenAI-compatible response includes `usage`:
```python
r = client.chat.completions.create(...)
print(r.usage)  # CompletionUsage(prompt_tokens=N, completion_tokens=M, total_tokens=N+M)
```

### 2. Middleware to log to a metrics counter
```python
# middleware_cost.py
from prometheus_client import Counter

TOKENS_USED = Counter(
    "llm_tokens_total", "Tokens consumed",
    ["tenant","model","kind"],   # kind = prompt | completion
)
COST_USD = Counter(
    "llm_cost_usd_total", "Estimated cost in USD",
    ["tenant","model"],
)

# Cost per 1k tokens (lookup table)
PRICES = {
    "mistralai/Mistral-7B-Instruct-v0.3": (0.0001, 0.0002),  # self-hosted estimate
    "gpt-4o":                              (0.0025, 0.01),
    "claude-3.5-sonnet":                   (0.003, 0.015),
}

def log_usage(tenant, model, usage):
    prompt_p, comp_p = PRICES[model]
    TOKENS_USED.labels(tenant=tenant, model=model, kind="prompt").inc(usage.prompt_tokens)
    TOKENS_USED.labels(tenant=tenant, model=model, kind="completion").inc(usage.completion_tokens)
    cost = (usage.prompt_tokens/1000)*prompt_p + (usage.completion_tokens/1000)*comp_p
    COST_USD.labels(tenant=tenant, model=model).inc(cost)
```
Wrap your proxy so every LLM call goes through `log_usage(...)`.

### 3. Prometheus queries
```promql
# Total cost today by tenant
sum by (tenant) (increase(llm_cost_usd_total[1d]))

# Cost per million tokens by model
sum by (model) (rate(llm_cost_usd_total[5m]))
/
sum by (model) (rate(llm_tokens_total[5m])) * 1e6

# Top 10 spenders
topk(10, sum by (tenant) (increase(llm_cost_usd_total[1d])))
```

### 4. Grafana dashboard panels
- Daily cost by tenant (stacked bar over time)
- Cost per 1k tokens by model (bar gauge)
- Top tenants (table)
- Monthly burn (stat showing cumulative + projection)

### 5. Budget alert
```yaml
- alert: TenantDailyBudgetExceeded
  expr: |
    sum by (tenant) (increase(llm_cost_usd_total[1d])) > 100
  labels: { severity: warning }
  annotations:
    summary: "Tenant {{ $labels.tenant }} exceeded $100 today"
```

### 6. Hard cap via middleware (defensive)
```python
DAILY_LIMITS = {"team-a": 100.0}

def check_budget(tenant):
    # Query the counter for today's value
    today = ...  # Prometheus query result
    if today > DAILY_LIMITS.get(tenant, float("inf")):
        raise HTTPException(429, f"daily budget ${DAILY_LIMITS[tenant]} exhausted")
```

### 7. Cost-aware routing
Route low-priority traffic to cheaper models (small open-source LLM); route premium to expensive APIs.

## Validation
- [ ] Each request increments token counters by exact `usage` values.
- [ ] Cost counter matches manual computation.
- [ ] Alert fires when synthetic load pushes daily cost past threshold.

## Cleanup
None.

## Troubleshooting
- **Token counts in metrics 2× actual** — Middleware called twice; check for duplicate instrumentation.
- **Costs zero** — Pricing table missing model; default branch should warn loudly, not silently use 0.
- **Counters reset on pod restart** — Counter values are cumulative since process start. Use `increase()` over time windows, not raw values.
