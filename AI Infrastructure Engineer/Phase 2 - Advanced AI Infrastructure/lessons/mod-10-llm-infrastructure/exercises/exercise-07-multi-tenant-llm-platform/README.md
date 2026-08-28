# Exercise 07: Multi-Tenant LLM Platform

**Duration:** 4 hours
**Difficulty:** Advanced
**Prerequisites:** Exercises 03 + 06

## Objective

Build a multi-tenant LLM platform: shared GPU pool serving N models, per-tenant API keys + quotas + rate limits, request routing by tenant, isolated metering. Tenants don't know they're sharing.

## Requirements

1. Single ingress for all tenants.
2. Tenant authentication via API key.
3. Per-tenant rate limits + token quotas.
4. Request routing to the right model based on tenant config.
5. Per-tenant cost attribution + billing export.
6. Tenants isolated: tenant A's traffic spike doesn't degrade tenant B.

## Step-by-step

### Step 1 — Tenant config (15 min)
```yaml
# tenants.yaml
- id: tenant-a
  api_key_hash: <sha256>
  models_allowed: ["mistral-7b", "llama-3-8b"]
  rate_limit_rpm: 1000
  tokens_per_day: 10_000_000
  priority: 1
- id: tenant-b
  api_key_hash: <sha256>
  models_allowed: ["mistral-7b"]
  rate_limit_rpm: 100
  tokens_per_day: 1_000_000
  priority: 0
```

### Step 2 — Auth middleware (30 min)
```python
from fastapi import HTTPException, Header
import hashlib

TENANTS = load_tenants()

async def auth(authorization: str = Header(...)) -> dict:
    api_key = authorization.removeprefix("Bearer ")
    h = hashlib.sha256(api_key.encode()).hexdigest()
    for tenant in TENANTS:
        if tenant["api_key_hash"] == h:
            return tenant
    raise HTTPException(401, "invalid api key")
```

### Step 3 — Rate limit + quota (45 min)
```python
import redis
r = redis.Redis()

async def check_quota(tenant: dict, est_tokens: int = 1000):
    # Rate limit per minute
    key = f"rate:{tenant['id']}:{int(time.time()/60)}"
    count = r.incr(key)
    if count == 1: r.expire(key, 120)
    if count > tenant["rate_limit_rpm"]:
        raise HTTPException(429, "rate limit exceeded")
    
    # Daily token quota
    daily_key = f"tokens:{tenant['id']}:{time.strftime('%Y-%m-%d')}"
    used = int(r.get(daily_key) or 0)
    if used + est_tokens > tenant["tokens_per_day"]:
        raise HTTPException(429, "daily token quota exhausted")
```

### Step 4 — Request routing + priority (45 min)
```python
@app.post("/v1/chat/completions")
async def chat(req: dict, tenant = Depends(auth)):
    await check_quota(tenant)
    
    model = req["model"]
    if model not in tenant["models_allowed"]:
        raise HTTPException(403, f"model {model} not allowed")
    
    # Forward to vLLM with priority
    extra = {"priority": tenant["priority"]}
    r = await vllm_client.chat.completions.create(
        model=model, messages=req["messages"], extra_body=extra,
    )
    
    # Record actual usage
    r.usage_for_billing = r.usage
    record_usage(tenant, model, r.usage)
    return r
```

### Step 5 — Per-tenant metering (30 min)
```python
import time, json
def record_usage(tenant, model, usage):
    rec = {
        "ts": int(time.time()), "tenant": tenant["id"], "model": model,
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
    }
    # Append to S3 line-delimited JSON (daily file)
    s3.put_object(Bucket="usage", Key=f"{tenant['id']}/{date.today()}.jsonl",
                   Body=json.dumps(rec) + "\n", IfMatch=...)   # or use Kinesis Firehose
```

### Step 6 — Billing export (30 min)
Daily Athena query → per-tenant invoice:
```sql
SELECT tenant, model,
       SUM(prompt_tokens) AS prompt,
       SUM(completion_tokens) AS completion,
       SUM(prompt_tokens * 0.0001/1000 + completion_tokens * 0.0003/1000) AS cost_usd
FROM usage
WHERE date = current_date - 1
GROUP BY tenant, model;
```

### Step 7 — Isolation test (30 min)
Synthetic: tenant A hammers at 5000 rpm. Tenant B should still get < 200ms latency. Verify with parallel load tests.

## Deliverables

1. Working multi-tenant proxy in front of vLLM.
2. Auth, rate limit, quota, routing all working.
3. Usage records in S3.
4. Daily billing export query.
5. Isolation test report.

## Validation

- [ ] Wrong API key → 401.
- [ ] Exceeding rpm → 429.
- [ ] Disallowed model → 403.
- [ ] Tenant A overload doesn't degrade tenant B by > 20%.

## Common pitfalls

- **API keys in logs** — Mask in middleware.
- **Quota reset at midnight UTC vs tenant local time** — Pick one; document.
- **Priority alone doesn't isolate** — Need separate worker pools or rate limits for true isolation.
- **Billing inflation from streaming** — Streaming responses count tokens incrementally; double-counting easy.
