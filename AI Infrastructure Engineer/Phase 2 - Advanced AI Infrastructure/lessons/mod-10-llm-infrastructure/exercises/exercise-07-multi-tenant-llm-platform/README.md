# Exercise 07: Multi-Tenant LLM Platform

**Duration:** 4 hours
**Difficulty:** Advanced
**Prerequisites:** [Exercise 03](../exercise-03-vllm-deep-dive/README.md), [Exercise 06](../exercise-06-inference-optimization-llm/README.md), and [Lesson 07: LLM Platform Architecture](../../07-llm-platform-architecture.md)
**Hardware:** 1 GPU for vLLM, plus Docker for Redis and a database

## Goal

Build a gateway that lets several teams (tenants) share one pool of GPUs. Each tenant gets its own API key, limits, allowed models, and usage bill. One tenant's traffic spike must not hurt the others.

## Why it matters

GPUs are expensive, so almost every company ends up sharing them between teams or customers. Sharing creates three problems: who is allowed to do what, who pays for what, and how to stop one noisy user from slowing everyone down. A gateway is where you solve all three.

You can build this from scratch, as you will here, to learn what is going on. In production, many teams start from an existing gateway such as LiteLLM, Envoy AI Gateway, or Kong, and add their own rules on top. Keep that in mind when you write your recommendations at the end.

## Before you start

You need:
- vLLM serving a model, started with `--scheduling-policy priority`.
- Redis, for counters.
- A place to store usage records. Use Postgres or ClickHouse for this exercise.
- A FastAPI app, which will be your gateway.

Sketch the request path before you code: tenant → gateway (auth, limits, route) → vLLM → gateway (record usage) → tenant.

## Steps

### Step 1: Tenant config (15 min)

Keep the config in a file or table, not in code. Store a hash of each key, never the key itself.

```yaml
# tenants.yaml
- id: tenant-a
  api_key_sha256: "<hash>"
  models_allowed: ["llama-8b", "qwen-7b"]
  requests_per_minute: 1000
  tokens_per_day: 10000000
  max_concurrent: 16
  priority: 0            # lower number = served first in vLLM
- id: tenant-b
  api_key_sha256: "<hash>"
  models_allowed: ["qwen-7b"]
  requests_per_minute: 100
  tokens_per_day: 1000000
  max_concurrent: 4
  priority: 5
```

### Step 2: Authentication (30 min)

Look the key's hash up directly instead of looping over tenants. A dictionary lookup on the hash is fast, and the stored hash is not secret-equivalent to the key.

```python
import hashlib
from fastapi import Header, HTTPException

BY_HASH = {t["api_key_sha256"]: t for t in load_tenants()}

async def auth(authorization: str = Header(...)) -> dict:
    key = authorization.removeprefix("Bearer ").strip()
    tenant = BY_HASH.get(hashlib.sha256(key.encode()).hexdigest())
    if tenant is None:
        raise HTTPException(401, "invalid api key")
    return tenant
```

Use long random API keys (32+ random bytes). A plain SHA-256 is fine for those, but it would not be safe for human-chosen passwords.

### Step 3: Rate limits, quotas, and concurrency (45 min)

Do all three with Redis. Each counter must be updated in one atomic step, because many requests arrive at once.

- **Requests per minute:** an `INCR` on a key that includes the current minute, with an expiry.
- **Daily tokens:** you do not know the output length in advance. **Reserve** an estimate when the request starts, then **correct** the counter with the real usage when it finishes.
- **Concurrent requests:** the main protection against a noisy neighbor. Count requests in flight per tenant and return `429` when a tenant is at its limit.

```python
async def admit(tenant: dict, est_tokens: int):
    minute_key = f"rpm:{tenant['id']}:{int(time.time() // 60)}"
    n = await r.incr(minute_key)
    if n == 1:
        await r.expire(minute_key, 120)
    if n > tenant["requests_per_minute"]:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "30"})

    day_key = f"tok:{tenant['id']}:{time.strftime('%Y-%m-%d', time.gmtime())}"
    if await r.incrby(day_key, est_tokens) > tenant["tokens_per_day"]:
        await r.decrby(day_key, est_tokens)          # give the reservation back
        raise HTTPException(429, "daily token quota exhausted")
```

Always return `Retry-After` on a `429`, so clients can back off properly.

### Step 4: Routing and priority (45 min)

```python
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, tenant=Depends(auth)):
    if req.model not in tenant["models_allowed"]:
        raise HTTPException(403, f"model {req.model} is not allowed for this tenant")
    cap = min(req.max_tokens or 1024, MAX_OUTPUT)    # never trust the client's number
    await admit(tenant, est_tokens=estimate_prompt_tokens(req) + cap)
    resp = await upstream[req.model].chat.completions.create(
        model=req.model, messages=req.messages, max_tokens=cap,
        extra_body={"priority": tenant["priority"]},
    )
    await record_usage(tenant, req.model, resp.usage)
    return resp
```

Remember to also decrement the in-flight counter in a `finally` block, so a failed request does not leak a slot.

### Step 5: Metering (30 min)

Record the real usage for every request: tenant, model, prompt tokens, completion tokens, and a timestamp. Use the `usage` numbers that vLLM returns, since that is what you will bill from.

Do not append lines to an S3 object. S3 objects cannot be appended to, so concurrent writers will overwrite each other. Instead, insert rows into Postgres or ClickHouse, or send events through a queue (Kafka, Redpanda, or Kinesis) and have a consumer write them out in batches.

For streaming responses, ask for usage in the final chunk (`stream_options={"include_usage": True}`) and record it once at the end. Counting chunks as you go leads to double-counting and wrong totals.

### Step 6: Billing export (30 min)

Write a daily query that turns usage into a cost per tenant and model. Keep prices in a table, not hard-coded, and give input and output tokens separate prices.

```sql
SELECT u.tenant, u.model,
       SUM(u.prompt_tokens)     AS prompt_tokens,
       SUM(u.completion_tokens) AS completion_tokens,
       SUM(u.prompt_tokens * p.input_per_million / 1e6
         + u.completion_tokens * p.output_per_million / 1e6) AS cost_usd
FROM usage u JOIN prices p USING (model)
WHERE u.ts >= date_trunc('day', now()) - interval '1 day'
  AND u.ts <  date_trunc('day', now())
GROUP BY u.tenant, u.model;
```

Decide whether the "day" is UTC or the tenant's own time zone, and write that choice down.

### Step 7: Isolation test (30 min)

Run a load test where tenant A sends far more than its limit while tenant B sends a normal trickle. Measure tenant B's p95 latency with and without tenant A's flood.

Then switch off the concurrency cap and run it again. You should see why priority alone is not isolation: a priority only decides who goes first in the queue, and one tenant can still fill the queue.

## Deliverables

1. A working gateway in front of vLLM, with auth, limits, routing, and metering.
2. A `tenants.yaml` and a pricing table.
3. A billing query and a sample invoice for each tenant.
4. An isolation report with numbers, with and without the concurrency cap.
5. A short note: what would you take from an existing gateway instead of building, and why?

## Check yourself

- [ ] Wrong key → `401`. Over the rate limit → `429` with `Retry-After`. Disallowed model → `403`.
- [ ] Billed tokens match vLLM's reported usage, including streamed requests.
- [ ] Tenant B's p95 latency rises by less than 20% while tenant A floods.
- [ ] No API key appears anywhere in your logs.

## Common pitfalls

- **Keys in logs.** Mask the `Authorization` header in middleware and in any error reporter.
- **Trusting the client's `max_tokens`.** Cap it on the server side, or one request can eat a whole day's quota.
- **Counters that leak.** If a request fails after you reserve tokens or a concurrency slot, you must release them.
- **Priority is not isolation.** You also need per-tenant caps, or separate worker pools for tenants who need hard guarantees.
- **Double-counting streams.** Take usage from the final chunk. Do not add up chunks.
- **Slow auth path.** Auth runs on every request, so keep it a single lookup and cache tenant config in memory.
