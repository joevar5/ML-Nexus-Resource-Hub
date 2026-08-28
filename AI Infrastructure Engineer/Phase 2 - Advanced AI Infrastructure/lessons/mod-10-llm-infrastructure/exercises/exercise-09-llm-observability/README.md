# Exercise 09: LLM-Specific Observability

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercise 06 + mod-08 monitoring

## Objective

Instrument an LLM serving stack with LLM-specific metrics: TTFT, throughput tokens/s, prompt + completion length distributions, model selection counts, per-tenant usage, and prompt-response logging with sampling.

## Why this matters

Generic HTTP metrics (latency, error rate) don't tell you "is the LLM working?" LLM-specific signals are: streaming health, token throughput, response quality drift, abuse patterns.

## Requirements

1. Custom Prometheus metrics specific to LLM serving.
2. Log every request with prompt + completion (sampled at 1%).
3. Grafana dashboard purpose-built for LLMs.
4. Alert on abnormal token usage spikes.
5. Anonymized prompt corpus for security analysis.

## Step-by-step

### Step 1 — Custom metrics (45 min)
```python
from prometheus_client import Histogram, Counter, Summary

TTFT = Histogram("llm_ttft_seconds", "Time to first token", ["model", "tenant"],
                  buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
THROUGHPUT = Histogram("llm_tokens_per_second", "Generation throughput", ["model"],
                        buckets=(10, 25, 50, 100, 200, 500))
PROMPT_LEN = Histogram("llm_prompt_tokens", "Prompt token count", ["model"],
                        buckets=(100, 500, 1000, 4000, 8000, 16000, 32000))
COMP_LEN = Histogram("llm_completion_tokens", "Completion token count", ["model"],
                      buckets=(50, 100, 250, 500, 1000, 2000))
MODEL_SELECT = Counter("llm_model_selected_total", "Model routing decisions",
                        ["model", "tier"])

async def chat(req):
    t0 = time.perf_counter()
    first_token = None
    tokens = 0
    async for chunk in stream:
        if first_token is None:
            TTFT.labels(req.model, req.tenant).observe(time.perf_counter() - t0)
            first_token = time.perf_counter()
        tokens += 1
    duration = time.perf_counter() - first_token
    THROUGHPUT.labels(req.model).observe(tokens / duration)
    PROMPT_LEN.labels(req.model).observe(req.usage.prompt_tokens)
    COMP_LEN.labels(req.model).observe(req.usage.completion_tokens)
```

### Step 2 — Sampled request logging (30 min)
```python
import random
def maybe_log(req, resp):
    if random.random() < 0.01:    # 1% sample
        s3_log({
            "ts": time.time(),
            "tenant": req.tenant,
            "model": req.model,
            "prompt": anonymize(req.messages),
            "completion": resp.choices[0].message.content,
            "usage": dict(resp.usage),
        })

def anonymize(messages):
    # Strip PII: emails, phone numbers, anything matching SSN/credit card patterns
    text = json.dumps(messages)
    text = re.sub(r"\b\w+@\w+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", "[PHONE]", text)
    return json.loads(text)
```

### Step 3 — Grafana dashboard (60 min)
Panels:
- **Requests/sec by model + tenant**
- **TTFT p50/p95/p99 by model**
- **Throughput (tokens/s) distribution**
- **Prompt length distribution heatmap** (catches abuse / runaway prompts)
- **Completion length p99** (catches loops)
- **Token budget burn-down per tenant**
- **Model selection breakdown** (routing health)
- **Errors by type** (rate limit, model error, validation)

### Step 4 — Anomaly alerts (30 min)
```yaml
- alert: PromptTokenSpike
  expr: |
    histogram_quantile(0.99, sum by (le, tenant) (rate(llm_prompt_tokens_bucket[5m])))
    > 16000
  labels: { severity: warning }
  annotations: { summary: "Tenant {{ $labels.tenant }} sending unusually long prompts" }

- alert: CompletionExceedsAverage
  expr: |
    histogram_quantile(0.99, sum by (le) (rate(llm_completion_tokens_bucket[5m])))
    > 2 * histogram_quantile(0.50, sum by (le) (rate(llm_completion_tokens_bucket[1d])))
  labels: { severity: info }
  annotations: { summary: "Completion lengths 2× typical — possible loop or stuck generation" }
```

### Step 5 — Cost dashboard (15 min)
Tie back to exercise 06 of mod-06: per-tenant token cost over time.

## Deliverables

1. Instrumented service with all 5 custom metrics.
2. Sampled prompt log in S3.
3. Grafana dashboard JSON.
4. Alert rules deployed.

## Validation

- [ ] All metrics scraped by Prometheus.
- [ ] Dashboard renders all panels.
- [ ] Sampled log captures ~1% of requests, PII-redacted.
- [ ] Synthetic abuse (huge prompts) triggers alert.

## Common pitfalls

- **Logging raw prompts → leaked PII** — Anonymize before logging; consider not logging at all in regulated environments.
- **TTFT measurement on non-streaming** — Only meaningful for streaming responses.
- **Token counter mismatch with billing** — Use the same usage object Stripe/billing reads.
