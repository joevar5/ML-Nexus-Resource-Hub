# Exercise 09: LLM Observability

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** [Exercise 06](../exercise-06-inference-optimization-llm/README.md) and Module 08 (Monitoring)
**Hardware:** Docker, plus a vLLM server or the gateway from Exercise 07

## Goal

Make an LLM service easy to see into. You will collect LLM-specific metrics, record traces of each request, log a sample of prompts safely, build a dashboard, and set up alerts that catch problems before users do.

## Why it matters

A normal web service can be "up" with 200 responses and still be failing. An LLM can be slow, stuck in a loop, expensive, or giving worse answers, and none of that shows up in HTTP error rates. You need signals that match how LLMs fail.

## Before you start

Know where your numbers come from. There are three layers:

1. **The inference server.** vLLM already exposes a Prometheus endpoint at `/metrics`, with queue length, running requests, KV cache usage, and TTFT and latency histograms. Check those first, before writing any code. Metric names change between versions, so list what your server actually exposes.
2. **Your gateway or application.** This is the only place that knows the tenant, the feature, the model chosen, and the user-visible latency.
3. **Quality signals.** User feedback, evaluation scores on sampled traffic, and refusal rates. These are slower but tell you the most.

## Steps

### Step 1: Look at what you already have (20 min)

Start your server, open `/metrics`, and write down the metrics for queue depth, active requests, KV cache usage, TTFT, and token counts. Plot them in Prometheus or Grafana. Decide which gaps your own code needs to fill.

### Step 2: Add your own metrics (45 min)

Add the ones the server cannot know. Choose labels carefully, since every unique label combination creates a new time series.

```python
from prometheus_client import Counter, Histogram

TTFT = Histogram("llm_ttft_seconds", "Time to first token", ["model"],
                 buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
TOKENS = Counter("llm_tokens_total", "Tokens processed", ["model", "tenant", "kind"])  # kind: prompt|completion
REQUESTS = Counter("llm_requests_total", "Requests", ["model", "tenant", "status"])
PROMPT_LEN = Histogram("llm_prompt_tokens", "Prompt length", ["model"],
                       buckets=(100, 500, 1000, 4000, 8000, 16000, 32000))
COMPLETION_LEN = Histogram("llm_completion_tokens", "Completion length", ["model"],
                           buckets=(50, 100, 250, 500, 1000, 2000, 4000))
```

Measure time to first token on the **first chunk** of a streaming response. Take token counts from the `usage` object, not from counting chunks, because one chunk is not always one token.

Keep labels low-cardinality: model, tenant, and status are fine. Never label with a user ID, request ID, or prompt text, because that will overload Prometheus.

### Step 3: Add tracing (30 min)

Metrics tell you that something is slow. Traces show where. Use OpenTelemetry, with one trace per request and spans for auth, retrieval, the model call, and any guardrails.

The OpenTelemetry project defines standard "GenAI" attributes for LLM calls (model name, token counts, and so on), so use those names instead of inventing your own, and check the current spec because it is still evolving. Libraries and tools such as OpenLLMetry, OpenInference, Langfuse, and Arize Phoenix can capture most of this automatically.

Send traces to a backend such as Jaeger, Tempo, Langfuse, or Phoenix. Pick one and look at a trace of a slow request.

### Step 4: Log sampled prompts safely (30 min)

Reading real prompts and answers is the best way to find quality problems. It is also how personal data leaks. Be deliberate.

- Sample a small share of traffic (for example 1%), plus **every** error and every request a user rated badly.
- Redact personal data **before** it is stored. Regular expressions catch emails and phone numbers. A dedicated tool such as Microsoft Presidio catches much more. Test your redaction against examples and accept that it is never perfect.
- Set a retention period and restrict who can read the logs.
- In regulated settings, log metadata only (lengths, model, timing) and no content.

### Step 5: Build the dashboard (45 min)

Group panels by the question they answer:

| Question | Panels |
|---|---|
| Is it healthy? | Requests per second, error rate by type, queue depth |
| Is it fast? | TTFT p50/p95/p99, inter-token latency, tokens per second |
| Is it full? | KV cache usage, running vs waiting requests, GPU utilization |
| Is it weird? | Prompt length heatmap, completion length p99, truncation rate |
| What does it cost? | Tokens per tenant and model, estimated spend |
| Is it good? | User ratings, eval score on sampled traffic, refusal rate |

### Step 6: Alerts (30 min)

Alert on what hurts users, and send the alert to someone who can fix it. Include a link to the dashboard in every alert.

```yaml
- alert: HighTimeToFirstToken
  expr: histogram_quantile(0.95, sum by (le, model) (rate(llm_ttft_seconds_bucket[5m]))) > 2
  for: 10m
  labels: { severity: page }
  annotations: { summary: "p95 TTFT above 2s for {{ $labels.model }}" }

- alert: TokenSpend
  expr: sum by (tenant) (rate(llm_tokens_total[1h])) > 3 * sum by (tenant) (rate(llm_tokens_total[1h] offset 1d))
  for: 30m
  labels: { severity: ticket }
  annotations: { summary: "{{ $labels.tenant }} token usage is 3x yesterday" }
```

Add at least one alert for a stuck or looping generation (completion length far above normal) and one for a full request queue.

### Step 7: Test that it works (20 min)

Break something on purpose and watch the alert fire:
- Send a flood of long prompts.
- Use a prompt that makes the model repeat itself.
- Stop the GPU server for a minute.

An alert that you have never seen fire is an alert you cannot trust.

## Deliverables

1. The instrumented service, with traces and metrics.
2. A Grafana dashboard exported as JSON.
3. Alert rules, and a screenshot or log of at least one firing in your test.
4. A logging policy: sample rate, redaction, retention, and who has access.

## Check yourself

- [ ] Prometheus scrapes both the server's metrics and yours.
- [ ] You can open one trace and see where the time went.
- [ ] The prompt log contains no emails or phone numbers from your test data.
- [ ] A deliberate failure triggers the matching alert.

## Common pitfalls

- **High-cardinality labels.** User IDs or prompt text as labels will crash your metrics system.
- **Logging raw prompts.** They contain personal data, secrets, and customer content. Redact first, and consider not logging at all.
- **Measuring TTFT without streaming.** Without streaming, the first byte arrives with the last, so TTFT equals total latency.
- **Counting chunks as tokens.** Use the usage numbers, and make sure they match your billing source.
- **Only watching system metrics.** A model can answer fast and be wrong. Track a quality signal too.
