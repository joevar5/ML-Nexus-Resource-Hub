# Module 08: Monitoring & Observability — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-08-monitoring-observability/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What does this PromQL query calculate?
```promql
rate(http_requests_total[5m])
```

**Answer:** B) Average requests per second over last 5 minutes

**Explanation:**
`rate()` computes the per-second average rate of increase of a counter over the specified range window. With a `[5m]` window, it returns the average requests-per-second across the most recent 5 minutes, smoothing short spikes. To get totals you would use `increase()`, and for per-minute rates you would multiply by 60.

**Common Mistakes:**
- Picking A (total requests) — that would be `increase(http_requests_total[5m])`, not `rate()`.
- Picking C (current request rate) — `rate()` returns a 5-minute average; the instantaneous rate would use `irate()`.

**Related Material:** `lessons/mod-08-monitoring-observability/01-prometheus-metrics.md`

---

## Question 2
**Q:** What is the purpose of a recording rule in Prometheus?

**Answer:** B) Precompute expensive queries and store results as new time series

**Explanation:**
Recording rules let you evaluate complex or frequently used PromQL expressions on a schedule and persist the results as new time series. This dramatically speeds up dashboards and alerts that would otherwise re-execute heavy aggregations on every query. They are the standard tool for keeping high-traffic Grafana panels responsive.

**Common Mistakes:**
- Picking C (record alert history) — alert history lives in Alertmanager / external storage, not recording rules.
- Picking D (backup metrics) — backups are handled by remote_write or snapshots, not recording rules.

**Related Material:** `lessons/mod-08-monitoring-observability/01-prometheus-metrics.md`

---

## Question 3
**Q:** What does `for: 5m` mean in a Prometheus alert rule?

**Answer:** C) Alert only fires if condition is true for 5 consecutive minutes

**Explanation:**
The `for` clause specifies how long an alert expression must continuously evaluate to true before the alert transitions from "pending" to "firing". This suppresses transient spikes and reduces noise from short-lived anomalies. Evaluation interval (how often the rule runs) is a separate setting (`evaluation_interval`).

**Common Mistakes:**
- Picking A (fires every 5 minutes) — `for` does not control notification cadence; that is configured in Alertmanager via `repeat_interval`.
- Picking B (evaluated every 5 minutes) — evaluation frequency is set globally, not by `for`.

**Related Material:** `lessons/mod-08-monitoring-observability/06-alerting-strategies.md`

---

## Question 4
**Q:** Which PromQL function should you use to calculate the 95th percentile latency from a histogram?

**Answer:** B) histogram_quantile()

**Explanation:**
`histogram_quantile(0.95, sum(rate(metric_bucket[5m])) by (le))` interpolates the 95th percentile from the cumulative bucket counters that Prometheus histograms expose. The function requires the `le` (less-than-or-equal) label, which is why histograms must be defined with cumulative buckets. There is no built-in `percentile_95()` in PromQL.

**Common Mistakes:**
- Picking A (avg_over_time) — averages hide tail latency, which is exactly what P95/P99 is designed to expose.
- Picking D (percentile_95) — this function does not exist in PromQL; it's a plausible-sounding distractor.

**Related Material:** `lessons/mod-08-monitoring-observability/01-prometheus-metrics.md`

---

## Question 5
**Q:** What is the main advantage of Prometheus's pull-based architecture?

**Answer:** B) Easier to detect when targets are down

**Explanation:**
Because Prometheus actively scrapes targets, a failed scrape is immediately visible as a missing `up` metric (value `0`), making target health a first-class concept. Pull also simplifies access control (only Prometheus needs to reach targets) and makes service discovery natural via Kubernetes / Consul integrations. In push systems, a silent agent can look identical to a healthy one with no traffic.

**Common Mistakes:**
- Picking A (lower network overhead) — pull vs push has similar overhead; pull can actually be heavier with many targets.
- Picking C (faster data collection) — speed is determined by scrape interval, not the pull/push model.

**Related Material:** `lessons/mod-08-monitoring-observability/01-prometheus-metrics.md`

---

## Question 6
**Q:** What is a "span" in distributed tracing?

**Answer:** B) A single operation within a trace

**Explanation:**
A span represents one unit of work — a function call, RPC, DB query — with a start time, duration, attributes, and a parent span ID linking it into a tree. A trace is the collection of all spans sharing a trace ID, forming the request's path through services. Spans are the fundamental building block emitted by OpenTelemetry SDKs.

**Common Mistakes:**
- Picking A (total trace duration) — that is the trace's root-span duration, not a span itself.
- Picking D (network connection) — spans are logical operations, not transport-layer constructs.

**Related Material:** `lessons/mod-08-monitoring-observability/02-distributed-tracing.md`

---

## Question 7
**Q:** What protocol/framework has become the industry standard for instrumentation?

**Answer:** C) OpenTelemetry

**Explanation:**
OpenTelemetry (OTel) is the CNCF-graduated standard that unifies APIs, SDKs, and the OTLP wire protocol for metrics, logs, and traces across vendors and languages. It replaced the earlier OpenTracing and OpenCensus projects and is now supported by virtually every observability backend (Jaeger, Tempo, Datadog, etc.). Choosing OTel avoids vendor lock-in for instrumentation code.

**Common Mistakes:**
- Picking A (Zipkin) or B (Jaeger) — these are tracing *backends*; both now consume OTLP from OpenTelemetry SDKs.
- Picking D (Prometheus) — Prometheus is a metrics system, not a general instrumentation framework.

**Related Material:** `lessons/mod-08-monitoring-observability/02-distributed-tracing.md`

---

## Question 8
**Q:** What is the main advantage of Grafana Tempo over Jaeger?

**Answer:** C) Lower storage costs (object storage, minimal indexing)

**Explanation:**
Tempo stores traces directly in object storage (S3, GCS, Azure Blob) and indexes only the trace ID, relying on TraceQL plus correlation from metrics/logs to find interesting traces. This makes it dramatically cheaper than Jaeger's per-attribute indexing, especially at scale where 100% retention becomes feasible. The design choice fits the "store everything, query via correlation" paradigm.

**Common Mistakes:**
- Picking A (Better UI) — both use Grafana for visualization in modern stacks; UI is not the differentiator.
- Picking B (Faster queries) — Tempo can actually be slower for tag-based search; the win is cost, not speed.

**Related Material:** `lessons/mod-08-monitoring-observability/02-distributed-tracing.md`

---

## Question 9
**Q:** What is the Population Stability Index (PSI) used for?

**Answer:** B) Detecting data drift between two distributions

**Explanation:**
PSI quantifies how much a feature's distribution has shifted between a reference period (often training data) and a comparison period (current production), by comparing bucketed proportions. Common thresholds: PSI < 0.1 = no significant drift, 0.1–0.25 = moderate, > 0.25 = significant drift requiring action. It's widely used in credit-risk models and is a staple of ML observability dashboards.

**Common Mistakes:**
- Picking A (model accuracy) — accuracy is measured against labeled ground truth, not via PSI.
- Picking D (GPU utilization) — that is an infrastructure metric, unrelated to statistical drift.

**Related Material:** `lessons/mod-08-monitoring-observability/03-ml-model-observability.md`

---

## Question 10
**Q:** What does SHAP stand for in the context of ML explainability?

**Answer:** B) SHapley Additive exPlanations

**Explanation:**
SHAP applies Shapley values from cooperative game theory to attribute each feature's marginal contribution to a model's prediction, producing additive, locally-accurate explanations. It is widely used for both global feature importance and per-prediction interpretability, and integrates with frameworks like XGBoost, LightGBM, and PyTorch. Monitoring SHAP values over time can reveal feature-importance drift even when raw distributions look stable.

**Common Mistakes:**
- Picking A or C — these are plausible-sounding distractors built from arbitrary words; SHAP specifically derives from Lloyd Shapley.
- Picking D (System Health and Performance) — confuses an ML interpretability technique with a generic ops acronym.

**Related Material:** `lessons/mod-08-monitoring-observability/03-ml-model-observability.md`

---
