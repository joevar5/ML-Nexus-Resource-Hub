# Module 08: Monitoring & Observability — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-08-monitoring-observability/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What are the three pillars of observability?

**Answer:** B) Metrics, Logs, Traces

**Explanation:**
The three pillars of observability are metrics (aggregated numerical measurements over time), logs (discrete timestamped event records), and traces (the path of a request through a distributed system). Together they provide complementary signals that let engineers answer not only *what* is happening but *why*. Each pillar covers blind spots of the others — for example, metrics show trends, logs reveal context, and traces expose cross-service causality.

**Common Mistakes:**
- Picking A (Monitoring, Logging, Tracing) — "Monitoring" is a discipline, not a pillar; the pillar is metrics, the data type used for monitoring.
- Picking D (CPU, Memory, Disk) — these are infrastructure resource categories, not observability pillars.

**Related Material:** `lessons/mod-08-monitoring-observability/01-introduction-observability.md`

---

## Question 2
**Q:** Which metric type in Prometheus can only increase over time?

**Answer:** C) Counter

**Explanation:**
A Prometheus counter is a monotonically increasing value that only resets on process restart — perfect for cumulative quantities like requests served, errors encountered, or bytes processed. You typically apply `rate()` or `increase()` to counters to derive a per-second rate. Gauges, in contrast, go up and down, while histograms and summaries record observation distributions.

**Common Mistakes:**
- Picking A (Gauge) — gauges fluctuate freely (e.g., temperature, queue depth), so they are the opposite of counters.
- Picking B (Histogram) — histograms use multiple bucket counters internally, but the histogram type itself records distributions, not a single monotonic value.

**Related Material:** `lessons/mod-08-monitoring-observability/02-prometheus-metrics.md`

---

## Question 3
**Q:** What is the primary difference between monitoring and observability?

**Answer:** B) Monitoring answers "what is broken", observability answers "why it's broken"

**Explanation:**
Monitoring traditionally watches known failure modes via predefined dashboards and alerts — it tells you *that* something is wrong. Observability goes further by providing the rich, high-cardinality data needed to ask new, unforeseen questions about system behavior, helping you understand *why* it broke. In practice, monitoring is a subset of an observable system.

**Common Mistakes:**
- Picking C (monitoring is only metrics) — both disciplines use metrics, logs, and traces; the difference is in approach, not data type.
- Picking D (no difference) — they are related but not synonymous; conflating them leads to under-investing in tooling for unknown unknowns.

**Related Material:** `lessons/mod-08-monitoring-observability/01-introduction-observability.md`

---

## Question 4
**Q:** In the context of ML systems, what does "data drift" refer to?

**Answer:** B) Changes in input data distribution compared to training data

**Explanation:**
Data drift (also called covariate shift) occurs when the statistical distribution of incoming features at inference time diverges from the distribution the model was trained on. Even if the model itself doesn't change, its predictions degrade because it is being applied outside its learned input space. Detecting drift early (via PSI, KS tests, etc.) allows retraining before accuracy collapses.

**Common Mistakes:**
- Picking A (accuracy decrease over time) — that is "concept drift" or simply model degradation, which is often a *consequence* of data drift, not its definition.
- Picking C (network latency) — drift is a statistical property of data, unrelated to network performance.

**Related Material:** `lessons/mod-08-monitoring-observability/07-ml-model-observability.md`

---

## Question 5
**Q:** Which of the following is NOT one of the "Four Golden Signals"?

**Answer:** C) Security

**Explanation:**
Google SRE's Four Golden Signals are Latency, Traffic, Errors, and Saturation — the minimum signals to instrument for any user-facing service. Security, while critical, is a separate domain handled by different controls and telemetry (audit logs, intrusion detection). The Golden Signals focus on user-perceived reliability and capacity headroom.

**Common Mistakes:**
- Picking A (Latency) or B (Traffic) — both are core Golden Signals and were likely chosen by confusing the question wording.
- Picking D (Saturation) — saturation (how "full" a resource is) is the fourth pillar and is often forgotten compared to the more familiar latency/traffic/errors.

**Related Material:** `lessons/mod-08-monitoring-observability/01-introduction-observability.md`

---

## Question 6
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

**Related Material:** `lessons/mod-08-monitoring-observability/02-prometheus-metrics.md`

---

## Question 7
**Q:** What is the purpose of a recording rule in Prometheus?

**Answer:** B) Precompute expensive queries and store results as new time series

**Explanation:**
Recording rules let you evaluate complex or frequently used PromQL expressions on a schedule and persist the results as new time series. This dramatically speeds up dashboards and alerts that would otherwise re-execute heavy aggregations on every query. They are the standard tool for keeping high-traffic Grafana panels responsive.

**Common Mistakes:**
- Picking C (record alert history) — alert history lives in Alertmanager / external storage, not recording rules.
- Picking D (backup metrics) — backups are handled by remote_write or snapshots, not recording rules.

**Related Material:** `lessons/mod-08-monitoring-observability/02-prometheus-metrics.md`

---

## Question 8
**Q:** What does `for: 5m` mean in a Prometheus alert rule?

**Answer:** C) Alert only fires if condition is true for 5 consecutive minutes

**Explanation:**
The `for` clause specifies how long an alert expression must continuously evaluate to true before the alert transitions from "pending" to "firing". This suppresses transient spikes and reduces noise from short-lived anomalies. Evaluation interval (how often the rule runs) is a separate setting (`evaluation_interval`).

**Common Mistakes:**
- Picking A (fires every 5 minutes) — `for` does not control notification cadence; that is configured in Alertmanager via `repeat_interval`.
- Picking B (evaluated every 5 minutes) — evaluation frequency is set globally, not by `for`.

**Related Material:** `lessons/mod-08-monitoring-observability/06-alerting-strategies.md`

---

## Question 9
**Q:** Which PromQL function should you use to calculate the 95th percentile latency from a histogram?

**Answer:** B) histogram_quantile()

**Explanation:**
`histogram_quantile(0.95, sum(rate(metric_bucket[5m])) by (le))` interpolates the 95th percentile from the cumulative bucket counters that Prometheus histograms expose. The function requires the `le` (less-than-or-equal) label, which is why histograms must be defined with cumulative buckets. There is no built-in `percentile_95()` in PromQL.

**Common Mistakes:**
- Picking A (avg_over_time) — averages hide tail latency, which is exactly what P95/P99 is designed to expose.
- Picking D (percentile_95) — this function does not exist in PromQL; it's a plausible-sounding distractor.

**Related Material:** `lessons/mod-08-monitoring-observability/02-prometheus-metrics.md`

---

## Question 10
**Q:** What is the main advantage of Prometheus's pull-based architecture?

**Answer:** B) Easier to detect when targets are down

**Explanation:**
Because Prometheus actively scrapes targets, a failed scrape is immediately visible as a missing `up` metric (value `0`), making target health a first-class concept. Pull also simplifies access control (only Prometheus needs to reach targets) and makes service discovery natural via Kubernetes / Consul integrations. In push systems, a silent agent can look identical to a healthy one with no traffic.

**Common Mistakes:**
- Picking A (lower network overhead) — pull vs push has similar overhead; pull can actually be heavier with many targets.
- Picking C (faster data collection) — speed is determined by scrape interval, not the pull/push model.

**Related Material:** `lessons/mod-08-monitoring-observability/02-prometheus-metrics.md`

---

## Question 11
**Q:** What is the purpose of variables in Grafana dashboards?

**Answer:** B) Create dynamic, reusable dashboards with filters

**Explanation:**
Grafana variables (template variables) let users select values at the top of a dashboard — e.g., environment, namespace, model name — that get substituted into every panel's query. This turns one dashboard into a reusable view across many entities, eliminating dashboard sprawl. They can be populated from data source queries, constants, or custom lists.

**Common Mistakes:**
- Picking A (store query results) — Grafana caches query results separately; variables hold selectable values, not result sets.
- Picking C (define alert thresholds) — thresholds are configured per-panel or per-alert rule, not via variables.

**Related Material:** `lessons/mod-08-monitoring-observability/03-grafana-visualization.md`

---

## Question 12
**Q:** Which Grafana panel type is best for showing a single current value with a threshold indicator?

**Answer:** C) Gauge

**Explanation:**
A gauge panel visualizes a single scalar value against configurable color thresholds (e.g., green/yellow/red bands), making the current state and "how close to danger" instantly readable. It is ideal for KPIs like utilization percentages or SLO budgets. Time series shows trends, tables show many values, and heatmaps show density distributions.

**Common Mistakes:**
- Picking A (Time series) — better for showing values over time, not a single instantaneous reading.
- Picking D (Heatmap) — heatmaps display distributions across two dimensions, not a single value.

**Related Material:** `lessons/mod-08-monitoring-observability/03-grafana-visualization.md`

---

## Question 13
**Q:** In Grafana, what does a transformation do?

**Answer:** B) Modifies query results before visualization

**Explanation:**
Transformations run client-side on the data returned by a query — joining series, renaming fields, filtering rows, computing derived columns, etc. — before the panel renders. They are the right tool when you cannot or do not want to push the logic into the underlying query language. The chain executes in order, and each step's output feeds the next.

**Common Mistakes:**
- Picking A (convert data source types) — Grafana does not transmute data sources; that is configured at the data-source level.
- Picking C (theme change) — themes are user/UI settings, unrelated to data.

**Related Material:** `lessons/mod-08-monitoring-observability/03-grafana-visualization.md`

---

## Question 14
**Q:** What is the purpose of provisioning in Grafana?

**Answer:** B) Automatically configure data sources and dashboards via files

**Explanation:**
Provisioning uses YAML/JSON files (typically mounted into the container) to declare data sources, dashboards, alerting rules, and notification channels so that Grafana can be reproduced as code. This enables GitOps workflows, version control of dashboards, and reliable disaster recovery. It removes click-ops drift between environments.

**Common Mistakes:**
- Picking A (allocate memory) — resource allocation is handled by the container runtime, not Grafana.
- Picking C (create user accounts) — user provisioning is a related but separate concern, typically tied to OAuth/LDAP.

**Related Material:** `lessons/mod-08-monitoring-observability/03-grafana-visualization.md`

---

## Question 15
**Q:** What is the main advantage of structured logging over unstructured logging?

**Answer:** C) Machine-parseable and queryable

**Explanation:**
Structured logs (typically JSON) carry explicit key-value fields, so log aggregators can index, filter, aggregate, and join on them without fragile regex extraction. This unlocks queries like "show all errors where `user_id=42` and `latency_ms>500`". Unstructured plain-text logs require parsing pipelines that break whenever the message format changes.

**Common Mistakes:**
- Picking A (easier for humans) — raw JSON is actually *harder* to skim than free text; tooling compensates.
- Picking B (less disk space) — JSON is typically larger than plain text due to key names and quoting.

**Related Material:** `lessons/mod-08-monitoring-observability/04-logging-elk-loki.md`

---

## Question 16
**Q:** In the ELK stack, what component is responsible for parsing and transforming log data?

**Answer:** B) Logstash

**Explanation:**
Logstash is the ETL engine of the ELK stack — it ingests events from many inputs, applies filter plugins (grok, mutate, geoip, etc.) to parse and enrich them, and forwards the results to Elasticsearch or other outputs. Elasticsearch stores and indexes; Kibana visualizes; Filebeat ships raw lines. Modern stacks often replace Logstash with lighter Beats + Elasticsearch ingest pipelines, but the question targets the classic ELK roles.

**Common Mistakes:**
- Picking A (Elasticsearch) — Elasticsearch indexes and searches, but parsing happens upstream in Logstash (or ingest pipelines).
- Picking D (Filebeat) — Filebeat is a lightweight shipper that forwards lines with minimal processing.

**Related Material:** `lessons/mod-08-monitoring-observability/04-logging-elk-loki.md`

---

## Question 17
**Q:** What is the primary difference between Loki and Elasticsearch?

**Answer:** B) Loki indexes only labels, not full text

**Explanation:**
Loki was designed "Prometheus-style" — it indexes a small set of labels per stream and stores log content compressed in object storage, then performs full-text matching at query time via grep-like filters. This drastically reduces index size and cost compared to Elasticsearch's full inverted index, at the trade-off of slower ad-hoc text searches. The result is a much cheaper logging stack for high-volume, label-rich workloads.

**Common Mistakes:**
- Picking A (Loki is slower) — Loki can actually be faster for label-scoped queries; full-text scans on huge ranges are where Elasticsearch wins.
- Picking D (more storage) — Loki uses *less* storage because it skips the full-text index.

**Related Material:** `lessons/mod-08-monitoring-observability/04-logging-elk-loki.md`

---

## Question 18
**Q:** What is a "span" in distributed tracing?

**Answer:** B) A single operation within a trace

**Explanation:**
A span represents one unit of work — a function call, RPC, DB query — with a start time, duration, attributes, and a parent span ID linking it into a tree. A trace is the collection of all spans sharing a trace ID, forming the request's path through services. Spans are the fundamental building block emitted by OpenTelemetry SDKs.

**Common Mistakes:**
- Picking A (total trace duration) — that is the trace's root-span duration, not a span itself.
- Picking D (network connection) — spans are logical operations, not transport-layer constructs.

**Related Material:** `lessons/mod-08-monitoring-observability/05-distributed-tracing.md`

---

## Question 19
**Q:** What protocol/framework has become the industry standard for instrumentation?

**Answer:** C) OpenTelemetry

**Explanation:**
OpenTelemetry (OTel) is the CNCF-graduated standard that unifies APIs, SDKs, and the OTLP wire protocol for metrics, logs, and traces across vendors and languages. It replaced the earlier OpenTracing and OpenCensus projects and is now supported by virtually every observability backend (Jaeger, Tempo, Datadog, etc.). Choosing OTel avoids vendor lock-in for instrumentation code.

**Common Mistakes:**
- Picking A (Zipkin) or B (Jaeger) — these are tracing *backends*; both now consume OTLP from OpenTelemetry SDKs.
- Picking D (Prometheus) — Prometheus is a metrics system, not a general instrumentation framework.

**Related Material:** `lessons/mod-08-monitoring-observability/05-distributed-tracing.md`

---

## Question 20
**Q:** What is the main advantage of Grafana Tempo over Jaeger?

**Answer:** C) Lower storage costs (object storage, minimal indexing)

**Explanation:**
Tempo stores traces directly in object storage (S3, GCS, Azure Blob) and indexes only the trace ID, relying on TraceQL plus correlation from metrics/logs to find interesting traces. This makes it dramatically cheaper than Jaeger's per-attribute indexing, especially at scale where 100% retention becomes feasible. The design choice fits the "store everything, query via correlation" paradigm.

**Common Mistakes:**
- Picking A (Better UI) — both use Grafana for visualization in modern stacks; UI is not the differentiator.
- Picking B (Faster queries) — Tempo can actually be slower for tag-based search; the win is cost, not speed.

**Related Material:** `lessons/mod-08-monitoring-observability/05-distributed-tracing.md`

---

## Question 21
**Q:** What should every alert include to be actionable?

**Answer:** B) Runbook URL or clear remediation steps

**Explanation:**
An actionable alert tells the responder *what is wrong, why it matters, and what to do about it*. Embedding a runbook link (or inline steps) in the alert annotations turns a 3 a.m. page from panic into procedure and shortens mean-time-to-resolve. Alerts without remediation guidance are a primary driver of alert fatigue and slow incident response.

**Common Mistakes:**
- Picking A (color coding) — color helps in dashboards but does nothing for an on-call engineer reading a Slack/PagerDuty message.
- Picking D (historical trend graph) — useful context, but not the single most important element of actionability.

**Related Material:** `lessons/mod-08-monitoring-observability/06-alerting-strategies.md`

---

## Question 22
**Q:** What is the purpose of inhibition rules in Alertmanager?

**Answer:** B) Suppress lower-priority alerts when higher-priority ones are firing

**Explanation:**
Inhibition rules let one alert silence another based on matching labels — for example, when a "cluster down" alert fires, you can inhibit all the "service unreachable" alerts that are merely symptoms. This prevents alert storms during cascading failures and keeps responders focused on the root cause. It is distinct from silences (manual, time-bounded muting) and grouping (batching notifications).

**Common Mistakes:**
- Picking A (prevent alerts from being sent) — that is too broad; silences serve that purpose. Inhibition is conditional on another alert firing.
- Picking C (slow down notifications) — that is `repeat_interval` / `group_wait`, not inhibition.

**Related Material:** `lessons/mod-08-monitoring-observability/06-alerting-strategies.md`

---

## Question 23
**Q:** What is "alert fatigue"?

**Answer:** B) Team ignoring alerts due to too many false positives

**Explanation:**
Alert fatigue is the desensitization that occurs when responders are flooded with noisy, low-value, or false-positive alerts; eventually they start ignoring or auto-acknowledging everything, including real incidents. It is the single largest threat to on-call effectiveness. The cure is ruthless tuning: every alert must be actionable, urgent, and well-scoped — otherwise convert it to a dashboard.

**Common Mistakes:**
- Picking A (alerts taking too long) — that is alert latency, a separate (and rarer) problem.
- Picking C (CPU usage) — alerts have negligible CPU cost; this is a distractor.

**Related Material:** `lessons/mod-08-monitoring-observability/06-alerting-strategies.md`

---

## Question 24
**Q:** What is the Population Stability Index (PSI) used for?

**Answer:** B) Detecting data drift between two distributions

**Explanation:**
PSI quantifies how much a feature's distribution has shifted between a reference period (often training data) and a comparison period (current production), by comparing bucketed proportions. Common thresholds: PSI < 0.1 = no significant drift, 0.1–0.25 = moderate, > 0.25 = significant drift requiring action. It's widely used in credit-risk models and is a staple of ML observability dashboards.

**Common Mistakes:**
- Picking A (model accuracy) — accuracy is measured against labeled ground truth, not via PSI.
- Picking D (GPU utilization) — that is an infrastructure metric, unrelated to statistical drift.

**Related Material:** `lessons/mod-08-monitoring-observability/07-ml-model-observability.md`

---

## Question 25
**Q:** What does SHAP stand for in the context of ML explainability?

**Answer:** B) SHapley Additive exPlanations

**Explanation:**
SHAP applies Shapley values from cooperative game theory to attribute each feature's marginal contribution to a model's prediction, producing additive, locally-accurate explanations. It is widely used for both global feature importance and per-prediction interpretability, and integrates with frameworks like XGBoost, LightGBM, and PyTorch. Monitoring SHAP values over time can reveal feature-importance drift even when raw distributions look stable.

**Common Mistakes:**
- Picking A or C — these are plausible-sounding distractors built from arbitrary words; SHAP specifically derives from Lloyd Shapley.
- Picking D (System Health and Performance) — confuses an ML interpretability technique with a generic ops acronym.

**Related Material:** `lessons/mod-08-monitoring-observability/07-ml-model-observability.md`

---
