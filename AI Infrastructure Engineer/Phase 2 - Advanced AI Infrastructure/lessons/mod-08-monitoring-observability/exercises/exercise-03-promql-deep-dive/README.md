# Exercise 03: PromQL Deep Dive

**Duration:** 2.5 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 01-02; Prometheus running

## Objective

Write 25 PromQL queries spanning instant vectors, range vectors, aggregations, histograms, recording rules, and subqueries — each against a real metric on a running cluster. Build a personal `PROMQL_CHEATSHEET.md` you'd keep next to your terminal.

## Why this matters

PromQL is the single most useful query language for an infrastructure engineer. Engineers who think in PromQL diagnose 10× faster than engineers who scroll through Grafana hoping a panel will explain the problem.

## Setup

Run the kube-prometheus-stack (per mod-101 lab 05) plus the iris-api with traffic load. You now have real per-pod, per-namespace, per-cluster metrics.

## The 25 queries

For each, write the query, run it in Prometheus UI, screenshot or copy the result, and write 1-2 sentences explaining what it tells you.

### Counters and rates (1-5)
1. Total HTTP requests across all services in the last hour.
2. Per-second request rate, grouped by HTTP method and status code, last 5 minutes.
3. Top 5 services by traffic in the last 10 minutes.
4. Difference: `irate` vs `rate` — when to use each (write a 2-sentence comparison and demonstrate with a bursty traffic pattern).
5. Per-pod CPU usage rate over the last 2 minutes.

### Histograms and quantiles (6-10)
6. p50 latency for `iris-api` over the last 5 minutes.
7. p95 latency for `iris-api` over the last 5 minutes, grouped by URL path.
8. Difference between p95 and p50 (latency spread).
9. How many requests took longer than 500ms in the last 10 minutes? (use `histogram_count` with bucket filtering)
10. SLI: fraction of requests under 200ms.

### Aggregations and operators (11-15)
11. Total requests across all pods, divided by the number of pods (per-pod average).
12. Container memory usage as a percentage of container limit.
13. Top 3 namespaces by total CPU usage.
14. Predict_linear: how full will the disk be in 4 hours at the current rate?
15. Whether ALL pods of a deployment are Ready (label_join + group_left trick).

### Subqueries and recording rules (16-20)
16. Max p95 latency over any 5-minute window in the past hour (subquery).
17. Average per-minute request rate over the last hour, then averaged over the last 15 minutes (nested aggregation).
18. Recording rule that pre-computes per-path request rate at 1-minute resolution.
19. Recording rule for cluster-wide error rate, evaluated every 30s.
20. Query whose runtime would be 100× faster if you used the recording rule from #18.

### Real diagnostics (21-25)
21. "Which pod is hogging the CPU right now?" — find the top 3 pods by CPU.
22. "We deployed at 14:00 — did error rate change?" — compare error rate 1 hour before and after a timestamp.
23. "Is the database the bottleneck?" — joint analysis of API latency and database query latency.
24. "Are restarts increasing?" — `changes()` over container restart counter, top 10 pods.
25. "What's our error budget burn rate right now?" — multi-window burn alert query (from mod-08 lab 05).

## Step-by-step

### Step 1 — Set up (15 min)
Ensure Prometheus + grafana + iris-api stack is running. Generate continuous load (e.g., `wrk -t 4 -c 32 -d 1h` against `/v1/predict`).

### Step 2 — Work through queries 1-5 (30 min)
Take screenshots. Write notes per query.

### Step 3 — Queries 6-10 (histograms) (45 min)
Histograms are the trickiest. Spend time on the bucket math; verify quantile estimates against a synthetic ground truth.

### Step 4 — Queries 11-15 (45 min)
Group operators (`group_left`, `group_right`) take practice. Use real label sets from your kube-state-metrics.

### Step 5 — Queries 16-20 (recording rules) (30 min)
Add the rules to `prometheus.yml`, reload, validate.

### Step 6 — Queries 21-25 (diagnostics) (30 min)
For each, frame it as a real ops question first. Then write the query.

### Step 7 — Compile cheatsheet (15 min)
`PROMQL_CHEATSHEET.md` with the top 10 queries you'd actually want at 3 AM.

## Deliverables

1. A populated `QUERIES.md` with all 25 queries + explanations + screenshots.
2. A `prometheus-rules.yml` with the 2 recording rules from steps 18-19.
3. `PROMQL_CHEATSHEET.md` (≤ 1 page).

## Validation

- [ ] All 25 queries return non-null results against your running stack.
- [ ] You can explain the difference between `rate` and `irate` correctly.
- [ ] Recording rule's pre-computed value matches the original query within float epsilon.
- [ ] You used `group_left` correctly in at least one query (a hard concept worth practicing).

## Stretch goals

- Add 10 more queries specific to **ML observability**: feature drift PSI, prediction-class distribution shift, model-version rollout monitoring.
- Translate the cheatsheet to a Grafana **dashboard JSON** you'd ship to a new team.
- Write a `promtool` test file validating one of your recording rules' correctness.

## Common pitfalls

- **`rate()` on a gauge** — Returns nonsense; rates apply to counters. Use `delta` for gauges.
- **Histogram quantile lies on sparse data** — < 100 samples per bucket = noise. Widen the window.
- **Subqueries are expensive** — Each subquery re-evaluates the inner expression; use recording rules instead for hot dashboards.
- **Forgetting `[5m]`** — Bare counter values are cumulative since process start. Always wrap in `rate()` / `increase()` over a window.
