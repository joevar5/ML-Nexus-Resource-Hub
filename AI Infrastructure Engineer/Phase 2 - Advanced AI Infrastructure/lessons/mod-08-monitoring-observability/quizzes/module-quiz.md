# Module 08: Monitoring & Observability - Quiz

## Instructions
- This quiz covers the four lessons in this module: Prometheus, distributed tracing, ML model observability, and SLOs
- 10 multiple choice questions + 3 practical scenarios
- Passing score: 80% (8/10 correct)
- Estimated time: 25 minutes

---

## Section 1: Prometheus & PromQL (Questions 1-5)

### Question 1
What does this PromQL query calculate?
```promql
rate(http_requests_total[5m])
```

A) Total requests in last 5 minutes
B) Average requests per second over last 5 minutes
C) Current request rate
D) Number of requests per minute

**Answer: B**

### Question 2
What is the purpose of a recording rule in Prometheus?

A) Record all metrics to a file
B) Precompute expensive queries and store results as new time series
C) Record alert history
D) Create backup copies of metrics

**Answer: B**

### Question 3
What does `for: 5m` mean in a Prometheus alert rule?

A) Alert fires every 5 minutes
B) Alert is evaluated every 5 minutes
C) Alert only fires if condition is true for 5 consecutive minutes
D) Alert expires after 5 minutes

**Answer: C**

### Question 4
Which PromQL function should you use to calculate the 95th percentile latency from a histogram?

A) avg_over_time()
B) histogram_quantile()
C) rate()
D) percentile_95()

**Answer: B**

### Question 5
What is the main advantage of Prometheus's pull-based architecture?

A) Lower network overhead
B) Easier to detect when targets are down
C) Faster data collection
D) Simpler client configuration

**Answer: B**

---

## Section 2: Distributed Tracing (Questions 6-8)

### Question 6
What is a "span" in distributed tracing?

A) The total duration of a trace
B) A single operation within a trace
C) The time between two log messages
D) A network connection

**Answer: B**

### Question 7
What protocol/framework has become the industry standard for instrumentation?

A) Zipkin
B) Jaeger
C) OpenTelemetry
D) Prometheus

**Answer: C**

### Question 8
What is the main advantage of Grafana Tempo over Jaeger?

A) Better UI
B) Faster queries
C) Lower storage costs (object storage, minimal indexing)
D) More features

**Answer: C**

---

## Section 3: ML-Specific Monitoring (Questions 9-10)

### Question 9
What is the Population Stability Index (PSI) used for?

A) Measuring model accuracy
B) Detecting data drift between two distributions
C) Calculating training loss
D) Monitoring GPU utilization

**Answer: B**

### Question 10
What does SHAP stand for in the context of ML explainability?

A) Statistical Hypothesis And Prediction
B) SHapley Additive exPlanations
C) Sampling Heuristic for Accurate Predictions
D) System Health and Performance

**Answer: B**

---

## Section 4: Practical Scenarios (Short Answer)

### Scenario 1: High Latency Investigation

**Situation:** Your ML model serving endpoint suddenly has P99 latency increase from 100ms to 2000ms.

**Question:** List 5 metrics/logs/traces you would check to diagnose the issue, and explain why.

**Sample Answer:**
1. **GPU utilization** (`gpu_utilization`) - Check if GPUs are saturated (>95%)
2. **GPU memory** (`gpu_memory_used/total`) - Check for memory exhaustion
3. **Batch size distribution** (`histogram_quantile(batch_size)`) - Unusually large batches cause slower processing
4. **Error logs** (search for "ERROR" or "OOM") - Identify crashes or memory errors
5. **Traces** (filter by high duration) - See which stage of pipeline is slow (preprocessing, inference, postprocessing)
6. **Recent deployments** (`kubectl get events`) - Check if new version was deployed

### Scenario 2: Designing Alerts

**Situation:** You're setting up monitoring for a new fraud detection ML service that processes 10,000 transactions/second.

**Question:** Design 3 critical alerts (specify metric, threshold, duration, and severity).

**Sample Answer:**

1. **High Error Rate**
   - Metric: `(sum(rate(predictions_total{status="error"}[5m])) / sum(rate(predictions_total[5m]))) * 100`
   - Threshold: `> 5%`
   - Duration: `for: 5m`
   - Severity: Critical
   - Reason: >5% errors impacts many users

2. **Model Accuracy Drop**
   - Metric: `model_accuracy{dataset="validation"}`
   - Threshold: `< 0.90`
   - Duration: `for: 10m`
   - Severity: Critical
   - Reason: Below 90% accuracy means model is not performing adequately

3. **Prediction Throughput Low**
   - Metric: `sum(rate(predictions_total[5m]))`
   - Threshold: `< 5000` (50% of expected)
   - Duration: `for: 5m`
   - Severity: Warning
   - Reason: Significant drop in throughput may indicate scaling issues

### Scenario 3: Data Drift Detection

**Situation:** Your model's accuracy has degraded from 95% to 88% over the past week. You suspect data drift.

**Question:** Describe the steps you would take to confirm and quantify the drift.

**Sample Answer:**

1. **Collect reference data:** Get feature distributions from training set or last known good period

2. **Collect current data:** Gather recent production inference data (past week)

3. **Calculate drift metrics per feature:**
   - For numerical: Kolmogorov-Smirnov (KS) test, PSI
   - For categorical: Chi-squared test

4. **Identify drifted features:**
   ```python
   # PSI > 0.25 = significant drift
   # KS test p-value < 0.05 = drift detected
   ```

5. **Visualize distributions:**
   - Plot histograms comparing reference vs current
   - Create drift dashboard in Grafana

6. **Correlate with accuracy:**
   - Check if features with highest drift correlate with accuracy drop

7. **Root cause analysis:**
   - Check logs for data pipeline changes
   - Verify feature engineering logic
   - Check for seasonality or external events

8. **Action:**
   - Retrain model on recent data
   - Update feature engineering
   - Set up continuous drift monitoring

---

## Scoring Guide

- 10 questions and 3 scenarios, 1 point each (13 points)
- Passing score: 80% (11/13)
