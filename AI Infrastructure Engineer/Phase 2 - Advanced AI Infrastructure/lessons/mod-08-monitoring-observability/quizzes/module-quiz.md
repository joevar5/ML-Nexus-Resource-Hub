# Module 08: Monitoring & Observability - Quiz

## Instructions
- This quiz covers all lessons in Module 08
- 25 multiple choice questions + 3 practical scenarios
- Passing score: 80% (20/25 correct)
- Estimated time: 45 minutes

---

## Section 1: Fundamentals (Questions 1-5)

### Question 1
What are the three pillars of observability?

A) Monitoring, Logging, Tracing
B) Metrics, Logs, Traces
C) Alerts, Dashboards, Reports
D) CPU, Memory, Disk

**Answer: B**

### Question 2
Which metric type in Prometheus can only increase over time?

A) Gauge
B) Histogram
C) Counter
D) Summary

**Answer: C**

### Question 3
What is the primary difference between monitoring and observability?

A) Monitoring is newer than observability
B) Monitoring answers "what is broken", observability answers "why it's broken"
C) Monitoring is only for metrics, observability includes logs
D) There is no difference

**Answer: B**

### Question 4
In the context of ML systems, what does "data drift" refer to?

A) Gradual decrease in model accuracy over time
B) Changes in input data distribution compared to training data
C) Network latency variations
D) Storage capacity reduction

**Answer: B**

### Question 5
Which of the following is NOT one of the "Four Golden Signals"?

A) Latency
B) Traffic
C) Security
D) Saturation

**Answer: C**

---

## Section 2: Prometheus & PromQL (Questions 6-10)

### Question 6
What does this PromQL query calculate?
```promql
rate(http_requests_total[5m])
```

A) Total requests in last 5 minutes
B) Average requests per second over last 5 minutes
C) Current request rate
D) Number of requests per minute

**Answer: B**

### Question 7
What is the purpose of a recording rule in Prometheus?

A) Record all metrics to a file
B) Precompute expensive queries and store results as new time series
C) Record alert history
D) Create backup copies of metrics

**Answer: B**

### Question 8
What does `for: 5m` mean in a Prometheus alert rule?

A) Alert fires every 5 minutes
B) Alert is evaluated every 5 minutes
C) Alert only fires if condition is true for 5 consecutive minutes
D) Alert expires after 5 minutes

**Answer: C**

### Question 9
Which PromQL function should you use to calculate the 95th percentile latency from a histogram?

A) avg_over_time()
B) histogram_quantile()
C) rate()
D) percentile_95()

**Answer: B**

### Question 10
What is the main advantage of Prometheus's pull-based architecture?

A) Lower network overhead
B) Easier to detect when targets are down
C) Faster data collection
D) Simpler client configuration

**Answer: B**

---

## Section 3: Grafana & Visualization (Questions 11-14)

### Question 11
What is the purpose of variables in Grafana dashboards?

A) Store temporary query results
B) Create dynamic, reusable dashboards with filters
C) Define alert thresholds
D) Configure data source connections

**Answer: B**

### Question 12
Which Grafana panel type is best for showing a single current value with a threshold indicator?

A) Time series
B) Table
C) Gauge
D) Heatmap

**Answer: C**

### Question 13
In Grafana, what does a transformation do?

A) Converts between data source types
B) Modifies query results before visualization
C) Changes dashboard theme
D) Translates text to different languages

**Answer: B**

### Question 14
What is the purpose of provisioning in Grafana?

A) Allocate more memory to Grafana
B) Automatically configure data sources and dashboards via files
C) Create user accounts
D) Schedule dashboard refreshes

**Answer: B**

---

## Section 4: Logging (Questions 15-17)

### Question 15
What is the main advantage of structured logging over unstructured logging?

A) Easier for humans to read
B) Takes less disk space
C) Machine-parseable and queryable
D) Faster to write

**Answer: C**

### Question 16
In the ELK stack, what component is responsible for parsing and transforming log data?

A) Elasticsearch
B) Logstash
C) Kibana
D) Filebeat

**Answer: B**

### Question 17
What is the primary difference between Loki and Elasticsearch?

A) Loki is slower
B) Loki indexes only labels, not full text
C) Loki doesn't support queries
D) Loki requires more storage

**Answer: B**

---

## Section 5: Distributed Tracing (Questions 18-20)

### Question 18
What is a "span" in distributed tracing?

A) The total duration of a trace
B) A single operation within a trace
C) The time between two log messages
D) A network connection

**Answer: B**

### Question 19
What protocol/framework has become the industry standard for instrumentation?

A) Zipkin
B) Jaeger
C) OpenTelemetry
D) Prometheus

**Answer: C**

### Question 20
What is the main advantage of Grafana Tempo over Jaeger?

A) Better UI
B) Faster queries
C) Lower storage costs (object storage, minimal indexing)
D) More features

**Answer: C**

---

## Section 6: Alerting (Questions 21-23)

### Question 21
What should every alert include to be actionable?

A) Color coding
B) Runbook URL or clear remediation steps
C) List of all team members
D) Historical trend graph

**Answer: B**

### Question 22
What is the purpose of inhibition rules in Alertmanager?

A) Prevent alerts from being sent
B) Suppress lower-priority alerts when higher-priority ones are firing
C) Slow down alert notifications
D) Block external notifications

**Answer: B**

### Question 23
What is "alert fatigue"?

A) Alerts taking too long to send
B) Team ignoring alerts due to too many false positives
C) Alerts consuming too much CPU
D) Alerts expiring before being acknowledged

**Answer: B**

---

## Section 7: ML-Specific Monitoring (Questions 24-25)

### Question 24
What is the Population Stability Index (PSI) used for?

A) Measuring model accuracy
B) Detecting data drift between two distributions
C) Calculating training loss
D) Monitoring GPU utilization

**Answer: B**

### Question 25
What does SHAP stand for in the context of ML explainability?

A) Statistical Hypothesis And Prediction
B) SHapley Additive exPlanations
C) Sampling Heuristic for Accurate Predictions
D) System Health and Performance

**Answer: B**

---

## Section 8: Practical Scenarios (Short Answer)

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

- **Questions 1-25:** 1 point each (25 points total)
- **Scenarios 1-3:** Bonus points for comprehensive answers
- **Passing score:** 20/25 (80%)

## Answer Key Summary

1. B  | 2. C  | 3. B  | 4. B  | 5. C
6. B  | 7. B  | 8. C  | 9. B  | 10. B
11. B | 12. C | 13. B | 14. B | 15. C
16. B | 17. B | 18. B | 19. C | 20. C
21. B | 22. B | 23. B | 24. B | 25. B

---

## Additional Study Resources

If you scored < 80%, review:
- **Prometheus basics** (Questions 6-10): Lesson 02
- **Grafana** (Questions 11-14): Lesson 03
- **Logging** (Questions 15-17): Lesson 04
- **Tracing** (Questions 18-20): Lesson 05
- **Alerting** (Questions 21-23): Lesson 06
- **ML Monitoring** (Questions 24-25): Lesson 07

---

**Good luck!** 🚀
