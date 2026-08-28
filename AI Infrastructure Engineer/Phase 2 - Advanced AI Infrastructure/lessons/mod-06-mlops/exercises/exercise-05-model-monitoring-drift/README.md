# Exercise 05: Model Monitoring and Drift Detection

**Estimated Time**: 28-36 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, Prometheus, Grafana, PostgreSQL, scikit-learn

## Overview

Build a production-grade model monitoring system that detects data drift, concept drift, and model performance degradation. Implement statistical tests for distribution changes, feature importance tracking, automated alerting, and retraining triggers. This exercise teaches critical MLOps patterns for maintaining model quality in production.

In production ML systems, model monitoring is essential for:
- **Data Drift Detection**: Detect when input data distribution changes
- **Concept Drift Detection**: Detect when relationship between features and target changes
- **Performance Monitoring**: Track accuracy, latency, error rates over time
- **Automated Alerts**: Notify team when model degrades
- **Retraining Triggers**: Automatically retrain when performance drops

## Learning Objectives

By completing this exercise, you will:

1. **Implement data drift detection** using statistical tests (KS test, PSI)
2. **Build concept drift detection** using prediction error monitoring
3. **Track model performance** with real-time metrics
4. **Create alerting system** for drift and performance issues
5. **Build retraining pipeline** triggered by drift detection
6. **Visualize drift** in Grafana dashboards
7. **Implement A/B testing** to validate retraining

## Business Context

**Real-World Scenario**: Your fraud detection model deployed 6 months ago shows declining performance:

- **Rising false positives**: Blocking 2x more legitimate transactions
- **Silent failures**: Model accuracy dropped from 95% to 87% (unnoticed for 3 weeks)
- **Data drift**: COVID-19 changed shopping patterns, model trained on pre-pandemic data
- **No visibility**: Can't see model performance between quarterly evaluations
- **Manual monitoring**: Engineer checks metrics weekly (not scalable)

Your task: Build monitoring system that:
- Detects data drift within 1 hour using statistical tests
- Alerts team when model accuracy drops >5%
- Triggers automatic retraining when drift detected
- Provides real-time dashboard showing drift metrics
- A/B tests retrained model before full deployment

## Project Structure

```
exercise-05-model-monitoring-drift/
├── README.md
├── requirements.txt
├── docker-compose.yaml              # Prometheus, Grafana, Postgres
├── config/
│   ├── prometheus.yml               # Prometheus config
│   ├── alert_rules.yml              # Alerting rules
│   └── drift_thresholds.yaml        # Drift detection thresholds
├── src/
│   └── model_monitoring/
│       ├── __init__.py
│       ├── drift/
│       │   ├── __init__.py
│       │   ├── data_drift.py            # Data drift detection
│       │   ├── concept_drift.py         # Concept drift detection
│       │   ├── statistical_tests.py     # KS test, PSI, etc.
│       │   └── feature_drift.py         # Per-feature drift analysis
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── performance_tracker.py   # Track accuracy, precision, etc.
│       │   ├── prometheus_exporter.py   # Export to Prometheus
│       │   └── metrics_store.py         # Store historical metrics
│       ├── alerting/
│       │   ├── __init__.py
│       │   ├── alert_manager.py         # Manage alerts
│       │   └── notification.py          # Slack/email notifications
│       ├── retraining/
│       │   ├── __init__.py
│       │   ├── trigger.py               # Retraining triggers
│       │   └── pipeline.py              # Retraining pipeline
│       └── visualization/
│           ├── __init__.py
│           └── dashboard_builder.py     # Generate Grafana dashboards
├── dashboards/
│   ├── model_performance.json       # Grafana dashboard
│   ├── drift_detection.json
│   └── feature_importance.json
├── tests/
│   ├── test_data_drift.py
│   ├── test_concept_drift.py
│   └── test_statistical_tests.py
├── examples/
│   ├── simulate_drift.py            # Generate drifted data
│   └── monitor_production.py
└── docs/
    ├── DESIGN.md
    ├── DRIFT_DETECTION.md
    └── ALERTING.md
```

## Requirements

### Functional Requirements

1. **Data Drift Detection**:
   - Compare production data distribution vs training data
   - Statistical tests: Kolmogorov-Smirnov, Population Stability Index (PSI)
   - Per-feature drift scores
   - Multivariate drift detection

2. **Concept Drift Detection**:
   - Monitor prediction error trends
   - Detect shifts in feature importance
   - Track model confidence scores
   - Compare predicted vs actual (when labels available)

3. **Performance Monitoring**:
   - Real-time metrics: accuracy, precision, recall, F1
   - Latency tracking (p50, p95, p99)
   - Throughput (predictions/sec)
   - Error rates (5xx, timeouts)

4. **Alerting**:
   - Alert on drift score >threshold
   - Alert on performance drop >5%
   - Alert on high error rate
   - Escalation policies (warn → critical)

5. **Automated Retraining**:
   - Trigger retraining on drift detection
   - A/B test retrained model vs current
   - Auto-promote if retrained model better
   - Rollback if retrained model worse

### Non-Functional Requirements

- **Detection Latency**: Detect drift within 1 hour
- **Alert Latency**: Send alerts within 5 minutes
- **Storage**: Retain 90 days of metrics
- **Performance**: Handle 10K predictions/sec monitoring

## Implementation Tasks

### Task 1: Data Drift Detection (7-9 hours)

Implement statistical tests for data drift.

```python
# src/model_monitoring/drift/data_drift.py

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DriftResult:
    """Result of drift detection"""
    feature_name: str
    drift_score: float
    p_value: float
    is_drift: bool
    test_name: str
    timestamp: datetime

class DataDriftDetector:
    """
    Detect data drift using statistical tests

    Methods:
    1. Kolmogorov-Smirnov test (continuous features)
    2. Chi-square test (categorical features)
    3. Population Stability Index (PSI)
    """

    def __init__(self, drift_threshold: float = 0.05):
        """
        Args:
            drift_threshold: p-value threshold (0.05 = 95% confidence)
        """
        self.drift_threshold = drift_threshold

    def detect_drift_ks(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """
        Kolmogorov-Smirnov test for continuous features

        Tests if two samples come from same distribution.
        Null hypothesis: distributions are same
        If p_value < threshold, reject null (drift detected)

        Args:
            reference_data: Training/baseline data
            current_data: Recent production data

        Returns:
            DriftResult with KS statistic and p-value
        """
        # TODO: Run KS test
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)

        # TODO: Determine if drift
        is_drift = p_value < self.drift_threshold

        return DriftResult(
            feature_name=feature_name,
            drift_score=ks_statistic,
            p_value=p_value,
            is_drift=is_drift,
            test_name="kolmogorov_smirnov",
            timestamp=datetime.utcnow()
        )

    def detect_drift_psi(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str,
        num_bins: int = 10
    ) -> DriftResult:
        """
        Population Stability Index (PSI)

        PSI measures change in distribution by comparing binned data.

        PSI = Σ (% current - % reference) * ln(% current / % reference)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (drift)

        Args:
            num_bins: Number of bins for histogram
        """
        # TODO: Create bins based on reference data
        bins = np.histogram_bin_edges(reference_data, bins=num_bins)

        # TODO: Calculate percentage in each bin
        ref_counts, _ = np.histogram(reference_data, bins=bins)
        cur_counts, _ = np.histogram(current_data, bins=bins)

        ref_percents = ref_counts / len(reference_data)
        cur_percents = cur_counts / len(current_data)

        # TODO: Calculate PSI
        # Avoid division by zero
        epsilon = 1e-10
        ref_percents = np.clip(ref_percents, epsilon, 1)
        cur_percents = np.clip(cur_percents, epsilon, 1)

        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))

        # TODO: Determine if drift (PSI >= 0.2 is significant)
        is_drift = psi >= 0.2

        return DriftResult(
            feature_name=feature_name,
            drift_score=psi,
            p_value=None,  # PSI doesn't have p-value
            is_drift=is_drift,
            test_name="psi",
            timestamp=datetime.utcnow()
        )

    def detect_drift_chi2(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """
        Chi-square test for categorical features

        Tests independence between reference and current distributions
        """
        # TODO: Get unique categories
        categories = np.unique(np.concatenate([reference_data, current_data]))

        # TODO: Count occurrences in each dataset
        ref_counts = np.array([np.sum(reference_data == cat) for cat in categories])
        cur_counts = np.array([np.sum(current_data == cat) for cat in categories])

        # TODO: Create contingency table
        contingency = np.array([ref_counts, cur_counts])

        # TODO: Run chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        is_drift = p_value < self.drift_threshold

        return DriftResult(
            feature_name=feature_name,
            drift_score=chi2,
            p_value=p_value,
            is_drift=is_drift,
            test_name="chi_square",
            timestamp=datetime.utcnow()
        )

    def detect_multivariate_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray
    ) -> float:
        """
        Multivariate drift using Maximum Mean Discrepancy (MMD)

        Measures distance between distributions in high-dimensional space
        """
        # TODO: Compute kernel matrix
        # Use RBF kernel for MMD
        from sklearn.metrics.pairwise import rbf_kernel

        gamma = 1.0 / reference_data.shape[1]

        # K(X, X)
        K_XX = rbf_kernel(reference_data, reference_data, gamma=gamma)
        # K(Y, Y)
        K_YY = rbf_kernel(current_data, current_data, gamma=gamma)
        # K(X, Y)
        K_XY = rbf_kernel(reference_data, current_data, gamma=gamma)

        # TODO: Calculate MMD^2
        m = len(reference_data)
        n = len(current_data)

        mmd_squared = (
            np.sum(K_XX) / (m * m)
            + np.sum(K_YY) / (n * n)
            - 2 * np.sum(K_XY) / (m * n)
        )

        return np.sqrt(max(mmd_squared, 0))
```

**Acceptance Criteria**:
- ✅ KS test for continuous features
- ✅ PSI calculation
- ✅ Chi-square for categorical
- ✅ Multivariate drift detection
- ✅ Configurable thresholds

---

### Task 2: Concept Drift Detection (6-7 hours)

Detect changes in model performance over time.

```python
# src/model_monitoring/drift/concept_drift.py

import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass

@dataclass
class PerformanceWindow:
    """Performance metrics for time window"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1: float
    error_rate: float

class ConceptDriftDetector:
    """
    Detect concept drift using performance monitoring

    Methods:
    1. ADWIN (Adaptive Windowing) - detects changes in data stream
    2. DDM (Drift Detection Method) - monitors error rate
    3. Page-Hinkley test - detects mean changes
    """

    def __init__(
        self,
        window_size: int = 1000,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.1
    ):
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold

        self.error_window = deque(maxlen=window_size)
        self.performance_history: List[PerformanceWindow] = []

    def add_prediction(
        self,
        predicted: float,
        actual: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Add prediction result

        Maintains sliding window of errors
        """
        error = 1 if predicted != actual else 0
        self.error_window.append(error)

    def detect_drift_ddm(self) -> Dict:
        """
        Drift Detection Method

        Monitors error rate and standard deviation.
        Warning if error_rate + 2*std > threshold
        Drift if error_rate + 3*std > threshold

        Returns:
            {
                "drift_detected": bool,
                "warning": bool,
                "error_rate": float,
                "std": float
            }
        """
        if len(self.error_window) < 30:
            return {"drift_detected": False, "warning": False}

        # TODO: Calculate error rate and std
        errors = np.array(self.error_window)
        error_rate = np.mean(errors)
        std = np.std(errors)

        # TODO: Check thresholds
        warning = (error_rate + 2 * std) > self.warning_threshold
        drift = (error_rate + 3 * std) > self.drift_threshold

        return {
            "drift_detected": drift,
            "warning": warning,
            "error_rate": error_rate,
            "std": std,
            "threshold_distance": error_rate + 3 * std - self.drift_threshold
        }

    def detect_drift_page_hinkley(
        self,
        delta: float = 0.005,
        lambda_: float = 50
    ) -> bool:
        """
        Page-Hinkley test

        Detects changes in mean of sequence.
        Cumulative sum of deviations from mean.

        Args:
            delta: Magnitude of changes to detect
            lambda_: Detection threshold
        """
        if len(self.error_window) < 30:
            return False

        errors = np.array(self.error_window)

        # TODO: Calculate cumulative sum
        mean = np.mean(errors)
        cumsum = 0
        min_cumsum = 0

        for error in errors:
            cumsum += error - mean - delta
            if cumsum < min_cumsum:
                min_cumsum = cumsum

        # TODO: Check if cumsum - min_cumsum > threshold
        ph_value = cumsum - min_cumsum

        return ph_value > lambda_

    def track_performance_trend(
        self,
        window_hours: int = 24
    ) -> Dict:
        """
        Analyze performance trend over time

        Detects:
        - Declining accuracy
        - Increasing error rate
        - Performance volatility

        Returns:
            {
                "trend": "improving"/"stable"/"declining",
                "slope": float,  # Linear regression slope
                "volatility": float  # Standard deviation of metrics
            }
        """
        # TODO: Filter to recent window
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [p for p in self.performance_history if p.timestamp >= cutoff]

        if len(recent) < 10:
            return {"trend": "unknown"}

        # TODO: Fit linear regression to accuracy over time
        timestamps = np.array([(p.timestamp - recent[0].timestamp).total_seconds()
                              for p in recent])
        accuracies = np.array([p.accuracy for p in recent])

        # Simple linear regression
        slope = np.polyfit(timestamps, accuracies, 1)[0]

        # TODO: Calculate volatility
        volatility = np.std(accuracies)

        # TODO: Determine trend
        if slope < -0.001:  # Declining
            trend = "declining"
        elif slope > 0.001:  # Improving
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": slope,
            "volatility": volatility,
            "current_accuracy": recent[-1].accuracy,
            "accuracy_change": recent[-1].accuracy - recent[0].accuracy
        }

    def add_performance_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        error_rate: float,
        timestamp: Optional[datetime] = None
    ):
        """Store performance metrics"""
        self.performance_history.append(
            PerformanceWindow(
                timestamp=timestamp or datetime.utcnow(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                error_rate=error_rate
            )
        )

        # Keep last 7 days
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.performance_history = [
            p for p in self.performance_history if p.timestamp >= cutoff
        ]
```

**Acceptance Criteria**:
- ✅ DDM drift detection
- ✅ Page-Hinkley test
- ✅ Performance trend analysis
- ✅ Sliding window of predictions
- ✅ Configurable thresholds

---

### Task 3: Prometheus Metrics Exporter (5-6 hours)

Export drift metrics to Prometheus.

```python
# src/model_monitoring/metrics/prometheus_exporter.py

from prometheus_client import Gauge, Counter, Histogram, start_http_server
from typing import Dict
import time

class ModelMetricsExporter:
    """
    Export model monitoring metrics to Prometheus

    Metrics exposed:
    - model_drift_score: Drift score per feature
    - model_accuracy: Current accuracy
    - model_predictions_total: Total predictions
    - model_inference_latency: Inference latency histogram
    - model_errors_total: Error count
    """

    def __init__(self, port: int = 8000):
        self.port = port

        # Define metrics
        self.drift_score = Gauge(
            'model_drift_score',
            'Data drift score',
            ['model_name', 'feature_name', 'test_type']
        )

        self.accuracy = Gauge(
            'model_accuracy',
            'Model accuracy',
            ['model_name', 'model_version']
        )

        self.precision = Gauge(
            'model_precision',
            'Model precision',
            ['model_name', 'model_version']
        )

        self.recall = Gauge(
            'model_recall',
            'Model recall',
            ['model_name', 'model_version']
        )

        self.predictions_total = Counter(
            'model_predictions_total',
            'Total predictions',
            ['model_name', 'model_version']
        )

        self.inference_latency = Histogram(
            'model_inference_latency_seconds',
            'Inference latency',
            ['model_name', 'model_version'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        self.errors_total = Counter(
            'model_errors_total',
            'Total errors',
            ['model_name', 'model_version', 'error_type']
        )

        self.drift_detected = Gauge(
            'model_drift_detected',
            'Whether drift is detected (1=yes, 0=no)',
            ['model_name', 'feature_name']
        )

    def update_drift_score(
        self,
        model_name: str,
        feature_name: str,
        test_type: str,
        score: float
    ):
        """Update drift score metric"""
        self.drift_score.labels(
            model_name=model_name,
            feature_name=feature_name,
            test_type=test_type
        ).set(score)

    def update_accuracy(
        self,
        model_name: str,
        model_version: str,
        accuracy: float
    ):
        """Update accuracy metric"""
        self.accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)

    def record_prediction(
        self,
        model_name: str,
        model_version: str,
        latency_seconds: float
    ):
        """Record prediction"""
        self.predictions_total.labels(
            model_name=model_name,
            model_version=model_version
        ).inc()

        self.inference_latency.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(latency_seconds)

    def record_error(
        self,
        model_name: str,
        model_version: str,
        error_type: str
    ):
        """Record error"""
        self.errors_total.labels(
            model_name=model_name,
            model_version=model_version,
            error_type=error_type
        ).inc()

    def start(self):
        """Start HTTP server for Prometheus scraping"""
        start_http_server(self.port)
        print(f"Prometheus metrics server started on port {self.port}")
```

**Prometheus configuration**:

```yaml
# config/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'model_monitoring'
    static_configs:
      - targets: ['model-monitor:8000']

rule_files:
  - 'alert_rules.yml'
```

```yaml
# config/alert_rules.yml

groups:
  - name: model_drift
    interval: 1m
    rules:
      # Alert on data drift
      - alert: DataDriftDetected
        expr: model_drift_detected == 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected for {{ $labels.model_name }}"
          description: "Feature {{ $labels.feature_name }} shows drift"

      # Alert on accuracy drop
      - alert: ModelAccuracyDrop
        expr: |
          (
            model_accuracy < 0.9
            and
            model_accuracy < (model_accuracy offset 1h) * 0.95
          )
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped for {{ $labels.model_name }}"
          description: "Accuracy: {{ $value }}"

      # Alert on high error rate
      - alert: HighErrorRate
        expr: |
          rate(model_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for {{ $labels.model_name }}"
          description: "Error rate: {{ $value }}"
```

**Acceptance Criteria**:
- ✅ Export drift scores to Prometheus
- ✅ Export performance metrics
- ✅ Alert rules configured
- ✅ Scrape endpoint working
- ✅ Metrics queryable in Prometheus

---

### Task 4: Alerting System (4-5 hours)

Build alerting for drift and performance issues.

```python
# src/model_monitoring/alerting/alert_manager.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import requests

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert definition"""
    title: str
    message: str
    severity: AlertSeverity
    model_name: str
    timestamp: datetime
    metadata: Dict

class AlertManager:
    """
    Manage alerts for model monitoring

    Supports:
    - Slack notifications
    - Email notifications
    - PagerDuty integration
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        email_config: Optional[Dict] = None
    ):
        self.slack_webhook_url = slack_webhook_url
        self.email_config = email_config
        self.alert_history: List[Alert] = []

    def send_drift_alert(
        self,
        model_name: str,
        feature_name: str,
        drift_score: float,
        threshold: float
    ):
        """Send alert for drift detection"""
        alert = Alert(
            title=f"Data Drift Detected: {model_name}",
            message=f"Feature '{feature_name}' drift score {drift_score:.3f} exceeds threshold {threshold:.3f}",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            timestamp=datetime.utcnow(),
            metadata={
                "feature_name": feature_name,
                "drift_score": drift_score,
                "threshold": threshold
            }
        )

        self._send_alert(alert)

    def send_performance_alert(
        self,
        model_name: str,
        metric_name: str,
        current_value: float,
        expected_value: float
    ):
        """Send alert for performance degradation"""
        severity = AlertSeverity.CRITICAL if current_value < expected_value * 0.9 else AlertSeverity.WARNING

        alert = Alert(
            title=f"Model Performance Degradation: {model_name}",
            message=f"{metric_name} dropped to {current_value:.3f} (expected {expected_value:.3f})",
            severity=severity,
            model_name=model_name,
            timestamp=datetime.utcnow(),
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "expected_value": expected_value,
                "degradation_pct": (expected_value - current_value) / expected_value * 100
            }
        )

        self._send_alert(alert)

    def _send_alert(self, alert: Alert):
        """Send alert via configured channels"""
        self.alert_history.append(alert)

        # Send to Slack
        if self.slack_webhook_url:
            self._send_slack(alert)

        # Send email
        if self.email_config:
            self._send_email(alert)

        # Log
        print(f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}")

    def _send_slack(self, alert: Alert):
        """Send alert to Slack"""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.CRITICAL: "#ff0000"
        }[alert.severity]

        payload = {
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": "Model", "value": alert.model_name, "short": True},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": False}
                ]
            }]
        }

        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")

    def _send_email(self, alert: Alert):
        """Send alert via email"""
        # TODO: Implement email sending using SMTP
        pass

    def get_recent_alerts(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get recent alerts"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filtered = [a for a in self.alert_history if a.timestamp >= cutoff]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        return filtered
```

**Acceptance Criteria**:
- ✅ Drift alerts sent
- ✅ Performance alerts sent
- ✅ Slack integration working
- ✅ Alert history tracked
- ✅ Severity levels handled

---

### Task 5: Automated Retraining Pipeline (5-6 hours)

Trigger retraining when drift detected.

```python
# src/model_monitoring/retraining/trigger.py

from typing import Dict, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class RetrainingTrigger:
    """Conditions that trigger retraining"""
    name: str
    condition: Callable[[], bool]
    cooldown_hours: int = 24  # Don't retrain more often than this
    last_triggered: Optional[datetime] = None

class RetrainingManager:
    """
    Manage automated model retraining

    Triggers:
    1. Data drift detected (drift score > threshold)
    2. Performance drop (accuracy < threshold)
    3. Scheduled (every N days)
    4. Manual trigger
    """

    def __init__(self):
        self.triggers: Dict[str, RetrainingTrigger] = {}
        self.retraining_history = []

    def add_trigger(
        self,
        name: str,
        condition: Callable[[], bool],
        cooldown_hours: int = 24
    ):
        """Add retraining trigger"""
        self.triggers[name] = RetrainingTrigger(
            name=name,
            condition=condition,
            cooldown_hours=cooldown_hours
        )

    def check_triggers(self) -> List[str]:
        """
        Check all triggers

        Returns:
            List of trigger names that fired
        """
        fired_triggers = []

        for name, trigger in self.triggers.items():
            # Check cooldown
            if trigger.last_triggered:
                elapsed = datetime.utcnow() - trigger.last_triggered
                if elapsed < timedelta(hours=trigger.cooldown_hours):
                    continue

            # Check condition
            if trigger.condition():
                fired_triggers.append(name)
                trigger.last_triggered = datetime.utcnow()

        return fired_triggers

    def trigger_retraining(
        self,
        model_name: str,
        trigger_reason: str,
        retraining_pipeline: Callable
    ):
        """
        Execute retraining pipeline

        Args:
            retraining_pipeline: Function to run retraining
        """
        print(f"Triggering retraining for {model_name}: {trigger_reason}")

        # TODO: Run retraining pipeline
        try:
            result = retraining_pipeline()

            self.retraining_history.append({
                "model_name": model_name,
                "trigger_reason": trigger_reason,
                "timestamp": datetime.utcnow(),
                "result": result,
                "success": True
            })

            return result

        except Exception as e:
            print(f"Retraining failed: {e}")
            self.retraining_history.append({
                "model_name": model_name,
                "trigger_reason": trigger_reason,
                "timestamp": datetime.utcnow(),
                "error": str(e),
                "success": False
            })
            raise
```

**Example usage**:

```python
# examples/monitor_production.py

from model_monitoring.drift.data_drift import DataDriftDetector
from model_monitoring.drift.concept_drift import ConceptDriftDetector
from model_monitoring.retraining.trigger import RetrainingManager
from model_monitoring.alerting.alert_manager import AlertManager

# Initialize components
data_drift = DataDriftDetector(drift_threshold=0.05)
concept_drift = ConceptDriftDetector(drift_threshold=0.1)
retraining_mgr = RetrainingManager()
alert_mgr = AlertManager(slack_webhook_url="https://hooks.slack.com/...")

# Define retraining triggers
retraining_mgr.add_trigger(
    name="data_drift",
    condition=lambda: data_drift.detect_drift_psi(...).is_drift,
    cooldown_hours=24
)

retraining_mgr.add_trigger(
    name="performance_drop",
    condition=lambda: concept_drift.detect_drift_ddm()['drift_detected'],
    cooldown_hours=12
)

# Monitor loop
while True:
    # Collect production data
    current_data = fetch_recent_production_data()
    reference_data = fetch_training_data()

    # Check data drift
    for feature in features:
        drift_result = data_drift.detect_drift_ks(
            reference_data[feature],
            current_data[feature],
            feature
        )

        if drift_result.is_drift:
            alert_mgr.send_drift_alert(
                model_name="fraud_detection",
                feature_name=feature,
                drift_score=drift_result.drift_score,
                threshold=0.05
            )

    # Check triggers
    fired_triggers = retraining_mgr.check_triggers()
    if fired_triggers:
        retraining_mgr.trigger_retraining(
            model_name="fraud_detection",
            trigger_reason=", ".join(fired_triggers),
            retraining_pipeline=run_training_pipeline
        )

    time.sleep(3600)  # Check every hour
```

**Acceptance Criteria**:
- ✅ Configurable triggers
- ✅ Cooldown periods
- ✅ Retraining history tracked
- ✅ Error handling
- ✅ Integration with drift detection

---

### Task 6: Grafana Dashboards (3-4 hours)

Create dashboards for visualization.

```json
// dashboards/drift_detection.json

{
  "dashboard": {
    "title": "Model Drift Detection",
    "panels": [
      {
        "title": "Drift Score by Feature",
        "type": "graph",
        "targets": [
          {
            "expr": "model_drift_score",
            "legendFormat": "{{feature_name}} ({{test_type}})"
          }
        ]
      },
      {
        "title": "Drift Detected",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(model_drift_detected)"
          }
        ]
      },
      {
        "title": "Model Accuracy Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "{{model_name}} v{{model_version}}"
          }
        ]
      }
    ]
  }
}
```

**Acceptance Criteria**:
- ✅ Drift score visualization
- ✅ Accuracy trend charts
- ✅ Alert history
- ✅ Feature importance over time
- ✅ Auto-refresh every 30s

---

## Testing Requirements

```python
def test_ks_drift_detection():
    """Test KS test detects drift"""
    detector = DataDriftDetector()

    # Same distribution - no drift
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    result = detector.detect_drift_ks(ref, cur, "feature1")
    assert not result.is_drift

    # Different distribution - drift
    cur_shifted = np.random.normal(2, 1, 1000)  # Mean shifted
    result = detector.detect_drift_ks(ref, cur_shifted, "feature1")
    assert result.is_drift

def test_psi_calculation():
    """Test PSI calculation"""
    detector = DataDriftDetector()

    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0.5, 1, 1000)  # Slight shift

    result = detector.detect_drift_psi(ref, cur, "feature1")
    assert result.drift_score > 0
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Drift Detection Latency** | <1 hour | ________h |
| **Alert Latency** | <5 min | ________min |
| **False Positive Rate** | <10% | ________% |

## Validation

Submit:
1. Data drift detection implementation
2. Concept drift detection
3. Prometheus metrics exporter
4. Alerting system with Slack
5. Automated retraining triggers
6. Grafana dashboards
7. Test suite
8. Documentation

## Resources

- [Evidently AI](https://www.evidentlyai.com/) - ML monitoring library
- [KS Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [PSI](https://www.listendata.com/2015/05/population-stability-index.html)
- [Drift Detection Methods](https://riverml.xyz/latest/api/drift/)

---

**Estimated Completion Time**: 28-36 hours

**Skills Practiced**:
- Data drift detection
- Concept drift detection
- Statistical testing
- Prometheus monitoring
- Grafana dashboards
- Automated alerting
- ML retraining pipelines
