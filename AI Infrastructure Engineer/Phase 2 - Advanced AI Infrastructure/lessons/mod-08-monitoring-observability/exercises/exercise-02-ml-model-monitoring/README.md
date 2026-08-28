# Exercise 02: ML Model Monitoring and Drift Detection

**Estimated Time**: 30-38 hours

## Business Context

Your company operates 15 ML models in production serving 5 million predictions/day. Recent incidents highlight critical monitoring gaps:

**Incident #1 - Silent Model Degradation**:
- Fraud detection model accuracy dropped from 94% to 76% over 3 months
- Took 2 months to detect (discovered only during quarterly review)
- Cost: $2.3M in undetected fraud losses
- Root cause: Data drift (customer behavior changed post-pandemic)

**Incident #2 - Feature Store Outage**:
- Feature store cache failure caused 500ms → 5s latency spike
- P95 latency SLO violated (500ms target)
- Detected after 45 minutes (customer complaints)
- Cost: 12,000 failed requests, angry customers

**Incident #3 - Model Version Rollout Bug**:
- New model version (v3.2) had prediction bias against certain demographics
- Took 1 week to detect through manual review
- Cost: Regulatory investigation, PR crisis

The VP of ML has mandated a **comprehensive ML model monitoring system** that:
1. **Detects data drift** automatically within 24 hours
2. **Tracks model performance** in real-time (accuracy, precision, recall)
3. **Monitors prediction quality** (distribution, outliers, bias)
4. **Alerts on degradation** before customer impact
5. **Enables fast rollback** (< 5 minutes to previous model version)

## Learning Objectives

After completing this exercise, you will be able to:

1. Implement comprehensive ML model performance monitoring
2. Detect data drift using statistical tests (KS test, PSI, Jensen-Shannon divergence)
3. Track prediction distributions and detect anomalies
4. Monitor feature importance and detect feature drift
5. Implement A/B testing infrastructure for model comparisons
6. Build automated model rollback based on performance degradation
7. Create ML-specific dashboards and alerts

## Prerequisites

- Module 06 Exercise 05 (Model Monitoring & Drift) - concepts
- Module 08 Exercise 01 (Observability Stack) - infrastructure
- Understanding of ML model evaluation metrics
- Python programming (intermediate to advanced)
- Basic statistics knowledge

## Problem Statement

Build an **ML Model Monitoring System** that:

1. **Tracks model performance** continuously (accuracy, latency, throughput)
2. **Detects data drift** in input features using statistical tests
3. **Monitors prediction distributions** for anomalies
4. **Implements A/B testing** for safe model rollouts
5. **Automates rollback** when performance degrades
6. **Provides visibility** through dashboards and alerts

### Success Metrics

- Data drift detected within 24 hours (vs 2 months baseline)
- Model performance tracked in real-time (<1 minute lag)
- Automated rollback within 5 minutes of performance degradation
- Zero silent model failures (100% detection rate)
- A/B test infrastructure supporting 10+ concurrent tests
- ML monitoring dashboard load time <3 seconds

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│              ML Model Monitoring System                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Prediction      │    │  Reference       │                 │
│  │  Service         │───▶│  Data Store      │                 │
│  │  (ML Model)      │    │  (PostgreSQL)    │                 │
│  └──────────────────┘    └──────────────────┘                 │
│           │                        │                            │
│           │ Predictions            │ Baseline stats             │
│           ▼                        ▼                            │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Drift Detector  │◀───│  Statistical     │                 │
│  │  - KS test       │    │  Tests           │                 │
│  │  - PSI           │    │  - Compare       │                 │
│  │  - JS divergence │    │    distributions │                 │
│  └──────────────────┘    └──────────────────┘                 │
│           │                                                     │
│           │ Drift alerts                                        │
│           ▼                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Performance     │    │  A/B Test        │                 │
│  │  Tracker         │◀───│  Controller      │                 │
│  │  - Accuracy      │    │  - Traffic split │                 │
│  │  - Precision     │    │  - Metrics       │                 │
│  │  - Recall        │    │    comparison    │                 │
│  └──────────────────┘    └──────────────────┘                 │
│           │                        │                            │
│           │                        │                            │
│           ▼                        ▼                            │
│  ┌──────────────────┐    ┌──────────────────┐                 │
│  │  Auto Rollback   │    │  Prometheus      │                 │
│  │  Engine          │◀───│  (Metrics)       │                 │
│  │  - Detect degr.  │    │                  │                 │
│  │  - Rollback v.   │    └──────────────────┘                 │
│  └──────────────────┘             │                            │
│           │                        │                            │
│           │                        ▼                            │
│           │               ┌──────────────────┐                 │
│           └──────────────▶│  Grafana         │                 │
│                           │  ML Dashboards   │                 │
│                           └──────────────────┘                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: Data Drift Detection (8-10 hours)

Implement statistical tests to detect when input data distribution changes.

#### 1.1 Drift Detector

Create `src/drift_detection/drift_detector.py`:

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftResult:
    feature_name: str
    drift_detected: bool
    severity: DriftSeverity
    test_statistic: float
    p_value: float
    threshold: float
    description: str

class KolmogorovSmirnovTest:
    """
    Kolmogorov-Smirnov test for continuous feature drift.

    Compares two distributions (reference vs current) to detect drift.

    Good for:
    - Continuous numerical features
    - Detecting any distribution change

    Not good for:
    - Categorical features
    - Small sample sizes (<100 samples)
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """
        TODO: Perform KS test to detect drift

        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(reference_data, current_data)

        # Interpretation:
        # - p_value < 0.05: Distributions are different (drift detected)
        # - p_value >= 0.05: No significant drift

        # Severity based on p-value:
        # - p < 0.001: CRITICAL
        # - p < 0.01: HIGH
        # - p < 0.05: MEDIUM
        # - p >= 0.05: NONE

        Return DriftResult with test results
        """
        pass

class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for categorical and continuous features.

    PSI measures how much the distribution has shifted.

    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate drift (investigate)
    - PSI >= 0.2: Significant drift (action required)

    Good for:
    - Both categorical and continuous features
    - Easy to interpret (single number)
    - Industry standard in credit risk

    Formula:
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    """

    THRESHOLDS = {
        'low': 0.1,
        'medium': 0.15,
        'high': 0.2
    }

    def calculate_psi(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        TODO: Calculate Population Stability Index

        Steps:
        1. Create bins from reference data:
           bins = np.percentile(reference_data, np.linspace(0, 100, num_bins + 1))

        2. Calculate % of samples in each bin for reference and current:
           ref_percents = histogram of reference data
           cur_percents = histogram of current data

        3. Calculate PSI:
           psi = 0
           for i in range(num_bins):
               psi += (cur_percents[i] - ref_percents[i]) * np.log(cur_percents[i] / ref_percents[i])

        Handle edge cases:
        - If bin has 0 samples, add small epsilon (1e-6) to avoid log(0)

        Return PSI value
        """
        pass

    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """
        TODO: Detect drift using PSI

        psi = self.calculate_psi(reference_data, current_data)

        # Determine severity
        if psi < self.THRESHOLDS['low']:
            severity = DriftSeverity.NONE
        elif psi < self.THRESHOLDS['medium']:
            severity = DriftSeverity.LOW
        elif psi < self.THRESHOLDS['high']:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.CRITICAL

        Return DriftResult
        """
        pass

class JensenShannonDivergence:
    """
    Jensen-Shannon Divergence for distribution comparison.

    Symmetric version of KL divergence.
    Bounded: 0 <= JS <= 1

    Good for:
    - Comparing probability distributions
    - More stable than KL divergence
    - Works for categorical and continuous features

    JSD Interpretation:
    - JSD < 0.1: Distributions very similar
    - 0.1 <= JSD < 0.3: Moderate difference
    - JSD >= 0.3: Significant difference
    """

    def calculate_jsd(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        num_bins: int = 50
    ) -> float:
        """
        TODO: Calculate Jensen-Shannon Divergence

        Steps:
        1. Create histograms (probability distributions):
           bins = np.linspace(min(both), max(both), num_bins)
           p, _ = np.histogram(reference_data, bins=bins, density=True)
           q, _ = np.histogram(current_data, bins=bins, density=True)

           # Normalize to probabilities
           p = p / np.sum(p)
           q = q / np.sum(q)

        2. Calculate JSD:
           m = 0.5 * (p + q)
           jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

           where kl_divergence(a, b) = Σ a[i] × log(a[i] / b[i])

        3. Add epsilon to avoid log(0)

        Return JSD value
        """
        pass

class DriftDetector:
    """Main drift detection coordinator."""

    def __init__(self):
        self.ks_test = KolmogorovSmirnovTest()
        self.psi_test = PopulationStabilityIndex()
        self.jsd_test = JensenShannonDivergence()

    def detect_multivariate_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, DriftResult]:
        """
        TODO: Detect drift across multiple features

        For each feature:
        1. Extract reference and current data
        2. Run multiple drift tests:
           - KS test (continuous features)
           - PSI (all features)
           - JSD (all features)

        3. Aggregate results (drift detected if ANY test shows drift)

        Return dict: feature_name -> DriftResult
        """
        results = {}

        for feature in features:
            ref_data = reference_df[feature].values
            cur_data = current_df[feature].values

            # TODO: Run tests
            ks_result = self.ks_test.detect_drift(ref_data, cur_data, feature)
            psi_result = self.psi_test.detect_drift(ref_data, cur_data, feature)
            jsd_result = self.jsd_test.detect_drift(ref_data, cur_data, feature)

            # TODO: Aggregate (use most severe result)
            # ...

            results[feature] = ...

        return results

    def generate_drift_report(self, drift_results: Dict[str, DriftResult]) -> str:
        """
        TODO: Generate human-readable drift report

        Example:
        === Data Drift Detection Report ===
        Date: 2023-10-25 14:30:00 UTC

        CRITICAL Drift Detected (3 features):
        1. transaction_amount: PSI=0.45 (threshold: 0.2)
           - Distribution has shifted significantly
           - Recommend: Retrain model or investigate data source

        2. merchant_category: KS test p-value=0.001
           - Category distribution changed
           - Recommend: Check for new merchant categories

        MEDIUM Drift (2 features):
        ...

        NO Drift (10 features):
        ...

        Overall Status: ACTION REQUIRED
        Recommended Action: Retrain model within 7 days
        """
        pass
```

#### 1.2 Continuous Monitoring

Create `src/drift_detection/drift_monitor.py`:

```python
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd

class ContinuousDriftMonitor:
    """
    Continuously monitor for data drift.

    Runs drift detection every N hours and alerts on drift.
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        reference_data_path: str,
        check_interval_hours: int = 6
    ):
        self.drift_detector = drift_detector
        self.reference_data = pd.read_parquet(reference_data_path)
        self.check_interval_hours = check_interval_hours

    def fetch_recent_predictions(
        self,
        hours: int = 24
    ) -> pd.DataFrame:
        """
        TODO: Fetch recent prediction data from database

        Query PostgreSQL for predictions from last N hours:

        SELECT
            feature_1,
            feature_2,
            ...
            prediction,
            prediction_timestamp
        FROM predictions
        WHERE prediction_timestamp > NOW() - INTERVAL '{hours} hours'
        LIMIT 10000  -- Sufficient sample size for statistical tests

        Return DataFrame with same schema as reference_data
        """
        pass

    def run_drift_check(self):
        """
        TODO: Run drift detection check

        1. Fetch recent predictions
        2. Run drift detector
        3. If drift detected:
           - Log to Prometheus metrics
           - Send alert to Slack/PagerDuty
           - Store drift report
        4. Update drift status dashboard
        """
        print(f"[{datetime.now()}] Running drift detection check...")

        # TODO: Fetch data
        current_data = self.fetch_recent_predictions(hours=24)

        if len(current_data) < 1000:
            print(f"Insufficient data: {len(current_data)} samples (need 1000+)")
            return

        # TODO: Run drift detection
        features = [col for col in current_data.columns if col not in ['prediction', 'timestamp']]
        drift_results = self.drift_detector.detect_multivariate_drift(
            self.reference_data,
            current_data,
            features
        )

        # TODO: Check for critical drift
        critical_drifts = [
            feature for feature, result in drift_results.items()
            if result.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]
        ]

        if critical_drifts:
            self._alert_drift_detected(critical_drifts, drift_results)

        # TODO: Log metrics to Prometheus
        self._log_drift_metrics(drift_results)

    def _alert_drift_detected(
        self,
        drifted_features: List[str],
        drift_results: Dict[str, DriftResult]
    ):
        """
        TODO: Send alerts on drift detection

        Send to:
        1. Prometheus (increment drift_detected counter)
        2. Slack (#ml-alerts channel)
        3. PagerDuty (if CRITICAL severity)

        Include:
        - Which features drifted
        - Severity
        - Recommended action
        """
        pass

    def _log_drift_metrics(self, drift_results: Dict[str, DriftResult]):
        """
        TODO: Log drift metrics to Prometheus

        Metrics:
        - drift_detected{feature="amount", severity="high"} = 1
        - psi_score{feature="amount"} = 0.25
        - ks_statistic{feature="amount"} = 0.15
        """
        pass

    def start(self):
        """
        TODO: Start continuous monitoring

        schedule.every(self.check_interval_hours).hours.do(self.run_drift_check)

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled jobs
        """
        pass
```

### Part 2: Model Performance Tracking (7-9 hours)

Track model accuracy, precision, recall in real-time.

#### 2.1 Performance Metrics Calculator

Create `src/performance/metrics_tracker.py`:

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, List
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelPerformance:
    timestamp: datetime
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    sample_size: int

class PerformanceTracker:
    """
    Track model performance metrics in real-time.

    Challenge: In production, we don't always have ground truth labels immediately.

    Solutions:
    1. Delayed labels: Wait for labels (e.g., fraud confirmed after 24 hours)
    2. Proxy metrics: Use click-through rate as proxy for relevance
    3. Human labeling: Sample and label subset for evaluation
    """

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> ModelPerformance:
        """
        TODO: Calculate all performance metrics

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        # ROC AUC (requires probabilities)
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        else:
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        Return ModelPerformance object
        """
        pass

    def track_performance_over_time(
        self,
        performance_history: List[ModelPerformance],
        window_days: int = 30
    ) -> Dict:
        """
        TODO: Analyze performance trends

        Calculate:
        1. Current metrics vs 30-day average
        2. Performance degradation rate (slope of accuracy over time)
        3. Statistical significance of degradation (t-test)

        Return:
        {
            'current_accuracy': 0.92,
            'avg_accuracy_30d': 0.94,
            'degradation_rate': -0.001,  # Accuracy dropping 0.1% per day
            'degradation_significant': True,
            'days_until_slo_violation': 15  # At current rate, SLO violated in 15 days
        }
        """
        pass

    def detect_performance_degradation(
        self,
        current_performance: ModelPerformance,
        baseline_performance: ModelPerformance,
        threshold: float = 0.02  # 2% degradation
    ) -> bool:
        """
        TODO: Detect if performance has degraded significantly

        Check if any metric has decreased by more than threshold:
        - Accuracy drop > 2%
        - Precision drop > 2%
        - Recall drop > 2%

        Return True if degradation detected
        """
        pass

class PerformanceMonitor:
    """Continuous performance monitoring with alerting."""

    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.performance_history: List[ModelPerformance] = []

    def fetch_ground_truth_labels(
        self,
        prediction_ids: List[str]
    ) -> Dict[str, int]:
        """
        TODO: Fetch ground truth labels for predictions

        In production, labels may arrive with delay:
        - Fraud detection: Label arrives when fraud confirmed (24-72 hours)
        - Recommendations: Click/no-click label arrives immediately
        - Credit risk: Default/no-default label arrives after 30-90 days

        Query database:
        SELECT prediction_id, ground_truth_label, label_timestamp
        FROM prediction_labels
        WHERE prediction_id IN (...)
        AND label_timestamp IS NOT NULL

        Return dict: prediction_id -> label
        """
        pass

    def evaluate_recent_predictions(self, hours: int = 24):
        """
        TODO: Evaluate predictions from last N hours

        1. Fetch predictions with ground truth labels
        2. Calculate performance metrics
        3. Compare to baseline
        4. Alert if degradation detected
        5. Log metrics to Prometheus
        """
        pass
```

### Part 3: Prediction Distribution Monitoring (6-8 hours)

Monitor prediction outputs for anomalies.

#### 3.1 Prediction Analyzer

Create `src/prediction/prediction_analyzer.py`:

```python
class PredictionDistributionMonitor:
    """
    Monitor distribution of model predictions for anomalies.

    Unusual prediction patterns can indicate:
    - Model bug (always predicting same class)
    - Data drift (input distribution changed)
    - Feature pipeline bug (features corrupted)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.baseline_distribution = None

    def analyze_prediction_distribution(
        self,
        predictions: np.ndarray,
        prediction_probabilities: Optional[np.ndarray] = None
    ) -> Dict:
        """
        TODO: Analyze prediction distribution for anomalies

        For classification:
        1. Class balance:
           - Is model predicting mostly one class?
           - Expected: fraud_rate ~5%, if predicting 50%, something wrong

        2. Confidence distribution:
           - Are predictions confident (prob near 0 or 1)?
           - Or uncertain (prob near 0.5)?
           - Low confidence may indicate out-of-distribution inputs

        3. Prediction rate over time:
           - Sudden spike in positive predictions?
           - Could indicate data shift or model bug

        For regression:
        1. Prediction range:
           - Are predictions within expected range?
           - Outliers? (predictions > 3 std dev from mean)

        2. Distribution shape:
           - Has distribution changed? (should be similar to training)

        Return analysis results
        """
        pass

    def detect_prediction_anomalies(
        self,
        current_predictions: np.ndarray,
        baseline_predictions: np.ndarray
    ) -> List[str]:
        """
        TODO: Detect anomalies in prediction distribution

        Anomalies to check:
        1. Class imbalance shift:
           - Baseline: 5% positive, Current: 20% positive → ALERT

        2. Confidence collapse:
           - All predictions near 0.5 (model uncertain) → ALERT

        3. Prediction spike:
           - Sudden 10× increase in positive predictions → ALERT

        4. Out-of-range predictions:
           - Regression model predicting negative when only positive valid → ALERT

        Return list of detected anomalies
        """
        anomalies = []

        # TODO: Implement checks

        return anomalies
```

### Part 4: A/B Testing Infrastructure (5-7 hours)

Build infrastructure for safe model rollouts.

#### 4.1 A/B Test Controller

Create `src/ab_testing/ab_controller.py`:

```python
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import random

class TrafficSplitStrategy(Enum):
    RANDOM = "random"  # Randomly assign to A or B
    HASH_BASED = "hash_based"  # Consistent assignment based on user_id
    GRADUAL_ROLLOUT = "gradual_rollout"  # Gradually increase B traffic

@dataclass
class ABTestConfig:
    name: str
    model_a_version: str  # Control (baseline)
    model_b_version: str  # Treatment (new version)
    traffic_split_percent_b: float  # % of traffic to model B (0-100)
    strategy: TrafficSplitStrategy
    success_metric: str  # 'accuracy', 'precision', 'roc_auc'
    minimum_sample_size: int = 1000
    statistical_significance_threshold: float = 0.05

class ABTestController:
    """
    A/B testing for model rollouts.

    Safely test new model versions by:
    1. Routing % of traffic to new model
    2. Comparing performance metrics
    3. Auto-promote if better, rollback if worse
    """

    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}

    def create_ab_test(self, config: ABTestConfig):
        """
        TODO: Create new A/B test

        1. Validate config
        2. Deploy both model versions
        3. Start routing traffic
        4. Initialize metrics collection

        self.active_tests[config.name] = config
        """
        pass

    def route_prediction_request(
        self,
        test_name: str,
        user_id: str
    ) -> str:
        """
        TODO: Route request to model A or B based on strategy

        If strategy == RANDOM:
            if random.random() < (traffic_split / 100):
                return model_b_version
            else:
                return model_a_version

        If strategy == HASH_BASED:
            # Consistent assignment (same user always gets same model)
            hash_value = hash(user_id) % 100
            if hash_value < traffic_split:
                return model_b_version
            else:
                return model_a_version

        If strategy == GRADUAL_ROLLOUT:
            # Increase traffic_split over time (e.g., 1%, 5%, 25%, 50%, 100%)
            # See gradual_rollout() method

        Return model_version to use
        """
        pass

    def collect_ab_test_metrics(
        self,
        test_name: str,
        model_version: str,
        prediction: int,
        ground_truth: Optional[int] = None
    ):
        """
        TODO: Collect metrics for A/B test comparison

        Store in database:
        - test_name
        - model_version (A or B)
        - prediction
        - ground_truth (if available)
        - timestamp

        This data will be used to calculate statistical significance
        """
        pass

    def analyze_ab_test_results(self, test_name: str) -> Dict:
        """
        TODO: Analyze A/B test results

        1. Fetch metrics for model A and model B
        2. Calculate success metric (accuracy, precision, etc.)
        3. Perform statistical significance test:
           - Z-test for proportions (accuracy, precision)
           - T-test for continuous metrics

        4. Determine winner:
           - If B significantly better (p < 0.05): B wins
           - If A significantly better: A wins (rollback B)
           - If no significant difference: Inconclusive

        Return:
        {
            'model_a_metric': 0.92,
            'model_b_metric': 0.94,
            'improvement': 0.02,  # +2%
            'p_value': 0.001,
            'statistically_significant': True,
            'winner': 'model_b',
            'recommendation': 'Promote model B to 100% traffic'
        }
        """
        pass

    def auto_promote_or_rollback(self, test_name: str):
        """
        TODO: Automatically promote winner or rollback loser

        1. Analyze test results
        2. If model B wins:
           - Gradually increase traffic to 100%
           - Mark model B as production
           - End test

        3. If model A wins (B is worse):
           - Rollback: Route all traffic to model A
           - Alert team about failed rollout
           - End test

        4. If inconclusive:
           - Continue test (need more data)
        """
        pass

    def gradual_rollout(
        self,
        test_name: str,
        rollout_schedule: List[Tuple[int, float]]
    ):
        """
        TODO: Gradually increase traffic to new model

        rollout_schedule = [
            (60, 1),      # 1% traffic for 60 minutes
            (120, 5),     # 5% traffic for 120 minutes
            (240, 25),    # 25% traffic for 240 minutes
            (480, 50),    # 50% traffic for 480 minutes
            (0, 100),     # 100% traffic (full rollout)
        ]

        At each stage:
        - Update traffic split
        - Monitor for performance degradation
        - If degradation: rollback to previous stage
        - If OK: proceed to next stage
        """
        pass
```

### Part 5: Automated Rollback (4-6 hours)

Implement automatic rollback on performance degradation.

#### 5.1 Rollback Engine

Create `src/rollback/auto_rollback.py`:

```python
class AutoRollbackEngine:
    """
    Automatically rollback model deployments on performance degradation.

    Triggers:
    1. Accuracy drops >2% from baseline
    2. Error rate >1%
    3. Latency P95 >500ms
    4. Critical drift detected
    """

    def __init__(self, performance_tracker: PerformanceTracker):
        self.performance_tracker = performance_tracker
        self.rollback_history: List[Dict] = []

    def monitor_and_rollback(self):
        """
        TODO: Continuously monitor performance and rollback if needed

        Run every 5 minutes:
        1. Fetch recent performance metrics
        2. Check rollback triggers
        3. If trigger activated:
           - Execute rollback
           - Alert team
           - Log incident

        schedule.every(5).minutes.do(self._check_performance)
        """
        pass

    def _check_performance(self):
        """
        TODO: Check if rollback needed

        Fetch current performance:
        - Accuracy, precision, recall
        - Error rate
        - Latency P95

        Compare to baseline (last stable version):
        - If accuracy drop >2%: ROLLBACK
        - If error rate >1%: ROLLBACK
        - If latency >500ms: ROLLBACK

        If multiple models degraded:
        - Rollback most critical first
        """
        pass

    def execute_rollback(
        self,
        model_name: str,
        current_version: str,
        rollback_to_version: str,
        reason: str
    ):
        """
        TODO: Execute model rollback

        Steps:
        1. Validate rollback_to_version exists and is healthy
        2. Update model routing config:
           - Route 100% traffic to rollback_to_version
           - Keep current_version deployed for debugging

        3. Verify rollback success:
           - Wait 2 minutes
           - Check metrics improved
           - If not improved: escalate to on-call

        4. Log rollback:
           - Store in rollback_history
           - Send alert to Slack/PagerDuty
           - Create incident ticket

        5. Post-rollback:
           - Analyze root cause
           - Fix issue in current_version
           - Re-deploy with A/B test
        """
        print(f"🚨 ROLLBACK: {model_name} {current_version} → {rollback_to_version}")
        print(f"Reason: {reason}")

        # TODO: Implement rollback logic

        # Log rollback
        self.rollback_history.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'from_version': current_version,
            'to_version': rollback_to_version,
            'reason': reason
        })
```

## Acceptance Criteria

### Functional Requirements

- [ ] Data drift detection implemented (KS test, PSI, JSD)
- [ ] Drift monitoring runs every 6 hours automatically
- [ ] Model performance tracked (accuracy, precision, recall, ROC-AUC)
- [ ] Prediction distribution monitoring detects anomalies
- [ ] A/B testing infrastructure supports 10+ concurrent tests
- [ ] Automated rollback triggers on performance degradation
- [ ] ML monitoring dashboard shows all metrics

### Performance Requirements

- [ ] Drift detection completes in <2 minutes for 10K samples
- [ ] Performance metrics updated in real-time (<1 minute lag)
- [ ] A/B test analysis completes in <5 seconds
- [ ] Rollback executes in <5 minutes
- [ ] Dashboard load time <3 seconds

### Operational Requirements

- [ ] Data drift detected within 24 hours (100% detection rate)
- [ ] Zero silent model failures (all degradation detected)
- [ ] Automated rollback success rate >95%
- [ ] False positive alert rate <5%

### Code Quality

- [ ] All drift detection algorithms have unit tests
- [ ] Statistical tests validated against known datasets
- [ ] Comprehensive logging and error handling
- [ ] Documentation with usage examples

## Testing Strategy

### Unit Tests

```python
# tests/test_drift_detection.py
def test_ks_test_detects_drift():
    """Verify KS test detects distribution shift."""
    # TODO: Create reference and shifted distributions
    # Verify drift detected
```

### Integration Tests

```python
# tests/test_ab_testing.py
def test_ab_test_workflow():
    """Test complete A/B test workflow."""
    # 1. Create test
    # 2. Route traffic
    # 3. Collect metrics
    # 4. Analyze results
    # 5. Verify promotion/rollback
```

## Deliverables

1. **Source Code** (`src/`):
   - `drift_detection/` - Drift detector, continuous monitor
   - `performance/` - Performance tracker, metrics calculator
   - `prediction/` - Prediction distribution analyzer
   - `ab_testing/` - A/B test controller
   - `rollback/` - Auto-rollback engine

2. **Dashboards** (`dashboards/`):
   - ML model performance dashboard (Grafana JSON)
   - Data drift dashboard
   - A/B test comparison dashboard

3. **Documentation** (`docs/`):
   - `ML_MONITORING_GUIDE.md` - Complete guide
   - `DRIFT_DETECTION.md` - Drift detection algorithms
   - `AB_TESTING.md` - A/B testing best practices
   - `ROLLBACK_PROCEDURES.md` - Rollback procedures

4. **Deployment** (`kubernetes/`):
   - Drift monitor deployment
   - Performance tracker deployment
   - A/B test controller deployment

## Bonus Challenges

1. **Feature Importance Tracking** (+6 hours):
   - Monitor feature importance over time
   - Alert when important features become irrelevant
   - Detect feature leakage

2. **Bias Detection** (+8 hours):
   - Monitor prediction bias across demographics
   - Implement fairness metrics (demographic parity, equalized odds)
   - Alert on bias increase

3. **Explainability Monitoring** (+6 hours):
   - Track SHAP values for predictions
   - Detect when model explanations change
   - Monitor model trust scores

## Resources

### Documentation

- [Evidently AI - ML Monitoring](https://docs.evidentlyai.com/)
- [Alibi Detect - Drift Detection](https://docs.seldon.io/projects/alibi-detect/)
- [Sklearn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Research Papers

- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [A Survey on Concept Drift Adaptation](https://dl.acm.org/doi/10.1145/2523813)

## Submission

Submit your implementation via Git:

```bash
git add .
git commit -m "Complete Exercise 02: ML Model Monitoring"
git push origin exercise-02-ml-model-monitoring
```

Ensure your submission includes:
- Complete drift detection implementation
- Performance tracking system
- A/B testing infrastructure
- Auto-rollback engine
- Dashboards and documentation

---

**Estimated Time Breakdown**:
- Part 1 (Drift Detection): 8-10 hours
- Part 2 (Performance Tracking): 7-9 hours
- Part 3 (Prediction Monitoring): 6-8 hours
- Part 4 (A/B Testing): 5-7 hours
- Part 5 (Auto Rollback): 4-6 hours
- **Total**: 30-38 hours
