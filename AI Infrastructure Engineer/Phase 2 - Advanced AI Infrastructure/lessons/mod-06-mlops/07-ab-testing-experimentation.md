# Lesson 07: A/B Testing & Experimentation

## Overview
A/B testing and experimentation are critical for validating ML model improvements in production. This lesson covers designing, running, and analyzing experiments to measure model performance and business impact.

**Duration:** 2-3 hours
**Prerequisites:** Understanding of statistical testing, ML metrics, production deployments
**Learning Objectives:**
- Design statistically sound ML experiments
- Implement A/B testing infrastructure
- Analyze experiment results and make decisions
- Handle common pitfalls and challenges
- Measure business impact of ML models

---

## 1. Introduction to ML Experimentation

### 1.1 Why A/B Testing for ML?

**Traditional ML Evaluation:**
```python
# Offline evaluation
accuracy = 0.92  # On test set
precision = 0.89
recall = 0.91

# ❌ Problem: Doesn't measure real-world impact
# - Test set may not represent production distribution
# - Metrics may not correlate with business outcomes
# - No measurement of user behavior changes
```

**A/B Testing:**
```python
# Online evaluation with real users
control_group = {
    'model': 'v1',
    'accuracy': 0.92,
    'click_through_rate': 0.15,  # 15% of recommendations clicked
    'revenue_per_user': 45.20
}

treatment_group = {
    'model': 'v2',
    'accuracy': 0.94,  # Better accuracy
    'click_through_rate': 0.18,  # 20% improvement!
    'revenue_per_user': 52.30  # 15.7% revenue increase
}

# ✅ Measures actual business impact
```

### 1.2 Experiment Design Framework

```
┌─────────────────────────────────────────────────────────┐
│              ML Experiment Lifecycle                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. Hypothesis Formation                                 │
│     └─ "New model will increase CTR by 10%"             │
│                                                           │
│  2. Experiment Design                                    │
│     ├─ Sample size calculation                          │
│     ├─ Traffic allocation (90% control, 10% treatment)  │
│     └─ Success metrics definition                       │
│                                                           │
│  3. Implementation                                       │
│     ├─ Feature flags / routing logic                    │
│     ├─ Instrumentation / logging                        │
│     └─ Guardrail metrics                                │
│                                                           │
│  4. Execution & Monitoring                              │
│     ├─ Data quality checks                              │
│     ├─ Real-time dashboards                             │
│     └─ Anomaly detection                                │
│                                                           │
│  5. Analysis & Decision                                  │
│     ├─ Statistical significance testing                  │
│     ├─ Business impact assessment                       │
│     └─ Launch decision                                   │
│                                                           │
│  6. Post-Launch Monitoring                              │
│     └─ Continued metric tracking                        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Designing ML Experiments

### 2.1 Hypothesis and Metrics

```python
# experiments/experiment_config.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class MetricType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"

@dataclass
class Metric:
    name: str
    metric_type: MetricType
    direction: str  # "increase" or "decrease"
    threshold: float  # Minimum detectable effect
    description: str

@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment"""

    # Experiment metadata
    experiment_id: str
    name: str
    hypothesis: str
    owner: str

    # Variants
    control_variant: str = "control"
    treatment_variants: List[str] = None

    # Traffic allocation
    traffic_allocation: Dict[str, float] = None

    # Metrics
    primary_metrics: List[Metric] = None
    secondary_metrics: List[Metric] = None
    guardrail_metrics: List[Metric] = None

    # Statistical parameters
    significance_level: float = 0.05  # Alpha
    statistical_power: float = 0.80   # 1 - Beta
    minimum_sample_size: int = 10000

    # Duration
    planned_duration_days: int = 14

# Example experiment configuration
recommendation_model_experiment = ExperimentConfig(
    experiment_id="rec_model_v2_2024_01",
    name="Recommendation Model V2 Launch",
    hypothesis="New collaborative filtering model will increase CTR by 10%",
    owner="ml-team@example.com",

    control_variant="model_v1",
    treatment_variants=["model_v2"],

    traffic_allocation={
        "model_v1": 0.90,  # 90% control
        "model_v2": 0.10   # 10% treatment
    },

    primary_metrics=[
        Metric(
            name="click_through_rate",
            metric_type=MetricType.PRIMARY,
            direction="increase",
            threshold=0.10,  # 10% relative increase
            description="Percentage of recommendations clicked"
        )
    ],

    secondary_metrics=[
        Metric(
            name="revenue_per_user",
            metric_type=MetricType.SECONDARY,
            direction="increase",
            threshold=0.05,  # 5% relative increase
            description="Average revenue per user during experiment"
        ),
        Metric(
            name="session_duration",
            metric_type=MetricType.SECONDARY,
            direction="increase",
            threshold=0.03,
            description="Average session duration in minutes"
        )
    ],

    guardrail_metrics=[
        Metric(
            name="error_rate",
            metric_type=MetricType.GUARDRAIL,
            direction="decrease",
            threshold=0.01,  # Must stay below 1%
            description="API error rate"
        ),
        Metric(
            name="p99_latency",
            metric_type=MetricType.GUARDRAIL,
            direction="decrease",
            threshold=100,  # Must stay below 100ms
            description="99th percentile latency"
        )
    ],

    significance_level=0.05,
    statistical_power=0.80,
    minimum_sample_size=10000,
    planned_duration_days=14
)
```

### 2.2 Sample Size Calculation

```python
# experiments/sample_size.py
import numpy as np
from scipy import stats
from typing import Tuple

def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    significance_level: float = 0.05,
    statistical_power: float = 0.80
) -> Tuple[int, dict]:
    """
    Calculate required sample size for A/B test

    Args:
        baseline_rate: Current conversion/click rate
        minimum_detectable_effect: Minimum effect to detect (e.g., 0.10 for 10% improvement)
        significance_level: Alpha (Type I error rate)
        statistical_power: 1 - Beta (Type II error rate)

    Returns:
        Required sample size per variant
    """

    # Expected treatment rate
    treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        baseline_rate * (1 - baseline_rate) +
        treatment_rate * (1 - treatment_rate)
    )

    # Z-scores
    z_alpha = stats.norm.ppf(1 - significance_level / 2)  # Two-tailed
    z_beta = stats.norm.ppf(statistical_power)

    # Sample size per variant
    n = (
        (z_alpha + z_beta) ** 2 *
        pooled_std ** 2 /
        (treatment_rate - baseline_rate) ** 2
    )

    sample_size = int(np.ceil(n))

    # Calculate expected duration
    daily_users = 100000  # Example
    days_required = np.ceil(sample_size * 2 / daily_users)

    return sample_size, {
        'sample_size_per_variant': sample_size,
        'total_sample_size': sample_size * 2,
        'estimated_days': int(days_required),
        'baseline_rate': baseline_rate,
        'expected_treatment_rate': treatment_rate,
        'minimum_detectable_effect': minimum_detectable_effect,
        'significance_level': significance_level,
        'statistical_power': statistical_power
    }

# Example usage
sample_size, details = calculate_sample_size(
    baseline_rate=0.15,  # Current CTR of 15%
    minimum_detectable_effect=0.10,  # Detect 10% improvement
    significance_level=0.05,
    statistical_power=0.80
)

print(f"Required sample size: {sample_size:,} per variant")
print(f"Total users needed: {details['total_sample_size']:,}")
print(f"Estimated duration: {details['estimated_days']} days")
```

---

## 3. Implementing A/B Testing Infrastructure

### 3.1 User Assignment and Routing

```python
# experiments/assignment.py
import hashlib
from typing import Dict, Optional
from dataclasses import dataclass
import redis

@dataclass
class Assignment:
    user_id: str
    experiment_id: str
    variant: str
    timestamp: int

class ExperimentAssigner:
    """Assign users to experiment variants"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def assign_variant(
        self,
        user_id: str,
        experiment_id: str,
        traffic_allocation: Dict[str, float]
    ) -> str:
        """
        Assign user to a variant using consistent hashing

        Ensures:
        - Same user always gets same variant
        - Traffic split matches allocation percentages
        """

        # Check if user already assigned
        cache_key = f"assignment:{experiment_id}:{user_id}"
        cached = self.redis.get(cache_key)

        if cached:
            return cached.decode('utf-8')

        # Hash user_id + experiment_id for consistent assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Normalize to [0, 1]
        assignment_value = (hash_value % 10000) / 10000.0

        # Determine variant based on traffic allocation
        cumulative = 0.0
        for variant, allocation in sorted(traffic_allocation.items()):
            cumulative += allocation
            if assignment_value < cumulative:
                # Cache assignment
                self.redis.setex(
                    cache_key,
                    86400 * 30,  # 30 days TTL
                    variant
                )
                return variant

        # Fallback to control
        return "control"

    def get_assignment(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[str]:
        """Get existing assignment if any"""
        cache_key = f"assignment:{experiment_id}:{user_id}"
        cached = self.redis.get(cache_key)
        return cached.decode('utf-8') if cached else None

# Usage in model serving
class ExperimentAwareModelServer:
    """Model server with A/B testing support"""

    def __init__(self):
        self.assigner = ExperimentAssigner(redis.Redis(host='localhost'))
        self.models = {
            'model_v1': load_model('models:/model/v1'),
            'model_v2': load_model('models:/model/v2')
        }

    async def predict(self, user_id: str, features: Dict):
        """Make prediction with A/B testing"""

        # Get or assign variant
        variant = self.assigner.assign_variant(
            user_id=user_id,
            experiment_id="rec_model_v2_2024_01",
            traffic_allocation={
                'model_v1': 0.90,
                'model_v2': 0.10
            }
        )

        # Get corresponding model
        model = self.models[variant]

        # Make prediction
        prediction = model.predict(features)

        # Log for analysis
        log_prediction(
            user_id=user_id,
            experiment_id="rec_model_v2_2024_01",
            variant=variant,
            prediction=prediction,
            features=features
        )

        return {
            'prediction': prediction,
            'variant': variant
        }
```

### 3.2 Event Logging and Tracking

```python
# experiments/logging.py
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import boto3
from typing import Dict, Any

@dataclass
class ExperimentEvent:
    """Event logged during experiment"""
    event_id: str
    timestamp: datetime
    user_id: str
    experiment_id: str
    variant: str
    event_type: str  # "impression", "click", "conversion", etc.
    properties: Dict[str, Any]

class ExperimentLogger:
    """Log experiment events to analytics system"""

    def __init__(self, kinesis_stream: str):
        self.kinesis = boto3.client('kinesis')
        self.stream_name = kinesis_stream

    def log_event(self, event: ExperimentEvent):
        """Log event to Kinesis"""
        record = {
            'Data': json.dumps(asdict(event), default=str),
            'PartitionKey': event.user_id
        }

        self.kinesis.put_record(
            StreamName=self.stream_name,
            **record
        )

    def log_impression(
        self,
        user_id: str,
        experiment_id: str,
        variant: str,
        items: list
    ):
        """Log when user sees recommendations"""
        event = ExperimentEvent(
            event_id=f"{user_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            event_type="impression",
            properties={
                'items': items,
                'num_items': len(items)
            }
        )
        self.log_event(event)

    def log_click(
        self,
        user_id: str,
        experiment_id: str,
        variant: str,
        item_id: str,
        position: int
    ):
        """Log when user clicks recommendation"""
        event = ExperimentEvent(
            event_id=f"{user_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            event_type="click",
            properties={
                'item_id': item_id,
                'position': position
            }
        )
        self.log_event(event)

    def log_conversion(
        self,
        user_id: str,
        experiment_id: str,
        variant: str,
        revenue: float
    ):
        """Log when user makes purchase"""
        event = ExperimentEvent(
            event_id=f"{user_id}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant,
            event_type="conversion",
            properties={
                'revenue': revenue
            }
        )
        self.log_event(event)

# Integration with model serving
@app.post("/recommend")
async def recommend(user_id: str):
    # Get variant assignment
    variant = assigner.assign_variant(user_id, "rec_model_v2_2024_01", {...})

    # Get recommendations
    items = model.predict(user_id)

    # Log impression
    logger.log_impression(
        user_id=user_id,
        experiment_id="rec_model_v2_2024_01",
        variant=variant,
        items=items
    )

    return {"items": items}

@app.post("/track/click")
async def track_click(user_id: str, item_id: str, position: int):
    # Get user's variant
    variant = assigner.get_assignment(user_id, "rec_model_v2_2024_01")

    # Log click
    logger.log_click(
        user_id=user_id,
        experiment_id="rec_model_v2_2024_01",
        variant=variant,
        item_id=item_id,
        position=position
    )

    return {"status": "logged"}
```

---

## 4. Analyzing Experiment Results

### 4.1 Statistical Significance Testing

```python
# experiments/analysis.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple

class ExperimentAnalyzer:
    """Analyze A/B test results"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def proportion_test(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int
    ) -> Dict:
        """
        Two-proportion z-test for binary metrics (CTR, conversion rate, etc.)
        """

        # Calculate proportions
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total

        # Pooled proportion
        p_pool = (control_conversions + treatment_conversions) / (control_total + treatment_total)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))

        # Z-score
        z_score = (p2 - p1) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Confidence interval
        se_diff = np.sqrt(p1 * (1 - p1) / control_total + p2 * (1 - p2) / treatment_total)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff

        # Relative lift
        relative_lift = (p2 - p1) / p1 if p1 > 0 else 0

        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'absolute_difference': p2 - p1,
            'relative_lift': relative_lift,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_sizes': {
                'control': control_total,
                'treatment': treatment_total
            }
        }

    def continuous_metric_test(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray
    ) -> Dict:
        """
        T-test for continuous metrics (revenue, session duration, etc.)
        """

        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(
            treatment_values,
            control_values,
            equal_var=False
        )

        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        relative_lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0

        # Confidence interval
        se = np.sqrt(
            np.var(control_values) / len(control_values) +
            np.var(treatment_values) / len(treatment_values)
        )
        ci_lower = (treatment_mean - control_mean) - 1.96 * se
        ci_upper = (treatment_mean - control_mean) + 1.96 * se

        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'absolute_difference': treatment_mean - control_mean,
            'relative_lift': relative_lift,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_sizes': {
                'control': len(control_values),
                'treatment': len(treatment_values)
            }
        }

    def analyze_experiment(
        self,
        experiment_data: pd.DataFrame,
        metrics_config: List[Metric]
    ) -> Dict:
        """
        Complete experiment analysis
        """
        results = {}

        for metric in metrics_config:
            if metric.name in ['click_through_rate', 'conversion_rate']:
                # Binary metric - use proportion test
                control = experiment_data[experiment_data['variant'] == 'control']
                treatment = experiment_data[experiment_data['variant'] == 'treatment']

                result = self.proportion_test(
                    control_conversions=control[f'{metric.name}_converted'].sum(),
                    control_total=len(control),
                    treatment_conversions=treatment[f'{metric.name}_converted'].sum(),
                    treatment_total=len(treatment)
                )

            else:
                # Continuous metric - use t-test
                control_values = experiment_data[
                    experiment_data['variant'] == 'control'
                ][metric.name].values

                treatment_values = experiment_data[
                    experiment_data['variant'] == 'treatment'
                ][metric.name].values

                result = self.continuous_metric_test(control_values, treatment_values)

            results[metric.name] = result

        return results

# Usage
analyzer = ExperimentAnalyzer(significance_level=0.05)

# Load experiment data
experiment_data = pd.read_parquet('s3://experiments/rec_model_v2_2024_01/data.parquet')

# Analyze
results = analyzer.analyze_experiment(
    experiment_data=experiment_data,
    metrics_config=recommendation_model_experiment.primary_metrics
)

# Print results
for metric_name, result in results.items():
    print(f"\n{'='*60}")
    print(f"Metric: {metric_name}")
    print(f"Control: {result['control_rate']:.4f}")
    print(f"Treatment: {result['treatment_rate']:.4f}")
    print(f"Relative Lift: {result['relative_lift']*100:+.2f}%")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {'✅ YES' if result['is_significant'] else '❌ NO'}")
    print(f"95% CI: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
```

### 4.2 Sequential Testing and Early Stopping

```python
# experiments/sequential_testing.py
import numpy as np
from scipy import stats

class SequentialTester:
    """
    Sequential testing for early stopping decisions
    Uses alpha spending approach
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        num_looks: int = 5
    ):
        self.significance_level = significance_level
        self.num_looks = num_looks
        self.alpha_spent = self._calculate_alpha_spending()

    def _calculate_alpha_spending(self) -> list:
        """
        Calculate alpha spending at each look (O'Brien-Fleming)
        """
        alphas = []
        for k in range(1, self.num_looks + 1):
            z = stats.norm.ppf(1 - self.significance_level / 2)
            z_k = z * np.sqrt(self.num_looks / k)
            alpha_k = 2 * (1 - stats.norm.cdf(z_k))
            alphas.append(alpha_k)
        return alphas

    def should_stop(
        self,
        look_number: int,
        p_value: float
    ) -> Tuple[bool, str]:
        """
        Decide if experiment should stop at this look

        Returns:
            (should_stop, reason)
        """

        if look_number > len(self.alpha_spent):
            return False, "exceeded_max_looks"

        threshold = self.alpha_spent[look_number - 1]

        if p_value < threshold:
            return True, "significant_result"

        if look_number == self.num_looks:
            return True, "max_duration_reached"

        return False, "continue"

# Usage
tester = SequentialTester(significance_level=0.05, num_looks=5)

# Check at each analysis point
for look in range(1, 6):
    # Run analysis
    p_value = analyzer.proportion_test(...)['p_value']

    should_stop, reason = tester.should_stop(look, p_value)

    print(f"Look {look}: p-value={p_value:.4f}, threshold={tester.alpha_spent[look-1]:.4f}")

    if should_stop:
        print(f"✅ Stop experiment: {reason}")
        break
```

---

## 5. Common Pitfalls and Solutions

### 5.1 Novelty Effect

```python
def detect_novelty_effect(experiment_data: pd.DataFrame):
    """
    Check if treatment effect changes over time (novelty effect)
    """
    # Split experiment into time cohorts
    experiment_data['cohort'] = pd.qcut(
        experiment_data['days_since_start'],
        q=4,
        labels=['week1', 'week2', 'week3', 'week4']
    )

    # Analyze each cohort
    for cohort in ['week1', 'week2', 'week3', 'week4']:
        cohort_data = experiment_data[experiment_data['cohort'] == cohort]

        result = analyzer.proportion_test(
            control_conversions=cohort_data[
                cohort_data['variant'] == 'control'
            ]['converted'].sum(),
            control_total=len(cohort_data[cohort_data['variant'] == 'control']),
            treatment_conversions=cohort_data[
                cohort_data['variant'] == 'treatment'
            ]['converted'].sum(),
            treatment_total=len(cohort_data[cohort_data['variant'] == 'treatment'])
        )

        print(f"{cohort}: Lift = {result['relative_lift']*100:+.2f}%")

    # If lift decreases significantly over time → novelty effect
```

### 5.2 Sample Ratio Mismatch (SRM)

```python
def check_sample_ratio_mismatch(
    control_count: int,
    treatment_count: int,
    expected_ratio: float = 0.5
) -> Dict:
    """
    Check if observed traffic split matches expected

    Sample Ratio Mismatch can indicate:
    - Bugs in assignment logic
    - Bot traffic
    - Performance issues causing user drop-off
    """

    total = control_count + treatment_count
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)

    # Chi-square test
    chi_square = (
        (control_count - expected_control) ** 2 / expected_control +
        (treatment_count - expected_treatment) ** 2 / expected_treatment
    )

    p_value = 1 - stats.chi2.cdf(chi_square, df=1)

    return {
        'control_count': control_count,
        'treatment_count': treatment_count,
        'expected_ratio': expected_ratio,
        'observed_ratio': control_count / total,
        'chi_square': chi_square,
        'p_value': p_value,
        'has_srm': p_value < 0.001,  # Very low threshold
        'warning': '⚠️ Sample Ratio Mismatch detected!' if p_value < 0.001 else '✅ No SRM'
    }

# Check for SRM
srm_result = check_sample_ratio_mismatch(
    control_count=89500,
    treatment_count=10800,  # Expected 10000
    expected_ratio=0.9
)

if srm_result['has_srm']:
    print("❌ SRM detected - investigate before trusting results!")
```

---

## 6. Experiment Dashboard

```python
# experiments/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def create_experiment_dashboard(experiment_id: str):
    """
    Real-time experiment monitoring dashboard
    """
    st.title(f"Experiment: {experiment_id}")

    # Load latest data
    data = load_experiment_data(experiment_id)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Users",
            f"{len(data):,}",
            delta=f"+{len(data) - len(data.shift(1)):,}"
        )

    with col2:
        control_ctr = data[data['variant'] == 'control']['ctr'].mean()
        treatment_ctr = data[data['variant'] == 'treatment']['ctr'].mean()
        lift = (treatment_ctr - control_ctr) / control_ctr * 100

        st.metric(
            "CTR Lift",
            f"{lift:+.2f}%",
            delta=f"{treatment_ctr:.4f} vs {control_ctr:.4f}"
        )

    with col3:
        result = analyzer.proportion_test(...)
        st.metric(
            "P-value",
            f"{result['p_value']:.4f}",
            delta="Significant ✅" if result['is_significant'] else "Not sig ⏳"
        )

    with col4:
        days_running = (datetime.now() - experiment_start_date).days
        st.metric(
            "Days Running",
            days_running,
            delta=f"{planned_duration - days_running} days remaining"
        )

    # Time series plot
    st.subheader("Metric Over Time")

    fig = go.Figure()

    for variant in ['control', 'treatment']:
        variant_data = data[data['variant'] == variant]
        daily = variant_data.groupby('date')['ctr'].mean()

        fig.add_trace(go.Scatter(
            x=daily.index,
            y=daily.values,
            name=variant,
            mode='lines+markers'
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Click-Through Rate",
        hovermode='x unified'
    )

    st.plotly_chart(fig)

    # Guardrail metrics
    st.subheader("Guardrail Metrics")

    guardrail_status = check_guardrails(data)

    for metric, status in guardrail_status.items():
        if status['healthy']:
            st.success(f"✅ {metric}: {status['value']:.4f} (threshold: {status['threshold']})")
        else:
            st.error(f"❌ {metric}: {status['value']:.4f} exceeds threshold {status['threshold']}")

    # Statistical analysis
    st.subheader("Statistical Analysis")

    results = analyzer.analyze_experiment(data, metrics_config)

    for metric_name, result in results.items():
        with st.expander(f"{metric_name}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Control:** {result['control_rate']:.4f}")
                st.write(f"**Treatment:** {result['treatment_rate']:.4f}")
                st.write(f"**Lift:** {result['relative_lift']*100:+.2f}%")

            with col2:
                st.write(f"**P-value:** {result['p_value']:.4f}")
                st.write(f"**Significant:** {'✅ Yes' if result['is_significant'] else '❌ No'}")
                st.write(f"**95% CI:** [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")

# Run dashboard
# streamlit run experiments/dashboard.py
```

---

## Summary

In this lesson, you learned:

✅ **Experiment Design:**
- Hypothesis formation and metrics selection
- Sample size calculation
- Traffic allocation strategies

✅ **Implementation:**
- User assignment with consistent hashing
- Event logging and tracking
- Integration with model serving

✅ **Analysis:**
- Statistical significance testing
- Sequential testing for early stopping
- Detecting common pitfalls (SRM, novelty effect)

✅ **Monitoring:**
- Real-time experiment dashboards
- Guardrail metrics
- Decision making frameworks

---

## Additional Resources

- [Trustworthy Online Controlled Experiments](https://experimentguide.com/) - Comprehensive guide to A/B testing
- [Evan Miller's A/B Testing Tools](https://www.evanmiller.org/ab-testing/) - Sample size calculators
- [Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)

---

## Next Lesson

**Lesson 08: MLOps Best Practices & Governance** - Learn best practices for running ML systems in production and governance frameworks.
