# Lesson 07: ML-Specific Monitoring and Model Observability

## Learning Objectives
By the end of this lesson, you will be able to:
- Monitor ML model performance in production
- Track data drift and model degradation
- Implement feature store monitoring
- Monitor training pipelines and experiments
- Set up model explainability and fairness monitoring
- Build comprehensive ML observability dashboards
- Integrate ML monitoring with existing observability stack

## Prerequisites
- Completion of Lessons 01-06 (Observability fundamentals)
- Understanding of ML concepts (training, inference, drift)
- Familiarity with Prometheus, Grafana, and Python
- Experience with ML frameworks (PyTorch, TensorFlow, or scikit-learn)

## Introduction

Traditional infrastructure monitoring focuses on system health (CPU, memory, latency). ML observability extends this to track **model behavior and performance**, including:
- Prediction quality over time
- Input data distribution shifts
- Feature importance changes
- Model fairness and bias
- Experiment tracking and comparison

### Why ML-Specific Monitoring?

1. **Models degrade**: Unlike traditional software, ML models can silently fail
2. **Data drift**: Input distributions change, reducing model accuracy
3. **Concept drift**: The relationship between features and target changes
4. **Training monitoring**: Track experiments, hyperparameters, and convergence
5. **Compliance**: Monitor fairness, bias, and explainability requirements

---

## 1. Model Performance Metrics

### Classification Models

```python
from prometheus_client import Gauge, Histogram
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Define metrics
MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy score',
    ['model_name', 'model_version', 'dataset']
)

MODEL_PRECISION = Gauge(
    'model_precision',
    'Model precision score',
    ['model_name', 'model_version', 'class']
)

MODEL_RECALL = Gauge(
    'model_recall',
    'Model recall score',
    ['model_name', 'model_version', 'class']
)

MODEL_F1 = Gauge(
    'model_f1_score',
    'Model F1 score',
    ['model_name', 'model_version', 'class']
)

MODEL_AUC = Gauge(
    'model_auc_roc',
    'Model AUC-ROC score',
    ['model_name', 'model_version']
)

CONFUSION_MATRIX = Gauge(
    'model_confusion_matrix',
    'Confusion matrix elements',
    ['model_name', 'model_version', 'true_class', 'predicted_class']
)

def update_classification_metrics(
    y_true, y_pred, y_prob,
    model_name, model_version, class_names
):
    """Update classification metrics"""

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    MODEL_ACCURACY.labels(
        model_name=model_name,
        model_version=model_version,
        dataset='production'
    ).set(accuracy)

    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    for i, class_name in enumerate(class_names):
        MODEL_PRECISION.labels(
            model_name=model_name,
            model_version=model_version,
            class_name=class_name
        ).set(precision[i])

        MODEL_RECALL.labels(
            model_name=model_name,
            model_version=model_version,
            class_name=class_name
        ).set(recall[i])

        MODEL_F1.labels(
            model_name=model_name,
            model_version=model_version,
            class_name=class_name
        ).set(f1[i])

    # AUC-ROC (if probabilities available)
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        MODEL_AUC.labels(
            model_name=model_name,
            model_version=model_version
        ).set(auc)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            CONFUSION_MATRIX.labels(
                model_name=model_name,
                model_version=model_version,
                true_class=true_class,
                predicted_class=pred_class
            ).set(cm[i, j])
```

### Regression Models

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

# Define metrics
MODEL_RMSE = Gauge(
    'model_rmse',
    'Root Mean Squared Error',
    ['model_name', 'model_version']
)

MODEL_MAE = Gauge(
    'model_mae',
    'Mean Absolute Error',
    ['model_name', 'model_version']
)

MODEL_R2 = Gauge(
    'model_r2_score',
    'R-squared score',
    ['model_name', 'model_version']
)

MODEL_MAPE = Gauge(
    'model_mape',
    'Mean Absolute Percentage Error',
    ['model_name', 'model_version']
)

PREDICTION_ERROR_DISTRIBUTION = Histogram(
    'model_prediction_error',
    'Distribution of prediction errors',
    ['model_name', 'model_version'],
    buckets=[-100, -50, -10, -5, -1, 0, 1, 5, 10, 50, 100]
)

def update_regression_metrics(y_true, y_pred, model_name, model_version):
    """Update regression metrics"""

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    MODEL_RMSE.labels(
        model_name=model_name,
        model_version=model_version
    ).set(rmse)

    MODEL_MAE.labels(
        model_name=model_name,
        model_version=model_version
    ).set(mae)

    MODEL_R2.labels(
        model_name=model_name,
        model_version=model_version
    ).set(r2)

    MODEL_MAPE.labels(
        model_name=model_name,
        model_version=model_version
    ).set(mape)

    # Error distribution
    errors = y_pred - y_true
    for error in errors:
        PREDICTION_ERROR_DISTRIBUTION.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(error)
```

---

## 2. Data Drift Detection

### Statistical Drift Detection

```python
from scipy.stats import ks_2samp, chi2_contingency
from prometheus_client import Gauge, Counter
import numpy as np

# Metrics
FEATURE_DRIFT_SCORE = Gauge(
    'feature_drift_score',
    'KS statistic for feature drift',
    ['model_name', 'feature_name']
)

DRIFT_DETECTED = Counter(
    'drift_detected_total',
    'Number of times drift was detected',
    ['model_name', 'feature_name', 'drift_type']
)

class DriftDetector:
    """Detect data drift using statistical tests"""

    def __init__(self, reference_data, model_name, threshold=0.05):
        self.reference_data = reference_data
        self.model_name = model_name
        self.threshold = threshold

    def check_numerical_drift(self, feature_name, current_data):
        """
        Check drift in numerical features using Kolmogorov-Smirnov test
        """
        reference = self.reference_data[feature_name]
        statistic, p_value = ks_2samp(reference, current_data)

        # Update metric
        FEATURE_DRIFT_SCORE.labels(
            model_name=self.model_name,
            feature_name=feature_name
        ).set(statistic)

        # Check for drift
        if p_value < self.threshold:
            DRIFT_DETECTED.labels(
                model_name=self.model_name,
                feature_name=feature_name,
                drift_type='numerical'
            ).inc()

            return True, statistic, p_value

        return False, statistic, p_value

    def check_categorical_drift(self, feature_name, current_data):
        """
        Check drift in categorical features using chi-squared test
        """
        reference = self.reference_data[feature_name]

        # Create contingency table
        ref_counts = reference.value_counts()
        curr_counts = current_data.value_counts()

        # Align indices
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

        contingency = np.array([ref_aligned, curr_aligned])
        statistic, p_value, _, _ = chi2_contingency(contingency)

        # Update metric
        FEATURE_DRIFT_SCORE.labels(
            model_name=self.model_name,
            feature_name=feature_name
        ).set(statistic)

        if p_value < self.threshold:
            DRIFT_DETECTED.labels(
                model_name=self.model_name,
                feature_name=feature_name,
                drift_type='categorical'
            ).inc()

            return True, statistic, p_value

        return False, statistic, p_value

# Usage
import pandas as pd

reference_df = pd.read_csv('reference_data.csv')
detector = DriftDetector(reference_df, model_name='fraud-detector')

# Check drift on new data
current_batch = pd.read_csv('current_batch.csv')

for column in reference_df.columns:
    if reference_df[column].dtype in ['float64', 'int64']:
        drift, stat, pval = detector.check_numerical_drift(
            column,
            current_batch[column]
        )
    else:
        drift, stat, pval = detector.check_categorical_drift(
            column,
            current_batch[column]
        )

    if drift:
        print(f"DRIFT DETECTED in {column}: stat={stat:.4f}, p={pval:.4f}")
```

### Population Stability Index (PSI)

```python
import numpy as np
from prometheus_client import Gauge

PSI_SCORE = Gauge(
    'feature_psi_score',
    'Population Stability Index',
    ['model_name', 'feature_name']
)

def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index

    PSI < 0.1: No significant change
    PSI 0.1-0.25: Moderate change
    PSI > 0.25: Significant change
    """
    # Create bins
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    # Calculate distributions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculate PSI
    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log(actual_percents / expected_percents)
    )

    return psi

# Usage
for feature in features:
    psi = calculate_psi(
        reference_df[feature],
        current_df[feature]
    )

    PSI_SCORE.labels(
        model_name='fraud-detector',
        feature_name=feature
    ).set(psi)

    if psi > 0.25:
        print(f"Significant drift in {feature}: PSI = {psi:.3f}")
```

---

## 3. Training Pipeline Monitoring

### Experiment Tracking with MLflow

```python
import mlflow
from prometheus_client import Gauge, Counter
import logging

# Metrics
EXPERIMENT_RUN = Counter(
    'mlflow_experiment_runs_total',
    'Total number of experiment runs',
    ['experiment_name', 'status']
)

TRAINING_METRIC = Gauge(
    'training_metric_value',
    'Training metric value',
    ['experiment_name', 'run_id', 'metric_name']
)

TRAINING_DURATION = Gauge(
    'training_duration_seconds',
    'Training duration in seconds',
    ['experiment_name', 'run_id']
)

class MLflowMonitor:
    """Monitor MLflow experiments"""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.logger = logging.getLogger(__name__)

    def track_training(self, model, train_fn, **params):
        """
        Track training with MLflow and export metrics to Prometheus
        """
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            try:
                # Log parameters
                mlflow.log_params(params)

                # Train model
                import time
                start_time = time.time()

                metrics = train_fn(model, **params)

                duration = time.time() - start_time

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Export to Prometheus
                for metric_name, value in metrics.items():
                    TRAINING_METRIC.labels(
                        experiment_name=self.experiment_name,
                        run_id=run_id,
                        metric_name=metric_name
                    ).set(value)

                TRAINING_DURATION.labels(
                    experiment_name=self.experiment_name,
                    run_id=run_id
                ).set(duration)

                # Mark as successful
                EXPERIMENT_RUN.labels(
                    experiment_name=self.experiment_name,
                    status='success'
                ).inc()

                self.logger.info(
                    f"Training completed: run_id={run_id}, duration={duration:.2f}s"
                )

                return run_id, metrics

            except Exception as e:
                # Mark as failed
                EXPERIMENT_RUN.labels(
                    experiment_name=self.experiment_name,
                    status='failed'
                ).inc()

                self.logger.error(f"Training failed: {e}", exc_info=True)
                raise

# Usage
monitor = MLflowMonitor('fraud-detection-experiments')

def train_model(model, learning_rate, batch_size, epochs):
    """Training function"""
    # Training logic here
    return {
        'train_loss': 0.123,
        'val_loss': 0.145,
        'train_accuracy': 0.95,
        'val_accuracy': 0.93
    }

run_id, metrics = monitor.track_training(
    model,
    train_model,
    learning_rate=0.001,
    batch_size=32,
    epochs=10
)
```

---

## 4. Model Explainability Monitoring

### SHAP Values Tracking

```python
import shap
import numpy as np
from prometheus_client import Gauge, Histogram

FEATURE_IMPORTANCE = Gauge(
    'model_feature_importance',
    'SHAP-based feature importance',
    ['model_name', 'feature_name']
)

SHAP_VALUE_DISTRIBUTION = Histogram(
    'model_shap_values',
    'Distribution of SHAP values',
    ['model_name', 'feature_name'],
    buckets=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0]
)

def monitor_feature_importance(model, X, feature_names, model_name):
    """Track feature importance using SHAP"""

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values (feature importance)
    importance = np.abs(shap_values).mean(axis=0)

    for i, feature in enumerate(feature_names):
        # Feature importance
        FEATURE_IMPORTANCE.labels(
            model_name=model_name,
            feature_name=feature
        ).set(importance[i])

        # SHAP value distribution
        for value in shap_values[:, i]:
            SHAP_VALUE_DISTRIBUTION.labels(
                model_name=model_name,
                feature_name=feature
            ).observe(value)
```

---

## 5. Model Fairness Monitoring

```python
from prometheus_client import Gauge
import numpy as np

FAIRNESS_METRIC = Gauge(
    'model_fairness_metric',
    'Fairness metric value',
    ['model_name', 'protected_attribute', 'metric_type']
)

def calculate_fairness_metrics(
    y_true, y_pred, sensitive_attribute, model_name
):
    """
    Calculate fairness metrics across sensitive attributes
    """
    unique_groups = np.unique(sensitive_attribute)

    # Demographic parity difference
    positive_rates = {}
    for group in unique_groups:
        mask = sensitive_attribute == group
        positive_rate = y_pred[mask].mean()
        positive_rates[group] = positive_rate

    dp_diff = max(positive_rates.values()) - min(positive_rates.values())

    FAIRNESS_METRIC.labels(
        model_name=model_name,
        protected_attribute='all',
        metric_type='demographic_parity_diff'
    ).set(dp_diff)

    # Equal opportunity difference (TPR parity)
    tpr_by_group = {}
    for group in unique_groups:
        mask = (sensitive_attribute == group) & (y_true == 1)
        if mask.sum() > 0:
            tpr = y_pred[mask].mean()
            tpr_by_group[group] = tpr

    if len(tpr_by_group) > 0:
        eo_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())

        FAIRNESS_METRIC.labels(
            model_name=model_name,
            protected_attribute='all',
            metric_type='equal_opportunity_diff'
        ).set(eo_diff)

    return {
        'demographic_parity_diff': dp_diff,
        'equal_opportunity_diff': eo_diff if len(tpr_by_group) > 0 else None
    }
```

---

## 6. Complete ML Observability Dashboard

**Grafana Dashboard JSON (Excerpt):**

```json
{
  "dashboard": {
    "title": "ML Model Observability",
    "panels": [
      {
        "title": "Model Accuracy Over Time",
        "type": "timeseries",
        "targets": [{
          "expr": "model_accuracy{model_name=\"fraud-detector\"}",
          "legendFormat": "{{ model_version }}"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 0.85, "color": "yellow"},
                {"value": 0.92, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "title": "Feature Drift (PSI)",
        "type": "bargauge",
        "targets": [{
          "expr": "feature_psi_score{model_name=\"fraud-detector\"}"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.1, "color": "yellow"},
                {"value": 0.25, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "title": "Training Loss",
        "type": "timeseries",
        "targets": [{
          "expr": "training_metric_value{metric_name=\"train_loss\"}",
          "legendFormat": "Run {{ run_id }}"
        }]
      },
      {
        "title": "Prediction Distribution",
        "type": "heatmap",
        "targets": [{
          "expr": "rate(model_prediction_value_bucket[5m])"
        }]
      }
    ]
  }
}
```

---

## Summary

In this lesson, you learned:

✅ Monitoring ML model performance (accuracy, precision, recall, etc.)
✅ Detecting data drift with statistical tests (KS test, PSI, Chi-squared)
✅ Tracking training pipelines and experiments with MLflow
✅ Monitoring model explainability with SHAP values
✅ Measuring model fairness across protected attributes
✅ Building comprehensive ML observability dashboards

## Next Steps

- **Lesson 08**: Best practices and observability culture
- **Practice**: Implement drift detection for your ML models
- **Exercise**: Create a complete ML observability pipeline

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Evidently AI (Drift Detection)](https://docs.evidentlyai.com/)
- [Fairlearn (Fairness Assessment)](https://fairlearn.org/)
- [Alibi (Model Explainability)](https://docs.seldon.io/projects/alibi/en/stable/)

---

**Estimated Time:** 5-6 hours
**Difficulty:** Advanced
**Prerequisites:** Lessons 01-06, ML fundamentals, Statistics
