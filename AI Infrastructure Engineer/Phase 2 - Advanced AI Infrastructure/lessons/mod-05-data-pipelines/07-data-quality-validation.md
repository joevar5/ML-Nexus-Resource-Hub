# Lesson 07: Data Quality and Validation

## Overview
Data quality is critical for ML systems - poor data quality leads to unreliable models and incorrect predictions. This lesson covers comprehensive data validation, quality monitoring, and remediation strategies for ML pipelines.

**Duration:** 6-8 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, pandas/PySpark, understanding of data pipelines

## Learning Objectives
By the end of this lesson, you will be able to:
- Implement comprehensive data validation frameworks
- Detect data quality issues in ML pipelines
- Build automated data quality monitoring
- Handle schema evolution and data drift
- Create data contracts and expectations
- Implement data quality gates in CI/CD

---

## 1. Introduction to Data Quality

### 1.1 Dimensions of Data Quality

```
┌────────────────────────────────────────────────────────┐
│         Data Quality Dimensions for ML                  │
├────────────────────────────────────────────────────────┤
│ 1. Completeness    - No missing required values        │
│ 2. Validity        - Values within expected ranges     │
│ 3. Accuracy        - Values represent true state       │
│ 4. Consistency     - No contradictions across sources  │
│ 5. Timeliness      - Data is up-to-date               │
│ 6. Uniqueness      - No unwanted duplicates            │
│ 7. Integrity       - Referential relationships intact  │
└────────────────────────────────────────────────────────┘
```

### 1.2 Why Data Quality Matters for ML

```python
# Impact of poor data quality on ML

# Scenario 1: Missing values
# Training: 95% accuracy
# Production: 60% accuracy (missing features filled with 0)

# Scenario 2: Data drift
# Training: Age range [18-65]
# Production: Age range [0-120] with outliers → poor predictions

# Scenario 3: Label leakage
# Training includes future information → overfitted model

# Scenario 4: Inconsistent encoding
# Training: country codes ['US', 'UK', 'CA']
# Production: country codes ['USA', 'United Kingdom', 'Canada']
```

### 1.3 Cost of Poor Data Quality

**Direct costs:**
- Failed predictions → business losses
- Model retraining costs
- Debugging and remediation time
- Customer trust damage

**Indirect costs:**
- Team productivity loss
- Delayed feature releases
- Technical debt accumulation

---

## 2. Data Validation Framework

### 2.1 Basic Validation Rules

```python
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataValidator:
    """Comprehensive data validation framework"""

    def __init__(self):
        self.validation_results = []

    def check_not_null(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = 0.0
    ) -> bool:
        """Check columns have at most threshold fraction of nulls"""
        results = {}

        for col in columns:
            null_fraction = df[col].isna().sum() / len(df)
            passed = null_fraction <= threshold

            results[col] = {
                'passed': passed,
                'null_fraction': null_fraction,
                'threshold': threshold
            }

            if not passed:
                self.validation_results.append({
                    'check': 'not_null',
                    'column': col,
                    'passed': False,
                    'details': f"{null_fraction:.2%} nulls (threshold: {threshold:.2%})"
                })

        return all(r['passed'] for r in results.values())

    def check_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """Check numeric column values are within range"""
        values = df[column].dropna()

        violations = 0
        if min_value is not None:
            violations += (values < min_value).sum()
        if max_value is not None:
            violations += (values > max_value).sum()

        passed = violations == 0

        if not passed:
            self.validation_results.append({
                'check': 'range',
                'column': column,
                'passed': False,
                'details': f"{violations} values out of range [{min_value}, {max_value}]"
            })

        return passed

    def check_values_in_set(
        self,
        df: pd.DataFrame,
        column: str,
        allowed_values: set
    ) -> bool:
        """Check categorical column contains only allowed values"""
        unique_values = set(df[column].dropna().unique())
        invalid_values = unique_values - allowed_values

        passed = len(invalid_values) == 0

        if not passed:
            self.validation_results.append({
                'check': 'values_in_set',
                'column': column,
                'passed': False,
                'details': f"Invalid values: {invalid_values}"
            })

        return passed

    def check_unique(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> bool:
        """Check columns have unique values"""
        duplicates = df.duplicated(subset=columns).sum()
        passed = duplicates == 0

        if not passed:
            self.validation_results.append({
                'check': 'unique',
                'columns': columns,
                'passed': False,
                'details': f"{duplicates} duplicate rows found"
            })

        return passed

    def check_data_freshness(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
        max_age_hours: int
    ) -> bool:
        """Check data is not too old"""
        latest_timestamp = pd.to_datetime(df[timestamp_column]).max()
        age = datetime.now() - latest_timestamp

        passed = age <= timedelta(hours=max_age_hours)

        if not passed:
            self.validation_results.append({
                'check': 'freshness',
                'column': timestamp_column,
                'passed': False,
                'details': f"Data is {age.total_seconds()/3600:.1f} hours old (max: {max_age_hours})"
            })

        return passed

    def check_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        expected_mean: float,
        tolerance: float = 0.2
    ) -> bool:
        """Check distribution hasn't shifted significantly"""
        actual_mean = df[column].mean()
        diff = abs(actual_mean - expected_mean) / expected_mean

        passed = diff <= tolerance

        if not passed:
            self.validation_results.append({
                'check': 'distribution',
                'column': column,
                'passed': False,
                'details': f"Mean shifted by {diff:.2%} (tolerance: {tolerance:.2%})"
            })

        return passed

    def get_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        total_checks = len(self.validation_results)
        failed_checks = sum(1 for r in self.validation_results if not r['passed'])

        return {
            'total_checks': total_checks,
            'failed_checks': failed_checks,
            'success_rate': 1 - (failed_checks / max(total_checks, 1)),
            'failures': [r for r in self.validation_results if not r['passed']]
        }

# Usage
validator = DataValidator()

# Load data
df = pd.read_csv('training_data.csv')

# Run validations
validator.check_not_null(df, ['user_id', 'age', 'country'], threshold=0.01)
validator.check_range(df, 'age', min_value=18, max_value=100)
validator.check_values_in_set(df, 'country', {'US', 'UK', 'CA', 'AU'})
validator.check_unique(df, ['user_id'])
validator.check_data_freshness(df, 'created_at', max_age_hours=24)
validator.check_distribution(df, 'age', expected_mean=35.0, tolerance=0.2)

# Get report
report = validator.get_report()
print(f"Validation success rate: {report['success_rate']:.2%}")
for failure in report['failures']:
    print(f"❌ {failure['check']} failed: {failure['details']}")
```

---

## 3. Great Expectations Framework

### 3.1 Introduction to Great Expectations

Great Expectations is an industry-standard framework for data validation.

```bash
# Install Great Expectations
pip install great-expectations==0.18.0
```

### 3.2 Basic Great Expectations Usage

```python
import great_expectations as gx
from great_expectations.dataset import PandasDataset

# Create expectations suite
df = pd.read_csv('training_data.csv')
ge_df = PandasDataset(df)

# Define expectations
ge_df.expect_column_to_exist('user_id')
ge_df.expect_column_values_to_not_be_null('user_id')
ge_df.expect_column_values_to_be_unique('user_id')

ge_df.expect_column_values_to_be_between('age', min_value=18, max_value=100)
ge_df.expect_column_values_to_be_in_set('country', ['US', 'UK', 'CA', 'AU'])

ge_df.expect_column_mean_to_be_between('age', min_value=30, max_value=40)
ge_df.expect_column_stdev_to_be_between('age', min_value=5, max_value=15)

# Validate
results = ge_df.validate()
print(f"Success: {results['success']}")
print(f"Evaluated expectations: {results['statistics']['evaluated_expectations']}")
print(f"Successful expectations: {results['statistics']['successful_expectations']}")
```

### 3.3 Advanced Great Expectations

```python
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    InMemoryStoreBackendDefaults
)

class MLDataQualityChecker:
    """ML-focused data quality checker with Great Expectations"""

    def __init__(self):
        # Create in-memory context
        data_context_config = DataContextConfig(
            store_backend_defaults=InMemoryStoreBackendDefaults()
        )
        self.context = BaseDataContext(project_config=data_context_config)

    def create_training_data_expectations(self, suite_name: str):
        """Create expectations for training data"""
        self.context.add_or_update_expectation_suite(suite_name)
        suite = self.context.get_expectation_suite(suite_name)

        # Schema expectations
        expectations = [
            # Required columns
            {
                "expectation_type": "expect_table_columns_to_match_ordered_list",
                "kwargs": {
                    "column_list": [
                        "user_id", "age", "country", "spend",
                        "feature_1", "feature_2", "label"
                    ]
                }
            },
            # No nulls in critical columns
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "user_id"}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "label"}
            },
            # Data types
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "user_id", "type_": "str"}
            },
            {
                "expectation_type": "expect_column_values_to_be_of_type",
                "kwargs": {"column": "age", "type_": "int"}
            },
            # Value constraints
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "column": "age",
                    "min_value": 18,
                    "max_value": 100
                }
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "country",
                    "value_set": ["US", "UK", "CA", "AU"]
                }
            },
            # Distribution checks
            {
                "expectation_type": "expect_column_mean_to_be_between",
                "kwargs": {
                    "column": "age",
                    "min_value": 30,
                    "max_value": 40
                }
            },
            # Uniqueness
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {"column": "user_id"}
            },
            # Row count
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {
                    "min_value": 1000,
                    "max_value": 10000000
                }
            }
        ]

        for exp in expectations:
            suite.add_expectation(exp)

        self.context.save_expectation_suite(suite)
        return suite

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        checkpoint_name: str = "ml_checkpoint"
    ) -> Dict[str, Any]:
        """Validate dataframe against expectations"""

        # Create checkpoint
        checkpoint_config = {
            "name": checkpoint_name,
            "config_version": 1,
            "class_name": "SimpleCheckpoint",
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": "pandas_datasource",
                        "data_connector_name": "runtime_data_connector",
                        "data_asset_name": "training_data"
                    },
                    "expectation_suite_name": suite_name
                }
            ]
        }

        # Add datasource
        datasource_config = {
            "name": "pandas_datasource",
            "class_name": "Datasource",
            "execution_engine": {
                "class_name": "PandasExecutionEngine"
            },
            "data_connectors": {
                "runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["batch_id"]
                }
            }
        }

        self.context.add_or_update_datasource(**datasource_config)
        self.context.add_or_update_checkpoint(**checkpoint_config)

        # Run validation
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name="training_data",
            runtime_parameters={"batch_data": df},
            batch_identifiers={"batch_id": "training_batch"}
        )

        results = self.context.run_checkpoint(
            checkpoint_name=checkpoint_name,
            batch_request=batch_request
        )

        return results

# Usage
checker = MLDataQualityChecker()
checker.create_training_data_expectations("training_data_suite")

df = pd.read_csv('training_data.csv')
results = checker.validate_dataframe(df, "training_data_suite")

print(f"Validation success: {results.success}")
```

---

## 4. Schema Validation and Evolution

### 4.1 Schema Enforcement

```python
from typing import Dict, Type
import pandera as pa
from pandera import Column, Check, DataFrameSchema

class MLDataSchema:
    """Define and enforce ML data schemas"""

    @staticmethod
    def training_data_schema() -> DataFrameSchema:
        """Schema for training data"""
        return DataFrameSchema(
            columns={
                "user_id": Column(
                    dtype="str",
                    checks=[
                        Check(lambda s: s.str.match(r'^user_\d+$').all(),
                              error="user_id must match pattern 'user_XXX'"),
                        Check(lambda s: ~s.duplicated().any(),
                              error="user_id must be unique")
                    ],
                    nullable=False
                ),
                "age": Column(
                    dtype="int",
                    checks=[
                        Check.in_range(min_value=18, max_value=100),
                        Check(lambda s: s.median() > 25,
                              error="Median age too low")
                    ],
                    nullable=False
                ),
                "country": Column(
                    dtype="str",
                    checks=[
                        Check.isin(["US", "UK", "CA", "AU"])
                    ],
                    nullable=False
                ),
                "spend": Column(
                    dtype="float",
                    checks=[
                        Check.greater_than_or_equal_to(0),
                        Check.less_than(10000)
                    ],
                    nullable=True
                ),
                "label": Column(
                    dtype="int",
                    checks=[Check.isin([0, 1])],
                    nullable=False
                )
            },
            checks=[
                # Table-level checks
                Check(lambda df: len(df) >= 1000,
                      error="Dataset too small (< 1000 rows)"),
                Check(lambda df: df['label'].mean() > 0.01,
                      error="Label imbalance too extreme"),
                Check(lambda df: df['label'].mean() < 0.99,
                      error="Label imbalance too extreme")
            ],
            strict=True,
            coerce=True
        )

    @staticmethod
    def inference_data_schema() -> DataFrameSchema:
        """Schema for inference data (no label required)"""
        return DataFrameSchema(
            columns={
                "user_id": Column(dtype="str", nullable=False),
                "age": Column(dtype="int", checks=[Check.in_range(18, 100)]),
                "country": Column(dtype="str", checks=[Check.isin(["US", "UK", "CA", "AU"])]),
                "spend": Column(dtype="float", checks=[Check.greater_than_or_equal_to(0)])
            },
            strict=False,  # Allow extra columns
            coerce=True
        )

# Usage
schema = MLDataSchema.training_data_schema()

try:
    df = pd.read_csv('training_data.csv')
    validated_df = schema.validate(df, lazy=True)
    print("✅ Data validation passed")

except pa.errors.SchemaErrors as e:
    print("❌ Data validation failed:")
    print(e.failure_cases)
```

### 4.2 Schema Evolution

```python
from typing import List, Dict
import json

class SchemaVersionManager:
    """Manage schema versions and compatibility"""

    def __init__(self, schema_registry_path: str):
        self.schema_registry_path = schema_registry_path
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict]:
        """Load schema registry"""
        try:
            with open(self.schema_registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def register_schema(
        self,
        name: str,
        version: str,
        schema: Dict[str, Any]
    ):
        """Register new schema version"""
        if name not in self.schemas:
            self.schemas[name] = {}

        self.schemas[name][version] = {
            'schema': schema,
            'timestamp': datetime.now().isoformat()
        }

        self._save_schemas()

    def _save_schemas(self):
        """Persist schema registry"""
        with open(self.schema_registry_path, 'w') as f:
            json.dump(self.schemas, f, indent=2)

    def check_compatibility(
        self,
        name: str,
        old_version: str,
        new_version: str
    ) -> bool:
        """Check if schema evolution is backward compatible"""
        old_schema = self.schemas[name][old_version]['schema']
        new_schema = self.schemas[name][new_version]['schema']

        old_cols = set(old_schema['columns'].keys())
        new_cols = set(new_schema['columns'].keys())

        # Backward compatibility rules:
        # 1. Can't remove required columns
        # 2. Can add optional columns
        # 3. Can't make columns more restrictive

        removed_cols = old_cols - new_cols
        if removed_cols:
            print(f"❌ Removed columns: {removed_cols}")
            return False

        for col in old_cols & new_cols:
            old_nullable = old_schema['columns'][col].get('nullable', False)
            new_nullable = new_schema['columns'][col].get('nullable', False)

            if old_nullable and not new_nullable:
                print(f"❌ Column {col} became non-nullable")
                return False

        print("✅ Schema evolution is backward compatible")
        return True

# Usage
manager = SchemaVersionManager('schema_registry.json')

# Register v1
schema_v1 = {
    'columns': {
        'user_id': {'type': 'str', 'nullable': False},
        'age': {'type': 'int', 'nullable': False}
    }
}
manager.register_schema('user_features', 'v1', schema_v1)

# Register v2 (adds optional column)
schema_v2 = {
    'columns': {
        'user_id': {'type': 'str', 'nullable': False},
        'age': {'type': 'int', 'nullable': False},
        'country': {'type': 'str', 'nullable': True}  # New optional column
    }
}
manager.register_schema('user_features', 'v2', schema_v2)

# Check compatibility
is_compatible = manager.check_compatibility('user_features', 'v1', 'v2')
```

---

## 5. Data Quality Monitoring

### 5.1 Statistical Monitoring

```python
import numpy as np
from scipy import stats
from typing import Tuple

class DataDriftMonitor:
    """Monitor data drift over time"""

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df
        self.baseline_stats = self._compute_stats(baseline_df)

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute statistical profile"""
        stats_dict = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }

        for col in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[col].value_counts(normalize=True)
            stats_dict[col] = {
                'distribution': value_counts.to_dict(),
                'cardinality': len(value_counts),
                'top_value': value_counts.index[0],
                'top_value_freq': value_counts.iloc[0]
            }

        return stats_dict

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect drift using statistical tests"""
        drift_report = {
            'has_drift': False,
            'drifted_columns': [],
            'details': {}
        }

        # Numerical columns: KS test
        for col in current_df.select_dtypes(include=[np.number]).columns:
            if col not in self.baseline_df.columns:
                continue

            baseline_values = self.baseline_df[col].dropna()
            current_values = current_df[col].dropna()

            statistic, p_value = stats.ks_2samp(baseline_values, current_values)

            drifted = p_value < threshold

            drift_report['details'][col] = {
                'test': 'kolmogorov_smirnov',
                'statistic': statistic,
                'p_value': p_value,
                'drifted': drifted,
                'baseline_mean': baseline_values.mean(),
                'current_mean': current_values.mean(),
                'mean_diff_pct': (
                    (current_values.mean() - baseline_values.mean()) /
                    baseline_values.mean() * 100
                )
            }

            if drifted:
                drift_report['has_drift'] = True
                drift_report['drifted_columns'].append(col)

        # Categorical columns: Chi-square test
        for col in current_df.select_dtypes(include=['object', 'category']).columns:
            if col not in self.baseline_df.columns:
                continue

            baseline_dist = self.baseline_df[col].value_counts()
            current_dist = current_df[col].value_counts()

            # Align distributions
            all_categories = set(baseline_dist.index) | set(current_dist.index)
            baseline_counts = [baseline_dist.get(cat, 0) for cat in all_categories]
            current_counts = [current_dist.get(cat, 0) for cat in all_categories]

            statistic, p_value = stats.chisquare(
                f_obs=current_counts,
                f_exp=baseline_counts
            )

            drifted = p_value < threshold

            drift_report['details'][col] = {
                'test': 'chi_square',
                'statistic': statistic,
                'p_value': p_value,
                'drifted': drifted
            }

            if drifted:
                drift_report['has_drift'] = True
                drift_report['drifted_columns'].append(col)

        return drift_report

# Usage
baseline_df = pd.read_csv('baseline_data.csv')
monitor = DataDriftMonitor(baseline_df)

current_df = pd.read_csv('current_data.csv')
drift_report = monitor.detect_drift(current_df)

if drift_report['has_drift']:
    print(f"⚠️  Data drift detected in columns: {drift_report['drifted_columns']}")
    for col in drift_report['drifted_columns']:
        details = drift_report['details'][col]
        print(f"  {col}: p-value = {details['p_value']:.4f}")
else:
    print("✅ No significant data drift detected")
```

### 5.2 Continuous Monitoring Pipeline

```python
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class DataQualityMetrics:
    """Data quality metrics snapshot"""
    timestamp: datetime
    row_count: int
    null_counts: Dict[str, int]
    duplicate_count: int
    mean_values: Dict[str, float]
    std_values: Dict[str, float]
    validation_score: float

class ContinuousQualityMonitor:
    """Continuous data quality monitoring"""

    def __init__(self, validator: DataValidator, drift_monitor: DataDriftMonitor):
        self.validator = validator
        self.drift_monitor = drift_monitor
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)

    def monitor_batch(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Monitor single batch of data"""
        # Run validations
        self.validator.check_not_null(df, df.columns.tolist(), threshold=0.05)
        self.validator.check_unique(df, ['user_id'])

        validation_report = self.validator.get_report()

        # Check drift
        drift_report = self.drift_monitor.detect_drift(df)

        # Compute metrics
        metrics = DataQualityMetrics(
            timestamp=datetime.now(),
            row_count=len(df),
            null_counts={col: df[col].isna().sum() for col in df.columns},
            duplicate_count=df.duplicated().sum(),
            mean_values={
                col: df[col].mean()
                for col in df.select_dtypes(include=[np.number]).columns
            },
            std_values={
                col: df[col].std()
                for col in df.select_dtypes(include=[np.number]).columns
            },
            validation_score=validation_report['success_rate']
        )

        self.metrics_history.append(metrics)

        # Alert on issues
        if validation_report['failed_checks'] > 0:
            self.logger.warning(
                f"Data quality issues: {validation_report['failed_checks']} checks failed"
            )

        if drift_report['has_drift']:
            self.logger.warning(
                f"Data drift detected: {drift_report['drifted_columns']}"
            )

        return metrics

    def get_quality_trends(self) -> Dict[str, List[float]]:
        """Analyze quality trends over time"""
        trends = {
            'timestamps': [m.timestamp for m in self.metrics_history],
            'row_counts': [m.row_count for m in self.metrics_history],
            'validation_scores': [m.validation_score for m in self.metrics_history]
        }
        return trends

# Usage with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta

def monitor_data_quality(**context):
    """Airflow task for data quality monitoring"""
    # Load data
    df = pd.read_csv(context['ti'].xcom_pull(task_ids='load_data'))

    # Monitor
    monitor = ContinuousQualityMonitor(validator, drift_monitor)
    metrics = monitor.monitor_batch(df)

    # Store metrics
    context['ti'].xcom_push(key='quality_metrics', value=metrics.__dict__)

    # Fail task if quality too low
    if metrics.validation_score < 0.95:
        raise Exception(f"Data quality too low: {metrics.validation_score:.2%}")

dag = DAG(
    'data_quality_monitoring',
    schedule_interval='@hourly',
    default_args={'retries': 2}
)

quality_check = PythonOperator(
    task_id='check_quality',
    python_callable=monitor_data_quality,
    dag=dag
)
```

---

## 6. Handling Data Quality Issues

### 6.1 Automated Remediation

```python
class DataQualityRemediator:
    """Automated data quality issue remediation"""

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: Dict[str, str]
    ) -> pd.DataFrame:
        """Handle missing values per column"""
        df = df.copy()

        for col, method in strategy.items():
            if method == 'drop':
                df = df.dropna(subset=[col])
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'constant':
                df[col] = df[col].fillna(0)

        return df

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers from numerical column"""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            return df[(df[column] >= lower) & (df[column] <= upper)]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            return df[z_scores < threshold]

        return df

    @staticmethod
    def deduplicate(
        df: pd.DataFrame,
        subset: List[str],
        keep: str = 'first'
    ) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates(subset=subset, keep=keep)

# Usage
remediator = DataQualityRemediator()

# Handle missing values
missing_strategy = {
    'age': 'median',
    'country': 'mode',
    'spend': 'mean'
}
df = remediator.handle_missing_values(df, missing_strategy)

# Remove outliers
df = remediator.remove_outliers(df, 'spend', method='iqr', threshold=1.5)

# Deduplicate
df = remediator.deduplicate(df, subset=['user_id'], keep='last')
```

---

## 7. Best Practices

### 7.1 Data Quality Checklist

✅ **DO:**
- Validate data at every pipeline stage
- Monitor data quality continuously
- Set up alerts for quality degradation
- Document data quality expectations
- Version control your validation rules
- Test validation logic itself

❌ **DON'T:**
- Skip validation for "trusted" sources
- Ignore small quality issues
- Hard-code validation thresholds
- Mix validation and transformation logic
- Assume data quality is static

### 7.2 Integration with CI/CD

```python
# tests/test_data_quality.py
import pytest

def test_training_data_schema():
    """Test training data meets schema requirements"""
    df = pd.read_csv('data/training_data.csv')
    schema = MLDataSchema.training_data_schema()

    # Should not raise exception
    validated_df = schema.validate(df)

    assert len(validated_df) > 1000
    assert validated_df['label'].mean() > 0.01

def test_no_data_drift():
    """Test current data hasn't drifted from baseline"""
    baseline_df = pd.read_csv('data/baseline.csv')
    current_df = pd.read_csv('data/current.csv')

    monitor = DataDriftMonitor(baseline_df)
    drift_report = monitor.detect_drift(current_df, threshold=0.01)

    assert not drift_report['has_drift'], \
        f"Data drift detected: {drift_report['drifted_columns']}"

def test_data_freshness():
    """Test data is recent enough"""
    df = pd.read_csv('data/training_data.csv')
    latest = pd.to_datetime(df['timestamp']).max()
    age_hours = (datetime.now() - latest).total_seconds() / 3600

    assert age_hours < 24, f"Data is {age_hours:.1f} hours old"
```

---

## 8. Summary

Key takeaways:
- ✅ Data quality is critical for ML success
- ✅ Validate early, validate often
- ✅ Monitor for drift and degradation
- ✅ Automate quality checks in pipelines
- ✅ Use established frameworks (Great Expectations, Pandera)
- ✅ Handle schema evolution carefully

**Next Lesson:** [08 - Pipeline Monitoring and Error Handling](./08-pipeline-monitoring-errors.md)
