# Lesson 01: Introduction to MLOps

## Overview
MLOps (Machine Learning Operations) brings DevOps principles and practices to machine learning systems. This lesson introduces the fundamentals of MLOps, explains why it's critical for production ML, and provides a framework for building reliable ML systems.

**Duration:** 4-5 hours
**Difficulty:** Beginner to Intermediate
**Prerequisites:** Basic ML knowledge, understanding of DevOps concepts

## Learning Objectives
By the end of this lesson, you will be able to:
- Define MLOps and explain its importance
- Understand the complete ML lifecycle
- Identify common challenges in production ML
- Assess ML system maturity levels
- Design MLOps architectures
- Choose appropriate tools for MLOps

---

## 1. What is MLOps?

### 1.1 Definition

**MLOps** is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

```
MLOps = ML + DevOps + Data Engineering

┌─────────────────────────────────────────────┐
│          Machine Learning (ML)              │
│   • Model development                       │
│   • Feature engineering                     │
│   • Hyperparameter tuning                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│             MLOps Practices                 │
│   • Automation                              │
│   • Monitoring                              │
│   • Reproducibility                         │
│   • Collaboration                           │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼─────────┐  ┌────────▼──────────┐
│    DevOps       │  │ Data Engineering  │
│ • CI/CD         │  │ • Data pipelines  │
│ • Infrastructure│  │ • Data quality    │
│ • Monitoring    │  │ • Feature stores  │
└─────────────────┘  └───────────────────┘
```

### 1.2 Why MLOps Matters

**Traditional Software vs ML Systems:**

| Aspect | Traditional Software | ML Systems |
|--------|---------------------|------------|
| Code changes | Code only | Code + Data + Model |
| Testing | Unit/integration tests | Model validation + data validation |
| Deployment | Binary artifacts | Models + code + config |
| Monitoring | System metrics | Model performance + data drift |
| Debugging | Stack traces | Model behavior analysis |
| Updates | Code releases | Retraining + deployment |

**Statistics:**
- 87% of ML projects never make it to production (VentureBeat, 2019)
- Only 22% of companies using ML have successfully deployed a model (Gartner, 2020)
- Average time to deploy ML model: 8-90 days (MLOps Community Survey, 2021)

**Why MLOps is essential:**
1. **Reproducibility**: Ensure experiments can be recreated
2. **Reliability**: Build dependable ML systems
3. **Scalability**: Handle increasing model complexity and volume
4. **Collaboration**: Enable teams to work together effectively
5. **Compliance**: Meet regulatory and governance requirements
6. **Speed**: Accelerate model development and deployment

---

## 2. The ML Lifecycle

### 2.1 Complete ML Lifecycle

```
┌──────────────────────────────────────────────────────────┐
│                    ML Lifecycle                          │
└──────────────────────────────────────────────────────────┘

1. Business Problem Definition
   ↓
2. Data Collection & Preparation
   ├─ Data ingestion
   ├─ Data validation
   ├─ Data cleaning
   └─ Feature engineering
   ↓
3. Model Development
   ├─ Experiment tracking
   ├─ Model training
   ├─ Hyperparameter tuning
   └─ Model evaluation
   ↓
4. Model Validation
   ├─ Performance metrics
   ├─ Business metrics
   └─ Fairness/bias check
   ↓
5. Model Deployment
   ├─ Model packaging
   ├─ Infrastructure setup
   └─ Release strategy
   ↓
6. Model Monitoring
   ├─ Performance tracking
   ├─ Data drift detection
   └─ Alert management
   ↓
7. Model Maintenance
   ├─ Retraining
   ├─ Version management
   └─ Decommissioning

   (Loop back to step 2 or 3)
```

### 2.2 Continuous Processes in MLOps

```python
# Example: ML Lifecycle Implementation

class MLLifecycle:
    """Orchestrate complete ML lifecycle"""

    def __init__(self):
        self.experiment_tracker = MLflowTracker()
        self.data_validator = DataValidator()
        self.model_registry = ModelRegistry()
        self.monitor = ModelMonitor()

    def execute_lifecycle(self, config):
        """Execute full ML lifecycle"""

        # 1. Data preparation
        raw_data = self.collect_data(config['data_source'])
        self.data_validator.validate(raw_data)
        clean_data = self.prepare_data(raw_data)

        # 2. Feature engineering
        features = self.engineer_features(clean_data)
        self.data_validator.validate_features(features)

        # 3. Model training
        with self.experiment_tracker.start_run():
            model = self.train_model(features, config['model_params'])
            metrics = self.evaluate_model(model, features)

            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.log_model(model)

        # 4. Model validation
        if self.validate_model(model, metrics, config['thresholds']):
            # 5. Model registration
            model_version = self.model_registry.register_model(
                model,
                stage='staging'
            )

            # 6. Deploy to staging
            self.deploy_model(model_version, environment='staging')

            # 7. Monitor performance
            self.monitor.setup_monitoring(model_version)

            return model_version
        else:
            raise ModelValidationError("Model did not meet requirements")
```

---

## 3. MLOps vs DevOps

### 3.1 Key Differences

```
┌─────────────────────────────────────────────────────────┐
│              DevOps         vs        MLOps             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Artifacts:                                             │
│  • Code                      • Code + Data + Models     │
│                                                          │
│  Testing:                                               │
│  • Unit tests                • Unit + Model + Data tests│
│  • Integration tests         • A/B testing              │
│                                                          │
│  Deployment:                                            │
│  • Rolling updates           • Canary + shadow mode     │
│  • Blue-green                • Multi-model serving      │
│                                                          │
│  Monitoring:                                            │
│  • System metrics            • Model performance        │
│  • Error rates               • Data drift               │
│  • Latency                   • Feature distribution     │
│                                                          │
│  Triggers:                                              │
│  • Code changes              • Data changes             │
│                              • Model degradation        │
│                              • Schedule                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Additional MLOps Complexity

```python
# DevOps: Deploy when code changes
def devops_pipeline():
    code = checkout_code()
    run_tests(code)
    build_artifact(code)
    deploy(artifact)

# MLOps: Deploy when code, data, OR model changes
def mlops_pipeline():
    # Multiple triggers
    code = checkout_code()
    data = fetch_latest_data()

    # Validate everything
    run_code_tests(code)
    run_data_tests(data)

    # Train and validate model
    model = train_model(code, data)
    model_metrics = evaluate_model(model)

    # Check if deployment needed
    if should_deploy(model_metrics, data_drift):
        # Package everything
        artifact = package_model(model, code, data_schema)

        # Deploy with monitoring
        deploy(artifact)
        monitor_model_performance(model)

        # Check for drift
        if detect_drift():
            trigger_retraining()
```

---

## 4. Common Challenges in Production ML

### 4.1 Technical Challenges

**1. Reproducibility**
```python
# Problem: Can't reproduce experiment results

# Bad: Non-reproducible experiment
def train_model():
    data = load_data()  # Which version?
    model = RandomForest()  # What hyperparameters?
    model.fit(data)  # What seed?
    return model  # No tracking

# Good: Reproducible experiment
import mlflow

def train_model(data_version, seed=42, **model_params):
    with mlflow.start_run():
        # Log everything
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("seed", seed)
        mlflow.log_params(model_params)

        # Load versioned data
        data = load_data(version=data_version)

        # Train with fixed seed
        model = RandomForest(random_state=seed, **model_params)
        model.fit(data.X_train, data.y_train)

        # Log model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(evaluate(model, data))

        return model
```

**2. Data Quality Issues**
- Missing values in production
- Schema changes
- Data drift
- Label quality problems

**3. Model Performance Degradation**
- Concept drift
- Feature distribution shift
- Seasonal patterns
- Adversarial inputs

**4. Deployment Complexity**
- Multiple deployment targets (cloud, edge, mobile)
- Latency requirements
- Scalability needs
- Resource constraints

### 4.2 Organizational Challenges

**1. Team Silos**
```
Data Scientists → Models in notebooks
        ↓ (handoff gap)
ML Engineers → Production systems
        ↓ (handoff gap)
DevOps → Infrastructure management
```

**Solution: Cross-functional MLOps teams**
```
┌────────────────────────────────────┐
│      MLOps Team                    │
├────────────────────────────────────┤
│ • Data Scientists                  │
│ • ML Engineers                     │
│ • DevOps Engineers                 │
│ • Data Engineers                   │
│ • Product Managers                 │
└────────────────────────────────────┘
```

**2. Lack of Standards**
- No standard model format
- Inconsistent naming conventions
- Undocumented models
- No approval processes

**3. Limited Monitoring**
- No visibility into model performance
- Late detection of issues
- Unclear impact of problems

---

## 5. MLOps Maturity Model

### 5.1 Maturity Levels

```
Level 0: Manual Process
├─ Manual data analysis
├─ Notebook-based development
├─ Manual model deployment
└─ No monitoring

Level 1: DevOps, No MLOps
├─ Automated build & test
├─ Manual model training
├─ Manual deployment
└─ Basic logging

Level 2: Automated Training
├─ Experiment tracking
├─ Automated training pipeline
├─ Model registry
└─ Manual deployment

Level 3: Automated Deployment
├─ CI/CD for ML
├─ Automated model validation
├─ Automated deployment
└─ Basic monitoring

Level 4: Full MLOps Automation
├─ Automated everything
├─ Feature store
├─ A/B testing
├─ Advanced monitoring
└─ Automated retraining
```

### 5.2 Assessment Framework

```python
class MLOpsMaturityAssessment:
    """Assess MLOps maturity level"""

    def __init__(self):
        self.dimensions = {
            'data_management': [
                'Data versioning',
                'Data validation',
                'Feature store',
                'Data lineage'
            ],
            'model_development': [
                'Experiment tracking',
                'Hyperparameter tuning',
                'Model versioning',
                'Reproducibility'
            ],
            'deployment': [
                'Automated deployment',
                'Multiple environments',
                'Rollback capability',
                'A/B testing'
            ],
            'monitoring': [
                'Performance tracking',
                'Data drift detection',
                'Alert system',
                'Automated retraining'
            ],
            'governance': [
                'Model approval workflow',
                'Documentation',
                'Audit trail',
                'Compliance checks'
            ]
        }

    def assess(self, responses: dict) -> dict:
        """
        Assess maturity for each dimension

        Args:
            responses: Dict of {dimension: {capability: score}}

        Returns:
            Maturity report
        """
        results = {}

        for dimension, capabilities in self.dimensions.items():
            scores = [responses.get(dimension, {}).get(cap, 0)
                     for cap in capabilities]

            avg_score = sum(scores) / len(scores)

            results[dimension] = {
                'score': avg_score,
                'level': self._score_to_level(avg_score),
                'capabilities': dict(zip(capabilities, scores))
            }

        overall_score = sum(r['score'] for r in results.values()) / len(results)

        return {
            'overall_level': self._score_to_level(overall_score),
            'overall_score': overall_score,
            'dimensions': results,
            'recommendations': self._get_recommendations(results)
        }

    def _score_to_level(self, score: float) -> int:
        """Convert score (0-1) to maturity level (0-4)"""
        if score < 0.2:
            return 0
        elif score < 0.4:
            return 1
        elif score < 0.6:
            return 2
        elif score < 0.8:
            return 3
        else:
            return 4

    def _get_recommendations(self, results: dict) -> list:
        """Generate improvement recommendations"""
        recommendations = []

        for dimension, data in results.items():
            if data['score'] < 0.6:
                low_caps = [
                    cap for cap, score in data['capabilities'].items()
                    if score < 0.5
                ]
                recommendations.append({
                    'dimension': dimension,
                    'priority': 'high',
                    'improvements': low_caps
                })

        return recommendations

# Usage
assessment = MLOpsMaturityAssessment()

# Score each capability (0 = none, 1 = basic, 2 = intermediate, 3 = advanced)
responses = {
    'data_management': {
        'Data versioning': 2,
        'Data validation': 1,
        'Feature store': 0,
        'Data lineage': 1
    },
    'model_development': {
        'Experiment tracking': 2,
        'Hyperparameter tuning': 2,
        'Model versioning': 1,
        'Reproducibility': 1
    },
    # ... other dimensions
}

report = assessment.assess(responses)
print(f"Overall Maturity Level: {report['overall_level']}")
print(f"Recommendations: {report['recommendations']}")
```

---

## 6. MLOps Architecture

### 6.1 Reference Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    MLOps Platform                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐      ┌──────────────────┐         │
│  │  Feature Store   │      │ Experiment       │         │
│  │  - Online store  │      │ Tracking         │         │
│  │  - Offline store │      │ (MLflow)         │         │
│  └──────────────────┘      └──────────────────┘         │
│                                                            │
│  ┌──────────────────┐      ┌──────────────────┐         │
│  │  Model Registry  │      │ Training         │         │
│  │  - Versions      │      │ Pipeline         │         │
│  │  - Metadata      │      │ (Airflow)        │         │
│  └──────────────────┘      └──────────────────┘         │
│                                                            │
│  ┌──────────────────┐      ┌──────────────────┐         │
│  │  Model Serving   │      │ Monitoring       │         │
│  │  - REST API      │      │ - Metrics        │         │
│  │  - Batch         │      │ - Alerts         │         │
│  └──────────────────┘      └──────────────────┘         │
│                                                            │
└────────────────────────────────────────────────────────────┘
            ↓                           ↓
    ┌──────────────┐          ┌──────────────┐
    │ Data Sources │          │   Consumers  │
    └──────────────┘          └──────────────┘
```

### 6.2 Component Selection

```python
# Example: MLOps stack configuration

mlops_stack = {
    'experiment_tracking': {
        'tool': 'MLflow',
        'alternatives': ['W&B', 'Neptune', 'Comet'],
        'reason': 'Open source, mature, good integrations'
    },
    'feature_store': {
        'tool': 'Feast',
        'alternatives': ['Tecton', 'AWS Feature Store', 'Databricks'],
        'reason': 'Open source, flexible, cloud-agnostic'
    },
    'model_registry': {
        'tool': 'MLflow Model Registry',
        'alternatives': ['DVC', 'ModelDB', 'custom'],
        'reason': 'Integrated with experiment tracking'
    },
    'pipeline_orchestration': {
        'tool': 'Airflow',
        'alternatives': ['Kubeflow', 'Prefect', 'Dagster'],
        'reason': 'Mature, scalable, large ecosystem'
    },
    'model_serving': {
        'tool': 'FastAPI + Docker',
        'alternatives': ['TensorFlow Serving', 'Torchserve', 'Seldon'],
        'reason': 'Flexible, lightweight, easy to customize'
    },
    'monitoring': {
        'tool': 'Prometheus + Grafana',
        'alternatives': ['Datadog', 'New Relic', 'Evidently'],
        'reason': 'Industry standard, good for custom metrics'
    }
}
```

---

## 7. MLOps Tools Ecosystem

### 7.1 Tool Categories

```python
mlops_tools = {
    'Experiment Tracking': [
        'MLflow', 'Weights & Biases', 'Neptune.ai',
        'Comet ML', 'TensorBoard'
    ],
    'Data Versioning': [
        'DVC', 'Pachyderm', 'LakeFS', 'Delta Lake'
    ],
    'Feature Stores': [
        'Feast', 'Tecton', 'Hopsworks', 'AWS Feature Store'
    ],
    'Model Serving': [
        'TensorFlow Serving', 'TorchServe', 'Seldon Core',
        'BentoML', 'KServe'
    ],
    'Workflow Orchestration': [
        'Airflow', 'Kubeflow Pipelines', 'Metaflow',
        'Prefect', 'Dagster'
    ],
    'Model Monitoring': [
        'Evidently', 'Fiddler', 'Arize', 'WhyLabs',
        'NannyML'
    ],
    'ML Platforms': [
        'Databricks', 'SageMaker', 'Vertex AI',
        'Azure ML', 'Kubeflow'
    ]
}
```

---

## 8. Getting Started with MLOps

### 8.1 Implementation Roadmap

**Phase 1: Foundation (Weeks 1-4)**
- Set up experiment tracking
- Implement basic CI/CD
- Add logging and monitoring
- Document current processes

**Phase 2: Automation (Weeks 5-8)**
- Automate training pipeline
- Add model registry
- Implement automated testing
- Set up staging environment

**Phase 3: Production (Weeks 9-12)**
- Deploy to production
- Add monitoring and alerting
- Implement A/B testing
- Create runbooks

**Phase 4: Optimization (Ongoing)**
- Continuous improvement
- Advanced features (feature store, etc.)
- Team training
- Process refinement

### 8.2 Quick Start Checklist

```python
mlops_quickstart = {
    'immediate': [
        '□ Start tracking experiments (MLflow)',
        '□ Version control data (DVC)',
        '□ Document model cards',
        '□ Set up basic monitoring'
    ],
    'short_term': [
        '□ Implement CI/CD for models',
        '□ Create model registry',
        '□ Automate deployment',
        '□ Add data validation'
    ],
    'long_term': [
        '□ Build feature store',
        '□ Implement A/B testing',
        '□ Advanced monitoring',
        '□ Automated retraining'
    ]
}
```

---

## 9. Summary

Key takeaways:
- ✅ MLOps is essential for production ML success
- ✅ ML lifecycle is more complex than software lifecycle
- ✅ Maturity models help assess and improve practices
- ✅ Architecture should match organizational needs
- ✅ Start simple, iterate and improve
- ✅ Focus on automation and monitoring

**Next Lesson:** [02 - MLflow Experiment Tracking](./02-mlflow-experiment-tracking.md)
