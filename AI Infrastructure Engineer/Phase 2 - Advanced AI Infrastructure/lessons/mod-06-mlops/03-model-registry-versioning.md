# Lesson 03: Model Registry & Versioning

## Overview
A Model Registry is a centralized repository for storing, versioning, and managing ML models throughout their lifecycle. This lesson covers model versioning strategies, MLflow Model Registry, and governance workflows for production ML systems.

**Duration:** 5-6 hours
**Difficulty:** Intermediate
**Prerequisites:** MLflow Tracking basics, understanding of software versioning

## Learning Objectives
- Understand model versioning concepts and strategies
- Implement MLflow Model Registry
- Manage model lifecycle stages
- Create approval workflows
- Track model lineage and metadata
- Integrate registry with deployment systems

---

## 1. Model Versioning Fundamentals

### 1.1 Why Version Models?

```python
# Problem: Without versioning
model = joblib.load('fraud_model.pkl')  # Which version? When trained?

# Solution: With versioning
model = mlflow.pyfunc.load_model('models:/fraud_model/3')  # Version 3
```

**Key reasons:**
- Reproducibility
- Rollback capability
- A/B testing
- Audit trail
- Collaborative development

### 1.2 Versioning Strategies

```
Semantic Versioning (MAJOR.MINOR.PATCH)
├─ MAJOR: Breaking changes (new algorithm, features)
├─ MINOR: New features (backward compatible)
└─ PATCH: Bug fixes, retraining

Timestamp Versioning (YYYYMMDD-HHMMSS)
└─ Simple, chronological

Git-style Hashing
└─ Content-addressable, unique identifiers
```

---

## 2. MLflow Model Registry

### 2.1 Architecture

```
┌──────────────────────────────────────────────┐
│          MLflow Model Registry               │
├──────────────────────────────────────────────┤
│                                              │
│  Model: fraud_detection                      │
│  ├─ Version 1 (Archived)                    │
│  ├─ Version 2 (Staging)                     │
│  ├─ Version 3 (Production) ◄── Active       │
│  └─ Version 4 (None)                        │
│                                              │
│  Metadata:                                   │
│  • Description                               │
│  • Tags                                      │
│  • Metrics                                   │
│  • Source run                                │
└──────────────────────────────────────────────┘
```

### 2.2 Model Lifecycle Stages

```python
from mlflow.tracking import MlflowClient

# Stages in MLflow Model Registry
stages = {
    'None': 'Initial registration, not yet validated',
    'Staging': 'Under validation, ready for testing',
    'Production': 'Deployed to production',
    'Archived': 'Deprecated, no longer in use'
}
```

---

## 3. Registering Models

### 3.1 Basic Registration

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Train and log model
with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="fraud_detection"
    )

    # Log metrics
    mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93})

print(f"Model logged to run: {run.info.run_id}")
```

### 3.2 Programmatic Registration

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from existing run
model_uri = f"runs:/{run_id}/model"
model_details = client.create_registered_model("fraud_detection")

# Create model version
model_version = client.create_model_version(
    name="fraud_detection",
    source=model_uri,
    run_id=run_id,
    description="Random Forest with 100 trees"
)

print(f"Model version: {model_version.version}")
```

### 3.3 Model with Metadata

```python
def register_model_with_metadata(
    model_name: str,
    run_id: str,
    description: str,
    tags: dict,
    metrics: dict
):
    """Register model with comprehensive metadata"""
    client = MlflowClient()

    # Create model version
    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=description
    )

    # Add tags
    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key=key,
            value=value
        )

    # Add performance metrics as tags
    for metric, value in metrics.items():
        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key=f"metric_{metric}",
            value=str(value)
        )

    return mv

# Usage
mv = register_model_with_metadata(
    model_name="fraud_detection",
    run_id=run.info.run_id,
    description="Production fraud detection model",
    tags={
        "team": "risk-ml",
        "framework": "sklearn",
        "algorithm": "random_forest"
    },
    metrics={
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96
    }
)
```

---

## 4. Managing Model Versions

### 4.1 Transitioning Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to Staging
client.transition_model_version_stage(
    name="fraud_detection",
    version=3,
    stage="Staging",
    archive_existing_versions=False
)

# Transition to Production
client.transition_model_version_stage(
    name="fraud_detection",
    version=3,
    stage="Production",
    archive_existing_versions=True  # Archive previous production versions
)

# Archive old version
client.transition_model_version_stage(
    name="fraud_detection",
    version=1,
    stage="Archived"
)
```

### 4.2 Querying Versions

```python
# Get all versions of a model
versions = client.search_model_versions(f"name='fraud_detection'")

for mv in versions:
    print(f"Version {mv.version}: {mv.current_stage}")
    print(f"  Run ID: {mv.run_id}")
    print(f"  Status: {mv.status}")

# Get latest version by stage
latest_production = client.get_latest_versions(
    "fraud_detection",
    stages=["Production"]
)[0]

latest_staging = client.get_latest_versions(
    "fraud_detection",
    stages=["Staging"]
)[0]

print(f"Production version: {latest_production.version}")
print(f"Staging version: {latest_staging.version}")
```

### 4.3 Loading Models by Version

```python
import mlflow.pyfunc

# Load specific version
model_v3 = mlflow.pyfunc.load_model("models:/fraud_detection/3")

# Load latest by stage
production_model = mlflow.pyfunc.load_model(
    "models:/fraud_detection/Production"
)

staging_model = mlflow.pyfunc.load_model(
    "models:/fraud_detection/Staging"
)

# Make predictions
predictions = production_model.predict(X_test)
```

---

## 5. Model Approval Workflows

### 5.1 Automated Validation

```python
class ModelValidator:
    """Validate models before promotion"""

    def __init__(self, client: MlflowClient):
        self.client = client

    def validate_metrics(self, model_version, thresholds: dict) -> bool:
        """Check if model meets metric thresholds"""
        run = self.client.get_run(model_version.run_id)
        metrics = run.data.metrics

        for metric, threshold in thresholds.items():
            if metric not in metrics:
                return False
            if metrics[metric] < threshold:
                return False

        return True

    def validate_against_champion(
        self,
        challenger_version,
        champion_version,
        test_data
    ) -> bool:
        """Compare challenger against current champion"""
        challenger_model = mlflow.pyfunc.load_model(
            f"models:/{challenger_version.name}/{challenger_version.version}"
        )
        champion_model = mlflow.pyfunc.load_model(
            f"models:/{champion_version.name}/{champion_version.version}"
        )

        challenger_acc = evaluate_model(challenger_model, test_data)
        champion_acc = evaluate_model(champion_model, test_data)

        return challenger_acc > champion_acc

    def approve_for_staging(
        self,
        model_name: str,
        version: int,
        thresholds: dict
    ) -> bool:
        """Approve model for staging if it passes validation"""
        mv = self.client.get_model_version(model_name, version)

        if self.validate_metrics(mv, thresholds):
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )

            # Add approval metadata
            self.client.set_model_version_tag(
                model_name,
                version,
                "validated",
                "true"
            )

            return True

        return False

# Usage
validator = ModelValidator(client)

approved = validator.approve_for_staging(
    model_name="fraud_detection",
    version=4,
    thresholds={"accuracy": 0.9, "f1_score": 0.85}
)

if approved:
    print("Model approved for staging")
```

### 5.2 Manual Approval Workflow

```python
class ApprovalWorkflow:
    """Manage manual approval process"""

    def __init__(self, client: MlflowClient):
        self.client = client

    def request_approval(
        self,
        model_name: str,
        version: int,
        approver: str,
        reason: str
    ):
        """Create approval request"""
        self.client.set_model_version_tag(
            model_name,
            version,
            "approval_status",
            "pending"
        )
        self.client.set_model_version_tag(
            model_name,
            version,
            "approver",
            approver
        )
        self.client.set_model_version_tag(
            model_name,
            version,
            "approval_reason",
            reason
        )

        # Send notification
        self.notify_approver(approver, model_name, version)

    def approve(
        self,
        model_name: str,
        version: int,
        approver: str
    ):
        """Approve model for production"""
        self.client.set_model_version_tag(
            model_name,
            version,
            "approval_status",
            "approved"
        )
        self.client.set_model_version_tag(
            model_name,
            version,
            "approved_by",
            approver
        )
        self.client.set_model_version_tag(
            model_name,
            version,
            "approved_at",
            datetime.now().isoformat()
        )

        # Transition to production
        self.client.transition_model_version_stage(
            model_name,
            version,
            "Production",
            archive_existing_versions=True
        )

    def reject(
        self,
        model_name: str,
        version: int,
        reason: str
    ):
        """Reject model"""
        self.client.set_model_version_tag(
            model_name,
            version,
            "approval_status",
            "rejected"
        )
        self.client.set_model_version_tag(
            model_name,
            version,
            "rejection_reason",
            reason
        )

# Usage
workflow = ApprovalWorkflow(client)

# Request approval
workflow.request_approval(
    model_name="fraud_detection",
    version=4,
    approver="ml-lead@company.com",
    reason="Improved accuracy by 2%"
)

# Approve
workflow.approve(
    model_name="fraud_detection",
    version=4,
    approver="ml-lead@company.com"
)
```

---

## 6. Model Lineage & Metadata

### 6.1 Tracking Model Lineage

```python
def log_model_lineage(
    model_name: str,
    version: int,
    training_data_version: str,
    parent_model_version: int = None,
    dependencies: dict = None
):
    """Log complete model lineage"""
    client = MlflowClient()

    # Training data lineage
    client.set_model_version_tag(
        model_name,
        version,
        "training_data_version",
        training_data_version
    )

    # Parent model (for retraining)
    if parent_model_version:
        client.set_model_version_tag(
            model_name,
            version,
            "parent_model_version",
            str(parent_model_version)
        )

    # Dependencies
    if dependencies:
        for dep, ver in dependencies.items():
            client.set_model_version_tag(
                model_name,
                version,
                f"dependency_{dep}",
                ver
            )

    # Code version
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()

        client.set_model_version_tag(
            model_name,
            version,
            "git_commit",
            git_commit
        )
    except:
        pass

# Usage
log_model_lineage(
    model_name="fraud_detection",
    version=4,
    training_data_version="2024-01-15",
    parent_model_version=3,
    dependencies={
        "sklearn": "1.3.0",
        "pandas": "2.0.0"
    }
)
```

### 6.2 Model Cards

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelCard:
    """Structured model documentation"""
    model_name: str
    version: int
    description: str
    intended_use: str
    training_data: Dict
    performance_metrics: Dict
    ethical_considerations: List[str]
    limitations: List[str]

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'version': self.version,
            'description': self.description,
            'intended_use': self.intended_use,
            'training_data': self.training_data,
            'performance_metrics': self.performance_metrics,
            'ethical_considerations': self.ethical_considerations,
            'limitations': self.limitations
        }

    def save_to_registry(self, client: MlflowClient):
        """Save model card to registry"""
        import json

        card_json = json.dumps(self.to_dict(), indent=2)

        client.set_model_version_tag(
            self.model_name,
            self.version,
            "model_card",
            card_json
        )

# Usage
card = ModelCard(
    model_name="fraud_detection",
    version=4,
    description="Random Forest classifier for credit card fraud detection",
    intended_use="Real-time fraud scoring for transaction approvals",
    training_data={
        "source": "transactions_2023",
        "size": "1M records",
        "date_range": "2023-01-01 to 2023-12-31"
    },
    performance_metrics={
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95
    },
    ethical_considerations=[
        "Potential bias in historical fraud patterns",
        "False positives may impact customer experience"
    ],
    limitations=[
        "Performance degrades on new fraud patterns",
        "Requires retraining quarterly"
    ]
)

card.save_to_registry(client)
```

---

## 7. Integration with Deployment

### 7.1 Deployment Automation

```python
class ModelDeploymentManager:
    """Manage model deployments from registry"""

    def __init__(self, client: MlflowClient):
        self.client = client

    def deploy_production_model(
        self,
        model_name: str,
        deployment_target: str
    ):
        """Deploy current production model"""
        # Get production version
        versions = self.client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not versions:
            raise ValueError(f"No production version for {model_name}")

        prod_version = versions[0]

        # Load model
        model_uri = f"models:/{model_name}/Production"

        # Deploy based on target
        if deployment_target == "kubernetes":
            self.deploy_to_k8s(model_uri, prod_version)
        elif deployment_target == "sagemaker":
            self.deploy_to_sagemaker(model_uri, prod_version)
        elif deployment_target == "lambda":
            self.deploy_to_lambda(model_uri, prod_version)

        # Record deployment
        self.client.set_model_version_tag(
            model_name,
            prod_version.version,
            "deployed_to",
            deployment_target
        )
        self.client.set_model_version_tag(
            model_name,
            prod_version.version,
            "deployed_at",
            datetime.now().isoformat()
        )

    def rollback_deployment(
        self,
        model_name: str,
        previous_version: int
    ):
        """Rollback to previous version"""
        self.client.transition_model_version_stage(
            model_name,
            previous_version,
            "Production",
            archive_existing_versions=True
        )
```

---

## 8. Best Practices

### 8.1 Model Registry Checklist

✅ **DO:**
- Use descriptive model names
- Document models thoroughly
- Version all dependencies
- Implement approval workflows
- Track model lineage
- Clean up archived models
- Use semantic versioning

❌ **DON'T:**
- Skip validation before promotion
- Bypass approval workflows
- Leave models in staging indefinitely
- Delete production models
- Mix unrelated models in same registry

### 8.2 Production Example

```python
# Complete production workflow
def production_model_workflow():
    """End-to-end model registry workflow"""
    client = MlflowClient()

    # 1. Train model
    with mlflow.start_run() as run:
        model = train_model()
        metrics = evaluate_model(model)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(metrics)

    # 2. Register model
    mv = register_model_with_metadata(
        model_name="fraud_detection",
        run_id=run.info.run_id,
        description="Quarterly retrained model",
        tags={"quarter": "Q1-2024"},
        metrics=metrics
    )

    # 3. Validate
    validator = ModelValidator(client)
    if validator.approve_for_staging(
        "fraud_detection",
        mv.version,
        {"accuracy": 0.9, "f1_score": 0.85}
    ):
        print(f"Version {mv.version} moved to Staging")

    # 4. Request approval
    workflow = ApprovalWorkflow(client)
    workflow.request_approval(
        "fraud_detection",
        mv.version,
        "ml-lead@company.com",
        "Quarterly retrained model with improved metrics"
    )

    # 5. Deploy (after approval)
    deployer = ModelDeploymentManager(client)
    deployer.deploy_production_model(
        "fraud_detection",
        "kubernetes"
    )
```

---

## 9. Summary

Key takeaways:
- ✅ Model Registry provides centralized model management
- ✅ Lifecycle stages organize model progression
- ✅ Automated validation ensures quality
- ✅ Approval workflows provide governance
- ✅ Model lineage enables reproducibility
- ✅ Integration with deployment automates releases

**Next Lesson:** [04 - Feature Stores Engineering](./04-feature-stores-engineering.md)
