# Lesson 08: MLOps Best Practices & Governance

## Overview
Running ML systems in production requires robust practices, clear governance, and strong team coordination. This lesson covers best practices, organizational patterns, and governance frameworks for successful MLOps.

**Duration:** 2-3 hours
**Prerequisites:** Understanding of MLOps lifecycle, production ML systems
**Learning Objectives:**
- Implement MLOps best practices
- Establish model governance frameworks
- Manage technical debt in ML systems
- Build effective ML teams and processes
- Ensure model fairness, security, and compliance

---

## 1. MLOps Maturity Model

### 1.1 Levels of MLOps Maturity

```
┌──────────────────────────────────────────────────────────┐
│                MLOps Maturity Levels                      │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  Level 0: Manual Process                                  │
│  ├─ Manual training and deployment                        │
│  ├─ No automation                                         │
│  ├─ Notebooks and scripts                                 │
│  └─ No monitoring                                         │
│                                                            │
│  Level 1: ML Pipeline Automation                          │
│  ├─ Automated training pipeline                           │
│  ├─ Experiment tracking                                   │
│  ├─ Model registry                                        │
│  └─ Basic monitoring                                      │
│                                                            │
│  Level 2: CI/CD Pipeline Automation                       │
│  ├─ Automated testing (code, data, models)                │
│  ├─ CI/CD for ML pipelines                               │
│  ├─ Feature store                                         │
│  ├─ Model versioning                                      │
│  └─ Deployment automation                                 │
│                                                            │
│  Level 3: Automated MLOps                                 │
│  ├─ Full CI/CD/CT (Continuous Training)                  │
│  ├─ Automated retraining triggers                         │
│  ├─ A/B testing and canary deployments                    │
│  ├─ Automated monitoring and alerting                     │
│  ├─ Model governance                                      │
│  └─ Feature drift detection and auto-remediation         │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Assessing Your MLOps Maturity

```python
# mlops_maturity_assessment.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class MaturityLevel(Enum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3

@dataclass
class MaturityCriteria:
    category: str
    criteria: str
    level: MaturityLevel
    implemented: bool

class MLOpsMaturityAssessment:
    """Assess MLOps maturity level"""

    def __init__(self):
        self.criteria = self._define_criteria()

    def _define_criteria(self) -> List[MaturityCriteria]:
        return [
            # Level 1 criteria
            MaturityCriteria("Training", "Automated training pipeline", MaturityLevel.LEVEL_1, False),
            MaturityCriteria("Tracking", "Experiment tracking (MLflow/W&B)", MaturityLevel.LEVEL_1, False),
            MaturityCriteria("Registry", "Model registry", MaturityLevel.LEVEL_1, False),
            MaturityCriteria("Monitoring", "Basic model monitoring", MaturityLevel.LEVEL_1, False),

            # Level 2 criteria
            MaturityCriteria("Testing", "Automated data validation", MaturityLevel.LEVEL_2, False),
            MaturityCriteria("Testing", "Automated model validation", MaturityLevel.LEVEL_2, False),
            MaturityCriteria("CI/CD", "CI/CD pipeline for ML", MaturityLevel.LEVEL_2, False),
            MaturityCriteria("Features", "Feature store", MaturityLevel.LEVEL_2, False),
            MaturityCriteria("Deployment", "Automated deployment", MaturityLevel.LEVEL_2, False),

            # Level 3 criteria
            MaturityCriteria("Training", "Continuous training", MaturityLevel.LEVEL_3, False),
            MaturityCriteria("Deployment", "Canary/A-B testing", MaturityLevel.LEVEL_3, False),
            MaturityCriteria("Monitoring", "Drift detection", MaturityLevel.LEVEL_3, False),
            MaturityCriteria("Monitoring", "Auto-retraining triggers", MaturityLevel.LEVEL_3, False),
            MaturityCriteria("Governance", "Model governance framework", MaturityLevel.LEVEL_3, False),
        ]

    def assess(self, implemented_criteria: List[str]) -> Dict:
        """
        Assess maturity based on implemented criteria

        Args:
            implemented_criteria: List of criteria names that are implemented

        Returns:
            Assessment results with maturity level
        """
        # Mark implemented criteria
        for criterion in self.criteria:
            if criterion.criteria in implemented_criteria:
                criterion.implemented = True

        # Calculate maturity level
        level_scores = {
            MaturityLevel.LEVEL_1: self._check_level(MaturityLevel.LEVEL_1),
            MaturityLevel.LEVEL_2: self._check_level(MaturityLevel.LEVEL_2),
            MaturityLevel.LEVEL_3: self._check_level(MaturityLevel.LEVEL_3),
        }

        # Determine overall level (must complete all criteria for a level)
        overall_level = MaturityLevel.LEVEL_0
        for level in [MaturityLevel.LEVEL_1, MaturityLevel.LEVEL_2, MaturityLevel.LEVEL_3]:
            if level_scores[level]['percentage'] >= 80:  # 80% threshold
                overall_level = level

        return {
            'overall_level': overall_level,
            'level_scores': level_scores,
            'recommendations': self._generate_recommendations(overall_level)
        }

    def _check_level(self, level: MaturityLevel) -> Dict:
        """Check completion for a specific level"""
        level_criteria = [c for c in self.criteria if c.level == level]
        implemented = [c for c in level_criteria if c.implemented]

        return {
            'total': len(level_criteria),
            'implemented': len(implemented),
            'percentage': len(implemented) / len(level_criteria) * 100 if level_criteria else 0,
            'missing': [c.criteria for c in level_criteria if not c.implemented]
        }

    def _generate_recommendations(self, current_level: MaturityLevel) -> List[str]:
        """Generate recommendations for next steps"""
        if current_level == MaturityLevel.LEVEL_0:
            return [
                "Implement experiment tracking (MLflow or Weights & Biases)",
                "Set up model registry for version management",
                "Create automated training pipeline",
                "Add basic model monitoring"
            ]
        elif current_level == MaturityLevel.LEVEL_1:
            return [
                "Implement automated data validation (Great Expectations)",
                "Add model validation tests",
                "Set up CI/CD pipeline for ML code",
                "Implement feature store (Feast)",
                "Automate model deployment"
            ]
        elif current_level == MaturityLevel.LEVEL_2:
            return [
                "Implement continuous training with automatic triggers",
                "Add canary/A-B testing for deployments",
                "Implement drift detection and monitoring",
                "Set up model governance framework",
                "Add automatic retraining based on performance degradation"
            ]
        else:
            return ["Maintain and optimize current MLOps practices"]

# Usage
assessment = MLOpsMaturityAssessment()

implemented = [
    "Automated training pipeline",
    "Experiment tracking (MLflow/W&B)",
    "Model registry",
    "Basic model monitoring",
    "Automated data validation"
]

results = assessment.assess(implemented)

print(f"Overall Maturity Level: {results['overall_level'].name}")
print(f"\nRecommendations:")
for rec in results['recommendations']:
    print(f"  - {rec}")
```

---

## 2. Model Governance Framework

### 2.1 Model Card Documentation

```python
# governance/model_card.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class ModelCard:
    """
    Standardized documentation for ML models
    Based on Google's Model Cards framework
    """

    # Model details
    model_name: str
    model_version: str
    model_type: str  # "classification", "regression", "ranking"
    model_architecture: str
    training_date: datetime
    owner_team: str
    contact_email: str

    # Intended use
    intended_use: str
    primary_use_cases: List[str]
    out_of_scope_uses: List[str]

    # Factors affecting model performance
    relevant_factors: List[str]  # demographics, geography, etc.
    evaluation_factors: Dict[str, any]

    # Metrics
    performance_metrics: Dict[str, float]
    decision_thresholds: Dict[str, float]

    # Training data
    training_data_description: str
    training_data_size: int
    training_data_date_range: tuple
    data_preprocessing: List[str]

    # Evaluation data
    evaluation_data_description: str
    evaluation_data_size: int

    # Ethical considerations
    sensitive_data: List[str]
    fairness_metrics: Dict[str, Dict[str, float]]
    potential_biases: List[str]

    # Limitations
    known_limitations: List[str]
    failure_modes: List[str]

    # Compliance
    regulatory_requirements: List[str]
    data_privacy_measures: List[str]

    def to_json(self, file_path: str):
        """Export model card to JSON"""
        with open(file_path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def to_html(self, file_path: str):
        """Generate HTML model card"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Card: {self.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                .metric {{ background: #ecf0f1; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Model Card: {self.model_name}</h1>
            <p><strong>Version:</strong> {self.model_version}</p>
            <p><strong>Owner:</strong> {self.owner_team} ({self.contact_email})</p>

            <h2>Model Details</h2>
            <p><strong>Type:</strong> {self.model_type}</p>
            <p><strong>Architecture:</strong> {self.model_architecture}</p>
            <p><strong>Training Date:</strong> {self.training_date}</p>

            <h2>Intended Use</h2>
            <p>{self.intended_use}</p>
            <h3>Primary Use Cases:</h3>
            <ul>
                {"".join(f"<li>{use_case}</li>" for use_case in self.primary_use_cases)}
            </ul>

            <h2>Performance Metrics</h2>
            {"".join(f'<div class="metric"><strong>{k}:</strong> {v:.4f}</div>' for k, v in self.performance_metrics.items())}

            <h2>Fairness Assessment</h2>
            {"".join(self._render_fairness_metrics())}

            <h2>Limitations</h2>
            <ul>
                {"".join(f"<li>{limitation}</li>" for limitation in self.known_limitations)}
            </ul>
        </body>
        </html>
        """

        with open(file_path, 'w') as f:
            f.write(html)

    def _render_fairness_metrics(self) -> List[str]:
        """Render fairness metrics as HTML"""
        html_parts = []
        for group, metrics in self.fairness_metrics.items():
            html_parts.append(f"<h3>{group}</h3>")
            for metric, value in metrics.items():
                html_parts.append(f'<div class="metric"><strong>{metric}:</strong> {value:.4f}</div>')
        return html_parts

# Example model card
recommendation_model_card = ModelCard(
    model_name="product_recommendation_model",
    model_version="v2.1.0",
    model_type="ranking",
    model_architecture="Two-tower neural network with attention",
    training_date=datetime(2024, 1, 15),
    owner_team="ML Platform Team",
    contact_email="ml-platform@example.com",

    intended_use="Recommend products to users based on browsing and purchase history",
    primary_use_cases=[
        "Homepage personalization",
        "Product detail page recommendations",
        "Email campaign targeting"
    ],
    out_of_scope_uses=[
        "Credit scoring",
        "Employment decisions",
        "Medical recommendations"
    ],

    relevant_factors=[
        "User purchase history",
        "Geographic location",
        "Device type",
        "Time of day"
    ],
    evaluation_factors={
        "demographics": ["age_group", "location"],
        "user_segments": ["new_users", "active_users", "churned_users"]
    },

    performance_metrics={
        "ndcg@10": 0.756,
        "map@10": 0.689,
        "click_through_rate": 0.182,
        "conversion_rate": 0.034
    },
    decision_thresholds={
        "min_confidence_score": 0.5,
        "max_recommendations": 10
    },

    training_data_description="User interaction data from e-commerce platform",
    training_data_size=50_000_000,
    training_data_date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    data_preprocessing=[
        "Remove bots and fraud accounts",
        "Filter items with <5 interactions",
        "Normalize item features",
        "Apply recency weighting"
    ],

    evaluation_data_description="Held-out data from December 2023",
    evaluation_data_size=5_000_000,

    sensitive_data=["user_id", "purchase_history", "location"],
    fairness_metrics={
        "age_groups": {
            "18-25": 0.175,
            "26-35": 0.182,
            "36-50": 0.179,
            "50+": 0.168
        },
        "geographic": {
            "urban": 0.185,
            "suburban": 0.180,
            "rural": 0.170
        }
    },
    potential_biases=[
        "May favor popular items over niche products",
        "Performance varies by geographic region",
        "New users have cold-start challenges"
    ],

    known_limitations=[
        "Requires minimum 5 interactions for personalization",
        "Performance degrades for very niche product categories",
        "Latency increases with catalog size >1M items"
    ],
    failure_modes=[
        "Returns generic recommendations for cold-start users",
        "May not adapt quickly to sudden preference changes",
        "Sensitive to seasonal trends"
    ],

    regulatory_requirements=[
        "GDPR compliance for EU users",
        "CCPA compliance for California users",
        "Right to explanation for automated decisions"
    ],
    data_privacy_measures=[
        "PII encrypted at rest",
        "User data anonymized after 90 days of inactivity",
        "User can request data deletion"
    ]
)

# Export model card
recommendation_model_card.to_json("model_cards/recommendation_model_v2.1.0.json")
recommendation_model_card.to_html("model_cards/recommendation_model_v2.1.0.html")
```

### 2.2 Model Approval Workflow

```python
# governance/approval_workflow.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

class ApprovalStage(Enum):
    DATA_REVIEW = "data_review"
    MODEL_REVIEW = "model_review"
    FAIRNESS_REVIEW = "fairness_review"
    SECURITY_REVIEW = "security_review"
    COMPLIANCE_REVIEW = "compliance_review"
    FINAL_APPROVAL = "final_approval"

@dataclass
class ApprovalRequest:
    """Request for model approval"""
    model_name: str
    model_version: str
    requested_by: str
    requested_at: datetime
    model_card: ModelCard
    stage: ApprovalStage
    status: ApprovalStatus
    reviewer: Optional[str] = None
    comments: List[str] = None
    reviewed_at: Optional[datetime] = None

class ModelApprovalWorkflow:
    """Manage model approval process"""

    def __init__(self):
        self.approval_stages = [
            ApprovalStage.DATA_REVIEW,
            ApprovalStage.MODEL_REVIEW,
            ApprovalStage.FAIRNESS_REVIEW,
            ApprovalStage.SECURITY_REVIEW,
            ApprovalStage.COMPLIANCE_REVIEW,
            ApprovalStage.FINAL_APPROVAL
        ]

    def initiate_approval(
        self,
        model_name: str,
        model_version: str,
        requested_by: str,
        model_card: ModelCard
    ) -> ApprovalRequest:
        """Start approval process"""

        request = ApprovalRequest(
            model_name=model_name,
            model_version=model_version,
            requested_by=requested_by,
            requested_at=datetime.now(),
            model_card=model_card,
            stage=ApprovalStage.DATA_REVIEW,
            status=ApprovalStatus.PENDING,
            comments=[]
        )

        # Notify data review team
        self._notify_reviewers(request)

        return request

    def review_stage(
        self,
        request: ApprovalRequest,
        reviewer: str,
        approved: bool,
        comments: str
    ) -> ApprovalRequest:
        """Review current stage"""

        request.reviewer = reviewer
        request.reviewed_at = datetime.now()
        request.comments.append(f"[{request.stage.value}] {reviewer}: {comments}")

        if approved:
            # Move to next stage
            current_index = self.approval_stages.index(request.stage)

            if current_index < len(self.approval_stages) - 1:
                request.stage = self.approval_stages[current_index + 1]
                request.status = ApprovalStatus.PENDING
                self._notify_reviewers(request)
            else:
                # Final approval
                request.status = ApprovalStatus.APPROVED
                self._promote_to_production(request)
        else:
            request.status = ApprovalStatus.NEEDS_REVISION
            self._notify_requester(request)

        return request

    def _notify_reviewers(self, request: ApprovalRequest):
        """Notify relevant reviewers"""
        reviewer_emails = {
            ApprovalStage.DATA_REVIEW: "data-team@example.com",
            ApprovalStage.MODEL_REVIEW: "ml-team@example.com",
            ApprovalStage.FAIRNESS_REVIEW: "ethics-team@example.com",
            ApprovalStage.SECURITY_REVIEW: "security-team@example.com",
            ApprovalStage.COMPLIANCE_REVIEW: "compliance-team@example.com",
            ApprovalStage.FINAL_APPROVAL: "ml-lead@example.com"
        }

        email = reviewer_emails.get(request.stage)
        # Send email notification
        print(f"📧 Notifying {email} for {request.stage.value}")

    def _promote_to_production(self, request: ApprovalRequest):
        """Promote approved model to production"""
        # Update model registry
        # Trigger deployment pipeline
        print(f"✅ Model {request.model_name} v{request.model_version} approved for production")

# Usage
workflow = ModelApprovalWorkflow()

# Initiate approval
approval_request = workflow.initiate_approval(
    model_name="recommendation_model",
    model_version="v2.1.0",
    requested_by="ml-engineer@example.com",
    model_card=recommendation_model_card
)

# Data team reviews
approval_request = workflow.review_stage(
    request=approval_request,
    reviewer="data-scientist@example.com",
    approved=True,
    comments="Data quality checks passed. Training data is representative."
)

# Continue through stages...
```

---

## 3. Technical Debt Management

### 3.1 ML-Specific Technical Debt

```python
# debt/ml_technical_debt.py
from enum import Enum
from dataclasses import dataclass
from typing import List

class DebtCategory(Enum):
    DATA_DEBT = "data_debt"
    MODEL_DEBT = "model_debt"
    INFRASTRUCTURE_DEBT = "infrastructure_debt"
    MONITORING_DEBT = "monitoring_debt"
    DOCUMENTATION_DEBT = "documentation_debt"

@dataclass
class TechnicalDebt:
    """ML-specific technical debt item"""
    category: DebtCategory
    description: str
    impact: str  # "high", "medium", "low"
    effort: str  # "high", "medium", "low"
    created_at: str
    owner: str

class MLTechnicalDebtTracker:
    """Track and prioritize ML technical debt"""

    def __init__(self):
        self.debt_items: List[TechnicalDebt] = []

    def identify_common_debt(self) -> List[TechnicalDebt]:
        """
        Identify common ML technical debt patterns
        """
        return [
            # Data debt
            TechnicalDebt(
                category=DebtCategory.DATA_DEBT,
                description="Data pipeline depends on multiple legacy systems",
                impact="high",
                effort="high",
                created_at="2024-01-15",
                owner="data-team"
            ),
            TechnicalDebt(
                category=DebtCategory.DATA_DEBT,
                description="No automated data validation in pipeline",
                impact="high",
                effort="medium",
                created_at="2024-01-20",
                owner="ml-team"
            ),
            TechnicalDebt(
                category=DebtCategory.DATA_DEBT,
                description="Training data not versioned",
                impact="medium",
                effort="low",
                created_at="2024-02-01",
                owner="ml-team"
            ),

            # Model debt
            TechnicalDebt(
                category=DebtCategory.MODEL_DEBT,
                description="Multiple models doing similar tasks (duplication)",
                impact="medium",
                effort="high",
                created_at="2024-01-10",
                owner="ml-team"
            ),
            TechnicalDebt(
                category=DebtCategory.MODEL_DEBT,
                description="Complex ensemble of models difficult to maintain",
                impact="high",
                effort="high",
                created_at="2024-01-25",
                owner="ml-team"
            ),

            # Infrastructure debt
            TechnicalDebt(
                category=DebtCategory.INFRASTRUCTURE_DEBT,
                description="Manual model deployment process",
                impact="high",
                effort="medium",
                created_at="2024-01-05",
                owner="mlops-team"
            ),

            # Monitoring debt
            TechnicalDebt(
                category=DebtCategory.MONITORING_DEBT,
                description="No drift detection in production",
                impact="high",
                effort="medium",
                created_at="2024-02-10",
                owner="mlops-team"
            ),

            # Documentation debt
            TechnicalDebt(
                category=DebtCategory.DOCUMENTATION_DEBT,
                description="Model cards not updated for recent models",
                impact="medium",
                effort="low",
                created_at="2024-02-15",
                owner="ml-team"
            ),
        ]

    def prioritize_debt(self, debt_items: List[TechnicalDebt]) -> List[TechnicalDebt]:
        """
        Prioritize debt items based on impact and effort
        """
        priority_scores = {
            ('high', 'low'): 10,      # High impact, low effort - DO FIRST
            ('high', 'medium'): 8,
            ('medium', 'low'): 7,
            ('high', 'high'): 6,
            ('medium', 'medium'): 5,
            ('low', 'low'): 4,
            ('medium', 'high'): 3,
            ('low', 'medium'): 2,
            ('low', 'high'): 1,
        }

        def get_priority(item: TechnicalDebt) -> int:
            return priority_scores.get((item.impact, item.effort), 0)

        return sorted(debt_items, key=get_priority, reverse=True)

# Usage
tracker = MLTechnicalDebtTracker()
debt_items = tracker.identify_common_debt()
prioritized = tracker.prioritize_debt(debt_items)

print("Technical Debt Priority List:")
for i, item in enumerate(prioritized, 1):
    print(f"{i}. [{item.category.value}] {item.description}")
    print(f"   Impact: {item.impact}, Effort: {item.effort}, Owner: {item.owner}\n")
```

---

## 4. Team Organization and Processes

### 4.1 ML Team Structures

```
Option 1: Embedded ML Engineers
┌─────────────────────────────────────────┐
│          Product Team                    │
│  ├─ Product Manager                      │
│  ├─ Frontend Engineers                   │
│  ├─ Backend Engineers                    │
│  ├─ ML Engineer (Embedded)               │
│  └─ Data Engineer (Embedded)             │
└─────────────────────────────────────────┘

Pros:
- Close to product
- Fast iteration
- Domain expertise

Cons:
- Siloed ML practices
- Inconsistent tooling
- Difficult to share learnings

Option 2: Centralized ML Platform Team
┌─────────────────────────────────────────┐
│       ML Platform Team (Central)         │
│  ├─ ML Platform Engineers                │
│  ├─ MLOps Engineers                      │
│  └─ ML Infrastructure Engineers          │
└─────────────────────────────────────────┘
           │
           ├──────────┬──────────┬──────────
           │          │          │
    ┌──────▼────┐ ┌──▼──────┐ ┌─▼────────┐
    │ Product   │ │Product  │ │Product   │
    │  Team A   │ │ Team B  │ │ Team C   │
    └───────────┘ └─────────┘ └──────────┘

Pros:
- Consistent practices
- Shared tooling
- Efficient resource use

Cons:
- Distance from product
- Potential bottleneck
- Less domain expertise

Option 3: Hybrid (Recommended)
┌─────────────────────────────────────────┐
│       ML Platform Team (Central)         │
│  ├─ ML Platform Engineers                │
│  ├─ MLOps Engineers                      │
│  └─ Provides: Infrastructure, tools,     │
│              best practices               │
└─────────────────────────────────────────┘
           │
           ├──────────┬──────────┬──────────
           │          │          │
    ┌──────▼────┐ ┌──▼──────┐ ┌─▼────────┐
    │ Product   │ │Product  │ │Product   │
    │  Team A   │ │ Team B  │ │ Team C   │
    │  + Embedded│ │+ Embedded│ │+ Embedded│
    │  ML Eng   │ │ ML Eng  │ │ ML Eng   │
    └───────────┘ └─────────┘ └──────────┘

Pros:
- Best of both worlds
- Centralized infrastructure
- Embedded product knowledge

Cons:
- Requires good coordination
- More complex structure
```

### 4.2 ML Development Workflow

```python
# process/ml_workflow.py
from enum import Enum
from dataclasses import dataclass
from typing import List

class WorkflowStage(Enum):
    PROBLEM_DEFINITION = "problem_definition"
    DATA_EXPLORATION = "data_exploration"
    BASELINE_MODEL = "baseline_model"
    MODEL_DEVELOPMENT = "model_development"
    EVALUATION = "evaluation"
    DEPLOYMENT_PREP = "deployment_prep"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

@dataclass
class StageChecklist:
    """Checklist for each workflow stage"""
    stage: WorkflowStage
    required_artifacts: List[str]
    required_approvals: List[str]
    exit_criteria: List[str]

ml_workflow_stages = [
    StageChecklist(
        stage=WorkflowStage.PROBLEM_DEFINITION,
        required_artifacts=[
            "Problem statement document",
            "Success metrics defined",
            "Business impact assessment"
        ],
        required_approvals=["Product Manager", "ML Lead"],
        exit_criteria=[
            "Clear ML formulation",
            "Success metrics agreed",
            "Resources allocated"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.DATA_EXPLORATION,
        required_artifacts=[
            "Data quality report",
            "EDA notebook",
            "Feature feasibility analysis"
        ],
        required_approvals=["Data Science Lead"],
        exit_criteria=[
            "Data quality acceptable",
            "Sufficient data volume",
            "Features identified"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.BASELINE_MODEL,
        required_artifacts=[
            "Baseline model notebook",
            "Evaluation metrics",
            "Comparison to heuristics"
        ],
        required_approvals=["ML Engineer"],
        exit_criteria=[
            "Baseline beats heuristics",
            "Sufficient improvement potential"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.MODEL_DEVELOPMENT,
        required_artifacts=[
            "Experiment tracking",
            "Model comparison analysis",
            "Feature importance analysis"
        ],
        required_approvals=["ML Lead"],
        exit_criteria=[
            "Model meets performance thresholds",
            "Offline evaluation complete"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.EVALUATION,
        required_artifacts=[
            "Comprehensive evaluation report",
            "Fairness assessment",
            "Error analysis"
        ],
        required_approvals=["ML Lead", "Product Manager"],
        exit_criteria=[
            "Performance acceptable",
            "No fairness issues",
            "Error modes understood"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.DEPLOYMENT_PREP,
        required_artifacts=[
            "Model card",
            "Deployment plan",
            "Monitoring plan",
            "Rollback plan"
        ],
        required_approvals=["MLOps Lead", "Security Review"],
        exit_criteria=[
            "All documentation complete",
            "Infrastructure ready",
            "Monitoring configured"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.DEPLOYMENT,
        required_artifacts=[
            "A/B test plan",
            "Success criteria",
            "Communication plan"
        ],
        required_approvals=["ML Lead", "Product Manager"],
        exit_criteria=[
            "Deployment successful",
            "A/B test running",
            "No critical issues"
        ]
    ),

    StageChecklist(
        stage=WorkflowStage.MONITORING,
        required_artifacts=[
            "Weekly performance reports",
            "Incident logs",
            "Improvement backlog"
        ],
        required_approvals=["Ongoing"],
        exit_criteria=[
            "Performance stable",
            "No drift detected",
            "Incidents handled"
        ]
    ),
]
```

---

## 5. Best Practices Checklist

### 5.1 Development Best Practices

```markdown
## ML Development Best Practices

### Data Management
- [ ] Data versioning implemented (DVC, lakeFS)
- [ ] Data quality validation automated
- [ ] Training/test split documented and reproducible
- [ ] Data lineage tracked
- [ ] PII handling documented and compliant

### Experimentation
- [ ] All experiments logged (MLflow, W&B)
- [ ] Reproducible random seeds
- [ ] Hyperparameters tracked
- [ ] Environment captured (requirements.txt, Docker)
- [ ] Git commit hash recorded

### Code Quality
- [ ] Unit tests for data processing
- [ ] Unit tests for model code
- [ ] Integration tests for pipelines
- [ ] Code reviews required
- [ ] Linting and formatting enforced

### Model Development
- [ ] Baseline model established
- [ ] Multiple models compared
- [ ] Cross-validation performed
- [ ] Feature importance analyzed
- [ ] Error analysis completed

### Deployment
- [ ] Model card created
- [ ] Performance thresholds defined
- [ ] Monitoring configured
- [ ] Alerting set up
- [ ] Rollback plan documented

### Monitoring
- [ ] Model performance tracked
- [ ] Data drift detection active
- [ ] Prediction distribution monitored
- [ ] Latency/throughput monitored
- [ ] Business metrics tracked
```

---

## 6. Security and Compliance

### 6.1 Model Security

```python
# security/model_security.py
from typing import Dict, List
import hashlib

class ModelSecurityChecker:
    """Check model security posture"""

    def __init__(self):
        self.checks = [
            self.check_model_provenance,
            self.check_dependency_vulnerabilities,
            self.check_input_validation,
            self.check_access_controls,
            self.check_encryption,
        ]

    def audit_model(self, model_info: Dict) -> Dict:
        """Run security audit on model"""

        results = {}
        for check in self.checks:
            check_name = check.__name__
            try:
                passed, details = check(model_info)
                results[check_name] = {
                    'passed': passed,
                    'details': details
                }
            except Exception as e:
                results[check_name] = {
                    'passed': False,
                    'details': f"Check failed: {str(e)}"
                }

        overall_passed = all(r['passed'] for r in results.values())

        return {
            'overall_passed': overall_passed,
            'checks': results
        }

    def check_model_provenance(self, model_info: Dict) -> tuple:
        """Verify model provenance"""
        required_fields = ['git_commit', 'training_data_version', 'trainer']

        missing = [f for f in required_fields if f not in model_info]

        if missing:
            return False, f"Missing provenance fields: {missing}"

        return True, "Model provenance verified"

    def check_dependency_vulnerabilities(self, model_info: Dict) -> tuple:
        """Check for vulnerable dependencies"""
        # Integrate with tools like Safety, Snyk
        # For demo, simplified check
        vulnerable_packages = []  # Would come from vulnerability scanner

        if vulnerable_packages:
            return False, f"Vulnerable dependencies: {vulnerable_packages}"

        return True, "No known vulnerabilities"

    def check_input_validation(self, model_info: Dict) -> tuple:
        """Verify input validation exists"""
        has_schema = 'input_schema' in model_info
        has_validation = 'input_validation' in model_info

        if not (has_schema and has_validation):
            return False, "Missing input validation"

        return True, "Input validation configured"

    def check_access_controls(self, model_info: Dict) -> tuple:
        """Check access controls"""
        has_auth = model_info.get('authentication', False)
        has_authz = model_info.get('authorization', False)

        if not (has_auth and has_authz):
            return False, "Access controls not configured"

        return True, "Access controls configured"

    def check_encryption(self, model_info: Dict) -> tuple:
        """Check encryption"""
        model_encrypted = model_info.get('model_encrypted', False)
        data_encrypted = model_info.get('data_encrypted_in_transit', False)

        if not (model_encrypted and data_encrypted):
            return False, "Encryption not configured"

        return True, "Encryption configured"

# Usage
security_checker = ModelSecurityChecker()

model_info = {
    'git_commit': 'abc123',
    'training_data_version': 'v2.1',
    'trainer': 'ml-engineer@example.com',
    'input_schema': {...},
    'input_validation': True,
    'authentication': True,
    'authorization': True,
    'model_encrypted': True,
    'data_encrypted_in_transit': True
}

audit_results = security_checker.audit_model(model_info)

if audit_results['overall_passed']:
    print("✅ Security audit passed")
else:
    print("❌ Security audit failed")
    for check, result in audit_results['checks'].items():
        if not result['passed']:
            print(f"  - {check}: {result['details']}")
```

---

## Summary

In this lesson, you learned:

✅ **MLOps Maturity:**
- Assessing current maturity level
- Roadmap for improvement
- Best practices at each level

✅ **Governance:**
- Model cards and documentation
- Approval workflows
- Compliance requirements

✅ **Technical Debt:**
- Identifying ML-specific debt
- Prioritization frameworks
- Mitigation strategies

✅ **Team & Process:**
- Team structures
- Development workflows
- Best practices checklists

✅ **Security:**
- Model security audits
- Access controls
- Encryption requirements

---

## Additional Resources

- [MLOps: Continuous delivery and automation pipelines](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

---

## Module Complete!

Congratulations! You've completed the MLOps & Experiment Tracking module. You're now ready to build, deploy, and maintain production ML systems with best practices.

**Next Steps:**
- Review the module quiz
- Complete the hands-on exercises
- Explore additional resources
- Move to Module 07: GPU Computing & Distributed Training
