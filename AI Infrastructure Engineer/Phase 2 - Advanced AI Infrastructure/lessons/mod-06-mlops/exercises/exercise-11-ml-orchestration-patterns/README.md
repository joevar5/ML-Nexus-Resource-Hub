# Exercise 11: ML Pipeline Orchestration Patterns

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Exercises 02 (Airflow); mod-05 exercises 02 + 11

## Objective

Implement four ML orchestration patterns — DAG-per-model, parametric DAG, event-driven, and continuous training — and characterize their trade-offs. Pick the right pattern for three real scenarios.

## Why this matters

The choice of orchestration pattern shapes your platform's complexity. Picking the wrong one (e.g., DAG-per-model for 200 models) creates an unmanageable code mountain; picking too-late-binding (event-driven for routine batch) hides simple bugs.

## The four patterns

### Pattern 1: DAG per model
One Airflow DAG file per model. Explicit, easy to read, scales linearly in code with model count.

### Pattern 2: Parametric DAG
One DAG file driven by a YAML config; loops over models internally.

### Pattern 3: Event-driven
Pipelines trigger on events (new data, drift detected, schedule). Often with KEDA or Argo Events on K8s.

### Pattern 4: Continuous training
Streaming-style: ingest event → update features → re-train delta → deploy. No discrete "batch" run.

## Requirements

Implement each pattern for a 3-model platform (recs, fraud, ltv). For each:
1. Working code.
2. Add a 4th model and measure: how much new code? How much config? How long?
3. Document failure mode: what happens when 1 model's data is bad?

## Step-by-step

### Step 1 — Define the 3 models (15 min)
```yaml
# models.yaml
- name: recs-ranker
  schedule: "@daily"
  source: events.click
  features_view: recs_features
  algorithm: gradient_boosting
- name: fraud-detector
  schedule: "*/30 * * * *"          # every 30 min
  source: transactions
  features_view: fraud_features
  algorithm: random_forest
- name: ltv-predictor
  schedule: "@weekly"
  source: purchases
  features_view: ltv_features
  algorithm: xgboost
```

### Step 2 — Pattern 1: DAG per model (30 min)
Three files in `dags/`:
- `recs_ranker.py`
- `fraud_detector.py`
- `ltv_predictor.py`

Each is ~100 lines of slightly-varying code. Adding `cltv-predictor` = copy a file + edit ~20 lines.

```python
# dags/recs_ranker.py
from datetime import datetime
from airflow import DAG
from airflow.decorators import task

with DAG("recs_ranker", schedule="@daily", start_date=datetime(2026,1,1)) as dag:
    @task
    def fetch(): ...
    @task
    def train(): ...
    @task
    def evaluate(): ...
    fetch() >> train() >> evaluate()
```

### Step 3 — Pattern 2: Parametric DAG (45 min)
Single file, generates DAGs from config:
```python
# dags/factory.py
import yaml
from airflow import DAG
from airflow.decorators import task

CONFIG = yaml.safe_load(open("/opt/airflow/config/models.yaml"))

for model_cfg in CONFIG:
    dag_id = f"train_{model_cfg['name']}"
    
    with DAG(dag_id, schedule=model_cfg["schedule"],
             start_date=datetime(2026,1,1),
             tags=["ml", model_cfg["name"]]) as dag:
        @task(task_id=f"fetch_{model_cfg['name']}")
        def fetch(source=model_cfg["source"]): ...
        @task(task_id=f"train_{model_cfg['name']}")
        def train(algorithm=model_cfg["algorithm"]): ...
        @task(task_id=f"eval_{model_cfg['name']}")
        def evaluate(): ...
        fetch() >> train() >> evaluate()
    
    globals()[dag_id] = dag
```

Adding a 4th model = adding one entry to YAML. Zero new code.

### Step 4 — Pattern 3: Event-driven (60 min)
Use Argo Events + Argo Workflows on Kubernetes:
```yaml
# event-source.yaml
apiVersion: argoproj.io/v1alpha1
kind: EventSource
metadata: { name: s3-new-data }
spec:
  type: s3
  s3:
    bucket: { name: ml-raw }
    events: [s3:ObjectCreated:*]
```
```yaml
# sensor.yaml — translates event into workflow trigger
apiVersion: argoproj.io/v1alpha1
kind: Sensor
metadata: { name: train-on-new-data }
spec:
  triggers:
    - template:
        argoWorkflow:
          source:
            resource: { ... ref to WorkflowTemplate train-model ... }
          parameters:
            - src: { dependencyName: s3-new-data, dataKey: body.s3.object.key }
              dest: spec.arguments.parameters.0.value
```

New file arrives → workflow triggers → model trains on just that data slice.

### Step 5 — Pattern 4: Continuous training (60 min)
Streaming: every N events, update model incrementally.
```python
# online_update.py
import joblib, redis, json
from kafka import KafkaConsumer
from sklearn.linear_model import SGDClassifier

model = joblib.load("model.joblib")
r = redis.Redis()
c = KafkaConsumer("events", value_deserializer=lambda v: json.loads(v))

BATCH_SIZE = 1000
buffer = []
for msg in c:
    buffer.append(msg.value)
    if len(buffer) >= BATCH_SIZE:
        X = [...]; y = [...]
        model.partial_fit(X, y, classes=[0,1])
        joblib.dump(model, "model.joblib")
        r.publish("model:updated", json.dumps({"trained_on": len(buffer)}))
        buffer = []
```
Production: pair with shadow deployment to validate each update before promoting.

### Step 6 — Compare (30 min)
```markdown
| Pattern | Adding 4th model | Failure isolation | Operational complexity |
|---|---|---|---|
| DAG per model | +1 file, ~100 LOC | Excellent (model A's bug can't touch model B) | Low |
| Parametric | +1 YAML entry | Medium (parsing bug breaks all) | Low |
| Event-driven | +1 sensor + workflow template | Excellent | High (Argo learning curve) |
| Continuous | Substantial refactor | Poor (streaming bug affects everything) | High |
```

### Step 7 — Scenario picks (15 min)
Three scenarios:
1. **3 routine batch models, small team** — Pattern 2 (parametric, fewest moving parts).
2. **50 batch models with custom logic per team** — Pattern 1 (DAG per model; teams own theirs).
3. **Real-time fraud requiring < 1h end-to-end** — Pattern 3 (event-driven).
4. **Recommender that must adapt to user actions within minutes** — Pattern 4 (continuous).

Defend each pick in `SCENARIO_PICKS.md`.

## Deliverables

1. Working code for all 4 patterns.
2. `COMPARISON.md` matrix.
3. `SCENARIO_PICKS.md` recommendations.
4. `OPERATIONS.md`: how each pattern fails and how you'd debug it.

## Validation

- [ ] All 4 patterns run end-to-end with the 3-model setup.
- [ ] Adding a 4th model is measured and documented per pattern.
- [ ] Failure isolation tested: deliberately break one model; observe blast radius.

## Stretch goals

- Add **Prefect** as a 5th pattern; compare to Airflow ergonomics.
- Implement a **DAG generation tool**: from a higher-level YAML, produce both per-model and parametric variants.
- Add **observability** comparable across patterns: per-model dashboards, alerting.

## Common pitfalls

- **Parametric DAG with shared state** — Bug in shared helper breaks all DAGs at once.
- **Event-driven without batched retries** — Burst of events fires N workflows; cluster oversubscribed.
- **Continuous training without canary** — Each partial_fit goes straight to prod. Validate first.
- **DAG-per-model with copy-paste drift** — 12 of 50 DAGs missing a retry config. Use a shared decorator.
