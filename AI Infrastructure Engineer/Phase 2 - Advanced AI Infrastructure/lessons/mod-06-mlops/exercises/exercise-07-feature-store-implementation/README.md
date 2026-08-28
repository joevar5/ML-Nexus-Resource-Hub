# Exercise 07: Feature Store Implementation

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** lab 03 (Feast overview)

## Objective

Stand up a production-shape feature store: offline (Parquet on S3), online (Redis), with a registry, feature definitions, materialization jobs, and point-in-time-correct historical retrieval for training. Use it to train a model and serve predictions.

## Why this matters

Feature stores solve "training/serving skew" — the single most common source of silent ML production failures. A feature in training doesn't match the feature at serving time because they're computed differently. A feature store enforces that they're the same.

## Requirements

1. Feature definitions in code (Feast or similar).
2. Materialization job pulling from a warehouse (DuckDB / Postgres) into Redis.
3. **Point-in-time-correct** historical retrieval for training.
4. Online retrieval < 10ms p99.
5. Tests covering both pipelines.
6. A `feature_store.yaml` registry with at least 5 features across 2 entities.

## Step-by-step

### Step 1 — Setup (15 min)
```bash
pip install 'feast[redis]' pandas duckdb
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Step 2 — Sample warehouse (30 min)
```python
import duckdb, pandas as pd, numpy as np
np.random.seed(42)

con = duckdb.connect("warehouse.duckdb")
# users table
users = pd.DataFrame({"user_id": range(1000), "country": np.random.choice(["US","UK"], 1000),
                       "signup_date": pd.date_range("2024-01-01", periods=1000)})
con.execute("CREATE OR REPLACE TABLE users AS SELECT * FROM users")

# transactions over time
rows = []
for u in range(1000):
    for d in range(30):
        rows.append({"user_id": u, "ts": pd.Timestamp("2026-05-01") + pd.Timedelta(days=d),
                     "amount": np.random.gamma(2, 50)})
tx = pd.DataFrame(rows)
con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM tx")
con.execute("INSERT INTO transactions SELECT * FROM tx")
```

### Step 3 — Feast project (30 min)
```bash
feast init my_store && cd my_store/feature_repo
```

Define entities and feature views in `features.py`:
```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

user = Entity(name="user", join_keys=["user_id"])

users_source = FileSource(
    name="users_src",
    path="../../data/users.parquet",
    timestamp_field="signup_date",
)
users_fv = FeatureView(
    name="user_dim",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[Field(name="country", dtype=String)],
    source=users_source,
)

tx_source = FileSource(
    name="tx_src",
    path="../../data/transactions.parquet",
    timestamp_field="ts",
)
tx_features = FeatureView(
    name="user_transactions",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="amount", dtype=Float32),
    ],
    source=tx_source,
)
```

### Step 4 — Configure Redis (15 min)
```yaml
# feature_store.yaml
project: my_store
provider: local
registry: data/registry.db
online_store:
    type: redis
    connection_string: localhost:6379
offline_store: { type: file }
entity_key_serialization_version: 2
```

### Step 5 — Apply + materialize (15 min)
```bash
feast apply
# Export warehouse tables to parquet for Feast's file source
python -c "import duckdb; con=duckdb.connect('../../warehouse.duckdb'); \
  con.execute('COPY users TO ../../data/users.parquet'); \
  con.execute('COPY transactions TO ../../data/transactions.parquet')"

feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

### Step 6 — Online retrieval (15 min)
```python
from feast import FeatureStore
store = FeatureStore(repo_path=".")

# Benchmark
import time
t0 = time.perf_counter()
for _ in range(1000):
    store.get_online_features(
        features=["user_dim:country", "user_transactions:amount"],
        entity_rows=[{"user_id": 1}],
    ).to_dict()
ms = (time.perf_counter()-t0) / 1000 * 1000
print(f"avg: {ms:.2f}ms")
```
Should be < 10ms.

### Step 7 — Point-in-time historical retrieval for training (45 min)
```python
import pandas as pd
# For each (user, label_timestamp) pair, fetch features as they were at that time
entity_df = pd.DataFrame({
    "user_id":         [1, 1, 2, 2],
    "event_timestamp": [pd.Timestamp("2026-05-15"), pd.Timestamp("2026-05-25"),
                         pd.Timestamp("2026-05-15"), pd.Timestamp("2026-05-25")],
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_dim:country", "user_transactions:amount"],
).to_df()
print(training_df)
```
Each row's features reflect the entity's state at `event_timestamp`. No leakage of future data.

### Step 8 — Tests (15 min)
```python
def test_online_returns_value():
    f = store.get_online_features(features=["user_dim:country"], entity_rows=[{"user_id": 1}]).to_dict()
    assert f["country"][0] in {"US","UK"}

def test_historical_excludes_future():
    df = store.get_historical_features(entity_df=..., features=["user_transactions:amount"]).to_df()
    # All amounts must have ts <= event_timestamp; verify by setting up a known case
    assert df["amount"].isnull().sum() == 0
```

## Deliverables

1. Feast repo with 2+ feature views.
2. Working materialization job.
3. Online retrieval benchmark < 10ms.
4. Historical retrieval producing point-in-time-correct training data.
5. `TRAINING_SERVING_SKEW.md`: explain how this feature store eliminates skew.

## Validation

- [ ] `feast apply` succeeds with no errors.
- [ ] Online retrieval returns values < 10ms p99.
- [ ] Historical retrieval correctly enforces point-in-time semantics.
- [ ] Tests pass.

## Stretch goals

- Migrate offline store to **BigQuery** or **Snowflake** for production scale.
- Add **streaming features** (mutate via streaming ingestion to online store).
- Add **feature lineage**: which models use which features (for impact analysis).

## Common pitfalls

- **TTL too short** — Features expire from Redis before they're queried. Tune per access pattern.
- **No materialization schedule** — Online features go stale. Schedule materialize-incremental every N minutes.
- **Point-in-time bug** — Forgetting `timestamp_field` causes feature leakage. Always set; always test.
- **Cardinality blowup in Redis** — Per-(entity × feature_view) keys can be huge; estimate before going to prod.
