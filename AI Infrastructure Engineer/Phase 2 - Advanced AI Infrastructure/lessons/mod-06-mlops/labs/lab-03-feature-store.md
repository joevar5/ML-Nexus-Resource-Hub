# Lab 03: Stand Up a Feast Feature Store

**Duration:** 75 min  **Prerequisites:** Python 3.11+, Redis

## Objective
Define feature views in Feast, materialize them from a Parquet source to an online Redis store, and retrieve features both online (low-latency) and offline (training).

## Steps

### 1. Install Feast
```bash
python -m venv venv && source venv/bin/activate
pip install 'feast[redis]' pandas
```

### 2. Initialize a repo
```bash
feast init my_store
cd my_store/feature_repo
```

### 3. Generate sample data
```python
# data/driver_stats.parquet generation
import pandas as pd, numpy as np, pyarrow.parquet as pq
ids = list(range(1000, 1005))
ts = pd.date_range("2026-05-01", periods=24, freq="H")
rows = []
for d in ids:
    for t in ts:
        rows.append({"driver_id": d, "event_timestamp": t,
                     "trips_today": int(np.random.poisson(8)),
                     "avg_rating":  float(np.round(np.random.normal(4.7, 0.2), 2))})
pd.DataFrame(rows).to_parquet("data/driver_stats.parquet")
```

### 4. Feature definitions
```python
# features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

driver = Entity(name="driver_id", join_keys=["driver_id"])

src = FileSource(
    name="driver_stats_src",
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
)

driver_stats = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="trips_today", dtype=Int64),
        Field(name="avg_rating",  dtype=Float32),
    ],
    source=src,
    online=True,
)
```

### 5. Configure Redis online store
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

Start redis: `docker run -d -p 6379:6379 redis:7-alpine`.

### 6. Apply and materialize
```bash
feast apply
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

### 7. Online retrieval
```python
from feast import FeatureStore
store = FeatureStore(repo_path=".")
print(store.get_online_features(
    features=["driver_stats:trips_today", "driver_stats:avg_rating"],
    entity_rows=[{"driver_id": 1000}, {"driver_id": 1003}],
).to_dict())
```

### 8. Offline retrieval (training set)
```python
import pandas as pd
entity_df = pd.DataFrame({"driver_id": [1000,1001], "event_timestamp": [pd.Timestamp("2026-05-01 10:00"), pd.Timestamp("2026-05-01 12:00")]})
print(store.get_historical_features(entity_df=entity_df,
        features=["driver_stats:trips_today","driver_stats:avg_rating"]).to_df())
```

## Validation
- [ ] `feast apply` returns no errors.
- [ ] Online retrieval returns values for both entity rows.
- [ ] Offline retrieval returns the historical values (point-in-time correct).

## Cleanup
```bash
docker stop $(docker ps -q --filter ancestor=redis:7-alpine)
deactivate && rm -rf my_store venv
```

## Troubleshooting
- **`Online store is empty`** — You skipped `feast materialize-incremental`. Online store is populated by explicit materialization, not on `apply`.
- **`Could not connect to redis`** — Wrong host in `feature_store.yaml`. Should be `localhost:6379` from your laptop.
- **Schema mismatch** — Dtype in `Field` must match the Parquet column type.
