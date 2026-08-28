# Exercise 11: Lambda Architecture — Batch + Streaming Hybrid

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercises 02 + 05 + lab 04 (Kafka)

## Objective

Implement a hybrid pipeline that combines batch (daily reprocessing of historical data) and streaming (real-time updates) to serve a single unified feature store. By the end you'll understand Lambda vs Kappa trade-offs and have working code for both layers.

## Why this matters

Most real production ML platforms run a hybrid: streaming for low-latency features, batch for accurate historical aggregates that streaming misses. Engineers who can wire both together correctly are rare; the most common bug is "streaming feature reflects this minute but batch feature is yesterday's", and reconciling that is the whole skill.

## Background — Lambda vs Kappa

**Lambda architecture:**
- Batch layer: source of truth, reprocesses everything daily, high accuracy.
- Streaming layer: real-time approximation, low latency, may be slightly stale or imprecise.
- Serving layer: combines both for a single answer.

**Kappa architecture:**
- Streaming-only; backfills via Kafka log replay.
- Simpler but requires replay-able streams (Kafka with infinite retention).
- Better when streaming has matured enough to replace batch entirely.

This exercise builds Lambda. Stretch is to migrate to Kappa.

## Requirements

1. **Batch layer** (Airflow + Spark) computes "purchases in last 30 days" per user, runs nightly.
2. **Streaming layer** (Kafka + Python consumer) increments "purchases since last batch" per user, runs continuously.
3. **Serving layer**: read both, return `batch_value + streaming_increment`.
4. **Reconciliation**: nightly batch run resets the streaming counter for each user.
5. **Late event handling**: events arriving > 1h late are written to a "late" partition, picked up by next batch.
6. **End-to-end test**: produce events, query feature store, verify correctness.

## Step-by-step

### Step 1 — Kafka + Redis up (15 min)
Use compose from prior labs.

### Step 2 — Event producer (15 min)
```python
# producer.py — simulates real-time purchase events
import json, time, random, uuid
from kafka import KafkaProducer

p = KafkaProducer(bootstrap_servers=["localhost:9092"], value_serializer=lambda v: json.dumps(v).encode())
while True:
    p.send("purchases", {
        "user_id": f"u{random.randint(1,100)}",
        "amount": round(random.uniform(5, 200), 2),
        "ts": int(time.time()),
        "event_id": str(uuid.uuid4()),
    })
    time.sleep(0.1)
```

### Step 3 — Streaming consumer (30 min)
```python
# stream_consumer.py
import json, redis
from kafka import KafkaConsumer

r = redis.Redis(decode_responses=True)
c = KafkaConsumer("purchases", bootstrap_servers=["localhost:9092"],
                  value_deserializer=lambda v: json.loads(v),
                  group_id="features-streaming")

for msg in c:
    e = msg.value
    # increment streaming counter
    r.hincrby(f"features:{e['user_id']}", "purchases_streaming", 1)
    r.hincrbyfloat(f"features:{e['user_id']}", "amount_streaming", e["amount"])
    # also write to a durable log for batch later
    r.xadd(f"audit:purchases:{e['user_id']}", {"data": json.dumps(e)})
```

### Step 4 — Batch layer (45 min)
```python
# batch_aggregate.py — Airflow PythonOperator
import duckdb, redis
con = duckdb.connect("warehouse.duckdb")
r = redis.Redis(decode_responses=True)

# Read everything in the last 30 days from the source-of-truth (warehouse table loaded by another pipeline)
agg = con.execute("""
    SELECT user_id,
           COUNT(*) AS purchases_30d,
           SUM(amount) AS amount_30d
    FROM purchases_warehouse
    WHERE ts >= NOW() - INTERVAL 30 DAY
    GROUP BY user_id
""").fetchdf()

# Write to Redis batch keys, AND reset streaming counters atomically
pipe = r.pipeline()
for _, row in agg.iterrows():
    pipe.hset(f"features:{row.user_id}",
              mapping={"purchases_batch": int(row.purchases_30d), "amount_batch": float(row.amount_30d)})
    pipe.hset(f"features:{row.user_id}", "purchases_streaming", 0)
    pipe.hset(f"features:{row.user_id}", "amount_streaming", 0)
pipe.execute()
```

### Step 5 — Serving layer (15 min)
```python
def get_features(user_id):
    f = r.hgetall(f"features:{user_id}")
    return {
        "purchases_30d": int(f.get("purchases_batch", 0)) + int(f.get("purchases_streaming", 0)),
        "amount_30d":   float(f.get("amount_batch", 0)) + float(f.get("amount_streaming", 0)),
    }
```

### Step 6 — Late event handler (30 min)
Producer simulates late events:
```python
# Once in a while, send an event with ts in the past
if random.random() < 0.05:
    e["ts"] = int(time.time()) - random.randint(3600, 86400*2)
```
Streaming consumer routes late events to a separate Kafka topic `purchases.late`. Batch layer reads both `purchases_warehouse` and `purchases.late` archive on next run.

### Step 7 — Test end-to-end (30 min)
```python
# integration_test.py
producer = ...
# Send 100 events for user u1
for _ in range(100): producer.send("purchases", {...user_id: "u1", amount: 10})
time.sleep(2)
# Query features
assert get_features("u1") == {"purchases_30d": 100, "amount_30d": 1000.0}

# Run batch aggregation
batch_aggregate()
# Streaming counters reset; batch reflects all 100 events
assert get_features("u1") == {"purchases_30d": 100, "amount_30d": 1000.0}

# More streaming events
for _ in range(10): producer.send("purchases", {...user_id: "u1", amount: 10})
time.sleep(1)
assert get_features("u1") == {"purchases_30d": 110, "amount_30d": 1100.0}
```

## Deliverables

1. Producer + streaming consumer + batch DAG + serving function.
2. Integration test demonstrating correctness across the boundary.
3. `ARCHITECTURE.md` describing the Lambda + reconciliation logic.
4. `TRADEOFFS.md`: when to choose Lambda vs Kappa for a given workload.

## Validation

- [ ] Real-time events reflected in features within 1s.
- [ ] Batch aggregate matches streaming + batch reset.
- [ ] Late events surface in subsequent batch run.
- [ ] Integration test green.

## Stretch goals

- Migrate to **Kappa**: drop the batch layer entirely, use Kafka log replay for backfill.
- Add **per-feature staleness**: serving layer indicates how stale each feature is.
- Add **canary**: write feature changes to a shadow keyspace and compare against production for N hours before promoting.

## Common pitfalls

- **Reset gap** — Streaming counter reset between batch read and write loses events arriving in that window. Use Redis transactions or a `last_batch_ts` marker to deduplicate.
- **Double counting** — Streaming event also in batch run. Tag every event with `event_id` and dedupe.
- **Diverging schemas** — Batch and streaming compute the same feature differently. Test against a synthetic dataset with known answer.
- **Backfill makes serving wrong** — During the backfill window, serving sees inconsistent state. Use feature flags to switch reads to a "frozen" snapshot during backfill.
