# Lab 04: Stream Events Through Kafka

**Duration:** 75 min  **Prerequisites:** Docker

## Objective
Run Kafka in KRaft mode (no Zookeeper), produce and consume messages in Python, and observe partitioning behavior.

## Steps

### 1. compose.yaml
```yaml
services:
  kafka:
    image: bitnami/kafka:3.7
    ports: ["9092:9092"]
    environment:
      - KAFKA_CFG_NODE_ID=0
      - KAFKA_CFG_PROCESS_ROLES=controller,broker
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
```
```bash
docker compose up -d
sleep 5
```

### 2. Create a topic with 3 partitions
```bash
docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --topic predictions --partitions 3 --replication-factor 1
docker compose exec kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### 3. Producer (Python)
```python
# producer.py
import json, random, time
from kafka import KafkaProducer

p = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode(),
    key_serializer=lambda k: k.encode(),
)
for i in range(100):
    user = f"user-{random.randint(1,5)}"          # 5 distinct keys → fewer partitions used
    p.send("predictions", key=user, value={"i": i, "user": user})
p.flush()
```
`pip install kafka-python`; run it. Watch broker log for incoming traffic.

### 4. Consumer (single)
```python
# consumer.py
from kafka import KafkaConsumer
import json

c = KafkaConsumer(
    "predictions",
    bootstrap_servers=["localhost:9092"],
    group_id="lab-04",
    auto_offset_reset="earliest",
    value_deserializer=lambda v: json.loads(v),
)
for msg in c:
    print(f"partition={msg.partition} offset={msg.offset} key={msg.key} value={msg.value}")
```

### 5. Multi-consumer in the same group
Start two consumers with the same `group_id`. Kafka assigns partitions across them. Stop one — the other picks up its partitions.

### 6. CLI inspection
```bash
docker compose exec kafka kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group lab-04 --describe
```

## Validation
- [ ] Topic `predictions` has 3 partitions.
- [ ] Producer publishes successfully (no exception).
- [ ] Consumer reads all 100 messages.
- [ ] With 2 consumers in the same group, each gets a non-overlapping subset.

## Cleanup
```bash
docker compose down -v
```

## Troubleshooting
- **`NoBrokersAvailable`** — Use `localhost:9092` not `kafka:9092` from your host machine.
- **Same partition for all messages** — All keys are the same, so they hash to the same partition. Increase key diversity.
- **`OffsetOutOfRange`** — Old consumer with stale offsets; reset with `--reset-offsets --to-earliest`.
