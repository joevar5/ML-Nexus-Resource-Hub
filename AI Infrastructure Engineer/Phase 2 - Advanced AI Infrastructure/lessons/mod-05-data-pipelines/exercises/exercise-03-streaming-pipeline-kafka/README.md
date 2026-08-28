# Exercise 03: Real-Time ML Feature Pipeline with Kafka

**Estimated Time**: 36-44 hours
**Difficulty**: Advanced
**Prerequisites**: Python 3.9+, Apache Kafka, Flink, Redis, PostgreSQL

## Overview

Build a production-grade real-time feature engineering pipeline using Apache Kafka, Flink, and feature stores. Process streaming events (user clicks, transactions) to compute features for ML models with <100ms latency. Implement exactly-once semantics, backfill historical features, and serve features to online inference with single-digit millisecond latency.

In production ML systems, real-time feature pipelines are critical for:
- **Low-Latency Inference**: Fraud detection (<50ms), recommendations (<100ms)
- **Feature Freshness**: Use events from last 5 minutes for predictions
- **Consistency**: Same features in training and serving (no train/serve skew)
- **Scale**: Process 100K+ events/second with auto-scaling
- **Reliability**: Exactly-once processing, no duplicate/lost events

## Learning Objectives

By completing this exercise, you will:

1. **Build streaming pipelines** with Kafka and Flink
2. **Implement feature engineering** on streaming data
3. **Manage feature stores** (online and offline)
4. **Handle late data and out-of-order events** with watermarks
5. **Implement exactly-once semantics** with Kafka transactions
6. **Build backfill pipelines** for historical features
7. **Monitor pipeline health** with metrics and alerting

## Business Context

**Real-World Scenario**: Your e-commerce platform needs real-time fraud detection for transactions. Current batch pipeline has issues:

- **Stale features**: Batch ETL runs every 6 hours, features are outdated
- **High fraud losses**: 2.5% fraud rate ($500K/month losses) due to delayed detection
- **No real-time signals**: Can't use "5 transactions in last 10 minutes" feature
- **Train/serve skew**: Training uses Spark, serving uses Python (different logic)
- **Scale issues**: Black Friday peaks at 50K transactions/sec, batch pipeline can't keep up

Your task: Build streaming pipeline that:
- Computes features within 50ms of event arrival
- Maintains feature consistency (same code for training/serving)
- Handles 100K events/sec with horizontal scaling
- Guarantees exactly-once processing (no duplicate features)
- Backfills historical features for model training

## Project Structure

```
exercise-03-streaming-pipeline-kafka/
├── README.md
├── requirements.txt
├── docker-compose.yaml              # Kafka, Flink, Redis, Postgres
├── config/
│   ├── kafka-topics.yaml            # Topic configurations
│   ├── flink-config.yaml            # Flink job config
│   └── feature-definitions.yaml     # Feature schemas
├── src/
│   └── feature_pipeline/
│       ├── __init__.py
│       ├── producers/
│       │   ├── __init__.py
│       │   ├── transaction_producer.py  # Generate sample events
│       │   └── clickstream_producer.py
│       ├── processors/
│       │   ├── __init__.py
│       │   ├── flink_processor.py       # Flink feature computation
│       │   ├── aggregations.py          # Time-window aggregations
│       │   └── transformations.py       # Feature transformations
│       ├── stores/
│       │   ├── __init__.py
│       │   ├── online_store.py          # Redis for online serving
│       │   ├── offline_store.py         # Postgres for training
│       │   └── feature_registry.py      # Feature metadata
│       ├── backfill/
│       │   ├── __init__.py
│       │   └── historical_loader.py     # Backfill historical features
│       ├── serving/
│       │   ├── __init__.py
│       │   └── feature_server.py        # FastAPI feature serving
│       └── monitoring/
│           ├── __init__.py
│           └── metrics.py               # Pipeline metrics
├── flink_jobs/
│   ├── fraud_features.py            # Flink job for fraud features
│   └── user_features.py             # Flink job for user features
├── tests/
│   ├── test_processors.py
│   ├── test_stores.py
│   ├── test_serving.py
│   └── fixtures/
│       └── sample_events.json
├── examples/
│   ├── produce_events.py
│   ├── consume_features.py
│   └── backfill_example.py
└── docs/
    ├── DESIGN.md
    ├── FEATURES.md                  # Feature catalog
    └── OPERATIONS.md
```

## Requirements

### Functional Requirements

1. **Event Ingestion**:
   - Consume events from Kafka topics (transactions, clicks)
   - Handle 100K+ events/second
   - Parse and validate event schemas
   - Handle malformed events gracefully

2. **Feature Engineering**:
   - Compute aggregations (count, sum, avg over time windows)
   - Time-based features (hour_of_day, day_of_week)
   - Ratio features (transaction_amount / avg_user_transaction)
   - Categorical encoding (one-hot, target encoding)

3. **Feature Storage**:
   - Online store (Redis): <5ms read latency for serving
   - Offline store (Postgres): Historical features for training
   - Feature registry: Metadata, versioning, lineage

4. **Feature Serving**:
   - REST API: GET /features/{entity_id}?features=f1,f2,f3
   - Batch retrieval: Multiple entities in single request
   - Point-in-time correctness for training

5. **Backfill**:
   - Compute historical features from batch data
   - Populate offline store for model training
   - Validate feature consistency (streaming vs batch)

### Non-Functional Requirements

- **Latency**: P95 feature computation <100ms
- **Throughput**: 100K events/second
- **Availability**: 99.9% uptime
- **Consistency**: Exactly-once semantics
- **Freshness**: Features updated within 1 second of event

## Implementation Tasks

### Task 1: Kafka Setup and Event Producers (6-7 hours)

Set up Kafka infrastructure and event producers.

```yaml
# docker-compose.yaml

version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: features
      POSTGRES_USER: features
      POSTGRES_PASSWORD: features123
    ports:
      - "5432:5432"

  flink-jobmanager:
    image: flink:1.18
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager

  flink-taskmanager:
    image: flink:1.18
    depends_on:
      - flink-jobmanager
    command: taskmanager
    environment:
      - JOB_MANAGER_RPC_ADDRESS=flink-jobmanager
```

```python
# src/feature_pipeline/producers/transaction_producer.py

from kafka import KafkaProducer
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import random
import time
from typing import Dict

@dataclass
class Transaction:
    """Transaction event schema"""
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: str
    location: Dict[str, float]  # {"lat": 37.7, "lon": -122.4}
    device_type: str  # "mobile", "web", "pos"
    is_fraud: bool = False  # Ground truth label

class TransactionProducer:
    """Produce transaction events to Kafka"""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            # Enable idempotence for exactly-once semantics
            enable_idempotence=True,
            acks='all',
            retries=3
        )
        self.topic = "transactions"

    def generate_transaction(self, user_id: str = None) -> Transaction:
        """
        Generate realistic transaction event

        TODO: Implement realistic data generation
        - Normal transactions: $10-500, low fraud rate
        - Fraud patterns: Multiple small transactions, unusual locations
        """
        user_id = user_id or f"user_{random.randint(1, 10000)}"

        # TODO: Generate realistic amount distribution
        # Normal: log-normal distribution around $50
        # Fraud: Often multiple small transactions
        is_fraud = random.random() < 0.025  # 2.5% fraud rate
        if is_fraud:
            amount = random.uniform(5, 50)  # Smaller amounts
        else:
            amount = random.lognormvariate(3.5, 1.0)  # ~$50 mean

        return Transaction(
            transaction_id=f"txn_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            user_id=user_id,
            merchant_id=f"merchant_{random.randint(1, 1000)}",
            amount=round(amount, 2),
            currency="USD",
            timestamp=datetime.utcnow().isoformat(),
            location={
                "lat": random.uniform(25, 50),
                "lon": random.uniform(-125, -65)
            },
            device_type=random.choice(["mobile", "web", "pos"]),
            is_fraud=is_fraud
        )

    def produce(self, transaction: Transaction):
        """
        Produce transaction to Kafka

        Use user_id as key for partitioning (all events for same user go to same partition)
        """
        # TODO: Send to Kafka
        future = self.producer.send(
            topic=self.topic,
            key=transaction.user_id,
            value=asdict(transaction)
        )

        # TODO: Handle delivery confirmation
        try:
            record_metadata = future.get(timeout=10)
            return record_metadata
        except Exception as e:
            print(f"Failed to produce transaction: {e}")
            raise

    def produce_stream(self, events_per_second: int = 100, duration_seconds: int = 60):
        """
        Produce continuous stream of transactions

        Args:
            events_per_second: Target throughput
            duration_seconds: How long to produce
        """
        interval = 1.0 / events_per_second
        start_time = time.time()

        produced = 0
        while (time.time() - start_time) < duration_seconds:
            transaction = self.generate_transaction()
            self.produce(transaction)
            produced += 1

            if produced % 1000 == 0:
                print(f"Produced {produced} transactions")

            time.sleep(interval)

        self.producer.flush()
        print(f"Produced {produced} transactions in {duration_seconds}s")

    def close(self):
        """Close producer"""
        self.producer.close()
```

**Acceptance Criteria**:
- ✅ Kafka cluster running
- ✅ Produce events with correct schema
- ✅ Configurable throughput (events/sec)
- ✅ Exactly-once producer semantics
- ✅ Realistic data distribution

---

### Task 2: Flink Feature Processor (9-11 hours)

Implement streaming feature computation with Flink.

```python
# flink_jobs/fraud_features.py

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetResetStrategy, KafkaSink
from pyflink.common import SimpleStringSchema, Types
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common.time import Time
import json
from datetime import datetime, timedelta

class FraudFeatureProcessor:
    """
    Flink job to compute fraud detection features

    Features:
    1. transaction_count_1h: Count of transactions in last 1 hour
    2. transaction_sum_1h: Sum of transaction amounts in last 1 hour
    3. transaction_avg_1h: Average transaction amount in last 1 hour
    4. unique_merchants_1h: Count of unique merchants in last 1 hour
    5. transaction_velocity: Transactions per minute in last 10 minutes
    6. amount_deviation: (current_amount - avg_amount) / std_amount
    """

    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_parallelism(4)

        # Enable checkpointing for fault tolerance
        self.env.enable_checkpointing(60000)  # Checkpoint every 60s

    def run(self):
        """Execute Flink streaming job"""

        # TODO: Configure Kafka source
        kafka_source = KafkaSource.builder() \
            .set_bootstrap_servers("localhost:9092") \
            .set_topics("transactions") \
            .set_group_id("fraud-feature-processor") \
            .set_starting_offsets(KafkaOffsetResetStrategy.LATEST) \
            .set_value_only_deserializer(SimpleStringSchema()) \
            .build()

        # TODO: Create watermark strategy for event time processing
        watermark_strategy = WatermarkStrategy \
            .for_bounded_out_of_orderness(timedelta(seconds=10)) \
            .with_timestamp_assigner(lambda event: self._extract_timestamp(event))

        # TODO: Create data stream
        stream = self.env.from_source(
            kafka_source,
            watermark_strategy,
            "Kafka Source"
        )

        # TODO: Parse JSON events
        parsed_stream = stream.map(
            lambda x: json.loads(x),
            output_type=Types.MAP(Types.STRING(), Types.STRING())
        )

        # TODO: Key by user_id
        keyed_stream = parsed_stream.key_by(lambda event: event['user_id'])

        # TODO: Compute windowed aggregations
        # 1-hour tumbling window for transaction counts/sums
        hourly_features = keyed_stream \
            .window(TumblingEventTimeWindows.of(Time.hours(1))) \
            .process(HourlyAggregationFunction())

        # 10-minute sliding window for velocity
        velocity_features = keyed_stream \
            .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1))) \
            .process(VelocityFunction())

        # TODO: Compute stateful features (running statistics)
        deviation_features = keyed_stream.process(DeviationFunction())

        # TODO: Merge feature streams
        all_features = hourly_features.union(velocity_features).union(deviation_features)

        # TODO: Write to online store (Redis) via Kafka
        kafka_sink = KafkaSink.builder() \
            .set_bootstrap_servers("localhost:9092") \
            .set_record_serializer(...) \
            .build()

        all_features.sink_to(kafka_sink)

        # TODO: Execute job
        self.env.execute("Fraud Feature Processor")

    def _extract_timestamp(self, event_json: str) -> int:
        """Extract timestamp from event for watermarking"""
        event = json.loads(event_json)
        dt = datetime.fromisoformat(event['timestamp'])
        return int(dt.timestamp() * 1000)  # milliseconds

class HourlyAggregationFunction:
    """
    Window function to compute hourly aggregations

    Output features:
    - transaction_count_1h
    - transaction_sum_1h
    - transaction_avg_1h
    - unique_merchants_1h
    """

    def process(self, key, context, elements):
        """
        Process window of transactions

        Args:
            key: user_id
            context: Window context
            elements: Iterable of transactions in window
        """
        # TODO: Aggregate transactions
        transactions = list(elements)
        count = len(transactions)
        amounts = [float(t['amount']) for t in transactions]
        merchants = set(t['merchant_id'] for t in transactions)

        # TODO: Compute features
        features = {
            "user_id": key,
            "timestamp": context.window().max_timestamp(),
            "transaction_count_1h": count,
            "transaction_sum_1h": sum(amounts),
            "transaction_avg_1h": sum(amounts) / count if count > 0 else 0,
            "unique_merchants_1h": len(merchants)
        }

        yield json.dumps(features)

class VelocityFunction:
    """Compute transaction velocity (transactions per minute)"""

    def process(self, key, context, elements):
        transactions = list(elements)
        window_duration_minutes = 10

        velocity = len(transactions) / window_duration_minutes

        features = {
            "user_id": key,
            "timestamp": context.window().max_timestamp(),
            "transaction_velocity": velocity
        }

        yield json.dumps(features)

class DeviationFunction:
    """
    Compute amount deviation using running statistics

    Maintains running mean and std for each user
    """

    def __init__(self):
        # TODO: Initialize state
        # Use Flink's ValueState to maintain per-user statistics
        pass

    def process_element(self, transaction, ctx):
        """
        Process single transaction

        Update running statistics and compute deviation
        """
        # TODO: Get current statistics from state
        # TODO: Update with new transaction amount
        # TODO: Compute deviation: (amount - mean) / std
        # TODO: Emit feature
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Flink job processes Kafka events
- ✅ Compute time-window aggregations
- ✅ Handle late data with watermarks
- ✅ Maintain state for running statistics
- ✅ Emit features to output topic

---

### Task 3: Feature Stores (8-10 hours)

Implement online and offline feature stores.

```python
# src/feature_pipeline/stores/online_store.py

import redis
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta

class OnlineFeatureStore:
    """
    Online feature store using Redis

    Design:
    - Key: feature:{entity_type}:{entity_id}:{feature_name}
    - Value: JSON {"value": 123.45, "timestamp": "2024-01-25T10:00:00Z"}
    - TTL: 7 days (features auto-expire)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
        self.default_ttl = timedelta(days=7)

    def write_feature(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Write feature to online store

        Args:
            entity_type: "user", "transaction", etc.
            entity_id: Specific entity ID
            feature_name: Name of feature
            value: Feature value
            timestamp: Event timestamp
        """
        key = self._build_key(entity_type, entity_id, feature_name)

        feature_data = {
            "value": value,
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        }

        # TODO: Write to Redis with TTL
        self.client.setex(
            key,
            self.default_ttl,
            json.dumps(feature_data)
        )

    def write_features_batch(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """
        Write multiple features for entity in single operation

        Use Redis pipeline for atomic batch write
        """
        pipeline = self.client.pipeline()

        for feature_name, value in features.items():
            key = self._build_key(entity_type, entity_id, feature_name)
            feature_data = {
                "value": value,
                "timestamp": (timestamp or datetime.utcnow()).isoformat()
            }
            pipeline.setex(key, self.default_ttl, json.dumps(feature_data))

        # TODO: Execute pipeline
        pipeline.execute()

    def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Get multiple features for entity

        Returns:
            {"feature1": 123.45, "feature2": 67.89, "feature3": None}
        """
        # TODO: Use Redis MGET for batch retrieval
        keys = [self._build_key(entity_type, entity_id, f) for f in feature_names]
        values = self.client.mget(keys)

        # TODO: Parse results
        result = {}
        for feature_name, value_json in zip(feature_names, values):
            if value_json:
                data = json.loads(value_json)
                result[feature_name] = data['value']
            else:
                result[feature_name] = None

        return result

    def _build_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        """Build Redis key"""
        return f"feature:{entity_type}:{entity_id}:{feature_name}"
```

```python
# src/feature_pipeline/stores/offline_store.py

import psycopg2
from psycopg2.extras import execute_batch
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

class OfflineFeatureStore:
    """
    Offline feature store using PostgreSQL

    Schema:
    - features table: entity_type, entity_id, feature_name, value, event_timestamp, ingestion_timestamp
    - Optimized for point-in-time queries for training

    Index on (entity_type, entity_id, feature_name, event_timestamp)
    """

    def __init__(self, connection_string: str = "postgresql://features:features123@localhost/features"):
        self.conn = psycopg2.connect(connection_string)
        self._initialize_schema()

    def _initialize_schema(self):
        """Create tables and indexes"""
        with self.conn.cursor() as cur:
            # TODO: Create features table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    entity_type VARCHAR(50) NOT NULL,
                    entity_id VARCHAR(100) NOT NULL,
                    feature_name VARCHAR(100) NOT NULL,
                    value DOUBLE PRECISION,
                    event_timestamp TIMESTAMP NOT NULL,
                    ingestion_timestamp TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (entity_type, entity_id, feature_name, event_timestamp)
                )
            """)

            # TODO: Create index for point-in-time queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_pit
                ON features (entity_type, entity_id, feature_name, event_timestamp DESC)
            """)

            self.conn.commit()

    def write_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, float],
        event_timestamp: datetime
    ):
        """
        Write features to offline store

        Args:
            features: {"feature1": 123.45, "feature2": 67.89}
        """
        with self.conn.cursor() as cur:
            rows = [
                (entity_type, entity_id, feature_name, value, event_timestamp)
                for feature_name, value in features.items()
            ]

            # TODO: Batch insert
            execute_batch(
                cur,
                """
                INSERT INTO features (entity_type, entity_id, feature_name, value, event_timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (entity_type, entity_id, feature_name, event_timestamp)
                DO UPDATE SET value = EXCLUDED.value
                """,
                rows
            )

            self.conn.commit()

    def get_features_point_in_time(
        self,
        entity_type: str,
        entity_ids: List[str],
        feature_names: List[str],
        event_timestamp: datetime
    ) -> pd.DataFrame:
        """
        Get features as of specific timestamp (point-in-time correctness)

        For each entity and feature, returns the latest value BEFORE event_timestamp.
        Critical for training to avoid label leakage.

        Returns:
            DataFrame with columns: entity_id, feature_name, value, event_timestamp
        """
        with self.conn.cursor() as cur:
            # TODO: Query latest features before timestamp
            query = """
                SELECT DISTINCT ON (entity_id, feature_name)
                    entity_id,
                    feature_name,
                    value,
                    event_timestamp
                FROM features
                WHERE entity_type = %s
                  AND entity_id = ANY(%s)
                  AND feature_name = ANY(%s)
                  AND event_timestamp <= %s
                ORDER BY entity_id, feature_name, event_timestamp DESC
            """

            cur.execute(query, (entity_type, entity_ids, feature_names, event_timestamp))

            # TODO: Convert to DataFrame
            rows = cur.fetchall()
            return pd.DataFrame(
                rows,
                columns=['entity_id', 'feature_name', 'value', 'event_timestamp']
            )

    def get_feature_range(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get time series of feature values

        Useful for debugging and analysis
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT event_timestamp, value
                FROM features
                WHERE entity_type = %s
                  AND entity_id = %s
                  AND feature_name = %s
                  AND event_timestamp BETWEEN %s AND %s
                ORDER BY event_timestamp
            """, (entity_type, entity_id, feature_name, start_time, end_time))

            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=['event_timestamp', 'value'])
```

**Acceptance Criteria**:
- ✅ Online store with <5ms read latency
- ✅ Offline store with point-in-time queries
- ✅ Batch write operations
- ✅ Proper indexing for performance
- ✅ TTL for online features

---

### Task 4: Feature Serving API (5-6 hours)

Build REST API for feature serving.

```python
# src/feature_pipeline/serving/feature_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from ..stores.online_store import OnlineFeatureStore
from ..stores.offline_store import OfflineFeatureStore

app = FastAPI(title="Feature Serving API")

# Initialize stores
online_store = OnlineFeatureStore()
offline_store = OfflineFeatureStore()

class FeatureRequest(BaseModel):
    """Request model for feature retrieval"""
    entity_type: str
    entity_id: str
    feature_names: List[str]

class BatchFeatureRequest(BaseModel):
    """Request model for batch feature retrieval"""
    entity_type: str
    entity_ids: List[str]
    feature_names: List[str]

class FeatureResponse(BaseModel):
    """Response model"""
    entity_id: str
    features: Dict[str, Optional[float]]
    timestamp: str

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/features/online", response_model=FeatureResponse)
def get_online_features(request: FeatureRequest):
    """
    Get features from online store (for inference)

    Example:
    POST /features/online
    {
        "entity_type": "user",
        "entity_id": "user_12345",
        "feature_names": ["transaction_count_1h", "transaction_avg_1h"]
    }

    Response:
    {
        "entity_id": "user_12345",
        "features": {
            "transaction_count_1h": 5.0,
            "transaction_avg_1h": 127.50
        },
        "timestamp": "2024-01-25T10:00:00Z"
    }
    """
    try:
        # TODO: Fetch from online store
        features = online_store.get_features(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            feature_names=request.feature_names
        )

        return FeatureResponse(
            entity_id=request.entity_id,
            features=features,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/batch", response_model=List[FeatureResponse])
def get_batch_features(request: BatchFeatureRequest):
    """
    Get features for multiple entities

    Used for batch inference
    """
    try:
        results = []
        for entity_id in request.entity_ids:
            features = online_store.get_features(
                entity_type=request.entity_type,
                entity_id=entity_id,
                feature_names=request.feature_names
            )
            results.append(FeatureResponse(
                entity_id=entity_id,
                features=features,
                timestamp=datetime.utcnow().isoformat()
            ))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/offline/{entity_type}/{entity_id}")
def get_offline_features(
    entity_type: str,
    entity_id: str,
    feature_names: str,  # Comma-separated
    event_timestamp: str  # ISO format
):
    """
    Get features from offline store (for training)

    Point-in-time query to avoid label leakage
    """
    try:
        features_list = feature_names.split(',')
        timestamp = datetime.fromisoformat(event_timestamp)

        df = offline_store.get_features_point_in_time(
            entity_type=entity_type,
            entity_ids=[entity_id],
            feature_names=features_list,
            event_timestamp=timestamp
        )

        # TODO: Pivot DataFrame to feature dict
        features = {}
        for _, row in df.iterrows():
            features[row['feature_name']] = row['value']

        return {
            "entity_id": entity_id,
            "features": features,
            "event_timestamp": event_timestamp
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """
    Expose metrics for monitoring

    Returns:
    - Cache hit rate
    - Request latency
    - Feature freshness
    """
    # TODO: Implement metrics collection
    return {
        "requests_total": 1000,
        "cache_hit_rate": 0.95,
        "avg_latency_ms": 3.2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Acceptance Criteria**:
- ✅ REST API for online features
- ✅ Batch feature retrieval
- ✅ Point-in-time offline queries
- ✅ <5ms p95 latency for online
- ✅ Metrics endpoint

---

### Task 5: Historical Backfill (5-6 hours)

Implement backfill for historical features.

```python
# src/feature_pipeline/backfill/historical_loader.py

from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
from ..stores.offline_store import OfflineFeatureStore
from ..processors.aggregations import compute_hourly_features

class HistoricalBackfill:
    """
    Backfill historical features from batch data

    Use case: Train model on last 90 days of features
    """

    def __init__(self, offline_store: OfflineFeatureStore):
        self.offline_store = offline_store

    def backfill_from_dataframe(
        self,
        transactions_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Compute features from historical transaction data

        Args:
            transactions_df: Historical transactions with columns:
                - transaction_id, user_id, amount, timestamp, etc.
            start_date: Backfill start date
            end_date: Backfill end date
        """
        # TODO: Filter to date range
        df = transactions_df[
            (transactions_df['timestamp'] >= start_date) &
            (transactions_df['timestamp'] <= end_date)
        ].copy()

        # TODO: Compute features using same logic as streaming
        # Critical: Use SAME feature computation code to avoid train/serve skew

        # Group by user and compute aggregations
        for user_id in df['user_id'].unique():
            user_transactions = df[df['user_id'] == user_id].sort_values('timestamp')

            # TODO: Compute time-window features for each transaction
            for idx, row in user_transactions.iterrows():
                event_time = row['timestamp']

                # Get transactions in 1-hour window before this one
                window_start = event_time - timedelta(hours=1)
                window_txns = user_transactions[
                    (user_transactions['timestamp'] >= window_start) &
                    (user_transactions['timestamp'] < event_time)
                ]

                # Compute features
                features = compute_hourly_features(window_txns)

                # Write to offline store
                self.offline_store.write_features(
                    entity_type="user",
                    entity_id=user_id,
                    features=features,
                    event_timestamp=event_time
                )

        print(f"Backfilled features for {df['user_id'].nunique()} users")

    def validate_consistency(
        self,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        timestamp: datetime
    ) -> Dict:
        """
        Validate feature consistency between streaming and batch

        Compare feature values computed by:
        1. Streaming pipeline (in offline store)
        2. Batch backfill

        Should be identical (or within tolerance for floating point)
        """
        # TODO: Get streaming value
        streaming_df = self.offline_store.get_feature_range(
            entity_type, entity_id, feature_name,
            timestamp - timedelta(minutes=1),
            timestamp + timedelta(minutes=1)
        )

        # TODO: Get batch value (would need separate backfill run)
        # TODO: Compare values
        # TODO: Return validation result

        return {
            "feature": feature_name,
            "streaming_value": 0.0,
            "batch_value": 0.0,
            "difference": 0.0,
            "consistent": True
        }
```

**Acceptance Criteria**:
- ✅ Backfill from historical data
- ✅ Use same feature logic as streaming
- ✅ Validate streaming vs batch consistency
- ✅ Handle large datasets efficiently
- ✅ Write to offline store

---

### Task 6: Monitoring and Metrics (3-4 hours)

Implement pipeline monitoring.

```python
# src/feature_pipeline/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Define metrics
feature_compute_duration = Histogram(
    'feature_compute_duration_seconds',
    'Time to compute features',
    ['feature_name']
)

feature_freshness = Gauge(
    'feature_freshness_seconds',
    'Age of feature value',
    ['entity_type', 'feature_name']
)

events_processed = Counter(
    'events_processed_total',
    'Total events processed',
    ['source']
)

feature_store_latency = Histogram(
    'feature_store_latency_seconds',
    'Feature store operation latency',
    ['operation', 'store_type']
)

def track_duration(metric_name: str):
    """Decorator to track function duration"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            feature_compute_duration.labels(feature_name=metric_name).observe(duration)
            return result
        return wrapper
    return decorator

class PipelineMonitor:
    """Monitor pipeline health"""

    def __init__(self):
        self.alerts = []

    def check_feature_freshness(
        self,
        online_store,
        entity_type: str,
        entity_id: str,
        feature_name: str,
        max_age_seconds: int = 300
    ) -> bool:
        """
        Check if feature is fresh enough

        Alert if feature value is >5 minutes old
        """
        # TODO: Get feature with timestamp
        # TODO: Calculate age
        # TODO: Update gauge
        # TODO: Alert if stale
        pass

    def check_kafka_lag(self, consumer_group: str, topic: str) -> int:
        """
        Check Kafka consumer lag

        Alert if lag >100K messages
        """
        # TODO: Query Kafka for consumer lag
        # TODO: Alert if high lag
        pass
```

**Acceptance Criteria**:
- ✅ Prometheus metrics exposed
- ✅ Track compute duration
- ✅ Monitor feature freshness
- ✅ Check Kafka lag
- ✅ Alerting on anomalies

---

## Testing Requirements

```python
def test_feature_computation():
    """Test feature computation logic"""
    transactions = [
        {"amount": 100, "merchant_id": "m1"},
        {"amount": 200, "merchant_id": "m2"},
        {"amount": 150, "merchant_id": "m1"}
    ]

    features = compute_hourly_features(transactions)

    assert features['transaction_count_1h'] == 3
    assert features['transaction_sum_1h'] == 450
    assert features['transaction_avg_1h'] == 150
    assert features['unique_merchants_1h'] == 2

def test_point_in_time_correctness():
    """Test point-in-time feature retrieval"""
    # Ensure no label leakage
    # Features at time T should only use data before T
    pass
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Feature Latency (P95)** | <100ms | ________ms |
| **Throughput** | 100K events/sec | ________ |
| **Online Store Latency** | <5ms | ________ms |
| **Feature Freshness** | <1s | ________s |

## Validation

Submit:
1. Complete implementation
2. Flink job processing events
3. Feature stores (online + offline)
4. Feature serving API
5. Backfill pipeline
6. Test suite
7. Documentation

## Resources

- [Apache Kafka](https://kafka.apache.org/documentation/)
- [Apache Flink](https://flink.apache.org/)
- [Feast Feature Store](https://feast.dev/)
- [Feature Store for ML](https://www.featurestore.org/)

---

**Estimated Completion Time**: 36-44 hours

**Skills Practiced**:
- Streaming data pipelines
- Apache Kafka, Flink
- Feature engineering
- Online/offline stores
- Exactly-once semantics
- Point-in-time correctness
