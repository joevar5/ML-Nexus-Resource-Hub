# Lesson 04: Logging with ELK Stack and Loki

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand the importance of centralized logging for AI infrastructure
- Set up and configure the ELK Stack (Elasticsearch, Logstash, Kibana)
- Deploy and use Grafana Loki for log aggregation
- Implement structured logging in Python applications
- Query and analyze logs effectively
- Design log retention and archival strategies
- Integrate logs with metrics for comprehensive observability

## Prerequisites
- Completion of Lessons 01-03 (Observability, Prometheus, Grafana)
- Understanding of Docker and Kubernetes
- Python programming experience
- Basic Linux command-line knowledge

## Introduction

Logs are the detailed record of events in your systems. For AI infrastructure, logs are crucial for debugging model failures, tracking data pipeline issues, understanding GPU errors, and auditing ML experiments. Unlike metrics (which are aggregated numbers) and traces (which track requests), logs provide detailed, timestamped records of what happened.

### Why Centralized Logging for ML Systems?

1. **Debugging**: Trace errors in training jobs, inference pipelines, and data processing
2. **Audit trail**: Track who deployed which model, data access patterns, experiment parameters
3. **Performance analysis**: Identify slow operations, resource contention, I/O bottlenecks
4. **Compliance**: Maintain records for data governance and model explainability
5. **Correlation**: Link logs with metrics and traces for root cause analysis

---

## 1. Logging Fundamentals

### Log Levels

Standard log levels (from most to least severe):

| Level | Purpose | Example |
|-------|---------|---------|
| **FATAL/CRITICAL** | System is unusable | GPU driver crash, OOM killer |
| **ERROR** | Failure in operation | Model loading failed, inference error |
| **WARN** | Potential issue | High GPU temperature, slow query |
| **INFO** | General information | Model deployed, batch completed |
| **DEBUG** | Detailed diagnostic | Function parameters, intermediate values |
| **TRACE** | Very detailed | Loop iterations, all function calls |

### Structured vs. Unstructured Logging

**Unstructured (Plain Text):**
```
2025-10-15 10:23:45 Model bert-base processed 1000 requests in 5.2 seconds
```
- Human-readable
- Hard to parse and query
- Difficult to aggregate

**Structured (JSON):**
```json
{
  "timestamp": "2025-10-15T10:23:45Z",
  "level": "INFO",
  "model_name": "bert-base",
  "model_version": "v1.2",
  "requests_processed": 1000,
  "duration_seconds": 5.2,
  "gpu_id": 0,
  "message": "Batch processing completed"
}
```
- Machine-parseable
- Easy to query and filter
- Enables aggregation and analysis

### Logging Best Practices

1. **Use structured logging** (JSON format)
2. **Include context** (request ID, user ID, model name, etc.)
3. **Log at appropriate levels** (not everything at INFO)
4. **Avoid logging sensitive data** (passwords, PII, API keys)
5. **Use correlation IDs** to track requests across services
6. **Include timing information** for performance analysis
7. **Log errors with stack traces**

---

## 2. ELK Stack (Elasticsearch, Logstash, Kibana)

The ELK Stack is a powerful suite for log aggregation, processing, storage, and visualization.

### Architecture

```
┌─────────────┐
│ Application │
│   (Logs)    │
└──────┬──────┘
       │
       ↓
┌─────────────┐      ┌──────────────┐      ┌───────────┐
│  Filebeat/  │─────>│   Logstash   │─────>│Elasticsearch│
│  Fluentd    │      │  (Process &  │      │  (Store &  │
│  (Collect)  │      │   Transform) │      │   Index)   │
└─────────────┘      └──────────────┘      └─────┬──────┘
                                                  │
                                                  ↓
                                           ┌──────────┐
                                           │  Kibana  │
                                           │(Visualize│
                                           │  Query)  │
                                           └──────────┘
```

**Components:**
- **Elasticsearch**: Distributed search and analytics engine (storage)
- **Logstash**: Data processing pipeline (ingestion and transformation)
- **Kibana**: Visualization and query interface
- **Filebeat/Fluentd**: Lightweight log shippers (collection)

### Installing ELK Stack

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms2g -Xmx2g
      - xpack.security.enabled=false
      - xpack.security.http.ssl.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
      - "9300:9300"
    networks:
      - elk
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200"]
      interval: 30s
      timeout: 10s
      retries: 5

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    container_name: logstash
    volumes:
      - ./logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    ports:
      - "5000:5000/tcp"  # Logstash TCP input
      - "5000:5000/udp"  # Logstash UDP input
      - "9600:9600"      # Logstash monitoring
    environment:
      - LS_JAVA_OPTS=-Xmx1g -Xms1g
    networks:
      - elk
    depends_on:
      elasticsearch:
        condition: service_healthy

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    networks:
      - elk
    depends_on:
      elasticsearch:
        condition: service_healthy

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.10.0
    container_name: filebeat
    user: root
    volumes:
      - ./filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: filebeat -e -strict.perms=false
    networks:
      - elk
    depends_on:
      - elasticsearch
      - logstash

volumes:
  elasticsearch_data:

networks:
  elk:
    driver: bridge
```

### Logstash Configuration

**logstash/pipeline/ml-logs.conf:**

```ruby
input {
  # TCP input for structured logs
  tcp {
    port => 5000
    codec => json_lines
    tags => ["tcp"]
  }

  # Filebeat input
  beats {
    port => 5044
    tags => ["beats"]
  }
}

filter {
  # Parse JSON logs
  if [message] =~ /^{.*}$/ {
    json {
      source => "message"
    }
  }

  # Add timestamp if missing
  if ![timestamp] {
    ruby {
      code => "event.set('timestamp', Time.now.utc.iso8601)"
    }
  }

  # Parse log level
  if [level] {
    mutate {
      uppercase => ["level"]
    }
  }

  # Extract model information
  if [model_name] {
    mutate {
      add_field => {
        "[@metadata][index_prefix]" => "ml-logs-models"
      }
    }
  }

  # Parse GPU logs
  if [gpu_id] {
    mutate {
      convert => {
        "gpu_id" => "integer"
      }
      add_field => {
        "[@metadata][index_prefix]" => "ml-logs-gpu"
      }
    }
  }

  # Parse training job logs
  if [job_type] == "training" {
    mutate {
      add_field => {
        "[@metadata][index_prefix]" => "ml-logs-training"
      }
    }

    # Extract epoch and loss if present
    grok {
      match => {
        "message" => "Epoch %{NUMBER:epoch:int}.*loss[:=]\s*%{NUMBER:loss:float}"
      }
      tag_on_failure => []
    }
  }

  # Add environment tags
  mutate {
    add_field => {
      "environment" => "${ENV:dev}"
      "cluster" => "${CLUSTER:local}"
    }
  }

  # Remove unnecessary fields
  mutate {
    remove_field => ["@version", "host"]
  }
}

output {
  # Output to Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index_prefix]:ml-logs}-%{+YYYY.MM.dd}"
  }

  # Debug output (optional)
  # stdout {
  #   codec => rubydebug
  # }
}
```

### Filebeat Configuration

**filebeat/filebeat.yml:**

```yaml
filebeat.inputs:
  # Container logs
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'
    processors:
      - add_docker_metadata:
          host: "unix:///var/run/docker.sock"
      - decode_json_fields:
          fields: ["message"]
          target: ""
          overwrite_keys: true

  # Application logs from file
  - type: log
    enabled: true
    paths:
      - /var/log/ml-platform/*.log
    fields:
      service: ml-platform
    json.keys_under_root: true
    json.add_error_key: true

output.logstash:
  hosts: ["logstash:5044"]
  compression_level: 3

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

---

## 3. Grafana Loki

Loki is a log aggregation system designed to be cost-effective and easy to operate, inspired by Prometheus.

### Loki vs. ELK

| Feature | Loki | ELK |
|---------|------|-----|
| **Storage** | Object storage (S3, GCS) | Elasticsearch |
| **Indexing** | Labels only (like Prometheus) | Full-text indexing |
| **Query language** | LogQL (similar to PromQL) | Elasticsearch DSL |
| **Cost** | Lower (less indexing) | Higher (full indexing) |
| **Setup complexity** | Simpler | More complex |
| **Search performance** | Slower for unindexed fields | Faster full-text search |
| **Integration** | Native Grafana integration | Separate Kibana |

### Installing Loki with Promtail

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  loki:
    image: grafana/loki:2.9.0
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/config.yml
      - loki_data:/loki
    command: -config.file=/etc/loki/config.yml
    networks:
      - monitoring

  promtail:
    image: grafana/promtail:2.9.0
    container_name: promtail
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - monitoring
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
    networks:
      - monitoring
    depends_on:
      - loki

volumes:
  loki_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

### Loki Configuration

**loki-config.yml:**

```yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2023-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h  # 7 days
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20
  per_stream_rate_limit: 5MB
  per_stream_rate_limit_burst: 20MB

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 30d

compactor:
  working_directory: /loki/compactor
  shared_store: filesystem
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
```

### Promtail Configuration

**promtail-config.yml:**

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker container logs
  - job_name: containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
    pipeline_stages:
      - json:
          expressions:
            level: level
            message: message
            timestamp: timestamp
            model_name: model_name
            gpu_id: gpu_id
      - labels:
          level:
          model_name:
          gpu_id:
      - timestamp:
          source: timestamp
          format: RFC3339

  # ML platform logs
  - job_name: ml-platform
    static_configs:
      - targets:
          - localhost
        labels:
          job: ml-platform
          __path__: /var/log/ml-platform/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            msg: message
            model: model_name
            version: model_version
      - labels:
          level:
          model:
          version:

  # Training job logs
  - job_name: training-jobs
    static_configs:
      - targets:
          - localhost
        labels:
          job: training
          __path__: /var/log/training/*.log
    pipeline_stages:
      - regex:
          expression: '^.*Epoch (?P<epoch>\d+).*loss[:=]\s*(?P<loss>[\d.]+)'
      - labels:
          epoch:
      - metrics:
          training_loss:
            type: Gauge
            description: "Training loss"
            source: loss
            config:
              action: set
```

---

## 4. Structured Logging in Python

### Using Python's logging Module

**Basic Structured Logging:**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON logs"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'model_name'):
            log_data['model_name'] = record.model_name
        if hasattr(record, 'model_version'):
            log_data['model_version'] = record.model_version
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logging
def setup_logging(log_file='app.log', log_level=logging.INFO):
    """Configure structured logging"""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler with JSON formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Console handler (human-readable for development)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Usage
logger = setup_logging()

logger.info(
    "Model inference completed",
    extra={
        'model_name': 'bert-base',
        'model_version': 'v1.2',
        'batch_size': 32,
        'latency_ms': 45.2
    }
)
```

### Using python-json-logger

```bash
pip install python-json-logger
```

```python
from pythonjsonlogger import jsonlogger
import logging

# Setup
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Usage with extra fields
logger.info(
    "Processing batch",
    extra={
        'model_name': 'resnet50',
        'batch_size': 64,
        'gpu_id': 0
    }
)
```

### Advanced Logging with structlog

```bash
pip install structlog
```

```python
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Bind context that persists across log calls
logger = logger.bind(
    model_name="bert-base",
    model_version="v1.2",
    environment="production"
)

# Log with additional context
logger.info("inference_started", batch_size=32, request_id="abc123")
logger.info("inference_completed", latency_ms=45.2, predictions_count=32)
logger.error("inference_failed", error="OOM", gpu_id=0)
```

### Complete ML Application Logging Example

```python
import structlog
from fastapi import FastAPI, Request
from contextvars import ContextVar
import uuid
import time

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

# Context variable for request ID
request_id_ctx = ContextVar("request_id", default=None)

app = FastAPI()
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Add request ID to all logs"""
    request_id = str(uuid.uuid4())
    request_id_ctx.set(request_id)

    # Bind request ID to context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )

    start_time = time.time()

    logger.info("request_started")

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_seconds=duration
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "request_failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=duration
        )
        raise

@app.post("/predict")
async def predict(data: dict):
    """ML inference endpoint with comprehensive logging"""

    logger.info(
        "inference_started",
        model_name=data.get("model"),
        batch_size=len(data.get("inputs", []))
    )

    try:
        # Simulate inference
        time.sleep(0.1)

        logger.info(
            "inference_completed",
            predictions_count=len(data.get("inputs", [])),
            model_name=data.get("model")
        )

        return {"predictions": [0.95, 0.87]}

    except Exception as e:
        logger.error(
            "inference_error",
            error=str(e),
            error_type=type(e).__name__,
            model_name=data.get("model")
        )
        raise
```

---

## 5. Querying Logs

### LogQL (Loki)

LogQL is similar to PromQL for querying logs.

**Basic Queries:**

```logql
# All logs from ml-serving service
{service="ml-serving"}

# ERROR level logs
{service="ml-serving"} |= "ERROR"
{service="ml-serving"} | json | level="ERROR"

# Logs for specific model
{service="ml-serving"} | json | model_name="bert-base"

# Logs with latency > 1 second
{service="ml-serving"} | json | latency_ms > 1000

# Count error logs per minute
rate({service="ml-serving"} | json | level="ERROR"[1m])

# Extract and aggregate numeric fields
sum by (model_name) (
  rate({service="ml-serving"} | json | unwrap latency_ms [5m])
)
```

**Advanced LogQL:**

```logql
# Regex pattern matching
{service="training"} |~ "Epoch \\d+ .*loss"

# Multiple conditions
{service="ml-serving", environment="prod"}
  | json
  | model_name=~"bert.*"
  | latency_ms > 500

# Aggregate by label
sum by (model_name, model_version) (
  count_over_time({service="ml-serving"} | json [5m])
)

# Calculate percentiles
quantile_over_time(0.95,
  {service="ml-serving"}
    | json
    | unwrap latency_ms [5m]
) by (model_name)
```

### Elasticsearch Query DSL

**Basic Queries (Kibana Discover):**

```json
# Simple text search
{
  "query": {
    "match": {
      "message": "error"
    }
  }
}

# Filtered search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "level": "ERROR" } },
        { "match": { "model_name": "bert-base" } }
      ],
      "filter": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-1h"
            }
          }
        }
      ]
    }
  }
}

# Aggregation (count by model)
{
  "size": 0,
  "aggs": {
    "models": {
      "terms": {
        "field": "model_name.keyword",
        "size": 10
      }
    }
  }
}
```

---

## 6. Log Retention and Archival

### Loki Retention

```yaml
# loki-config.yml
limits_config:
  retention_period: 30d  # Keep logs for 30 days

compactor:
  working_directory: /loki/compactor
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
```

### Elasticsearch Index Lifecycle Management (ILM)

```json
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "50GB",
            "max_age": "1d"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": {
            "number_of_shards": 1
          },
          "forcemerge": {
            "max_num_segments": 1
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "freeze": {}
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

---

## Summary

In this lesson, you learned:

✅ Importance of centralized logging for AI infrastructure
✅ Setting up ELK Stack (Elasticsearch, Logstash, Kibana)
✅ Deploying Grafana Loki for cost-effective log aggregation
✅ Implementing structured logging in Python applications
✅ Querying logs with LogQL and Elasticsearch DSL
✅ Designing log retention and archival strategies
✅ Best practices for logging in ML systems

## Next Steps

- **Lesson 05**: Learn distributed tracing with Jaeger and Tempo
- **Practice**: Implement structured logging in your ML applications
- **Exercise**: Set up centralized logging for a multi-service ML platform

## Additional Resources

- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [LogQL Guide](https://grafana.com/docs/loki/latest/logql/)
- [Structlog Documentation](https://www.structlog.org/)

---

**Estimated Time:** 4-6 hours
**Difficulty:** Intermediate
**Prerequisites:** Lessons 01-03, Python, Docker
