# Lesson 03: Grafana for Visualization

## Learning Objectives
By the end of this lesson, you will be able to:
- Install and configure Grafana for AI infrastructure monitoring
- Connect Grafana to Prometheus and other data sources
- Create effective dashboards for ML system monitoring
- Design panels with appropriate visualizations
- Use Grafana variables and templates for dynamic dashboards
- Set up dashboard alerts and notifications
- Share and version control dashboards

## Prerequisites
- Completion of Lesson 02 (Prometheus for Metrics Collection)
- Understanding of Prometheus and PromQL
- Basic knowledge of data visualization concepts
- Familiarity with JSON (helpful but not required)

## Introduction

Grafana is an open-source analytics and visualization platform that transforms raw metrics into actionable insights. For AI infrastructure engineers, Grafana is essential for visualizing GPU utilization, model performance, training progress, and infrastructure health in real-time.

### Why Grafana for AI Infrastructure?

1. **Multi-source support**: Visualize data from Prometheus, Loki, Elasticsearch, and more
2. **Rich visualizations**: Time series, heatmaps, gauges, tables - perfect for ML metrics
3. **Dynamic dashboards**: Variables and templating for multi-model monitoring
4. **Alerting**: Integrated alerting with multiple notification channels
5. **Sharing**: Export, import, and version control dashboards
6. **Community**: Thousands of pre-built dashboards available

---

## 1. Installing Grafana

### Docker Installation

**docker-compose.yml (with Prometheus):**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.1.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:3000
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

**Start the stack:**
```bash
docker-compose up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin / admin123
```

### Kubernetes Installation with Helm

```bash
# Install Grafana (or use kube-prometheus-stack)
helm install grafana grafana/grafana \
  --namespace monitoring \
  --create-namespace \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set adminPassword='admin123' \
  --set service.type=LoadBalancer

# Get admin password
kubectl get secret --namespace monitoring grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access
kubectl port-forward --namespace monitoring \
  svc/grafana 3000:80
```

### Configuration Provisioning

Automate data source and dashboard setup:

**grafana/provisioning/datasources/prometheus.yml:**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: 15s
      queryTimeout: 60s
      httpMethod: POST

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false

  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    editable: false
```

**grafana/provisioning/dashboards/default.yml:**
```yaml
apiVersion: 1

providers:
  - name: 'AI Infrastructure Dashboards'
    orgId: 1
    folder: 'ML Monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

---

## 2. Connecting Data Sources

### Adding Prometheus Data Source (UI)

1. **Navigate**: Configuration → Data Sources → Add data source
2. **Select**: Prometheus
3. **Configure**:
   - **Name**: Prometheus
   - **URL**: `http://prometheus:9090` (Docker) or service URL (K8s)
   - **Access**: Server (default)
   - **Scrape interval**: 15s
4. **Save & Test**

### Data Source Configuration (Advanced)

```yaml
# Custom headers for authentication
httpHeaderName1: Authorization
httpHeaderValue1: Bearer ${PROMETHEUS_TOKEN}

# Query timeout
timeout: 60

# Custom query parameters
customQueryParameters: 'key1=value1&key2=value2'

# TLS settings
tlsAuth: true
tlsAuthWithCACert: true
```

---

## 3. Creating Dashboards

### Dashboard Fundamentals

**Dashboard Structure:**
```
Dashboard
├── Variables (filters, selectors)
├── Rows (organizational containers)
│   ├── Panel 1 (visualization)
│   ├── Panel 2 (visualization)
│   └── Panel 3 (visualization)
└── Time Range Picker
```

### Creating Your First Dashboard

**Manual Creation:**

1. **Create**: Dashboards → New Dashboard → Add visualization
2. **Select Data Source**: Choose Prometheus
3. **Query**: Enter PromQL query
4. **Visualize**: Choose panel type (Time series, Gauge, etc.)
5. **Configure**: Set title, units, thresholds
6. **Save**: Name and save dashboard

### Panel Types for ML Monitoring

| Panel Type | Use Case | Example Metric |
|-----------|----------|----------------|
| **Time Series** | Trends over time | GPU utilization, latency |
| **Gauge** | Current value vs threshold | Memory usage, temperature |
| **Stat** | Single value with sparkline | Request count, error rate |
| **Bar Gauge** | Compare multiple values | Model comparison |
| **Heatmap** | Distribution over time | Latency percentiles |
| **Table** | Structured data | Model inventory |
| **Pie Chart** | Proportions | Request distribution by model |
| **Graph (deprecated)** | Legacy time series | (Use Time Series instead) |

---

## 4. Dashboard Examples for AI Infrastructure

### Example 1: GPU Monitoring Dashboard

**JSON Dashboard Configuration:**

```json
{
  "dashboard": {
    "title": "GPU Infrastructure Monitoring",
    "tags": ["gpu", "ml", "infrastructure"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "GPU Utilization by Node",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "avg(gpu_utilization) by (node)",
            "legendFormat": "{{node}}",
            "refId": "A"
          }
        ],
        "options": {
          "legend": {"displayMode": "table", "calcs": ["mean", "max"]},
          "tooltip": {"mode": "multi"}
        },
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 70, "color": "yellow"},
                {"value": 90, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "GPU Memory Usage",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "(sum(gpu_memory_used_bytes) by (node) / sum(gpu_memory_total_bytes) by (node)) * 100",
            "legendFormat": "{{node}}",
            "refId": "A"
          }
        ],
        "options": {
          "showThresholdLabels": true,
          "showThresholdMarkers": true
        },
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 80, "color": "yellow"},
                {"value": 95, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "GPU Temperature",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "gpu_temperature_celsius",
            "legendFormat": "GPU {{gpu_id}} - {{node}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "celsius",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 75, "color": "yellow"},
                {"value": 85, "color": "red"}
              ]
            }
          }
        }
      }
    ]
  }
}
```

### Example 2: Model Serving Performance Dashboard

**Panel Queries:**

**1. Request Rate by Model:**
```promql
# Query
sum(rate(model_requests_total[5m])) by (model_name, version)

# Panel: Time Series
# Legend: {{model_name}} v{{version}}
# Y-axis: requests/sec
```

**2. P95 Latency:**
```promql
# Query
histogram_quantile(0.95,
  sum(rate(model_prediction_duration_seconds_bucket[5m])) by (model_name, le)
)

# Panel: Time Series
# Legend: {{model_name}} (P95)
# Y-axis: seconds
# Thresholds: < 0.1s (green), < 0.5s (yellow), >= 0.5s (red)
```

**3. Error Rate:**
```promql
# Query
(sum(rate(model_requests_total{status="error"}[5m])) by (model_name) /
 sum(rate(model_requests_total[5m])) by (model_name)) * 100

# Panel: Stat
# Unit: percent (0-100)
# Thresholds: < 1% (green), < 5% (yellow), >= 5% (red)
```

**4. Active Requests:**
```promql
# Query
sum(active_requests) by (model_name)

# Panel: Bar Gauge
# Orientation: Horizontal
# Display mode: Gradient
```

**5. Prediction Throughput:**
```promql
# Query
sum(rate(model_predictions_total[1m])) by (model_name)

# Panel: Stat
# Unit: predictions/sec
# Color mode: Value
```

### Example 3: Training Job Monitoring

**Panel Configuration:**

```json
{
  "panels": [
    {
      "title": "Training Loss",
      "type": "timeseries",
      "targets": [{
        "expr": "avg(training_loss) by (job_name, epoch)",
        "legendFormat": "{{job_name}} - Epoch {{epoch}}"
      }],
      "fieldConfig": {
        "defaults": {
          "custom": {
            "lineInterpolation": "smooth",
            "showPoints": "always"
          }
        }
      }
    },
    {
      "title": "Training Throughput (samples/sec)",
      "type": "stat",
      "targets": [{
        "expr": "rate(training_samples_processed[5m])"
      }],
      "options": {
        "graphMode": "area",
        "colorMode": "value"
      }
    },
    {
      "title": "GPU Utilization During Training",
      "type": "heatmap",
      "targets": [{
        "expr": "avg(gpu_utilization{job=\"training\"}) by (gpu_id)",
        "format": "time_series"
      }],
      "dataFormat": "tsbuckets"
    }
  ]
}
```

---

## 5. Dashboard Variables and Templating

Variables make dashboards dynamic and reusable.

### Creating Variables

**Query Variable (Model Selector):**

1. **Navigate**: Dashboard Settings → Variables → Add variable
2. **Configure**:
   ```
   Name: model_name
   Type: Query
   Data source: Prometheus
   Query: label_values(model_requests_total, model_name)
   Refresh: On Dashboard Load
   Multi-value: true
   Include All option: true
   ```

3. **Use in Queries**:
   ```promql
   sum(rate(model_requests_total{model_name="$model_name"}[5m]))
   ```

### Common Variable Types

**1. Query Variable (from Prometheus labels):**
```yaml
Name: node
Query: label_values(gpu_utilization, node)
Usage: gpu_utilization{node="$node"}
```

**2. Custom Variable (fixed list):**
```yaml
Name: environment
Type: Custom
Values: dev,staging,prod
Usage: model_requests_total{env="$environment"}
```

**3. Interval Variable (dynamic time ranges):**
```yaml
Name: interval
Type: Interval
Values: 1m,5m,15m,30m,1h
Auto: true
Usage: rate(metric[$interval])
```

**4. Data Source Variable:**
```yaml
Name: datasource
Type: Data source
Plugin: Prometheus
Usage: (Select in panel)
```

### Advanced Variable Examples

**Chained Variables:**

```yaml
# Variable 1: Cluster
Name: cluster
Query: label_values(kube_node_info, cluster)

# Variable 2: Namespace (filtered by cluster)
Name: namespace
Query: label_values(kube_pod_info{cluster="$cluster"}, namespace)

# Variable 3: Pod (filtered by cluster and namespace)
Name: pod
Query: label_values(kube_pod_info{cluster="$cluster", namespace="$namespace"}, pod)
```

**Usage in Dashboard:**
```promql
avg(container_cpu_usage{
  cluster="$cluster",
  namespace="$namespace",
  pod="$pod"
})
```

---

## 6. Transformations and Calculations

Grafana transformations allow you to modify query results before visualization.

### Common Transformations

**1. Merge Multiple Queries:**
```json
{
  "transformations": [
    {
      "id": "merge",
      "options": {}
    }
  ]
}
```

**2. Filter by Value:**
```json
{
  "transformations": [
    {
      "id": "filterByValue",
      "options": {
        "filters": [
          {
            "fieldName": "gpu_utilization",
            "config": {
              "id": "greater",
              "options": {
                "value": 50
              }
            }
          }
        ]
      }
    }
  ]
}
```

**3. Add Field from Calculation:**
```json
{
  "transformations": [
    {
      "id": "calculateField",
      "options": {
        "mode": "binary",
        "binary": {
          "left": "gpu_memory_used",
          "operator": "/",
          "right": "gpu_memory_total"
        },
        "alias": "memory_utilization_ratio"
      }
    }
  ]
}
```

**4. Organize Fields (Rename/Hide):**
```json
{
  "transformations": [
    {
      "id": "organize",
      "options": {
        "excludeByName": {
          "__name__": true,
          "job": true
        },
        "renameByName": {
          "model_name": "Model",
          "version": "Version"
        }
      }
    }
  ]
}
```

---

## 7. Alerting in Grafana

### Setting Up Alerts

**Alert Rule (High GPU Temperature):**

```json
{
  "alert": {
    "name": "High GPU Temperature",
    "conditions": [
      {
        "evaluator": {
          "type": "gt",
          "params": [85]
        },
        "operator": {
          "type": "and"
        },
        "query": {
          "params": ["A", "5m", "now"]
        },
        "reducer": {
          "type": "avg"
        },
        "type": "query"
      }
    ],
    "executionErrorState": "alerting",
    "frequency": "1m",
    "handler": 1,
    "message": "GPU temperature exceeded 85°C",
    "name": "High GPU Temperature",
    "noDataState": "no_data",
    "notifications": [
      {
        "uid": "slack-notifications"
      }
    ]
  }
}
```

### Notification Channels

**Slack Integration:**

```yaml
# grafana/provisioning/notifiers/slack.yml
notifiers:
  - name: Slack Alerts
    type: slack
    uid: slack-notifications
    org_id: 1
    is_default: true
    send_reminder: true
    settings:
      url: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
      recipient: '#ml-alerts'
      username: Grafana
      icon_emoji: ':grafana:'
      mentionChannel: channel
```

**Email Notification:**

```yaml
notifiers:
  - name: Email Alerts
    type: email
    uid: email-notifications
    org_id: 1
    settings:
      addresses: ml-team@company.com;ops@company.com
      singleEmail: false
```

### Alert Example: Model Performance Degradation

**Grafana UI Configuration:**

1. **Create Alert Rule**:
   - Name: `Model Latency Degradation`
   - Query: `histogram_quantile(0.95, sum(rate(model_prediction_duration_seconds_bucket[5m])) by (model_name, le))`
   - Condition: `WHEN avg() OF query(A, 5m, now) IS ABOVE 1.0`
   - State: `Alerting`

2. **Configure Notifications**:
   - Notification channel: Slack, Email
   - Message: `Model {{model_name}} P95 latency is {{ value }}s (threshold: 1.0s)`

3. **Alert Frequency**:
   - Evaluate every: `1m`
   - For: `5m` (alert after sustained condition)

---

## 8. Best Practices

### Dashboard Design

**1. Organization:**
- Group related panels in rows
- Use consistent naming conventions
- Tag dashboards appropriately
- Create folders for different teams/services

**2. Visual Hierarchy:**
- Most important metrics at the top
- Use size to indicate importance
- Consistent color schemes
- Limit panels to 10-15 per dashboard

**3. Performance:**
- Optimize queries (avoid expensive regex)
- Use recording rules for complex queries
- Set appropriate refresh intervals
- Limit time ranges for high-cardinality data

### Query Optimization

**Bad:**
```promql
# High cardinality, slow
rate(metric{user_id=~".*"}[5m])
```

**Good:**
```promql
# Aggregated, fast
sum(rate(metric[5m])) by (service)
```

### Dashboard Naming Conventions

```
[Team/Service] - [Purpose] - [Environment]

Examples:
- ML Platform - GPU Monitoring - Production
- Model Serving - BERT Performance - Staging
- Data Pipeline - Airflow Metrics - All Envs
```

---

## 9. Sharing and Version Control

### Exporting Dashboards

**JSON Export:**
```bash
# Manual: Dashboard → Share → Export → Save to file

# API Export
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  http://localhost:3000/api/dashboards/uid/gpu-monitoring \
  | jq .dashboard > gpu-dashboard.json
```

### Version Control with Git

**Repository Structure:**
```
grafana-dashboards/
├── README.md
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yml
│   └── dashboards/
│       └── default.yml
├── dashboards/
│   ├── gpu-monitoring.json
│   ├── model-serving.json
│   └── training-jobs.json
└── alerts/
    └── ml-alerts.yml
```

**Automated Deployment:**

```yaml
# .github/workflows/deploy-dashboards.yml
name: Deploy Grafana Dashboards

on:
  push:
    branches: [main]
    paths:
      - 'dashboards/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy Dashboards
        env:
          GRAFANA_URL: ${{ secrets.GRAFANA_URL }}
          GRAFANA_API_KEY: ${{ secrets.GRAFANA_API_KEY }}
        run: |
          for dashboard in dashboards/*.json; do
            curl -X POST \
              -H "Authorization: Bearer $GRAFANA_API_KEY" \
              -H "Content-Type: application/json" \
              -d @$dashboard \
              $GRAFANA_URL/api/dashboards/db
          done
```

---

## 10. Complete Dashboard Example

**ML Platform Overview Dashboard (JSON):**

```json
{
  "dashboard": {
    "title": "ML Platform - Overview",
    "tags": ["ml", "overview", "production"],
    "timezone": "browser",
    "editable": true,
    "graphTooltip": 1,
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "30s",

    "templating": {
      "list": [
        {
          "name": "model_name",
          "type": "query",
          "datasource": "Prometheus",
          "query": "label_values(model_requests_total, model_name)",
          "refresh": 1,
          "multi": true,
          "includeAll": true
        },
        {
          "name": "interval",
          "type": "interval",
          "auto": true,
          "auto_count": 30,
          "options": [
            {"text": "1m", "value": "1m"},
            {"text": "5m", "value": "5m"},
            {"text": "15m", "value": "15m"}
          ]
        }
      ]
    },

    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [{
          "expr": "sum(rate(model_requests_total{model_name=~\"$model_name\"}[$interval])) by (model_name)",
          "legendFormat": "{{model_name}}"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "id": 2,
        "title": "P95 Latency",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(model_prediction_duration_seconds_bucket{model_name=~\"$model_name\"}[$interval])) by (model_name, le))",
          "legendFormat": "{{model_name}}"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.5, "color": "yellow"},
                {"value": 1.0, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Error Rate %",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
        "targets": [{
          "expr": "(sum(rate(model_requests_total{model_name=~\"$model_name\", status=\"error\"}[$interval])) / sum(rate(model_requests_total{model_name=~\"$model_name\"}[$interval]))) * 100"
        }],
        "options": {
          "graphMode": "area",
          "colorMode": "value"
        },
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 1, "color": "yellow"},
                {"value": 5, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "GPU Utilization",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
        "targets": [{
          "expr": "avg(gpu_utilization)"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        }
      }
    ]
  }
}
```

---

## Summary

In this lesson, you learned:

✅ Installing and configuring Grafana for AI infrastructure
✅ Connecting Prometheus and other data sources
✅ Creating dashboards with various panel types
✅ Using variables and templating for dynamic dashboards
✅ Setting up alerts and notifications
✅ Dashboard best practices and optimization
✅ Version controlling dashboards with Git
✅ Complete examples for GPU, model serving, and training monitoring

## Next Steps

- **Lesson 04**: Learn about logging with ELK Stack and Loki
- **Practice**: Create dashboards for your ML infrastructure
- **Exercise**: Build a comprehensive model serving dashboard

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Pre-built Dashboards](https://grafana.com/grafana/dashboards/)
- [Grafana Community](https://community.grafana.com/)

---

**Estimated Time:** 3-5 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 02, Prometheus basics
