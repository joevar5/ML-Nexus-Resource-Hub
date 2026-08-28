# Exercise 04: Kubernetes Cluster Autoscaler with Custom Metrics

**Estimated Time**: 35-43 hours
**Difficulty**: Advanced
**Prerequisites**: Kubernetes, Python 3.9+, Helm, Prometheus, Custom Metrics API

## Overview

Build a production-grade cluster autoscaling system that scales Kubernetes nodes based on custom metrics (GPU utilization, model inference queue depth, cost optimization). Implement intelligent scaling decisions using Horizontal Pod Autoscaler (HPA), Vertical Pod Autoscaler (VPA), and Cluster Autoscaler with custom metric providers. This exercise teaches advanced Kubernetes autoscaling patterns essential for cost-efficient ML infrastructure.

In production ML infrastructure, intelligent autoscaling is critical for:
- **Cost Optimization**: Scale down during low traffic (save 60-70% on compute costs)
- **Performance**: Scale up proactively before queue depths increase
- **GPU Efficiency**: Pack GPU workloads to maximize utilization
- **SLA Compliance**: Maintain p95 latency <100ms during traffic spikes
- **Multi-Tenancy**: Fair resource allocation across teams/models

## Learning Objectives

By completing this exercise, you will:

1. **Implement HPA with custom metrics** from Prometheus
2. **Deploy VPA** for right-sizing pod resource requests
3. **Configure Cluster Autoscaler** for node scaling
4. **Build custom metrics provider** exposing ML-specific metrics
5. **Implement predictive autoscaling** using time-series forecasting
6. **Optimize scaling decisions** balancing cost and performance
7. **Handle GPU node scaling** with constraints (zone availability, quotas)

## Business Context

**Real-World Scenario**: Your ML platform serves 50 models across 3 Kubernetes clusters (dev, staging, prod). Current issues:

- **High costs**: Production cluster runs 20 GPU nodes 24/7 ($14,400/day), but only needed 12-16 hours/day
- **Slow scale-up**: Traffic spikes cause 5-minute pod pending times (Cluster Autoscaler takes 3-4 min to provision nodes)
- **Wasted capacity**: Pods request 4 CPU/8Gi but use 1.5 CPU/3Gi (62% waste)
- **GPU underutilization**: Some GPU pods use 30% GPU while requesting 100%
- **Manual intervention**: Ops team manually scales before known traffic spikes (Black Friday, product launches)

Your task: Build an autoscaler that:
- Scales based on inference queue depth (target: <10 requests waiting)
- Predicts traffic spikes 15 minutes ahead, scales proactively
- Right-sizes pod requests to reduce waste by 40%+
- Prioritizes spot instances to reduce costs by 60%
- Handles GPU-specific constraints (nvidia.com/gpu taints, quotas)

## Project Structure

```
exercise-04-k8s-cluster-autoscaler/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Poetry config
├── helm/
│   └── custom-metrics-adapter/        # Helm chart for metrics adapter
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── apiservice.yaml
│           └── rbac.yaml
├── kubernetes/
│   ├── hpa/
│   │   ├── cpu-based-hpa.yaml         # Standard CPU-based HPA
│   │   ├── custom-metric-hpa.yaml     # HPA using custom metrics
│   │   └── external-metric-hpa.yaml   # HPA using external metrics (SQS queue)
│   ├── vpa/
│   │   ├── vpa-recommender.yaml       # VPA in recommendation mode
│   │   └── vpa-auto.yaml              # VPA in auto mode
│   ├── cluster-autoscaler/
│   │   ├── deployment.yaml            # Cluster Autoscaler deployment
│   │   └── configmap.yaml             # CA configuration
│   └── priorities/
│       ├── priority-classes.yaml      # Pod priority classes
│       └── pod-disruption-budgets.yaml
├── src/
│   └── autoscaler/
│       ├── __init__.py
│       ├── metrics_provider.py        # Custom metrics API provider
│       ├── metrics_collector.py       # Collect metrics from Prometheus
│       ├── predictor.py               # Time-series forecasting
│       ├── optimizer.py               # Scaling decision optimization
│       ├── cost_calculator.py         # Cost-aware scaling
│       ├── gpu_scheduler.py           # GPU-aware scheduling
│       └── cli.py                     # CLI for testing
├── tests/
│   ├── test_metrics_provider.py
│   ├── test_predictor.py
│   ├── test_optimizer.py
│   ├── test_cost_calculator.py
│   └── fixtures/
│       └── sample_metrics.json
├── examples/
│   ├── deploy_ml_service_with_hpa.yaml
│   ├── test_scale_up.sh
│   └── test_scale_down.sh
├── benchmarks/
│   ├── scale_up_latency.sh            # Measure time to scale
│   ├── cost_comparison.py             # Compare costs vs baseline
│   └── load_test.py                   # Generate traffic for testing
└── docs/
    ├── DESIGN.md                      # Architecture decisions
    ├── METRICS.md                     # Custom metrics reference
    └── TROUBLESHOOTING.md             # Common issues
```

## Requirements

### Functional Requirements

Your autoscaling system must:

1. **HPA (Horizontal Pod Autoscaler)**:
   - Scale based on CPU/memory utilization
   - Scale based on custom metrics (requests_per_second, queue_depth)
   - Scale based on external metrics (SQS queue length, Pub/Sub backlog)
   - Respect min/max replicas and scaling policies

2. **VPA (Vertical Pod Autoscaler)**:
   - Recommend resource requests based on actual usage
   - Auto-update pod requests (with pod restarts)
   - Prevent over/under-provisioning

3. **Cluster Autoscaler**:
   - Add nodes when pods are pending due to insufficient resources
   - Remove underutilized nodes (<50% utilization for 10+ minutes)
   - Respect node pool constraints (min/max nodes, instance types)
   - Handle multiple node pools (CPU-only, GPU, spot instances)

4. **Custom Metrics Provider**:
   - Expose ML-specific metrics via Kubernetes Custom Metrics API
   - Aggregate metrics from Prometheus
   - Support metrics like: inference_queue_depth, gpu_utilization, model_latency_p95

5. **Predictive Scaling**:
   - Forecast traffic 15 minutes ahead using time-series models
   - Trigger scale-up before traffic arrives
   - Handle daily/weekly patterns

6. **Cost Optimization**:
   - Prefer spot instances when possible
   - Calculate cost per request
   - Recommend optimal instance types

### Non-Functional Requirements

- **Scale-up Latency**: <90 seconds from metric breach to pods running
- **Scale-down Safety**: Don't scale down too aggressively (avoid flapping)
- **Accuracy**: Predictive scaling accuracy >80%
- **Cost Savings**: Achieve 40-60% cost reduction vs. static provisioning

## Implementation Tasks

### Task 1: Custom Metrics Provider (7-9 hours)

Build a metrics provider that exposes custom metrics to Kubernetes.

```python
# src/autoscaler/metrics_provider.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import prometheus_client
from kubernetes import client, config

@dataclass
class CustomMetric:
    """Represents a custom metric for HPA"""
    name: str  # inference_queue_depth
    namespace: str
    target_name: str  # deployment/model-server
    target_kind: str  # Deployment, StatefulSet
    value: float
    timestamp: datetime

class MetricsProvider:
    """
    Implements Kubernetes Custom Metrics API

    Exposes metrics that HPA can use for scaling decisions.
    Integrates with Prometheus to fetch actual metric values.
    """

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        config.load_kube_config()
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()

    def get_metric(
        self,
        namespace: str,
        metric_name: str,
        target_name: str,
        target_kind: str = "Deployment"
    ) -> Optional[CustomMetric]:
        """
        Get custom metric value for a Kubernetes resource

        Example HPA usage:
        ```yaml
        metrics:
        - type: Object
          object:
            metric:
              name: inference_queue_depth
            describedObject:
              apiVersion: apps/v1
              kind: Deployment
              name: model-server
            target:
              type: Value
              value: "10"  # Scale when queue > 10
        ```

        Implementation:
        1. Fetch metric from Prometheus
        2. Aggregate across all pods in deployment
        3. Return current value
        """
        # TODO: Build Prometheus query
        # Example: avg(inference_queue_depth{namespace="prod", deployment="model-server"})
        query = self._build_prometheus_query(namespace, metric_name, target_name)

        # TODO: Execute query against Prometheus
        result = self._query_prometheus(query)

        # TODO: Parse result and return CustomMetric
        if result:
            return CustomMetric(
                name=metric_name,
                namespace=namespace,
                target_name=target_name,
                target_kind=target_kind,
                value=float(result),
                timestamp=datetime.now()
            )
        return None

    def list_metrics(self, namespace: str) -> List[str]:
        """
        List all available custom metrics in namespace

        Returns metric names that HPA can query
        """
        # TODO: Query Prometheus for available metrics
        # TODO: Filter to metrics relevant for autoscaling
        raise NotImplementedError

    def _build_prometheus_query(
        self,
        namespace: str,
        metric_name: str,
        target_name: str
    ) -> str:
        """
        Build PromQL query for metric

        Example queries:
        - inference_queue_depth: avg(inference_queue_depth{deployment="model-server"})
        - gpu_utilization: avg(DCGM_FI_DEV_GPU_UTIL{pod=~"model-server-.*"})
        - requests_per_second: rate(http_requests_total{deployment="model-server"}[1m])
        """
        # TODO: Construct PromQL query based on metric type
        raise NotImplementedError

    def _query_prometheus(self, query: str) -> Optional[float]:
        """Execute Prometheus query and return scalar result"""
        import requests

        # TODO: Send query to Prometheus HTTP API
        # POST /api/v1/query with query parameter
        response = requests.post(
            f"{self.prometheus_url}/api/v1/query",
            data={"query": query}
        )

        # TODO: Parse JSON response
        # Extract value from result[0].value[1]
        if response.status_code == 200:
            data = response.json()
            if data.get("data", {}).get("result"):
                return float(data["data"]["result"][0]["value"][1])
        return None
```

```python
# src/autoscaler/metrics_collector.py

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta
import requests

@dataclass
class MetricSample:
    """Time-series metric sample"""
    timestamp: datetime
    value: float

class MetricsCollector:
    """Collect and aggregate metrics from Prometheus"""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def get_time_series(
        self,
        metric_name: str,
        labels: Dict[str, str],
        duration: timedelta,
        step: str = "1m"
    ) -> List[MetricSample]:
        """
        Get time-series data for metric

        Args:
            metric_name: Prometheus metric name
            labels: Label filters (e.g., {"deployment": "model-server"})
            duration: How far back to query
            step: Sample interval (1m, 5m, etc.)

        Returns:
            List of samples with timestamps and values
        """
        # TODO: Build Prometheus range query
        # GET /api/v1/query_range?query=...&start=...&end=...&step=...

        end_time = datetime.now()
        start_time = end_time - duration

        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        query = f"{metric_name}{{{label_str}}}"

        # TODO: Execute query
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": step
            }
        )

        # TODO: Parse results into MetricSample objects
        raise NotImplementedError

    def get_current_pod_metrics(self, namespace: str, deployment: str) -> Dict[str, float]:
        """
        Get current resource metrics for all pods in deployment

        Returns:
            {
                "cpu_usage_cores": 2.5,
                "memory_usage_bytes": 4294967296,
                "gpu_utilization_percent": 85.0,
                "requests_per_second": 450.0
            }
        """
        # TODO: Query Prometheus for pod metrics
        # TODO: Aggregate across all pods
        raise NotImplementedError

    def get_node_metrics(self) -> List[Dict]:
        """
        Get metrics for all nodes

        Returns:
            [
                {
                    "node": "node-1",
                    "cpu_utilization": 0.45,
                    "memory_utilization": 0.62,
                    "gpu_count": 1,
                    "gpu_utilization": 0.85
                }
            ]
        """
        # TODO: Query node-level metrics
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Expose custom metrics via Kubernetes API
- ✅ Fetch metrics from Prometheus
- ✅ Support time-series queries
- ✅ HPA can scale based on custom metrics
- ✅ Handle multiple metrics per deployment

---

### Task 2: Time-Series Predictor (6-8 hours)

Implement predictive autoscaling using time-series forecasting.

```python
# src/autoscaler/predictor.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from .metrics_collector import MetricSample

@dataclass
class Forecast:
    """Predicted metric value"""
    timestamp: datetime
    predicted_value: float
    confidence_interval: tuple  # (lower, upper)
    model_accuracy: float  # R² score

class TimeSeriesPredictor:
    """
    Predict future metric values for proactive scaling

    Strategies:
    1. Simple moving average (baseline)
    2. Linear regression on recent trend
    3. Seasonal decomposition (daily/weekly patterns)
    4. ARIMA model (advanced)
    """

    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours

    def predict(
        self,
        historical_data: List[MetricSample],
        horizon: timedelta = timedelta(minutes=15)
    ) -> Forecast:
        """
        Predict metric value N minutes into the future

        Args:
            historical_data: Past metric samples
            horizon: How far ahead to predict

        Returns:
            Forecast with predicted value and confidence
        """
        if len(historical_data) < 10:
            raise ValueError("Need at least 10 samples for prediction")

        # TODO: Extract features and target values
        X, y = self._prepare_data(historical_data)

        # TODO: Train regression model
        model = LinearRegression()
        model.fit(X, y)

        # TODO: Predict future value
        future_timestamp = datetime.now() + horizon
        future_features = self._extract_features(future_timestamp)
        predicted_value = model.predict([future_features])[0]

        # TODO: Calculate confidence interval
        # Use residuals to estimate uncertainty
        residuals = y - model.predict(X)
        std_error = np.std(residuals)
        confidence = (
            predicted_value - 2 * std_error,
            predicted_value + 2 * std_error
        )

        # TODO: Calculate model accuracy (R²)
        accuracy = model.score(X, y)

        return Forecast(
            timestamp=future_timestamp,
            predicted_value=max(0, predicted_value),  # Can't be negative
            confidence_interval=confidence,
            model_accuracy=accuracy
        )

    def detect_pattern(self, historical_data: List[MetricSample]) -> str:
        """
        Detect traffic pattern (constant, trending, seasonal)

        Returns: "constant", "increasing", "decreasing", "daily_seasonal", "weekly_seasonal"
        """
        # TODO: Analyze variance, trend, seasonality
        # TODO: Use autocorrelation to detect patterns
        raise NotImplementedError

    def _prepare_data(self, samples: List[MetricSample]) -> tuple:
        """
        Prepare data for training

        Extract features:
        - Hour of day (0-23)
        - Day of week (0-6)
        - Time since start (minutes)
        - Recent trend (slope of last 10 points)

        Target: metric value
        """
        X = []
        y = []

        for i, sample in enumerate(samples):
            features = self._extract_features(sample.timestamp, samples[:i])
            X.append(features)
            y.append(sample.value)

        return np.array(X), np.array(y)

    def _extract_features(
        self,
        timestamp: datetime,
        historical_samples: Optional[List[MetricSample]] = None
    ) -> List[float]:
        """
        Extract features from timestamp

        Features:
        - hour_of_day: 0-23 (capture daily patterns)
        - day_of_week: 0-6 (capture weekly patterns)
        - is_weekend: 0/1
        - minutes_since_midnight: 0-1439
        - recent_trend: slope of last 10 samples (if available)
        """
        features = [
            timestamp.hour,
            timestamp.weekday(),
            1 if timestamp.weekday() >= 5 else 0,  # is_weekend
            timestamp.hour * 60 + timestamp.minute,
        ]

        # Calculate recent trend if historical data available
        if historical_samples and len(historical_samples) >= 10:
            recent = historical_samples[-10:]
            values = [s.value for s in recent]
            trend = (values[-1] - values[0]) / 10  # Avg change per sample
            features.append(trend)
        else:
            features.append(0)

        return features
```

**Acceptance Criteria**:
- ✅ Predict metric values 15 minutes ahead
- ✅ Detect daily/weekly patterns
- ✅ Provide confidence intervals
- ✅ Accuracy >80% for stable workloads
- ✅ Handle sudden spikes gracefully

---

### Task 3: Scaling Optimizer (7-9 hours)

Implement intelligent scaling decision logic.

```python
# src/autoscaler/optimizer.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

class ScaleDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"

@dataclass
class ScalingDecision:
    """Decision to scale pods or nodes"""
    direction: ScaleDirection
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float  # 0.0-1.0
    estimated_cost_impact: float  # USD per hour
    estimated_latency_impact: float  # milliseconds (p95)

@dataclass
class ScalingPolicy:
    """Rules for scaling behavior"""
    min_replicas: int = 1
    max_replicas: int = 100
    target_metric_value: float = 80.0  # Target utilization %
    scale_up_threshold: float = 0.8  # Scale up if >80% of target
    scale_down_threshold: float = 0.5  # Scale down if <50% of target
    scale_up_cooldown: timedelta = timedelta(minutes=3)
    scale_down_cooldown: timedelta = timedelta(minutes=10)
    scale_up_factor: float = 1.5  # Multiply by 1.5 when scaling up
    scale_down_factor: float = 0.7  # Multiply by 0.7 when scaling down

class ScalingOptimizer:
    """Make intelligent scaling decisions"""

    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.last_scale_up: Optional[datetime] = None
        self.last_scale_down: Optional[datetime] = None

    def decide(
        self,
        current_metric: float,
        predicted_metric: Optional[float],
        current_replicas: int,
        current_latency_p95: float
    ) -> ScalingDecision:
        """
        Decide whether to scale and by how much

        Decision logic:
        1. Check if in cooldown period
        2. Compare current metric vs target
        3. Consider predicted future metric (proactive scaling)
        4. Calculate target replicas
        5. Respect min/max bounds
        6. Estimate cost and latency impact

        Args:
            current_metric: Current metric value (e.g., CPU utilization %)
            predicted_metric: Predicted metric value in 15 minutes
            current_replicas: Current pod count
            current_latency_p95: Current p95 latency in ms

        Returns:
            ScalingDecision
        """
        now = datetime.now()

        # TODO: Check cooldown
        if self._in_cooldown(now, ScaleDirection.UP):
            return ScalingDecision(
                direction=ScaleDirection.NONE,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                reason="In scale-up cooldown period",
                confidence=1.0,
                estimated_cost_impact=0,
                estimated_latency_impact=0
            )

        # TODO: Determine if scaling needed
        # Use predicted metric if available and higher than current
        metric_to_use = current_metric
        use_prediction = False
        if predicted_metric and predicted_metric > current_metric * 1.2:
            metric_to_use = predicted_metric
            use_prediction = True

        # TODO: Calculate utilization ratio
        utilization_ratio = metric_to_use / self.policy.target_metric_value

        # TODO: Determine direction
        direction = ScaleDirection.NONE
        if utilization_ratio > self.policy.scale_up_threshold:
            direction = ScaleDirection.UP
        elif utilization_ratio < self.policy.scale_down_threshold:
            if not self._in_cooldown(now, ScaleDirection.DOWN):
                direction = ScaleDirection.DOWN

        # TODO: Calculate target replicas
        if direction == ScaleDirection.UP:
            # Scale up by factor or proportionally to utilization
            target_replicas = int(current_replicas * max(
                self.policy.scale_up_factor,
                utilization_ratio
            ))
            target_replicas = min(target_replicas, self.policy.max_replicas)
            self.last_scale_up = now

        elif direction == ScaleDirection.DOWN:
            target_replicas = int(current_replicas * self.policy.scale_down_factor)
            target_replicas = max(target_replicas, self.policy.min_replicas)
            self.last_scale_down = now

        else:
            target_replicas = current_replicas

        # TODO: Estimate impacts
        cost_impact = self._estimate_cost_impact(current_replicas, target_replicas)
        latency_impact = self._estimate_latency_impact(
            current_replicas,
            target_replicas,
            current_latency_p95
        )

        # TODO: Calculate confidence
        confidence = 0.9 if use_prediction else 1.0

        reason = self._generate_reason(
            direction,
            metric_to_use,
            utilization_ratio,
            use_prediction
        )

        return ScalingDecision(
            direction=direction,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            reason=reason,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_latency_impact=latency_impact
        )

    def _in_cooldown(self, now: datetime, direction: ScaleDirection) -> bool:
        """Check if in cooldown period for given direction"""
        if direction == ScaleDirection.UP:
            if self.last_scale_up:
                elapsed = now - self.last_scale_up
                return elapsed < self.policy.scale_up_cooldown
        elif direction == ScaleDirection.DOWN:
            if self.last_scale_down:
                elapsed = now - self.last_scale_down
                return elapsed < self.policy.scale_down_cooldown
        return False

    def _estimate_cost_impact(self, current: int, target: int) -> float:
        """
        Estimate hourly cost change

        Assume: $0.10 per pod per hour (adjust based on instance type)
        """
        pod_cost_per_hour = 0.10
        delta_replicas = target - current
        return delta_replicas * pod_cost_per_hour

    def _estimate_latency_impact(
        self,
        current: int,
        target: int,
        current_latency: float
    ) -> float:
        """
        Estimate latency change

        Simple model: latency inversely proportional to replicas
        latency_new = latency_current * (current_replicas / target_replicas)
        """
        if target == 0:
            return float('inf')
        ratio = current / target
        new_latency = current_latency * ratio
        return new_latency - current_latency

    def _generate_reason(
        self,
        direction: ScaleDirection,
        metric: float,
        ratio: float,
        predicted: bool
    ) -> str:
        """Generate human-readable scaling reason"""
        pred_str = " (predicted)" if predicted else ""
        if direction == ScaleDirection.UP:
            return f"Metric{pred_str} {metric:.1f} exceeds target (utilization: {ratio:.1%})"
        elif direction == ScaleDirection.DOWN:
            return f"Metric{pred_str} {metric:.1f} below target (utilization: {ratio:.1%})"
        else:
            return f"Metric{pred_str} {metric:.1f} within acceptable range"
```

**Acceptance Criteria**:
- ✅ Make scaling decisions based on metrics
- ✅ Respect cooldown periods
- ✅ Use predictions for proactive scaling
- ✅ Estimate cost and latency impacts
- ✅ Prevent flapping (rapid scale up/down)

---

### Task 4: Cost Calculator (5-6 hours)

Implement cost-aware scaling with spot instance support.

```python
# src/autoscaler/cost_calculator.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class InstanceType(Enum):
    """Kubernetes node instance types"""
    CPU_SMALL = "t3.large"  # 2 vCPU, 8 GB
    CPU_MEDIUM = "t3.xlarge"  # 4 vCPU, 16 GB
    CPU_LARGE = "t3.2xlarge"  # 8 vCPU, 32 GB
    GPU_T4 = "g4dn.xlarge"  # 4 vCPU, 16 GB, 1x T4
    GPU_V100 = "p3.2xlarge"  # 8 vCPU, 61 GB, 1x V100
    GPU_A10 = "g5.xlarge"  # 4 vCPU, 16 GB, 1x A10

@dataclass
class InstanceCost:
    """Cost information for instance type"""
    instance_type: str
    on_demand_price: float  # USD per hour
    spot_price: float  # USD per hour (average)
    vcpus: int
    memory_gb: float
    gpus: int

@dataclass
class CostOptimizationRecommendation:
    """Recommendation to optimize costs"""
    current_cost_per_hour: float
    optimized_cost_per_hour: float
    savings_per_hour: float
    savings_percent: float
    actions: List[str]  # ["Use spot instances", "Right-size pods", etc.]

class CostCalculator:
    """Calculate and optimize infrastructure costs"""

    # Pricing data (update with actual prices)
    INSTANCE_PRICING: Dict[str, InstanceCost] = {
        "t3.large": InstanceCost("t3.large", 0.0832, 0.025, 2, 8, 0),
        "t3.xlarge": InstanceCost("t3.xlarge", 0.1664, 0.050, 4, 16, 0),
        "t3.2xlarge": InstanceCost("t3.2xlarge", 0.3328, 0.100, 8, 32, 0),
        "g4dn.xlarge": InstanceCost("g4dn.xlarge", 0.526, 0.158, 4, 16, 1),
        "p3.2xlarge": InstanceCost("p3.2xlarge", 3.06, 0.918, 8, 61, 1),
        "g5.xlarge": InstanceCost("g5.xlarge", 1.006, 0.302, 4, 16, 1),
    }

    def calculate_cluster_cost(
        self,
        node_counts: Dict[str, int],
        use_spot: bool = False
    ) -> float:
        """
        Calculate total cluster cost per hour

        Args:
            node_counts: {"t3.xlarge": 5, "g4dn.xlarge": 2}
            use_spot: Use spot instance pricing

        Returns:
            Total cost in USD per hour
        """
        total_cost = 0.0
        for instance_type, count in node_counts.items():
            pricing = self.INSTANCE_PRICING.get(instance_type)
            if not pricing:
                continue

            price = pricing.spot_price if use_spot else pricing.on_demand_price
            total_cost += price * count

        return total_cost

    def recommend_optimization(
        self,
        current_nodes: Dict[str, int],
        pod_requirements: List[Dict],
        allow_spot: bool = True
    ) -> CostOptimizationRecommendation:
        """
        Recommend cost optimizations

        Args:
            current_nodes: Current node counts by type
            pod_requirements: [{"cpu": 2, "memory_gb": 4, "gpu": 0}, ...]
            allow_spot: Allow spot instance recommendations

        Returns:
            Optimization recommendations
        """
        # TODO: Calculate current cost
        current_cost = self.calculate_cluster_cost(current_nodes, use_spot=False)

        # TODO: Calculate optimal node configuration
        # Use bin-packing algorithm to minimize nodes needed
        optimal_nodes = self._bin_pack_pods(pod_requirements)

        # TODO: Calculate optimized cost
        optimized_cost = self.calculate_cluster_cost(optimal_nodes, use_spot=allow_spot)

        # TODO: Generate actionable recommendations
        actions = []
        if allow_spot:
            actions.append("Use spot instances (70% savings)")
        if sum(optimal_nodes.values()) < sum(current_nodes.values()):
            actions.append(f"Reduce nodes from {sum(current_nodes.values())} to {sum(optimal_nodes.values())}")

        return CostOptimizationRecommendation(
            current_cost_per_hour=current_cost,
            optimized_cost_per_hour=optimized_cost,
            savings_per_hour=current_cost - optimized_cost,
            savings_percent=((current_cost - optimized_cost) / current_cost * 100) if current_cost > 0 else 0,
            actions=actions
        )

    def _bin_pack_pods(self, pod_requirements: List[Dict]) -> Dict[str, int]:
        """
        Find optimal node configuration using bin-packing

        Simple greedy algorithm:
        1. Sort pods by resource requirements (largest first)
        2. Try to fit each pod on existing nodes
        3. Add new node if pod doesn't fit

        Returns:
            Optimal node counts by instance type
        """
        # TODO: Implement bin-packing algorithm
        # TODO: Consider CPU, memory, GPU constraints
        # TODO: Return node counts
        raise NotImplementedError

    def estimate_monthly_cost(
        self,
        hourly_cost: float,
        utilization_pattern: Optional[List[float]] = None
    ) -> float:
        """
        Estimate monthly cost with variable utilization

        Args:
            hourly_cost: Cost per hour at full capacity
            utilization_pattern: List of 24 hourly utilization values (0.0-1.0)

        Returns:
            Estimated monthly cost in USD
        """
        if utilization_pattern:
            # Average utilization across day
            avg_utilization = sum(utilization_pattern) / len(utilization_pattern)
            daily_cost = hourly_cost * avg_utilization * 24
        else:
            # Assume constant utilization
            daily_cost = hourly_cost * 24

        monthly_cost = daily_cost * 30
        return monthly_cost
```

**Acceptance Criteria**:
- ✅ Calculate cluster costs accurately
- ✅ Recommend cost optimizations
- ✅ Support spot instance pricing
- ✅ Implement bin-packing for pod placement
- ✅ Estimate monthly costs with utilization patterns

---

### Task 5: GPU Scheduler (4-5 hours)

Implement GPU-aware scheduling logic.

```python
# src/autoscaler/gpu_scheduler.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from kubernetes import client

@dataclass
class GPUNode:
    """Node with GPU resources"""
    name: str
    gpu_count: int
    gpu_type: str  # "nvidia-tesla-t4", "nvidia-tesla-v100"
    gpus_allocated: int
    gpus_available: int
    is_spot: bool

@dataclass
class GPUPodRequest:
    """Pod requesting GPU resources"""
    pod_name: str
    namespace: str
    gpu_count: int
    gpu_type: Optional[str] = None  # Specific GPU type or None
    cpu_request: float = 1.0
    memory_gb: float = 4.0

class GPUScheduler:
    """Handle GPU-specific scheduling constraints"""

    def __init__(self):
        config.load_kube_config()
        self.k8s_core = client.CoreV1Api()

    def get_gpu_nodes(self) -> List[GPUNode]:
        """
        List all GPU nodes in cluster

        Identifies nodes with nvidia.com/gpu capacity
        """
        # TODO: List nodes
        nodes = self.k8s_core.list_node()

        gpu_nodes = []
        for node in nodes.items:
            # TODO: Check if node has GPUs
            capacity = node.status.capacity
            allocatable = node.status.allocatable

            if "nvidia.com/gpu" in capacity:
                gpu_count = int(capacity["nvidia.com/gpu"])
                gpus_available = int(allocatable["nvidia.com/gpu"])

                # TODO: Determine GPU type from node labels
                gpu_type = node.metadata.labels.get("gpu-type", "unknown")

                # TODO: Check if spot instance
                is_spot = "spot" in node.metadata.labels.get("node-lifecycle", "")

                gpu_nodes.append(GPUNode(
                    name=node.metadata.name,
                    gpu_count=gpu_count,
                    gpu_type=gpu_type,
                    gpus_allocated=gpu_count - gpus_available,
                    gpus_available=gpus_available,
                    is_spot=is_spot
                ))

        return gpu_nodes

    def find_node_for_pod(
        self,
        pod_request: GPUPodRequest,
        prefer_spot: bool = True
    ) -> Optional[GPUNode]:
        """
        Find best node for GPU pod

        Priority:
        1. Node with requested GPU type (if specified)
        2. Node with available GPUs
        3. Spot instances (if prefer_spot)
        4. Node with fewest free GPUs (bin-packing)

        Returns None if no suitable node found
        """
        # TODO: Get all GPU nodes
        gpu_nodes = self.get_gpu_nodes()

        # TODO: Filter by GPU type if specified
        if pod_request.gpu_type:
            gpu_nodes = [n for n in gpu_nodes if n.gpu_type == pod_request.gpu_type]

        # TODO: Filter by available GPUs
        gpu_nodes = [n for n in gpu_nodes if n.gpus_available >= pod_request.gpu_count]

        if not gpu_nodes:
            return None

        # TODO: Sort by preference
        # Prefer spot, then fewest available GPUs (bin-packing)
        gpu_nodes.sort(
            key=lambda n: (
                not n.is_spot if prefer_spot else n.is_spot,
                n.gpus_available
            )
        )

        return gpu_nodes[0]

    def calculate_gpu_utilization(self) -> Dict[str, float]:
        """
        Calculate GPU utilization across cluster

        Returns:
            {
                "node-1": 0.85,  # 85% utilized
                "node-2": 0.50
            }
        """
        # TODO: Get GPU metrics from Prometheus (DCGM_FI_DEV_GPU_UTIL)
        # TODO: Average per node
        raise NotImplementedError

    def should_scale_gpu_nodes(
        self,
        pending_gpu_pods: int,
        average_utilization: float
    ) -> bool:
        """
        Decide if GPU nodes should be added

        Scale up if:
        - Pods are pending due to insufficient GPUs
        - Average GPU utilization > 80%

        Returns:
            True if should add GPU nodes
        """
        return pending_gpu_pods > 0 or average_utilization > 0.80
```

**Acceptance Criteria**:
- ✅ List GPU nodes with availability
- ✅ Find optimal node for GPU pod
- ✅ Calculate GPU utilization
- ✅ Recommend GPU node scaling
- ✅ Support GPU type constraints (T4, V100, A10)

---

### Task 6: Integration and CLI (6-7 hours)

Integrate all components and build CLI.

```python
# src/autoscaler/cli.py

import click
from pathlib import Path
from .metrics_provider import MetricsProvider
from .metrics_collector import MetricsCollector
from .predictor import TimeSeriesPredictor
from .optimizer import ScalingOptimizer, ScalingPolicy
from .cost_calculator import CostCalculator
from .gpu_scheduler import GPUScheduler

@click.group()
def cli():
    """Kubernetes Cluster Autoscaler with Custom Metrics"""
    pass

@cli.command()
@click.option('--prometheus-url', default='http://prometheus:9090')
@click.option('--namespace', default='default')
@click.option('--deployment', required=True)
def analyze(prometheus_url: str, namespace: str, deployment: str):
    """Analyze current metrics and scaling decisions"""
    # TODO: Collect current metrics
    # TODO: Get predictions
    # TODO: Show scaling recommendations
    pass

@cli.command()
@click.option('--prometheus-url', default='http://prometheus:9090')
@click.option('--metric', required=True, help='Metric to forecast')
@click.option('--namespace', default='default')
@click.option('--deployment', required=True)
@click.option('--horizon', default=15, help='Minutes to forecast ahead')
def predict(prometheus_url: str, metric: str, namespace: str, deployment: str, horizon: int):
    """Predict future metric values"""
    # TODO: Collect historical data
    # TODO: Run prediction
    # TODO: Display forecast with confidence interval
    pass

@cli.command()
def cost():
    """Analyze cluster costs and optimization opportunities"""
    # TODO: Get current node counts
    # TODO: Calculate costs
    # TODO: Show optimization recommendations
    pass

@cli.command()
def gpu():
    """Show GPU node status and utilization"""
    # TODO: List GPU nodes
    # TODO: Show utilization
    # TODO: Recommend scaling
    pass

@cli.command()
@click.option('--namespace', default='default')
@click.option('--deployment', required=True)
@click.option('--metric', required=True)
@click.option('--target-value', type=float, required=True)
@click.option('--min-replicas', type=int, default=1)
@click.option('--max-replicas', type=int, default=10)
def create_hpa(namespace: str, deployment: str, metric: str, target_value: float,
               min_replicas: int, max_replicas: int):
    """Create HPA with custom metric"""
    # TODO: Generate HPA YAML
    # TODO: Apply to cluster
    pass

if __name__ == '__main__':
    cli()
```

**Acceptance Criteria**:
- ✅ CLI commands work end-to-end
- ✅ Integration with Prometheus
- ✅ Create HPAs programmatically
- ✅ Display cost analysis
- ✅ Show GPU status

---

## Testing Requirements

### Unit Tests

```python
def test_scaling_decision_scale_up():
    """Test scale-up decision logic"""
    policy = ScalingPolicy(
        min_replicas=1,
        max_replicas=10,
        target_metric_value=80.0
    )
    optimizer = ScalingOptimizer(policy)

    decision = optimizer.decide(
        current_metric=95.0,  # Above threshold
        predicted_metric=None,
        current_replicas=3,
        current_latency_p95=50.0
    )

    assert decision.direction == ScaleDirection.UP
    assert decision.target_replicas > 3

def test_cost_calculation():
    """Test cost calculator"""
    calc = CostCalculator()
    cost = calc.calculate_cluster_cost(
        {"t3.xlarge": 5, "g4dn.xlarge": 2},
        use_spot=False
    )
    assert cost > 0
    expected = (0.1664 * 5) + (0.526 * 2)
    assert abs(cost - expected) < 0.01
```

### Integration Tests

```bash
# tests/integration/test_autoscaler.sh

#!/bin/bash
set -e

echo "Deploying test workload..."
kubectl apply -f tests/fixtures/test-deployment.yaml

echo "Creating HPA with custom metrics..."
python -m autoscaler.cli create-hpa \
    --deployment test-app \
    --metric requests_per_second \
    --target-value 100 \
    --min-replicas 2 \
    --max-replicas 10

echo "Generating load..."
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://test-app; done"

echo "Verifying scale-up..."
sleep 60
REPLICAS=$(kubectl get deployment test-app -o jsonpath='{.spec.replicas}')
if [ "$REPLICAS" -gt 2 ]; then
    echo "✅ Autoscaling working! Scaled to $REPLICAS replicas"
else
    echo "❌ Autoscaling failed"
    exit 1
fi
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Scale-Up Latency** | <90s | ________s |
| **Prediction Accuracy** | >80% | ________% |
| **Cost Savings** | 40-60% | ________% |
| **Flapping Events** | <1 per day | ________ |

## Validation

Submit:
1. Complete implementation (all 6 tasks)
2. Test suite with >75% coverage
3. Performance benchmarks
4. Cost analysis report showing savings
5. Documentation (DESIGN.md, METRICS.md, TROUBLESHOOTING.md)

## Resources

- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Custom Metrics API](https://github.com/kubernetes-sigs/custom-metrics-apiserver)
- [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [VPA](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)

---

**Estimated Completion Time**: 35-43 hours

**Skills Practiced**:
- Kubernetes autoscaling (HPA, VPA, CA)
- Custom metrics API
- Time-series forecasting
- Cost optimization
- GPU scheduling
