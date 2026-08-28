# Exercise 05: Service Mesh Observability with Istio

**Estimated Time**: 30-38 hours
**Difficulty**: Advanced
**Prerequisites**: Kubernetes, Istio, Python 3.9+, Prometheus, Grafana, Jaeger

## Overview

Build a comprehensive observability platform for microservices using Istio service mesh. Implement distributed tracing, traffic monitoring, SLI/SLO tracking, and automated canary deployments with rollback based on error rates. This exercise teaches production-grade service mesh patterns essential for operating complex ML inference pipelines at scale.

In production ML infrastructure, service mesh observability is critical for:
- **Distributed Tracing**: Track requests across 10+ microservices
- **Traffic Management**: Canary deployments, A/B testing for model versions
- **SLO Monitoring**: Ensure 99.9% availability, <100ms p95 latency
- **Security**: mTLS, authorization policies, audit logs
- **Chaos Engineering**: Inject faults to test resilience

## Learning Objectives

By completing this exercise, you will:

1. **Deploy Istio service mesh** with production configuration
2. **Implement distributed tracing** with Jaeger/Zipkin
3. **Create SLI/SLO dashboards** in Grafana
4. **Build automated canary deployment** system
5. **Implement fault injection** for chaos engineering
6. **Monitor golden signals** (latency, traffic, errors, saturation)
7. **Configure mTLS** and authorization policies

## Business Context

**Real-World Scenario**: Your ML platform serves 20 models through a microservices architecture (API gateway → feature service → model inference → post-processing → response). Current challenges:

- **Debug complexity**: Request failures span 5+ services, no visibility into where errors occur
- **Slow rollouts**: New model deployments are all-or-nothing (risky)
- **No SLO tracking**: Can't prove you meet 99.9% availability SLA
- **Security gaps**: Inter-service communication not encrypted
- **Performance issues**: Can't identify which service adds latency

Your task: Build observability platform that:
- Traces every request end-to-end with <5ms overhead
- Deploys models via canary (5% → 25% → 50% → 100%) with auto-rollback
- Tracks error budget consumption (0.1% errors allowed per month)
- Enforces mTLS between all services
- Provides flame graphs showing request latency breakdown

## Project Structure

```
exercise-05-service-mesh-observability/
├── README.md
├── requirements.txt
├── istio/
│   ├── install.yaml                  # Istio installation config
│   ├── telemetry.yaml                # Telemetry configuration
│   └── mesh-config.yaml              # Mesh-wide settings
├── kubernetes/
│   ├── apps/
│   │   ├── api-gateway.yaml          # Frontend API
│   │   ├── feature-service.yaml      # Feature engineering
│   │   ├── model-inference.yaml      # ML model serving
│   │   └── post-processor.yaml       # Result processing
│   ├── virtual-services/
│   │   ├── canary-deployment.yaml    # Canary routing rules
│   │   └── fault-injection.yaml      # Chaos engineering
│   ├── destination-rules/
│   │   ├── circuit-breaker.yaml      # Circuit breaker config
│   │   └── load-balancing.yaml       # Load balancing policy
│   └── security/
│       ├── peer-authentication.yaml  # mTLS enforcement
│       └── authorization-policy.yaml # AuthZ rules
├── src/
│   └── observability/
│       ├── __init__.py
│       ├── slo_tracker.py            # SLO/SLI monitoring
│       ├── canary_controller.py      # Automated canary deployments
│       ├── trace_analyzer.py         # Distributed trace analysis
│       ├── error_budget.py           # Error budget calculator
│       ├── alert_manager.py          # Alert routing
│       └── cli.py                    # CLI tools
├── dashboards/
│   ├── golden-signals.json           # Grafana dashboard
│   ├── slo-dashboard.json            # SLO tracking
│   └── trace-analysis.json           # Trace visualization
├── tests/
│   ├── test_slo_tracker.py
│   ├── test_canary_controller.py
│   └── test_trace_analyzer.py
├── examples/
│   ├── deploy_canary.sh
│   ├── inject_faults.sh
│   └── analyze_traces.py
└── docs/
    ├── DESIGN.md
    ├── SLO_DEFINITIONS.md
    └── RUNBOOK.md
```

## Requirements

### Functional Requirements

1. **Distributed Tracing**:
   - Trace 100% of requests with sampling (configurable)
   - Correlate traces across all services
   - Extract latency per service
   - Identify error sources

2. **SLO Monitoring**:
   - Define SLIs (availability, latency, throughput)
   - Track error budget consumption
   - Alert when error budget depleting quickly
   - Generate SLO compliance reports

3. **Canary Deployments**:
   - Gradual traffic shift (5% → 25% → 50% → 100%)
   - Auto-promote if error rate <0.5%
   - Auto-rollback if error rate >2%
   - Manual approval gates

4. **Traffic Management**:
   - Route by header (A/B testing)
   - Circuit breakers (open after 5 consecutive failures)
   - Retry policies (3 retries with exponential backoff)
   - Timeouts (5s default)

5. **Security**:
   - Enforce mTLS cluster-wide
   - Namespace isolation
   - Service-to-service authorization

### Non-Functional Requirements

- **Tracing Overhead**: <5ms per request
- **Canary Promotion Time**: 15 minutes (5min per stage)
- **Alert Latency**: <30 seconds from SLO breach
- **Dashboard Load Time**: <2 seconds

## Implementation Tasks

### Task 1: Istio Installation and Configuration (5-6 hours)

Install Istio with production settings.

```yaml
# istio/install.yaml

apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: production-istio
spec:
  profile: production

  # Enable telemetry
  meshConfig:
    enableTracing: true
    defaultConfig:
      tracing:
        sampling: 100.0  # Sample 100% of requests
        zipkin:
          address: jaeger-collector.istio-system:9411

    # Enable access logging
    accessLogFile: /dev/stdout
    accessLogFormat: |
      [%START_TIME%] "%REQ(:METHOD)% %REQ(X-ENVOY-ORIGINAL-PATH?:PATH)% %PROTOCOL%"
      %RESPONSE_CODE% %RESPONSE_FLAGS% %BYTES_RECEIVED% %BYTES_SENT%
      %DURATION% "%REQ(X-FORWARDED-FOR)%" "%REQ(USER-AGENT)%"
      "%REQ(X-REQUEST-ID)%" "%REQ(:AUTHORITY)%" "%UPSTREAM_HOST%"

  components:
    pilot:
      k8s:
        resources:
          requests:
            cpu: 500m
            memory: 2Gi

    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
        hpaSpec:
          minReplicas: 2
          maxReplicas: 10

    egressGateways:
    - name: istio-egressgateway
      enabled: true

  # Addons
  addonComponents:
    prometheus:
      enabled: true
    grafana:
      enabled: true
    tracing:
      enabled: true
    kiali:
      enabled: true
```

```bash
# Installation script
#!/bin/bash

# Install Istio
istioctl install -f istio/install.yaml -y

# Label namespace for automatic sidecar injection
kubectl label namespace default istio-injection=enabled

# Deploy Jaeger for tracing
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/jaeger.yaml

# Deploy Prometheus
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/prometheus.yaml

# Deploy Grafana
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/grafana.yaml

# Deploy Kiali (service mesh visualization)
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml

echo "✅ Istio installed successfully"
```

**Acceptance Criteria**:
- ✅ Istio control plane healthy
- ✅ Sidecars auto-injected in labeled namespaces
- ✅ Telemetry addons accessible
- ✅ mTLS enforced cluster-wide

---

### Task 2: SLO Tracker (7-8 hours)

Implement SLO/SLI monitoring system.

```python
# src/observability/slo_tracker.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

class SLIType(Enum):
    """Types of Service Level Indicators"""
    AVAILABILITY = "availability"  # % of successful requests
    LATENCY = "latency"  # % of requests < threshold
    THROUGHPUT = "throughput"  # requests per second

@dataclass
class SLI:
    """Service Level Indicator definition"""
    name: str
    type: SLIType
    query: str  # PromQL query
    threshold: float  # For latency: 100ms, for availability: N/A

@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    service: str
    sli: SLI
    target: float  # e.g., 99.9% (0.999)
    window: timedelta  # Rolling window (30 days)

@dataclass
class ErrorBudget:
    """Remaining error budget for SLO"""
    slo: SLO
    budget_total: float  # Total allowed errors (0.1% = 0.001)
    budget_consumed: float  # Errors consumed so far
    budget_remaining: float  # Remaining budget
    burn_rate: float  # Rate of consumption (errors per hour)
    time_to_exhaustion: Optional[timedelta]  # When budget runs out

class SLOTracker:
    """Track SLOs and error budgets"""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    def check_slo(self, slo: SLO, now: Optional[datetime] = None) -> ErrorBudget:
        """
        Check SLO compliance and calculate error budget

        Args:
            slo: SLO to check
            now: Current time (defaults to datetime.now())

        Returns:
            ErrorBudget with current status
        """
        now = now or datetime.now()

        # TODO: Query Prometheus for SLI metric
        if slo.sli.type == SLIType.AVAILABILITY:
            # Query: (sum(rate(requests_total{service="X",status=~"2.."}[window])) /
            #         sum(rate(requests_total{service="X"}[window])))
            current_availability = self._query_availability(slo)
            budget_total = 1.0 - slo.target  # e.g., 0.001 for 99.9%
            budget_consumed = 1.0 - current_availability

        elif slo.sli.type == SLIType.LATENCY:
            # Query: histogram_quantile(0.95, rate(request_duration_bucket[window])) < threshold
            current_latency_compliance = self._query_latency(slo)
            budget_total = 1.0 - slo.target
            budget_consumed = 1.0 - current_latency_compliance

        else:
            raise ValueError(f"Unsupported SLI type: {slo.sli.type}")

        budget_remaining = budget_total - budget_consumed

        # TODO: Calculate burn rate (errors per hour)
        burn_rate = self._calculate_burn_rate(slo, window=timedelta(hours=1))

        # TODO: Estimate time to exhaustion
        if burn_rate > 0 and budget_remaining > 0:
            time_to_exhaustion = timedelta(hours=budget_remaining / burn_rate)
        else:
            time_to_exhaustion = None

        return ErrorBudget(
            slo=slo,
            budget_total=budget_total,
            budget_consumed=budget_consumed,
            budget_remaining=budget_remaining,
            burn_rate=burn_rate,
            time_to_exhaustion=time_to_exhaustion
        )

    def _query_availability(self, slo: SLO) -> float:
        """
        Query availability SLI from Prometheus

        Returns availability as decimal (0.999 = 99.9%)
        """
        import requests

        # Build PromQL query
        window_str = f"{int(slo.window.total_seconds())}s"
        query = f"""
        sum(rate(istio_requests_total{{
            destination_service=~"{slo.service}.*",
            response_code=~"2.."
        }}[{window_str}]))
        /
        sum(rate(istio_requests_total{{
            destination_service=~"{slo.service}.*"
        }}[{window_str}]))
        """

        # TODO: Execute query
        response = requests.post(
            f"{self.prometheus_url}/api/v1/query",
            data={"query": query}
        )

        # TODO: Parse result
        if response.status_code == 200:
            data = response.json()
            if data.get("data", {}).get("result"):
                return float(data["data"]["result"][0]["value"][1])

        return 0.0

    def _query_latency(self, slo: SLO) -> float:
        """
        Query latency SLI from Prometheus

        Returns % of requests meeting latency threshold
        """
        # TODO: Query histogram_quantile for p95/p99
        # TODO: Compare to threshold
        # TODO: Return compliance percentage
        raise NotImplementedError

    def _calculate_burn_rate(self, slo: SLO, window: timedelta) -> float:
        """
        Calculate rate of error budget consumption

        Returns errors per hour
        """
        # TODO: Query error rate over recent window
        # TODO: Calculate hourly burn rate
        raise NotImplementedError

    def generate_report(self, slos: List[SLO]) -> Dict:
        """
        Generate SLO compliance report

        Returns:
            {
                "report_time": "2024-01-25T10:00:00Z",
                "slos": [
                    {
                        "name": "api-availability",
                        "compliant": True,
                        "current": 99.95,
                        "target": 99.9,
                        "error_budget_remaining": 0.05
                    }
                ],
                "summary": {
                    "total_slos": 5,
                    "compliant": 4,
                    "non_compliant": 1,
                    "at_risk": 1  # Burning budget quickly
                }
            }
        """
        # TODO: Check all SLOs
        # TODO: Aggregate results
        # TODO: Return structured report
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Track availability SLO (99.9%)
- ✅ Track latency SLO (p95 <100ms)
- ✅ Calculate error budgets accurately
- ✅ Estimate time to budget exhaustion
- ✅ Generate compliance reports

---

### Task 3: Canary Controller (7-9 hours)

Build automated canary deployment system.

```python
# src/observability/canary_controller.py

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum
from kubernetes import client, config
import time

class CanaryStage(Enum):
    """Canary deployment stages"""
    STAGE_0 = 0    # Baseline (0% canary)
    STAGE_1 = 5    # 5% canary traffic
    STAGE_2 = 25   # 25% canary traffic
    STAGE_3 = 50   # 50% canary traffic
    STAGE_4 = 100  # 100% canary (promote complete)

class CanaryStatus(Enum):
    """Canary deployment status"""
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    namespace: str
    service_name: str
    baseline_version: str  # v1
    canary_version: str  # v2

    # Traffic shifting stages
    stages: List[int] = None  # [5, 25, 50, 100]
    stage_duration: timedelta = timedelta(minutes=5)

    # Success criteria
    max_error_rate: float = 0.02  # 2%
    max_latency_p95: float = 100.0  # ms

    # Rollback criteria
    rollback_error_rate: float = 0.05  # 5%
    rollback_latency_p95: float = 200.0  # ms

@dataclass
class CanaryMetrics:
    """Metrics for canary analysis"""
    error_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    requests_per_second: float

class CanaryController:
    """Manage automated canary deployments"""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        config.load_kube_config()
        self.k8s_networking = client.NetworkingV1Api()
        self.k8s_custom = client.CustomObjectsApi()

    def deploy_canary(self, config: CanaryConfig) -> CanaryStatus:
        """
        Execute canary deployment with automated promotion/rollback

        Workflow:
        1. Start at 0% canary (baseline only)
        2. For each stage (5%, 25%, 50%, 100%):
           a. Update VirtualService to shift traffic
           b. Wait for stage_duration
           c. Analyze metrics (error rate, latency)
           d. If metrics good: promote to next stage
           e. If metrics bad: rollback to baseline
        3. If reach 100%: complete deployment
        """
        print(f"Starting canary deployment: {config.baseline_version} → {config.canary_version}")

        stages = config.stages or [5, 25, 50, 100]

        for stage_pct in stages:
            print(f"\n📊 Stage: {stage_pct}% canary traffic")

            # TODO: Update VirtualService traffic split
            self._update_traffic_split(
                namespace=config.namespace,
                service=config.service_name,
                baseline_weight=100 - stage_pct,
                canary_weight=stage_pct
            )

            # TODO: Wait for traffic to stabilize
            print(f"Waiting {config.stage_duration.total_seconds()}s for metrics...")
            time.sleep(config.stage_duration.total_seconds())

            # TODO: Collect metrics for canary version
            canary_metrics = self._get_metrics(
                config.namespace,
                config.service_name,
                config.canary_version
            )

            baseline_metrics = self._get_metrics(
                config.namespace,
                config.service_name,
                config.baseline_version
            )

            print(f"Canary metrics: error_rate={canary_metrics.error_rate:.2%}, "
                  f"p95={canary_metrics.latency_p95:.1f}ms")
            print(f"Baseline metrics: error_rate={baseline_metrics.error_rate:.2%}, "
                  f"p95={baseline_metrics.latency_p95:.1f}ms")

            # TODO: Decide: promote, rollback, or continue
            decision = self._analyze_metrics(canary_metrics, baseline_metrics, config)

            if decision == "rollback":
                print("❌ Metrics failed! Rolling back...")
                self._rollback(config.namespace, config.service_name, config.baseline_version)
                return CanaryStatus.ROLLED_BACK

            elif decision == "promote":
                print(f"✅ Metrics pass! Promoting to {stage_pct}%")
                continue

            else:
                print("⚠️  Metrics inconclusive, continuing...")

        # Reached 100%, deployment successful
        print("\n🎉 Canary deployment completed successfully!")
        return CanaryStatus.SUCCEEDED

    def _update_traffic_split(
        self,
        namespace: str,
        service: str,
        baseline_weight: int,
        canary_weight: int
    ) -> None:
        """
        Update Istio VirtualService to split traffic

        Example VirtualService:
        ```yaml
        apiVersion: networking.istio.io/v1beta1
        kind: VirtualService
        metadata:
          name: model-inference
        spec:
          hosts:
          - model-inference
          http:
          - route:
            - destination:
                host: model-inference
                subset: v1  # baseline
              weight: 95
            - destination:
                host: model-inference
                subset: v2  # canary
              weight: 5
        ```
        """
        # TODO: Get current VirtualService
        vs = self.k8s_custom.get_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            namespace=namespace,
            plural="virtualservices",
            name=service
        )

        # TODO: Update weights
        vs["spec"]["http"][0]["route"][0]["weight"] = baseline_weight
        vs["spec"]["http"][0]["route"][1]["weight"] = canary_weight

        # TODO: Apply updated VirtualService
        self.k8s_custom.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            namespace=namespace,
            plural="virtualservices",
            name=service,
            body=vs
        )

    def _get_metrics(
        self,
        namespace: str,
        service: str,
        version: str
    ) -> CanaryMetrics:
        """
        Get metrics for specific service version

        Queries Prometheus for:
        - Error rate: rate(istio_requests_total{destination_version="v2", response_code=~"5.."}[1m])
        - Latency: histogram_quantile(0.95, rate(istio_request_duration_milliseconds_bucket[1m]))
        """
        import requests

        # TODO: Query error rate
        error_query = f"""
        sum(rate(istio_requests_total{{
            destination_service=~"{service}.*",
            destination_version="{version}",
            response_code=~"5.."
        }}[1m]))
        /
        sum(rate(istio_requests_total{{
            destination_service=~"{service}.*",
            destination_version="{version}"
        }}[1m]))
        """

        # TODO: Query latency percentiles
        latency_p95_query = f"""
        histogram_quantile(0.95,
          rate(istio_request_duration_milliseconds_bucket{{
            destination_service=~"{service}.*",
            destination_version="{version}"
          }}[1m])
        )
        """

        # TODO: Execute queries and parse results
        error_rate = self._query_prometheus(error_query) or 0.0
        latency_p95 = self._query_prometheus(latency_p95_query) or 0.0

        return CanaryMetrics(
            error_rate=error_rate,
            latency_p50=0.0,  # TODO: Query p50
            latency_p95=latency_p95,
            latency_p99=0.0,  # TODO: Query p99
            requests_per_second=0.0  # TODO: Query RPS
        )

    def _analyze_metrics(
        self,
        canary: CanaryMetrics,
        baseline: CanaryMetrics,
        config: CanaryConfig
    ) -> str:
        """
        Analyze metrics to decide: promote, rollback, or continue

        Returns: "promote", "rollback", or "continue"
        """
        # TODO: Check if canary exceeds rollback thresholds
        if canary.error_rate > config.rollback_error_rate:
            return "rollback"
        if canary.latency_p95 > config.rollback_latency_p95:
            return "rollback"

        # TODO: Check if canary meets success criteria
        # Compare canary to baseline (canary should not be significantly worse)
        if canary.error_rate > baseline.error_rate * 2:  # 2x worse
            return "rollback"
        if canary.latency_p95 > baseline.latency_p95 * 1.5:  # 50% worse
            return "rollback"

        # Metrics acceptable
        return "promote"

    def _rollback(self, namespace: str, service: str, baseline_version: str) -> None:
        """Rollback to 100% baseline traffic"""
        self._update_traffic_split(namespace, service, 100, 0)
        print(f"Rolled back to {baseline_version}")

    def _query_prometheus(self, query: str) -> Optional[float]:
        """Execute Prometheus query"""
        import requests
        response = requests.post(
            f"{self.prometheus_url}/api/v1/query",
            data={"query": query}
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("data", {}).get("result"):
                return float(data["data"]["result"][0]["value"][1])
        return None
```

**Acceptance Criteria**:
- ✅ Deploy canary in stages (5% → 100%)
- ✅ Auto-promote if metrics good
- ✅ Auto-rollback if metrics bad
- ✅ Compare canary vs baseline metrics
- ✅ Update Istio VirtualService correctly

---

### Task 4: Trace Analyzer (5-6 hours)

Analyze distributed traces to identify latency sources.

```python
# src/observability/trace_analyzer.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests

@dataclass
class Span:
    """Represents a single span in distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service_name: str
    operation_name: str
    start_time: datetime
    duration_ms: float
    tags: Dict[str, str]
    logs: List[Dict]

@dataclass
class Trace:
    """Complete distributed trace"""
    trace_id: str
    spans: List[Span]

    @property
    def total_duration_ms(self) -> float:
        """Total request duration"""
        if not self.spans:
            return 0.0
        return max(span.start_time.timestamp() * 1000 + span.duration_ms for span in self.spans) - \
               min(span.start_time.timestamp() * 1000 for span in self.spans)

    @property
    def critical_path(self) -> List[Span]:
        """Find critical path (longest sequential chain)"""
        # TODO: Build span tree, find longest path
        raise NotImplementedError

@dataclass
class LatencyBreakdown:
    """Latency contribution by service"""
    service_name: str
    total_duration_ms: float
    percentage: float

class TraceAnalyzer:
    """Analyze distributed traces from Jaeger"""

    def __init__(self, jaeger_url: str):
        self.jaeger_url = jaeger_url

    def get_traces(
        self,
        service: str,
        start: datetime,
        end: datetime,
        limit: int = 100
    ) -> List[Trace]:
        """
        Fetch traces from Jaeger

        Uses Jaeger HTTP API:
        GET /api/traces?service=X&start=...&end=...&limit=100
        """
        # TODO: Build query parameters
        params = {
            "service": service,
            "start": int(start.timestamp() * 1000000),  # microseconds
            "end": int(end.timestamp() * 1000000),
            "limit": limit
        }

        # TODO: Fetch from Jaeger API
        response = requests.get(f"{self.jaeger_url}/api/traces", params=params)

        # TODO: Parse JSON response into Trace objects
        if response.status_code == 200:
            data = response.json()
            traces = []
            for trace_data in data.get("data", []):
                spans = self._parse_spans(trace_data)
                traces.append(Trace(
                    trace_id=trace_data["traceID"],
                    spans=spans
                ))
            return traces

        return []

    def analyze_latency(self, trace: Trace) -> List[LatencyBreakdown]:
        """
        Break down latency by service

        Returns:
            [
                LatencyBreakdown("api-gateway", 15.2, 10%),
                LatencyBreakdown("feature-service", 45.8, 30%),
                LatencyBreakdown("model-inference", 82.5, 55%),
                LatencyBreakdown("post-processor", 7.5, 5%)
            ]
        """
        # TODO: Group spans by service
        service_durations = {}
        for span in trace.spans:
            if span.service_name not in service_durations:
                service_durations[span.service_name] = 0.0
            service_durations[span.service_name] += span.duration_ms

        # TODO: Calculate percentages
        total = trace.total_duration_ms
        breakdowns = []
        for service, duration in service_durations.items():
            breakdowns.append(LatencyBreakdown(
                service_name=service,
                total_duration_ms=duration,
                percentage=(duration / total * 100) if total > 0 else 0
            ))

        # Sort by duration (descending)
        breakdowns.sort(key=lambda b: b.total_duration_ms, reverse=True)
        return breakdowns

    def find_slow_traces(
        self,
        service: str,
        threshold_ms: float = 100.0,
        window: timedelta = timedelta(hours=1)
    ) -> List[Trace]:
        """
        Find traces exceeding latency threshold

        Useful for identifying performance regressions
        """
        end = datetime.now()
        start = end - window

        # TODO: Fetch traces
        traces = self.get_traces(service, start, end, limit=1000)

        # TODO: Filter by duration
        slow_traces = [t for t in traces if t.total_duration_ms > threshold_ms]

        return slow_traces

    def find_errors(
        self,
        service: str,
        window: timedelta = timedelta(hours=1)
    ) -> List[Trace]:
        """Find traces with errors"""
        end = datetime.now()
        start = end - window

        traces = self.get_traces(service, start, end)

        # TODO: Filter traces with error tags
        error_traces = []
        for trace in traces:
            for span in trace.spans:
                if span.tags.get("error") == "true" or span.tags.get("http.status_code", "").startswith("5"):
                    error_traces.append(trace)
                    break

        return error_traces

    def _parse_spans(self, trace_data: Dict) -> List[Span]:
        """Parse Jaeger trace JSON into Span objects"""
        # TODO: Extract spans from trace data
        # TODO: Handle nested structure
        raise NotImplementedError
```

**Acceptance Criteria**:
- ✅ Fetch traces from Jaeger
- ✅ Calculate latency breakdown by service
- ✅ Find slow traces (>threshold)
- ✅ Find error traces
- ✅ Identify critical path in traces

---

### Task 5: Alert Manager (3-4 hours)

Implement alert routing based on SLO violations.

```python
# src/observability/alert_manager.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from .slo_tracker import ErrorBudget, SLO

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"  # Page on-call
    WARNING = "warning"  # Notify team
    INFO = "info"  # Log only

@dataclass
class Alert:
    """Alert definition"""
    title: str
    description: str
    severity: AlertSeverity
    slo: Optional[SLO] = None
    error_budget: Optional[ErrorBudget] = None
    runbook_url: Optional[str] = None

class AlertManager:
    """Generate and route alerts"""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    def check_error_budget(self, budget: ErrorBudget) -> Optional[Alert]:
        """
        Generate alert if error budget critical

        Alert if:
        - Budget <10% remaining (WARNING)
        - Budget exhausted (CRITICAL)
        - Burn rate >5x (will exhaust in <6 hours) (CRITICAL)
        """
        # TODO: Check remaining budget
        if budget.budget_remaining <= 0:
            return Alert(
                title=f"SLO Violated: {budget.slo.name}",
                description=f"Error budget exhausted! Current: {budget.budget_consumed:.2%}",
                severity=AlertSeverity.CRITICAL,
                slo=budget.slo,
                error_budget=budget
            )

        # TODO: Check burn rate
        if budget.time_to_exhaustion and budget.time_to_exhaustion < timedelta(hours=6):
            return Alert(
                title=f"High Error Budget Burn Rate: {budget.slo.name}",
                description=f"Budget will exhaust in {budget.time_to_exhaustion}",
                severity=AlertSeverity.CRITICAL,
                slo=budget.slo,
                error_budget=budget
            )

        # TODO: Check low budget
        budget_pct = budget.budget_remaining / budget.budget_total if budget.budget_total > 0 else 1.0
        if budget_pct < 0.1:  # <10% remaining
            return Alert(
                title=f"Low Error Budget: {budget.slo.name}",
                description=f"Only {budget_pct:.1%} budget remaining",
                severity=AlertSeverity.WARNING,
                slo=budget.slo,
                error_budget=budget
            )

        return None

    def send_alert(self, alert: Alert) -> None:
        """
        Send alert to webhook (Slack, PagerDuty, etc.)

        Payload:
        {
            "title": "SLO Violated: api-availability",
            "severity": "critical",
            "description": "...",
            "timestamp": "2024-01-25T10:00:00Z"
        }
        """
        import requests
        from datetime import datetime

        if not self.webhook_url:
            print(f"[{alert.severity.value.upper()}] {alert.title}: {alert.description}")
            return

        # TODO: Format payload
        payload = {
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "timestamp": datetime.now().isoformat()
        }

        # TODO: Send to webhook
        requests.post(self.webhook_url, json=payload)
```

**Acceptance Criteria**:
- ✅ Generate alerts for SLO violations
- ✅ Alert on high burn rate
- ✅ Send to webhook (Slack/PagerDuty)
- ✅ Include runbook links

---

### Task 6: CLI and Integration (3-4 hours)

Build CLI and integrate all components.

```python
# src/observability/cli.py

import click
from .slo_tracker import SLOTracker, SLO, SLI, SLIType
from .canary_controller import CanaryController, CanaryConfig
from .trace_analyzer import TraceAnalyzer
from datetime import timedelta

@click.group()
def cli():
    """Service Mesh Observability Tools"""
    pass

@cli.command()
@click.option('--service', required=True)
@click.option('--prometheus-url', default='http://prometheus:9090')
def check_slo(service: str, prometheus_url: str):
    """Check SLO compliance"""
    # TODO: Define SLOs
    # TODO: Check compliance
    # TODO: Display results
    pass

@cli.command()
@click.option('--namespace', default='default')
@click.option('--service', required=True)
@click.option('--baseline', required=True, help='Baseline version (e.g., v1)')
@click.option('--canary', required=True, help='Canary version (e.g., v2)')
@click.option('--prometheus-url', default='http://prometheus:9090')
def deploy_canary(namespace: str, service: str, baseline: str, canary: str, prometheus_url: str):
    """Deploy canary with automated promotion/rollback"""
    controller = CanaryController(prometheus_url)

    config = CanaryConfig(
        namespace=namespace,
        service_name=service,
        baseline_version=baseline,
        canary_version=canary
    )

    status = controller.deploy_canary(config)
    click.echo(f"Deployment status: {status.value}")

@cli.command()
@click.option('--service', required=True)
@click.option('--jaeger-url', default='http://jaeger:16686')
@click.option('--threshold-ms', type=float, default=100.0)
def analyze_traces(service: str, jaeger_url: str, threshold_ms: float):
    """Analyze slow traces"""
    analyzer = TraceAnalyzer(jaeger_url)

    slow_traces = analyzer.find_slow_traces(service, threshold_ms)

    click.echo(f"Found {len(slow_traces)} slow traces (>{threshold_ms}ms)")
    for trace in slow_traces[:10]:
        breakdown = analyzer.analyze_latency(trace)
        click.echo(f"\nTrace {trace.trace_id}: {trace.total_duration_ms:.1f}ms")
        for b in breakdown:
            click.echo(f"  {b.service_name}: {b.total_duration_ms:.1f}ms ({b.percentage:.1f}%)")

if __name__ == '__main__':
    cli()
```

**Acceptance Criteria**:
- ✅ CLI commands work end-to-end
- ✅ Integration with Prometheus/Jaeger
- ✅ Canary deployment automation
- ✅ Trace analysis

---

## Testing Requirements

```python
def test_slo_tracker():
    """Test SLO tracking"""
    # Mock Prometheus responses
    # Test availability calculation
    # Test error budget math

def test_canary_controller():
    """Test canary deployment"""
    # Mock Kubernetes API
    # Test traffic splitting
    # Test rollback logic

def test_trace_analyzer():
    """Test trace analysis"""
    # Mock Jaeger responses
    # Test latency breakdown
    # Test slow trace detection
```

## Expected Results

| Metric | Target | Measured |
|--------|--------|----------|
| **Tracing Overhead** | <5ms | ________ms |
| **Canary Duration** | 15min | ________min |
| **Alert Latency** | <30s | ________s |

## Validation

Submit:
1. Complete implementation
2. Grafana dashboards (golden signals, SLO)
3. Canary deployment demo
4. Trace analysis examples
5. Documentation

## Resources

- [Istio Documentation](https://istio.io/latest/docs/)
- [Jaeger Tracing](https://www.jaegertracing.io/)
- [SLO Workshop](https://sre.google/workbook/implementing-slos/)
- [Flagger (Canary Tool)](https://flagger.app/)

---

**Estimated Completion Time**: 30-38 hours

**Skills Practiced**:
- Istio service mesh
- Distributed tracing
- SLO/SLI monitoring
- Canary deployments
- Traffic management
