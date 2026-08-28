"""
Prometheus Metrics for LLM Infrastructure

This module implements comprehensive monitoring metrics for LLM serving.
Metrics track performance, usage, costs, and system health.

Learning Objectives:
- Understand Prometheus metrics types
- Learn LLM-specific monitoring patterns
- Implement custom metrics collectors
- Track performance and business metrics
- Build monitoring dashboards

Key Concepts:
- Counter: Monotonically increasing (requests, errors)
- Gauge: Can go up/down (GPU memory, active requests)
- Histogram: Distribution of values (latency, token count)
- Summary: Similar to histogram with percentiles
- Labels: Dimensions for filtering (model, user, endpoint)

Metrics Categories:
- Request metrics (count, latency, errors)
- Token metrics (input/output tokens, throughput)
- Model metrics (GPU utilization, memory, batch size)
- Cost metrics (cost per request, total spend)
- Quality metrics (user feedback, error types)
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

# TODO: Import Prometheus client
# try:
#     from prometheus_client import (
#         Counter, Gauge, Histogram, Summary,
#         CollectorRegistry, generate_latest,
#         CONTENT_TYPE_LATEST
#     )
# except ImportError:
#     raise ImportError("prometheus_client not installed. Install with: pip install prometheus-client")


# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

# TODO: Define request metrics
# Request counter with labels
# llm_requests_total = Counter(
#     'llm_requests_total',
#     'Total number of LLM requests',
#     ['model', 'endpoint', 'status']
# )

# TODO: Define latency histogram
# llm_request_duration_seconds = Histogram(
#     'llm_request_duration_seconds',
#     'Request duration in seconds',
#     ['model', 'endpoint'],
#     buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
# )

# TODO: Define token metrics
# llm_tokens_total = Counter(
#     'llm_tokens_total',
#     'Total number of tokens processed',
#     ['model', 'direction']  # direction: input/output
# )

# TODO: Define throughput gauge
# llm_tokens_per_second = Gauge(
#     'llm_tokens_per_second',
#     'Token generation throughput',
#     ['model']
# )

# TODO: Define GPU metrics
# llm_gpu_memory_used_bytes = Gauge(
#     'llm_gpu_memory_used_bytes',
#     'GPU memory used in bytes',
#     ['gpu_id', 'model']
# )

# llm_gpu_utilization_percent = Gauge(
#     'llm_gpu_utilization_percent',
#     'GPU utilization percentage',
#     ['gpu_id', 'model']
# )

# TODO: Define cost metrics
# llm_cost_usd_total = Counter(
#     'llm_cost_usd_total',
#     'Total cost in USD',
#     ['model', 'customer']
# )

# TODO: Define active requests
# llm_active_requests = Gauge(
#     'llm_active_requests',
#     'Number of active requests being processed',
#     ['model']
# )

# TODO: Define error metrics
# llm_errors_total = Counter(
#     'llm_errors_total',
#     'Total number of errors',
#     ['model', 'error_type']
# )


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """
    Collect and expose Prometheus metrics for LLM serving.

    TODO: Implement metrics collection
    - Track all request metrics
    - Monitor GPU resources
    - Calculate costs
    - Expose metrics endpoint
    """

    def __init__(
        self,
        model_name: str,
        cost_per_1k_tokens: Optional[Dict[str, float]] = None
    ):
        """
        Initialize metrics collector.

        Args:
            model_name: Name of the LLM model
            cost_per_1k_tokens: Cost configuration
        """
        # TODO: Store configuration
        # self.model_name = model_name
        # self.cost_per_1k_tokens = cost_per_1k_tokens or {
        #     "input": 0.0002,
        #     "output": 0.0002
        # }

        # TODO: Initialize counters for tracking
        # self._request_start_times: Dict[str, float] = {}

        pass

    def record_request(
        self,
        endpoint: str,
        status: str = "success"
    ) -> None:
        """
        Record a completed request.

        TODO: Implement request recording
        - Increment request counter
        - Label with model, endpoint, status

        Args:
            endpoint: API endpoint (/generate, /rag-generate)
            status: Request status (success, error, timeout)
        """
        # TODO: Increment counter
        # llm_requests_total.labels(
        #     model=self.model_name,
        #     endpoint=endpoint,
        #     status=status
        # ).inc()

        pass

    def record_latency(
        self,
        endpoint: str,
        duration_seconds: float
    ) -> None:
        """
        Record request latency.

        TODO: Implement latency recording
        - Observe duration in histogram
        - Label with model, endpoint

        Args:
            endpoint: API endpoint
            duration_seconds: Request duration
        """
        # TODO: Observe latency
        # llm_request_duration_seconds.labels(
        #     model=self.model_name,
        #     endpoint=endpoint
        # ).observe(duration_seconds)

        pass

    def record_tokens(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """
        Record token usage.

        TODO: Implement token recording
        - Increment input token counter
        - Increment output token counter
        - Calculate and update throughput

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # TODO: Record input tokens
        # llm_tokens_total.labels(
        #     model=self.model_name,
        #     direction="input"
        # ).inc(input_tokens)

        # TODO: Record output tokens
        # llm_tokens_total.labels(
        #     model=self.model_name,
        #     direction="output"
        # ).inc(output_tokens)

        pass

    def record_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        customer: str = "default"
    ) -> float:
        """
        Record and return cost.

        TODO: Implement cost recording
        - Calculate cost based on tokens
        - Increment cost counter
        - Return calculated cost

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            customer: Customer identifier

        Returns:
            Cost in USD
        """
        # TODO: Calculate cost
        # input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens["input"]
        # output_cost = (output_tokens / 1000) * self.cost_per_1k_tokens["output"]
        # total_cost = input_cost + output_cost

        # TODO: Record cost
        # llm_cost_usd_total.labels(
        #     model=self.model_name,
        #     customer=customer
        # ).inc(total_cost)

        # return total_cost

        pass

    def record_error(
        self,
        error_type: str
    ) -> None:
        """
        Record an error.

        TODO: Implement error recording
        - Increment error counter
        - Label with model, error type

        Args:
            error_type: Type of error (timeout, oom, invalid_input, etc.)
        """
        # TODO: Increment error counter
        # llm_errors_total.labels(
        #     model=self.model_name,
        #     error_type=error_type
        # ).inc()

        pass

    def update_gpu_metrics(
        self,
        gpu_id: int,
        memory_used_bytes: int,
        utilization_percent: float
    ) -> None:
        """
        Update GPU metrics.

        TODO: Implement GPU metrics update
        - Set GPU memory gauge
        - Set GPU utilization gauge

        Args:
            gpu_id: GPU device ID
            memory_used_bytes: Memory used in bytes
            utilization_percent: GPU utilization (0-100)
        """
        # TODO: Update GPU metrics
        # llm_gpu_memory_used_bytes.labels(
        #     gpu_id=str(gpu_id),
        #     model=self.model_name
        # ).set(memory_used_bytes)
        #
        # llm_gpu_utilization_percent.labels(
        #     gpu_id=str(gpu_id),
        #     model=self.model_name
        # ).set(utilization_percent)

        pass

    def set_active_requests(self, count: int) -> None:
        """
        Set number of active requests.

        TODO: Implement active request tracking
        - Set gauge to current count

        Args:
            count: Current number of active requests
        """
        # TODO: Set active requests gauge
        # llm_active_requests.labels(
        #     model=self.model_name
        # ).set(count)

        pass


# ============================================================================
# DECORATORS
# ============================================================================

def track_request(
    collector: MetricsCollector,
    endpoint: str
):
    """
    Decorator to track request metrics.

    TODO: Implement request tracking decorator
    - Track request count
    - Track latency
    - Track errors
    - Track active requests

    Args:
        collector: MetricsCollector instance
        endpoint: Endpoint name

    Example:
        @track_request(metrics_collector, "/generate")
        async def generate(request):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # TODO: Increment active requests
            # collector.set_active_requests(...)

            # TODO: Track start time
            # start_time = time.time()

            try:
                # TODO: Execute function
                # result = await func(*args, **kwargs)

                # TODO: Record success
                # duration = time.time() - start_time
                # collector.record_request(endpoint, status="success")
                # collector.record_latency(endpoint, duration)

                # return result
                pass

            except Exception as e:
                # TODO: Record error
                # duration = time.time() - start_time
                # collector.record_request(endpoint, status="error")
                # collector.record_latency(endpoint, duration)
                # collector.record_error(type(e).__name__)
                # raise

            finally:
                # TODO: Decrement active requests
                # collector.set_active_requests(...)
                pass

        return wrapper
    return decorator


def track_tokens(collector: MetricsCollector):
    """
    Decorator to track token usage.

    TODO: Implement token tracking decorator
    - Extract token counts from response
    - Record tokens
    - Calculate cost

    Args:
        collector: MetricsCollector instance
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # TODO: Execute function
            # result = await func(*args, **kwargs)

            # TODO: Extract token counts from result
            # if hasattr(result, 'usage'):
            #     input_tokens = result.usage.prompt_tokens
            #     output_tokens = result.usage.completion_tokens
            #
            #     collector.record_tokens(input_tokens, output_tokens)
            #     collector.record_cost(input_tokens, output_tokens)

            # return result
            pass

        return wrapper
    return decorator


# ============================================================================
# GPU MONITORING
# ============================================================================

class GPUMonitor:
    """
    Monitor GPU metrics.

    TODO: Implement GPU monitoring
    - Query GPU stats periodically
    - Update Prometheus metrics
    - Support multiple GPUs
    - Handle NVIDIA and AMD GPUs
    """

    def __init__(
        self,
        collector: MetricsCollector,
        update_interval: int = 5
    ):
        """
        Initialize GPU monitor.

        Args:
            collector: MetricsCollector instance
            update_interval: Update interval in seconds
        """
        # TODO: Store configuration
        # self.collector = collector
        # self.update_interval = update_interval

        # TODO: Check if nvidia-smi available
        # self.has_nvidia = self._check_nvidia()

        pass

    def _check_nvidia(self) -> bool:
        """
        Check if NVIDIA GPU available.

        TODO: Implement NVIDIA check
        - Try importing pynvml
        - Initialize NVML
        - Return availability

        Returns:
            True if NVIDIA GPU available
        """
        # TODO: Check for NVIDIA GPU
        # try:
        #     import pynvml
        #     pynvml.nvmlInit()
        #     return True
        # except Exception:
        #     return False

        pass

    def get_gpu_stats(self) -> list:
        """
        Get current GPU statistics.

        TODO: Implement GPU stats collection
        - Query all GPUs
        - Get memory usage
        - Get utilization
        - Return list of stats

        Returns:
            List of dicts with GPU stats
        """
        # TODO: Get GPU stats
        # try:
        #     import pynvml
        #     device_count = pynvml.nvmlDeviceGetCount()
        #     stats = []
        #
        #     for i in range(device_count):
        #         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        #         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #         util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        #
        #         stats.append({
        #             "gpu_id": i,
        #             "memory_used": mem_info.used,
        #             "memory_total": mem_info.total,
        #             "utilization": util.gpu
        #         })
        #
        #     return stats
        # except Exception as e:
        #     logger.error(f"Failed to get GPU stats: {e}")
        #     return []

        pass

    def update_metrics(self) -> None:
        """
        Update GPU metrics.

        TODO: Implement metrics update
        - Get current GPU stats
        - Update Prometheus metrics
        """
        # TODO: Update metrics
        # stats = self.get_gpu_stats()
        # for stat in stats:
        #     self.collector.update_gpu_metrics(
        #         gpu_id=stat["gpu_id"],
        #         memory_used_bytes=stat["memory_used"],
        #         utilization_percent=stat["utilization"]
        #     )

        pass

    async def start_monitoring(self) -> None:
        """
        Start continuous GPU monitoring.

        TODO: Implement continuous monitoring
        - Run update loop
        - Sleep between updates
        - Handle shutdown gracefully
        """
        # TODO: Continuous monitoring
        # import asyncio
        # while True:
        #     self.update_metrics()
        #     await asyncio.sleep(self.update_interval)

        pass


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

def create_metrics_endpoint():
    """
    Create FastAPI endpoint for Prometheus metrics.

    TODO: Implement metrics endpoint
    - Return Prometheus metrics format
    - Set correct content type

    Returns:
        FastAPI endpoint function
    """
    # TODO: Create endpoint
    # from fastapi import Response
    #
    # async def metrics():
    #     return Response(
    #         content=generate_latest(),
    #         media_type=CONTENT_TYPE_LATEST
    #     )
    #
    # return metrics

    pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example Usage:

from fastapi import FastAPI
from .metrics import MetricsCollector, create_metrics_endpoint, track_request

# Initialize metrics collector
metrics = MetricsCollector(
    model_name="llama-2-7b-chat",
    cost_per_1k_tokens={
        "input": 0.0002,
        "output": 0.0002
    }
)

# Create FastAPI app
app = FastAPI()

# Add metrics endpoint
app.get("/metrics")(create_metrics_endpoint())

# Use decorator on endpoints
@app.post("/generate")
@track_request(metrics, "/generate")
async def generate(request: GenerateRequest):
    # Process request
    result = llm_server.generate(request.prompt)

    # Record tokens and cost
    metrics.record_tokens(
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens
    )
    cost = metrics.record_cost(
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
        customer=request.customer_id
    )

    return result

# Start GPU monitoring
import asyncio

gpu_monitor = GPUMonitor(metrics, update_interval=5)
asyncio.create_task(gpu_monitor.start_monitoring())

# Manual metric recording
metrics.record_request("/generate", status="success")
metrics.record_latency("/generate", 0.342)
metrics.record_tokens(input_tokens=50, output_tokens=100)
metrics.record_error("timeout")

# Query metrics (Prometheus will scrape /metrics endpoint)
# Example Prometheus queries:
#
# Request rate:
#   rate(llm_requests_total[5m])
#
# Error rate:
#   rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])
#
# P95 latency:
#   histogram_quantile(0.95, llm_request_duration_seconds_bucket)
#
# Tokens per second:
#   rate(llm_tokens_total{direction="output"}[1m])
#
# GPU utilization:
#   avg(llm_gpu_utilization_percent)
#
# Total cost:
#   llm_cost_usd_total
"""
