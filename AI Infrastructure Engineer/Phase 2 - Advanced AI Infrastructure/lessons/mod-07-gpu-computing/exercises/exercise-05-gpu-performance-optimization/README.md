# Exercise 05: GPU Performance Optimization and Profiling

**Estimated Time**: 32-40 hours

## Business Context

Your ML team is spending $180,000/month on GPU compute for model training. Recent profiling shows:
- Average GPU utilization: 45% (should be >80%)
- Training throughput: 2,400 samples/sec (expected 5,000+)
- Memory utilization: 60% of available VRAM
- Multi-GPU scaling efficiency: 65% (should be >90%)

The VP of Engineering has tasked you with **optimizing GPU performance to reduce training time by 50%** and decrease cloud costs by $90,000/month. You need to build a comprehensive GPU profiling and optimization framework.

## Learning Objectives

After completing this exercise, you will be able to:

1. Profile GPU workloads using NVIDIA Nsight Systems, Nsight Compute, and PyTorch Profiler
2. Identify and fix GPU performance bottlenecks (kernel efficiency, memory bandwidth, data loading)
3. Optimize CUDA kernels and mixed-precision training
4. Implement efficient multi-GPU data parallelism with optimal communication patterns
5. Apply advanced memory optimizations (gradient checkpointing, activation offloading, pinned memory)
6. Design automated performance regression detection systems

## Prerequisites

- Module 07 Exercise 04 (GPU Cluster Management)
- Understanding of deep learning training loops
- Basic familiarity with PyTorch or TensorFlow
- Linux command-line proficiency
- Python programming (advanced level)

## Problem Statement

Build a **GPU Performance Optimization Framework** that:

1. **Profiles GPU workloads** to identify bottlenecks
2. **Recommends optimizations** based on profiling data
3. **Applies automated fixes** for common performance issues
4. **Validates improvements** through A/B testing
5. **Detects performance regressions** in CI/CD pipelines

### Success Metrics

- Training throughput increased by 100%+ (2,400 → 5,000+ samples/sec)
- GPU utilization improved to >85%
- Multi-GPU scaling efficiency >90% for 2-8 GPUs
- Memory utilization optimized to >80% without OOM errors
- Automated detection of performance regressions >5%

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Performance Optimization System            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Profile Collector│      │  Bottleneck      │            │
│  │  - Nsight Systems │─────▶│  Analyzer        │            │
│  │  - Nsight Compute │      │  - Kernel time   │            │
│  │  - PyTorch Profile│      │  - Memory BW     │            │
│  │  - DCGM metrics   │      │  - Data loading  │            │
│  └──────────────────┘      └──────────────────┘            │
│           │                          │                       │
│           │                          ▼                       │
│           │                 ┌──────────────────┐            │
│           │                 │  Optimization    │            │
│           │                 │  Recommender     │            │
│           │                 │  - Mixed prec.   │            │
│           │                 │  - Kernel fusion │            │
│           │                 │  - Memory opts   │            │
│           │                 └──────────────────┘            │
│           │                          │                       │
│           ▼                          ▼                       │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Automated       │      │  Performance     │            │
│  │  Optimizer       │◀─────│  Validator       │            │
│  │  - Auto mixed    │      │  - A/B testing   │            │
│  │  - Gradient ckpt │      │  - Regression    │            │
│  │  - Data prefetch │      │    detection     │            │
│  └──────────────────┘      └──────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Part 1: GPU Profiling Framework (8-10 hours)

Build a comprehensive profiling system that collects performance data from multiple sources.

#### 1.1 Profile Collector

Create `src/profiling/profile_collector.py`:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import subprocess
import json
from pathlib import Path

class ProfilerType(Enum):
    NSIGHT_SYSTEMS = "nsys"
    NSIGHT_COMPUTE = "ncu"
    PYTORCH_PROFILER = "pytorch"
    DCGM = "dcgm"

@dataclass
class ProfileResult:
    profiler_type: ProfilerType
    profile_path: str
    summary: Dict
    recommendations: List[str]

class NsightSystemsProfiler:
    """
    Profile entire application using NVIDIA Nsight Systems.
    Identifies high-level bottlenecks: kernels, data transfers, CPU overhead.
    """

    def profile(
        self,
        command: str,
        output_path: str,
        duration: Optional[int] = None
    ) -> ProfileResult:
        """
        TODO: Run nsys profile command
        - Execute: nsys profile -o {output_path} --stats=true {command}
        - Capture CUDA kernel times, memory transfers, CPU activity
        - Parse output report (JSON format)
        - Extract key metrics:
          - Total kernel time (should be >80% of GPU time)
          - Memory transfer time (should be <10%)
          - CPU overhead (gaps between kernel launches)

        Example output to parse:
        {
            "cuda_kernels": [
                {"name": "volta_sgemm", "time_ns": 1234567, "grid": [128, 1, 1]}
            ],
            "cuda_memory": [
                {"type": "HtoD", "size_bytes": 4096, "time_ns": 500}
            ]
        }
        """
        pass

    def analyze_timeline(self, profile_path: str) -> Dict:
        """
        TODO: Analyze timeline for common issues
        - GPU idle periods (gaps between kernels >1ms)
        - CPU-GPU synchronization points (cudaDeviceSynchronize)
        - Memory transfer overlaps with compute
        - Return bottleneck report:
          {
              "gpu_idle_time_ms": 150,
              "sync_overhead_ms": 45,
              "memory_transfer_concurrent": True/False
          }
        """
        pass

class NsightComputeProfiler:
    """
    Profile individual CUDA kernels using NVIDIA Nsight Compute.
    Provides detailed metrics: SM efficiency, memory bandwidth, occupancy.
    """

    def profile_kernel(
        self,
        command: str,
        kernel_name: str,
        output_path: str
    ) -> ProfileResult:
        """
        TODO: Profile specific kernel with detailed metrics
        - Execute: ncu --kernel-name {kernel_name} -o {output_path} {command}
        - Collect metrics:
          - SM efficiency (should be >80%)
          - Memory bandwidth utilization (should be >60% of peak)
          - Achieved occupancy (should be >50%)
          - Warp execution efficiency
          - Register/shared memory usage

        Key metrics to extract:
        - sm__throughput.avg.pct_of_peak_sustained_elapsed
        - dram__throughput.avg.pct_of_peak_sustained_elapsed
        - sm__warps_active.avg.pct_of_peak_sustained_active
        """
        pass

    def identify_bottlenecks(self, profile_result: ProfileResult) -> List[str]:
        """
        TODO: Identify kernel-level bottlenecks
        - If SM efficiency <60%: "Compute bound - optimize algorithm"
        - If memory BW >80% but SM <60%: "Memory bound - reduce memory access"
        - If occupancy <30%: "Low occupancy - reduce register usage or increase blocks"
        - If warp efficiency <80%: "Branch divergence - reduce conditional code"
        """
        pass

class PyTorchProfiler:
    """
    Profile PyTorch training loops with operator-level granularity.
    """

    def profile_training_step(
        self,
        model,
        data_loader,
        num_steps: int = 10,
        output_path: str = "./pytorch_profile"
    ) -> ProfileResult:
        """
        TODO: Profile PyTorch training using torch.profiler

        import torch.profiler as profiler

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=profiler.tensorboard_trace_handler(output_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for step, batch in enumerate(data_loader):
                if step >= num_steps:
                    break

                # TODO: Run training step
                outputs = model(batch['input'])
                loss = criterion(outputs, batch['target'])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                prof.step()

        # TODO: Parse profiler output
        # Extract top 10 time-consuming operations
        # Identify data loading bottlenecks (CPU time while GPU idle)
        # Memory allocation overhead
        """
        pass

    def analyze_operator_breakdown(self, profile_path: str) -> Dict:
        """
        TODO: Analyze which PyTorch operators consume most time
        - Group by operation type (conv2d, matmul, activation, etc.)
        - Calculate % of total training time
        - Identify optimization opportunities:
          - Can operations be fused? (e.g., conv + bn + relu)
          - Are there unnecessary data copies?
          - Is autocast (mixed precision) being used?
        """
        pass

class ProfileCollector:
    """Main interface for collecting all profiling data."""

    def __init__(self):
        self.nsys_profiler = NsightSystemsProfiler()
        self.ncu_profiler = NsightComputeProfiler()
        self.pytorch_profiler = PyTorchProfiler()

    def profile_comprehensive(
        self,
        command: str,
        output_dir: str,
        include_kernel_details: bool = True
    ) -> Dict[ProfilerType, ProfileResult]:
        """
        TODO: Run all profilers and collect results
        1. Nsight Systems for high-level timeline
        2. Nsight Compute for top 5 kernels (if include_kernel_details)
        3. PyTorch Profiler for operator breakdown

        Return consolidated report with all findings
        """
        pass
```

#### 1.2 Testing Requirements

Create `tests/test_profiling.py`:

```python
def test_nsight_systems_profiling():
    """Test that Nsight Systems captures kernel times correctly."""
    # TODO: Run simple CUDA program and verify profiling works
    pass

def test_pytorch_profiler_integration():
    """Test PyTorch profiler with dummy model."""
    # TODO: Profile simple model training and verify output
    pass
```

### Part 2: Bottleneck Analysis Engine (7-9 hours)

Build an automated system to analyze profiling data and identify performance issues.

#### 2.1 Bottleneck Analyzer

Create `src/analysis/bottleneck_analyzer.py`:

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class BottleneckType(Enum):
    KERNEL_EFFICIENCY = "kernel_efficiency"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    DATA_LOADING = "data_loading"
    CPU_GPU_SYNC = "cpu_gpu_sync"
    MULTI_GPU_COMM = "multi_gpu_comm"
    MEMORY_ALLOCATION = "memory_allocation"

@dataclass
class Bottleneck:
    type: BottleneckType
    severity: float  # 0-1, where 1 is critical
    description: str
    evidence: Dict
    recommendations: List[str]

class KernelAnalyzer:
    """Analyze CUDA kernel performance."""

    THRESHOLDS = {
        'sm_efficiency_min': 60.0,      # SM efficiency should be >60%
        'memory_bw_utilization_min': 60.0,  # Memory BW should be >60%
        'occupancy_min': 30.0,          # Occupancy should be >30%
        'warp_efficiency_min': 80.0     # Warp efficiency should be >80%
    }

    def analyze(self, kernel_metrics: Dict) -> List[Bottleneck]:
        """
        TODO: Analyze kernel performance metrics

        Check each metric against thresholds:
        1. SM Efficiency:
           - <40%: Critical - kernel is severely underutilizing GPU
           - 40-60%: Warning - room for optimization
           - >60%: Good

        2. Memory Bandwidth:
           - >80% with low SM efficiency: Memory bound
           - Recommend: reduce memory access, use shared memory, coalesce access

        3. Occupancy:
           - <30%: Too few threads per SM
           - Recommend: increase block size, reduce register usage

        4. Warp Efficiency:
           - <80%: Branch divergence or uncoalesced memory access
           - Recommend: remove conditionals, align memory access

        Return list of Bottleneck objects with specific recommendations
        """
        bottlenecks = []

        # TODO: Implement threshold checks
        sm_efficiency = kernel_metrics.get('sm_efficiency', 0)
        if sm_efficiency < self.THRESHOLDS['sm_efficiency_min']:
            severity = 1.0 - (sm_efficiency / 100.0)
            bottlenecks.append(Bottleneck(
                type=BottleneckType.KERNEL_EFFICIENCY,
                severity=severity,
                description=f"Low SM efficiency: {sm_efficiency:.1f}%",
                evidence={'sm_efficiency': sm_efficiency},
                recommendations=[
                    "Profile with Nsight Compute to identify compute bottlenecks",
                    "Consider kernel fusion to reduce launch overhead",
                    "Optimize algorithm for better parallelism"
                ]
            ))

        # TODO: Add similar checks for memory BW, occupancy, warp efficiency

        return bottlenecks

class DataLoadingAnalyzer:
    """Analyze data loading pipeline performance."""

    def analyze(self, timeline_data: Dict) -> List[Bottleneck]:
        """
        TODO: Detect data loading bottlenecks

        Look for patterns:
        1. GPU idle while CPU prepares next batch:
           - Evidence: Gaps >5ms between kernel launches
           - Recommendation: Increase num_workers in DataLoader

        2. Data transfer time > compute time:
           - Evidence: H2D memory transfers take >50% of step time
           - Recommendations:
             - Use pinned memory (pin_memory=True)
             - Prefetch to GPU (non_blocking=True)
             - Reduce batch size if data doesn't fit in GPU memory

        3. CPU preprocessing bottleneck:
           - Evidence: CPU time > GPU time
           - Recommendations:
             - Move preprocessing to GPU (using DALI or kornia)
             - Simplify transforms
             - Use faster image decoding (turbojpeg)
        """
        pass

class MultiGPUAnalyzer:
    """Analyze multi-GPU communication patterns."""

    def analyze(self, nccl_metrics: Dict, num_gpus: int) -> List[Bottleneck]:
        """
        TODO: Analyze multi-GPU scaling efficiency

        Calculate scaling efficiency:
        - Ideal speedup: N GPUs → N× faster
        - Actual speedup = (single_gpu_time / multi_gpu_time)
        - Scaling efficiency = (actual_speedup / num_gpus) × 100%

        Bottlenecks:
        1. Gradient synchronization overhead:
           - If NCCL allreduce time > 10% of step time
           - Recommendations:
             - Use gradient accumulation to amortize comm cost
             - Enable NCCL_P2P_DISABLE=0 for NVLink
             - Use FP16 gradients (automatic with mixed precision)

        2. Load imbalance:
           - If max_gpu_time / min_gpu_time > 1.1
           - Some GPUs finish earlier, wait for stragglers
           - Recommendation: Balance data distribution

        3. Communication-computation overlap:
           - DDP should overlap gradient allreduce with backward pass
           - Check if gradient_as_bucket_view=True
        """
        pass

class BottleneckAnalyzer:
    """Main bottleneck analysis coordinator."""

    def __init__(self):
        self.kernel_analyzer = KernelAnalyzer()
        self.data_loading_analyzer = DataLoadingAnalyzer()
        self.multi_gpu_analyzer = MultiGPUAnalyzer()

    def analyze_all(
        self,
        profile_results: Dict
    ) -> List[Bottleneck]:
        """
        TODO: Run all analyzers and consolidate findings
        1. Analyze kernel performance
        2. Analyze data loading pipeline
        3. Analyze multi-GPU communication (if applicable)
        4. Rank bottlenecks by severity
        5. Return top 5 most critical issues
        """
        pass

    def generate_report(self, bottlenecks: List[Bottleneck]) -> str:
        """
        TODO: Generate human-readable report

        Format:
        === GPU Performance Analysis Report ===

        Critical Issues (severity >0.7):
        1. [KERNEL_EFFICIENCY] Low SM efficiency: 45.2%
           Recommendations:
           - Profile with Nsight Compute to identify compute bottlenecks
           - Consider kernel fusion to reduce launch overhead

        Warnings (severity 0.4-0.7):
        2. [DATA_LOADING] GPU idle time detected: 12% of training time
           Recommendations:
           - Increase DataLoader num_workers from 4 to 8
           - Enable pinned memory

        Estimated Performance Improvement: 2.1× faster training
        """
        pass
```

### Part 3: Automated Optimization Framework (9-11 hours)

Implement automated fixes for common performance issues.

#### 3.1 Optimization Strategies

Create `src/optimization/optimizers.py`:

```python
from typing import Dict, Any
import torch
import torch.nn as nn

class MixedPrecisionOptimizer:
    """
    Implement automatic mixed precision (AMP) training.
    Uses FP16 for compute-intensive ops, FP32 for numerically sensitive ops.

    Benefits:
    - 2-3× faster training on Tensor Core GPUs (V100, A100)
    - 50% reduction in memory usage
    - Minimal accuracy impact (<0.1% typically)
    """

    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def wrap_training_step(self, model, criterion, optimizer):
        """
        TODO: Wrap training step with autocast

        def training_step(batch):
            with torch.cuda.amp.autocast():
                outputs = model(batch['input'])
                loss = criterion(outputs, batch['target'])

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            return loss.item()

        return training_step
        """
        pass

    def estimate_speedup(self, model: nn.Module) -> float:
        """
        TODO: Estimate speedup from mixed precision

        Count operations by type:
        - MatMul, Conv2d: High speedup (2-3×) from Tensor Cores
        - BatchNorm, LayerNorm: Medium speedup (1.5×)
        - Activations: Low speedup (1.2×)

        Weighted average based on operation counts
        """
        pass

class GradientCheckpointingOptimizer:
    """
    Implement gradient checkpointing to reduce memory usage.

    Trade-off:
    - Memory: 40-60% reduction (enables larger batch sizes)
    - Speed: 20-30% slower (recompute activations during backward)

    Net benefit: Often 1.5× faster due to larger batches
    """

    def apply_checkpointing(
        self,
        model: nn.Module,
        checkpoint_segments: int = 4
    ) -> nn.Module:
        """
        TODO: Apply gradient checkpointing to model

        For transformer models:
        - Checkpoint every N transformer blocks
        - Keep final layer activations (small memory cost)

        Example for HuggingFace models:
        model.gradient_checkpointing_enable()

        For custom models:
        from torch.utils.checkpoint import checkpoint

        def forward_with_checkpointing(x):
            # Divide model into segments
            for segment in self.segments:
                x = checkpoint(segment, x)
            return x
        """
        pass

class DataLoadingOptimizer:
    """Optimize PyTorch DataLoader configuration."""

    def optimize_dataloader(
        self,
        dataset,
        batch_size: int,
        current_config: Dict
    ) -> Dict:
        """
        TODO: Recommend optimal DataLoader settings

        Determine optimal num_workers:
        - Run benchmark with num_workers = [0, 2, 4, 8, 16]
        - Measure samples/sec for each
        - Select configuration with highest throughput

        Other optimizations:
        - pin_memory=True: Faster H2D transfers (uses pinned memory)
        - prefetch_factor=2: Prefetch 2 batches per worker
        - persistent_workers=True: Keep workers alive between epochs

        Return:
        {
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'estimated_speedup': 1.4
        }
        """
        pass

class KernelFusionOptimizer:
    """Optimize model by fusing operations."""

    def fuse_conv_bn_relu(self, model: nn.Module) -> nn.Module:
        """
        TODO: Fuse Conv2d + BatchNorm2d + ReLU

        Reduces kernel launches from 3 to 1:
        - Conv2d: y = Wx + b
        - BatchNorm: y = γ(y - μ) / σ + β
        - ReLU: y = max(0, y)

        Fused kernel does all three in one pass

        For inference (eval mode):
        torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

        For training:
        More complex - need custom fused op or use torch.compile()
        """
        pass

    def apply_torch_compile(
        self,
        model: nn.Module,
        mode: str = "reduce-overhead"
    ) -> nn.Module:
        """
        TODO: Apply torch.compile() for automatic kernel fusion

        PyTorch 2.0+ feature - automatically fuses operations

        model = torch.compile(model, mode=mode)

        Modes:
        - "reduce-overhead": Optimize for many small ops (RNNs)
        - "max-autotune": Maximum optimization, longer compile time
        - "default": Balanced

        Typical speedup: 1.3-2× depending on model
        """
        pass

class MemoryOptimizer:
    """Optimize GPU memory usage."""

    def find_optimal_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 512
    ) -> int:
        """
        TODO: Binary search for maximum batch size that fits in memory

        Algorithm:
        1. Start with batch_size = max_batch_size
        2. Try forward + backward pass
        3. If OOM: batch_size //= 2
        4. If success: try batch_size * 1.5
        5. Repeat until convergence

        Return largest batch size that doesn't OOM with 10% safety margin
        """
        pass

    def enable_activation_checkpointing(
        self,
        model: nn.Module,
        checkpoint_ratio: float = 0.5
    ):
        """
        TODO: Checkpoint 50% of transformer blocks to save memory
        See GradientCheckpointingOptimizer for implementation
        """
        pass

class AutoOptimizer:
    """Automatically apply optimizations based on bottleneck analysis."""

    def __init__(self):
        self.mixed_precision = MixedPrecisionOptimizer()
        self.gradient_checkpointing = GradientCheckpointingOptimizer()
        self.data_loading = DataLoadingOptimizer()
        self.kernel_fusion = KernelFusionOptimizer()
        self.memory = MemoryOptimizer()

    def optimize(
        self,
        model: nn.Module,
        data_loader,
        bottlenecks: List[Bottleneck],
        config: Dict
    ) -> Dict[str, Any]:
        """
        TODO: Apply optimizations based on identified bottlenecks

        Decision tree:
        1. If KERNEL_EFFICIENCY bottleneck:
           - Apply mixed precision (AMP)
           - Apply torch.compile() for kernel fusion

        2. If MEMORY_BANDWIDTH bottleneck:
           - Enable gradient checkpointing
           - Find optimal batch size

        3. If DATA_LOADING bottleneck:
           - Optimize DataLoader (num_workers, pinned memory)

        4. If MULTI_GPU_COMM bottleneck:
           - Enable gradient accumulation
           - Use DDP with gradient_as_bucket_view=True

        Return:
        {
            'optimizations_applied': ['mixed_precision', 'torch_compile'],
            'estimated_speedup': 2.3,
            'memory_saved_gb': 4.2,
            'new_config': {...}
        }
        """
        pass
```

#### 3.2 Testing Requirements

Create `tests/test_optimizations.py`:

```python
def test_mixed_precision_speedup():
    """Verify AMP provides speedup on Tensor Core GPUs."""
    # TODO: Compare training time with/without AMP
    pass

def test_gradient_checkpointing_memory():
    """Verify gradient checkpointing reduces memory usage."""
    # TODO: Measure memory usage with/without checkpointing
    pass
```

### Part 4: Performance Validation & Regression Detection (8-10 hours)

Build a system to validate optimizations and detect regressions.

#### 4.1 Performance Validator

Create `src/validation/performance_validator.py`:

```python
from dataclasses import dataclass
from typing import Dict, List
import time
import torch

@dataclass
class PerformanceMetrics:
    samples_per_second: float
    gpu_utilization: float
    memory_usage_gb: float
    multi_gpu_scaling_efficiency: float
    step_time_ms: float

class PerformanceBenchmark:
    """Run controlled performance benchmarks."""

    def benchmark_training(
        self,
        model: torch.nn.Module,
        data_loader,
        num_steps: int = 100,
        warmup_steps: int = 10
    ) -> PerformanceMetrics:
        """
        TODO: Benchmark training performance

        1. Warmup phase (10 steps):
           - Allows GPU clocks to ramp up
           - Fills caches
           - Stabilizes memory allocator

        2. Measurement phase (100 steps):
           - Record time for each step
           - Sample GPU metrics every 10 steps:
             - nvidia-smi for utilization, memory
             - DCGM for detailed metrics

        3. Calculate metrics:
           - samples_per_second = (num_steps × batch_size) / total_time
           - gpu_utilization = average SM activity %
           - memory_usage_gb = peak memory allocated
           - step_time_ms = average time per step

        Return PerformanceMetrics object
        """
        pass

    def benchmark_multi_gpu_scaling(
        self,
        model: torch.nn.Module,
        data_loader,
        gpu_counts: List[int] = [1, 2, 4, 8]
    ) -> Dict[int, PerformanceMetrics]:
        """
        TODO: Measure multi-GPU scaling efficiency

        For each GPU count:
        1. Initialize DDP with world_size = gpu_count
        2. Run benchmark
        3. Calculate scaling efficiency:
           speedup = time_1gpu / time_Ngpu
           efficiency = speedup / N

        Return metrics for each configuration
        """
        pass

class ABTestValidator:
    """A/B test baseline vs optimized configurations."""

    def compare_configurations(
        self,
        baseline_config: Dict,
        optimized_config: Dict,
        model_fn,
        data_loader,
        num_runs: int = 3
    ) -> Dict:
        """
        TODO: Statistical comparison of configurations

        1. Run each configuration num_runs times
        2. Calculate mean and std dev for metrics
        3. Perform t-test to determine statistical significance
        4. Report improvement:
           - Speedup: optimized_samples_per_sec / baseline_samples_per_sec
           - Memory savings: baseline_memory_gb - optimized_memory_gb
           - Confidence interval (95%)

        Return:
        {
            'speedup': 2.1,
            'speedup_confidence_interval': (1.9, 2.3),
            'p_value': 0.001,
            'statistically_significant': True,
            'baseline_metrics': PerformanceMetrics(...),
            'optimized_metrics': PerformanceMetrics(...)
        }
        """
        pass

class RegressionDetector:
    """Detect performance regressions in CI/CD pipeline."""

    def __init__(self, metrics_db_path: str):
        self.metrics_db_path = metrics_db_path
        # TODO: Initialize connection to metrics database (SQLite or PostgreSQL)

    def record_benchmark(
        self,
        commit_sha: str,
        branch: str,
        metrics: PerformanceMetrics,
        metadata: Dict
    ):
        """
        TODO: Record benchmark results to database

        Table schema:
        CREATE TABLE benchmarks (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            commit_sha VARCHAR(40),
            branch VARCHAR(100),
            samples_per_second FLOAT,
            gpu_utilization FLOAT,
            memory_usage_gb FLOAT,
            step_time_ms FLOAT,
            gpu_type VARCHAR(50),
            num_gpus INT,
            batch_size INT
        );

        Insert current benchmark results
        """
        pass

    def detect_regression(
        self,
        current_metrics: PerformanceMetrics,
        baseline_branch: str = "main",
        regression_threshold: float = 0.05  # 5% slowdown
    ) -> Dict:
        """
        TODO: Detect if current performance is worse than baseline

        1. Query last 10 benchmarks from baseline_branch
        2. Calculate baseline mean and std dev
        3. Compare current metrics:
           - If current_samples_per_sec < baseline_mean × (1 - threshold):
             REGRESSION DETECTED

        4. Identify likely culprit:
           - Binary search through recent commits
           - Find first commit where regression appears

        Return:
        {
            'regression_detected': True/False,
            'current_samples_per_sec': 4200,
            'baseline_samples_per_sec': 4800,
            'performance_delta': -12.5,  # % change
            'threshold': -5.0,
            'likely_culprit_commit': 'abc123def',
            'recommendation': 'Revert commit or investigate...'
        }
        """
        pass

    def generate_performance_report(
        self,
        start_date: str,
        end_date: str
    ) -> str:
        """
        TODO: Generate performance trends over time

        Query benchmarks in date range
        Generate matplotlib charts:
        1. Samples/sec over time (line chart)
        2. GPU utilization over time
        3. Memory usage over time

        Identify:
        - Performance improvements (commits that increased throughput)
        - Regressions (commits that decreased throughput)
        - Trends (gradual improvement/degradation)

        Save charts and return summary report
        """
        pass

class PerformanceValidator:
    """Main validation coordinator."""

    def __init__(self, metrics_db_path: str):
        self.benchmark = PerformanceBenchmark()
        self.ab_test = ABTestValidator()
        self.regression_detector = RegressionDetector(metrics_db_path)

    def validate_optimization(
        self,
        baseline_model,
        optimized_model,
        data_loader,
        commit_sha: str
    ) -> Dict:
        """
        TODO: Complete validation workflow

        1. Benchmark baseline configuration
        2. Benchmark optimized configuration
        3. A/B test comparison with statistical significance
        4. Record results to database
        5. Check for regressions against main branch
        6. Generate validation report

        Return comprehensive validation results
        """
        pass
```

#### 4.2 CI/CD Integration

Create `.github/workflows/performance-benchmark.yml`:

```yaml
name: GPU Performance Benchmark

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu, a100]  # GPU runner required

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

      - name: Run performance benchmark
        run: |
          python scripts/run_benchmark.py \
            --commit-sha ${{ github.sha }} \
            --branch ${{ github.ref_name }} \
            --output benchmark_results.json

      - name: Detect performance regression
        id: regression
        run: |
          python scripts/detect_regression.py \
            --results benchmark_results.json \
            --threshold 0.05

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('benchmark_results.json'));

            const comment = `## GPU Performance Benchmark Results

            **Samples/sec**: ${results.samples_per_second.toFixed(0)} (${results.performance_delta > 0 ? '+' : ''}${results.performance_delta.toFixed(1)}%)
            **GPU Utilization**: ${results.gpu_utilization.toFixed(1)}%
            **Memory Usage**: ${results.memory_usage_gb.toFixed(2)} GB

            ${results.regression_detected ? '⚠️ **Performance regression detected!**' : '✅ No regression detected'}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: Fail if regression detected
        if: steps.regression.outputs.regression == 'true'
        run: |
          echo "Performance regression detected!"
          exit 1
```

## Acceptance Criteria

### Functional Requirements

- [ ] Profile collector captures data from Nsight Systems, Nsight Compute, and PyTorch Profiler
- [ ] Bottleneck analyzer identifies top 5 performance issues with severity scores
- [ ] Automated optimizer applies fixes (AMP, gradient checkpointing, DataLoader tuning)
- [ ] A/B testing validates optimizations with statistical significance (p < 0.05)
- [ ] Regression detector alerts on >5% performance degradation
- [ ] CI/CD integration runs benchmarks on every PR

### Performance Requirements

- [ ] Training throughput improved by 100%+ (2,400 → 5,000+ samples/sec)
- [ ] GPU utilization >85% (up from 45%)
- [ ] Multi-GPU scaling efficiency >90% for 2-8 GPUs
- [ ] Memory utilization >80% without OOM errors
- [ ] Profiling overhead <5% (minimal impact on training time)

### Code Quality

- [ ] All profiling and optimization code has 80%+ test coverage
- [ ] Comprehensive error handling (profile failures, OOM during optimization)
- [ ] Detailed logging for all optimization decisions
- [ ] Type hints for all public functions
- [ ] Docstrings with examples for all classes

## Testing Strategy

### Unit Tests

```python
# tests/test_kernel_analyzer.py
def test_kernel_bottleneck_detection():
    """Test that low SM efficiency is correctly flagged."""
    metrics = {'sm_efficiency': 35.0}
    analyzer = KernelAnalyzer()
    bottlenecks = analyzer.analyze(metrics)
    assert len(bottlenecks) > 0
    assert bottlenecks[0].type == BottleneckType.KERNEL_EFFICIENCY

# tests/test_mixed_precision.py
def test_amp_speedup():
    """Verify AMP provides 2×+ speedup on A100."""
    # Requires GPU testing environment
    pass
```

### Integration Tests

```python
# tests/test_end_to_end.py
def test_full_optimization_pipeline():
    """Test complete workflow: profile → analyze → optimize → validate."""
    # 1. Profile baseline
    # 2. Identify bottlenecks
    # 3. Apply optimizations
    # 4. Validate improvement
    pass
```

### Manual Testing

1. **Profile real training workload**:
   ```bash
   python src/profiling/profile_collector.py \
     --command "python train.py" \
     --output ./profiles/baseline
   ```

2. **Analyze bottlenecks**:
   ```bash
   python src/analysis/bottleneck_analyzer.py \
     --profile ./profiles/baseline \
     --output bottleneck_report.json
   ```

3. **Apply optimizations**:
   ```bash
   python src/optimization/auto_optimizer.py \
     --bottlenecks bottleneck_report.json \
     --model-config config/model.yaml \
     --output optimized_config.yaml
   ```

4. **Validate improvements**:
   ```bash
   python src/validation/performance_validator.py \
     --baseline config/model.yaml \
     --optimized optimized_config.yaml \
     --output validation_report.json
   ```

## Deliverables

1. **Source Code** (`src/`):
   - `profiling/profile_collector.py` - Multi-source profiling system
   - `analysis/bottleneck_analyzer.py` - Automated bottleneck detection
   - `optimization/optimizers.py` - Optimization strategies (AMP, checkpointing, etc.)
   - `validation/performance_validator.py` - A/B testing and regression detection

2. **Tests** (`tests/`):
   - Unit tests for all analyzers and optimizers
   - Integration test for full pipeline
   - Benchmark scripts for validation

3. **CI/CD Integration** (`.github/workflows/`):
   - `performance-benchmark.yml` - Automated benchmarking on PRs

4. **Documentation** (`docs/`):
   - `OPTIMIZATION_GUIDE.md` - How to use the optimization framework
   - `PROFILING_GUIDE.md` - How to profile GPU workloads
   - `BENCHMARKING.md` - Performance benchmarking best practices

5. **Example Scripts** (`scripts/`):
   - `run_benchmark.py` - Run performance benchmark
   - `detect_regression.py` - Detect performance regressions
   - `optimize_model.py` - Apply optimizations to existing training script

## Bonus Challenges

1. **Custom CUDA Kernel Optimization** (+8 hours):
   - Identify slow PyTorch operators in profiling data
   - Write optimized CUDA kernels for hotspots
   - Integrate custom kernels via PyTorch C++ extension
   - Validate 2-5× speedup for specific operations

2. **Multi-Node GPU Profiling** (+6 hours):
   - Profile multi-node distributed training
   - Identify network communication bottlenecks
   - Optimize NCCL communication patterns
   - Measure scaling efficiency up to 32 GPUs

3. **Energy Efficiency Optimization** (+5 hours):
   - Measure GPU power consumption (watts)
   - Optimize for performance-per-watt
   - Dynamic voltage/frequency scaling
   - Generate energy efficiency reports

## Resources

### Official Documentation

- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Tutorials

- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Sample Code

```python
# Example: Complete optimization workflow

from profiling.profile_collector import ProfileCollector
from analysis.bottleneck_analyzer import BottleneckAnalyzer
from optimization.optimizers import AutoOptimizer
from validation.performance_validator import PerformanceValidator

# 1. Profile baseline
collector = ProfileCollector()
profile_results = collector.profile_comprehensive(
    command="python train.py --config baseline.yaml",
    output_dir="./profiles/baseline"
)

# 2. Analyze bottlenecks
analyzer = BottleneckAnalyzer()
bottlenecks = analyzer.analyze_all(profile_results)
print(analyzer.generate_report(bottlenecks))

# 3. Apply optimizations
optimizer = AutoOptimizer()
optimizations = optimizer.optimize(
    model=model,
    data_loader=train_loader,
    bottlenecks=bottlenecks,
    config=baseline_config
)

# 4. Validate improvements
validator = PerformanceValidator(metrics_db_path="./metrics.db")
validation_results = validator.validate_optimization(
    baseline_model=baseline_model,
    optimized_model=optimized_model,
    data_loader=train_loader,
    commit_sha="abc123"
)

print(f"Speedup: {validation_results['speedup']:.2f}×")
print(f"Memory saved: {validation_results['memory_saved_gb']:.2f} GB")
```

## Submission

Submit your implementation via Git:

```bash
git add .
git commit -m "Complete Exercise 05: GPU Performance Optimization"
git push origin exercise-05-gpu-performance-optimization
```

Ensure your submission includes:
- All source code with comprehensive TODO comments implemented
- Unit and integration tests with >80% coverage
- CI/CD workflow for automated benchmarking
- Documentation with usage examples
- Performance benchmark results showing >2× improvement

---

**Estimated Time Breakdown**:
- Part 1 (Profiling Framework): 8-10 hours
- Part 2 (Bottleneck Analysis): 7-9 hours
- Part 3 (Automated Optimization): 9-11 hours
- Part 4 (Validation & Regression Detection): 8-10 hours
- **Total**: 32-40 hours
