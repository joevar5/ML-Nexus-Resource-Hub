# Lesson 08: Advanced GPU Optimization

## Learning Objectives

By the end of this lesson, you will be able to:

1. Profile GPU performance using NVIDIA Nsight and PyTorch Profiler
2. Identify and fix GPU performance bottlenecks
3. Optimize batch sizes for maximum throughput
4. Understand and optimize GPU kernel launches
5. Implement production-ready GPU monitoring
6. Apply best practices for GPU infrastructure in production

## Introduction

This lesson covers advanced GPU optimization techniques for production ML systems. We'll explore profiling tools, performance tuning, and operational best practices to maximize GPU utilization and cost-effectiveness.

## Performance Profiling Tools

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity
import torch

def profile_training_step(model, data, target):
    """Profile a single training step"""

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("forward"):
            output = model(data)

        with record_function("loss"):
            loss = F.cross_entropy(output, target)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    # Export for TensorBoard
    prof.export_chrome_trace("trace.json")

    return prof

# Usage
model = MyModel().cuda()
data = torch.randn(32, 3, 224, 224).cuda()
target = torch.randint(0, 10, (32,)).cuda()

prof = profile_training_step(model, data, target)
```

### NVIDIA Nsight Systems

```bash
# Profile entire Python script
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report.qdrep \
    python train.py

# View in Nsight Systems GUI
nsys-ui report.qdrep

# Command-line stats
nsys stats report.qdrep
```

### NVIDIA Nsight Compute

```bash
# Profile specific CUDA kernels
ncu \
    --set full \
    --export=kernel_profile \
    python train.py

# Analyze specific kernel
ncu \
    --kernel-name="volta_sgemm" \
    --launch-skip=10 \
    --launch-count=1 \
    python train.py
```

### Custom NVTX Markers

```python
# Add NVTX ranges for better profiling
import torch.cuda.nvtx as nvtx

def train_with_nvtx(model, train_loader):
    """Training with NVTX markers for profiling"""

    for epoch in range(num_epochs):
        nvtx.range_push(f"epoch_{epoch}")

        for batch_idx, (data, target) in enumerate(train_loader):
            nvtx.range_push("data_loading")
            data, target = data.cuda(), target.cuda()
            nvtx.range_pop()

            nvtx.range_push("forward")
            output = model(data)
            loss = criterion(output, target)
            nvtx.range_pop()

            nvtx.range_push("backward")
            loss.backward()
            nvtx.range_pop()

            nvtx.range_push("optimizer")
            optimizer.step()
            optimizer.zero_grad()
            nvtx.range_pop()

        nvtx.range_pop()  # End epoch

# NVTX ranges appear in Nsight Systems timeline
```

## Identifying Performance Bottlenecks

### GPU Utilization Analysis

```python
import nvidia_smi
import time

class GPUMonitor:
    """Real-time GPU monitoring"""

    def __init__(self):
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def get_utilization(self):
        """Get current GPU utilization"""
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)

        return {
            'gpu_util': util.gpu,
            'mem_util': util.memory,
            'mem_used_gb': mem.used / 1024**3,
            'mem_total_gb': mem.total / 1024**3
        }

    def diagnose_bottleneck(self):
        """Diagnose performance bottlenecks"""
        stats = self.get_utilization()

        print(f"GPU Utilization: {stats['gpu_util']}%")
        print(f"Memory Usage: {stats['mem_used_gb']:.1f} / {stats['mem_total_gb']:.1f} GB")

        # Diagnose issues
        if stats['gpu_util'] < 50:
            print("\n⚠ Low GPU Utilization!")
            print("Possible causes:")
            print("  - Data loading bottleneck")
            print("  - Small batch size")
            print("  - CPU preprocessing too slow")
            print("  - Excessive CPU-GPU synchronization")

        if stats['mem_util'] > 90:
            print("\n⚠ High Memory Pressure!")
            print("Possible optimizations:")
            print("  - Reduce batch size")
            print("  - Enable gradient checkpointing")
            print("  - Use mixed precision")

        if stats['gpu_util'] > 80 and stats['mem_util'] < 50:
            print("\n✓ Good GPU utilization, memory underutilized")
            print("  - Could increase batch size")

        if stats['gpu_util'] < 50 and stats['mem_util'] > 80:
            print("\n⚠ Memory bound but low utilization")
            print("  - Kernel launch overhead")
            print("  - Memory bandwidth bottleneck")

    def __del__(self):
        nvidia_smi.nvmlShutdown()

# Usage
monitor = GPUMonitor()
monitor.diagnose_bottleneck()
```

### Kernel Analysis

```python
def analyze_kernel_performance(prof):
    """Analyze individual kernel performance"""

    # Get CUDA kernels
    cuda_events = [
        event for event in prof.key_averages()
        if event.device_type == torch.profiler.DeviceType.CUDA
    ]

    # Sort by time
    cuda_events.sort(key=lambda x: x.cuda_time_total, reverse=True)

    print("Top 10 CUDA Kernels:")
    for i, event in enumerate(cuda_events[:10]):
        print(f"\n{i+1}. {event.key}")
        print(f"   Total Time: {event.cuda_time_total/1000:.2f} ms")
        print(f"   Calls: {event.count}")
        print(f"   Avg Time: {event.cuda_time_total/event.count/1000:.2f} ms")

        # Check for inefficiencies
        if event.count > 1000:
            print(f"   ⚠ High call count - consider batching")

        if event.cuda_time_total / prof.self_cpu_time_total > 0.5:
            print(f"   ⚠ Dominates runtime - optimize this kernel")
```

## Batch Size Optimization

### Finding Optimal Batch Size

```python
import time
import torch

def find_optimal_batch_size(model, input_shape, start_batch=1, max_batch=256):
    """
    Find batch size that maximizes throughput
    without exceeding memory limits
    """
    model = model.cuda()
    model.eval()

    results = []

    batch_size = start_batch
    while batch_size <= max_batch:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Create batch
            data = torch.randn(batch_size, *input_shape).cuda()

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(data)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                for _ in range(20):
                    _ = model(data)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Calculate throughput
            throughput = (batch_size * 20) / elapsed  # samples/sec
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB

            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'memory_gb': memory_used,
                'latency_ms': (elapsed / 20) * 1000
            })

            print(f"Batch {batch_size:4d}: "
                  f"{throughput:8.1f} samples/sec, "
                  f"{memory_used:5.2f} GB, "
                  f"{(elapsed/20)*1000:6.2f} ms")

            # Increase batch size
            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size}: OOM")
                break
            else:
                raise e

    # Find optimal
    optimal = max(results, key=lambda x: x['throughput'])

    print(f"\nOptimal batch size: {optimal['batch_size']}")
    print(f"  Throughput: {optimal['throughput']:.1f} samples/sec")
    print(f"  Memory: {optimal['memory_gb']:.2f} GB")
    print(f"  Latency: {optimal['latency_ms']:.2f} ms")

    return optimal

# Usage
model = ResNet50()
optimal = find_optimal_batch_size(model, input_shape=(3, 224, 224))
```

### Dynamic Batch Sizing

```python
class DynamicBatchSampler:
    """
    Dynamically adjust batch size based on GPU memory
    """
    def __init__(self, dataset, initial_batch_size=32, max_batch_size=256):
        self.dataset = dataset
        self.batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        i = 0
        while i < len(indices):
            batch_indices = indices[i:i+self.batch_size]
            yield batch_indices
            i += self.batch_size

    def handle_oom(self):
        """Reduce batch size after OOM"""
        self.batch_size = max(self.batch_size // 2, 1)
        self.oom_count += 1
        print(f"OOM! Reducing batch size to {self.batch_size}")

    def increase_batch_size(self):
        """Try to increase batch size"""
        if self.batch_size < self.max_batch_size:
            self.batch_size = min(self.batch_size * 2, self.max_batch_size)
            print(f"Increasing batch size to {self.batch_size}")

# Usage
sampler = DynamicBatchSampler(dataset)

for epoch in range(num_epochs):
    for batch_indices in sampler:
        batch = [dataset[i] for i in batch_indices]

        try:
            train_step(batch)

            # Periodically try to increase batch size
            if random.random() < 0.01:  # 1% chance
                sampler.increase_batch_size()

        except RuntimeError as e:
            if "out of memory" in str(e):
                sampler.handle_oom()
                torch.cuda.empty_cache()
            else:
                raise e
```

## Data Loading Optimization

### Profiling DataLoader

```python
import time

def profile_dataloader(loader, num_batches=100):
    """Profile data loading performance"""

    load_times = []
    transfer_times = []
    total_start = time.time()

    for i, (data, target) in enumerate(loader):
        if i >= num_batches:
            break

        # Measure GPU transfer time
        transfer_start = time.time()
        data_gpu = data.cuda(non_blocking=True)
        target_gpu = target.cuda(non_blocking=True)
        torch.cuda.synchronize()
        transfer_time = time.time() - transfer_start

        transfer_times.append(transfer_time)

    total_time = time.time() - total_start

    avg_transfer = sum(transfer_times) / len(transfer_times)
    avg_iteration = total_time / num_batches
    data_loading_time = avg_iteration - avg_transfer

    print(f"DataLoader Performance:")
    print(f"  Total iteration time: {avg_iteration*1000:.2f} ms")
    print(f"  Data loading time: {data_loading_time*1000:.2f} ms")
    print(f"  GPU transfer time: {avg_transfer*1000:.2f} ms")
    print(f"  Throughput: {num_batches/total_time:.1f} batches/sec")

    # Diagnose
    if data_loading_time / avg_iteration > 0.5:
        print("\n⚠ Data loading is bottleneck!")
        print("  - Increase num_workers")
        print("  - Simplify preprocessing")
        print("  - Use faster storage")

    if avg_transfer / avg_iteration > 0.3:
        print("\n⚠ GPU transfer is bottleneck!")
        print("  - Enable pin_memory")
        print("  - Use non_blocking=True")

# Test different configurations
configs = [
    {'num_workers': 0, 'pin_memory': False, 'prefetch_factor': 2},
    {'num_workers': 4, 'pin_memory': False, 'prefetch_factor': 2},
    {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2},
    {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 4},
]

for config in configs:
    print(f"\nConfig: {config}")
    loader = DataLoader(dataset, batch_size=32, **config)
    profile_dataloader(loader)
```

### Optimized DataLoader Configuration

```python
def create_optimal_dataloader(dataset, batch_size, num_gpus=1):
    """Create optimally configured DataLoader"""

    # Determine num_workers based on CPU cores
    import os
    cpu_count = os.cpu_count() or 4
    num_workers = min(cpu_count, 8)  # Diminishing returns after ~8

    # Adjust for distributed training
    if num_gpus > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,              # Faster GPU transfer
        prefetch_factor=2,            # Prefetch 2 batches per worker
        persistent_workers=True,      # Keep workers alive
        drop_last=True,               # Consistent batch sizes
    )

    return loader
```

## Production Monitoring

### GPU Metrics Collection

```python
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram
import nvidia_smi

class GPUMetricsCollector:
    """Collect GPU metrics for Prometheus"""

    def __init__(self):
        nvidia_smi.nvmlInit()

        # Define metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )

        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id']
        )

        self.gpu_memory_total = Gauge(
            'gpu_memory_total_bytes',
            'GPU total memory in bytes',
            ['gpu_id']
        )

        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id']
        )

        self.gpu_power_usage = Gauge(
            'gpu_power_usage_watts',
            'GPU power usage in Watts',
            ['gpu_id']
        )

    def collect_metrics(self):
        """Collect metrics from all GPUs"""
        device_count = nvidia_smi.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

            # Utilization
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_utilization.labels(gpu_id=i).set(util.gpu)

            # Memory
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_memory_used.labels(gpu_id=i).set(mem.used)
            self.gpu_memory_total.labels(gpu_id=i).set(mem.total)

            # Temperature
            temp = nvidia_smi.nvmlDeviceGetTemperature(handle, 0)
            self.gpu_temperature.labels(gpu_id=i).set(temp)

            # Power
            power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            self.gpu_power_usage.labels(gpu_id=i).set(power)

    def start_http_server(self, port=8000):
        """Start Prometheus metrics server"""
        prometheus_client.start_http_server(port)
        print(f"Metrics server running on port {port}")

    def run_forever(self, interval=10):
        """Collect metrics periodically"""
        import time

        while True:
            self.collect_metrics()
            time.sleep(interval)

# Usage
collector = GPUMetricsCollector()
collector.start_http_server(8000)
collector.run_forever(interval=10)
```

### Training Metrics Dashboard

```python
from torch.utils.tensorboard import SummaryWriter
import nvidia_smi

class TrainingMonitor:
    """Monitor training with GPU metrics"""

    def __init__(self, log_dir='runs'):
        self.writer = SummaryWriter(log_dir)
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def log_step(self, step, loss, lr, batch_time):
        """Log training step metrics"""
        # Training metrics
        self.writer.add_scalar('Loss/train', loss, step)
        self.writer.add_scalar('Learning_Rate', lr, step)
        self.writer.add_scalar('Time/batch', batch_time, step)

        # GPU metrics
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        temp = nvidia_smi.nvmlDeviceGetTemperature(self.handle, 0)

        self.writer.add_scalar('GPU/utilization', util.gpu, step)
        self.writer.add_scalar('GPU/memory_used_gb', mem.used/1024**3, step)
        self.writer.add_scalar('GPU/temperature', temp, step)

        # Throughput
        samples_per_sec = batch_size / batch_time
        self.writer.add_scalar('Throughput/samples_per_sec', samples_per_sec, step)

    def close(self):
        self.writer.close()
        nvidia_smi.nvmlShutdown()

# Usage
monitor = TrainingMonitor()

for step, (data, target) in enumerate(train_loader):
    start_time = time.time()

    # Training step
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    batch_time = time.time() - start_time

    # Log metrics
    monitor.log_step(
        step=step,
        loss=loss.item(),
        lr=optimizer.param_groups[0]['lr'],
        batch_time=batch_time
    )

monitor.close()
```

## Production Best Practices

### 1. Warmup and Profiling

```python
def train_with_warmup_profiling(model, train_loader):
    """Training with warmup and periodic profiling"""

    # Warmup (fill caches, compile kernels)
    print("Warming up...")
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            if i >= 10:
                break
            output = model(data.cuda())

    print("Warmup complete")

    # Training with periodic profiling
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            # Profile every 1000 batches
            if batch_idx % 1000 == 0:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                # Check for performance regression
                analyze_profile(prof, batch_idx)
            else:
                # Normal training
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
```

### 2. Automatic Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp_best_practices(model, train_loader):
    """Production AMP training"""

    scaler = GradScaler()

    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            # Forward with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"Warning: Loss is {loss.item()}, skipping batch")
                continue

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping (unscales internally)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Monitor scale factor
            if batch_idx % 100 == 0:
                scale = scaler.get_scale()
                print(f"Loss scale: {scale}")
```

### 3. Error Handling and Recovery

```python
def train_with_error_handling(model, train_loader, checkpoint_dir):
    """Robust training with error handling"""

    for epoch in range(num_epochs):
        try:
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    # Training step
                    output = model(data.cuda())
                    loss = criterion(output, target.cuda())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Handle OOM
                        print(f"OOM at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e

        except KeyboardInterrupt:
            # Save checkpoint on interrupt
            print("Interrupted! Saving checkpoint...")
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)
            raise

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
```

## Summary

In this lesson, you learned:

1. **Profiling Tools**: PyTorch Profiler, Nsight Systems, Nsight Compute
2. **Bottleneck Identification**: GPU utilization, kernel analysis, memory profiling
3. **Batch Size Optimization**: Finding optimal batch size for throughput
4. **Data Loading**: Profiling and optimizing DataLoader performance
5. **Production Monitoring**: Prometheus metrics, TensorBoard dashboards
6. **Best Practices**: Warmup, AMP, error handling, checkpointing

## Key Takeaways

- **Profile before optimizing** - measure to find real bottlenecks
- **GPU utilization <80%** indicates potential for optimization
- **Optimal batch size** balances throughput and memory
- **Data loading** can be a hidden bottleneck - profile it
- **Monitor in production** - track GPU metrics continuously
- **Use AMP by default** for modern GPUs (Volta+)
- **Handle errors gracefully** - OOM, NaN, interrupts

## Further Reading

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)

## Module Complete!

Congratulations! You've completed **Module 07: GPU Computing & Distributed Training**. You now have the skills to:

- Leverage GPUs effectively for ML workloads
- Implement distributed training across multiple GPUs/nodes
- Optimize GPU memory usage
- Profile and tune GPU performance
- Deploy GPU infrastructure in production

## Next Steps

Continue to:
- **Module 08**: Monitoring & Observability
- **Module 09**: Infrastructure as Code
- **Module 10**: LLM Infrastructure

Or apply your knowledge in:
- **Project 02**: MLOps Pipeline with distributed training
- **Project 03**: LLM Deployment with GPU optimization

---

**You're now ready to build world-class GPU infrastructure for AI!**
