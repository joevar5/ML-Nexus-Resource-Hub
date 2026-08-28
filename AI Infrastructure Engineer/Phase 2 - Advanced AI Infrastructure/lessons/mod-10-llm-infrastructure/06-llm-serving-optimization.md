# Lesson 06: LLM Serving Optimization

## Table of Contents
1. [Introduction](#introduction)
2. [Model Quantization Techniques](#model-quantization-techniques)
3. [Flash Attention 2](#flash-attention-2)
4. [Continuous Batching](#continuous-batching)
5. [Dynamic Batching Strategies](#dynamic-batching-strategies)
6. [Multi-GPU Inference](#multi-gpu-inference)
7. [Pipeline Parallelism](#pipeline-parallelism)
8. [KV Cache Management](#kv-cache-management)
9. [Prompt Caching](#prompt-caching)
10. [Speculative Decoding](#speculative-decoding)
11. [Benchmarking LLM Inference](#benchmarking-llm-inference)
12. [Cost vs Performance Trade-offs](#cost-vs-performance-trade-offs)
13. [Production Optimization Checklist](#production-optimization-checklist)
14. [Summary](#summary)

## Introduction

Optimizing LLM serving is critical for both cost efficiency and user experience. This lesson covers advanced techniques to maximize throughput, minimize latency, and reduce infrastructure costs for LLM inference.

### Learning Objectives

After completing this lesson, you will be able to:
- Implement various quantization techniques (GPTQ, AWQ, GGUF, bitsandbytes)
- Use Flash Attention 2 for faster inference
- Implement continuous and dynamic batching
- Configure multi-GPU inference with tensor and pipeline parallelism
- Optimize KV cache management
- Implement prompt caching strategies
- Use speculative decoding for faster generation
- Benchmark and analyze LLM performance
- Make informed cost vs performance trade-off decisions

## Model Quantization Techniques

### Understanding Quantization

Quantization reduces the precision of model weights and activations, trading minimal accuracy for significant memory and compute savings.

```python
# Quantization precision comparison
quantization_comparison = {
    'FP32': {
        'bits_per_param': 32,
        'memory_factor': 1.0,
        'speed_factor': 1.0,
        'accuracy': 'Baseline',
        'use_case': 'Research, maximum accuracy'
    },
    'FP16': {
        'bits_per_param': 16,
        'memory_factor': 0.5,
        'speed_factor': 1.5,
        'accuracy': '~100% of FP32',
        'use_case': 'Standard production deployment'
    },
    'INT8': {
        'bits_per_param': 8,
        'memory_factor': 0.25,
        'speed_factor': 2.0,
        'accuracy': '98-99% of FP32',
        'use_case': 'Memory-constrained deployments'
    },
    'INT4': {
        'bits_per_param': 4,
        'memory_factor': 0.125,
        'speed_factor': 2.5,
        'accuracy': '95-98% of FP32',
        'use_case': 'Edge devices, maximum throughput'
    }
}


def calculate_quantization_savings(model_size_b: float, from_precision: str, to_precision: str):
    """
    Calculate memory and cost savings from quantization
    """
    precision_bytes = {
        'FP32': 4,
        'FP16': 2,
        'BF16': 2,
        'INT8': 1,
        'INT4': 0.5
    }

    original_memory_gb = model_size_b * precision_bytes[from_precision]
    quantized_memory_gb = model_size_b * precision_bytes[to_precision]

    memory_reduction = original_memory_gb / quantized_memory_gb
    memory_saved_gb = original_memory_gb - quantized_memory_gb

    return {
        'original_memory_gb': round(original_memory_gb, 2),
        'quantized_memory_gb': round(quantized_memory_gb, 2),
        'memory_saved_gb': round(memory_saved_gb, 2),
        'reduction_factor': f"{memory_reduction:.1f}x",
        'percentage_saved': f"{(memory_saved_gb / original_memory_gb) * 100:.1f}%"
    }


# Example: Llama 2 70B
print("Llama 2 70B: FP16 → INT4")
print(calculate_quantization_savings(70, 'FP16', 'INT4'))
# Can fit 70B model in 35GB instead of 140GB!
```

### GPTQ (Generative Pre-trained Transformer Quantization)

GPTQ is a post-training quantization method optimized for generative models.

```python
# Install dependencies
# pip install auto-gptq optimum

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# GPTQ quantization configuration
gptq_config = GPTQConfig(
    bits=4,                          # Quantization bits
    dataset="c4",                    # Calibration dataset
    tokenizer=None,                  # Will be set below
    group_size=128,                  # Group size for quantization
    damp_percent=0.01,               # Damping percentage
    desc_act=False,                  # Activate descending order
    sym=True,                        # Symmetric quantization
    true_sequential=True,            # Sequential quantization
    use_cuda_fp16=True,              # Use CUDA FP16
    model_seqlen=2048,              # Model sequence length
)

# Load and quantize model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gptq_config.tokenizer = tokenizer

# Quantize (this takes time - run once, save the result)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=gptq_config,
    device_map="auto"
)

# Save quantized model
model.save_pretrained("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")

# Later, load the quantized model (fast)
quantized_model = AutoModelForCausalLM.from_pretrained(
    "./llama-2-7b-gptq-4bit",
    device_map="auto"
)
```

### AWQ (Activation-aware Weight Quantization)

AWQ protects important weights based on activation statistics, achieving better accuracy than GPTQ.

```python
# Install dependencies
# pip install autoawq

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-4bit"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Quantize (using calibration data)
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Load for inference
model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True,  # Fuse layers for faster inference
    device_map="auto"
)
```

### GGUF (GPT-Generated Unified Format)

GGUF is used primarily with llama.cpp for CPU and edge deployment.

```bash
# Convert model to GGUF format
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert Hugging Face model to GGUF
python convert.py /path/to/llama-2-7b --outtype q4_0 --outfile llama-2-7b-q4_0.gguf

# Quantization types:
# q4_0: 4-bit (smallest, fastest)
# q4_1: 4-bit (better quality)
# q5_0: 5-bit
# q5_1: 5-bit (better quality)
# q8_0: 8-bit (best quality)

# Run inference
./main -m llama-2-7b-q4_0.gguf -p "Once upon a time" -n 128
```

### bitsandbytes Quantization

bitsandbytes provides easy-to-use quantization for Hugging Face models.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 8-bit quantization
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,           # Outlier threshold
    llm_int8_has_fp16_weight=False,   # Use INT8 for all
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_8bit,
    device_map="auto"
)

# 4-bit quantization (NF4)
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_use_double_quant=True,   # Double quantization
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_4bit,
    device_map="auto"
)

# Memory usage comparison
print(f"8-bit model memory: {model_8bit.get_memory_footprint() / 1e9:.2f} GB")
print(f"4-bit model memory: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
```

### Quantization Comparison

```python
# Comprehensive quantization benchmark
class QuantizationBenchmark:
    """
    Compare different quantization methods
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}

    def benchmark_method(self, method: str, config: dict):
        """
        Benchmark a specific quantization method
        """
        import time
        import torch

        # Load model based on method
        start_time = time.time()

        if method == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif method == 'gptq':
            model = AutoModelForCausalLM.from_pretrained(
                config['model_path'],
                device_map="auto"
            )
        elif method == 'awq':
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_quantized(
                config['model_path'],
                fuse_layers=True,
                device_map="auto"
            )
        elif method == 'bnb_8bit':
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif method == 'bnb_4bit':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )

        load_time = time.time() - start_time

        # Get memory usage
        memory_gb = model.get_memory_footprint() / 1e9

        # Benchmark inference
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)

        # Measure
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
        inference_time = (time.time() - start_time) / num_runs

        # Calculate tokens per second
        tokens_generated = 50
        tokens_per_sec = tokens_generated / inference_time

        self.results[method] = {
            'load_time_sec': round(load_time, 2),
            'memory_gb': round(memory_gb, 2),
            'inference_time_sec': round(inference_time, 3),
            'tokens_per_sec': round(tokens_per_sec, 2)
        }

        return self.results[method]

    def compare_all(self):
        """
        Compare all available quantization methods
        """
        methods = ['fp16', 'bnb_8bit', 'bnb_4bit']

        for method in methods:
            print(f"\nBenchmarking {method}...")
            try:
                result = self.benchmark_method(method, {})
                print(f"Results: {result}")
            except Exception as e:
                print(f"Failed: {e}")

        return self.results


# Usage
benchmark = QuantizationBenchmark("meta-llama/Llama-2-7b-hf")
results = benchmark.compare_all()
```

## Flash Attention 2

### What is Flash Attention?

Flash Attention is an optimized attention algorithm that reduces memory usage and improves speed without changing the output.

**Key benefits:**
- 2-4x faster attention computation
- Reduced memory usage (enables longer contexts)
- Mathematically equivalent to standard attention

```python
# Using Flash Attention 2 with vLLM
from vllm import LLM, SamplingParams

# Flash Attention 2 is enabled by default in vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    # Flash Attention 2 automatically used if available
)

# For Hugging Face Transformers
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Enable Flash Attention 2
)
```

### Flash Attention Performance Impact

```python
# Flash Attention benchmark comparison
class FlashAttentionBenchmark:
    """
    Compare standard attention vs Flash Attention
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def benchmark_attention(self, use_flash: bool, sequence_length: int):
        """
        Benchmark with/without Flash Attention
        """
        import time
        import torch

        # Load model
        if use_flash:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"  # Standard attention
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create long prompt
        prompt = "Hello " * (sequence_length // 2)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10)

        # Measure
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)

        avg_time = (time.time() - start_time) / num_runs

        # Memory usage
        memory_gb = torch.cuda.max_memory_allocated() / 1e9

        return {
            'attention_type': 'Flash Attention 2' if use_flash else 'Standard',
            'sequence_length': sequence_length,
            'avg_time_sec': round(avg_time, 3),
            'memory_gb': round(memory_gb, 2),
            'tokens_per_sec': round(50 / avg_time, 2)
        }

    def compare(self, sequence_lengths=[512, 1024, 2048, 4096]):
        """
        Compare Flash vs Standard across different sequence lengths
        """
        results = []

        for seq_len in sequence_lengths:
            print(f"\nTesting sequence length: {seq_len}")

            # Standard attention
            std_result = self.benchmark_attention(False, seq_len)
            print(f"Standard: {std_result}")

            # Flash attention
            flash_result = self.benchmark_attention(True, seq_len)
            print(f"Flash: {flash_result}")

            # Calculate improvement
            speedup = std_result['avg_time_sec'] / flash_result['avg_time_sec']
            memory_savings = std_result['memory_gb'] - flash_result['memory_gb']

            results.append({
                'sequence_length': seq_len,
                'standard_time': std_result['avg_time_sec'],
                'flash_time': flash_result['avg_time_sec'],
                'speedup': f"{speedup:.2f}x",
                'memory_savings_gb': round(memory_savings, 2)
            })

        return results


# Usage
benchmark = FlashAttentionBenchmark("meta-llama/Llama-2-7b-chat-hf")
results = benchmark.compare()
```

### Enabling Flash Attention in Production

```python
# Dockerfile for Flash Attention support
"""
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y python3-pip git

# Install Flash Attention 2
RUN pip install packaging ninja
RUN pip install flash-attn --no-build-isolation

# Install vLLM with Flash Attention support
RUN pip install vllm

# Your application code
COPY . /app
WORKDIR /app

CMD ["python", "serve.py"]
"""

# Kubernetes deployment with Flash Attention
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server-flash-attn
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-server
  template:
    metadata:
      labels:
        app: llm-server
    spec:
      containers:
      - name: vllm
        image: your-registry/vllm-flash-attn:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 32Gi
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-2-7b-chat-hf"
        - name: MAX_MODEL_LEN
          value: "4096"
        ports:
        - containerPort: 8000
"""
```

## Continuous Batching

### Understanding Continuous Batching

Traditional batching waits for all requests to complete before processing new ones. Continuous batching (also called "iteration-level batching") adds new requests as soon as slots become available.

```python
# Visualization of batching strategies
"""
Traditional Static Batching:
Time: 0s    1s    2s    3s    4s    5s
Req1: [================]
Req2: [================]
Req3: [================]
Req4:                   [================]
Req5:                   [================]
Req6:                   [================]

Continuous Batching:
Time: 0s    1s    2s    3s    4s    5s
Req1: [================]
Req2: [================]
Req3: [================]
Req4:      [================]
Req5:           [================]
Req6:                [================]

Result: Continuous batching achieves higher GPU utilization and lower average latency!
"""
```

### vLLM Continuous Batching

vLLM implements continuous batching by default through its PagedAttention mechanism:

```python
from vllm import LLM, SamplingParams
from typing import List

# Initialize vLLM with optimized settings for continuous batching
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    max_num_batched_tokens=8192,      # Max tokens in a batch
    max_num_seqs=256,                 # Max concurrent sequences
    gpu_memory_utilization=0.90,      # Leave headroom for dynamic batching
    enable_prefix_caching=True,       # Cache common prefixes
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

# Process multiple requests (continuous batching happens automatically)
prompts = [
    "Write a story about",
    "Explain quantum computing",
    "What is the capital of",
    # ... hundreds of prompts
]

# vLLM automatically batches and schedules these efficiently
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### Continuous Batching Metrics

```python
# Monitor continuous batching performance
class ContinuousBatchingMonitor:
    """
    Monitor continuous batching metrics
    """
    def __init__(self):
        self.metrics = {
            'batch_sizes': [],
            'queue_lengths': [],
            'gpu_utilization': [],
            'throughput': []
        }

    def collect_metrics(self, batch_size: int, queue_length: int, gpu_util: float):
        """
        Collect metrics during serving
        """
        self.metrics['batch_sizes'].append(batch_size)
        self.metrics['queue_lengths'].append(queue_length)
        self.metrics['gpu_utilization'].append(gpu_util)

    def calculate_throughput(self, tokens_generated: int, time_elapsed: float):
        """
        Calculate tokens per second
        """
        throughput = tokens_generated / time_elapsed
        self.metrics['throughput'].append(throughput)
        return throughput

    def report(self):
        """
        Generate performance report
        """
        import numpy as np

        report = {
            'avg_batch_size': np.mean(self.metrics['batch_sizes']),
            'p95_batch_size': np.percentile(self.metrics['batch_sizes'], 95),
            'avg_queue_length': np.mean(self.metrics['queue_lengths']),
            'avg_gpu_utilization': np.mean(self.metrics['gpu_utilization']),
            'avg_throughput': np.mean(self.metrics['throughput'])
        }

        return report


# Integration with vLLM server
"""
# In your vLLM server code:
monitor = ContinuousBatchingMonitor()

# After each batch
monitor.collect_metrics(
    batch_size=current_batch_size,
    queue_length=len(request_queue),
    gpu_util=get_gpu_utilization()
)

# Periodically log report
if time.time() - last_report_time > 60:
    print(monitor.report())
    last_report_time = time.time()
"""
```

## Dynamic Batching Strategies

### Adaptive Batch Sizing

```python
class AdaptiveBatcher:
    """
    Dynamically adjust batch size based on request patterns
    """
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        target_latency_ms: float = 100
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.current_batch_size = min_batch_size
        self.latency_history = []

    def adjust_batch_size(self, observed_latency_ms: float):
        """
        Adjust batch size based on observed latency
        """
        self.latency_history.append(observed_latency_ms)

        # Keep last 100 measurements
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)

        # Calculate average latency
        avg_latency = sum(self.latency_history) / len(self.latency_history)

        # Adjust batch size
        if avg_latency < self.target_latency_ms * 0.8:
            # Latency is good, increase batch size
            self.current_batch_size = min(
                self.current_batch_size + 1,
                self.max_batch_size
            )
        elif avg_latency > self.target_latency_ms * 1.2:
            # Latency is too high, decrease batch size
            self.current_batch_size = max(
                self.current_batch_size - 1,
                self.min_batch_size
            )

        return self.current_batch_size

    def get_batch_size(self) -> int:
        """
        Get current recommended batch size
        """
        return self.current_batch_size


# Usage in serving loop
batcher = AdaptiveBatcher(
    min_batch_size=1,
    max_batch_size=64,
    target_latency_ms=100
)

while True:
    # Get current batch size
    batch_size = batcher.get_batch_size()

    # Process batch
    start_time = time.time()
    outputs = process_batch(requests[:batch_size])
    latency_ms = (time.time() - start_time) * 1000

    # Adjust for next iteration
    batcher.adjust_batch_size(latency_ms)
```

### Priority-Based Batching

```python
from queue import PriorityQueue
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedRequest:
    """
    Request with priority for batching
    """
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    params: Any = field(compare=False)


class PriorityBatcher:
    """
    Batch requests based on priority
    """
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.queue = PriorityQueue()

    def add_request(
        self,
        request_id: str,
        prompt: str,
        priority: int = 5,  # 1 = highest, 10 = lowest
        params: dict = None
    ):
        """
        Add request to queue with priority
        """
        request = PrioritizedRequest(
            priority=priority,
            timestamp=time.time(),
            request_id=request_id,
            prompt=prompt,
            params=params or {}
        )
        self.queue.put(request)

    def get_batch(self):
        """
        Get next batch, prioritizing high-priority requests
        """
        batch = []
        while len(batch) < self.batch_size and not self.queue.empty():
            batch.append(self.queue.get())

        return batch

    def has_requests(self):
        """
        Check if there are pending requests
        """
        return not self.queue.empty()


# Usage
batcher = PriorityBatcher(batch_size=8)

# Add requests with different priorities
batcher.add_request("req1", "Regular prompt", priority=5)
batcher.add_request("req2", "Important prompt", priority=1)  # Processed first
batcher.add_request("req3", "Low priority prompt", priority=10)

# Get batch (high priority first)
batch = batcher.get_batch()
```

## Multi-GPU Inference

### Tensor Parallelism

Tensor parallelism splits individual layers across GPUs for large models that don't fit on a single GPU.

```python
# vLLM with tensor parallelism
from vllm import LLM, SamplingParams

# Split model across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,           # Use 4 GPUs
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    dtype="float16"
)

# Usage is the same
outputs = llm.generate(prompts, sampling_params)
```

### Tensor Parallelism Configuration

```python
# Calculate optimal tensor parallel size
def calculate_tensor_parallel_size(
    model_size_b: float,
    available_gpus: list,
    precision: str = "fp16"
):
    """
    Calculate optimal number of GPUs for tensor parallelism
    """
    # Memory calculation
    precision_bytes = {'fp32': 4, 'fp16': 2, 'int8': 1}
    model_memory_gb = model_size_b * precision_bytes[precision]

    # Add overhead for KV cache and activations (50%)
    total_memory_needed = model_memory_gb * 1.5

    # GPU memory (GB)
    gpu_memory = {
        'T4': 16,
        'A10G': 24,
        'A100-40GB': 40,
        'A100-80GB': 80
    }

    configurations = []

    for gpu_type in available_gpus:
        single_gpu_memory = gpu_memory[gpu_type]

        # Calculate minimum GPUs needed
        min_gpus = int(np.ceil(total_memory_needed / single_gpu_memory))

        # Tensor parallel size should be power of 2
        tp_sizes = [1, 2, 4, 8]
        valid_tp_sizes = [tp for tp in tp_sizes if tp >= min_gpus]

        for tp in valid_tp_sizes:
            memory_per_gpu = total_memory_needed / tp
            utilization = memory_per_gpu / single_gpu_memory

            if utilization <= 0.95:  # Leave some headroom
                configurations.append({
                    'gpu_type': gpu_type,
                    'tensor_parallel_size': tp,
                    'memory_per_gpu_gb': round(memory_per_gpu, 2),
                    'utilization': f"{utilization * 100:.1f}%",
                    'total_gpus_needed': tp
                })

    return configurations


# Example: Llama 2 70B
configs = calculate_tensor_parallel_size(
    model_size_b=70,
    available_gpus=['A10G', 'A100-40GB', 'A100-80GB'],
    precision='fp16'
)

print("Optimal configurations for Llama 2 70B:")
for config in configs:
    print(config)
```

### Ray Serve for Multi-Model Multi-GPU

```python
# Deploy multiple models across GPUs with Ray Serve
from ray import serve
import ray
from vllm import LLM, SamplingParams

# Initialize Ray
ray.init()

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4}
)
class LLMDeployment:
    def __init__(self, model_name: str):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90
        )
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=256
        )

    def __call__(self, request):
        prompt = request.query_params.get("prompt")
        outputs = self.llm.generate([prompt], self.sampling_params)
        return {"text": outputs[0].outputs[0].text}


# Deploy multiple models
serve.run(
    LLMDeployment.bind(model_name="meta-llama/Llama-2-7b-chat-hf"),
    name="llama-7b",
    route_prefix="/v1/llama-7b"
)

serve.run(
    LLMDeployment.bind(model_name="mistralai/Mistral-7B-Instruct-v0.1"),
    name="mistral-7b",
    route_prefix="/v1/mistral-7b"
)

# Each model automatically gets its own GPU(s) via Ray's scheduling
```

## Pipeline Parallelism

### Understanding Pipeline Parallelism

Pipeline parallelism splits the model into stages, with each stage on a different GPU. Good for very large models and high throughput.

```python
# Pipeline parallelism visualization
"""
GPU 0: [Layers 0-7  ]  →  [Layers 0-7  ]  →  [Layers 0-7  ]
GPU 1: [Layers 8-15 ]  →  [Layers 8-15 ]  →  [Layers 8-15 ]
GPU 2: [Layers 16-23]  →  [Layers 16-23]  →  [Layers 16-23]
GPU 3: [Layers 24-31]  →  [Layers 24-31]  →  [Layers 24-31]

Time:  Batch 1         Batch 2         Batch 3

Pipeline parallelism allows processing multiple batches simultaneously!
"""
```

### DeepSpeed Inference Pipeline

```python
# DeepSpeed inference with pipeline parallelism
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-70b-hf"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DeepSpeed inference configuration
ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        }
    },
    "tensor_parallel": {
        "tp_size": 4  # 4-way tensor parallelism
    },
    "pipeline_parallel": {
        "pp_size": 2  # 2-way pipeline parallelism
    }
}

# Initialize DeepSpeed engine
engine = deepspeed.init_inference(
    model=model,
    config=ds_config,
    tensor_parallel={"tp_size": 4},
    dtype=torch.float16,
    replace_with_kernel_inject=True
)

# Use model for inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = engine.module.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## KV Cache Management

### Understanding KV Cache

The KV (Key-Value) cache stores attention keys and values from previous tokens to avoid recomputation during generation.

```python
# KV cache memory calculation
def calculate_kv_cache_memory(
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    precision_bytes: int = 2  # FP16
):
    """
    Calculate KV cache memory requirements
    """
    # Keys and Values for each layer
    # Each: (batch_size, num_heads, seq_length, head_dim)
    head_dim = hidden_size // num_attention_heads

    # Memory per layer (keys + values)
    memory_per_layer = (
        2 *  # keys and values
        batch_size *
        num_attention_heads *
        sequence_length *
        head_dim *
        precision_bytes
    )

    total_memory = memory_per_layer * num_layers

    return {
        'memory_per_layer_mb': memory_per_layer / (1024**2),
        'total_memory_mb': total_memory / (1024**2),
        'total_memory_gb': total_memory / (1024**3)
    }


# Example: Llama 2 7B with batch of 8
kv_cache = calculate_kv_cache_memory(
    batch_size=8,
    sequence_length=2048,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32
)
print(f"KV Cache for batch of 8: {kv_cache['total_memory_gb']:.2f} GB")
```

### PagedAttention (vLLM's Approach)

PagedAttention manages KV cache like virtual memory, allowing efficient memory usage and dynamic batching.

```python
# vLLM with optimized KV cache management
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.90,  # Reserve memory for KV cache
    # PagedAttention automatically manages KV cache efficiently
)

# vLLM benefits:
# - Non-contiguous KV cache (like virtual memory paging)
# - Efficient memory sharing for common prefixes
# - Dynamic allocation based on actual sequence length
```

### KV Cache Optimization Strategies

```python
class KVCacheOptimizer:
    """
    Strategies for optimizing KV cache usage
    """
    @staticmethod
    def estimate_max_batch_size(
        available_memory_gb: float,
        model_memory_gb: float,
        sequence_length: int,
        kv_cache_per_token_mb: float
    ):
        """
        Estimate maximum batch size given memory constraints
        """
        # Available memory for KV cache
        kv_cache_memory_gb = available_memory_gb - model_memory_gb

        # Memory per sequence
        memory_per_sequence_mb = sequence_length * kv_cache_per_token_mb
        memory_per_sequence_gb = memory_per_sequence_mb / 1024

        # Maximum batch size
        max_batch_size = int(kv_cache_memory_gb / memory_per_sequence_gb)

        return max(1, max_batch_size)

    @staticmethod
    def optimize_for_throughput(
        target_throughput_tokens_per_sec: int,
        model_tokens_per_sec_single: int,
        available_memory_gb: float,
        model_memory_gb: float
    ):
        """
        Find optimal configuration for target throughput
        """
        # Calculate required batch size
        required_batch = target_throughput_tokens_per_sec / model_tokens_per_sec_single

        # Check if memory allows this batch size
        max_batch = KVCacheOptimizer.estimate_max_batch_size(
            available_memory_gb=available_memory_gb,
            model_memory_gb=model_memory_gb,
            sequence_length=2048,  # Assume 2K context
            kv_cache_per_token_mb=0.5  # Rough estimate
        )

        if required_batch <= max_batch:
            return {
                'feasible': True,
                'recommended_batch_size': int(required_batch),
                'max_batch_size': max_batch
            }
        else:
            return {
                'feasible': False,
                'required_batch_size': int(required_batch),
                'max_batch_size': max_batch,
                'suggestion': 'Need more GPUs or smaller model'
            }


# Usage
optimizer = KVCacheOptimizer()

config = optimizer.optimize_for_throughput(
    target_throughput_tokens_per_sec=1000,
    model_tokens_per_sec_single=50,
    available_memory_gb=80,
    model_memory_gb=14
)
print(config)
```

## Prompt Caching

### Automatic Prefix Caching

Cache common prompt prefixes to avoid recomputing them.

```python
# vLLM with prefix caching
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True,  # Enable prefix caching
    max_model_len=4096
)

# Common system prompt
system_prompt = """You are a helpful AI assistant. You are knowledgeable,
accurate, and always strive to provide the best possible answers."""

# Multiple requests with same prefix
prompts = [
    f"{system_prompt}\n\nUser: What is Python?\nAssistant:",
    f"{system_prompt}\n\nUser: What is JavaScript?\nAssistant:",
    f"{system_prompt}\n\nUser: What is Rust?\nAssistant:",
]

# First request processes full prompt
# Subsequent requests reuse cached system prompt computation!
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

### Manual Response Caching

```python
import hashlib
import json
from typing import Optional
import redis

class LLMResponseCache:
    """
    Cache LLM responses for identical requests
    """
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.ttl_seconds = 3600  # 1 hour cache

    def _generate_cache_key(self, prompt: str, params: dict) -> str:
        """
        Generate cache key from prompt and parameters
        """
        cache_input = {
            'prompt': prompt,
            'params': params
        }
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def get(self, prompt: str, params: dict) -> Optional[str]:
        """
        Get cached response if available
        """
        key = self._generate_cache_key(prompt, params)
        cached = self.redis_client.get(key)

        if cached:
            # Update stats
            self.redis_client.incr("cache_hits")
            return cached

        self.redis_client.incr("cache_misses")
        return None

    def set(self, prompt: str, params: dict, response: str):
        """
        Cache response
        """
        key = self._generate_cache_key(prompt, params)
        self.redis_client.setex(key, self.ttl_seconds, response)

    def get_stats(self):
        """
        Get cache statistics
        """
        hits = int(self.redis_client.get("cache_hits") or 0)
        misses = int(self.redis_client.get("cache_misses") or 0)
        total = hits + misses

        return {
            'hits': hits,
            'misses': misses,
            'total_requests': total,
            'hit_rate': (hits / total * 100) if total > 0 else 0
        }


# Usage with LLM serving
cache = LLMResponseCache()

def generate_with_cache(prompt: str, params: dict):
    """
    Generate response with caching
    """
    # Check cache first
    cached_response = cache.get(prompt, params)
    if cached_response:
        return cached_response

    # Generate new response
    response = llm.generate([prompt], SamplingParams(**params))
    response_text = response[0].outputs[0].text

    # Cache result
    cache.set(prompt, params, response_text)

    return response_text
```

### Semantic Caching

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class SemanticCache:
    """
    Cache based on semantic similarity rather than exact matches
    """
    def __init__(self, similarity_threshold: float = 0.95):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []  # List of (embedding, prompt, response)
        self.similarity_threshold = similarity_threshold

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text
        """
        return self.embedding_model.encode(text)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response if semantically similar prompt exists
        """
        if not self.cache:
            return None

        # Get embedding for new prompt
        prompt_embedding = self._get_embedding(prompt)

        # Find most similar cached prompt
        max_similarity = 0
        best_match = None

        for cached_embedding, cached_prompt, cached_response in self.cache:
            similarity = self._cosine_similarity(prompt_embedding, cached_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = cached_response

        # Return if above threshold
        if max_similarity >= self.similarity_threshold:
            return best_match

        return None

    def set(self, prompt: str, response: str):
        """
        Add prompt and response to cache
        """
        embedding = self._get_embedding(prompt)
        self.cache.append((embedding, prompt, response))

        # Limit cache size
        if len(self.cache) > 1000:
            self.cache.pop(0)


# Usage
semantic_cache = SemanticCache(similarity_threshold=0.95)

# First query
prompt1 = "What is machine learning?"
response1 = generate_llm_response(prompt1)
semantic_cache.set(prompt1, response1)

# Similar query (will hit cache)
prompt2 = "Can you explain machine learning?"
cached = semantic_cache.get(prompt2)  # Returns response1!
```

## Speculative Decoding

### What is Speculative Decoding?

Speculative decoding uses a small "draft" model to generate candidate tokens quickly, then verifies them with the larger target model in parallel. This can achieve 2-3x speedup.

```python
# Speculative decoding with vLLM
from vllm import LLM, SamplingParams

# Target model (large, accurate)
target_model = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4
)

# Draft model (small, fast)
draft_model = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1
)

# Enable speculative decoding
target_model.enable_speculative_decoding(
    draft_model=draft_model,
    num_speculative_tokens=5  # Generate 5 tokens ahead
)

# Use normally - speculative decoding happens automatically
outputs = target_model.generate(prompts, sampling_params)
```

### Implementing Speculative Decoding

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    """
    Implement speculative decoding manually
    """
    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        num_speculative_tokens: int = 5
    ):
        # Load models
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.num_speculative_tokens = num_speculative_tokens

    def generate(self, prompt: str, max_tokens: int = 100):
        """
        Generate text using speculative decoding
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.target_model.device
        )

        generated_tokens = []
        accepted_tokens = 0
        total_drafted_tokens = 0

        while len(generated_tokens) < max_tokens:
            # Step 1: Draft model generates K tokens
            draft_tokens = self._draft_generate(
                input_ids,
                num_tokens=self.num_speculative_tokens
            )
            total_drafted_tokens += len(draft_tokens)

            # Step 2: Target model verifies all at once (parallel)
            accepted = self._verify_tokens(input_ids, draft_tokens)
            accepted_tokens += len(accepted)

            # Step 3: Add accepted tokens
            generated_tokens.extend(accepted)
            input_ids = torch.cat([
                input_ids,
                torch.tensor([accepted], device=input_ids.device)
            ], dim=1)

            # If not all tokens accepted, generate one more with target model
            if len(accepted) < len(draft_tokens):
                next_token = self._target_generate_one(input_ids)
                generated_tokens.append(next_token)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=input_ids.device)
                ], dim=1)
                accepted_tokens += 1

        # Calculate acceptance rate
        acceptance_rate = accepted_tokens / total_drafted_tokens if total_drafted_tokens > 0 else 0

        # Decode
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            'text': output_text,
            'acceptance_rate': acceptance_rate,
            'total_tokens': len(generated_tokens)
        }

    def _draft_generate(self, input_ids: torch.Tensor, num_tokens: int):
        """
        Generate tokens with draft model
        """
        draft_tokens = []

        with torch.no_grad():
            current_ids = input_ids

            for _ in range(num_tokens):
                outputs = self.draft_model(current_ids)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                draft_tokens.append(next_token.item())
                current_ids = torch.cat([
                    current_ids,
                    next_token.unsqueeze(0)
                ], dim=1)

        return draft_tokens

    def _verify_tokens(self, input_ids: torch.Tensor, draft_tokens: list):
        """
        Verify draft tokens with target model (parallel)
        """
        # Create input with all draft tokens
        all_ids = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=input_ids.device)
        ], dim=1)

        # Get target model's predictions for all positions at once
        with torch.no_grad():
            outputs = self.target_model(all_ids)
            predictions = outputs.logits[:, -len(draft_tokens)-1:-1, :].argmax(dim=-1)

        # Check which tokens match
        accepted = []
        for i, (pred, draft) in enumerate(zip(predictions[0], draft_tokens)):
            if pred.item() == draft:
                accepted.append(draft)
            else:
                break  # Stop at first mismatch

        return accepted

    def _target_generate_one(self, input_ids: torch.Tensor):
        """
        Generate one token with target model
        """
        with torch.no_grad():
            outputs = self.target_model(input_ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        return next_token.item()


# Usage
decoder = SpeculativeDecoder(
    target_model_name="meta-llama/Llama-2-13b-chat-hf",
    draft_model_name="meta-llama/Llama-2-7b-chat-hf",
    num_speculative_tokens=5
)

result = decoder.generate("Once upon a time", max_tokens=100)
print(f"Generated: {result['text']}")
print(f"Acceptance rate: {result['acceptance_rate']:.2%}")
```

## Benchmarking LLM Inference

### Comprehensive Benchmark Suite

```python
import time
import torch
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """
    Results from a benchmark run
    """
    metric_name: str
    value: float
    unit: str
    percentile: Optional[int] = None


class LLMBenchmark:
    """
    Comprehensive LLM inference benchmarking
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def benchmark_latency(
        self,
        prompts: List[str],
        max_tokens: int = 50,
        num_warmup: int = 3,
        num_runs: int = 10
    ):
        """
        Benchmark inference latency
        """
        # Warmup
        for _ in range(num_warmup):
            inputs = self.tokenizer(prompts[0], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=10)

        # Measure
        latencies = []
        for _ in range(num_runs):
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
                latency = time.time() - start_time

                latencies.append(latency * 1000)  # Convert to ms

        # Calculate statistics
        self.results['latency'] = {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'std_ms': np.std(latencies)
        }

        return self.results['latency']

    def benchmark_throughput(
        self,
        prompts: List[str],
        max_tokens: int = 50,
        duration_seconds: int = 60
    ):
        """
        Benchmark throughput (requests per second)
        """
        start_time = time.time()
        num_requests = 0
        total_tokens = 0

        while time.time() - start_time < duration_seconds:
            prompt = prompts[num_requests % len(prompts)]
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

            num_requests += 1
            total_tokens += outputs.shape[1]

        elapsed = time.time() - start_time

        self.results['throughput'] = {
            'requests_per_second': num_requests / elapsed,
            'tokens_per_second': total_tokens / elapsed,
            'total_requests': num_requests,
            'total_tokens': total_tokens,
            'duration_seconds': elapsed
        }

        return self.results['throughput']

    def benchmark_memory(self):
        """
        Benchmark GPU memory usage
        """
        torch.cuda.reset_peak_memory_stats()

        # Run a few inferences
        prompt = "Hello, world!"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=100)

        # Get memory stats
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        self.results['memory'] = {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'peak_gb': peak_gb
        }

        return self.results['memory']

    def benchmark_scaling(
        self,
        base_prompt: str,
        sequence_lengths: List[int] = [128, 256, 512, 1024, 2048]
    ):
        """
        Benchmark how latency scales with sequence length
        """
        scaling_results = []

        for seq_len in sequence_lengths:
            # Create prompt of specific length
            tokens = self.tokenizer.encode(base_prompt)
            while len(tokens) < seq_len:
                tokens.extend(tokens[:min(len(tokens), seq_len - len(tokens))])
            tokens = tokens[:seq_len]
            prompt = self.tokenizer.decode(tokens)

            # Measure latency
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            latencies = []
            for _ in range(5):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_new_tokens=50)
                latencies.append((time.time() - start_time) * 1000)

            scaling_results.append({
                'sequence_length': seq_len,
                'avg_latency_ms': np.mean(latencies),
                'tokens_per_second': 50 / (np.mean(latencies) / 1000)
            })

        self.results['scaling'] = scaling_results
        return scaling_results

    def generate_report(self):
        """
        Generate comprehensive benchmark report
        """
        print("=" * 60)
        print("LLM INFERENCE BENCHMARK REPORT")
        print("=" * 60)

        if 'latency' in self.results:
            print("\nLatency Metrics:")
            for key, value in self.results['latency'].items():
                print(f"  {key}: {value:.2f}")

        if 'throughput' in self.results:
            print("\nThroughput Metrics:")
            for key, value in self.results['throughput'].items():
                print(f"  {key}: {value:.2f}")

        if 'memory' in self.results:
            print("\nMemory Metrics:")
            for key, value in self.results['memory'].items():
                print(f"  {key}: {value:.2f}")

        if 'scaling' in self.results:
            print("\nScaling Analysis:")
            for result in self.results['scaling']:
                print(f"  Seq Length {result['sequence_length']}: "
                      f"{result['avg_latency_ms']:.2f}ms, "
                      f"{result['tokens_per_second']:.2f} tokens/sec")

        print("=" * 60)


# Usage
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

benchmark = LLMBenchmark(model, tokenizer)

# Run benchmarks
prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "Write a story about"
]

benchmark.benchmark_latency(prompts)
benchmark.benchmark_throughput(prompts, duration_seconds=30)
benchmark.benchmark_memory()
benchmark.benchmark_scaling("Hello world")

# Generate report
benchmark.generate_report()
```

## Cost vs Performance Trade-offs

### Optimization Decision Framework

```python
class CostPerformanceOptimizer:
    """
    Framework for making cost vs performance trade-off decisions
    """
    def __init__(self):
        self.gpu_costs = {
            'T4': 0.40,
            'A10G': 1.20,
            'A100-40GB': 3.50,
            'A100-80GB': 5.00
        }

        self.performance_factors = {
            'T4': 1.0,
            'A10G': 2.5,
            'A100-40GB': 5.0,
            'A100-80GB': 5.5
        }

    def analyze_configuration(
        self,
        gpu_type: str,
        quantization: str,
        batch_size: int,
        requests_per_day: int,
        target_latency_ms: int
    ):
        """
        Analyze a specific configuration
        """
        # Base performance
        base_performance = self.performance_factors[gpu_type]

        # Quantization impact
        quant_speedup = {
            'fp16': 1.0,
            'int8': 1.5,
            'int4': 2.0
        }[quantization]

        # Effective performance
        effective_performance = base_performance * quant_speedup

        # Batch size impact on latency
        latency_penalty = 1 + (batch_size - 1) * 0.1

        # Estimated latency
        estimated_latency_ms = 100 / effective_performance * latency_penalty

        # Meets SLA?
        meets_sla = estimated_latency_ms <= target_latency_ms

        # Cost calculation
        requests_per_second = requests_per_day / 86400
        gpus_needed = max(1, requests_per_second / (effective_performance * batch_size))

        monthly_cost = gpus_needed * self.gpu_costs[gpu_type] * 24 * 30

        # Cost per request
        cost_per_request = monthly_cost / requests_per_day / 30

        return {
            'configuration': {
                'gpu_type': gpu_type,
                'quantization': quantization,
                'batch_size': batch_size
            },
            'performance': {
                'estimated_latency_ms': round(estimated_latency_ms, 2),
                'meets_sla': meets_sla,
                'effective_performance': round(effective_performance, 2)
            },
            'cost': {
                'gpus_needed': round(gpus_needed, 2),
                'monthly_cost': round(monthly_cost, 2),
                'cost_per_request': round(cost_per_request, 6)
            },
            'score': self._calculate_score(
                meets_sla, cost_per_request, estimated_latency_ms, target_latency_ms
            )
        }

    def _calculate_score(
        self,
        meets_sla: bool,
        cost_per_request: float,
        latency_ms: float,
        target_latency_ms: int
    ):
        """
        Calculate overall score (higher is better)
        """
        if not meets_sla:
            return 0  # Doesn't meet requirements

        # Score based on cost efficiency and latency margin
        latency_margin = (target_latency_ms - latency_ms) / target_latency_ms
        cost_score = 1 / (cost_per_request * 1000)  # Lower cost = higher score

        return cost_score * (1 + latency_margin)

    def find_optimal_configuration(
        self,
        requests_per_day: int,
        target_latency_ms: int,
        budget_per_month: float = None
    ):
        """
        Find optimal configuration
        """
        configurations = []

        for gpu_type in ['T4', 'A10G', 'A100-40GB']:
            for quantization in ['fp16', 'int8', 'int4']:
                for batch_size in [1, 2, 4, 8, 16]:
                    config = self.analyze_configuration(
                        gpu_type=gpu_type,
                        quantization=quantization,
                        batch_size=batch_size,
                        requests_per_day=requests_per_day,
                        target_latency_ms=target_latency_ms
                    )

                    # Filter by budget if specified
                    if budget_per_month and config['cost']['monthly_cost'] > budget_per_month:
                        continue

                    if config['score'] > 0:
                        configurations.append(config)

        # Sort by score
        configurations.sort(key=lambda x: x['score'], reverse=True)

        return configurations[:5]  # Top 5


# Usage
optimizer = CostPerformanceOptimizer()

# Find best configuration
optimal_configs = optimizer.find_optimal_configuration(
    requests_per_day=100000,
    target_latency_ms=200,
    budget_per_month=5000
)

print("Top 5 configurations:")
for i, config in enumerate(optimal_configs, 1):
    print(f"\n{i}. Configuration:")
    print(f"   GPU: {config['configuration']['gpu_type']}")
    print(f"   Quantization: {config['configuration']['quantization']}")
    print(f"   Batch Size: {config['configuration']['batch_size']}")
    print(f"   Latency: {config['performance']['estimated_latency_ms']}ms")
    print(f"   Monthly Cost: ${config['cost']['monthly_cost']}")
    print(f"   Cost per Request: ${config['cost']['cost_per_request']}")
    print(f"   Score: {config['score']:.2f}")
```

## Production Optimization Checklist

```python
# Production optimization checklist
production_checklist = {
    "Model Selection": {
        "tasks": [
            "Choose smallest model that meets accuracy requirements",
            "Benchmark multiple model sizes",
            "Consider fine-tuned smaller models vs larger base models",
            "Evaluate domain-specific models"
        ],
        "verification": "Run A/B tests to validate model choice"
    },
    "Quantization": {
        "tasks": [
            "Test INT8 quantization (98-99% accuracy, 50% cost reduction)",
            "Consider INT4 for non-critical workloads",
            "Benchmark GPTQ vs AWQ vs bitsandbytes",
            "Validate output quality after quantization"
        ],
        "verification": "Quality metrics within 2% of baseline"
    },
    "Serving Framework": {
        "tasks": [
            "Use vLLM for production (best throughput)",
            "Enable Flash Attention 2",
            "Configure continuous batching",
            "Enable prefix caching for common prompts"
        ],
        "verification": "Achieve 3x+ throughput vs naive implementation"
    },
    "GPU Configuration": {
        "tasks": [
            "Right-size GPU tier (don't over-provision)",
            "Configure tensor parallelism for large models",
            "Set optimal gpu_memory_utilization (0.85-0.95)",
            "Monitor actual GPU utilization"
        ],
        "verification": "GPU utilization consistently >70%"
    },
    "Batching": {
        "tasks": [
            "Configure max_num_batched_tokens appropriately",
            "Set max_num_seqs based on workload",
            "Monitor batch size distribution",
            "Tune timeout parameters"
        ],
        "verification": "Average batch size >8 under load"
    },
    "Caching": {
        "tasks": [
            "Implement response caching (Redis)",
            "Enable prefix caching in vLLM",
            "Consider semantic caching for similar queries",
            "Set appropriate cache TTLs"
        ],
        "verification": "Cache hit rate >30%"
    },
    "Monitoring": {
        "tasks": [
            "Track latency percentiles (p50, p95, p99)",
            "Monitor GPU utilization and memory",
            "Track throughput (tokens/sec, requests/sec)",
            "Monitor cost per request",
            "Alert on SLA violations"
        ],
        "verification": "Full observability dashboard operational"
    },
    "Cost Optimization": {
        "tasks": [
            "Use spot instances where appropriate",
            "Implement auto-scaling",
            "Right-size for actual traffic patterns",
            "Review and optimize monthly"
        ],
        "verification": "Cost per request tracked and optimized"
    }
}


# Automated checklist verification
class ProductionReadinessChecker:
    """
    Verify production optimization checklist
    """
    def __init__(self):
        self.checks = {}

    def check_quantization(self, model):
        """
        Verify quantization is enabled
        """
        # Check if model is quantized
        quantized = hasattr(model, 'quantization_config')
        self.checks['quantization'] = quantized
        return quantized

    def check_flash_attention(self, model):
        """
        Verify Flash Attention is enabled
        """
        # Check model config
        has_flash = getattr(model.config, '_attn_implementation', None) == 'flash_attention_2'
        self.checks['flash_attention'] = has_flash
        return has_flash

    def check_gpu_utilization(self, target_utilization: float = 0.70):
        """
        Verify GPU utilization meets target
        """
        import torch
        if not torch.cuda.is_available():
            return False

        # Get current utilization (this is simplified)
        allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        self.checks['gpu_utilization'] = allocated >= target_utilization
        return allocated >= target_utilization

    def check_monitoring(self, prometheus_endpoint: str):
        """
        Verify monitoring is configured
        """
        import requests
        try:
            response = requests.get(f"{prometheus_endpoint}/metrics", timeout=5)
            has_monitoring = response.status_code == 200
            self.checks['monitoring'] = has_monitoring
            return has_monitoring
        except:
            self.checks['monitoring'] = False
            return False

    def generate_report(self):
        """
        Generate readiness report
        """
        total_checks = len(self.checks)
        passed_checks = sum(self.checks.values())

        print("=" * 60)
        print("PRODUCTION READINESS REPORT")
        print("=" * 60)
        print(f"\nPassed: {passed_checks}/{total_checks} checks\n")

        for check_name, passed in self.checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {check_name}")

        print("\n" + "=" * 60)

        if passed_checks == total_checks:
            print("✓ System is production ready!")
        else:
            print("✗ System needs optimization before production deployment")

        return passed_checks == total_checks
```

## Summary

This lesson covered comprehensive optimization techniques for LLM serving:

### Key Takeaways

1. **Quantization**: 50-75% cost reduction with minimal quality loss
   - INT8: Best quality/cost trade-off
   - INT4: Maximum cost reduction
   - GPTQ, AWQ for post-training quantization

2. **Flash Attention 2**: 2-4x speedup with no quality loss
   - Enabled by default in vLLM
   - Enables longer context windows

3. **Continuous Batching**: 3-10x throughput improvement
   - Automatic in vLLM
   - Maximizes GPU utilization

4. **Multi-GPU Inference**: Scale to largest models
   - Tensor parallelism for model splitting
   - Pipeline parallelism for throughput

5. **Caching**: 30-60% cost reduction
   - Response caching for identical requests
   - Prefix caching for common prompts
   - Semantic caching for similar queries

6. **Speculative Decoding**: 2-3x speedup
   - Use small draft model + large target model
   - Best for high-quality requirements

7. **Benchmarking**: Measure everything
   - Latency, throughput, memory
   - Scaling characteristics
   - Cost per request

8. **Production Checklist**: Systematic optimization
   - Quantization + Flash Attention + Caching = baseline
   - Monitor and iterate
   - Balance cost vs performance

### Next Steps

In the next lesson, we'll cover complete LLM platform architecture, including multi-model serving, API gateways, monitoring, and security.

---

**Next Lesson**: [07-llm-platform-architecture.md](./07-llm-platform-architecture.md)
