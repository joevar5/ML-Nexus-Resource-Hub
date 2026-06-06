# Exercise 05: Robust Exception Handling in ML Pipelines

## Overview
This exercise covers designing fault-tolerant ML pipelines. You will learn to handle common system and data errors, create custom domain exceptions, write retry decorators with exponential backoff, and use context managers for clean resource cleanup (e.g., releasing GPU memory).

* **Time**: 45-60 mins
* **Level**: Intermediate

---

## Part 1: Try-Except-Else-Finally

Catching specific exceptions is critical. Never use a blank `except:` block, as it can hide genuine syntax or system errors.

### 1. Robust Data Processing (`exception_basics.py`)
Run this script to see how validation errors are handled without crashing the training run.

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch(batch: list) -> dict:
    results = {"outputs": [], "errors": []}
    
    try:
        if not batch:
            raise ValueError("Empty batch provided")
            
        for i, val in enumerate(batch):
            try:
                # Expecting float/int features
                scaled = float(val) * 2.0
                results["outputs"].append(scaled)
            except (TypeError, ValueError) as e:
                results["errors"].append(f"Index {i}: {e}")
                
    except ValueError as e:
        logger.error(f"Critical batch error: {e}")
        results["errors"].append(str(e))
    else:
        logger.info("Batch processed successfully without critical failure.")
    finally:
        logger.info(f"Summary: {len(results['outputs'])} processed, {len(results['errors'])} errors")
        
    return results

# Test run
print(process_batch([1.2, "corrupted_text", 3.4]))
```

---

## Part 2: Custom Exceptions for ML

Custom exceptions clarify exactly what part of your pipeline failed (e.g., GPU Out-Of-Memory, data drift, or missing model checkpoints).

### 1. Define & Raise Domain Exceptions (`custom_exceptions.py`)

```python
class MLException(Exception):
    """Base exception class for our ML pipeline."""
    pass

class GPUOutOfMemory(MLException):
    """Raised when tensor allocation exceeds GPU memory."""
    def __init__(self, batch_size: int, device_id: int):
        self.batch_size = batch_size
        self.device_id = device_id
        super().__init__(f"CUDA OOM on GPU:{device_id} with batch_size={batch_size}.")

class DataValidationError(MLException):
    """Raised when incoming features fail schema tests."""
    pass

# Usage
def allocate_gpu_memory(batch_size: int):
    if batch_size > 1024:
        raise GPUOutOfMemory(batch_size, device_id=0)
    print(f"Allocated memory for batch size {batch_size}")
```

---

## Part 3: Retry Logic with Exponential Backoff

Transient errors—like a temporary network drop during a Hugging Face model download or an S3 sync—should be retried automatically.

### 1. Resilient Decorator (`resilience.py`)

```python
import time
import random
import functools
from typing import Callable, Any

def retry_backoff(max_retries: int = 3, initial_delay: float = 0.5, factor: float = 2.0):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        print(f"❌ {func.__name__} failed permanently after {max_retries} retries.")
                        raise e
                    print(f"⚠️ {func.__name__} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= factor
            return None
        return wrapper
    return decorator

@retry_backoff(max_retries=3)
def fetch_remote_weights(url: str) -> str:
    if random.random() > 0.3:
        raise ConnectionError("Network timeout.")
    return "weights_v1.bin"

# Test run
print(f"Result: {fetch_remote_weights('https://s3/model')}")
```

---

## Part 4: Context Managers for Cleanups

Use context managers (`__enter__` and `__exit__`) to guarantee cleanup operations, like freeing GPU memory or deleting temporary checkpoints.

### 1. GPU Memory Tracker (`gpu_context.py`)

```python
class GPUContext:
    def __init__(self, device_id: int):
        self.device_id = device_id

    def __enter__(self):
        print(f"⚡ GPU:{self.device_id} initialized and locking memory context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"🧹 Clearing CUDA cache on GPU:{self.device_id}.")
        if exc_type:
            print(f"💥 Exception suppressed or logged: {exc_val}")
        # Returning False propagates the exception
        return False

# Usage
with GPUContext(device_id=0):
    print("Running forward pass...")
    # Raising an error to test cleanup
    # raise RuntimeError("CUDA illegal memory access")
```

---

## Challenge: Build a Resilient Loader
Write a Python script that:
1. Tries to load a local dataset JSON file.
2. If the file is missing, it raises a custom `DataValidationError` and triggers a fallback function to download the dataset.
3. Decorate the download function with `retry_backoff` so it handles transient network failures gracefully.

---

## Quick Checklist
1. What is the difference between catching `Exception` and a specific error like `KeyError`?
2. When should you use `else` inside a `try-except` block?
3. If an error is raised inside a `with` block, does the context manager's `__exit__` method still run?
