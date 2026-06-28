# Exercise 03: Reusable ML Utilities & Modules

## Overview
This exercise covers writing clean, reusable, and modular Python code for machine learning infrastructure. You'll work with type hints, flexible signatures (`*args`/`**kwargs`), decorators, custom packaging, and functional programming.

* **Time**: 45-60 mins
* **Level**: Intermediate

---

## Part 1: Type Hints & Flexible Arguments

Type hints make your ML pipelines self-documenting and prevent shape/type bugs before runtime. Combining them with flexible arguments (`*args`, `**kwargs`) allows you to build highly adaptable utilities.

### 1. Robust Splits & Flexible Metrics (`ml_fns.py`)

```python
from typing import List, Tuple, Dict, Any, Optional
import random

# Type hinting lists and returning structured tuples
def split_data(
    data: List[Any], 
    train_ratio: float = 0.8, 
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any]]:
    """Splits a dataset into training and validation sets."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    
    data_copy = data.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * train_ratio)
    return data_copy[:split_idx], data_copy[split_idx:]

# Using *args and **kwargs for flexible logging
def log_metrics(step: int, *args: float, **kwargs: float) -> None:
    """Logs positional loss values and named evaluation metrics."""
    print(f"[Step {step:03d}]")
    if args:
        print(f"  Losses: {', '.join(f'{x:.4f}' for x in args)}")
    if kwargs:
        metrics = [f"{k}={v:.4f}" for k, v in kwargs.items()]
        print(f"  Metrics: {', '.join(metrics)}")

# Example run
train, val = split_data(list(range(10)), train_ratio=0.7, seed=42)
print(f"Train: {train} | Val: {val}")

log_metrics(10, 0.421, 0.389, accuracy=0.912, f1=0.887)
```

**Your Turn:**
1. Modify `log_metrics` to write the logged outputs to a local file instead of just printing them.
2. Write a function `build_layer` that takes `filters: int` and arbitrary `**kwargs` to return a layer configuration dictionary containing all parameters.

---

## Part 2: Decorators for ML Workflows

Decorators allow you to cleanly wrap functions with common patterns like performance timing, logging, or error retries without cluttering your core algorithms.

### 1. Timing & Caching (`decorators.py`)

```python
import time
import functools
from typing import Callable, Any

def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """Measures and logs function execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"⏱️  {func.__name__} took {duration:.4f}s")
        return result
    return wrapper

def memoize(func: Callable[..., Any]) -> Callable[..., Any]:
    """Caches computed results based on stringified inputs."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Let's test them out
@time_it
@memoize
def heavy_feature_calc(n: int) -> int:
    time.sleep(1.0) # Simulating a heavy calculation
    return sum(i * i for i in range(n))

print(heavy_feature_calc(100_000))  # Takes ~1 second
print(heavy_feature_calc(100_000))  # Instant cache hit!
```

---

## Part 3: Custom Modules & Packages

Keeping your utilities in a structured module keeps your main script clean and readable.

Create a folder structure like this:
```text
ml_utils/
  ├── __init__.py
  └── metrics.py
```

### 1. The Metrics Module (`ml_utils/metrics.py`)
```python
from typing import List

def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

def precision(y_true: List[int], y_pred: List[int]) -> float:
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    pred_pos = sum(y_pred)
    return true_pos / pred_pos if pred_pos > 0 else 0.0
```

### 2. The Package Entry (`ml_utils/__init__.py`)
```python
from .metrics import accuracy, precision

__all__ = ["accuracy", "precision"]
```

---

## Part 4: Functional Pipelines (`map`, `filter`, `lambda`)

Functional syntax helps construct concise, declarative data transformations.

### 1. Data Pipeline (`pipelines.py`)

```python
# Raw prediction confidence scores
confidences = [0.12, 0.89, 0.45, 0.73, 0.95, 0.05]

# 1. Map probabilities to binary labels using a lambda
predictions = list(map(lambda x: 1 if x >= 0.5 else 0, confidences))
print(f"Predictions: {predictions}")

# 2. Filter out low-confidence predictions
confident_scores = list(filter(lambda x: x > 0.7, confidences))
print(f"High-confidence scores: {confident_scores}")
```

---

## Challenge: Build a Metric Evaluator Pipeline
Write a script that:
1. Accepts a batch of simulated probabilities and true labels.
2. Uses functional tools (`map`) to threshold the probabilities.
3. Import your `ml_utils` package to compute classification accuracy.
4. Wrap the entire operation in a decorator to log how long the evaluation took.

---

## Quick Checklist
1. What is the difference between `*args` and `**kwargs`?
2. Why should you use `functools.wraps` inside your custom decorators?
3. How does Python's `sys.path` affect how modules are imported?
