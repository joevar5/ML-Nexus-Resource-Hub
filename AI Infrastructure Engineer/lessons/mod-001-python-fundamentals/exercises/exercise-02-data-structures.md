# Exercise 02: Python Data Structures for ML Data

## Overview
This hands-on exercise focuses on using Python's native data structures—**lists, dictionaries, sets, and tuples**—to build clean, efficient machine learning data pipelines. You'll tackle real ML patterns like batching, feature store configuration, data deduplication, and dataset splits.

* **Time**: 45-60 mins
* **Level**: Beginner to Intermediate

---

## Part 1: Lists & Batching

Lists are the workhorses of data streaming. In ML, you'll use them constantly for batching, transformations, and file tracking.

### 1. Slicing & Operations (`list_ops.py`)
Save and run this to see how we load, batch, and shuffle image paths.

```python
# Sample image filenames in a dataset
images = [f"img_{i:04d}.jpg" for i in range(1, 9)]

# Add new files
images.append("img_0009.jpg")
images.extend(["img_0010.jpg", "img_0011.jpg"])

# Quick check
print(f"Total dataset size: {len(images)}")
print(f"First image: {images[0]} | Last image: {images[-1]}")

# Batching using slices (Batch size = 4)
batch_size = 4
batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
for idx, batch in enumerate(batches):
    print(f"Batch {idx+1}: {batch}")
```

**Your Turn:**
1. Filter the `images` list to only include files where the number is odd.
2. Implement a sliding window batcher (e.g., batch size 4, stride 2).

---

## Part 2: Dictionaries for Configs & Metadata

Dictionaries are perfect for managing model hyper-parameters, training metrics, and feature schemas.

### 1. Comprehensions & Nested Queries (`dict_ops.py`)
Run this to see how we filter metrics and parse complex experiment runs.

```python
# Model training metrics
metrics = {"accuracy": 0.925, "loss": 0.142, "val_accuracy": 0.891, "val_loss": 0.210}

# Format and filter: Keep only metrics above 90% accuracy
high_metrics = {k: f"{v*100:.1f}%" for k, v in metrics.items() if "accuracy" in k and v > 0.90}
print(f"Top Performance Metrics: {high_metrics}")

# Multi-experiment logs
runs = {
    "run_1": {"model": "resnet50", "acc": 0.92, "status": "completed"},
    "run_2": {"model": "vgg16", "acc": 0.88, "status": "completed"},
    "run_3": {"model": "vit", "acc": 0.95, "status": "failed"} # Oh no, OOM!
}

# Find the best completed model
completed_runs = {k: v for k, v in runs.items() if v["status"] == "completed"}
best_run = max(completed_runs.items(), key=lambda x: x[1]["acc"])
print(f"Best completed run: {best_run[0]} ({best_run[1]['model']} - Acc: {best_run[1]['acc']})")
```

**Your Turn:**
1. Write a feature manager that maps feature names (e.g., `"age"`, `"income"`) to dicts containing their `dtype` and `importance`.
2. Add a helper function to export this feature store configuration to a JSON file.

---

## Part 3: Sets for Data Integrity

Data leakage (accidentally putting training data into your validation/test sets) is a silent killer in ML. Sets make leakage checks trivial.

### 1. Leakage & Overlap Checks (`set_ops.py`)

```python
# Unique IDs of dataset samples
train_ids = {101, 102, 103, 104, 105, 106, 107}
val_ids = {106, 107, 108, 109}
test_ids = {110, 111, 112}

# 1. Check for data leakage (intersection)
leakage = train_ids & val_ids
if leakage:
    print(f"⚠️ DATA LEAKAGE DETECTED! Overlapping IDs: {leakage}")

# 2. Extract clean training set (difference)
clean_train = train_ids - val_ids
print(f"Clean training set: {clean_train}")

# 3. Verify train/test are completely separate
print(f"Is train/test split clean? {train_ids.isdisjoint(test_ids)}")
```

**Your Turn:**
* Write a quick deduplication function that removes duplicates from a list of filenames while preserving their original order (hint: track seen elements in a set).

---

## Part 4: Tuples for Immutability

Tuples are clean, lightweight, and immutable. Use them for fixed metadata, model architectures, or returning multiple values from a function.

### 1. Unpacking & Fixed Configurations (`tuple_ops.py`)

```python
# Fixed metadata: (model_name, version, input_shape)
model_info = ("EfficientNet-B0", "v2.1", (224, 224, 3))

# Unpacking
name, ver, shape = model_info
print(f"Deploying {name} ({ver}) with input shape {shape}")

# Named tuples make things even cleaner
from collections import namedtuple
Layer = namedtuple('Layer', ['name', 'filters', 'kernel_size'])

conv1 = Layer('conv_1', 64, (3, 3))
print(f"Layer {conv1.name} has {conv1.filters} filters of size {conv1.kernel_size}")
```

---

## Part 5: The Challenge — Dataset Manager

Put it all together. Build a clean dataset manager class that maintains sample details, generates dataset splits, and ensures no data leakage occurs.

```python
# dataset_manager.py
import random
from typing import Dict, List, Set, Tuple

class MLDatasetManager:
    def __init__(self):
        # sample_id -> {'path': str, 'label': str}
        self.samples: Dict[int, Dict[str, str]] = {}
        self.train_ids: Set[int] = set()
        self.val_ids: Set[int] = set()

    def add_sample(self, sample_id: int, filepath: str, label: str):
        if sample_id in self.samples:
            return
        self.samples[sample_id] = {"path": filepath, "label": label}

    def split(self, train_ratio: float = 0.8, seed: int = 42):
        """Randomly splits the sample IDs into train and validation sets."""
        random.seed(seed)
        all_ids = list(self.samples.keys())
        random.shuffle(all_ids)
        
        split_idx = int(len(all_ids) * train_ratio)
        self.train_ids = set(all_ids[:split_idx])
        self.val_ids = set(all_ids[split_idx:])

    def verify(self) -> bool:
        """Returns True if splits are clean and cover all samples."""
        has_leakage = not self.train_ids.isdisjoint(self.val_ids)
        covers_all = (self.train_ids | self.val_ids) == set(self.samples.keys())
        return not has_leakage and covers_all

# Quick test
manager = MLDatasetManager()
for i in range(10):
    manager.add_sample(i, f"data/img_{i}.png", "cat" if i % 2 == 0 else "dog")

manager.split(train_ratio=0.7)
print(f"Train IDs: {manager.train_ids}")
print(f"Val IDs: {manager.val_ids}")
print(f"Split Verification: {'Passed' if manager.verify() else 'Failed'}")
```

---

## Quick Checklist
Before moving on, make sure you can answer:
1. When is a tuple better than a list?
2. What makes sets highly efficient for membership checks (`x in set`) compared to lists?
3. How do you prevent nested configuration dictionaries from raising `KeyError` exceptions when key names change?
