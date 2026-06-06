# Exercise 04: Reading and Writing ML Data Files

## Overview
This exercise covers reading and writing file formats commonly used in ML pipelines: CSV, JSON, YAML, and pickle. You will learn to use context managers, stream large datasets, serialize python objects safely, and build a unified file manager.

* **Time**: 45-60 mins
* **Level**: Intermediate

---

## Part 1: Tabular & Structured Configs (CSV, JSON, YAML)

### 1. File Formats in Action (`io_basics.py`)
Save and run this to read and write dataset CSVs, structured JSON metadata, and human-readable YAML configurations.

```python
import csv
import json
import yaml

# --- 1. CSV datasets ---
def write_predictions(filepath: str, ids: list[int], preds: list[float]) -> None:
    """Writes model inferences to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "prediction"])
        writer.writerows(zip(ids, preds))

write_predictions("preds.csv", [101, 102], [0.925, 0.141])

# --- 2. JSON metadata ---
def save_metadata(filepath: str, metadata: dict) -> None:
    """Saves model configurations to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

save_metadata("metadata.json", {"model": "ViT", "version": "1.0", "acc": 0.942})

# --- 3. YAML for human configuration ---
pipeline_config = {
    "model": {"type": "resnet", "layers": 50},
    "training": {"lr": 0.001, "batch_size": 64}
}

with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)
```

**Your Turn:**
1. Write a function that reads `preds.csv` back using `csv.DictReader` and returns a list of dictionaries.
2. Read `config.yaml` using `yaml.safe_load`.

---

## Part 2: Object Serialization & Generators

### 1. Pickle Serialization (`checkpoint_ops.py`)
Pickle allows quick saving of Python classes and objects (like tokenizer states or custom pipelines). 

> [!WARNING]
> Never unpickle files from untrusted sources. Pickle files can execute arbitrary code during loading.

```python
import pickle

checkpoint = {
    "epoch": 45,
    "model_weights": [0.12, -0.45, 0.92],
    "val_loss": 0.104
}

# Serialize (write binary)
with open("checkpoint.pkl", "wb") as f:
    pickle.dump(checkpoint, f)

# Deserialize (read binary)
with open("checkpoint.pkl", "rb") as f:
    loaded_checkpoint = pickle.load(f)
print(f"Restored to epoch: {loaded_checkpoint['epoch']}")
```

### 2. Large File Streaming (`streaming.py`)
Loading a 20GB dataset into memory at once will crash your environment. Use generators to stream it row-by-row:

```python
from typing import Iterator

def stream_csv_rows(filepath: str) -> Iterator[dict[str, str]]:
    """Yields rows one by one to keep memory consumption near zero."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

# Example usage
for row in stream_csv_rows("preds.csv"):
    print(f"Row: {row}")
```

---

## Part 3: The Challenge — Unified ML File Manager
Build a clean `MLFileManager` that uses `pathlib` and handles saving/loading of JSON, YAML, and CSV based on the file extension.

```python
# file_manager.py
from pathlib import Path
import json
import yaml
import csv
from typing import Any

class MLFileManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filepath: str, data: Any) -> None:
        path = self.base_dir / filepath
        ext = path.suffix.lower()
        
        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif ext in (".yaml", ".yml"):
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Extension {ext} not supported.")
        print(f"✓ Saved to {path}")

# Run quick test
fm = MLFileManager("outputs")
fm.save("run_config.json", {"epochs": 100})
fm.save("run_config.yaml", {"learning_rate": 0.005})
```

---

## Quick Checklist
1. What is the difference between `json.dump` and `json.dumps`?
2. Why should you always use `yaml.safe_load` instead of `yaml.load`?
3. How does `Path` from `pathlib` handle cross-platform path separators (macOS/Linux `/` vs. Windows `\`)?
