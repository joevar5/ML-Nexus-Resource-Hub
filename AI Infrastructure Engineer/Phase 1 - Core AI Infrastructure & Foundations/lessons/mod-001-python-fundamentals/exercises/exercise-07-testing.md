# Exercise 07: Testing ML Code with Pytest

## Overview
This exercise covers writing unit and integration tests for ML utility functions using Pytest. You will learn to use fixtures, parameterize test cases to cover multiple scenarios, mock external API dependencies, and write async unit tests.

* **Time**: 45-60 mins
* **Level**: Intermediate

---

## Part 1: Assertions & Fixtures

Fixtures help set up clean state (datasets, directories, model configs) before tests run, and tear them down afterward.

### 1. Basic Assertions & Fixtures (`tests/test_basics.py`)

```python
import pytest

# Simple ML function to test
def calculate_accuracy(y_true: list, y_pred: list) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Inputs must have equal length.")
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)

# 1. Define shared test data using fixtures
@pytest.fixture
def sample_data():
    return [1, 0, 1, 1], [1, 0, 0, 1]

# 2. Use fixture as a parameter in the test function
def test_accuracy_calculation(sample_data):
    y_true, y_pred = sample_data
    assert calculate_accuracy(y_true, y_pred) == 0.75

def test_accuracy_empty():
    assert calculate_accuracy([], []) == 0.0

def test_accuracy_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        calculate_accuracy([1, 0], [1])
```

---

## Part 2: Parameterization & Temp Files

Parameterization lets you run a single test function multiple times against different inputs.

### 1. Parameterized Range Tests (`tests/test_pipeline.py`)

```python
import pytest

def scale_feature(val: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val)

# Run this test 3 times with different inputs
@pytest.mark.parametrize(
    "val, min_val, max_val, expected",
    [
        (5.0, 0.0, 10.0, 0.5),
        (0.0, 0.0, 10.0, 0.0),
        (10.0, 0.0, 10.0, 1.0),
    ]
)
def test_scale_feature(val, min_val, max_val, expected):
    assert scale_feature(val, min_val, max_val) == expected

# 2. Writing to temporary file directories using Pytest's built-in `tmp_path` fixture
def test_save_metrics(tmp_path):
    temp_file = tmp_path / "metrics.txt"
    temp_file.write_text("loss: 0.123")
    
    assert temp_file.exists()
    assert temp_file.read_text() == "loss: 0.123"
```

---

## Part 3: Mocking API Calls & Objects

Use `unittest.mock` to simulate expensive API requests (like fetching model metadata from Hugging Face or sending data to a weights database) without making network calls.

### 1. Mocking HTTP requests (`tests/test_mocking.py`)
```python
from unittest.mock import patch, Mock

def fetch_model_status(model_id: str) -> dict:
    import requests
    response = requests.get(f"https://api.models/status/{model_id}")
    return response.json()

# Patch the requests.get call
@patch("requests.get")
def test_fetch_model_status(mock_get):
    # Setup mock response object
    mock_response = Mock()
    mock_response.json.return_value = {"status": "active", "epoch": 12}
    mock_get.return_value = mock_response

    result = fetch_model_status("bert-v2")
    assert result["status"] == "active"
    mock_get.assert_called_once_with("https://api.models/status/bert-v2")
```

---

## Part 4: Testing Asynchronous Code

Test `async` functions using the `pytest-asyncio` library.

### 1. Async Test Case (`tests/test_async.py`)
```python
import pytest
import asyncio

async def fetch_prediction_async(sample_id: int) -> float:
    await asyncio.sleep(0.1)  # Simulate API latency
    return 0.95

@pytest.mark.asyncio
async def test_fetch_prediction():
    pred = await fetch_prediction_async(42)
    assert pred == 0.95
```

---

## Running the Test Suite

Execute the following commands in your terminal:

```bash
# Run all tests in the tests/ directory
pytest tests/ -v

# Run only tests matching "accuracy" in their name
pytest -k "accuracy"

# Run tests and show code coverage percentage
pytest --cov=src tests/ --cov-report=term-missing
```

---

## Challenge: Test the Dataset Manager
Write tests in a new file `tests/test_dataset_manager.py` that:
1. Uses a fixture to instantiate a clean `MLDatasetManager` (from Exercise 02).
2. Tests the split function to ensure data is divided into training and validation sets correctly.
3. Parameterizes different test split ratios (e.g. `0.7`, `0.8`, `0.9`).
4. Uses assertions to verify that no data leakage exists between the resulting training and validation sets.

---

## Quick Checklist
1. What is the difference between a unit test and an integration test?
2. Why should tests be independent of each other?
3. How do Pytest markers (like `@pytest.mark.skipif`) help manage environment-specific tests (e.g., tests that require a GPU)?
