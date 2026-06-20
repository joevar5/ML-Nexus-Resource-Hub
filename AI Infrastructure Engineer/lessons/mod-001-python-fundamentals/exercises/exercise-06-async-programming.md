# Exercise 06: Asynchronous Operations in ML Pipelines

## Overview
This exercise covers using Python's `asyncio` to execute concurrent tasks. You will learn to load datasets, fetch model checkpoints from APIs, and run feature engineering tasks concurrently using `async/await` and `asyncio.gather`.

* **Time**: 45-60 mins
* **Level**: Intermediate to Advanced

---

## Part 1: Async Basics & Coroutines

Use `async` to write non-blocking code. This is extremely useful when your code is waiting on external I/O (disk reads, database queries, web calls).

### 1. Sequential vs. Concurrent execution (`async_basics.py`)

```python
import asyncio
import time

async def download_weights(name: str) -> dict:
    print(f"📥 Starting download: {name}")
    await asyncio.sleep(1.5)  # Simulate network latency
    print(f"- Downloaded: {name}")
    return {"model": name, "status": "loaded"}

async def load_features(dataset: str) -> list:
    print(f"📖 Reading dataset: {dataset}")
    await asyncio.sleep(0.8)  # Simulate disk I/O
    print(f"- Loaded: {dataset}")
    return [0.1, 0.4, 0.9]

async def main():
    # 1. Sequential Execution (Blocking)
    print("--- Sequential Execution ---")
    start = time.perf_counter()
    w1 = await download_weights("resnet50")
    f1 = await load_features("imagenet")
    print(f"Completed in {time.perf_counter() - start:.2f}s\n")

    # 2. Concurrent Execution (Non-blocking)
    print("--- Concurrent Execution ---")
    start = time.perf_counter()
    # Schedule both coroutines to run concurrently
    weights, features = await asyncio.gather(
        download_weights("resnet50"),
        load_features("imagenet")
    )
    print(f"Completed in {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

**Your Turn:**
1. Add a third async task `preprocess_features(features: list)` that sleeps for `0.5` seconds.
2. Chain the pipeline so preprocessing runs *only after* `load_features` finishes, but concurrently with `download_weights`.

---

## Part 2: Async File I/O & API Requests

Use `aiofiles` and `aiohttp` to read files and query APIs without blocking the main event loop.

### 1. Concurrent Network Calls (`async_api.py`)
```python
import asyncio
import aiohttp

async def fetch_model_config(session: aiohttp.ClientSession, model_id: str) -> dict:
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                return {"model": model_id, "downloads": data.get("downloads", 0)}
            return {"model": model_id, "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"model": model_id, "error": str(e)}

async def main():
    models = ["bert-base-uncased", "gpt2", "distilbert-base-uncased"]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_config(session, m) for m in models]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(r)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Part 3: Asynchronous Error Handling & Timeouts

In async programming, a single failing task inside `asyncio.gather` can either propagate instantly or be caught gracefully.

### 1. Robust Tasks (`async_errors.py`)

```python
import asyncio

async def query_model_server(server_id: int) -> str:
    await asyncio.sleep(1.0)
    if server_id == 2:
        raise ConnectionRefusedError(f"Server {server_id} down.")
    return f"Response from Server {server_id}"

async def main():
    servers = [1, 2, 3]
    tasks = [query_model_server(s) for s in servers]
    
    # return_exceptions=True captures errors as values instead of raising them immediately
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for server_id, result in zip(servers, results):
        if isinstance(result, Exception):
            print(f"❌ Server {server_id} failed: {result}")
        else:
            print(f"✓ {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Concurrency Cheat Sheet

| Mechanism | Best Used For | GIL Limitations | Overhead |
| :--- | :--- | :--- | :--- |
| **Asyncio** | I/O-bound tasks (APIs, network, database queries) | Yes (Runs on 1 thread) | Extremely low |
| **Threading** | I/O-bound tasks using legacy blocking libraries | Yes (Runs on 1 core) | Medium |
| **Multiprocessing** | CPU-bound computation (training models, heavy math) | No (Uses multiple cores) | High (Requires process spawn) |

---

## Challenge: Async Prediction Logger
Write a script that:
1. Simulates predicting batch elements asynchronously.
2. Writes each prediction to a separate file concurrently using `aiofiles`.
3. Ensures timeouts are handled gracefully so no file write takes longer than `2.0` seconds.

---

## Quick Checklist
1. What is the difference between synchronous execution and concurrent async execution?
2. When should you pass `return_exceptions=True` to `asyncio.gather()`?
3. Why does writing CPU-bound operations in `async` block the entire event loop?
