# Lab 01: Deploy an LLM with vLLM

**Duration:** 75 min  **Prerequisites:** GPU with ≥12GB VRAM; Docker with NVIDIA toolkit

## Objective
Run a 7B-parameter LLM (Mistral-7B or Llama-3-8B-Instruct) via vLLM's OpenAI-compatible server, benchmark throughput vs HuggingFace baseline, and use PagedAttention's KV-cache to fit more concurrent requests.

## Steps

### 1. Pull and start vLLM
```bash
docker run -d --gpus all --name vllm \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:v0.5.0 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --max-model-len 4096 \
  --dtype bfloat16
```
Wait ~5-10 min for download + load.

### 2. Verify
```bash
curl http://localhost:8000/v1/models
```

### 3. Single completion (OpenAI-compatible)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "Explain async I/O in 3 sentences."}],
    "max_tokens": 200
  }' | jq .
```

### 4. Benchmark throughput
```bash
pip install openai
python - <<'PY'
import asyncio, time
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

async def one():
    await client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role":"user","content":"Write a haiku about Kubernetes."}],
        max_tokens=64,
    )

async def bench(n=64, concurrency=16):
    sem = asyncio.Semaphore(concurrency)
    async def go():
        async with sem: await one()
    t0 = time.perf_counter()
    await asyncio.gather(*(go() for _ in range(n)))
    print(f"{n} reqs / {time.perf_counter()-t0:.1f}s = {n/(time.perf_counter()-t0):.1f} req/s")

asyncio.run(bench())
PY
```

### 5. Compare to HF transformers baseline
```python
from transformers import pipeline
import time
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device=0)
t0=time.perf_counter()
for _ in range(8): pipe("Write a haiku about Kubernetes.", max_new_tokens=64)
print(f"hf: {8/(time.perf_counter()-t0):.1f} req/s")
```
Expect vLLM 5-10× higher RPS at the same concurrency.

### 6. Monitor with vLLM metrics
vLLM exposes `/metrics` (Prometheus). Scrape it; key metrics: `vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`.

## Validation
- [ ] `/v1/models` lists the served model.
- [ ] Single completion returns reasonable text.
- [ ] Benchmark shows ≥10 req/s at concurrency 16 on a single 24GB GPU.
- [ ] `gpu_cache_usage_perc` grows under load.

## Cleanup
```bash
docker stop vllm && docker rm vllm
```

## Troubleshooting
- **OOM on load** — Reduce `--max-model-len` or use a quantized variant (`--quantization awq`).
- **`Cannot find HF token`** — Mistral requires HF auth; set `HF_TOKEN` and pass via `-e`.
- **Very slow first request** — CUDA graphs compiling; subsequent requests are fast.
