# Exercise 06: LLM Inference Optimization

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercise 03 (vLLM deep dive)

## Objective

For a 7B model, apply a chain of inference optimizations and measure: continuous batching, prefix caching, KV-cache compression, quantization (AWQ), TensorRT-LLM. Target 5×+ cumulative throughput improvement.

## Requirements

1. Baseline: HuggingFace transformers with batch=1.
2. Each optimization measured independently + cumulatively.
3. Final stack producing ≥5× baseline throughput.

## Step-by-step

### Step 1 — Baseline (15 min)
```python
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("...", torch_dtype=torch.bfloat16, device_map="auto")
model.train(False)

prompts = ["Tell me about Kubernetes."] * 32
t0 = time.perf_counter()
for p in prompts:
    inputs = tok(p, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100)
print(f"baseline: {32/(time.perf_counter()-t0):.2f} req/s")
```

### Step 2 — vLLM (continuous batching) (15 min)
Same model, served via vLLM. Drive same load. Expect 3-5× over baseline.

### Step 3 — Prefix caching (10 min)
`--enable-prefix-caching`. Workload with shared prompt prefix benefits.

### Step 4 — AWQ quantization (30 min)
```bash
# Quantize once (offline)
pip install autoawq
python -c "from awq import AutoAWQForCausalLM; m = AutoAWQForCausalLM.from_pretrained('Mistral-7B-Instruct-v0.3'); m.quantize(...); m.save_quantized('./mistral-awq')"

vllm serve ./mistral-awq --quantization awq
```
4-bit weights → 4× smaller, often 1.5× faster.

### Step 5 — KV cache compression (15 min)
`--kv-cache-dtype fp8` reduces KV memory by 50%; allows higher batch sizes.

### Step 6 — TensorRT-LLM (60 min)
Export to TRT-LLM engine; serve via Triton Inference Server.
```bash
# Compile
trtllm-build --model_config ... --output ./engine
# Serve via Triton
docker run nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 ...
```
TensorRT-LLM often 1.5-2× faster than vLLM on the same hardware.

### Step 7 — Continuous batching benchmark at concurrency (30 min)
Use a load tool (vegeta, oha, k6):
```bash
hey -z 60s -c 32 -m POST -T 'application/json' \
  -d '{"model":"...","messages":[...]}' http://localhost:8000/v1/chat/completions
```

### Step 8 — Final report (15 min)

| Optimization | Throughput (req/s @ c32) | TTFT p95 | Speedup vs baseline |
|---|---|---|---|
| baseline (HF) | ~1 | ~3000 ms | 1× |
| vLLM | ~5 | ~400 ms | 5× |
| + prefix caching | ~7 | ~200 ms | 7× |
| + AWQ | ~12 | ~250 ms | 12× |
| + KV fp8 | ~14 | ~240 ms | 14× |
| TRT-LLM | ~18 | ~150 ms | 18× |

## Deliverables

1. Working stack of each optimization.
2. `RESULTS.md` with measured table.
3. `RECOMMENDATIONS.md`: which optimizations are worth which complexity.

## Validation

- [ ] Each optimization measurable improvement.
- [ ] Cumulative speedup ≥ 5×.
- [ ] Quality (text output coherence) preserved at each step.

## Common pitfalls

- **Quantization quality regression** — AWQ usually safe; INT4 less so. Always run an eval set.
- **TRT-LLM build per GPU type** — Engines are GPU-specific.
- **TTFT vs throughput tradeoff** — Larger batch = better throughput but worse TTFT. Tune for workload.
