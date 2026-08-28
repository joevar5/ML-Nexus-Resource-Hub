# Exercise 09: GPU Inference Optimization

**Duration:** 3 hours
**Difficulty:** Intermediate+
**Prerequisites:** Exercises 03 + 08

## Objective

Take a trained ResNet-50 (or BERT-base) and apply five inference optimizations in sequence: `torch.compile`, fp16/bf16, int8 quantization (`torch.ao.quantization`), TensorRT export, and continuous batching. Measure cumulative speedup and accuracy delta at each step.

## Why this matters

Inference is where your model lives in production. The right inference stack can be 5-20× faster than baseline, with negligible accuracy loss. Engineers who can do the optimization chain become bottleneck removers.

## Requirements

For ResNet-50 (or BERT-base) on a 224×224 image (or 128-token input):

1. **Baseline**: `model.train(False)` + `torch.no_grad()`, fp32.
2. **`torch.compile`**: `model_compiled = torch.compile(model, mode="reduce-overhead")`.
3. **bf16 / fp16**: same model in lower precision.
4. **int8 quantization**: `torch.ao.quantization.quantize_dynamic`.
5. **TensorRT**: export to ONNX → import to TRT → benchmark.
6. **Continuous batching**: implement a simple async batcher that aggregates requests for 20ms windows.

Report for each step: latency (p50/p95), throughput (req/s at concurrency 32), accuracy on a held-out validation set.

## Step-by-step

### Step 1 — Baseline (15 min)
```python
import torch, time
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()
model.train(False)
x = torch.randn(1, 3, 224, 224, device="cuda")

def bench(model, x, iters=100):
    with torch.inference_mode():
        for _ in range(5): model(x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): model(x); torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

print(f"baseline fp32: {bench(model, x):.2f}ms")
```

### Step 2 — torch.compile (15 min)
```python
compiled = torch.compile(model, mode="reduce-overhead")
print(f"compile:        {bench(compiled, x):.2f}ms")
```
First call triggers compilation (slow). Subsequent calls are 1.5-3× faster.

### Step 3 — Lower precision (30 min)
```python
model_bf16 = model.to(torch.bfloat16)
x_bf16 = x.to(torch.bfloat16)
print(f"bf16:           {bench(model_bf16, x_bf16):.2f}ms")

compiled_bf16 = torch.compile(model_bf16, mode="reduce-overhead")
print(f"compile+bf16:   {bench(compiled_bf16, x_bf16):.2f}ms")
```

### Step 4 — Dynamic int8 quantization (45 min)
```python
import torch.ao.quantization as q

model_cpu = resnet50(weights=ResNet50_Weights.DEFAULT).cpu()
model_cpu.train(False)
# Dynamic only quantizes linear layers (CPU-only); for full GPU int8, use TensorRT.
qmodel = q.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
x_cpu = x.cpu().float()
print(f"int8 dyn (cpu): {bench(qmodel, x_cpu):.2f}ms")
# Note: this is CPU-only. For GPU int8, use TensorRT or vLLM-style quantization.
```

### Step 5 — TensorRT (60 min)
```bash
pip install torch-tensorrt
```
```python
import torch_tensorrt
trt_model = torch_tensorrt.compile(
    model.cuda(),
    inputs=[torch_tensorrt.Input(shape=[1,3,224,224], dtype=torch.float16)],
    enabled_precisions={torch.float16},
)
print(f"TensorRT fp16:  {bench(trt_model, x.half()):.2f}ms")
```

### Step 6 — Continuous batching (45 min)
Build an async server that aggregates inbound requests for up to 20ms or until 32 are queued:
```python
import asyncio
queue: asyncio.Queue = asyncio.Queue()

async def batcher():
    while True:
        items = [await queue.get()]
        deadline = time.monotonic() + 0.020
        while len(items) < 32:
            timeout = deadline - time.monotonic()
            if timeout <= 0: break
            try: items.append(await asyncio.wait_for(queue.get(), timeout))
            except asyncio.TimeoutError: break
        # batch inference
        xs = torch.stack([item.input for item in items]).cuda()
        with torch.inference_mode():
            preds = model(xs)
        for item, pred in zip(items, preds):
            item.future.set_result(pred)
```

Benchmark with `wrk` or `hey` at concurrency 64. Expect throughput 5-10× over per-request inference.

### Step 7 — Accuracy validation (15 min)
For each optimized variant, run inference on the ImageNet validation set's top-1 accuracy. Report any accuracy drop > 0.5%.

## Deliverables

1. `bench_inference.py` for all 6 variants.
2. `RESULTS.md` with latency + throughput + accuracy table.
3. `RECOMMENDATIONS.md`: which technique is worth it for typical ML inference services.
4. (Stretch) Live demo of the continuous batcher.

## Validation

- [ ] Each technique reports a measurement.
- [ ] Cumulative speedup baseline → fully optimized ≥ 5×.
- [ ] Accuracy drop ≤ 1% top-1 on ImageNet val for each technique.
- [ ] Continuous batcher achieves ≥ 5× throughput vs single-request at concurrency 32.

## Stretch goals

- Add **AWQ / GPTQ** 4-bit quantization for transformers (LLM serving).
- Use **vLLM** for the transformer benchmark; compare to TensorRT.
- Add **kernel fusion** profiling: `nvprof` or `nsys` to confirm fewer kernel launches with `torch.compile`.

## Common pitfalls

- **`torch.compile` slow first call** — Compilation can take 30s+ for the first inference. Pre-compile at service startup.
- **TensorRT fp16 introduces small accuracy regressions** — Usually acceptable; verify on your validation set.
- **Continuous batcher CPU-bound** — Python event loop overhead can swamp GPU; profile with `py-spy` if batched throughput plateaus.
- **`quantize_dynamic` is CPU-only** — For GPU int8, use TensorRT or vLLM. Don't confuse the APIs.
